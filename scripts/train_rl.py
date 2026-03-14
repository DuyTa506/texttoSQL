"""
Launch script for RL Training (Stage 2).

Supports two algorithms:
  - GRPO (default): reward-based optimization with vLLM sampling
  - DPO  (ablation): preference learning from chosen/rejected pairs

Supports multitask training (nl2sql + SQL correction) via --correction_data_path.
When omitted, trains on nl2sql data only (original baseline behaviour).

Usage:
    # GRPO – nl2sql only (baseline)
    python scripts/train_rl.py --sft_model_path ./checkpoints/sft/final

    # GRPO – multitask nl2sql + correction (80/20 mix)
    python scripts/train_rl.py \\
        --sft_model_path ./checkpoints/sft/final \\
        --train_data_path data/sft/train.jsonl \\
        --correction_data_path data/correction/train.jsonl \\
        --correction_mix_ratio 0.2

    # GRPO with custom settings
    python scripts/train_rl.py \\
        --sft_model_path ./checkpoints/sft/final \\
        --train_data_path data/sft/train.jsonl \\
        --num_generations 8 \\
        --max_steps 500

    # DPO ablation
    python scripts/train_rl.py \\
        --sft_model_path ./checkpoints/sft/final \\
        --algorithm dpo \\
        --train_data_path data/dpo/train.jsonl

    # Multi-GPU via torchrun
    torchrun --nproc_per_node=4 scripts/train_rl.py \\
        --sft_model_path ./checkpoints/sft/final \\
        --train_data_path data/sft/train.jsonl
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.config import LoRAConfig, RLConfig
from training.rl_trainer import DPOTrainerUnsloth, GRPOTrainerUnsloth

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="RL Training – Stage 2 (GRPO or DPO)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Algorithm ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--algorithm", default="grpo", choices=["grpo", "dpo"],
        help="RL algorithm: grpo (primary) or dpo (ablation)",
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--sft_model_path", default="./checkpoints/sft/final",
        help="Path to Stage-1 SFT checkpoint (Unsloth-compatible)",
    )
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument(
        "--load_in_4bit", action="store_true", default=False,
        help="Load model in 4-bit (QLoRA)",
    )

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument("--output_dir", default="./checkpoints/rl")

    # ── Data ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--train_data_path", default="data/sft/train.jsonl",
        help=(
            "JSONL training file. "
            "GRPO expects {prompt, answer[, schema_tables]}. "
            "DPO expects {prompt, chosen, rejected}."
        ),
    )
    parser.add_argument(
        "--db_dir", default="",
        help="Directory with SQLite DB files (optional; used for live execution reward)",
    )

    # ── Correction multitask (optional) ───────────────────────────────────────
    correction_group = parser.add_argument_group("Correction multitask options")
    correction_group.add_argument(
        "--correction_data_path", default="",
        help=(
            "Path to correction JSONL produced by scripts/mine_correction_data.py. "
            "When set, enables multitask GRPO: nl2sql + SQL correction. "
            "Leave empty (default) for nl2sql-only training."
        ),
    )
    correction_group.add_argument(
        "--correction_mix_ratio", type=float, default=0.2,
        help=(
            "Fraction of each batch drawn from correction samples (default 0.2 = 20%%). "
            "Only used when --correction_data_path is set."
        ),
    )

    # ── Training (shared) ─────────────────────────────────────────────────────
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument(
        "--max_steps", type=int, default=100,
        help="Max training steps (-1 → derived from num_epochs × dataset size)",
    )
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.001)

    # ── LoRA ──────────────────────────────────────────────────────────────────
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)

    # ── GRPO-specific ─────────────────────────────────────────────────────────
    grpo_group = parser.add_argument_group("GRPO options")
    grpo_group.add_argument(
        "--num_generations", type=int, default=4,
        help="Completions sampled per prompt (decrease if OOM)",
    )
    grpo_group.add_argument(
        "--grpo_temperature", type=float, default=1.0,
        help="Generation temperature for GRPO rollouts",
    )
    grpo_group.add_argument("--max_prompt_length", type=int, default=512)
    grpo_group.add_argument("--max_completion_length", type=int, default=2048)
    grpo_group.add_argument("--kl_coeff", type=float, default=0.05)
    grpo_group.add_argument("--vllm_min_p", type=float, default=0.1)
    grpo_group.add_argument("--vllm_seed", type=int, default=3407)

    # ── DPO-specific ──────────────────────────────────────────────────────────
    dpo_group = parser.add_argument_group("DPO options")
    dpo_group.add_argument("--dpo_beta", type=float, default=0.1,
                           help="KL penalty coefficient for DPO")
    dpo_group.add_argument("--dpo_max_length", type=int, default=1024)
    dpo_group.add_argument("--dpo_max_prompt_length", type=int, default=512)

    args = parser.parse_args()

    # ── Build RLConfig ────────────────────────────────────────────────────────
    lora = LoRAConfig(r=args.lora_r, lora_alpha=args.lora_alpha)

    config = RLConfig(
        algorithm=args.algorithm,
        sft_model_path=args.sft_model_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        lora=lora,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        # GRPO
        grpo_num_generations=args.num_generations,
        grpo_temperature=args.grpo_temperature,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        kl_coeff=args.kl_coeff,
        vllm_min_p=args.vllm_min_p,
        vllm_seed=args.vllm_seed,
        # DPO
        dpo_beta=args.dpo_beta,
        dpo_max_length=args.dpo_max_length,
        dpo_max_prompt_length=args.dpo_max_prompt_length,
        # Data
        train_data_path=args.train_data_path,
        db_dir=args.db_dir,
        # Correction multitask
        correction_data_path=args.correction_data_path,
        correction_mix_ratio=args.correction_mix_ratio,
    )

    # ── Log configuration ─────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("RL Training Configuration (Stage 2)")
    logger.info("=" * 60)
    logger.info("  Algorithm:     %s", config.algorithm.upper())
    logger.info("  SFT model:     %s", config.sft_model_path)
    logger.info("  4-bit:         %s", config.load_in_4bit)
    logger.info("  LoRA rank:     %d", config.lora.r)
    logger.info("  Seq length:    %d", config.max_seq_length)
    logger.info("  LR:            %.2e", config.learning_rate)
    logger.info("  Epochs:        %d", config.num_epochs)
    logger.info("  Train data:    %s", config.train_data_path)
    if config.correction_data_path:
        logger.info("  Correction:    %s (mix=%.0f%%)",
                    config.correction_data_path, config.correction_mix_ratio * 100)
    else:
        logger.info("  Correction:    disabled (nl2sql only)")
    logger.info("  Output:        %s", config.output_dir)
    if config.algorithm == "grpo":
        logger.info("  Generations:   %d", config.grpo_num_generations)
        logger.info("  Temperature:   %.2f", config.grpo_temperature)
        logger.info("  Max steps:     %d", args.max_steps)
        logger.info("  KL coeff:      %.4f", config.kl_coeff)
        logger.info("  vLLM min_p:    %.2f", config.vllm_min_p)
    else:  # dpo
        logger.info("  DPO beta:      %.2f", config.dpo_beta)
        logger.info("  Max length:    %d", config.dpo_max_length)
    logger.info("=" * 60)

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not Path(config.train_data_path).exists():
        logger.error("Training data not found: %s", config.train_data_path)
        logger.error("Run scripts/prepare_sft_data.py first to generate training data.")
        sys.exit(1)

    if not Path(config.sft_model_path).exists():
        logger.error("SFT checkpoint not found: %s", config.sft_model_path)
        logger.error("Run scripts/train_sft.py first to produce the Stage-1 checkpoint.")
        sys.exit(1)

    if config.correction_data_path and not Path(config.correction_data_path).exists():
        logger.error("Correction data not found: %s", config.correction_data_path)
        logger.error(
            "Run scripts/mine_correction_data.py first, "
            "or omit --correction_data_path for nl2sql-only training."
        )
        sys.exit(1)

    # ── Dispatch to trainer ───────────────────────────────────────────────────
    if config.algorithm == "grpo":
        trainer = GRPOTrainerUnsloth(config)
        trainer.setup()
        trainer.train()
        logger.info(
            "GRPO training complete! Checkpoint saved to %s/final", config.output_dir
        )

    else:  # dpo
        trainer = DPOTrainerUnsloth(config)
        trainer.setup()
        trainer.train()
        logger.info(
            "DPO training complete! Checkpoint saved to %s/final", config.output_dir
        )


if __name__ == "__main__":
    main()
