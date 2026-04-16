"""
Stage 2 RL Training Launch Script — GRPO or DPO.

Usage:
  # GRPO (default):
  python scripts/train_rl.py \\
      --sft_model_path ./checkpoints/sft/final \\
      --train_data_path data/rl/grpo_train.jsonl \\
      --output_dir ./checkpoints/rl \\
      --max_steps 500

  # DPO (ablation):
  python scripts/train_rl.py \\
      --algorithm dpo \\
      --sft_model_path ./checkpoints/sft/final \\
      --train_data_path data/rl/dpo_train.jsonl \\
      --output_dir ./checkpoints/rl_dpo
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Launch RL (GRPO/DPO) training.")

    # Algorithm
    parser.add_argument("--algorithm", default="grpo", choices=["grpo", "dpo"],
                        help="RL algorithm to use (default: grpo)")

    # Model
    parser.add_argument("--sft_model_path", default="./checkpoints/sft/final",
                        help="Path to SFT checkpoint to start RL from")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    # Training
    parser.add_argument("--output_dir", default="./checkpoints/rl")
    parser.add_argument("--train_data_path", default="data/rl/grpo_train.jsonl")
    parser.add_argument("--db_dir", default="datasets/data")
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Max training steps (-1 = use num_epochs)")
    parser.add_argument("--logging_steps", type=int, default=1)

    # GRPO-specific
    parser.add_argument("--grpo_num_generations", type=int, default=4,
                        help="Number of samples per prompt for GRPO (reduce if OOM)")
    parser.add_argument("--grpo_temperature", type=float, default=1.0)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_completion_length", type=int, default=2048)

    # DPO-specific
    parser.add_argument("--dpo_beta", type=float, default=0.1)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)

    return parser.parse_args()


def main():
    args = parse_args()

    from training.config import LoRAConfig, RLConfig
    from training.rl_trainer import GRPOTrainerUnsloth, DPOTrainerUnsloth

    lora_config = LoRAConfig(r=args.lora_r, lora_alpha=args.lora_alpha)

    config = RLConfig(
        algorithm=args.algorithm,
        sft_model_path=args.sft_model_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        lora=lora_config,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        train_data_path=args.train_data_path,
        db_dir=args.db_dir,
        grpo_num_generations=args.grpo_num_generations,
        grpo_temperature=args.grpo_temperature,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        dpo_beta=args.dpo_beta,
    )

    logger.info("RL config: algorithm=%s, model=%s", config.algorithm, config.sft_model_path)
    logger.info("Training data: %s", config.train_data_path)

    if config.algorithm == "grpo":
        trainer_obj = GRPOTrainerUnsloth(config)
    else:
        trainer_obj = DPOTrainerUnsloth(config)

    trainer_obj.setup()
    trainer_obj.train()


if __name__ == "__main__":
    main()
