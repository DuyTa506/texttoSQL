"""
Launch script for SFT Training (Stage 1).

Usage:
    python scripts/train_sft.py
    python scripts/train_sft.py --base_model ./models/Qwen3.5-2B
    python scripts/train_sft.py --load_in_4bit  # enable QLoRA
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.config import SFTConfig
from training.sft_trainer import SFTTrainerUnsloth

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="SFT Training (Stage 1)")

    # Model
    parser.add_argument("--base_model", default="./models/Qwen3.5-2B",
                        help="Model name or local path")
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--load_in_4bit", action="store_true", default=False,
                        help="Use QLoRA (4-bit quantization)")

    # Data
    parser.add_argument("--data_source", default="omnisql",
                        choices=["omnisql", "spider", "custom"])
    parser.add_argument("--omnisql_data_paths", nargs="+",
                        default=["data/sft/train.jsonl", "data/sft/dev.jsonl"])
    parser.add_argument("--train_data_path", default="")
    parser.add_argument("--eval_data_path", default="")

    # Training
    parser.add_argument("--output_dir", default="./checkpoints/sft")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=10)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)

    # Thinking
    parser.add_argument("--thinking_ratio", type=float, default=0.75)
    parser.add_argument("--no_thinking", action="store_true")

    args = parser.parse_args()

    # Build config
    config = SFTConfig(
        base_model=args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        enable_thinking=not args.no_thinking,
        thinking_data_ratio=args.thinking_ratio,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        data_source=args.data_source,
        train_data_path=args.train_data_path,
        eval_data_path=args.eval_data_path,
        omnisql_data_paths=args.omnisql_data_paths,
    )
    config.lora.r = args.lora_r
    config.lora.lora_alpha = args.lora_alpha

    logger.info("=" * 60)
    logger.info("SFT Training Configuration")
    logger.info("=" * 60)
    logger.info("  Model:         %s", config.base_model)
    logger.info("  4-bit:         %s", config.load_in_4bit)
    logger.info("  LoRA rank:     %d", config.lora.r)
    logger.info("  Seq length:    %d", config.max_seq_length)
    logger.info("  Batch size:    %d (x%d accum = %d effective)",
                config.batch_size, config.gradient_accumulation_steps,
                config.batch_size * config.gradient_accumulation_steps)
    logger.info("  Epochs:        %d", config.num_epochs)
    logger.info("  LR:            %.2e", config.learning_rate)
    logger.info("  Data source:   %s", config.data_source)
    logger.info("  Data paths:    %s", config.omnisql_data_paths)
    logger.info("  Thinking:      %s (ratio=%.0f%%)",
                config.enable_thinking, config.thinking_data_ratio * 100)
    logger.info("  Output:        %s", config.output_dir)
    logger.info("=" * 60)

    # Create trainer and run
    trainer = SFTTrainerUnsloth(config)
    trainer.setup()
    trainer.train()

    logger.info("SFT training complete! Checkpoint saved to %s/final", config.output_dir)


if __name__ == "__main__":
    main()
