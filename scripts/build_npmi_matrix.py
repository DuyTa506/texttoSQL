"""
Build NPMI co-occurrence matrix from training data.

Usage examples:

  # From OmniSQL processed data (recommended — uses full 2.5M+ corpus):
  python scripts/build_npmi_matrix.py \
      --data_paths data/train_synsql.json data/train_spider.json data/train_bird.json \
      --data_format omnisql \
      --output_path data/npmi_matrix.json \
      --min_count 3

  # From Spider raw data:
  python scripts/build_npmi_matrix.py \
      --data_paths data/spider/train_spider.json \
      --data_format spider \
      --output_path data/npmi_matrix.json \
      --min_count 3

  # With max samples (subsample for faster builds):
  python scripts/build_npmi_matrix.py \
      --data_paths data/train_synsql.json \
      --data_format omnisql \
      --output_path data/npmi_matrix.json \
      --max_samples 500000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.utils.npmi_scorer import NPMIScorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_data(data_paths: list[str], data_format: str, max_samples: int | None) -> list[dict]:
    """Load training examples from one or more JSON files."""
    all_examples: list[dict] = []

    for fpath in data_paths:
        logger.info("Loading data from: %s", fpath)

        try:
            import ijson

            with open(fpath, "r", encoding="utf-8") as f:
                count = 0
                for obj in ijson.items(f, "item"):
                    all_examples.append(obj)
                    count += 1
                    if max_samples and len(all_examples) >= max_samples:
                        break
                logger.info("  → loaded %d examples from %s", count, fpath)

        except ImportError:
            import json

            logger.warning("ijson not installed, using json.load() — may be slow for large files")
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            remaining = (max_samples - len(all_examples)) if max_samples else len(data)
            all_examples.extend(data[:remaining])
            logger.info("  → loaded %d examples from %s", min(remaining, len(data)), fpath)

        if max_samples and len(all_examples) >= max_samples:
            logger.info("Reached max_samples=%d, stopping.", max_samples)
            break

    logger.info("Total examples loaded: %d", len(all_examples))
    return all_examples


def main():
    parser = argparse.ArgumentParser(
        description="Build NPMI co-occurrence matrix for schema linking",
    )
    parser.add_argument(
        "--data_paths",
        nargs="+",
        required=True,
        help="Paths to training JSON files",
    )
    parser.add_argument(
        "--data_format",
        choices=["spider", "omnisql"],
        default="omnisql",
        help="Data format: 'spider' (question/query) or 'omnisql' (input_seq/output_seq)",
    )
    parser.add_argument(
        "--output_path",
        default="./data/npmi_matrix.json",
        help="Output path for NPMI matrix JSON",
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=3,
        help="Minimum co-occurrence count threshold",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max total samples to use (for faster testing)",
    )

    args = parser.parse_args()

    # Load data
    examples = load_data(args.data_paths, args.data_format, args.max_samples)

    # Build NPMI matrix
    scorer = NPMIScorer(min_count=args.min_count)
    scorer.build_matrix(examples, data_format=args.data_format)

    # Save
    scorer.save(args.output_path)
    logger.info("Done! Matrix saved to: %s", args.output_path)


if __name__ == "__main__":
    main()
