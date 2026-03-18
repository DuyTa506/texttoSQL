#!/usr/bin/env python3
"""
Multi-mode benchmark runner for Text-to-SQL pipeline.

Runs the pipeline across multiple retriever modes (hybrid, graph, merge)
with async/batch LLM inference enabled by default, and prints a comparison
table at the end.

This is a convenience wrapper around ``run_pipeline._benchmark_all()``.

Usage:
    # All 3 modes with async inference
    python scripts/run_benchmark.py --config configs/default.yaml \
        --override schema_graph.graph_path=data/schema_graphs/bird_dev_v3_rebuild.json \
        generation.provider=openai generation.model_path=gpt-4o-mini

    # Only graph + merge modes
    python scripts/run_benchmark.py --config configs/default.yaml \
        --modes graph merge \
        --override schema_graph.graph_path=data/schema_graphs/spider_full.json

    # With concurrency control
    python scripts/run_benchmark.py --config configs/default.yaml \
        --concurrency 10 --batch_size 16
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env file (if present)
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-mode benchmark runner for Text-to-SQL pipeline"
    )
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to config YAML file.")
    parser.add_argument(
        "--override", nargs="*", default=[], metavar="KEY=VALUE",
        help="Override config values, e.g. generation.provider=openai",
    )
    parser.add_argument(
        "--modes", nargs="*", default=["hybrid", "graph", "merge"],
        help="Retriever modes to benchmark (default: hybrid graph merge).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Examples per micro-batch (default: 32).",
    )
    parser.add_argument(
        "--concurrency", type=int, default=20,
        help="Max parallel API calls (default: 20).",
    )
    parser.add_argument(
        "--no_batch", action="store_true",
        help="Disable async/batch inference (run sequentially).",
    )
    parser.add_argument(
        "--reindex", action="store_true",
        help="Force re-indexing even if the index cache is valid.",
    )
    parser.add_argument(
        "--max_examples", type=int, default=0,
        help="Limit number of examples to process (0 = all).",
    )
    args = parser.parse_args()

    overrides = args.override or []
    if args.reindex:
        overrides.append("_reindex=true")
    if args.max_examples:
        overrides.append(f"_max_examples={args.max_examples}")

    from run_pipeline import _benchmark_all

    _benchmark_all(
        args.config,
        overrides=overrides,
        modes=args.modes,
        batch_enabled=not args.no_batch,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
    )


if __name__ == "__main__":
    main()
