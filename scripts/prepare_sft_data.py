"""
Prepare SFT data — merge OmniSQL datasets, parse thinking, and output
training-ready JSONL files.

This script:
  1. Merges train_spider.json + train_bird.json + train_synsql.json → single training set
  2. Merges dev_*.json → single dev/eval set
  3. Parses each (input_seq, output_seq) → (instruction, response) with <think> tags
  4. Outputs JSONL files ready for Qwen3 SFT with Unsloth

Usage:
  source .venv/bin/activate

  # Full merge (with SynSQL subsampling):
  python scripts/prepare_sft_data.py \
      --data_dir OmniSQL/datasets/data \
      --output_dir data/sft \
      --max_synsql 200000

  # Small test run:
  python scripts/prepare_sft_data.py \
      --data_dir OmniSQL/datasets/data \
      --output_dir data/sft \
      --max_synsql 1000 \
      --max_spider 500 \
      --max_bird 500
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class SFTExample:
    """A single SFT training/eval example."""
    instruction: str                # User prompt (schema + question)
    response: str                   # Model response (reasoning + SQL)
    source: str = ""                # Dataset origin: "spider", "bird", "synsql", etc.
    thinking_mode: str = "thinking" # "thinking" or "direct"

    # Raw parsed fields (for debugging / NPMI)
    schema_context: str = ""
    question: str = ""
    reasoning: str = ""
    sql: str = ""


# =============================================================================
# Parsing Logic
# =============================================================================

# Regex to extract SQL from ```sql ... ``` or ``` ... ``` blocks
_SQL_BLOCK_RE = re.compile(
    r"```(?:sql)?\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

# SQL complexity keywords for think/direct decision
_COMPLEX_SQL_PATTERNS = re.compile(
    r"\bJOIN\b|\bLEFT\b|\bRIGHT\b|\bCROSS\b|\bFULL\b"
    r"|\bSUBQUERY\b|\bEXISTS\b|\bIN\s*\(\s*SELECT"
    r"|\bWITH\b\s+\w+\s+AS"
    r"|\bOVER\s*\("
    r"|\bUNION\b|\bINTERSECT\b|\bEXCEPT\b"
    r"|\bCASE\b\s+WHEN"
    r"|\bHAVING\b"
    r"|\bGROUP\s+BY\b",
    re.IGNORECASE,
)


def is_complex_sql(sql: str) -> bool:
    """Determine if a SQL query is complex enough to require reasoning.

    Complex = any of: JOIN, subquery, CTE, window function, UNION,
    CASE WHEN, HAVING, GROUP BY.
    """
    return bool(_COMPLEX_SQL_PATTERNS.search(sql))


# Markers for extracting sections from input_seq
_SCHEMA_START_MARKERS = [
    "Database Schema:\n",
    "Database Schema :\n",
]
_SCHEMA_END_MARKERS = [
    "This schema describes",
    "\nQuestion:",
    "\nquestion:",
]
_QUESTION_START_MARKERS = [
    "Question:\n",
    "question:\n",
]
_QUESTION_END_MARKERS = [
    "\nInstructions:",
    "\ninstructions:",
    "\nOutput Format:",
    "\noutput format:",
]


def parse_input_seq(input_seq: str) -> tuple[str, str]:
    """Extract (schema_context, question) from OmniSQL input_seq.

    Handles variations in formatting across Spider, BIRD, and SynSQL datasets.
    """
    schema_context = ""
    question = ""

    # ---- Schema extraction ----
    schema_start = -1
    schema_marker_len = 0
    for marker in _SCHEMA_START_MARKERS:
        pos = input_seq.find(marker)
        if pos != -1:
            schema_start = pos
            schema_marker_len = len(marker)
            break

    if schema_start != -1:
        content_start = schema_start + schema_marker_len
        # Find end of schema section
        schema_end = len(input_seq)
        for marker in _SCHEMA_END_MARKERS:
            pos = input_seq.find(marker, content_start)
            if pos != -1 and pos < schema_end:
                schema_end = pos
        schema_context = input_seq[content_start:schema_end].strip()

    # ---- Question extraction ----
    q_start = -1
    q_marker_len = 0
    for marker in _QUESTION_START_MARKERS:
        pos = input_seq.find(marker)
        if pos != -1:
            q_start = pos
            q_marker_len = len(marker)
            break

    if q_start != -1:
        content_start = q_start + q_marker_len
        # Find end of question section
        q_end = len(input_seq)
        for marker in _QUESTION_END_MARKERS:
            pos = input_seq.find(marker, content_start)
            if pos != -1 and pos < q_end:
                q_end = pos
        question = input_seq[content_start:q_end].strip()

    return schema_context, question


def parse_output_seq(output_seq: str) -> tuple[str, str]:
    """Extract (reasoning, sql) from OmniSQL output_seq.

    The output_seq contains CoT reasoning (markdown steps) followed by
    SQL in a ```sql ... ``` code block. Takes the LAST code block as the
    final SQL answer.

    Returns (reasoning, sql).
    """
    # Find all SQL code blocks
    matches = list(_SQL_BLOCK_RE.finditer(output_seq))

    if matches:
        last_match = matches[-1]
        sql = last_match.group(1).strip()

        # Reasoning = everything before the last code block
        reasoning = output_seq[:last_match.start()].strip()

        # Clean up trailing "---" or markdown artifacts from reasoning
        reasoning = re.sub(r"\n---\s*$", "", reasoning).strip()
        reasoning = re.sub(r"###\s*(?:Final\s+)?(?:SQL\s+)?Query:?\s*$", "", reasoning, flags=re.IGNORECASE).strip()

        return reasoning, sql

    # Fallback: look for raw SELECT statement
    select_match = re.search(
        r"(SELECT\s+.+?)(?:\n\n|\Z)",
        output_seq,
        re.IGNORECASE | re.DOTALL,
    )
    if select_match:
        sql = select_match.group(1).strip()
        reasoning = output_seq[:select_match.start()].strip()
        return reasoning, sql

    # Nothing found — return as-is
    return "", output_seq.strip()


def format_thinking_example(schema_context: str, question: str, reasoning: str, sql: str) -> dict:
    """Format as Qwen3 thinking mode example with <think> tags."""
    instruction = (
        f"Given the SQLite database schema below, generate an executable SQL query "
        f"to answer the question.\n"
        f"Reason step-by-step about the schema and logic required before writing the query.\n\n"
        f"Schema:\n{schema_context}\n\n"
        f"Question: {question}"
    )

    # Wrap reasoning in <think> block
    if reasoning:
        response = f"<think>\n{reasoning}\n</think>\n\n```sql\n{sql}\n```"
    else:
        # No reasoning available — still include empty think block
        response = f"<think>\n\n</think>\n\n```sql\n{sql}\n```"

    return {
        "instruction": instruction,
        "response": response,
    }


def format_direct_example(schema_context: str, question: str, sql: str) -> dict:
    """Format as Qwen3 direct mode (no thinking) example."""
    instruction = (
        f"Given the SQLite database schema below, generate an executable SQL query "
        f"to answer the question.\n\n"
        f"Schema:\n{schema_context}\n\n"
        f"Question: {question}"
    )

    # Empty think block signals no-thinking mode to Qwen3
    response = f"<think>\n\n</think>\n\n```sql\n{sql}\n```"

    return {
        "instruction": instruction,
        "response": response,
    }


# =============================================================================
# Data Loading (streaming for large files)
# =============================================================================

def load_json_streaming(path: str, max_samples: Optional[int] = None) -> list[dict]:
    """Load JSON array with streaming support for large files."""
    file_size = os.path.getsize(path)
    basename = os.path.basename(path)

    # Use ijson for files > 500MB
    if file_size > 500_000_000:
        logger.info("Streaming large file: %s (%.1f GB)", basename, file_size / 1e9)
        try:
            import ijson
            results = []
            with open(path, "r", encoding="utf-8") as f:
                for obj in ijson.items(f, "item"):
                    results.append(obj)
                    if max_samples and len(results) >= max_samples:
                        break
                    if len(results) % 100_000 == 0:
                        logger.info("  ... loaded %d samples from %s", len(results), basename)
            return results
        except ImportError:
            logger.error("ijson not installed! Required for large file: %s", basename)
            logger.error("Install with: pip install ijson")
            sys.exit(1)
    else:
        logger.info("Loading file: %s (%.1f MB)", basename, file_size / 1e6)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if max_samples:
            data = data[:max_samples]
        return data


# =============================================================================
# Main Pipeline
# =============================================================================

def process_dataset(
    raw_data: list[dict],
    source: str,
    thinking_ratio: float = 0.75,
) -> list[SFTExample]:
    """Convert raw OmniSQL data to SFT examples with think/no-think mix.

    Parameters
    ----------
    raw_data : list[dict]
        OmniSQL format: [{input_seq, output_seq}, ...]
    source : str
        Dataset name for tracking.
    thinking_ratio : float
        Fraction of examples to format with thinking (default 0.75).

    Returns
    -------
    list[SFTExample]
    """
    examples = []
    skipped = 0

    for item in raw_data:
        input_seq = item.get("input_seq", "")
        output_seq = item.get("output_seq", "")

        if not input_seq or not output_seq:
            skipped += 1
            continue

        # Parse
        schema_context, question = parse_input_seq(input_seq)
        reasoning, sql = parse_output_seq(output_seq)

        # Validate — must have at least a question and SQL
        if not question or not sql:
            skipped += 1
            continue

        # SQL sanity check — must start with a SQL keyword
        sql_upper = sql.strip().upper()
        if not any(sql_upper.startswith(kw) for kw in ("SELECT", "INSERT", "UPDATE", "DELETE", "WITH", "CREATE", "DROP", "ALTER")):
            skipped += 1
            continue

        # Complexity-aware think/direct assignment:
        # - Complex queries (JOIN, subquery, CTE, etc.) ALWAYS use thinking
        # - Simple queries use random ratio (default 75/25)
        query_is_complex = is_complex_sql(sql)

        if query_is_complex:
            # Force thinking for complex queries — model must reason
            use_thinking = True
        else:
            use_thinking = random.random() < thinking_ratio

        if use_thinking:
            formatted = format_thinking_example(schema_context, question, reasoning, sql)
            mode = "thinking"
        else:
            formatted = format_direct_example(schema_context, question, sql)
            mode = "direct"

        examples.append(SFTExample(
            instruction=formatted["instruction"],
            response=formatted["response"],
            source=source,
            thinking_mode=mode,
            schema_context=schema_context,
            question=question,
            reasoning=reasoning,
            sql=sql,
        ))

    if skipped:
        logger.warning("Skipped %d invalid examples from %s", skipped, source)

    return examples


def save_jsonl(examples: list[SFTExample], output_path: str, include_metadata: bool = False):
    """Save examples as JSONL (one JSON object per line)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            if include_metadata:
                record = asdict(ex)
            else:
                # Minimal format for SFT training
                record = {
                    "instruction": ex.instruction,
                    "response": ex.response,
                }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Saved %d examples to %s", len(examples), path)


def main():
    parser = argparse.ArgumentParser(
        description="Merge and prepare OmniSQL datasets for SFT training",
    )
    parser.add_argument("--data_dir", required=True, help="Path to OmniSQL datasets/data directory")
    parser.add_argument("--output_dir", default="./data/sft", help="Output directory for prepared data")
    parser.add_argument("--max_synsql", type=int, default=None, help="Max samples from train_synsql.json")
    parser.add_argument("--max_spider", type=int, default=None, help="Max samples from train_spider.json")
    parser.add_argument("--max_bird", type=int, default=None, help="Max samples from train_bird.json")
    parser.add_argument("--thinking_ratio", type=float, default=0.75, help="Fraction of thinking examples (default 0.75)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--include_metadata", action="store_true", help="Include metadata in output JSONL")
    parser.add_argument("--shuffle", action="store_true", default=True, help="Shuffle final dataset")

    args = parser.parse_args()
    random.seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # ======================================================================
    # Step 1: Load and merge training data
    # ======================================================================
    logger.info("=" * 60)
    logger.info("STEP 1: Loading training datasets")
    logger.info("=" * 60)

    train_files = {
        "spider": ("train_spider.json", args.max_spider),
        "bird":   ("train_bird.json",   args.max_bird),
        "synsql": ("train_synsql.json", args.max_synsql),
    }

    all_train_examples: list[SFTExample] = []

    for source, (filename, max_samples) in train_files.items():
        filepath = data_dir / filename
        if not filepath.exists():
            logger.warning("Training file not found: %s — skipping", filepath)
            continue

        raw_data = load_json_streaming(str(filepath), max_samples=max_samples)
        logger.info("Processing %d raw samples from %s...", len(raw_data), filename)

        examples = process_dataset(raw_data, source=source, thinking_ratio=args.thinking_ratio)
        all_train_examples.extend(examples)

        logger.info("  %s: %d examples (%d thinking, %d direct)",
                     source, len(examples),
                     sum(1 for e in examples if e.thinking_mode == "thinking"),
                     sum(1 for e in examples if e.thinking_mode == "direct"))

    # Shuffle
    if args.shuffle:
        random.shuffle(all_train_examples)
        logger.info("Shuffled training set")

    # Save
    train_output = output_dir / "train.jsonl"
    save_jsonl(all_train_examples, str(train_output), include_metadata=args.include_metadata)

    # ======================================================================
    # Step 2: Load and merge dev/eval data
    # ======================================================================
    logger.info("=" * 60)
    logger.info("STEP 2: Loading dev/eval datasets")
    logger.info("=" * 60)

    dev_files = sorted(data_dir.glob("dev_*.json"))
    all_dev_examples: list[SFTExample] = []

    for filepath in dev_files:
        source = filepath.stem.replace("dev_", "")  # e.g., "spider", "bird", "ehrsql"
        raw_data = load_json_streaming(str(filepath))
        logger.info("Processing %d raw samples from %s...", len(raw_data), filepath.name)

        # Dev data: 100% thinking mode for eval
        examples = process_dataset(raw_data, source=source, thinking_ratio=1.0)
        all_dev_examples.extend(examples)

        logger.info("  %s: %d examples", source, len(examples))

    # Save
    if all_dev_examples:
        dev_output = output_dir / "dev.jsonl"
        save_jsonl(all_dev_examples, str(dev_output), include_metadata=args.include_metadata)

    # ======================================================================
    # Summary
    # ======================================================================
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("Training examples: %d", len(all_train_examples))
    logger.info("  - thinking mode: %d (%.1f%%)",
                sum(1 for e in all_train_examples if e.thinking_mode == "thinking"),
                100 * sum(1 for e in all_train_examples if e.thinking_mode == "thinking") / max(len(all_train_examples), 1))
    logger.info("  - direct mode:   %d (%.1f%%)",
                sum(1 for e in all_train_examples if e.thinking_mode == "direct"),
                100 * sum(1 for e in all_train_examples if e.thinking_mode == "direct") / max(len(all_train_examples), 1))

    # Source breakdown
    sources = {}
    for e in all_train_examples:
        sources[e.source] = sources.get(e.source, 0) + 1
    for src, cnt in sorted(sources.items()):
        logger.info("  - %s: %d", src, cnt)

    logger.info("Dev examples: %d", len(all_dev_examples))
    dev_sources = {}
    for e in all_dev_examples:
        dev_sources[e.source] = dev_sources.get(e.source, 0) + 1
    for src, cnt in sorted(dev_sources.items()):
        logger.info("  - %s: %d", src, cnt)

    logger.info("Output files:")
    logger.info("  - %s", train_output)
    if all_dev_examples:
        logger.info("  - %s", dev_output)
    logger.info("Done!")


if __name__ == "__main__":
    main()
