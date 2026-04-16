"""
RL Data Curation Pipeline for Text-to-SQL GRPO Training.

5-stage pipeline:
  Stage 1: Identify holdout data (not used in SFT), filter medium→hard
  Stage 2: Generate 3-5 SQL candidates per question with the SFT model
  Stage 3: Split by real execution accuracy (EX=True vs EX=False)
  Stage 4: Teacher-guided error correction for failed cases (via litellm or export)
  Stage 5: Assemble unified GRPO training shard

Usage:
  # Full pipeline:
  python scripts/prepare_rl_data.py \\
      --data_dir datasets/data \\
      --sft_train_jsonl data/sft/train.jsonl \\
      --sft_max_synsql 200000 \\
      --max_holdout 50000 \\
      --sft_model_path ./checkpoints/sft/final \\
      --num_candidates 5 \\
      --teacher_mode api \\
      --teacher_model gpt-4o-mini \\
      --output_dir data/rl

  # Resume from stage 3:
  python scripts/prepare_rl_data.py ... --stages 3,4,5

  # Export failed cases for manual Claude Code processing:
  python scripts/prepare_rl_data.py ... --stages 4 --teacher_mode export
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import random
import re
import sqlite3
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

# ---------------------------------------------------------------------------
# Regex helpers (same as rl_trainer.py)
# ---------------------------------------------------------------------------
SQL_BLOCK_RE = re.compile(r"```sql\s*(.+?)\s*```", flags=re.MULTILINE | re.DOTALL)
THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", flags=re.MULTILINE | re.DOTALL)
FULL_FORMAT_RE = re.compile(
    r"</think>.*?```sql\s*(.+?)\s*```[\s]*(?:<\|endoftext\|>)?[\s]*$",
    flags=re.MULTILINE | re.DOTALL,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RLCandidate:
    """Single holdout example enriched through all pipeline stages."""
    index: int                              # Index in source dataset
    source: str                             # "spider" | "bird" | "synsql"
    db_id: str
    question: str
    gold_sql: str
    schema_context: str
    db_path: str
    complexity: str                         # "simple" | "moderate" | "complex" | "highly_complex"
    schema_tables: str = ""                 # comma-sep table names
    # Stage 2
    predictions: list = field(default_factory=list)      # raw model outputs
    pred_sqls: list = field(default_factory=list)        # extracted SQL per prediction
    pred_errors: list = field(default_factory=list)      # error msg per prediction
    ex_results: list = field(default_factory=list)       # bool per prediction
    any_ex_pass: bool = False
    # Stage 4
    best_broken_sql: str = ""
    best_broken_error: str = ""
    teacher_reasoning: str = ""
    teacher_corrected_sql: str = ""



# ---------------------------------------------------------------------------
# Reused utilities from prepare_sft_data.py
# ---------------------------------------------------------------------------

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
    return bool(_COMPLEX_SQL_PATTERNS.search(sql))


def load_json_streaming(path: str, max_samples: Optional[int] = None, skip: int = 0) -> list[dict]:
    """Load JSON array with optional skip (for holdout) and streaming for large files."""
    file_size = os.path.getsize(path)
    basename = os.path.basename(path)

    if file_size > 500_000_000:
        logger.info("Streaming large file: %s (%.1f GB)", basename, file_size / 1e9)
        try:
            import ijson
            results = []
            idx = 0
            with open(path, "r", encoding="utf-8") as f:
                for obj in ijson.items(f, "item"):
                    if idx < skip:
                        idx += 1
                        continue
                    results.append(obj)
                    idx += 1
                    if max_samples and len(results) >= max_samples:
                        break
                    if len(results) % 100_000 == 0 and results:
                        logger.info("  ... loaded %d samples", len(results))
            return results
        except ImportError:
            logger.error("ijson not installed — required for large files. pip install ijson")
            sys.exit(1)
    else:
        logger.info("Loading file: %s (%.1f MB)", basename, file_size / 1e6)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if skip:
            data = data[skip:]
        if max_samples:
            data = data[:max_samples]
        return data


def parse_input_seq(input_seq: str) -> tuple[str, str]:
    """Extract (schema_context, question) from OmniSQL input_seq."""
    schema_context = ""
    question = ""

    schema_start = -1
    schema_marker_len = 0
    for marker in ["Database Schema:\n", "Database Schema :\n"]:
        pos = input_seq.find(marker)
        if pos != -1:
            schema_start = pos
            schema_marker_len = len(marker)
            break

    if schema_start != -1:
        content_start = schema_start + schema_marker_len
        schema_end = len(input_seq)
        for marker in ["This schema describes", "\nQuestion:", "\nquestion:"]:
            pos = input_seq.find(marker, content_start)
            if pos != -1 and pos < schema_end:
                schema_end = pos
        schema_context = input_seq[content_start:schema_end].strip()

    q_start = -1
    q_marker_len = 0
    for marker in ["Question:\n", "question:\n"]:
        pos = input_seq.find(marker)
        if pos != -1:
            q_start = pos
            q_marker_len = len(marker)
            break

    if q_start != -1:
        content_start = q_start + q_marker_len
        q_end = len(input_seq)
        for marker in ["\nInstructions:", "\ninstructions:", "\nOutput Format:", "\noutput format:"]:
            pos = input_seq.find(marker, content_start)
            if pos != -1 and pos < q_end:
                q_end = pos
        question = input_seq[content_start:q_end].strip()

    return schema_context, question


def save_jsonl(records: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            if isinstance(rec, RLCandidate):
                f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
            else:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Saved %d records → %s", len(records), path)


def load_rl_candidates(path: Path) -> list[RLCandidate]:
    candidates = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            c = RLCandidate(**d)
            candidates.append(c)
    logger.info("Loaded %d RLCandidates from %s", len(candidates), path)
    return candidates



# ---------------------------------------------------------------------------
# Stage 1: Identify holdout data
# ---------------------------------------------------------------------------

def resolve_db_path(source: str, db_id: str, data_dir: str) -> str:
    if source == "spider":
        return str(Path(data_dir) / "spider" / "database" / db_id / f"{db_id}.sqlite")
    elif source == "bird":
        return str(Path(data_dir) / "bird" / "train" / "train_databases" / db_id / f"{db_id}.sqlite")
    elif source == "synsql":
        return str(Path(data_dir) / "SynSQL-2.5M" / "databases" / db_id / f"{db_id}.sqlite")
    raise ValueError(f"Unknown source: {source}")


def detect_sft_counts(sft_train_jsonl: Optional[str]) -> dict[str, int]:
    """Count how many examples per source were used in SFT training.

    Requires JSONL was saved with --include_metadata (has 'source' field).
    Returns dict like {"spider": 7000, "bird": 9428, "synsql": 200000}.
    Falls back to empty dict if file missing or no metadata.
    """
    if not sft_train_jsonl or not Path(sft_train_jsonl).exists():
        return {}
    counts = {}
    try:
        with open(sft_train_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                src = d.get("source", "")
                if src:
                    counts[src] = counts.get(src, 0) + 1
    except Exception as e:
        logger.warning("Could not detect SFT counts from %s: %s", sft_train_jsonl, e)
        return {}
    logger.info("Auto-detected SFT counts: %s", counts)
    return counts


def classify_complexity(sql: str, synsql_label: str = "") -> str:
    """Return one of: simple, moderate, complex, highly_complex."""
    if synsql_label:
        label_map = {
            "Simple": "simple",
            "Moderate": "moderate",
            "Complex": "complex",
            "Highly Complex": "highly_complex",
        }
        mapped = label_map.get(synsql_label, "")
        if mapped:
            return mapped
    # Fallback to regex
    return "complex" if is_complex_sql(sql) else "simple"


def extract_tables_from_schema(schema_context: str) -> str:
    """Extract comma-separated table names from schema context string."""
    tables = re.findall(r'CREATE\s+TABLE\s+(\w+)', schema_context, re.IGNORECASE)
    return ",".join(tables)


def identify_holdout(args) -> list[RLCandidate]:
    """Stage 1: Load holdout data not used in SFT, filter by complexity."""
    logger.info("=" * 60)
    logger.info("STAGE 1: Identifying holdout data")
    logger.info("=" * 60)

    # Auto-detect SFT cutoffs if metadata available
    sft_counts = detect_sft_counts(getattr(args, "sft_train_jsonl", None))

    source_configs = []

    # --- Spider ---
    spider_sft = sft_counts.get("spider", args.sft_max_spider)
    spider_omni = Path(args.data_dir) / "train_spider.json"
    spider_orig = Path(args.data_dir) / "spider" / "train_spider.json"
    if spider_omni.exists() and spider_orig.exists() and spider_sft is not None:
        source_configs.append(("spider", spider_omni, spider_orig, spider_sft))
    else:
        logger.info("Skipping spider holdout (sft_max_spider not set or files missing)")

    # --- BIRD ---
    bird_sft = sft_counts.get("bird", args.sft_max_bird)
    bird_omni = Path(args.data_dir) / "train_bird.json"
    bird_orig = Path(args.data_dir) / "bird" / "train" / "train.json"
    if bird_omni.exists() and bird_orig.exists() and bird_sft is not None:
        source_configs.append(("bird", bird_omni, bird_orig, bird_sft))
    else:
        logger.info("Skipping bird holdout (sft_max_bird not set or files missing)")

    # --- SynSQL (primary holdout source) ---
    synsql_sft = sft_counts.get("synsql", args.sft_max_synsql)
    synsql_omni = Path(args.data_dir) / "train_synsql.json"
    synsql_orig = Path(args.data_dir) / "SynSQL-2.5M" / "data.json"
    if synsql_omni.exists() and synsql_orig.exists() and synsql_sft is not None:
        source_configs.append(("synsql", synsql_omni, synsql_orig, synsql_sft))
    else:
        logger.warning(
            "SynSQL holdout unavailable (sft_max_synsql=%s, omni=%s, orig=%s)",
            synsql_sft, synsql_omni.exists(), synsql_orig.exists()
        )

    if not source_configs:
        logger.error("No holdout sources configured. Check --sft_max_* args and data paths.")
        sys.exit(1)

    all_candidates: list[RLCandidate] = []
    rng = random.Random(args.seed)

    for source, omni_path, orig_path, sft_cutoff in source_configs:
        logger.info("Processing %s holdout (SFT used first %d records)...", source, sft_cutoff)

        # Load original dataset for db_id and gold_sql (starting from cutoff)
        orig_data = load_json_streaming(str(orig_path), skip=sft_cutoff)
        # Load OmniSQL processed data for schema + question (starting from cutoff)
        omni_data = load_json_streaming(str(omni_path), skip=sft_cutoff)

        n = min(len(orig_data), len(omni_data))
        logger.info("  %s holdout records available: %d", source, n)

        source_candidates: list[RLCandidate] = []
        skipped = 0

        for i in range(n):
            orig = orig_data[i]
            omni = omni_data[i]

            # Extract db_id and gold_sql from original dataset
            db_id = orig.get("db_id", "")
            if source == "spider":
                gold_sql = orig.get("query", "")
            elif source == "bird":
                gold_sql = orig.get("SQL", orig.get("query", ""))
            else:  # synsql
                gold_sql = orig.get("sql", orig.get("query", ""))

            if not db_id or not gold_sql:
                skipped += 1
                continue

            # Complexity filter (SynSQL has explicit label)
            synsql_label = orig.get("sql_complexity", "") if source == "synsql" else ""
            complexity = classify_complexity(gold_sql, synsql_label)

            complexity_rank = {"simple": 0, "moderate": 1, "complex": 2, "highly_complex": 3}
            min_rank = complexity_rank.get(args.min_complexity, 1)
            if complexity_rank.get(complexity, 0) < min_rank:
                continue

            # Parse schema + question from OmniSQL input_seq
            input_seq = omni.get("input_seq", "")
            if not input_seq:
                skipped += 1
                continue
            schema_context, question = parse_input_seq(input_seq)
            if not schema_context or not question:
                skipped += 1
                continue

            # Resolve SQLite db path and validate
            try:
                db_path = resolve_db_path(source, db_id, args.data_dir)
            except ValueError:
                skipped += 1
                continue
            if not Path(db_path).exists():
                skipped += 1
                continue

            schema_tables = extract_tables_from_schema(schema_context)

            source_candidates.append(RLCandidate(
                index=sft_cutoff + i,
                source=source,
                db_id=db_id,
                question=question,
                gold_sql=gold_sql,
                schema_context=schema_context,
                db_path=db_path,
                complexity=complexity,
                schema_tables=schema_tables,
            ))

        logger.info("  %s: %d candidates (skipped %d)", source, len(source_candidates), skipped)
        all_candidates.extend(source_candidates)

    # Shuffle and cap
    rng.shuffle(all_candidates)
    if args.max_holdout and len(all_candidates) > args.max_holdout:
        all_candidates = all_candidates[:args.max_holdout]
        logger.info("Capped holdout to %d records", args.max_holdout)

    complexity_dist = {}
    for c in all_candidates:
        complexity_dist[c.complexity] = complexity_dist.get(c.complexity, 0) + 1
    logger.info("Holdout complexity distribution: %s", complexity_dist)
    logger.info("Total holdout candidates: %d", len(all_candidates))

    return all_candidates



# ---------------------------------------------------------------------------
# Stage 2: Generate candidates with SFT model
# ---------------------------------------------------------------------------

def extract_sql_from_output(text: str) -> str:
    """Extract SQL from model output (Qwen3 format or legacy)."""
    match = FULL_FORMAT_RE.search(text)
    if match:
        return match.group(1).strip()
    match = SQL_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()
    for line in reversed(text.strip().split("\n")):
        if line.strip().upper().startswith("SQL:"):
            return line.strip()[4:].strip()
    return ""


def generate_candidates(holdout: list[RLCandidate], args) -> list[RLCandidate]:
    """Stage 2: Generate N SQL candidates per holdout example using the SFT model."""
    logger.info("=" * 60)
    logger.info("STAGE 2: Generating %d candidates per example", args.num_candidates)
    logger.info("=" * 60)

    from src.generation.inference import SQLInference

    inference = SQLInference(
        model_path=args.sft_model_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.generation_temperature,
    )

    checkpoint_path = Path(args.output_dir) / "stage2_candidates.jsonl"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: skip already-processed examples
    done_indices: set[int] = set()
    existing: list[RLCandidate] = []
    if checkpoint_path.exists():
        existing = load_rl_candidates(checkpoint_path)
        done_indices = {c.index for c in existing}
        logger.info("Resuming Stage 2: %d already processed", len(existing))

    remaining = [c for c in holdout if c.index not in done_indices]
    logger.info("Remaining to process: %d", len(remaining))

    batch_size = args.batch_size
    checkpoint_every = max(1, 1000 // batch_size)

    with open(checkpoint_path, "a", encoding="utf-8") as ckpt_f:
        for batch_start in range(0, len(remaining), batch_size):
            batch = remaining[batch_start: batch_start + batch_size]
            for candidate in batch:
                try:
                    raw_preds = inference.generate_n(
                        question=candidate.question,
                        schema_context=candidate.schema_context,
                        n=args.num_candidates,
                        temperature=args.generation_temperature,
                    )
                    candidate.predictions = [r["raw_output"] for r in raw_preds]
                    candidate.pred_sqls = [r["sql"] for r in raw_preds]
                except Exception as e:
                    logger.warning("Generation failed for index %d: %s", candidate.index, e)
                    candidate.predictions = []
                    candidate.pred_sqls = []

            # Checkpoint this batch
            for c in batch:
                ckpt_f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
            ckpt_f.flush()

            processed = batch_start + len(batch)
            if processed % (checkpoint_every * batch_size) == 0 or processed == len(remaining):
                logger.info("Stage 2: %d/%d processed", processed, len(remaining))

    # Reload full checkpoint
    all_candidates = load_rl_candidates(checkpoint_path)
    logger.info("Stage 2 complete: %d candidates processed", len(all_candidates))
    return all_candidates


# ---------------------------------------------------------------------------
# Stage 3: Split by execution accuracy
# ---------------------------------------------------------------------------

def execution_accuracy_with_error(
    predicted_sql: str,
    gold_sql: str,
    db_path: str,
    timeout: int = 5,
) -> tuple[bool, str]:
    """Execute predicted SQL vs gold SQL against real SQLite DB.

    Returns (match: bool, error_message: str).
    """
    if not predicted_sql.strip():
        return False, "empty SQL"
    if not Path(db_path).exists():
        return False, f"database not found: {db_path}"
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True,
                               timeout=timeout, check_same_thread=False)
        conn.execute("PRAGMA query_only = true")
        cursor = conn.cursor()
        cursor.execute(gold_sql)
        gold_results = set(map(tuple, cursor.fetchall()))
        cursor.execute(predicted_sql)
        pred_results = set(map(tuple, cursor.fetchall()))
        conn.close()
        return (gold_results == pred_results), ""
    except sqlite3.OperationalError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


def split_by_execution(candidates: list[RLCandidate]) -> tuple[list[RLCandidate], list[RLCandidate]]:
    """Stage 3: Split candidates by real execution accuracy."""
    logger.info("=" * 60)
    logger.info("STAGE 3: Splitting by execution accuracy")
    logger.info("=" * 60)

    ex_true: list[RLCandidate] = []
    ex_false: list[RLCandidate] = []

    for i, candidate in enumerate(candidates):
        candidate.ex_results = []
        candidate.pred_errors = []

        for pred_sql in candidate.pred_sqls:
            ok, err = execution_accuracy_with_error(pred_sql, candidate.gold_sql, candidate.db_path)
            candidate.ex_results.append(ok)
            candidate.pred_errors.append(err)

        candidate.any_ex_pass = any(candidate.ex_results)

        if candidate.any_ex_pass:
            ex_true.append(candidate)
        else:
            ex_false.append(candidate)

        if (i + 1) % 1000 == 0:
            logger.info("Stage 3: %d/%d evaluated (pass=%d, fail=%d)",
                        i + 1, len(candidates), len(ex_true), len(ex_false))

    pass_rate = len(ex_true) / max(len(candidates), 1) * 100
    logger.info("Stage 3 complete: EX=True %d (%.1f%%), EX=False %d (%.1f%%)",
                len(ex_true), pass_rate, len(ex_false), 100 - pass_rate)

    # Select best broken SQL for each failed candidate (highest token overlap with gold)
    for candidate in ex_false:
        if not candidate.pred_sqls:
            candidate.best_broken_sql = ""
            candidate.best_broken_error = "no predictions generated"
            continue
        best_idx = 0
        best_overlap = -1.0
        gold_tokens = set(candidate.gold_sql.lower().split())
        for j, sql in enumerate(candidate.pred_sqls):
            if not sql:
                continue
            pred_tokens = set(sql.lower().split())
            overlap = len(gold_tokens & pred_tokens) / max(len(gold_tokens), 1)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = j
        candidate.best_broken_sql = candidate.pred_sqls[best_idx]
        candidate.best_broken_error = candidate.pred_errors[best_idx] if candidate.pred_errors else ""

    return ex_true, ex_false



# ---------------------------------------------------------------------------
# Stage 4: Teacher error correction
# ---------------------------------------------------------------------------

TEACHER_PROMPT_TEMPLATE = """\
You are an expert SQL database engineer and debugger.

A student model attempted to write SQL for the following task but produced an incorrect query.
Your job is to diagnose the error and produce the corrected SQL with structured reasoning.

Database Schema:
{schema_context}

Question: {question}

Student's Broken SQL:
```sql
{broken_sql}
```

Error / Issue: {error_message}

Correct SQL for reference:
```sql
{gold_sql}
```

Respond in EXACTLY this format (no extra text before or after):
<think>
[DIAGNOSIS] Identify the specific error: what is wrong with the student's SQL?
[ANALYSIS] Explain why this error occurs given the schema and question.
[CORRECTION] Describe the fix: what needs to change and why.
</think>

```sql
[Corrected SQL query]
```"""


def _parse_teacher_response(response: str) -> tuple[str, str]:
    """Extract (reasoning, corrected_sql) from teacher response."""
    reasoning = ""
    corrected_sql = ""

    think_match = THINK_BLOCK_RE.search(response)
    if think_match:
        reasoning = think_match.group(1).strip()

    sql_match = SQL_BLOCK_RE.search(response)
    if sql_match:
        corrected_sql = sql_match.group(1).strip()

    return reasoning, corrected_sql


async def _call_teacher(
    candidate: RLCandidate,
    model: str,
    sem: asyncio.Semaphore,
    max_tokens: int = 1024,
) -> RLCandidate:
    """Call teacher model for a single failed candidate."""
    try:
        from litellm import acompletion
    except ImportError:
        logger.error("litellm not installed. pip install litellm")
        return candidate

    prompt = TEACHER_PROMPT_TEMPLATE.format(
        schema_context=candidate.schema_context,
        question=candidate.question,
        broken_sql=candidate.best_broken_sql or "(no SQL generated)",
        error_message=candidate.best_broken_error or "incorrect results",
        gold_sql=candidate.gold_sql,
    )

    async with sem:
        try:
            response = await acompletion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=max_tokens,
            )
            raw = response.choices[0].message.content.strip()
            reasoning, corrected_sql = _parse_teacher_response(raw)
            candidate.teacher_reasoning = reasoning
            candidate.teacher_corrected_sql = corrected_sql
        except Exception as e:
            logger.error("Teacher call failed for index %d: %s", candidate.index, e)

    return candidate


async def teacher_error_correction_api(
    ex_false: list[RLCandidate],
    args,
) -> list[RLCandidate]:
    """Stage 4 Mode A: Call teacher model via litellm API."""
    logger.info("Stage 4 (api): calling %s for %d failed cases", args.teacher_model, len(ex_false))

    sem = asyncio.Semaphore(args.teacher_concurrency)
    tasks = [_call_teacher(c, args.teacher_model, sem, args.teacher_max_tokens) for c in ex_false]
    results = await asyncio.gather(*tasks)

    success = sum(1 for c in results if c.teacher_corrected_sql)
    logger.info("Stage 4 complete: teacher produced corrections for %d/%d", success, len(results))
    return list(results)


def teacher_error_correction_export(ex_false: list[RLCandidate], args) -> list[RLCandidate]:
    """Stage 4 Mode B: Export failed cases for manual Claude Code / Codex CLI processing."""
    export_path = Path(args.output_dir) / "stage4_export_for_manual.jsonl"
    export_path.parent.mkdir(parents=True, exist_ok=True)

    exported = []
    with open(export_path, "w", encoding="utf-8") as f:
        for c in ex_false:
            stable_hash = hashlib.md5(c.question.encode()).hexdigest()[:8]
            rec_id = f"{c.db_id}__{stable_hash}"

            prompt_for_teacher = TEACHER_PROMPT_TEMPLATE.format(
                schema_context=c.schema_context,
                question=c.question,
                broken_sql=c.best_broken_sql or "(no SQL generated)",
                error_message=c.best_broken_error or "incorrect results",
                gold_sql=c.gold_sql,
            )

            export_rec = {
                "id": rec_id,
                "index": c.index,
                "source": c.source,
                "db_id": c.db_id,
                "schema_context": c.schema_context,
                "question": c.question,
                "broken_sql": c.best_broken_sql,
                "error_message": c.best_broken_error,
                "gold_sql": c.gold_sql,
                "prompt_for_teacher": prompt_for_teacher,
            }
            f.write(json.dumps(export_rec, ensure_ascii=False) + "\n")
            exported.append(rec_id)

    logger.info(
        "Stage 4 export: %d cases written to %s\n"
        "  → Process with Claude Code / Codex CLI, then save results to:\n"
        "  → data/rl/stage4_manual_results.jsonl\n"
        "  Format: {\"id\": \"...\", \"teacher_reasoning\": \"...\", \"teacher_corrected_sql\": \"...\"}",
        len(exported), export_path
    )
    return ex_false  # Return unchanged; Stage 5 will merge manual results


def load_manual_results(results_path: str, candidates: list[RLCandidate]) -> list[RLCandidate]:
    """Merge manually-produced teacher results back into candidates."""
    if not Path(results_path).exists():
        logger.warning("Manual results file not found: %s", results_path)
        return candidates

    results_map: dict[str, dict] = {}
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            results_map[d["id"]] = d

    merged = 0
    for c in candidates:
        stable_hash = hashlib.md5(c.question.encode()).hexdigest()[:8]
        rec_id = f"{c.db_id}__{stable_hash}"
        if rec_id in results_map:
            c.teacher_reasoning = results_map[rec_id].get("teacher_reasoning", "")
            c.teacher_corrected_sql = results_map[rec_id].get("teacher_corrected_sql", "")
            merged += 1

    logger.info("Merged manual teacher results: %d/%d candidates updated", merged, len(candidates))
    return candidates



# ---------------------------------------------------------------------------
# Stage 5: Assemble GRPO shard
# ---------------------------------------------------------------------------

def _build_standard_prompt(schema_context: str, question: str) -> str:
    return (
        "Given the database schema below, generate SQL to answer the question.\n\n"
        f"Schema:\n{schema_context}\n\n"
        f"Question: {question}"
    )


def _build_error_correction_prompt(
    schema_context: str,
    question: str,
    broken_sql: str,
    error_message: str,
) -> str:
    parts = [
        "The following SQL query was generated to answer a question but produces incorrect results.",
        "Fix the SQL query so that it correctly answers the question.",
        "",
        f"Schema:\n{schema_context}",
        "",
        f"Question: {question}",
        "",
        "Broken SQL:",
        "```sql",
        broken_sql or "(no SQL generated)",
        "```",
    ]
    if error_message:
        parts += ["", f"Error: {error_message}"]
    parts += ["", "Diagnose the error, explain the fix, then generate the corrected SQL."]
    return "\n".join(parts)


def assemble_grpo_shard(
    ex_true: list[RLCandidate],
    ex_corrected: list[RLCandidate],
    args,
) -> None:
    """Stage 5: Merge EX=True shard and error-correction shard into final GRPO JSONL."""
    logger.info("=" * 60)
    logger.info("STAGE 5: Assembling GRPO shard")
    logger.info("=" * 60)

    output_path = Path(args.output_dir) / "grpo_train.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []

    # --- Type A: standard text2sql (from EX=True) ---
    for c in ex_true:
        # Use the first passing prediction's prompt (same as gold prompt)
        rec = {
            "prompt": _build_standard_prompt(c.schema_context, c.question),
            "answer": c.gold_sql,
            "schema_tables": c.schema_tables,
            "task_type": "text2sql",
            "db_id": c.db_id,
            "db_path": c.db_path,
        }
        records.append(rec)

    # --- Type B: error correction (from Stage 4) ---
    max_ec = int(len(records) * args.error_correction_ratio / (1 - args.error_correction_ratio + 1e-9))
    ec_count = 0
    for c in ex_corrected:
        if not c.best_broken_sql:
            continue
        if args.error_correction_ratio < 1.0 and ec_count >= max_ec:
            break
        rec = {
            "prompt": _build_error_correction_prompt(
                c.schema_context, c.question,
                c.best_broken_sql, c.best_broken_error,
            ),
            "answer": c.gold_sql,
            "schema_tables": c.schema_tables,
            "task_type": "error_correction",
            "db_id": c.db_id,
            "db_path": c.db_path,
            "teacher_reasoning": c.teacher_reasoning,
        }
        records.append(rec)
        ec_count += 1

    # Shuffle the combined shard
    random.Random(args.seed).shuffle(records)

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_text2sql = sum(1 for r in records if r["task_type"] == "text2sql")
    n_ec = sum(1 for r in records if r["task_type"] == "error_correction")
    logger.info("GRPO shard saved to %s", output_path)
    logger.info("  text2sql:        %d (%.1f%%)", n_text2sql, 100 * n_text2sql / max(len(records), 1))
    logger.info("  error_correction: %d (%.1f%%)", n_ec, 100 * n_ec / max(len(records), 1))
    logger.info("  total:            %d", len(records))

    # Save metadata
    meta = {
        "total": len(records),
        "text2sql": n_text2sql,
        "error_correction": n_ec,
        "ex_true_available": len(ex_true),
        "ex_false_available": len(ex_corrected),
        "error_correction_ratio_target": args.error_correction_ratio,
    }
    meta_path = Path(args.output_dir) / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info("Metadata saved to %s", meta_path)



# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_stages(stages_str: str) -> set[int]:
    return {int(s.strip()) for s in stages_str.split(",")}


def main():
    parser = argparse.ArgumentParser(
        description="Build GRPO RL training data from SFT holdout examples."
    )

    # Data paths
    parser.add_argument("--data_dir", default="datasets/data",
                        help="Root dir of OmniSQL datasets/data")
    parser.add_argument("--sft_train_jsonl", default=None,
                        help="Path to SFT train.jsonl (for auto-detecting SFT counts, requires --include_metadata)")
    parser.add_argument("--output_dir", default="data/rl",
                        help="Output directory for all stage outputs")

    # Stage 1 args
    parser.add_argument("--sft_max_spider", type=int, default=None,
                        help="How many spider examples were used for SFT (holdout starts after this index)")
    parser.add_argument("--sft_max_bird", type=int, default=None,
                        help="How many bird examples were used for SFT")
    parser.add_argument("--sft_max_synsql", type=int, default=200000,
                        help="How many synsql examples were used for SFT (default: 200000)")
    parser.add_argument("--max_holdout", type=int, default=50000,
                        help="Max holdout candidates to process (default: 50000)")
    parser.add_argument("--min_complexity", default="moderate",
                        choices=["simple", "moderate", "complex", "highly_complex"],
                        help="Minimum SQL complexity for holdout filter (default: moderate)")
    parser.add_argument("--seed", type=int, default=42)

    # Stage 2 args
    parser.add_argument("--sft_model_path", default="./checkpoints/sft/final",
                        help="Path to SFT model checkpoint")
    parser.add_argument("--num_candidates", type=int, default=5,
                        help="Number of SQL candidates per question (default: 5)")
    parser.add_argument("--generation_temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)

    # Stage 4 args
    parser.add_argument("--teacher_mode", default="api", choices=["api", "export"],
                        help="'api': auto call litellm teacher; 'export': write cases for manual processing")
    parser.add_argument("--teacher_model", default="gpt-4o-mini",
                        help="LiteLLM model string for teacher (e.g. gpt-4o-mini, claude-sonnet-4-20250514, deepseek/deepseek-chat)")
    parser.add_argument("--teacher_concurrency", type=int, default=10)
    parser.add_argument("--teacher_max_tokens", type=int, default=1024)
    parser.add_argument("--stage4_results_path", default=None,
                        help="Path to manual results JSONL (used with --teacher_mode export after manual processing)")

    # Stage 5 args
    parser.add_argument("--error_correction_ratio", type=float, default=0.3,
                        help="Target fraction of error-correction examples in final shard (default: 0.3)")

    # Pipeline control
    parser.add_argument("--stages", default="1,2,3,4,5",
                        help="Comma-separated list of stages to run (default: 1,2,3,4,5)")

    args = parser.parse_args()
    stages = parse_stages(args.stages)
    output_dir = Path(args.output_dir)

    # ---- Stage 1 ----
    stage1_path = output_dir / "stage1_holdout.jsonl"
    if 1 in stages:
        holdout = identify_holdout(args)
        save_jsonl(holdout, stage1_path)
    else:
        holdout = load_rl_candidates(stage1_path)

    # ---- Stage 2 ----
    stage2_path = output_dir / "stage2_candidates.jsonl"
    if 2 in stages:
        holdout = generate_candidates(holdout, args)
        # Note: generate_candidates already checkpoints internally to stage2_candidates.jsonl
    elif 2 not in stages and stage2_path.exists():
        holdout = load_rl_candidates(stage2_path)

    # ---- Stage 3 ----
    stage3_true_path = output_dir / "stage3_ex_true.jsonl"
    stage3_false_path = output_dir / "stage3_ex_false.jsonl"
    if 3 in stages:
        ex_true, ex_false = split_by_execution(holdout)
        save_jsonl(ex_true, stage3_true_path)
        save_jsonl(ex_false, stage3_false_path)
    else:
        ex_true = load_rl_candidates(stage3_true_path) if stage3_true_path.exists() else []
        ex_false = load_rl_candidates(stage3_false_path) if stage3_false_path.exists() else []

    # ---- Stage 4 ----
    stage4_path = output_dir / "stage4_corrected.jsonl"
    if 4 in stages:
        if args.teacher_mode == "api":
            ex_corrected = asyncio.run(teacher_error_correction_api(ex_false, args))
            save_jsonl(ex_corrected, stage4_path)
        else:  # export
            ex_corrected = teacher_error_correction_export(ex_false, args)
            # If manual results are provided, merge them
            if args.stage4_results_path:
                ex_corrected = load_manual_results(args.stage4_results_path, ex_corrected)
                save_jsonl(ex_corrected, stage4_path)
            else:
                logger.info(
                    "Stage 4 export done. After manual processing, re-run with:\n"
                    "  --stages 4,5 --teacher_mode export "
                    "--stage4_results_path data/rl/stage4_manual_results.jsonl"
                )
                return  # Stop here — Stage 5 needs teacher results
    else:
        ex_corrected = load_rl_candidates(stage4_path) if stage4_path.exists() else []

    # ---- Stage 5 ----
    if 5 in stages:
        assemble_grpo_shard(ex_true, ex_corrected, args)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
