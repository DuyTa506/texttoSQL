"""
Mine Correction Data – generates correction training samples for multitask GRPO.

Flow:
  1. Load SFT checkpoint via SQLInference
  2. Run on train examples (filter by difficulty: medium + hard)
  3. Execute each generated SQL → collect ErrorType via SQLExecutor
  4. Filter: keep only WRONG_RESULT / SYNTAX_ERROR / NO_SUCH_COLUMN errors
  5. For each failure: call teacher LLM with correction prompt
     → teacher returns <think>fix reasoning</think> + corrected SQL
  6. Save as JSONL: {question, schema_context, wrong_sql, error_type,
                     error_message, corrected_sql, correction_reasoning, db_id}

CLI:
  python scripts/mine_correction_data.py \\
    --sft_model_path ./checkpoints/sft/final \\
    --data_path ./data/spider \\
    --db_dir ./data/spider/database \\
    --output_path ./data/correction/train.jsonl \\
    --teacher_model gpt-4o \\
    --max_samples 5000 \\
    --difficulties medium hard

Requirements:
  pip install ".[correction]"   # adds openai or anthropic SDK
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Teacher LLM prompt
# ---------------------------------------------------------------------------

_TEACHER_PROMPT_TEMPLATE = """\
You are an expert SQL developer. A student wrote a SQL query that has an error.
Your task is to analyze the error and generate the corrected SQL.

Schema:
{schema_context}

Question: {question}

Wrong SQL:
```sql
{wrong_sql}
```

Error type: {error_type}
Error message: {error_message}

Please:
1. Think step by step about why the SQL is wrong (<think>your reasoning</think>)
2. Generate the corrected SQL inside ```sql ... ```

Your response MUST follow this exact format:
<think>
[Your analysis of what went wrong and how to fix it]
</think>
```sql
[corrected SQL here]
```"""


def call_teacher_openai(
    prompt: str,
    model: str = "gpt-4o",
    api_key: str | None = None,
) -> str:
    """Call OpenAI API to generate correction reasoning + SQL."""
    try:
        import openai
    except ImportError:
        raise ImportError("pip install openai to use OpenAI teacher models")

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""


def call_teacher_anthropic(
    prompt: str,
    model: str = "claude-opus-4-5",
    api_key: str | None = None,
) -> str:
    """Call Anthropic API to generate correction reasoning + SQL."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("pip install anthropic to use Anthropic teacher models")

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text if message.content else ""


def call_teacher(
    prompt: str,
    model: str,
    api_key: str | None = None,
) -> str:
    """Route to the appropriate teacher LLM API."""
    if model.startswith("gpt") or model.startswith("o1") or model.startswith("o3"):
        return call_teacher_openai(prompt, model=model, api_key=api_key)
    elif model.startswith("claude"):
        return call_teacher_anthropic(prompt, model=model, api_key=api_key)
    else:
        raise ValueError(
            f"Unknown teacher model '{model}'. "
            "Supported: gpt-4o, gpt-4-turbo, claude-opus-4-5, claude-sonnet-4-5, etc."
        )


# ---------------------------------------------------------------------------
# SQL extraction from teacher response
# ---------------------------------------------------------------------------

import re

_SQL_BLOCK_RE = re.compile(r"```sql\s*(.+?)\s*```", re.DOTALL)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def extract_sql_from_teacher(response: str) -> str:
    match = _SQL_BLOCK_RE.search(response)
    return match.group(1).strip() if match else ""


def extract_reasoning_from_teacher(response: str) -> str:
    match = _THINK_RE.search(response)
    return match.group(1).strip() if match else ""


# ---------------------------------------------------------------------------
# Build schema context for a given example
# ---------------------------------------------------------------------------

def build_schema_context(db, example, retriever, linker, schema_filter, augmentor, all_chunks):
    """Run the retrieval pipeline to get schema context for an example."""
    aug_query = augmentor.augment(example.question, db)
    if isinstance(aug_query, list):
        raw_results = retriever.retrieve_multi(aug_query, db_id=example.db_id)
    else:
        raw_results = retriever.retrieve(aug_query, db_id=example.db_id)
    db_chunks = [c for c in all_chunks if c.db_id == example.db_id]
    expanded = linker.expand(raw_results, db, db_chunks)
    schema_context = schema_filter.filter_and_format(expanded, db)
    return schema_context


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    # ---- Load dataset ----
    from src.schema.schema_chunker import SchemaChunker
    from src.schema.schema_indexer import SchemaIndexer
    from src.data_parser import get_parser
    from src.post.sql_executor import ErrorType, SQLExecutor
    from src.retrieval.utils.bidirectional_linker import BidirectionalLinker
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.pre_retrieval.query_augmentor import QueryAugmentor
    from src.retrieval.utils.schema_filter import SchemaFilter

    logger.info("Loading dataset from: %s", args.data_path)
    adapter = get_parser("spider_v1")
    databases, examples = adapter.load(args.data_path)
    db_map = {db.db_id: db for db in databases}
    logger.info("Loaded %d databases, %d examples", len(databases), len(examples))

    # Filter by difficulty
    filtered_examples = [
        ex for ex in examples if ex.difficulty in args.difficulties
    ]
    logger.info(
        "After difficulty filter (%s): %d examples",
        args.difficulties, len(filtered_examples)
    )

    if args.max_samples and len(filtered_examples) > args.max_samples:
        import random
        random.seed(42)
        filtered_examples = random.sample(filtered_examples, args.max_samples)
        logger.info("Subsampled to %d examples", len(filtered_examples))

    # ---- Build retrieval pipeline ----
    chunker = SchemaChunker()
    all_chunks = chunker.chunk_many(databases)

    indexer = SchemaIndexer(
        embedding_model=args.embedding_model,
        persist_dir=args.chroma_dir,
        collection_name="schema_chunks_correction",
    )
    indexer.index(all_chunks, reset=True)

    augmentor = QueryAugmentor(strategy="keyword")
    retriever = HybridRetriever(
        indexer=indexer,
        chunks=all_chunks,
        bm25_top_k=30,
        semantic_top_k=30,
        rrf_k=60,
    )
    linker = BidirectionalLinker(max_expansion_depth=1)
    schema_filter = SchemaFilter(top_k=15)
    executor = SQLExecutor(db_dir=args.db_dir)

    # ---- Load SFT model for inference ----
    from src.generation.inference import SQLInference
    logger.info("Loading SFT model from: %s", args.sft_model_path)
    inference = SQLInference(
        model_path=args.sft_model_path,
        max_new_tokens=512,
        temperature=0.0,
    )

    # ---- Mine correction samples ----
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Track which error types to mine
    target_error_types = {
        ErrorType.WRONG_RESULT,
        ErrorType.SYNTAX_ERROR,
        ErrorType.NO_SUCH_COLUMN,
    }

    collected = 0
    skipped = 0
    errors_by_type: dict[str, int] = {}

    with open(output_path, "w", encoding="utf-8") as out_f:
        for i, example in enumerate(filtered_examples):
            db = db_map.get(example.db_id)
            if db is None:
                skipped += 1
                continue

            logger.info(
                "[%d/%d] Processing: db=%s, q=%s...",
                i + 1, len(filtered_examples),
                example.db_id, example.question[:60]
            )

            # Build schema context
            try:
                schema_context = build_schema_context(
                    db, example, retriever, linker, schema_filter, augmentor, all_chunks
                )
            except Exception as e:
                logger.warning("Schema context error: %s", e)
                skipped += 1
                continue

            # Generate SQL with SFT model
            try:
                gen_result = inference.generate(example.question, schema_context)
                predicted_sql = gen_result.get("sql", "")
            except Exception as e:
                logger.warning("Inference error: %s", e)
                skipped += 1
                continue

            if not predicted_sql:
                skipped += 1
                continue

            # Execute and classify
            exec_result = executor.execute(predicted_sql, example.db_id, gold_sql=example.query)

            # Skip successes and non-target error types
            if exec_result.success or exec_result.error_type not in target_error_types:
                skipped += 1
                continue

            error_name = exec_result.error_type.value
            errors_by_type[error_name] = errors_by_type.get(error_name, 0) + 1

            # Build teacher prompt
            from src.post.retry_loop import _ERROR_HINTS
            hint = _ERROR_HINTS.get(exec_result.error_type, "Review the SQL carefully.")
            teacher_prompt = _TEACHER_PROMPT_TEMPLATE.format(
                schema_context=schema_context,
                question=example.question,
                wrong_sql=predicted_sql,
                error_type=error_name,
                error_message=exec_result.error_message or "(no details)",
            )

            # Call teacher LLM
            try:
                teacher_response = call_teacher(
                    teacher_prompt,
                    model=args.teacher_model,
                    api_key=args.api_key,
                )
                corrected_sql = extract_sql_from_teacher(teacher_response)
                correction_reasoning = extract_reasoning_from_teacher(teacher_response)
            except Exception as e:
                logger.warning("Teacher LLM error: %s", e)
                # Rate-limit friendly: wait and retry once
                time.sleep(5.0)
                try:
                    teacher_response = call_teacher(
                        teacher_prompt,
                        model=args.teacher_model,
                        api_key=args.api_key,
                    )
                    corrected_sql = extract_sql_from_teacher(teacher_response)
                    correction_reasoning = extract_reasoning_from_teacher(teacher_response)
                except Exception as e2:
                    logger.error("Teacher LLM failed twice: %s", e2)
                    skipped += 1
                    continue

            if not corrected_sql:
                logger.debug("Teacher returned no SQL, skipping.")
                skipped += 1
                continue

            # Write correction sample
            sample = {
                "question": example.question,
                "schema_context": schema_context,
                "wrong_sql": predicted_sql,
                "error_type": error_name,
                "error_message": exec_result.error_message,
                "corrected_sql": corrected_sql,
                "correction_reasoning": correction_reasoning,
                "db_id": example.db_id,
                "difficulty": example.difficulty,
            }
            out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            collected += 1

            if collected % 50 == 0:
                logger.info(
                    "Collected %d correction samples so far. Errors: %s",
                    collected, errors_by_type
                )

            # Respect rate limits
            if args.sleep_between_calls > 0:
                time.sleep(args.sleep_between_calls)

    logger.info(
        "=== Mining complete: %d collected, %d skipped ===",
        collected, skipped
    )
    logger.info("Error type distribution: %s", errors_by_type)
    logger.info("Output saved to: %s", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mine correction data from SFT model failures"
    )
    parser.add_argument(
        "--sft_model_path", required=True,
        help="Path to SFT checkpoint to run inference with"
    )
    parser.add_argument(
        "--data_path", required=True,
        help="Path to Spider dataset directory"
    )
    parser.add_argument(
        "--db_dir", required=True,
        help="Path to SQLite database directory"
    )
    parser.add_argument(
        "--output_path", required=True,
        help="Output JSONL path for correction samples"
    )
    parser.add_argument(
        "--teacher_model", default="gpt-4o",
        help="Teacher model for generating corrections (gpt-4o, claude-opus-4-5, etc.)"
    )
    parser.add_argument(
        "--api_key", default=None,
        help="API key for teacher model (or set via OPENAI_API_KEY/ANTHROPIC_API_KEY env)"
    )
    parser.add_argument(
        "--max_samples", type=int, default=5000,
        help="Maximum number of train examples to process"
    )
    parser.add_argument(
        "--difficulties", nargs="+", default=["medium", "hard"],
        help="Difficulty levels to mine (default: medium hard)"
    )
    parser.add_argument(
        "--embedding_model",
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        help="Sentence-transformers model for schema indexing"
    )
    parser.add_argument(
        "--chroma_dir", default="./data/chroma_correction",
        help="ChromaDB persist directory for correction mining"
    )
    parser.add_argument(
        "--sleep_between_calls", type=float, default=0.5,
        help="Seconds to sleep between teacher LLM calls (rate limiting)"
    )

    args = parser.parse_args()
    main(args)
