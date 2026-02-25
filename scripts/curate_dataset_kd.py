"""
Knowledge Distillation (KD) Pipeline for Text-to-SQL Reasoning.

Uses a large teacher model (e.g., GPT-5, Claude 4.5, DeepSeek-V3) to 
generate step-by-step reasoning traces for text-to-SQL tasks.
The generated reasoning replaces the pseudo-rules in DataFormatter.

Usage:
    python scripts/curate_dataset_kd.py \
        --dataset_path data/spider/train_spider.json \
        --tables_path data/spider/tables.json \
        --schema_contexts_path data/schema_contexts_train.json \
        --output_path data/reasoning_cache.json \
        --model gpt-4o-mini
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys

# Using litellm for multi-provider support (OpenAI, Anthropic, Gemini, etc.)
# pip install litellm
try:
    from litellm import acompletion
except ImportError:
    print("Please install litellm: pip install litellm")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_schema_prompt(db_id: str, tables_data: list) -> str:
    """Generate basic schema string from spider tables.json data."""
    db_data = next((db for db in tables_data if db["db_id"] == db_id), None)
    if not db_data:
        return f"Database: {db_id}\n(Schema not found)"

    lines = [f"Database: {db_id}", ""]
    table_names = db_data["table_names_original"]
    col_names = db_data["column_names_original"]
    col_types = db_data["column_types"]
    primary_keys = db_data["primary_keys"]

    for i, t_name in enumerate(table_names):
        cols = []
        for j, (col_t_id, col_name) in enumerate(col_names):
            if col_t_id == i:
                is_pk = j in primary_keys
                ctype = col_types[j]
                cols.append(f"{col_name} ({ctype})" + (" [PK]" if is_pk else ""))
        lines.append(f"CREATE TABLE {t_name} ({', '.join(cols)})")

    return "\n".join(lines)


async def generate_reasoning(
    question: str,
    schema: str,
    gold_sql: str,
    model: str,
    sem: asyncio.Semaphore,
) -> str:
    """Call LLM API to reverse-engineer the reasoning."""
    prompt = f"""You are an expert SQL database engineer.
Given the database schema below and the question, someone has already written the correct SQL answer.
Your task is to write the step-by-step reasoning that leads to this correct SQL answer.
Explain which tables to use, how to join them, what conditions to apply, and why certain functions (like GROUB BY, ORDER BY, etc.) are used.

Schema:
{schema}

Question:
{question}

Correct SQL:
{gold_sql}

Please output ONLY the step-by-step reasoning in plain text. Do NOT wrap it in <think> tags. Do NOT repeat the Correct SQL at the end. Keep it concise and logical.
"""
    async with sem:
        try:
            response = await acompletion(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3, # Low temp for deterministic logic
                max_tokens=512,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating reasoning: {e}")
            return ""


async def process_batch(
    examples: list,
    tables_data: list,
    schema_contexts: dict,
    model: str,
    concurrency: int,
) -> dict:
    """Process all examples concurrently with a concurrency limit."""
    sem = asyncio.Semaphore(concurrency)
    cache = {}
    tasks = []
    keys = []

    for ex in examples:
        db_id = ex["db_id"]
        question = ex["question"]
        gold_sql = ex["query"]
        
        # Stable key matching the one in DataFormatter: db_id__MD5(question)
        stable_hash = hashlib.md5(question.encode("utf-8")).hexdigest()[:8]
        key = f"{db_id}__{stable_hash}"
        
        # KEY FIX: Use the retrieved Schema Context if available. 
        # This ensures the Teacher Model sees the exact same filtered schema as the SLM.
        if key in schema_contexts:
            schema = schema_contexts[key]
        else:
            # Fallback to full schema only if context is missing
            schema = generate_schema_prompt(db_id, tables_data)
            
        tasks.append(generate_reasoning(question, schema, gold_sql, model, sem))
        keys.append(key)

    logger.info(f"Starting API calls for {len(tasks)} examples...")
    results = await asyncio.gather(*tasks)

    for key, reasoning in zip(keys, results):
        if reasoning:
            cache[key] = reasoning

    return cache


def main():
    parser = argparse.ArgumentParser(description="Curate dataset with LLM reasoning (Knowledge Distillation).")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to raw json dataset (e.g. train_spider.json)")
    parser.add_argument("--tables_path", type=str, required=True, help="Path to tables.json (for schema)")
    parser.add_argument("--schema_contexts_path", type=str, default=None, help="Path to retrieved schema contexts JSON (generated by retrieval pipeline)")
    parser.add_argument("--output_path", type=str, default="data/reasoning_cache.json", help="Path to save output JSON dict")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LiteLLM model string")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent API requests")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N examples for testing")
    args = parser.parse_args()

    # Verify litellm keys are set (e.g. OPENAI_API_KEY)
    if "gpt" in args.model.lower() and not os.environ.get("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY environment variable is not set!")

    logger.info(f"Loading dataset from {args.dataset_path}")
    with open(args.dataset_path, "r", encoding="utf-8") as f:
        examples = json.load(f)

    with open(args.tables_path, "r", encoding="utf-8") as f:
        tables_data = json.load(f)

    # Load retrieved schema contexts if provided
    schema_contexts = {}
    if args.schema_contexts_path and os.path.exists(args.schema_contexts_path):
        logger.info(f"Loading schema contexts from {args.schema_contexts_path}")
        with open(args.schema_contexts_path, "r", encoding="utf-8") as f:
            schema_contexts = json.load(f)
    else:
        logger.warning(
            "No --schema_contexts_path provided. Teacher model will use the FULL database schema. "
            "For best SLM alignment, pass the filtered schemas from the retrieval pipeline."
        )

    if args.limit:
        examples = examples[:args.limit]
        logger.info(f"Limiting to {args.limit} examples")

    cache = asyncio.run(process_batch(
        examples, tables_data, schema_contexts, args.model, args.concurrency
    ))

    logger.info(f"Successfully generated reasoning for {len(cache)}/{len(examples)} examples.")
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved reasoning cache to {args.output_path}")


if __name__ == "__main__":
    main()
