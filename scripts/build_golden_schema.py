"""
Build Golden Schema Context for Text-to-SQL Training & KD.

This script parses the Gold SQL queries from the Spider dataset, extracts the 
exact tables and columns used (Golden Schema), and mixes them with 
randomly selected unused tables/columns (Noise).

The result is a "Golden Schema Context" that can be fed into:
1. Teacher LLM (GPT-4o) during Knowledge Distillation (KD) to generate reasoning.
2. SLM (Qwen3) during SFT to learn robust schema linking.

Usage:
    python scripts/build_golden_schema.py \
        --dataset_path data/spider/train_spider.json \
        --tables_path data/spider/tables.json \
        --output_path data/schema_contexts_golden.json \
        --noise_table_ratio 0.5
"""

import argparse
import hashlib
import json
import logging
import random
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Basic regex patterns for Spider SQL
TABLE_RE = re.compile(r'\b(?:FROM|JOIN)\s+([a-zA-Z0-9_]+)\b', re.IGNORECASE)
COLUMN_RE = re.compile(r'\b([a-zA-Z0-9_]+)\b', re.IGNORECASE)


class SpiderDatabase:
    """Helper class to manage and query Spider database schema."""
    def __init__(self, db_dict: dict):
        self.db_id = db_dict["db_id"]
        self.table_names = [t.lower() for t in db_dict["table_names_original"]]
        
        # Build column map: table_index -> list of column names
        self.columns = {i: [] for i in range(len(self.table_names))}
        self.primary_keys = db_dict["primary_keys"]
        self.column_types = db_dict["column_types"]
        self.all_cols = [] # (table_idx, name, type, is_pk)
        
        for j, (t_idx, c_name) in enumerate(db_dict["column_names_original"]):
            # t_idx == -1 means * (all columns)
            if t_idx >= 0:
                is_pk = j in self.primary_keys
                ctype = self.column_types[j]
                self.columns[t_idx].append((c_name.lower(), ctype, is_pk))
                self.all_cols.append((t_idx, c_name.lower(), ctype, is_pk))


def extract_gold_schema(sql: str, db: SpiderDatabase) -> tuple[set[int], set[str]]:
    """
    Extract gold tables and columns by simply matching tokens in the SQL
    against the known schema.
    Returns:
        gold_table_idxs: set of table indices used in the SQL.
        gold_columns: set of lowercase column names used in the SQL.
    """
    sql_lower = sql.lower()
    
    # 1. Look for table names following FROM or JOIN
    found_tables = TABLE_RE.findall(sql_lower)
    gold_table_idxs = set()
    for t_name in found_tables:
        t_name = t_name.strip()
        if t_name in db.table_names:
            gold_table_idxs.add(db.table_names.index(t_name))
            
    # 2. Heuristic fallback: if from/join parsing missed a table, 
    # check all tokens in SQL simply matching table names.
    tokens = [t.strip() for t in COLUMN_RE.findall(sql_lower)]
    for i, t_name in enumerate(db.table_names):
        if t_name in tokens:
            gold_table_idxs.add(i)
            
    # 3. Extract all valid column names mentioned in SQL
    gold_columns = set()
    for token in tokens:
        for t_idx, c_name, _, _ in db.all_cols:
            if token == c_name:
                gold_columns.add(c_name)
                # Auto-add the table index if a column belonging to it is found
                gold_table_idxs.add(t_idx)
                
    # Fallback to all tables if extraction failed completely
    if not gold_table_idxs and db.table_names:
        gold_table_idxs.add(0) 

    return gold_table_idxs, gold_columns


def build_schema_context(
    db: SpiderDatabase, 
    gold_table_idxs: set[int], 
    gold_columns: set[str],
    noise_table_ratio: float = 0.5
) -> str:
    """
    Build the CREATE TABLE strings for the Golden Context.
    Includes all gold tables + a fraction of random unused (noise) tables.
    Also injects ALL columns for selected tables (so model has to pick the right column).
    """
    all_idxs = set(range(len(db.table_names)))
    unused_idxs = list(all_idxs - gold_table_idxs)
    
    # Calculate how many noise tables to add
    num_noise = int(len(unused_idxs) * noise_table_ratio)
    noise_idxs = set(random.sample(unused_idxs, num_noise)) if num_noise > 0 else set()
    
    final_table_idxs = list(gold_table_idxs | noise_idxs)
    random.shuffle(final_table_idxs) # Shuffle so gold isn't always first
    
    lines = [f"Database: {db.db_id}", ""]
    
    for t_idx in final_table_idxs:
        t_name = db.table_names[t_idx]
        cols = []
        for c_name, ctype, is_pk in db.columns[t_idx]:
            # Always include the column (don't filter columns so model learns column-linking)
            cols.append(f"{c_name} ({ctype})" + (" [PK]" if is_pk else ""))
            
        lines.append(f"CREATE TABLE {t_name} ({', '.join(cols)})")
        
    return "\n".join(lines)


def process_dataset(examples: list, tables_data: list, noise_table_ratio: float) -> dict:
    """Process all examples and return a dictionary mapping 'db_id__hash' -> schema_context."""
    # Index databases for fast lookup
    db_dict = {db["db_id"]: SpiderDatabase(db) for db in tables_data}
    
    schema_contexts = {}
    
    for ex in examples:
        db_id = ex["db_id"]
        question = ex["question"]
        sql = ex["query"]
        
        if db_id not in db_dict:
            continue
            
        db = db_dict[db_id]
        
        # Stable hash key
        stable_hash = hashlib.md5(question.encode("utf-8")).hexdigest()[:8]
        key = f"{db_id}__{stable_hash}"
        
        # Extract Gold
        gold_table_idxs, gold_columns = extract_gold_schema(sql, db)
        
        # Build Context with Noise
        context = build_schema_context(db, gold_table_idxs, gold_columns, noise_table_ratio)
        schema_contexts[key] = context
        
    return schema_contexts


def main():
    parser = argparse.ArgumentParser(description="Build Golden Schema Contexts for SFT/KD.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to raw json dataset (e.g. train_spider.json)")
    parser.add_argument("--tables_path", type=str, required=True, help="Path to tables.json")
    parser.add_argument("--output_path", type=str, default="data/schema_contexts_golden.json", help="Path to save output JSON dict")
    parser.add_argument("--noise_table_ratio", type=float, default=0.5, help="Ratio of unused tables to include as noise (0.0 to 1.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)

    logger.info(f"Loading dataset from {args.dataset_path}")
    with open(args.dataset_path, "r", encoding="utf-8") as f:
        examples = json.load(f)

    logger.info(f"Loading tables from {args.tables_path}")
    with open(args.tables_path, "r", encoding="utf-8") as f:
        tables_data = json.load(f)

    logger.info("Extracting gold schemas and injecting noise...")
    schema_contexts = process_dataset(examples, tables_data, args.noise_table_ratio)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(schema_contexts, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Successfully generated {len(schema_contexts)} golden schema contexts.")
    logger.info(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
