#!/usr/bin/env python3
"""
Comprehensive Retrieval Analysis on BIRD Dev.

Metrics computed
----------------
Core (set-membership):
  table_recall, table_precision, table_f1
  column_recall, column_precision, column_f1
  perfect_recall%, n_missed, n_noisy

New (v2):
  1. Rank-quality   — NDCG@K, MRR, Hit@K  (how high up the gold tables are ranked)
  2. Join-complexity — recall/precision/F1 split by #gold-tables (1/2/3/4+)
  3. Difficulty      — recall/precision/F1 split by BIRD label (simple/moderate/challenging)
  4. FK coverage     — FK-pair recall (both ends of every needed FK present?)
                       orphaned FK rate (FK col retrieved but target table missing)
  5. Context budget  — avg schema tokens, token recall, redundant-table ratio
  6. Column-role     — recall split into PK / FK / regular columns
  7. Consistency     — recall std-dev per DB, zero-recall rate, exact-match rate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

# ── Project setup ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from src.data_parser import get_parser
from src.schema.schema_chunker import SchemaChunker
from src.schema.schema_indexer import SchemaIndexer
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.utils.bidirectional_linker import BidirectionalLinker
from src.pre_retrieval.query_augmentor import QueryAugmentor
from src.retrieval.utils.schema_filter import SchemaFilter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("retrieval_analysis")

DATA_PATH       = "./datasets/data/bird/dev_20240627"
CONFIG_PATH     = "./configs/default.yaml"
N_EXAMPLES      = 100
CHROMA_DIR_DEFAULT = "./data/chroma_db_bird_analysis"
COLLECTION_NAME    = "bird_analysis_chunks"


# =============================================================================
# OpenAI embedding wrapper
# =============================================================================

class _OpenAIEmbedderWrapper:
    def __init__(self, openai_model):
        self._model = openai_model

    def encode(self, texts, convert_to_numpy=True, **kwargs):
        vecs = self._model.embed(texts)
        if convert_to_numpy:
            return np.array(vecs, dtype=np.float32)
        return vecs


# =============================================================================
# SQL parsing helpers
# =============================================================================

def build_alias_map(sql: str) -> dict[str, str]:
    keywords = {
        "where", "on", "inner", "left", "right", "outer", "cross", "join",
        "natural", "full", "group", "order", "having", "limit", "union",
        "intersect", "except", "set", "and", "or", "as", "select", "from",
        "into", "values", "using",
    }
    alias_map: dict[str, str] = {}
    for m in re.finditer(r"(?:FROM|JOIN)\s+(\w+)\s+AS\s+(\w+)", sql, re.IGNORECASE):
        alias_map[m.group(2).lower()] = m.group(1).lower()
    for m in re.finditer(r"(?:FROM|JOIN)\s+(\w+)\s+(?!AS\b)(\w+)", sql, re.IGNORECASE):
        alias = m.group(2).lower()
        if alias not in keywords:
            alias_map[alias] = m.group(1).lower()
    return alias_map


def extract_gold_tables_and_columns(sql: str) -> tuple[set[str], set[str]]:
    """Return (gold_tables, gold_columns as 'table.col') from SQL."""
    gold_tables = {m.lower() for m in re.findall(r"(?:FROM|JOIN)\s+(\w+)", sql, re.IGNORECASE)}
    alias_map   = build_alias_map(sql)
    for t in gold_tables:
        alias_map[t] = t

    gold_columns: set[str] = set()
    for tbl, col in re.findall(r"(\w+)\.(\w+)", sql):
        tbl_l, col_l = tbl.lower(), col.lower()
        if col_l == "*":
            continue
        real = alias_map.get(tbl_l, tbl_l)
        if real in gold_tables:
            gold_columns.add(f"{real}.{col_l}")
    return gold_tables, gold_columns


def extract_retrieved_tables(retrieved_chunks: list[dict], top_k: int = 15) -> list[str]:
    """Return table names in *rank order* (score descending)."""
    sorted_chunks = sorted(retrieved_chunks, key=lambda x: x.get("score", 0.0), reverse=True)[:top_k]
    seen, ordered = set(), []
    for item in sorted_chunks:
        chunk = item.get("chunk")
        if chunk and chunk.table_name:
            t = chunk.table_name.lower()
            if t not in seen:
                seen.add(t)
                ordered.append(t)
    return ordered


def extract_retrieved_columns(retrieved_chunks: list[dict], db, top_k: int = 15) -> set[str]:
    """Return all columns for retrieved tables (schema-filter expands full tables)."""
    ret_tables = set(extract_retrieved_tables(retrieved_chunks, top_k))
    cols: set[str] = set()
    for table in db.tables:
        if table.name.lower() in ret_tables:
            for col in table.columns:
                cols.add(f"{table.name.lower()}.{col.name.lower()}")
    return cols


# =============================================================================
# Metric helpers
# =============================================================================

def safe_f1(r: float, p: float) -> float:
    return 2 * r * p / (r + p) if (r + p) > 0 else 0.0


def compute_ndcg(gold_tables: set[str], ranked_tables: list[str], k: int = 15) -> float:
    """NDCG@k — gold tables are equally relevant (binary relevance)."""
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, t in enumerate(ranked_tables[:k])
        if t in gold_tables
    )
    ideal = sum(
        1.0 / math.log2(rank + 2)
        for rank in range(min(len(gold_tables), k))
    )
    return dcg / ideal if ideal > 0 else 1.0


def compute_mrr(gold_tables: set[str], ranked_tables: list[str]) -> float:
    """Mean Reciprocal Rank — 1/rank of first gold table."""
    for rank, t in enumerate(ranked_tables, start=1):
        if t in gold_tables:
            return 1.0 / rank
    return 0.0


def compute_hit_at_k(gold_tables: set[str], ranked_tables: list[str], k: int) -> float:
    return 1.0 if any(t in gold_tables for t in ranked_tables[:k]) else 0.0


def count_schema_tokens(schema_context: str) -> int:
    """Approximate token count (whitespace split, good enough for budget estimates)."""
    return len(schema_context.split())


def compute_fk_coverage(
    gold_tables: set[str],
    retrieved_tables: set[str],
    db,
) -> dict:
    """
    FK pair recall: of all FK pairs (A→B) where both A and B are gold tables,
    what % have both ends retrieved?

    Orphaned FK rate: of FK columns in retrieved set whose FK-target table is
    a gold table, how many have their target table missing?
    """
    needed_pairs = [
        fk for fk in db.foreign_keys
        if fk.from_table.lower() in gold_tables and fk.to_table.lower() in gold_tables
    ]
    if needed_pairs:
        both_retrieved = sum(
            1 for fk in needed_pairs
            if fk.from_table.lower() in retrieved_tables
            and fk.to_table.lower() in retrieved_tables
        )
        fk_pair_recall = both_retrieved / len(needed_pairs)
    else:
        fk_pair_recall = 1.0  # no FK needed → trivially satisfied

    # Orphaned FK: retrieved a FK column but its target (gold) table is missing
    orphaned = 0
    fk_cols_retrieved = 0
    for fk in db.foreign_keys:
        if fk.from_table.lower() in retrieved_tables and fk.to_table.lower() in gold_tables:
            fk_cols_retrieved += 1
            if fk.to_table.lower() not in retrieved_tables:
                orphaned += 1

    orphaned_rate = orphaned / fk_cols_retrieved if fk_cols_retrieved > 0 else 0.0

    return {
        "fk_pair_recall":   fk_pair_recall,
        "n_needed_pairs":   len(needed_pairs),
        "orphaned_fk_rate": orphaned_rate,
        "n_orphaned":       orphaned,
    }


def compute_column_role_recall(
    gold_columns: set[str],
    retrieved_columns: set[str],
    db,
) -> dict:
    """
    Break column recall into PK / FK / regular columns.
    gold_columns is a set of 'table.col' strings.
    """
    # Build role lookups from DB schema
    pk_cols: set[str] = set()
    fk_cols: set[str] = set()
    for table in db.tables:
        for col in table.columns:
            key = f"{table.name.lower()}.{col.name.lower()}"
            if col.primary_key:
                pk_cols.add(key)
    for fk in db.foreign_keys:
        fk_cols.add(f"{fk.from_table.lower()}.{fk.from_column.lower()}")
        fk_cols.add(f"{fk.to_table.lower()}.{fk.to_column.lower()}")

    def role_recall(role_set: set[str]) -> float:
        gold_role = gold_columns & role_set
        if not gold_role:
            return None  # not applicable
        found = gold_role & retrieved_columns
        return len(found) / len(gold_role)

    pk_r  = role_recall(pk_cols)
    fk_r  = role_recall(fk_cols)
    reg_r = role_recall({c for c in gold_columns if c not in pk_cols and c not in fk_cols})

    return {"pk_col_recall": pk_r, "fk_col_recall": fk_r, "regular_col_recall": reg_r}


# =============================================================================
# Index / retriever builders (unchanged from original)
# =============================================================================

def build_embedding_encoder(idx_cfg: dict):
    provider   = idx_cfg.get("embedding_provider", "openai").lower()
    model_name = idx_cfg.get("embedding_model", "")
    if provider == "openai":
        from src.embeddings import OpenAIEmbeddingModel
        api_key  = idx_cfg.get("embedding_api_key", "") or os.environ.get("OPENAI_API_KEY", "")
        base_url = idx_cfg.get("embedding_base_url", "") or None
        logger.info("Using OpenAI embeddings: model=%s", model_name or "text-embedding-3-large")
        return OpenAIEmbeddingModel(
            model=model_name or "text-embedding-3-large",
            api_key=api_key or None,
            base_url=base_url,
        )
    if provider in ("huggingface", "hf", "local"):
        from src.embeddings import HuggingFaceEmbeddingModel
        logger.info("Using HuggingFace embeddings: model=%s", model_name)
        return HuggingFaceEmbeddingModel(model_name=model_name)
    return None


def compute_index_fingerprint(data_path: str, chunk_count: int) -> str:
    parts = [data_path, "bird", "openai", "text-embedding-3-large", COLLECTION_NAME, str(chunk_count)]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def check_index_cache(persist_dir: str, fingerprint: str) -> bool:
    fp_path = Path(persist_dir) / ".index_fingerprint"
    if not fp_path.exists():
        return False
    try:
        return json.loads(fp_path.read_text()).get("fingerprint") == fingerprint
    except Exception:
        return False


def save_index_fingerprint(persist_dir: str, fingerprint: str, chunk_count: int):
    from datetime import datetime
    fp_path = Path(persist_dir) / ".index_fingerprint"
    fp_path.parent.mkdir(parents=True, exist_ok=True)
    fp_path.write_text(json.dumps({
        "fingerprint": fingerprint,
        "chunk_count": chunk_count,
        "created_at": datetime.now().isoformat(),
    }))


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Retrieval analysis on BIRD dev")
    p.add_argument("--retriever", choices=["hybrid", "graph", "hybrid_merge"],
                   default="hybrid")
    p.add_argument("--graph_path", default="data/schema_graphs/bird_dev.json")
    p.add_argument("--n_examples", type=int, default=N_EXAMPLES)
    p.add_argument("--top_m",          type=int,   default=5)
    p.add_argument("--alpha",          type=float, default=0.7)
    p.add_argument("--max_nodes",      type=int,   default=20)
    p.add_argument("--score_threshold",type=float, default=0.01)
    p.add_argument("--no_linker",      action="store_true")
    p.add_argument("--rrf_k",          type=int,   default=60)
    p.add_argument("--chroma_dir",     default=CHROMA_DIR_DEFAULT,
                   help="ChromaDB persist directory (use separate dirs to run in parallel)")
    p.add_argument("--score_gap_ratio", type=float, default=3.0,
                   help="Elbow-cut threshold for pruning low-confidence tail nodes (v3)")
    p.add_argument("--use_fk_bridge", action="store_true", default=True,
                   help="Inject FK-connected bridge tables missing from PPR results (v3)")
    p.add_argument("--no_fk_bridge", dest="use_fk_bridge", action="store_false",
                   help="Disable FK bridge injection")
    p.add_argument("--value_scan", action="store_true", default=False,
                   help="Enable ValueScanner for seed boosting (v3)")
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args       = parse_args()
    use_graph  = args.retriever in ("graph", "hybrid_merge")
    use_merge  = args.retriever == "hybrid_merge"
    start_time = time.time()

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    idx_cfg = cfg["indexing"]
    ret_cfg = cfg["retrieval"]

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Loading BIRD dev data from: %s", DATA_PATH)
    parser = get_parser("bird")
    databases, examples = parser.load(DATA_PATH)
    logger.info("Loaded %d databases, %d examples", len(databases), len(examples))

    dev_examples = examples if args.n_examples == 0 else examples[:args.n_examples]
    logger.info("Analyzing %d examples", len(dev_examples))

    db_map         = {db.db_id: db for db in databases}
    example_db_ids = {ex.db_id for ex in dev_examples}
    relevant_dbs   = [db for db in databases if db.db_id in example_db_ids]

    chunker    = SchemaChunker(max_sample_values=cfg["chunking"].get("max_sample_values", 5))
    all_chunks = chunker.chunk_many(relevant_dbs)
    logger.info("Generated %d schema chunks for %d DBs", len(all_chunks), len(relevant_dbs))

    augmentor     = QueryAugmentor(strategy=cfg["augmentation"].get("strategy", "keyword"))
    schema_filter = SchemaFilter(top_k=ret_cfg["final_top_k"])

    def _build_hybrid():
        encoder    = build_embedding_encoder(idx_cfg)
        indexer    = SchemaIndexer(
            embedding_model=idx_cfg["embedding_model"],
            persist_dir=args.chroma_dir,
            collection_name=COLLECTION_NAME,
            batch_size=idx_cfg["batch_size"],
            encoder=encoder,
        )
        fingerprint = compute_index_fingerprint(DATA_PATH, len(all_chunks))
        if check_index_cache(args.chroma_dir, fingerprint):
            logger.info("Index cache hit — skipping re-indexing.")
        else:
            logger.info("Indexing %d chunks into ChromaDB...", len(all_chunks))
            indexer.index(all_chunks, reset=True)
            save_index_fingerprint(args.chroma_dir, fingerprint, len(all_chunks))
        return HybridRetriever(
            indexer=indexer,
            chunks=all_chunks,
            bm25_top_k=ret_cfg["bm25_top_k"],
            semantic_top_k=ret_cfg["semantic_top_k"],
            rrf_k=ret_cfg["rrf_k"],
        )

    if use_graph:
        from src.schema_graph import SchemaGraph
        from src.retrieval.graph_retriever import GraphRetriever
        from src.embeddings import OpenAIEmbeddingModel
        logger.info("Loading schema graph from: %s", args.graph_path)
        graph    = SchemaGraph.load(args.graph_path)
        embedder = _OpenAIEmbedderWrapper(OpenAIEmbeddingModel(model="text-embedding-3-large"))

        if use_merge:
            logger.info("Building HybridRetriever for hybrid_merge...")
            retriever = GraphRetriever(
                graph, embedder,
                top_m=args.top_m, alpha=args.alpha,
                max_nodes=args.max_nodes, score_threshold=args.score_threshold,
                hybrid_retriever=_build_hybrid(), rrf_k=args.rrf_k,
                score_gap_ratio=args.score_gap_ratio,
                use_fk_bridge=args.use_fk_bridge,
            )
            linker         = None
            retriever_name = "GraphRetriever+HybridRetriever (RRF)"
        else:
            retriever = GraphRetriever(
                graph, embedder,
                top_m=args.top_m, alpha=args.alpha,
                max_nodes=args.max_nodes, score_threshold=args.score_threshold,
                score_gap_ratio=args.score_gap_ratio,
                use_fk_bridge=args.use_fk_bridge,
            )
            linker         = None
            retriever_name = "GraphRetriever (PPR)"
    else:
        retriever = _build_hybrid()
        linker    = None if args.no_linker else BidirectionalLinker(
            max_expansion_depth=ret_cfg["bidirectional"]["max_expansion_depth"],
        )
        retriever_name = "HybridRetriever (BM25+Semantic)"

    # ── ValueScanner (v3 seed boosting) ────────────────────────────────────────
    value_scanner = None
    if args.value_scan and use_graph:
        from src.pre_retrieval.value_scanner import ValueScanner
        value_scanner = ValueScanner(max_values_per_col=500, top_k=5, min_score=0.75)
        logger.info("ValueScanner enabled for seed boosting")

    # ── Per-example loop ───────────────────────────────────────────────────────
    results = []

    per_db: defaultdict = defaultdict(lambda: defaultdict(list))

    # For aggregate new metrics
    ndcg_all, mrr_all, hit3_all, hit5_all = [], [], [], []
    fk_pair_recall_all, orphaned_rate_all  = [], []
    context_tokens_all, token_recall_all   = [], []
    pk_recall_all, fk_recall_all, reg_recall_all = [], [], []

    for i, ex in enumerate(dev_examples):
        db = db_map.get(ex.db_id)
        if db is None:
            logger.warning("DB not found: %s", ex.db_id)
            continue

        question = ex.question
        if getattr(ex, "evidence", "").strip():
            question = f"{ex.question}\nHint: {ex.evidence.strip()}"

        aug_query = augmentor.augment(question, db)

        # ── Value scanning (v3) ──
        value_matches = None
        if value_scanner is not None:
            try:
                value_matches = value_scanner.scan(question, db)
            except Exception:
                pass

        if isinstance(aug_query, list):
            if use_graph:
                raw = retriever.retrieve_multi(aug_query, db_id=ex.db_id, value_matches=value_matches)
            else:
                raw = retriever.retrieve_multi(aug_query, db_id=ex.db_id)
        else:
            if use_graph:
                raw = retriever.retrieve(aug_query, db_id=ex.db_id, value_matches=value_matches)
            else:
                raw = retriever.retrieve(aug_query, db_id=ex.db_id)

        if linker is not None:
            db_chunks = [c for c in all_chunks if c.db_id == ex.db_id]
            expanded  = linker.expand(raw, db, db_chunks)
        else:
            expanded = raw

        schema_context = schema_filter.filter_and_format(expanded, db)
        top_k          = ret_cfg["final_top_k"]

        # --- Core metrics ---
        ranked_tables    = extract_retrieved_tables(expanded, top_k=top_k)
        retrieved_tables = set(ranked_tables)
        retrieved_cols   = extract_retrieved_columns(expanded, db, top_k=top_k)
        gold_tables, gold_cols = extract_gold_tables_and_columns(ex.query)

        if gold_tables:
            table_recall    = len(gold_tables & retrieved_tables) / len(gold_tables)
            missed_tables   = gold_tables - retrieved_tables
        else:
            table_recall, missed_tables = 1.0, set()

        if retrieved_tables:
            table_precision = len(retrieved_tables & gold_tables) / len(retrieved_tables)
            extra_tables    = retrieved_tables - gold_tables
        else:
            table_precision, extra_tables = 0.0, set()

        if gold_cols:
            col_recall    = len(gold_cols & retrieved_cols) / len(gold_cols)
            missed_cols   = gold_cols - retrieved_cols
        else:
            col_recall, missed_cols = 1.0, set()

        col_precision = (
            len(retrieved_cols & gold_cols) / len(retrieved_cols)
            if retrieved_cols else 0.0
        )

        # --- 1. Rank-quality ---
        ndcg  = compute_ndcg(gold_tables, ranked_tables, k=top_k)
        mrr   = compute_mrr(gold_tables, ranked_tables)
        hit3  = compute_hit_at_k(gold_tables, ranked_tables, k=3)
        hit5  = compute_hit_at_k(gold_tables, ranked_tables, k=5)
        ndcg_all.append(ndcg); mrr_all.append(mrr)
        hit3_all.append(hit3); hit5_all.append(hit5)

        # --- 4. FK coverage ---
        fk_cov = compute_fk_coverage(gold_tables, retrieved_tables, db)
        fk_pair_recall_all.append(fk_cov["fk_pair_recall"])
        orphaned_rate_all.append(fk_cov["orphaned_fk_rate"])

        # --- 5. Context budget ---
        ctx_tokens  = count_schema_tokens(schema_context)
        gold_tokens = count_schema_tokens(
            schema_filter.filter_and_format(
                # Fake result: build a minimal retrieved list using gold tables only
                [{"chunk": type("C", (), {
                    "table_name": t, "chunk_type": "table",
                    "db_id": ex.db_id, "column_name": "", "content": ""
                })(), "score": 1.0} for t in gold_tables],
                db,
            )
        ) if gold_tables else 0
        tok_recall = gold_tokens / ctx_tokens if ctx_tokens > 0 else 1.0
        redundant_ratio = len(extra_tables) / len(retrieved_tables) if retrieved_tables else 0.0
        context_tokens_all.append(ctx_tokens)
        token_recall_all.append(tok_recall)

        # --- 6. Column-role recall ---
        col_roles = compute_column_role_recall(gold_cols, retrieved_cols, db)
        if col_roles["pk_col_recall"]      is not None: pk_recall_all.append(col_roles["pk_col_recall"])
        if col_roles["fk_col_recall"]      is not None: fk_recall_all.append(col_roles["fk_col_recall"])
        if col_roles["regular_col_recall"] is not None: reg_recall_all.append(col_roles["regular_col_recall"])

        # --- Store result record ---
        difficulty = getattr(ex, "difficulty", "unknown")
        result = {
            "idx":             i,
            "db_id":           ex.db_id,
            "question":        ex.question,
            "evidence":        getattr(ex, "evidence", ""),
            "difficulty":      difficulty,
            "gold_sql":        ex.query,
            "gold_tables":     sorted(gold_tables),
            "retrieved_tables":sorted(retrieved_tables),
            "ranked_tables":   ranked_tables,
            "missed_tables":   sorted(missed_tables),
            "extra_tables":    sorted(extra_tables),
            "gold_columns":    sorted(gold_cols),
            "missed_columns":  sorted(missed_cols),
            "table_recall":    table_recall,
            "table_precision": table_precision,
            "table_f1":        safe_f1(table_recall, table_precision),
            "col_recall":      col_recall,
            "col_precision":   col_precision,
            "col_f1":          safe_f1(col_recall, col_precision),
            "n_gold_tables":   len(gold_tables),
            "n_retrieved_tables": len(retrieved_tables),
            "n_db_tables":     len(db.tables),
            # Rank-quality
            "ndcg":   ndcg,
            "mrr":    mrr,
            "hit@3":  hit3,
            "hit@5":  hit5,
            # FK coverage
            "fk_pair_recall":   fk_cov["fk_pair_recall"],
            "orphaned_fk_rate": fk_cov["orphaned_fk_rate"],
            # Context budget
            "context_tokens":   ctx_tokens,
            "token_recall":     tok_recall,
            "redundant_ratio":  redundant_ratio,
            # Column role
            "pk_col_recall":      col_roles["pk_col_recall"],
            "fk_col_recall":      col_roles["fk_col_recall"],
            "regular_col_recall": col_roles["regular_col_recall"],
            # For schema display
            "schema_context": schema_context,
        }
        results.append(result)

        # Per-DB accumulation
        pdb = per_db[ex.db_id]
        for key in ["table_recall","table_precision","table_f1",
                    "col_recall","col_precision",
                    "ndcg","mrr","hit@3","hit@5",
                    "fk_pair_recall","context_tokens","token_recall","redundant_ratio"]:
            pdb[key].append(result[key])
        pdb["n"].append(1)

        if (i + 1) % 20 == 0:
            avg_r = sum(r["table_recall"] for r in results) / len(results)
            avg_p = sum(r["table_precision"] for r in results) / len(results)
            logger.info("Progress %d/%d — avg table recall=%.3f precision=%.3f",
                        i+1, len(dev_examples), avg_r, avg_p)

    elapsed = time.time() - start_time

    # =========================================================================
    # Helper: mean of list
    # =========================================================================
    def avg(lst): return sum(lst) / len(lst) if lst else 0.0
    def std(lst):
        if len(lst) < 2: return 0.0
        m = avg(lst)
        return math.sqrt(sum((x - m)**2 for x in lst) / len(lst))

    # =========================================================================
    # PRINT REPORT
    # =========================================================================
    W = 80
    sep = "─" * W

    def section(title):
        print(f"\n{sep}")
        print(f"  {title}")
        print(sep)

    print("\n" + "=" * W)
    print(f"  RETRIEVAL ANALYSIS — BIRD Dev ({len(results)} examples)")
    print(f"  Retriever : {retriever_name}")
    print(f"  Elapsed   : {elapsed:.1f}s")
    print("=" * W)

    # ── 0. Core Metrics ───────────────────────────────────────────────────────
    section("CORE METRICS (Set-Membership)")
    tr   = avg([r["table_recall"]    for r in results])
    tp   = avg([r["table_precision"] for r in results])
    tf1  = safe_f1(tr, tp)
    cr   = avg([r["col_recall"]      for r in results])
    cp   = avg([r["col_precision"]   for r in results])
    cf1  = safe_f1(cr, cp)
    perf = sum(1 for r in results if r["table_recall"] == 1.0)
    n_missed = sum(1 for r in results if r["missed_tables"])
    n_noisy  = sum(1 for r in results if len(r["extra_tables"]) > 3)
    zero_r   = sum(1 for r in results if r["table_recall"] == 0.0)
    exact_m  = sum(1 for r in results if not r["missed_tables"] and not r["extra_tables"])

    print(f"  Table  : Recall={tr:.4f}  Precision={tp:.4f}  F1={tf1:.4f}")
    print(f"  Column : Recall={cr:.4f}  Precision={cp:.4f}  F1={cf1:.4f}")
    print(f"  Perfect recall  : {perf}/{len(results)} ({100*perf/len(results):.1f}%)")
    print(f"  Exact match     : {exact_m}/{len(results)} ({100*exact_m/len(results):.1f}%)  (recall=1 AND precision=1)")
    print(f"  Zero recall     : {zero_r}/{len(results)} ({100*zero_r/len(results):.1f}%)  (no gold table found)")
    print(f"  Missed examples : {n_missed}  |  Noisy (>3 extra): {n_noisy}")

    # ── 1. Rank-Quality ───────────────────────────────────────────────────────
    section("RANK-QUALITY METRICS  (NDCG / MRR / Hit@K)")
    ndcg_v = avg(ndcg_all)
    mrr_v  = avg(mrr_all)
    hit3_v = avg(hit3_all)
    hit5_v = avg(hit5_all)
    print(f"  NDCG@{ret_cfg['final_top_k']:<2}          : {ndcg_v:.4f}")
    print(f"  MRR              : {mrr_v:.4f}  (1/rank of first gold table in ranked list)")
    print(f"  Hit@3            : {hit3_v:.4f}  (≥1 gold table in top-3)")
    print(f"  Hit@5            : {hit5_v:.4f}  (≥1 gold table in top-5)")

    # ── 2. Join-Complexity Stratification ─────────────────────────────────────
    section("JOIN-COMPLEXITY STRATIFICATION  (#gold tables needed)")
    complexity_buckets: dict[str, list] = {
        "1-table": [], "2-table": [], "3-table": [], "4+-table": []
    }
    for r in results:
        n = r["n_gold_tables"]
        bucket = "1-table" if n == 1 else "2-table" if n == 2 else "3-table" if n == 3 else "4+-table"
        complexity_buckets[bucket].append(r)

    print(f"  {'Complexity':<12} {'#Ex':>5}  {'Recall':>7} {'Prec':>7} {'F1':>7}  {'NDCG':>7} {'MRR':>7} {'Hit@3':>7}")
    print(f"  {'─'*12} {'─'*5}  {'─'*7} {'─'*7} {'─'*7}  {'─'*7} {'─'*7} {'─'*7}")
    for bucket, recs in complexity_buckets.items():
        if not recs: continue
        r_  = avg([r["table_recall"]    for r in recs])
        p_  = avg([r["table_precision"] for r in recs])
        f1_ = safe_f1(r_, p_)
        nd_ = avg([r["ndcg"] for r in recs])
        mr_ = avg([r["mrr"]  for r in recs])
        h3_ = avg([r["hit@3"] for r in recs])
        print(f"  {bucket:<12} {len(recs):>5}  {r_:>7.4f} {p_:>7.4f} {f1_:>7.4f}  {nd_:>7.4f} {mr_:>7.4f} {h3_:>7.4f}")

    # ── 3. Difficulty Stratification ──────────────────────────────────────────
    section("DIFFICULTY STRATIFICATION  (BIRD labels)")
    diff_buckets: dict[str, list] = defaultdict(list)
    for r in results:
        diff_buckets[r["difficulty"]].append(r)

    print(f"  {'Difficulty':<14} {'#Ex':>5}  {'Recall':>7} {'Prec':>7} {'F1':>7}  {'ColR':>7} {'ColP':>7} {'NDCG':>7}")
    print(f"  {'─'*14} {'─'*5}  {'─'*7} {'─'*7} {'─'*7}  {'─'*7} {'─'*7} {'─'*7}")
    for diff_label in ["simple", "moderate", "challenging"]:
        recs = diff_buckets.get(diff_label, [])
        if not recs: continue
        r_  = avg([r["table_recall"]    for r in recs])
        p_  = avg([r["table_precision"] for r in recs])
        f1_ = safe_f1(r_, p_)
        cr_ = avg([r["col_recall"]      for r in recs])
        cp_ = avg([r["col_precision"]   for r in recs])
        nd_ = avg([r["ndcg"] for r in recs])
        print(f"  {diff_label:<14} {len(recs):>5}  {r_:>7.4f} {p_:>7.4f} {f1_:>7.4f}  {cr_:>7.4f} {cp_:>7.4f} {nd_:>7.4f}")

    # ── 4. FK Coverage ─────────────────────────────────────────────────────────
    section("FK COVERAGE")
    fk_pr = avg(fk_pair_recall_all)
    orf   = avg(orphaned_rate_all)
    n_fk_needed = sum(r["n_gold_tables"] > 1 for r in results)
    print(f"  FK pair recall        : {fk_pr:.4f}  (both sides of needed FK retrieved)")
    print(f"  Orphaned FK rate      : {orf:.4f}  (FK col retrieved, target table missing)")
    print(f"  Multi-table examples  : {n_fk_needed}/{len(results)}  (queries needing ≥1 FK join)")

    # ── 5. Context Budget ──────────────────────────────────────────────────────
    section("CONTEXT BUDGET METRICS")
    avg_tok  = avg(context_tokens_all)
    avg_tokr = avg(token_recall_all)
    avg_rr   = avg([r["redundant_ratio"] for r in results])
    p25_tok  = sorted(context_tokens_all)[len(context_tokens_all)//4]
    p75_tok  = sorted(context_tokens_all)[3*len(context_tokens_all)//4]
    p95_tok  = sorted(context_tokens_all)[int(0.95*len(context_tokens_all))]
    over_2k  = sum(1 for t in context_tokens_all if t > 2000)
    print(f"  Avg schema tokens     : {avg_tok:.0f}  (p25={p25_tok} p75={p75_tok} p95={p95_tok})")
    print(f"  Token recall          : {avg_tokr:.4f}  (gold tokens / total context tokens)")
    print(f"  Redundant table ratio : {avg_rr:.4f}  (extra tables / total retrieved)")
    print(f"  Contexts >2000 tokens : {over_2k}/{len(results)} ({100*over_2k/len(results):.1f}%)")

    # ── 6. Column-Role Recall ──────────────────────────────────────────────────
    section("COLUMN-ROLE RECALL")
    print(f"  PK columns  : {avg(pk_recall_all):.4f}  ({len(pk_recall_all)} applicable examples)")
    print(f"  FK columns  : {avg(fk_recall_all):.4f}  ({len(fk_recall_all)} applicable examples)")
    print(f"  Regular cols: {avg(reg_recall_all):.4f}  ({len(reg_recall_all)} applicable examples)")

    # ── 7. Consistency ─────────────────────────────────────────────────────────
    section("CONSISTENCY / VARIANCE")
    print(f"  Recall std-dev (all examples): {std([r['table_recall'] for r in results]):.4f}")
    print(f"\n  {'Database':<35} {'n':>4}  {'Recall':>7} {'StdDev':>8} {'Zero%':>7} {'Exact%':>8}")
    print(f"  {'─'*35} {'─'*4}  {'─'*7} {'─'*8} {'─'*7} {'─'*8}")
    for db_id, pdb in sorted(per_db.items()):
        recalls = pdb["table_recall"]
        n       = len(recalls)
        db_r    = avg(recalls)
        db_std  = std(recalls)
        db_zero = sum(1 for x in recalls if x == 0.0)
        recs_db = [r for r in results if r["db_id"] == db_id]
        db_exact= sum(1 for r in recs_db if not r["missed_tables"] and not r["extra_tables"])
        print(f"  {db_id:<35} {n:>4}  {db_r:>7.3f} {db_std:>8.3f} {100*db_zero/n:>6.1f}% {100*db_exact/n:>7.1f}%")

    # ── Per-DB full table ──────────────────────────────────────────────────────
    section("PER-DATABASE — ALL METRICS")
    print(f"  {'Database':<30} {'n':>4}  "
          f"{'TblR':>6} {'TblP':>6} {'TblF1':>6}  "
          f"{'ColR':>6} {'ColP':>6}  "
          f"{'NDCG':>6} {'MRR':>6} {'H@3':>5}  "
          f"{'FKPair':>7} {'CtxTok':>7}")
    print(f"  {'─'*30} {'─'*4}  "
          f"{'─'*6} {'─'*6} {'─'*6}  "
          f"{'─'*6} {'─'*6}  "
          f"{'─'*6} {'─'*6} {'─'*5}  "
          f"{'─'*7} {'─'*7}")
    for db_id, pdb in sorted(per_db.items(), key=lambda x: -sum(x[1]["n"])):
        n   = sum(pdb["n"])
        r_  = avg(pdb["table_recall"])
        p_  = avg(pdb["table_precision"])
        f1_ = safe_f1(r_, p_)
        cr_ = avg(pdb["col_recall"])
        cp_ = avg(pdb["col_precision"])
        nd_ = avg(pdb["ndcg"])
        mr_ = avg(pdb["mrr"])
        h3_ = avg(pdb["hit@3"])
        fk_ = avg(pdb["fk_pair_recall"])
        ct_ = avg(pdb["context_tokens"])
        print(f"  {db_id:<30} {n:>4}  "
              f"{r_:>6.3f} {p_:>6.3f} {f1_:>6.3f}  "
              f"{cr_:>6.3f} {cp_:>6.3f}  "
              f"{nd_:>6.3f} {mr_:>6.3f} {h3_:>5.3f}  "
              f"{fk_:>7.3f} {ct_:>7.0f}")

    # ── Recall distribution ────────────────────────────────────────────────────
    section("RECALL DISTRIBUTION")
    bins = [("=1.00", lambda x: x == 1.0),
            (">=0.80", lambda x: x >= 0.80),
            (">=0.50", lambda x: x >= 0.50),
            ("<0.50",  lambda x: x <  0.50)]
    counts = {label: sum(1 for r in results if fn(r["table_recall"])) for label, fn in bins}
    # Make exclusive
    exclusive = {}
    exclusive["=1.00"]  = counts["=1.00"]
    exclusive[">=0.80"] = counts[">=0.80"] - counts["=1.00"]
    exclusive[">=0.50"] = counts[">=0.50"] - counts[">=0.80"]
    exclusive["<0.50"]  = counts["<0.50"]
    for label, cnt in exclusive.items():
        pct = 100 * cnt / len(results)
        bar = "█" * int(pct / 2)
        print(f"  {label:>7}: {cnt:>4} ({pct:5.1f}%)  {bar}")

    # ── Most missed / over-retrieved ──────────────────────────────────────────
    section("TOP MISSED & OVER-RETRIEVED TABLES")
    all_missed = Counter(
        f"{r['db_id']}.{t}" for r in results for t in r["missed_tables"]
    )
    all_extra  = Counter(
        f"{r['db_id']}.{t}" for r in results for t in r["extra_tables"]
    )
    print("  Most frequently MISSED (top 15):")
    for tbl, cnt in all_missed.most_common(15):
        print(f"    {tbl}: {cnt}×")
    print("\n  Most frequently OVER-RETRIEVED (top 15):")
    for tbl, cnt in all_extra.most_common(15):
        print(f"    {tbl}: {cnt}×")

    # ── Worst examples (for debugging) ────────────────────────────────────────
    section("NOTABLE EXAMPLES")
    worst_r = sorted([r for r in results if r["missed_tables"]], key=lambda r: r["table_recall"])
    if worst_r:
        r = worst_r[0]
        print(f"\n  *** WORST TABLE RECALL ***")
        print(f"  #{r['idx']} [{r['db_id']}] difficulty={r['difficulty']}")
        print(f"  Q: {r['question'][:120]}")
        print(f"  Gold: {r['gold_tables']}  Retrieved: {r['retrieved_tables']}")
        print(f"  Missed: {r['missed_tables']}  |  recall={r['table_recall']:.3f} ndcg={r['ndcg']:.3f} mrr={r['mrr']:.3f}")

    perfect_prec = [r for r in results if r["table_recall"] == 1.0 and not r["extra_tables"]]
    if perfect_prec:
        print(f"\n  *** PERFECT EXAMPLES (recall=1, prec=1): {len(perfect_prec)}/{len(results)} ***")

    high_ctx = sorted(results, key=lambda r: r["context_tokens"], reverse=True)[:3]
    print(f"\n  *** LARGEST CONTEXT WINDOWS ***")
    for r in high_ctx:
        print(f"  #{r['idx']} [{r['db_id']}] {r['context_tokens']} tokens  "
              f"gold={r['gold_tables']}  extra={r['extra_tables']}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    suffix = "_hybrid_merge" if use_merge else ("_graph" if use_graph else "")
    n_tag  = "_full" if args.n_examples == 0 else (
             f"_{args.n_examples}" if args.n_examples != N_EXAMPLES else "")
    out_path = PROJECT_ROOT / "results" / f"bird_retrieval_analysis{suffix}{n_tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Aggregate new metrics for JSON
    complexity_agg = {}
    for bucket, recs in complexity_buckets.items():
        if recs:
            r_ = avg([r["table_recall"]    for r in recs])
            p_ = avg([r["table_precision"] for r in recs])
            complexity_agg[bucket] = {
                "n": len(recs),
                "table_recall": r_, "table_precision": p_, "table_f1": safe_f1(r_, p_),
                "ndcg": avg([r["ndcg"] for r in recs]),
                "mrr":  avg([r["mrr"]  for r in recs]),
                "hit@3": avg([r["hit@3"] for r in recs]),
            }
    difficulty_agg = {}
    for diff_label, recs in diff_buckets.items():
        if recs:
            r_ = avg([r["table_recall"]    for r in recs])
            p_ = avg([r["table_precision"] for r in recs])
            difficulty_agg[diff_label] = {
                "n": len(recs),
                "table_recall": r_, "table_precision": p_, "table_f1": safe_f1(r_, p_),
                "col_recall": avg([r["col_recall"] for r in recs]),
                "ndcg": avg([r["ndcg"] for r in recs]),
            }

    save_results = [{k: v for k, v in r.items() if k != "schema_context"} for r in results]

    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "retriever": retriever_name,
                "n_examples": len(results),
                **({"graph_path": args.graph_path, "top_m": args.top_m,
                    "alpha": args.alpha, "max_nodes": args.max_nodes,
                    "score_threshold": args.score_threshold,
                    "score_gap_ratio": args.score_gap_ratio,
                    "use_fk_bridge": args.use_fk_bridge,
                    "value_scan": args.value_scan} if use_graph else {}),
            },
            "overall": {
                "table_recall":     tr,  "table_precision":  tp,  "table_f1":  tf1,
                "column_recall":    cr,  "column_precision": cp,  "column_f1": cf1,
                "ndcg":  ndcg_v, "mrr": mrr_v, "hit@3": hit3_v, "hit@5": hit5_v,
                "perfect_table_recall_pct": 100 * perf / len(results),
                "exact_match_pct":          100 * exact_m / len(results),
                "zero_recall_pct":          100 * zero_r  / len(results),
                "n_missed_table_examples":  n_missed,
                "n_noisy_examples":         n_noisy,
                "fk_pair_recall":           fk_pr,
                "orphaned_fk_rate":         orf,
                "avg_context_tokens":       avg_tok,
                "token_recall":             avg_tokr,
                "redundant_ratio":          avg_rr,
                "pk_col_recall":  avg(pk_recall_all),
                "fk_col_recall":  avg(fk_recall_all),
                "reg_col_recall": avg(reg_recall_all),
            },
            "by_join_complexity": complexity_agg,
            "by_difficulty":      difficulty_agg,
            "per_db": {
                db_id: {
                    "n_examples":      sum(pdb["n"]),
                    "table_recall":    avg(pdb["table_recall"]),
                    "table_precision": avg(pdb["table_precision"]),
                    "table_f1":        safe_f1(avg(pdb["table_recall"]), avg(pdb["table_precision"])),
                    "column_recall":   avg(pdb["col_recall"]),
                    "column_precision":avg(pdb["col_precision"]),
                    "ndcg":            avg(pdb["ndcg"]),
                    "mrr":             avg(pdb["mrr"]),
                    "hit@3":           avg(pdb["hit@3"]),
                    "fk_pair_recall":  avg(pdb["fk_pair_recall"]),
                    "avg_context_tokens": avg(pdb["context_tokens"]),
                }
                for db_id, pdb in per_db.items()
            },
            "missed_table_counts": dict(all_missed.most_common()),
            "extra_table_counts":  dict(all_extra.most_common()),
            "results": save_results,
        }, f, indent=2)
    print(f"\n  Results saved to: {out_path}")
    print(f"\n{'=' * W}\n  ANALYSIS COMPLETE\n{'=' * W}\n")


if __name__ == "__main__":
    main()
