"""
Pipeline script – indexes schemas, then runs RAG + evaluation.

Usage:
    python scripts/run_pipeline.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.schema_chunker import SchemaChunker
from src.data.schema_indexer import SchemaIndexer
from src.data.spider_v1_adapter import SpiderV1Adapter
from src.retrieval.bidirectional_linker import BidirectionalLinker
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.npmi_scorer import NPMIScorer
from src.retrieval.query_augmentor import QueryAugmentor
from src.retrieval.schema_filter import SchemaFilter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(config_path: str):
    cfg = load_config(config_path)

    # ---- Phase 2: Data Pipeline ----
    logger.info("=== Phase 2: Loading Dataset ===")
    adapter = SpiderV1Adapter()
    databases, examples = adapter.load(cfg["data"]["dataset_path"])
    logger.info("Loaded %d databases, %d examples", len(databases), len(examples))

    # Chunk schemas
    chunker = SchemaChunker(max_sample_values=cfg["chunking"].get("max_sample_values", 5))
    all_chunks = chunker.chunk_many(databases)
    logger.info("Generated %d schema chunks", len(all_chunks))

    # Index into ChromaDB
    idx_cfg = cfg["indexing"]
    indexer = SchemaIndexer(
        embedding_model=idx_cfg["embedding_model"],
        persist_dir=idx_cfg["chroma_persist_dir"],
        collection_name=idx_cfg["collection_name"],
        batch_size=idx_cfg["batch_size"],
    )
    indexer.index(all_chunks, reset=True)

    # ---- Phase 3: RAG Schema Linking ----
    logger.info("=== Phase 3: RAG Schema Linking ===")
    ret_cfg = cfg["retrieval"]

    augmentor = QueryAugmentor(strategy=cfg["augmentation"].get("strategy", "keyword"))

    # Optional: Load NPMI scorer
    npmi_scorer = None
    npmi_cfg = cfg.get("npmi", {})
    if npmi_cfg.get("enable") and npmi_cfg.get("matrix_path"):
        logger.info("Loading NPMI matrix from: %s", npmi_cfg["matrix_path"])
        npmi_scorer = NPMIScorer.load(npmi_cfg["matrix_path"])

    retriever = HybridRetriever(
        indexer=indexer,
        chunks=all_chunks,
        bm25_top_k=ret_cfg["bm25_top_k"],
        semantic_top_k=ret_cfg["semantic_top_k"],
        rrf_k=ret_cfg["rrf_k"],
        npmi_scorer=npmi_scorer,
        npmi_top_k=npmi_cfg.get("top_k", 30),
    )
    linker = BidirectionalLinker(
        max_expansion_depth=ret_cfg["bidirectional"]["max_expansion_depth"],
    )
    schema_filter = SchemaFilter(top_k=ret_cfg["final_top_k"])

    db_map = {db.db_id: db for db in databases}

    # Pre-compute retrieval for all dev examples (or a subset)
    dev_examples = [ex for ex in examples if True]  # filter dev set as needed
    retrieved_schemas: dict[str, list[dict]] = {}

    logger.info("Running retrieval on %d examples...", len(dev_examples))
    for ex in dev_examples:
        db = db_map.get(ex.db_id)
        if db is None:
            continue
        aug_query = augmentor.augment(ex.question, db)
        results = retriever.retrieve(aug_query, db_id=ex.db_id)
        db_chunks = [c for c in all_chunks if c.db_id == ex.db_id]
        expanded = linker.expand(results, db, db_chunks)

        key = f"{ex.db_id}__{hash(ex.question) % 10**8}"
        retrieved_schemas[key] = expanded

    logger.info("Retrieval complete. Stored %d schema contexts.", len(retrieved_schemas))
    logger.info("Pipeline ready for training (run train scripts) or evaluation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text-to-SQL Pipeline")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
