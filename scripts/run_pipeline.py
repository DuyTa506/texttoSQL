"""
Pipeline script – indexes schemas, then runs RAG + evaluation.

Extended with 4-phase architecture:
  Phase 1: Pre-retrieval (ValueScanner, QuestionDecomposer) — feature-flagged
  Phase 2: Retrieval (HybridRetriever + BidirectionalLinker) — unchanged core
  Phase 3: Generation (SQLInference with mode + n_candidates) — feature-flagged
  Phase 4: Post-generation (RetryLoop, CandidateSelector) — feature-flagged

All new phases are OFF by default (configs/default.yaml). Enable per phase:
  --override post_generation.retry_loop.enabled=true
  --override pre_retrieval.value_scan.enabled=true
  --override post_generation.candidates.n=3

Usage:
    python scripts/run_pipeline.py --config configs/default.yaml
    python scripts/run_pipeline.py --config configs/default.yaml \\
        --override post_generation.retry_loop.enabled=true
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
from src.evaluation.metrics import compute_metrics, execution_accuracy, exact_match, schema_recall, schema_precision
from src.retrieval.bidirectional_linker import BidirectionalLinker
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.npmi_scorer import NPMIScorer
from src.retrieval.query_augmentor import QueryAugmentor
from src.retrieval.schema_filter import SchemaFilter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Config helpers
# =============================================================================

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply CLI overrides of the form 'section.key=value' to config dict."""
    for override in overrides:
        if "=" not in override:
            logger.warning("Ignoring malformed override (no '='): %s", override)
            continue
        key_path, raw_value = override.split("=", 1)
        parts = key_path.strip().split(".")

        # Convert value to Python type
        value: object
        if raw_value.lower() == "true":
            value = True
        elif raw_value.lower() == "false":
            value = False
        else:
            try:
                value = int(raw_value)
            except ValueError:
                try:
                    value = float(raw_value)
                except ValueError:
                    value = raw_value

        # Navigate to the nested key and set
        node = cfg
        for part in parts[:-1]:
            if part not in node:
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value
        logger.info("Config override: %s = %r", key_path, value)

    return cfg


# =============================================================================
# Phase-component builders
# =============================================================================

def _build_retriever(
    cfg: dict,
    indexer,
    all_chunks: list,
    npmi_scorer=None,
):
    """
    Build the retriever for Phase 2.

    When schema_graph.enabled is true:
      - Loads the pre-built SchemaGraph from graph_path.
      - Instantiates a GraphRetriever (PPR-based).
      - If schema_graph.hybrid is also true, passes the HybridRetriever as a
        secondary signal so both PPR and BM25+semantic scores are fused via RRF.

    When schema_graph.enabled is false (default):
      - Returns a plain HybridRetriever (original baseline behaviour).

    The returned object always exposes:
      retrieve(query, *, db_id=None)        → list[dict]
      retrieve_multi(queries, *, db_id=None) → list[dict]
    """
    ret_cfg = cfg["retrieval"]
    npmi_cfg = cfg.get("npmi", {})
    sg_cfg = cfg.get("schema_graph", {})

    # Always build HybridRetriever — needed either as primary or hybrid fallback
    hybrid_retriever = HybridRetriever(
        indexer=indexer,
        chunks=all_chunks,
        bm25_top_k=ret_cfg["bm25_top_k"],
        semantic_top_k=ret_cfg["semantic_top_k"],
        rrf_k=ret_cfg["rrf_k"],
        npmi_scorer=npmi_scorer,
        npmi_top_k=npmi_cfg.get("top_k", 30),
    )

    if not sg_cfg.get("enabled", False):
        return hybrid_retriever

    # ── GraphRetriever path ────────────────────────────────────────────────────
    graph_path = sg_cfg.get("graph_path", "")
    if not graph_path or not Path(graph_path).exists():
        logger.warning(
            "schema_graph.enabled=true but graph_path '%s' not found — "
            "falling back to HybridRetriever. "
            "Build the graph with: python scripts/build_schema_graph.py --enrich",
            graph_path,
        )
        return hybrid_retriever

    try:
        from sentence_transformers import SentenceTransformer
        from src.schema_graph import SchemaGraph
        from src.schema_graph.graph_retriever import GraphRetriever
    except ImportError as exc:
        logger.warning(
            "GraphRetriever dependencies not available (%s) — "
            "falling back to HybridRetriever.",
            exc,
        )
        return hybrid_retriever

    logger.info("Loading SchemaGraph from %s ...", graph_path)
    graph = SchemaGraph.load(graph_path)

    # Attach ChromaDB node-embedding collection for fast ANN entry-point search.
    # Uses the same persist_dir as the schema_chunks collection (one DB instance).
    node_collection_name = sg_cfg.get("node_collection", "schema_graph_nodes")
    chroma_dir = cfg["indexing"].get("chroma_persist_dir", "./data/chroma_db")
    try:
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        chroma_client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        node_collection = chroma_client.get_collection(node_collection_name)
        graph.attach_chroma(node_collection)
        logger.info(
            "Attached ChromaDB collection '%s' to SchemaGraph (fast ANN mode).",
            node_collection_name,
        )
    except Exception as exc:
        logger.warning(
            "Could not attach ChromaDB collection '%s': %s — "
            "falling back to in-node linear scan. "
            "Run build_schema_graph.py --enrich to populate ChromaDB.",
            node_collection_name, exc,
        )

    embedding_model = cfg["indexing"]["embedding_model"]
    logger.info("Loading SentenceTransformer '%s' for GraphRetriever ...", embedding_model)
    embedder = SentenceTransformer(embedding_model)

    graph_retriever = GraphRetriever(
        graph,
        embedder,
        top_m=sg_cfg.get("top_m", 5),
        alpha=sg_cfg.get("alpha", 0.7),
        score_threshold=sg_cfg.get("score_threshold", 0.05),
        max_nodes=sg_cfg.get("max_nodes", 20),
        synonym_boost=sg_cfg.get("synonym_boost", 0.3),
        # Hybrid mode: fuse PPR + BM25/semantic via RRF
        hybrid_retriever=hybrid_retriever if sg_cfg.get("hybrid", False) else None,
        rrf_k=ret_cfg["rrf_k"],
    )
    mode = "graph+hybrid RRF" if sg_cfg.get("hybrid", False) else "graph PPR only"
    logger.info("GraphRetriever ready (%s, graph=%s)", mode, graph)
    return graph_retriever


def _build_value_scanner(pre_cfg: dict):
    """Build a ValueScanner if enabled, else return None."""
    vs_cfg = pre_cfg.get("value_scan", {})
    if not vs_cfg.get("enabled", False):
        return None
    from src.retrieval.value_scanner import ValueScanner
    logger.info("ValueScanner enabled (max_values=%d, top_k=%d)",
                vs_cfg.get("max_values", 500), vs_cfg.get("top_k", 5))
    return ValueScanner(
        max_values_per_col=vs_cfg.get("max_values", 500),
        top_k=vs_cfg.get("top_k", 5),
        min_score=vs_cfg.get("min_score", 0.75),
    )


def _build_decomposer(pre_cfg: dict):
    """Build a QuestionDecomposer if enabled, else return None."""
    dec_cfg = pre_cfg.get("decomposition", {})
    if not dec_cfg.get("enabled", False):
        return None
    from src.retrieval.question_decomposer import QuestionDecomposer
    logger.info("QuestionDecomposer enabled (threshold=%.2f)",
                dec_cfg.get("complexity_threshold", 0.6))
    return QuestionDecomposer(
        complexity_threshold=dec_cfg.get("complexity_threshold", 0.6),
        min_token_count=dec_cfg.get("min_token_count", 12),
    )


def _build_inference(gen_cfg: dict):
    """
    Build a SQLInference from the generation config.

    Returns None if no provider is configured (model_path empty and
    provider is 'local'), which disables Phase 3 generation in the pipeline
    (schema-linking only mode — useful for retrieval-only ablations).
    """
    from src.generation.inference import SQLInference

    provider = gen_cfg.get("provider", "local").lower()
    model_path = gen_cfg.get("model_path", "")

    # Skip if local provider with no model path (retrieval-only mode)
    if provider == "local" and not model_path:
        logger.warning(
            "generation.model_path is not set and provider='local'. "
            "Skipping Phase 3 generation (retrieval-only mode). "
            "Set generation.model_path or use provider='openai'."
        )
        return None

    try:
        inference = SQLInference.from_config(gen_cfg)
        logger.info(
            "SQLInference ready (provider=%s, model=%s).",
            provider,
            model_path or gen_cfg.get("model_path", "gpt-4o-mini"),
        )
        return inference
    except Exception as exc:
        logger.error("Failed to build SQLInference: %s", exc)
        raise


def _build_retry_loop(post_cfg: dict, executor, inference):
    """Build RetryLoop if enabled, else return None."""
    rl_cfg = post_cfg.get("retry_loop", {})
    if not rl_cfg.get("enabled", False):
        return None
    if inference is None:
        logger.warning("RetryLoop requested but no inference model configured — skipping.")
        return None
    from src.post.retry_loop import RetryConfig, RetryLoop
    logger.info("RetryLoop enabled (max_retries=%d)", rl_cfg.get("max_retries", 3))
    return RetryLoop(
        executor=executor,
        inference=inference,
        config=RetryConfig(
            max_retries=rl_cfg.get("max_retries", 3),
            enabled=True,
        ),
    )


def _build_candidate_selector(post_cfg: dict, executor):
    """Build CandidateSelector if n_candidates > 1, else return None."""
    cand_cfg = post_cfg.get("candidates", {})
    n = cand_cfg.get("n", 1)
    if n <= 1:
        return None
    if executor is None:
        return None
    from src.post.candidate_selector import CandidateSelector
    logger.info("CandidateSelector enabled (n=%d, strategy=%s)",
                n, cand_cfg.get("strategy", "execution_consistency"))
    return CandidateSelector(
        executor=executor,
        strategy=cand_cfg.get("strategy", "execution_consistency"),
    )


# =============================================================================
# Main pipeline
# =============================================================================

def main(config_path: str, overrides: list[str] | None = None):
    cfg = load_config(config_path)
    if overrides:
        cfg = apply_overrides(cfg, overrides)

    # ---- Load feature-flag configs ----
    pre_cfg = cfg.get("pre_retrieval", {})
    gen_cfg = cfg.get("generation", {})
    post_cfg = cfg.get("post_generation", {})
    eval_cfg = cfg.get("evaluation", {})

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

    # ---- Phase 2: Retrieval (HybridRetriever or GraphRetriever) ----
    logger.info("=== Phase 2: Schema Linking ===")
    ret_cfg = cfg["retrieval"]
    aug_strategy = cfg["augmentation"].get("strategy", "keyword")
    augmentor = QueryAugmentor(strategy=aug_strategy)

    # Optional: Load NPMI scorer
    npmi_scorer = None
    npmi_cfg = cfg.get("npmi", {})
    if npmi_cfg.get("enable") and npmi_cfg.get("matrix_path"):
        logger.info("Loading NPMI matrix from: %s", npmi_cfg["matrix_path"])
        npmi_scorer = NPMIScorer.load(npmi_cfg["matrix_path"])

    retriever = _build_retriever(cfg, indexer, all_chunks, npmi_scorer=npmi_scorer)
    linker = BidirectionalLinker(
        max_expansion_depth=ret_cfg["bidirectional"]["max_expansion_depth"],
    )
    schema_filter = SchemaFilter(top_k=ret_cfg["final_top_k"])

    db_map = {db.db_id: db for db in databases}

    # ---- Phase 1: Pre-Retrieval components (feature-flagged) ----
    value_scanner = _build_value_scanner(pre_cfg)
    decomposer = _build_decomposer(pre_cfg)

    # ---- Phase 3: Generation components (feature-flagged) ----
    inference = _build_inference(gen_cfg)
    gen_mode = gen_cfg.get("mode", "standard")
    n_candidates = gen_cfg.get("n_candidates", 1)

    # ---- Phase 4: Post-generation components (feature-flagged) ----
    executor = None
    retry_loop = None
    candidate_selector = None

    db_dir = eval_cfg.get("db_dir", "")
    if db_dir:
        from src.post.sql_executor import SQLExecutor
        executor = SQLExecutor(db_dir=db_dir)
        retry_loop = _build_retry_loop(post_cfg, executor, inference)
        candidate_selector = _build_candidate_selector(post_cfg, executor)
    elif post_cfg.get("retry_loop", {}).get("enabled") or n_candidates > 1:
        logger.warning(
            "evaluation.db_dir not set — RetryLoop and CandidateSelector disabled."
        )

    # ---- Per-example inference + evaluation loop ----
    dev_examples = examples  # use all examples (filter as needed)
    results: list[dict] = []

    if inference is None and (retry_loop is not None or n_candidates > 1):
        logger.info(
            "Note: inference components configured but no model_path set. "
            "Running retrieval-only mode."
        )

    logger.info("Running pipeline on %d examples...", len(dev_examples))

    for i, ex in enumerate(dev_examples):
        db = db_map.get(ex.db_id)
        if db is None:
            continue

        # === PHASE 1: Pre-Retrieval ===
        value_hints: str = ""
        if value_scanner is not None:
            try:
                matches = value_scanner.scan(ex.question, db)
                value_hints = value_scanner.to_schema_hints(matches)
            except Exception as e:
                logger.debug("ValueScanner error on example %d: %s", i, e)

        aug_result = augmentor.augment(
            ex.question,
            db,
            value_scanner=value_scanner if aug_strategy == "value" else None,
            decomposer=decomposer if aug_strategy == "decompose" else None,
        )

        # === PHASE 2: Retrieval ===
        if isinstance(aug_result, list):
            # Multi-query (decompose strategy)
            expanded = linker.expand(
                retriever.retrieve_multi(aug_result, db_id=ex.db_id),
                db,
                [c for c in all_chunks if c.db_id == ex.db_id],
            )
        else:
            raw_results = retriever.retrieve(aug_result, db_id=ex.db_id)
            db_chunks = [c for c in all_chunks if c.db_id == ex.db_id]
            expanded = linker.expand(raw_results, db, db_chunks)

        schema_context = schema_filter.filter_and_format(
            expanded, db, value_hints=value_hints if value_hints else None
        )

        # Track retrieved tables for schema metrics
        retrieved_tables = schema_filter.get_retrieved_tables(expanded)

        # === PHASE 3: Generation ===
        gen_result: dict | None = None
        if inference is not None:
            try:
                raw_gen = inference.generate(
                    ex.question,
                    schema_context,
                    mode=gen_mode,
                    n_candidates=n_candidates,
                )

                # Phase 4a: CandidateSelector (if n_candidates > 1)
                if isinstance(raw_gen, list) and candidate_selector is not None:
                    gen_result = candidate_selector.select(raw_gen, ex.db_id)
                elif isinstance(raw_gen, list):
                    gen_result = raw_gen[0] if raw_gen else {}
                else:
                    gen_result = raw_gen

            except Exception as e:
                logger.warning("Inference error on example %d: %s", i, e)
                gen_result = {"sql": "", "reasoning": "", "raw_output": ""}

        # === PHASE 4b: Retry Loop ===
        retry_count = 0
        correction_applied = False

        if gen_result is not None and retry_loop is not None:
            try:
                gen_result = retry_loop.run(
                    question=ex.question,
                    schema_context=schema_context,
                    initial_result=gen_result,
                    db_id=ex.db_id,
                    gold_sql=ex.query,
                )
                retry_count = gen_result.get("retry_count", 0)
                correction_applied = gen_result.get("correction_applied", False)
            except Exception as e:
                logger.warning("RetryLoop error on example %d: %s", i, e)

        # === Evaluation ===
        predicted_sql = gen_result.get("sql", "") if gen_result else ""
        gold_sql = ex.query

        db_path = db.db_path or ""
        ex_result: bool = False
        em_result: bool = False

        if predicted_sql and db_path:
            ex_result = execution_accuracy(predicted_sql, gold_sql, db_path)
            em_result = exact_match(predicted_sql, gold_sql)
        elif predicted_sql:
            em_result = exact_match(predicted_sql, gold_sql)

        recall = schema_recall(retrieved_tables, gold_sql)
        precision = schema_precision(retrieved_tables, gold_sql)

        results.append({
            "db_id": ex.db_id,
            "question": ex.question,
            "gold_sql": gold_sql,
            "predicted_sql": predicted_sql,
            "ex": ex_result,
            "em": em_result,
            "recall": recall,
            "precision": precision,
            "retry_count": retry_count,
            "correction_applied": correction_applied,
        })

        if (i + 1) % 100 == 0:
            partial = compute_metrics(results)
            logger.info(
                "Progress %d/%d — EX=%.3f EM=%.3f recall=%.3f precision=%.3f",
                i + 1, len(dev_examples),
                partial["execution_accuracy"],
                partial["exact_match"],
                partial["schema_recall"],
                partial["schema_precision"],
            )

    # ---- Aggregate Results ----
    if results:
        metrics = compute_metrics(results)
        # Add retry stats
        total_retried = sum(1 for r in results if r.get("retry_count", 0) > 0)
        total_corrected = sum(1 for r in results if r.get("correction_applied", False))
        avg_retries = sum(r.get("retry_count", 0) for r in results) / len(results)

        logger.info("=== Final Metrics ===")
        logger.info("Execution Accuracy : %.4f", metrics["execution_accuracy"])
        logger.info("Exact Match        : %.4f", metrics["exact_match"])
        logger.info("Schema Recall      : %.4f", metrics["schema_recall"])
        logger.info("Schema Precision   : %.4f", metrics["schema_precision"])
        logger.info("Total Examples     : %d", metrics["total_examples"])

        if retry_loop is not None:
            logger.info("Examples Retried   : %d", total_retried)
            logger.info("Corrections Applied: %d", total_corrected)
            logger.info("Avg Retries        : %.2f", avg_retries)
    else:
        logger.info("Retrieval complete. Stored %d schema contexts.", len(results))
        logger.info(
            "No inference model configured. "
            "Set generation.model_path in config to enable generation + evaluation."
        )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text-to-SQL Pipeline")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values, e.g. post_generation.retry_loop.enabled=true",
    )
    args = parser.parse_args()
    main(args.config, overrides=args.override)
