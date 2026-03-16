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
import hashlib
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env file (if present) so OPENAI_API_KEY etc. are available
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass  # python-dotenv is optional; users can export env vars manually

from src.data_parser import get_parser
from src.schema.schema_chunker import SchemaChunker
from src.schema.schema_indexer import SchemaIndexer
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

def _build_embedding_encoder(idx_cfg: dict):
    """
    Build an embedding encoder from the indexing config.

    When ``embedding_provider`` is ``"openai"`` (default), returns an
    :class:`OpenAIEmbeddingModel` — no torch required.

    When ``"huggingface"``, returns a :class:`HuggingFaceEmbeddingModel`
    (requires torch + sentence-transformers).

    Returns ``None`` if the provider is unrecognised, which makes
    :class:`SchemaIndexer` fall back to its built-in SentenceTransformer
    lazy-loader (backward compat).
    """
    provider = idx_cfg.get("embedding_provider", "openai").lower()
    model_name = idx_cfg.get("embedding_model", "")

    if provider == "openai":
        from src.embeddings import OpenAIEmbeddingModel

        api_key = idx_cfg.get("embedding_api_key", "") or os.environ.get("OPENAI_API_KEY", "")
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

    logger.warning(
        "Unknown embedding_provider '%s' — falling back to SentenceTransformer.",
        provider,
    )
    return None


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
        score_gap_ratio=sg_cfg.get("score_gap_ratio", 3.0),       # v3
        use_fk_bridge=sg_cfg.get("use_fk_bridge", True),          # v3
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
# Index caching (fingerprint-based)
# =============================================================================

def _compute_index_fingerprint(cfg: dict, chunk_count: int) -> str:
    """Compute a deterministic fingerprint for the current indexing config.

    The fingerprint changes when any of these change:
      - dataset_path, adapter
      - embedding_provider, embedding_model
      - collection_name
      - number of schema chunks
    """
    parts = [
        cfg.get("data", {}).get("dataset_path", ""),
        cfg.get("data", {}).get("adapter", "spider_v1"),
        cfg.get("indexing", {}).get("embedding_provider", "openai"),
        cfg.get("indexing", {}).get("embedding_model", ""),
        cfg.get("indexing", {}).get("collection_name", "schema_chunks"),
        str(chunk_count),
    ]
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _check_index_cache(persist_dir: str, fingerprint: str) -> bool:
    """Return True if the cached index matches the current fingerprint."""
    fp_path = Path(persist_dir) / ".index_fingerprint"
    if not fp_path.exists():
        return False
    try:
        stored = json.loads(fp_path.read_text())
        return stored.get("fingerprint") == fingerprint
    except Exception:
        return False


def _save_index_fingerprint(persist_dir: str, fingerprint: str, chunk_count: int):
    """Write the fingerprint so subsequent runs can skip indexing."""
    fp_path = Path(persist_dir) / ".index_fingerprint"
    fp_path.parent.mkdir(parents=True, exist_ok=True)
    fp_path.write_text(json.dumps({
        "fingerprint": fingerprint,
        "chunk_count": chunk_count,
        "created_at": datetime.now().isoformat(),
    }))


# =============================================================================
# Result saving (streaming JSONL + resume)
# =============================================================================

def _make_run_dir(cfg: dict, output_base: str = "./results") -> Path:
    """Create a deterministic run directory for this config.

    Layout::

        {output_base}/{adapter}__{model}__{mode}/
            results.jsonl      — one JSON object per line (streamed)
            predictions.sql    — predicted SQL (written at end)
            gold.sql           — gold SQL (written at end)
            metrics.json       — aggregate metrics (written at end)
            config.yaml        — config snapshot

    Uses a *stable* directory name (no timestamp) so re-runs with the same
    config resume from where they left off.
    """
    adapter = cfg.get("data", {}).get("adapter", "spider_v1")
    model = cfg.get("generation", {}).get("model_path", "none") or "none"
    model_tag = model.replace("/", "_").replace(".", "_")
    gen_mode = cfg.get("generation", {}).get("mode", "standard")
    setup_tag = f"{adapter}__{model_tag}__{gen_mode}"

    run_dir = Path(output_base) / setup_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    return run_dir


def _load_existing_results(run_dir: Path) -> list[dict]:
    """Load previously saved results from JSONL (for resume)."""
    jsonl_path = run_dir / "results.jsonl"
    if not jsonl_path.exists():
        return []
    results = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # skip corrupted lines
    return results


def _append_result(run_dir: Path, result: dict):
    """Append a single result to the JSONL file."""
    with open(run_dir / "results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def _save_final(run_dir: Path, results: list[dict], metrics: dict | None):
    """Write final summary files after the loop completes."""
    # Predicted SQL (one per line)
    with open(run_dir / "predictions.sql", "w", encoding="utf-8") as f:
        for r in results:
            f.write(r.get("predicted_sql", "").replace("\n", " ") + "\n")

    # Gold SQL (one per line)
    with open(run_dir / "gold.sql", "w", encoding="utf-8") as f:
        for r in results:
            f.write(r.get("gold_sql", "").replace("\n", " ") + "\n")

    # Metrics
    if metrics:
        with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)


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
    adapter_name = cfg["data"].get("adapter", "spider_v1")
    parser = get_parser(adapter_name)
    databases, examples = parser.load(cfg["data"]["dataset_path"])
    logger.info("Loaded %d databases, %d examples (adapter=%s)",
                len(databases), len(examples), adapter_name)

    # Chunk schemas
    chunker = SchemaChunker(max_sample_values=cfg["chunking"].get("max_sample_values", 5))
    all_chunks = chunker.chunk_many(databases)
    logger.info("Generated %d schema chunks", len(all_chunks))

    # Index into ChromaDB (skip if fingerprint matches)
    idx_cfg = cfg["indexing"]
    embedding_encoder = _build_embedding_encoder(idx_cfg)
    indexer = SchemaIndexer(
        embedding_model=idx_cfg["embedding_model"],
        persist_dir=idx_cfg["chroma_persist_dir"],
        collection_name=idx_cfg["collection_name"],
        batch_size=idx_cfg["batch_size"],
        encoder=embedding_encoder,
    )

    fingerprint = _compute_index_fingerprint(cfg, len(all_chunks))
    reindex = cfg.get("_reindex", False)

    if not reindex and _check_index_cache(idx_cfg["chroma_persist_dir"], fingerprint):
        logger.info("Index cache hit (fingerprint=%s) — skipping re-indexing.", fingerprint)
    else:
        if reindex:
            logger.info("--reindex flag set — forcing re-indexing.")
        indexer.index(all_chunks, reset=True)
        _save_index_fingerprint(idx_cfg["chroma_persist_dir"], fingerprint, len(all_chunks))

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
    max_examples = cfg.get("_max_examples", 0)
    dev_examples = examples[:max_examples] if max_examples else examples

    if inference is None and (retry_loop is not None or n_candidates > 1):
        logger.info(
            "Note: inference components configured but no model_path set. "
            "Running retrieval-only mode."
        )

    # Set up run directory for streaming results
    output_base = eval_cfg.get("output_dir", "./results")
    run_dir = _make_run_dir(cfg, output_base=output_base)

    # Resume: load existing results and skip already-completed examples
    results: list[dict] = _load_existing_results(run_dir)
    completed_keys: set[str] = set()
    for r in results:
        completed_keys.add(f"{r['db_id']}||{r['question']}")
    if results:
        logger.info("Resuming: %d examples already completed in %s", len(results), run_dir)

    logger.info("Running pipeline on %d examples (%d remaining)...",
                len(dev_examples), len(dev_examples) - len(results))

    for i, ex in enumerate(dev_examples):
        db = db_map.get(ex.db_id)
        if db is None:
            continue

        # Skip already-completed examples (resume)
        ex_key = f"{ex.db_id}||{ex.question}"
        if ex_key in completed_keys:
            continue

        # === Evidence injection (BIRD "evidence" field) ===
        question = ex.question
        if getattr(ex, "evidence", "") and ex.evidence.strip():
            question = f"{ex.question}\nHint: {ex.evidence.strip()}"

        # === PHASE 1: Pre-Retrieval ===
        value_hints: str = ""
        value_matches_list = None       # v3: for GraphRetriever seed boosting
        if value_scanner is not None:
            try:
                matches = value_scanner.scan(question, db)
                value_hints = value_scanner.to_schema_hints(matches)
                value_matches_list = matches  # v3: pass to retriever
            except Exception as e:
                logger.debug("ValueScanner error on example %d: %s", i, e)

        aug_result = augmentor.augment(
            question,
            db,
            value_scanner=value_scanner if aug_strategy == "value" else None,
            decomposer=decomposer if aug_strategy == "decompose" else None,
        )

        # === PHASE 2: Retrieval ===
        # v3: Pass value_matches to GraphRetriever for seed boosting.
        # HybridRetriever ignores unknown kwargs via its own signature,
        # so we use hasattr to detect GraphRetriever's value_matches support.
        retriever_kwargs = {"db_id": ex.db_id}
        if value_matches_list and hasattr(retriever, 'graph'):
            retriever_kwargs["value_matches"] = value_matches_list

        if isinstance(aug_result, list):
            # Multi-query (decompose strategy)
            expanded = linker.expand(
                retriever.retrieve_multi(aug_result, **retriever_kwargs),
                db,
                [c for c in all_chunks if c.db_id == ex.db_id],
            )
        else:
            raw_results = retriever.retrieve(aug_result, **retriever_kwargs)
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
                    question,
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
                    question=question,
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

        result_entry = {
            "db_id": ex.db_id,
            "question": question,  # includes evidence hint if present
            "evidence": getattr(ex, "evidence", ""),
            "schema_context": schema_context,
            "gold_sql": gold_sql,
            "predicted_sql": predicted_sql,
            "raw_output": gen_result.get("raw_output", "") if gen_result else "",
            "ex": ex_result,
            "em": em_result,
            "recall": recall,
            "precision": precision,
            "retry_count": retry_count,
            "correction_applied": correction_applied,
        }
        results.append(result_entry)
        _append_result(run_dir, result_entry)

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
    final_metrics = None
    if results:
        metrics = compute_metrics(results)
        # Add retry stats
        total_retried = sum(1 for r in results if r.get("retry_count", 0) > 0)
        total_corrected = sum(1 for r in results if r.get("correction_applied", False))
        avg_retries = sum(r.get("retry_count", 0) for r in results) / len(results)

        final_metrics = {
            **metrics,
            "total_retried": total_retried,
            "total_corrected": total_corrected,
            "avg_retries": round(avg_retries, 4),
        }

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

    # ---- Write final summary files ----
    _save_final(run_dir, results, final_metrics)
    logger.info("=== Results saved to: %s ===", run_dir)

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
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force re-indexing even if the index cache is valid.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=0,
        help="Limit number of examples to process (0 = all).",
    )
    args = parser.parse_args()
    overrides = args.override or []
    if args.reindex:
        overrides.append("_reindex=true")
    if args.max_examples:
        overrides.append(f"_max_examples={args.max_examples}")
    main(args.config, overrides=overrides)
