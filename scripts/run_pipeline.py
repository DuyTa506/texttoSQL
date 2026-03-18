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
import asyncio
import hashlib
import json
import logging
import os
import sys
import time
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
from src.retrieval.utils.bidirectional_linker import BidirectionalLinker
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.utils.npmi_scorer import NPMIScorer
from src.pre_retrieval.query_augmentor import QueryAugmentor
from src.retrieval.utils.schema_filter import SchemaFilter

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
        from src.schema_graph import SchemaGraph
        from src.retrieval.graph_retriever import GraphRetriever
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

    # Use the same embedding encoder already built for ChromaDB indexing.
    # GraphRetriever._embed() supports both BaseEmbeddingModel (embed_one)
    # and SentenceTransformer (encode) APIs, so no sentence-transformers needed.
    embedder = _build_embedding_encoder(cfg["indexing"])
    if embedder is None:
        # Fallback to SentenceTransformer if no encoder was built
        try:
            from sentence_transformers import SentenceTransformer
            embedding_model = cfg["indexing"]["embedding_model"]
            logger.info("Loading SentenceTransformer '%s' for GraphRetriever ...", embedding_model)
            embedder = SentenceTransformer(embedding_model)
        except ImportError:
            logger.warning("No embedding encoder available for GraphRetriever — falling back to HybridRetriever.")
            return hybrid_retriever
    else:
        logger.info("Using existing embedding encoder for GraphRetriever.")

    graph_retriever = GraphRetriever(
        graph,
        embedder,
        top_m=sg_cfg.get("top_m", 5),
        alpha=sg_cfg.get("alpha", 0.7),
        max_hops=sg_cfg.get("max_hops", 2),
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
    from src.pre_retrieval.value_scanner import ValueScanner
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
    from src.pre_retrieval.question_decomposer import QuestionDecomposer
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

    # Include retriever mode in directory name so parallel runs don't collide
    sg_cfg = cfg.get("schema_graph", {})
    if sg_cfg.get("enabled", False) and sg_cfg.get("hybrid", False):
        retriever_tag = "merge"
    elif sg_cfg.get("enabled", False):
        retriever_tag = "graph"
    else:
        retriever_tag = "hybrid"
    setup_tag = f"{adapter}__{model_tag}__{gen_mode}__{retriever_tag}"

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
# Phase 1+2 extracted helper  (reusable by both sync and async loops)
# =============================================================================

def _process_phases_1_2(
    question: str,
    ex,
    db,
    *,
    value_scanner,
    augmentor,
    aug_strategy: str,
    decomposer,
    retriever,
    linker,
    all_chunks: list,
    schema_filter,
    idx: int = 0,
) -> dict:
    """Run Phase 1 (pre-retrieval) + Phase 2 (retrieval) for one example.

    Returns a dict with:
      - question: str (with evidence hint if applicable)
      - schema_context: str
      - retrieved_tables: set[str]
      - value_hints: str
    """
    # === PHASE 1: Pre-Retrieval ===
    value_hints: str = ""
    value_matches_list = None
    if value_scanner is not None:
        try:
            matches = value_scanner.scan(question, db)
            value_hints = value_scanner.to_schema_hints(matches)
            value_matches_list = matches
        except Exception as e:
            logger.debug("ValueScanner error on example %d: %s", idx, e)

    aug_result = augmentor.augment(
        question,
        db,
        value_scanner=value_scanner if aug_strategy == "value" else None,
        decomposer=decomposer if aug_strategy == "decompose" else None,
    )

    # === PHASE 2: Retrieval ===
    retriever_kwargs = {"db_id": ex.db_id}
    if value_matches_list and hasattr(retriever, 'graph'):
        retriever_kwargs["value_matches"] = value_matches_list

    if isinstance(aug_result, list):
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
    retrieved_tables = schema_filter.get_retrieved_tables(expanded)

    return {
        "question": question,
        "schema_context": schema_context,
        "retrieved_tables": retrieved_tables,
        "value_hints": value_hints,
    }


# =============================================================================
# Async batch pipeline loop
# =============================================================================

def _run_async_pipeline_loop(
    *,
    dev_examples: list,
    db_map: dict,
    results: list[dict],
    completed_keys: set[str],
    run_dir: Path,
    # Phase 1+2 components
    value_scanner,
    augmentor,
    aug_strategy: str,
    decomposer,
    retriever,
    linker,
    all_chunks: list,
    schema_filter,
    # Phase 3 components
    inference,
    gen_mode: str,
    n_candidates: int,
    # Phase 4 components
    retry_loop,
    candidate_selector,
    # Batch config
    batch_size: int = 32,
    concurrency: int = 20,
):
    """Run the pipeline with async/batch LLM inference.

    Phases 1+2 run sequentially per example (fast, <20ms each).
    Phase 3 LLM calls are batched and run in parallel.
    Phases 4 + evaluation run sequentially per example.
    """
    asyncio.run(_async_pipeline_loop_inner(
        dev_examples=dev_examples,
        db_map=db_map,
        results=results,
        completed_keys=completed_keys,
        run_dir=run_dir,
        value_scanner=value_scanner,
        augmentor=augmentor,
        aug_strategy=aug_strategy,
        decomposer=decomposer,
        retriever=retriever,
        linker=linker,
        all_chunks=all_chunks,
        schema_filter=schema_filter,
        inference=inference,
        gen_mode=gen_mode,
        n_candidates=n_candidates,
        retry_loop=retry_loop,
        candidate_selector=candidate_selector,
        batch_size=batch_size,
        concurrency=concurrency,
    ))


async def _async_pipeline_loop_inner(
    *,
    dev_examples: list,
    db_map: dict,
    results: list[dict],
    completed_keys: set[str],
    run_dir: Path,
    value_scanner,
    augmentor,
    aug_strategy: str,
    decomposer,
    retriever,
    linker,
    all_chunks: list,
    schema_filter,
    inference,
    gen_mode: str,
    n_candidates: int,
    retry_loop,
    candidate_selector,
    batch_size: int,
    concurrency: int,
):
    """Inner async implementation of the batch pipeline loop."""
    from src.evaluation.metrics import compute_metrics, execution_accuracy, exact_match, schema_recall, schema_precision

    # Filter to pending examples
    pending = []
    for i, ex in enumerate(dev_examples):
        db = db_map.get(ex.db_id)
        if db is None:
            continue
        ex_key = f"{ex.db_id}||{ex.question}"
        if ex_key in completed_keys:
            continue
        # Prepare question with evidence hint
        question = ex.question
        if getattr(ex, "evidence", "") and ex.evidence.strip():
            question = f"{ex.question}\nHint: {ex.evidence.strip()}"
        pending.append((i, ex, db, question))

    if not pending:
        logger.info("All examples already completed — nothing to do.")
        return

    logger.info(
        "Async batch mode: %d pending examples, batch_size=%d, concurrency=%d",
        len(pending), batch_size, concurrency,
    )

    total_processed = 0
    t0 = time.time()

    # Process in micro-batches
    for batch_start in range(0, len(pending), batch_size):
        batch = pending[batch_start : batch_start + batch_size]
        batch_t0 = time.time()

        # ── Phase 1 + 2: sequential per example (fast) ────────────────────
        phase12_results = []
        for i, ex, db, question in batch:
            p12 = _process_phases_1_2(
                question, ex, db,
                value_scanner=value_scanner,
                augmentor=augmentor,
                aug_strategy=aug_strategy,
                decomposer=decomposer,
                retriever=retriever,
                linker=linker,
                all_chunks=all_chunks,
                schema_filter=schema_filter,
                idx=i,
            )
            phase12_results.append(p12)

        # ── Phase 3: batched LLM generation ───────────────────────────────
        gen_results: list = []
        if inference is not None:
            items = [
                (p12["question"], p12["schema_context"])
                for p12 in phase12_results
            ]
            try:
                gen_results = await inference.agenerate_batch(
                    items,
                    mode=gen_mode,
                    n_candidates=n_candidates,
                    concurrency=concurrency,
                )
            except Exception as exc:
                logger.warning("Batch inference error: %s — falling back to empty", exc)
                gen_results = [
                    {"sql": "", "reasoning": "", "raw_output": ""}
                    for _ in items
                ]
        else:
            gen_results = [None] * len(batch)

        # ── Phase 4 + Evaluation: sequential per example ──────────────────
        for j, (i, ex, db, question) in enumerate(batch):
            p12 = phase12_results[j]
            raw_gen = gen_results[j]

            # Phase 4a: CandidateSelector
            gen_result = None
            if raw_gen is not None:
                if isinstance(raw_gen, list) and candidate_selector is not None:
                    gen_result = candidate_selector.select(raw_gen, ex.db_id)
                elif isinstance(raw_gen, list):
                    gen_result = raw_gen[0] if raw_gen else {}
                else:
                    gen_result = raw_gen

            # Phase 4b: Retry Loop
            retry_count = 0
            correction_applied = False
            if gen_result is not None and retry_loop is not None:
                try:
                    gen_result = retry_loop.run(
                        question=p12["question"],
                        schema_context=p12["schema_context"],
                        initial_result=gen_result,
                        db_id=ex.db_id,
                        gold_sql=ex.query,
                    )
                    retry_count = gen_result.get("retry_count", 0)
                    correction_applied = gen_result.get("correction_applied", False)
                except Exception as e:
                    logger.warning("RetryLoop error on example %d: %s", i, e)

            # Evaluation
            predicted_sql = gen_result.get("sql", "") if gen_result else ""
            gold_sql = ex.query
            db_path = db.db_path or ""

            ex_result = False
            em_result = False
            if predicted_sql and db_path:
                ex_result = execution_accuracy(predicted_sql, gold_sql, db_path)
                em_result = exact_match(predicted_sql, gold_sql)
            elif predicted_sql:
                em_result = exact_match(predicted_sql, gold_sql)

            recall_val = schema_recall(p12["retrieved_tables"], gold_sql)
            precision_val = schema_precision(p12["retrieved_tables"], gold_sql)

            result_entry = {
                "db_id": ex.db_id,
                "question": p12["question"],
                "evidence": getattr(ex, "evidence", ""),
                "schema_context": p12["schema_context"],
                "gold_sql": gold_sql,
                "predicted_sql": predicted_sql,
                "raw_output": gen_result.get("raw_output", "") if gen_result else "",
                "ex": ex_result,
                "em": em_result,
                "recall": recall_val,
                "precision": precision_val,
                "retry_count": retry_count,
                "correction_applied": correction_applied,
            }
            results.append(result_entry)
            _append_result(run_dir, result_entry)

        total_processed += len(batch)
        batch_elapsed = time.time() - batch_t0

        if total_processed % 100 < batch_size or batch_start + batch_size >= len(pending):
            partial = compute_metrics(results)
            elapsed = time.time() - t0
            logger.info(
                "Batch progress %d/%d (%.1fs elapsed, %.2fs/batch) — "
                "EX=%.3f EM=%.3f recall=%.3f precision=%.3f",
                total_processed, len(pending), elapsed, batch_elapsed,
                partial["execution_accuracy"],
                partial["exact_match"],
                partial["schema_recall"],
                partial["schema_precision"],
            )


# =============================================================================
# Multi-mode benchmark orchestration
# =============================================================================

# Mode override presets
_MODE_OVERRIDES = {
    "hybrid": {
        "schema_graph.enabled": False,
        "schema_graph.hybrid": False,
    },
    "graph": {
        "schema_graph.enabled": True,
        "schema_graph.hybrid": False,
    },
    "merge": {
        "schema_graph.enabled": True,
        "schema_graph.hybrid": True,
    },
}


def _apply_mode_overrides(cfg: dict, mode: str) -> dict:
    """Apply mode-specific config overrides (returns a modified copy)."""
    import copy
    cfg = copy.deepcopy(cfg)
    overrides = _MODE_OVERRIDES.get(mode, {})
    for key_path, value in overrides.items():
        parts = key_path.split(".")
        node = cfg
        for part in parts[:-1]:
            if part not in node:
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value
    return cfg


def _run_single_mode(
    cfg: dict,
    mode_name: str,
    *,
    databases,
    examples,
    all_chunks,
    indexer,
    npmi_scorer,
    batch_enabled: bool,
    batch_size: int,
    concurrency: int,
) -> dict:
    """Run the pipeline for a single retriever mode.

    Returns the final metrics dict.
    """
    mode_cfg = _apply_mode_overrides(cfg, mode_name)

    logger.info("=" * 60)
    logger.info("=== Running mode: %s ===", mode_name.upper())
    logger.info("=" * 60)

    # Build retriever for this mode
    retriever = _build_retriever(mode_cfg, indexer, all_chunks, npmi_scorer=npmi_scorer)

    ret_cfg = mode_cfg["retrieval"]
    linker = BidirectionalLinker(
        max_expansion_depth=ret_cfg["bidirectional"]["max_expansion_depth"],
    )
    schema_filter_obj = SchemaFilter(top_k=ret_cfg["final_top_k"])

    pre_cfg = mode_cfg.get("pre_retrieval", {})
    gen_cfg = mode_cfg.get("generation", {})
    post_cfg = mode_cfg.get("post_generation", {})
    eval_cfg = mode_cfg.get("evaluation", {})

    aug_strategy = mode_cfg["augmentation"].get("strategy", "keyword")
    augmentor = QueryAugmentor(strategy=aug_strategy)

    value_scanner = _build_value_scanner(pre_cfg)
    decomposer = _build_decomposer(pre_cfg)
    inference = _build_inference(gen_cfg)
    gen_mode = gen_cfg.get("mode", "standard")
    n_candidates = gen_cfg.get("n_candidates", 1)

    # Phase 4 components
    db_dir = eval_cfg.get("db_dir", "")
    executor = None
    retry_loop = None
    candidate_selector = None
    if db_dir:
        from src.post.sql_executor import SQLExecutor
        executor = SQLExecutor(db_dir=db_dir)
        retry_loop = _build_retry_loop(post_cfg, executor, inference)
        candidate_selector = _build_candidate_selector(post_cfg, executor)

    db_map = {db.db_id: db for db in databases}

    max_examples = mode_cfg.get("_max_examples", 0)
    dev_examples = examples[:max_examples] if max_examples else examples

    # Use mode-specific run directory
    output_base = eval_cfg.get("output_dir", "./results")
    # Tag directory with mode name
    original_model = gen_cfg.get("model_path", "none") or "none"
    model_tag = original_model.replace("/", "_").replace(".", "_")
    adapter = mode_cfg.get("data", {}).get("adapter", "spider_v1")
    setup_tag = f"{adapter}__{model_tag}__{gen_mode}__{mode_name}"
    run_dir = Path(output_base) / setup_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(mode_cfg, f, default_flow_style=False)

    results: list[dict] = _load_existing_results(run_dir)
    completed_keys: set[str] = set()
    for r in results:
        completed_keys.add(f"{r['db_id']}||{r['question']}")
    if results:
        logger.info("[%s] Resuming: %d examples already completed", mode_name, len(results))

    t0 = time.time()

    if batch_enabled and inference is not None:
        _run_async_pipeline_loop(
            dev_examples=dev_examples,
            db_map=db_map,
            results=results,
            completed_keys=completed_keys,
            run_dir=run_dir,
            value_scanner=value_scanner,
            augmentor=augmentor,
            aug_strategy=aug_strategy,
            decomposer=decomposer,
            retriever=retriever,
            linker=linker,
            all_chunks=all_chunks,
            schema_filter=schema_filter_obj,
            inference=inference,
            gen_mode=gen_mode,
            n_candidates=n_candidates,
            retry_loop=retry_loop,
            candidate_selector=candidate_selector,
            batch_size=batch_size,
            concurrency=concurrency,
        )
    else:
        # Original sequential loop
        for i, ex in enumerate(dev_examples):
            db = db_map.get(ex.db_id)
            if db is None:
                continue
            ex_key = f"{ex.db_id}||{ex.question}"
            if ex_key in completed_keys:
                continue

            question = ex.question
            if getattr(ex, "evidence", "") and ex.evidence.strip():
                question = f"{ex.question}\nHint: {ex.evidence.strip()}"

            p12 = _process_phases_1_2(
                question, ex, db,
                value_scanner=value_scanner,
                augmentor=augmentor,
                aug_strategy=aug_strategy,
                decomposer=decomposer,
                retriever=retriever,
                linker=linker,
                all_chunks=all_chunks,
                schema_filter=schema_filter_obj,
                idx=i,
            )

            # Phase 3: Generation
            gen_result = None
            if inference is not None:
                try:
                    raw_gen = inference.generate(
                        p12["question"], p12["schema_context"],
                        mode=gen_mode, n_candidates=n_candidates,
                    )
                    if isinstance(raw_gen, list) and candidate_selector is not None:
                        gen_result = candidate_selector.select(raw_gen, ex.db_id)
                    elif isinstance(raw_gen, list):
                        gen_result = raw_gen[0] if raw_gen else {}
                    else:
                        gen_result = raw_gen
                except Exception as e:
                    logger.warning("Inference error on example %d: %s", i, e)
                    gen_result = {"sql": "", "reasoning": "", "raw_output": ""}

            # Phase 4b: Retry
            retry_count = 0
            correction_applied = False
            if gen_result is not None and retry_loop is not None:
                try:
                    gen_result = retry_loop.run(
                        question=p12["question"],
                        schema_context=p12["schema_context"],
                        initial_result=gen_result,
                        db_id=ex.db_id,
                        gold_sql=ex.query,
                    )
                    retry_count = gen_result.get("retry_count", 0)
                    correction_applied = gen_result.get("correction_applied", False)
                except Exception as e:
                    logger.warning("RetryLoop error on example %d: %s", i, e)

            # Evaluation
            predicted_sql = gen_result.get("sql", "") if gen_result else ""
            gold_sql = ex.query
            db_path = db.db_path or ""

            ex_result = False
            em_result = False
            if predicted_sql and db_path:
                ex_result = execution_accuracy(predicted_sql, gold_sql, db_path)
                em_result = exact_match(predicted_sql, gold_sql)
            elif predicted_sql:
                em_result = exact_match(predicted_sql, gold_sql)

            recall_val = schema_recall(p12["retrieved_tables"], gold_sql)
            precision_val = schema_precision(p12["retrieved_tables"], gold_sql)

            result_entry = {
                "db_id": ex.db_id,
                "question": p12["question"],
                "evidence": getattr(ex, "evidence", ""),
                "schema_context": p12["schema_context"],
                "gold_sql": gold_sql,
                "predicted_sql": predicted_sql,
                "raw_output": gen_result.get("raw_output", "") if gen_result else "",
                "ex": ex_result,
                "em": em_result,
                "recall": recall_val,
                "precision": precision_val,
                "retry_count": retry_count,
                "correction_applied": correction_applied,
            }
            results.append(result_entry)
            _append_result(run_dir, result_entry)

            if (i + 1) % 100 == 0:
                partial = compute_metrics(results)
                logger.info(
                    "[%s] Progress %d/%d — EX=%.3f EM=%.3f recall=%.3f precision=%.3f",
                    mode_name, i + 1, len(dev_examples),
                    partial["execution_accuracy"],
                    partial["exact_match"],
                    partial["schema_recall"],
                    partial["schema_precision"],
                )

    elapsed = time.time() - t0

    # Compute final metrics
    final_metrics = {}
    if results:
        metrics = compute_metrics(results)
        total_retried = sum(1 for r in results if r.get("retry_count", 0) > 0)
        total_corrected = sum(1 for r in results if r.get("correction_applied", False))
        avg_retries = sum(r.get("retry_count", 0) for r in results) / len(results)

        final_metrics = {
            **metrics,
            "mode": mode_name,
            "elapsed_seconds": round(elapsed, 1),
            "total_retried": total_retried,
            "total_corrected": total_corrected,
            "avg_retries": round(avg_retries, 4),
        }

        logger.info("=== [%s] Final Metrics (%.1fs) ===", mode_name, elapsed)
        logger.info("Execution Accuracy : %.4f", metrics["execution_accuracy"])
        logger.info("Exact Match        : %.4f", metrics["exact_match"])
        logger.info("Schema Recall      : %.4f", metrics["schema_recall"])
        logger.info("Schema Precision   : %.4f", metrics["schema_precision"])
        logger.info("Total Examples     : %d", metrics["total_examples"])

    _save_final(run_dir, results, final_metrics)
    logger.info("=== [%s] Results saved to: %s ===", mode_name, run_dir)

    return final_metrics


def _benchmark_all(
    config_path: str,
    overrides: list[str] | None = None,
    modes: list[str] | None = None,
    batch_enabled: bool = True,
    batch_size: int = 32,
    concurrency: int = 20,
):
    """Run the pipeline across multiple retriever modes and compare results.

    Shared one-time setup (data loading, ChromaDB indexing, model loading),
    then loops over modes.
    """
    cfg = load_config(config_path)
    if overrides:
        cfg = apply_overrides(cfg, overrides)

    modes = modes or ["hybrid", "graph", "merge"]

    # ---- Shared one-time setup ----
    logger.info("=== Benchmark Setup: Loading data + indexing ===")
    adapter_name = cfg["data"].get("adapter", "spider_v1")
    parser = get_parser(adapter_name)
    databases, examples = parser.load(cfg["data"]["dataset_path"])
    logger.info("Loaded %d databases, %d examples", len(databases), len(examples))

    # Chunk schemas
    chunker = SchemaChunker(max_sample_values=cfg["chunking"].get("max_sample_values", 5))
    all_chunks = chunker.chunk_many(databases)
    logger.info("Generated %d schema chunks", len(all_chunks))

    # Index into ChromaDB
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
        logger.info("Index cache hit — skipping re-indexing.")
    else:
        indexer.index(all_chunks, reset=True)
        _save_index_fingerprint(idx_cfg["chroma_persist_dir"], fingerprint, len(all_chunks))

    # Optional NPMI scorer
    npmi_scorer = None
    npmi_cfg = cfg.get("npmi", {})
    if npmi_cfg.get("enable") and npmi_cfg.get("matrix_path"):
        npmi_scorer = NPMIScorer.load(npmi_cfg["matrix_path"])

    # ---- Run each mode ----
    all_metrics = {}
    for mode_name in modes:
        if mode_name not in _MODE_OVERRIDES:
            logger.warning("Unknown mode '%s' — skipping. Valid: %s",
                          mode_name, list(_MODE_OVERRIDES.keys()))
            continue

        metrics = _run_single_mode(
            cfg, mode_name,
            databases=databases,
            examples=examples,
            all_chunks=all_chunks,
            indexer=indexer,
            npmi_scorer=npmi_scorer,
            batch_enabled=batch_enabled,
            batch_size=batch_size,
            concurrency=concurrency,
        )
        all_metrics[mode_name] = metrics

    # ---- Print comparison table ----
    if all_metrics:
        logger.info("")
        logger.info("=" * 72)
        logger.info("BENCHMARK COMPARISON")
        logger.info("=" * 72)
        logger.info(
            "%-10s  %8s  %8s  %8s  %8s  %8s",
            "Mode", "EX", "EM", "Recall", "Prec.", "Time(s)",
        )
        logger.info("-" * 72)
        for mode_name, m in all_metrics.items():
            if m:
                logger.info(
                    "%-10s  %8.4f  %8.4f  %8.4f  %8.4f  %8.1f",
                    mode_name,
                    m.get("execution_accuracy", 0),
                    m.get("exact_match", 0),
                    m.get("schema_recall", 0),
                    m.get("schema_precision", 0),
                    m.get("elapsed_seconds", 0),
                )
        logger.info("=" * 72)

    return all_metrics


# =============================================================================
# Main pipeline
# =============================================================================

def main(config_path: str, overrides: list[str] | None = None,
         batch_enabled: bool = False, batch_size: int = 32, concurrency: int = 20):
    cfg = load_config(config_path)
    if overrides:
        cfg = apply_overrides(cfg, overrides)

    # Read batch config from YAML (CLI args override)
    batch_cfg = cfg.get("batch", {})
    if not batch_enabled:
        batch_enabled = batch_cfg.get("enabled", False)
    if batch_size == 32:  # default — check config
        batch_size = batch_cfg.get("batch_size", 32)
    if concurrency == 20:  # default — check config
        concurrency = batch_cfg.get("concurrency", 20)

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

    if batch_enabled and inference is not None:
        logger.info("Async batch mode enabled (batch_size=%d, concurrency=%d)",
                    batch_size, concurrency)
        _run_async_pipeline_loop(
            dev_examples=dev_examples,
            db_map=db_map,
            results=results,
            completed_keys=completed_keys,
            run_dir=run_dir,
            value_scanner=value_scanner,
            augmentor=augmentor,
            aug_strategy=aug_strategy,
            decomposer=decomposer,
            retriever=retriever,
            linker=linker,
            all_chunks=all_chunks,
            schema_filter=schema_filter,
            inference=inference,
            gen_mode=gen_mode,
            n_candidates=n_candidates,
            retry_loop=retry_loop,
            candidate_selector=candidate_selector,
            batch_size=batch_size,
            concurrency=concurrency,
        )
    else:
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

            p12 = _process_phases_1_2(
                question, ex, db,
                value_scanner=value_scanner,
                augmentor=augmentor,
                aug_strategy=aug_strategy,
                decomposer=decomposer,
                retriever=retriever,
                linker=linker,
                all_chunks=all_chunks,
                schema_filter=schema_filter,
                idx=i,
            )

            # === PHASE 3: Generation ===
            gen_result: dict | None = None
            if inference is not None:
                try:
                    raw_gen = inference.generate(
                        p12["question"],
                        p12["schema_context"],
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
                        question=p12["question"],
                        schema_context=p12["schema_context"],
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

            recall = schema_recall(p12["retrieved_tables"], gold_sql)
            precision = schema_precision(p12["retrieved_tables"], gold_sql)

            result_entry = {
                "db_id": ex.db_id,
                "question": p12["question"],
                "evidence": getattr(ex, "evidence", ""),
                "schema_context": p12["schema_context"],
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
    # Batch / async inference flags
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Enable async/batch LLM inference for faster pipeline execution.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of examples per micro-batch (default: 32).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=20,
        help="Max parallel API calls for async inference (default: 20).",
    )
    # Multi-mode benchmark flags
    parser.add_argument(
        "--benchmark_all",
        action="store_true",
        help="Run all retriever modes (hybrid, graph, merge) and compare results.",
    )
    parser.add_argument(
        "--modes",
        nargs="*",
        default=["hybrid", "graph", "merge"],
        help="Retriever modes to benchmark (default: hybrid graph merge).",
    )
    args = parser.parse_args()
    overrides = args.override or []
    if args.reindex:
        overrides.append("_reindex=true")
    if args.max_examples:
        overrides.append(f"_max_examples={args.max_examples}")

    if args.benchmark_all:
        _benchmark_all(
            args.config,
            overrides=overrides,
            modes=args.modes,
            batch_enabled=args.batch,
            batch_size=args.batch_size,
            concurrency=args.concurrency,
        )
    else:
        main(
            args.config,
            overrides=overrides,
            batch_enabled=args.batch,
            batch_size=args.batch_size,
            concurrency=args.concurrency,
        )
