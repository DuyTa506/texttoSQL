"""
Build Schema Knowledge Graph — offline CLI script.

Runs the full 3-layer graph construction pipeline for one or more databases:

  Layer 1  Structural edges (DDL-derived, instant)
  Layer 2a Lexical semantic edges (token overlap, instant)
  LLM      Node enrichment — descriptions + synonyms (requires API key)
  Embed    Node embeddings via SentenceTransformer
  Layer 2b Embedding + synonym semantic edges (after enrichment)
  Layer 3  Statistical edges — SQL co-occurrence + DB value overlap

The completed graph is saved to a JSON file for fast loading at inference time.

Usage
-----
  # Spider 1.0 — structural + semantic only (no LLM, no stats)
  python scripts/build_schema_graph.py \\
      --data_path  data/spider \\
      --output     data/schema_graphs/spider.json

  # With LLM enrichment (adds descriptions, synonyms, embedding edges)
  python scripts/build_schema_graph.py \\
      --data_path  data/spider \\
      --output     data/schema_graphs/spider_enriched.json \\
      --enrich \\
      --enrich_model gpt-4o-mini \\
      --api_key  $OPENAI_API_KEY

  # Full pipeline: enrichment + statistical edges
  python scripts/build_schema_graph.py \\
      --data_path  data/spider \\
      --db_dir     data/spider/database \\
      --output     data/schema_graphs/spider_full.json \\
      --enrich \\
      --enrich_model gpt-4o-mini \\
      --statistical \\
      --value_overlap

  # Single database only
  python scripts/build_schema_graph.py \\
      --data_path  data/spider \\
      --output     data/schema_graphs/concert_singer.json \\
      --db_filter  concert_singer

  # Resume: reload existing graph and add statistical edges
  python scripts/build_schema_graph.py \\
      --load_existing data/schema_graphs/spider_enriched.json \\
      --output        data/schema_graphs/spider_full.json \\
      --data_path     data/spider \\
      --statistical

Output JSON can be loaded at inference time with:
  graph = SchemaGraph.load("data/schema_graphs/spider_full.json")
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.spider_v1_adapter import SpiderV1Adapter
from src.schema_graph import (
    SchemaGraph,
    SchemaGraphBuilder,
    NodeEnricher,
    build_semantic_edges,
    build_statistical_edges,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build Schema Knowledge Graph (all 3 layers)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Input / Output ────────────────────────────────────────────────────────
    p.add_argument(
        "--data_path", default="",
        help=(
            "Root directory of a Spider-format dataset "
            "(must contain tables.json).  Required unless --load_existing is given."
        ),
    )
    p.add_argument(
        "--db_dir", default="",
        help=(
            "Directory with SQLite DB files ({db_dir}/{db_id}/{db_id}.sqlite). "
            "Defaults to {data_path}/database.  Required for --value_overlap."
        ),
    )
    p.add_argument(
        "--output", default="data/schema_graphs/schema_graph.json",
        help="Path to write the final graph JSON.",
    )
    p.add_argument(
        "--load_existing", default="",
        help=(
            "Path to an already-built graph JSON.  If given, skip Layer 1/2 build "
            "and start from the loaded graph (useful for resuming or adding stats)."
        ),
    )
    p.add_argument(
        "--db_filter", default="",
        help="If set, process only this single database (by db_id).",
    )

    # ── Layer 2a: Lexical semantic edges ─────────────────────────────────────
    p.add_argument(
        "--lexical_threshold", type=float, default=0.4,
        help="Minimum Jaccard score for LEXICAL_SIMILAR edges.",
    )
    p.add_argument(
        "--no_same_table_pairs", action="store_true",
        help="Exclude column pairs within the same table from semantic edges.",
    )

    # ── LLM enrichment ────────────────────────────────────────────────────────
    enrich_group = p.add_argument_group("LLM enrichment (Layer 2b)")
    enrich_group.add_argument(
        "--enrich", action="store_true",
        help="Run LLM enrichment to generate descriptions and synonyms per node.",
    )
    enrich_group.add_argument(
        "--enrich_model", default="gpt-4o-mini",
        help="LLM model for node enrichment (OpenAI or Anthropic model name).",
    )
    enrich_group.add_argument(
        "--api_key", default="",
        help=(
            "API key for the enrichment LLM.  "
            "Falls back to OPENAI_API_KEY or ANTHROPIC_API_KEY env vars."
        ),
    )
    enrich_group.add_argument(
        "--enrich_batch_size", type=int, default=15,
        help="Number of nodes per LLM batch call.",
    )
    enrich_group.add_argument(
        "--enrich_sleep", type=float, default=0.5,
        help="Seconds to sleep between LLM batch calls (rate limit guard).",
    )
    enrich_group.add_argument(
        "--embedding_model", default="paraphrase-multilingual-mpnet-base-v2",
        help="SentenceTransformer model for embedding node descriptions.",
    )
    enrich_group.add_argument(
        "--chroma_persist_dir", default="./data/chroma_db",
        help=(
            "ChromaDB persistence directory for node embeddings. "
            "Defaults to the same directory as schema_chunks so one ChromaDB "
            "instance serves both collections. "
            "Set to empty string '' to fall back to in-node JSON storage."
        ),
    )
    enrich_group.add_argument(
        "--node_collection", default="schema_graph_nodes",
        help=(
            "ChromaDB collection name for graph node embeddings. "
            "Keep separate from 'schema_chunks' (Phase 2 chunk collection)."
        ),
    )
    enrich_group.add_argument(
        "--skip_existing_descriptions", action="store_true", default=True,
        help="Skip nodes that already have a description (default True).",
    )
    enrich_group.add_argument(
        "--force_regen", action="store_true",
        help="Re-generate all node descriptions even if they already exist.",
    )

    # ── Layer 3: Statistical edges ────────────────────────────────────────────
    stat_group = p.add_argument_group("Statistical edges (Layer 3)")
    stat_group.add_argument(
        "--statistical", action="store_true",
        help="Add SQL co-occurrence edges (CO_JOIN, CO_PREDICATE, CO_SELECT).",
    )
    stat_group.add_argument(
        "--value_overlap", action="store_true",
        help="Add VALUE_OVERLAP edges from live SQLite DISTINCT value comparison.",
    )
    stat_group.add_argument(
        "--join_threshold", type=float, default=0.10,
        help="Minimum Jaccard weight for CO_JOIN edges.",
    )
    stat_group.add_argument(
        "--predicate_threshold", type=float, default=0.10,
        help="Minimum Jaccard weight for CO_PREDICATE edges.",
    )
    stat_group.add_argument(
        "--select_threshold", type=float, default=0.10,
        help="Minimum Jaccard weight for CO_SELECT edges.",
    )
    stat_group.add_argument(
        "--value_overlap_threshold", type=float, default=0.20,
        help="Minimum Jaccard similarity of value sets for VALUE_OVERLAP edges.",
    )
    stat_group.add_argument(
        "--max_distinct_values", type=int, default=500,
        help="Skip TEXT columns with more than this many DISTINCT values.",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    t0 = time.time()

    # ── Resolve db_dir ────────────────────────────────────────────────────────
    db_dir = args.db_dir
    if not db_dir and args.data_path:
        db_dir = str(Path(args.data_path) / "database")

    # ── Step 0: Load or build graph ───────────────────────────────────────────
    if args.load_existing:
        logger.info("Loading existing graph from %s ...", args.load_existing)
        graph = SchemaGraph.load(args.load_existing)
        examples = []
        databases = []

        # If statistical edges are needed, still load examples for co-occurrence
        if (args.statistical or args.value_overlap) and args.data_path:
            logger.info("Loading examples from %s for statistical edges ...", args.data_path)
            adapter = SpiderV1Adapter()
            databases, examples = adapter.load(args.data_path)
    else:
        if not args.data_path:
            logger.error("Either --data_path or --load_existing must be provided.")
            sys.exit(1)
        if not Path(args.data_path).exists():
            logger.error("data_path not found: %s", args.data_path)
            sys.exit(1)

        logger.info("Loading dataset from %s ...", args.data_path)
        adapter = SpiderV1Adapter()
        databases, examples = adapter.load(args.data_path)
        logger.info("Loaded %d databases, %d examples", len(databases), len(examples))

        # Filter to a single DB if requested
        if args.db_filter:
            databases = [d for d in databases if d.db_id == args.db_filter]
            examples = [e for e in examples if e.db_id == args.db_filter]
            if not databases:
                logger.error("db_filter '%s' matched no databases.", args.db_filter)
                sys.exit(1)
            logger.info("Filtered to db_id='%s'", args.db_filter)

        # ── Step 1: Layer 1 — structural graph ────────────────────────────────
        logger.info("=== Step 1: Building structural graph (Layer 1) ===")
        builder = SchemaGraphBuilder()
        graph = builder.build_many(databases)
        _log_stats(graph, "After Layer 1")

        # ── Step 2a: Layer 2 lexical edges (no LLM needed) ────────────────────
        logger.info("=== Step 2a: Adding lexical semantic edges (Layer 2) ===")
        lex_edges = build_semantic_edges(
            graph,
            db_id=args.db_filter or None,
            lexical_threshold=args.lexical_threshold,
            embedding_threshold=2.0,      # >1.0 → disables embedding edges this pass
            synonym_min_overlap=999,       # effectively disabled this pass
            include_same_table_pairs=not args.no_same_table_pairs,
        )
        graph.add_edges(lex_edges)
        logger.info("Added %d lexical edges", len(lex_edges))

    # ── Step 3: LLM enrichment ────────────────────────────────────────────────
    if args.enrich:
        logger.info("=== Step 3: LLM node enrichment ===")
        api_key = (
            args.api_key
            or os.environ.get("OPENAI_API_KEY", "")
            or os.environ.get("ANTHROPIC_API_KEY", "")
        )
        if not api_key:
            logger.warning(
                "No API key provided.  Set --api_key or OPENAI_API_KEY / ANTHROPIC_API_KEY."
            )

        enricher = NodeEnricher(
            model=args.enrich_model,
            api_key=api_key or None,
        )

        result = enricher.enrich(
            graph,
            db_id=args.db_filter or None,
            batch_size=args.enrich_batch_size,
            skip_existing=args.skip_existing_descriptions and not args.force_regen,
            sleep_between_batches=args.enrich_sleep,
            force_regen=args.force_regen,
        )
        logger.info(
            "Enrichment: %d nodes enriched, %d failed, %d API calls",
            result.enriched, result.failed, result.api_calls,
        )

        # ── Step 4: Compute embeddings ─────────────────────────────────────────
        logger.info("=== Step 4: Computing node embeddings ===")
        chroma_dir = args.chroma_persist_dir or None   # empty str → None → in-node
        if chroma_dir:
            logger.info(
                "Node embeddings → ChromaDB collection '%s' at %s",
                args.node_collection, chroma_dir,
            )
        else:
            logger.info("Node embeddings → in-node JSON storage (no ChromaDB dir set)")
        n_embedded = enricher.embed_nodes(
            graph,
            db_id=args.db_filter or None,
            embedding_model=args.embedding_model,
            skip_existing=not args.force_regen,
            chroma_persist_dir=chroma_dir,
            chroma_collection_name=args.node_collection,
        )
        logger.info("Embedded %d nodes", n_embedded)

        # ── Step 5: Layer 2b — embedding + synonym edges ──────────────────────
        logger.info("=== Step 5: Adding embedding + synonym semantic edges (Layer 2b) ===")
        emb_syn_edges = build_semantic_edges(
            graph,
            db_id=args.db_filter or None,
            lexical_threshold=2.0,        # disables lexical this pass (already added)
            embedding_threshold=0.75,
            synonym_min_overlap=1,
            include_same_table_pairs=not args.no_same_table_pairs,
        )
        graph.add_edges(emb_syn_edges)
        logger.info("Added %d embedding/synonym edges", len(emb_syn_edges))
        _log_stats(graph, "After Layer 2b")

    # ── Step 6: Layer 3 — statistical edges ───────────────────────────────────
    needs_statistical = args.statistical or args.value_overlap
    if needs_statistical:
        logger.info("=== Step 6: Building statistical edges (Layer 3) ===")

        stat_edges = build_statistical_edges(
            graph,
            examples if args.statistical else [],
            db_dir=db_dir if args.value_overlap else "",
            db_id=args.db_filter or None,
            join_threshold=args.join_threshold,
            predicate_threshold=args.predicate_threshold,
            select_threshold=args.select_threshold,
            value_overlap_threshold=args.value_overlap_threshold,
            max_distinct_values=args.max_distinct_values,
        )
        graph.add_edges(stat_edges)
        logger.info("Added %d statistical edges", len(stat_edges))
        _log_stats(graph, "After Layer 3")

    # ── Save ──────────────────────────────────────────────────────────────────
    logger.info("=== Saving graph to %s ===", args.output)
    graph.save(args.output)

    elapsed = time.time() - t0
    logger.info("Done in %.1fs.  Final graph: %s", elapsed, graph)
    _print_summary(graph, args)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log_stats(graph: SchemaGraph, label: str) -> None:
    s = graph.stats()
    logger.info(
        "%s: %d nodes (%d tables, %d columns), %d edges, enriched=%d/%d",
        label,
        s["total_nodes"],
        s["nodes_by_type"].get("table", 0),
        s["nodes_by_type"].get("column", 0),
        s["total_edges"],
        s["enriched_columns"],
        s["total_columns"],
    )


def _print_summary(graph: SchemaGraph, args: argparse.Namespace) -> None:
    s = graph.stats()
    print("\n" + "=" * 58)
    print("Schema Graph Build Summary")
    print("=" * 58)
    print(f"  Output:            {args.output}")
    print(f"  Total nodes:       {s['total_nodes']}")
    print(f"    DB nodes:        {s['nodes_by_type'].get('database', 0)}")
    print(f"    Table nodes:     {s['nodes_by_type'].get('table', 0)}")
    print(f"    Column nodes:    {s['nodes_by_type'].get('column', 0)}")
    print(f"  Total edges:       {s['total_edges']}")
    for etype, cnt in sorted(s["edges_by_type"].items()):
        print(f"    {etype:<28} {cnt}")
    print(f"  Enriched columns:  {s['enriched_columns']}/{s['total_columns']}"
          f"  ({s['enrichment_ratio']:.0%})")
    print("=" * 58)


if __name__ == "__main__":
    main()
