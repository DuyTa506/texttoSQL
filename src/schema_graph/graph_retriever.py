"""
Graph Retriever — PPR-based schema linking via the Schema Knowledge Graph.

Drop-in replacement for ``HybridRetriever`` in ``run_pipeline.py``.

Provides the same public interface:
  retrieve(query, *, db_id=None)        → list[dict]
  retrieve_multi(queries, *, db_id=None) → list[dict]

Internally it runs Personalized PageRank over the ``SchemaGraph`` instead of
BM25 + ChromaDB vector search, and returns results in the same
``{id, content, chunk, score, source}`` dict format so the rest of the pipeline
(SchemaFilter, BidirectionalLinker) works without modification.

Architecture
------------
  Question
    ↓  embed via SentenceTransformer (same model as SchemaGraph embeddings)
  Cosine entry-point scoring against all COLUMN + TABLE nodes
    ↓  synonym boost (exact token match against node.synonyms)
  Top-M seed nodes
    ↓  Personalized PageRank over graph edges (alpha=0.7, max_hops=2)
  Final ranked node list
    ↓  Convert KGNode → chunk-format dict (with CREATE TABLE context)
  Output: list[dict]  compatible with SchemaFilter.filter_and_format()

The ``source`` field in returned dicts is "graph_ppr" so the pipeline can
distinguish graph-retrieved chunks from BM25/semantic chunks in logs.

Hybrid mode
-----------
When ``hybrid_retriever`` is also provided, results from both retrievers
are merged via RRF before returning.  This allows A/B ablation:
  - graph only:   GraphRetriever(graph, embedder)
  - hybrid:       GraphRetriever(graph, embedder, hybrid_retriever=hr)

Usage in run_pipeline.py (feature-flagged)
------------------------------------------
  if cfg.get("schema_graph", {}).get("enabled"):
      from src.schema_graph import SchemaGraph
      from src.schema_graph.graph_retriever import GraphRetriever
      graph    = SchemaGraph.load(cfg["schema_graph"]["graph_path"])
      embedder = SentenceTransformer(cfg["indexing"]["embedding_model"])
      retriever = GraphRetriever(graph, embedder)
  else:
      retriever = HybridRetriever(indexer, chunks, ...)
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Optional

import numpy as np

from src.schema_graph.graph_types import KGNode, NodeType
from src.schema_graph.edge_builders.structural_edges import tokenize_name

logger = logging.getLogger(__name__)


class GraphRetriever:
    """
    PPR-based schema retriever backed by a ``SchemaGraph``.

    Parameters
    ----------
    graph:
        A fully-built ``SchemaGraph`` (at minimum Layer 1; enriched for best quality).
    embedder:
        A ``SentenceTransformer`` instance already loaded.
        Must use the same model that was used to embed the graph nodes.
    top_m:
        Number of highest-cosine nodes used as PPR seeds per query.
    alpha:
        PPR damping factor.  Higher = PPR stays closer to seed nodes.
    ppr_max_iter:
        Maximum power-iteration steps passed to ``nx.pagerank`` (default 100).
    score_threshold:
        Nodes with PPR score below this are excluded.
    max_nodes:
        Hard cap on nodes returned per query.
    synonym_boost:
        Extra score added to nodes whose synonyms share tokens with the query.
    hybrid_retriever:
        Optional ``HybridRetriever`` instance.  When provided, its results are
        merged with PPR results via RRF before returning.
    hybrid_weight:
        RRF weight applied to graph-PPR scores vs. hybrid scores.
        1.0 = equal weight (standard RRF).
    """

    def __init__(
        self,
        graph,                      # SchemaGraph — no circular import at runtime
        embedder,                   # SentenceTransformer
        *,
        top_m: int = 5,
        alpha: float = 0.7,
        ppr_max_iter: int = 100,
        score_threshold: float = 0.05,
        max_nodes: int = 20,
        synonym_boost: float = 0.3,
        hybrid_retriever=None,      # HybridRetriever | None
        hybrid_weight: float = 1.0,
        rrf_k: int = 60,
    ) -> None:
        self.graph = graph
        self.embedder = embedder
        self.top_m = top_m
        self.alpha = alpha
        self.ppr_max_iter = ppr_max_iter
        self.score_threshold = score_threshold
        self.max_nodes = max_nodes
        self.synonym_boost = synonym_boost
        self.hybrid_retriever = hybrid_retriever
        self.hybrid_weight = hybrid_weight
        self.rrf_k = rrf_k

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        *,
        db_id: Optional[str] = None,
    ) -> list[dict]:
        """
        Retrieve relevant schema nodes for *query* via PPR.

        Returns
        -------
        list[dict]
            Dicts with keys: ``id``, ``content``, ``chunk``, ``score``, ``source``.
            ``chunk`` is None (graph nodes don't map to SchemaChunk objects).
            ``source`` is ``"graph_ppr"`` or ``"rrf"`` when hybrid mode is active.
        """
        # Embed the query
        q_emb = self._embed(query)
        synonym_tokens = _extract_synonym_tokens(query)

        # PPR retrieval from graph
        ppr_nodes = self.graph.retrieve(
            q_emb,
            db_id=db_id,
            top_m=self.top_m,
            alpha=self.alpha,
            max_iter=self.ppr_max_iter,
            score_threshold=self.score_threshold,
            max_nodes=self.max_nodes,
            synonym_tokens=synonym_tokens,
            synonym_boost=self.synonym_boost,
        )
        graph_results = self._nodes_to_dicts(ppr_nodes, db_id=db_id)

        # Hybrid merge if a secondary retriever is provided
        if self.hybrid_retriever is not None:
            hybrid_results = self.hybrid_retriever.retrieve(query, db_id=db_id)
            return self._rrf_merge(graph_results, hybrid_results)

        return graph_results

    def retrieve_multi(
        self,
        queries: list[str],
        *,
        db_id: Optional[str] = None,
    ) -> list[dict]:
        """
        Retrieve for multiple sub-queries and merge via RRF.

        Mirrors the ``HybridRetriever.retrieve_multi()`` interface used by
        the decompose strategy in ``run_pipeline.py``.

        Parameters
        ----------
        queries:
            Sub-queries (e.g. from ``QuestionDecomposer``).
        db_id:
            Restrict to one database.

        Returns
        -------
        list[dict]
            Merged, deduplicated results sorted by fused PPR score.
        """
        if not queries:
            return []
        if len(queries) == 1:
            return self.retrieve(queries[0], db_id=db_id)

        per_query: list[list[dict]] = [
            self.retrieve(q, db_id=db_id) for q in queries
        ]
        merged = self._rrf_merge(*per_query)

        # Deduplicate by node_id
        seen: set[str] = set()
        deduped: list[dict] = []
        for item in merged:
            if item["id"] not in seen:
                seen.add(item["id"])
                deduped.append(item)
        return deduped

    # ── Conversion ────────────────────────────────────────────────────────────

    def _nodes_to_dicts(
        self,
        nodes: list[KGNode],
        db_id: Optional[str] = None,
    ) -> list[dict]:
        """
        Convert a list of KGNodes to the chunk-format dicts expected by
        ``SchemaFilter.filter_and_format()``.

        The ``content`` field is constructed from node data so that
        SchemaFilter can still parse it (it inspects chunk metadata via
        ``chunk.chunk_type``, ``chunk.table_name``, ``chunk.column_name``).

        When no ``SchemaChunk`` object is available, we synthesise a minimal
        object via ``_SyntheticChunk`` so SchemaFilter doesn't break.
        """
        results: list[dict] = []
        for i, node in enumerate(nodes):
            content = _node_to_content(node)
            synthetic_chunk = _SyntheticChunk(
                db_id=node.db_id,
                chunk_type="table" if node.node_type == NodeType.TABLE else "column",
                table_name=node.table_name,
                column_name=node.column_name if node.node_type == NodeType.COLUMN else "",
                content=content,
            )
            results.append({
                "id": node.node_id,
                "content": content,
                "chunk": synthetic_chunk,
                "score": 1.0 / (i + 1),     # rank-based score (PPR already sorted)
                "source": "graph_ppr",
                "node": node,                # extra: downstream code can inspect
            })
        return results

    # ── RRF merge ─────────────────────────────────────────────────────────────

    def _rrf_merge(self, *result_lists: list[dict]) -> list[dict]:
        """Standard Reciprocal Rank Fusion over multiple result lists."""
        k = self.rrf_k
        rrf_scores: dict[str, float] = defaultdict(float)
        best_item: dict[str, dict] = {}

        for results in result_lists:
            for rank, item in enumerate(results, start=1):
                rid = item["id"]
                rrf_scores[rid] += 1.0 / (k + rank)
                if rid not in best_item:
                    best_item[rid] = item

        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
        merged = []
        for rid in sorted_ids:
            entry = {**best_item[rid]}
            entry["score"] = rrf_scores[rid]
            entry["source"] = "rrf"
            merged.append(entry)
        return merged

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        """Embed *text* using the stored SentenceTransformer."""
        emb = self.embedder.encode([text], convert_to_numpy=True)[0]
        return emb.astype(np.float32)


# ---------------------------------------------------------------------------
# Synthetic chunk — makes SchemaFilter work without real SchemaChunk objects
# ---------------------------------------------------------------------------


class _SyntheticChunk:
    """
    Minimal stand-in for ``SchemaChunk`` that satisfies SchemaFilter's
    attribute reads (``chunk_type``, ``table_name``, ``column_name``,
    ``db_id``, ``content``).

    SchemaFilter uses:
      chunk.chunk_type   → to group by table / column / fk / value
      chunk.table_name   → for grouping
      chunk.column_name  → for column display
      chunk.db_id        → for filtering
      chunk.content      → for the formatted output
    """

    __slots__ = ("db_id", "chunk_type", "table_name", "column_name", "content")

    def __init__(
        self,
        db_id: str,
        chunk_type: str,
        table_name: str,
        column_name: str,
        content: str,
    ) -> None:
        self.db_id = db_id
        self.chunk_type = chunk_type
        self.table_name = table_name
        self.column_name = column_name
        self.content = content

    def __repr__(self) -> str:
        return f"_SyntheticChunk({self.chunk_type}: {self.db_id}.{self.table_name})"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _node_to_content(node: KGNode) -> str:
    """
    Build a content string from a KGNode.

    Format mirrors what ``SchemaChunker`` produces so downstream code
    that pattern-matches on content strings still works.
    """
    if node.node_type == NodeType.TABLE:
        col_count = ""  # we don't have column count without graph traversal
        desc = f" — {node.description}" if node.description else ""
        return f"Table: {node.table_name}{desc}"

    elif node.node_type == NodeType.COLUMN:
        parts = [f"Column: {node.table_name}.{node.column_name}"]
        if node.dtype:
            parts.append(f"({node.dtype})")
        if node.is_pk:
            parts.append("[PK]")
        if node.is_fk:
            parts.append("[FK]")
        if node.description:
            parts.append(f"— {node.description}")
        if node.sample_values:
            sample = ", ".join(f"'{v}'" for v in node.sample_values[:3])
            parts.append(f"[e.g. {sample}]")
        return " ".join(parts)

    else:
        return node.node_id


def _extract_synonym_tokens(query: str) -> set[str]:
    """
    Extract lowercase tokens from the query for synonym matching.
    Filters out very short tokens and punctuation.
    """
    tokens = set()
    for tok in re.split(r"\W+", query.lower()):
        if len(tok) >= 3:
            tokens.add(tok)
    return tokens
