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
    ↓  synonym boost (exact token match against node.synonyms, NL stopwords removed)
  Top-M seed nodes
    ↓  Personalized PageRank over per-DB subgraph (alpha=0.7, max_hops=2)
  Real PPR scores preserved
    ↓  FK bridge table injection (missing JOIN hubs added with bridge score)
    ↓  Score gap pruning (elbow cut removes low-confidence tail)
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
      from src.retrieval.graph_retriever import GraphRetriever
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

import numpy as np

from src.schema_graph.graph_types import EdgeType, KGNode, NodeType
from src.schema_graph.edge_builders.structural_edges import tokenize_name
from src.retrieval.base_retriever import BaseRetriever

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fix 4: NL stopword set for synonym token extraction
# ---------------------------------------------------------------------------

_NL_STOPWORDS: frozenset[str] = frozenset({
    "what", "which", "who", "where", "when", "how", "why",
    "the", "are", "is", "was", "were", "has", "have", "had",
    "for", "and", "not", "but", "that", "this", "with", "from",
    "all", "any", "can", "list", "give", "show", "find", "get",
    "its", "into", "been", "will", "than", "you", "they", "their",
    "each", "also", "use", "per", "out", "more", "most",
})


class GraphRetriever(BaseRetriever):
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
        Hard cap on nodes returned per query (adaptive per DB size by default).
    synonym_boost:
        Extra score added to nodes whose synonyms share tokens with the query.
    hybrid_retriever:
        Optional ``HybridRetriever`` instance.  When provided, its results are
        merged with PPR results via RRF before returning.
    hybrid_weight:
        RRF weight applied to graph-PPR scores vs. hybrid scores.
        Values > 1.0 amplify graph results in the fused ranking.
        1.0 = equal weight (standard RRF).
    rrf_k:
        RRF smoothing constant (default 60).
    score_gap_ratio:
        Elbow-cut threshold: if consecutive PPR scores have ratio > this value,
        nodes below the gap are pruned.  Set to ``inf`` to disable.
    use_fk_bridge:
        If True, inject FK-connected bridge tables missing from the PPR result
        to improve recall on multi-join queries.
    """

    def __init__(
        self,
        graph,                      # SchemaGraph — no circular import at runtime
        embedder,                   # SentenceTransformer
        *,
        top_m: int = 5,
        alpha: float = 0.7,
        max_hops: int = 2,
        ppr_max_iter: int = 100,
        score_threshold: float = 0.05,
        max_nodes: int = 20,
        synonym_boost: float = 0.3,
        hybrid_retriever=None,      # HybridRetriever | None
        hybrid_weight: float = 1.0,
        rrf_k: int = 60,
        score_gap_ratio: float = 3.0,
        use_fk_bridge: bool = True,
    ) -> None:
        self.graph = graph
        self.embedder = embedder
        self.top_m = top_m
        self.alpha = alpha
        self.max_hops = max_hops
        self.ppr_max_iter = ppr_max_iter
        self.score_threshold = score_threshold
        self.max_nodes = max_nodes
        self.synonym_boost = synonym_boost
        self.hybrid_retriever = hybrid_retriever
        self.hybrid_weight = hybrid_weight
        self.rrf_k = rrf_k
        self.score_gap_ratio = score_gap_ratio
        self.use_fk_bridge = use_fk_bridge

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        *,
        db_id: str | None = None,
        value_matches: list | None = None,
    ) -> list[dict]:
        """
        Retrieve relevant schema nodes for *query* via PPR.

        Parameters
        ----------
        query:
            Natural language question.
        db_id:
            Restrict to one database.
        value_matches:
            Optional list of ``ValueMatch`` objects from ValueScanner (v3).
            When provided, matched columns/tables get boosted PPR seeds.

        Returns
        -------
        list[dict]
            Dicts with keys: ``id``, ``content``, ``chunk``, ``score``, ``source``.
            ``chunk`` is None (graph nodes don't map to SchemaChunk objects).
            ``source`` is ``"graph_ppr"`` or ``"rrf"`` when hybrid mode is active.
        """
        # Embed the query
        q_emb = self._embed(query)
        synonym_tokens = _extract_synonym_tokens(query)  # Fix 4: stopwords filtered

        # Fix 2: Adaptive max_nodes — cap based on DB table count so small DBs
        # don't return 20 nodes when they only have 5 tables.
        adaptive_max = self._adaptive_max_nodes(db_id)

        # v3: Convert ValueMatch list → {node_id → boost} dict
        value_boost: dict[str, float] | None = None
        if value_matches:
            value_boost = {}
            for m in value_matches:
                col_nid = f"{db_id}.{m.table_name}.{m.column_name}" if db_id else f"{m.table_name}.{m.column_name}"
                value_boost[col_nid] = max(value_boost.get(col_nid, 0), m.score * 0.5)
                # Also boost the parent table node
                tbl_nid = f"{db_id}.{m.table_name}" if db_id else m.table_name
                value_boost[tbl_nid] = max(value_boost.get(tbl_nid, 0), m.score * 0.3)

        # PPR retrieval from graph — now returns list[tuple[KGNode, float]] (Fix 1)
        ppr_nodes_with_scores: list[tuple[KGNode, float]] = self.graph.retrieve(
            q_emb,
            db_id=db_id,
            top_m=self.top_m,
            alpha=self.alpha,
            max_iter=self.max_hops * 50,
            score_threshold=self.score_threshold,
            max_nodes=adaptive_max,
            synonym_tokens=synonym_tokens,
            synonym_boost=self.synonym_boost,
            value_matched_nodes=value_boost,
        )

        # Fix 6: FK bridge table injection — must run BEFORE gap pruning so
        # bridge nodes with lower scores don't get pruned immediately.
        if self.use_fk_bridge and db_id is not None:
            ppr_nodes_with_scores = self._add_fk_bridge_tables(
                ppr_nodes_with_scores, db_id
            )

        # Fix 3: Score gap pruning — remove low-confidence tail nodes.
        # v3: Pass db_id for per-DB adaptive gap ratio.
        ppr_nodes_with_scores = self._apply_score_gap_pruning(
            ppr_nodes_with_scores, db_id=db_id
        )

        # Fix 1: Pass real scores to _nodes_to_dicts
        graph_results = self._nodes_to_dicts(ppr_nodes_with_scores, db_id=db_id)

        # Hybrid merge if a secondary retriever is provided
        if self.hybrid_retriever is not None:
            hybrid_results = self.hybrid_retriever.retrieve(query, db_id=db_id)
            # Fix 8: _rrf_merge now distinguishes graph vs. other lists and applies
            # self.hybrid_weight to graph results.
            return self._rrf_merge(graph_results, hybrid_results)

        return graph_results

    def retrieve_multi(
        self,
        queries: list[str],
        *,
        db_id: str | None = None,
        value_matches: list | None = None,
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
        value_matches:
            Optional list of ``ValueMatch`` objects (v3), passed to each
            sub-query retrieval.

        Returns
        -------
        list[dict]
            Merged, deduplicated results sorted by fused PPR score.
        """
        if not queries:
            return []
        if len(queries) == 1:
            return self.retrieve(queries[0], db_id=db_id, value_matches=value_matches)

        per_query: list[list[dict]] = [
            self.retrieve(q, db_id=db_id, value_matches=value_matches) for q in queries
        ]
        # Fix 8: For multi-query fusion all lists are equal-weight (no graph vs.
        # hybrid distinction), so use the first list as "graph" and the rest as
        # "others" — all receive equal RRF weight (hybrid_weight=1.0 effective).
        merged = self._rrf_merge(per_query[0], *per_query[1:])

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
        nodes_with_scores: list[tuple[KGNode, float]],  # Fix 1: accept tuples
        db_id: str | None = None,
    ) -> list[dict]:
        """
        Convert a list of (KGNode, score) tuples to the chunk-format dicts
        expected by ``SchemaFilter.filter_and_format()``.

        The ``content`` field is constructed from node data so that
        SchemaFilter can still parse it (it inspects chunk metadata via
        ``chunk.chunk_type``, ``chunk.table_name``, ``chunk.column_name``).

        When no ``SchemaChunk`` object is available, we synthesise a minimal
        object via ``_SyntheticChunk`` so SchemaFilter doesn't break.
        """
        results: list[dict] = []
        # Fix 1: Use real PPR score instead of 1/(i+1) rank-based scoring
        for node, score in nodes_with_scores:
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
                "score": score,          # Fix 1: real PPR score
                "source": "graph_ppr",
                "node": node,            # extra: downstream code can inspect
            })
        return results

    # ── RRF merge ─────────────────────────────────────────────────────────────

    def _rrf_merge(
        self,
        graph_results: list[dict],
        *other_lists: list[dict],
    ) -> list[dict]:
        """
        Reciprocal Rank Fusion over multiple result lists.

        Fix 8: ``graph_results`` is weighted by ``self.hybrid_weight`` in the
        RRF accumulator so the graph signal can be up- or down-weighted relative
        to the hybrid retriever's results (all ``other_lists`` use weight 1.0).

        Parameters
        ----------
        graph_results:
            PPR results from this retriever (weighted by ``hybrid_weight``).
        *other_lists:
            Additional result lists (e.g. HybridRetriever output), each with
            equal weight 1.0 in the fusion.
        """
        k = self.rrf_k
        rrf_scores: dict[str, float] = defaultdict(float)
        best_item: dict[str, dict] = {}

        # Graph results — weighted by hybrid_weight (Fix 8)
        for rank, item in enumerate(graph_results, start=1):
            rid = item["id"]
            rrf_scores[rid] += self.hybrid_weight / (k + rank)
            if rid not in best_item:
                best_item[rid] = item

        # Other result lists — standard RRF weight 1.0
        for results in other_lists:
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

    # ── Fix 2: Adaptive max_nodes ──────────────────────────────────────────────

    def _adaptive_max_nodes(self, db_id: str | None) -> int:
        """
        Compute an adaptive hard cap on returned nodes based on DB size.

        Small databases (e.g. 5 tables) should not return 20 nodes — that
        floods the prompt with irrelevant columns and collapses precision.
        The cap is ``min(max_nodes, max(6, db_table_count))``, ensuring we
        always return at least 6 nodes even for the smallest schemas.

        For multi-DB or unknown db_id, fall back to the configured max_nodes.
        """
        if db_id is None:
            return self.max_nodes
        db_table_count = sum(
            1 for n in self.graph.nodes_for_db(db_id)
            if n.node_type == NodeType.TABLE
        )
        if db_table_count == 0:
            return self.max_nodes
        return min(self.max_nodes, max(6, db_table_count))

    # ── Fix 6: FK bridge table recall boost ───────────────────────────────────

    def _add_fk_bridge_tables(
        self,
        nodes_with_scores: list[tuple[KGNode, float]],
        db_id: str,
    ) -> list[tuple[KGNode, float]]:
        """
        Inject FK-connected bridge tables that are missing from the PPR result.

        Analysis showed 130/192 Graph recall failures are due to JOIN hub tables
        (e.g. ``races``, ``member``, ``patient``) that PPR misses because it
        finds FK columns on both sides of a JOIN but not the intermediary table.

        For every FK column in the result, this method checks whether the
        FK-target table is already retrieved.  If not, the table node and its
        columns are added with a bridge score derived from the minimum result score.

        Parameters
        ----------
        nodes_with_scores:
            Current PPR result list (sorted by score desc).
        db_id:
            The database being queried (only same-DB bridges are added).

        Returns
        -------
        list[tuple[KGNode, float]]
            Extended list with bridge tables appended.
        """
        all_nodes = self.graph.nodes
        fk_edge_types = {
            EdgeType.FOREIGN_KEY, EdgeType.FOREIGN_KEY_REV,
            EdgeType.INFERRED_FK, EdgeType.INFERRED_FK_REV,  # v3: also follow inferred FKs
        }

        retrieved_tables: set[str] = {
            n.table_name for n, _ in nodes_with_scores
            if n.node_type == NodeType.TABLE
        }
        # Also count tables that have at least one column in the result
        for n, _ in nodes_with_scores:
            if n.node_type == NodeType.COLUMN:
                retrieved_tables.add(n.table_name)

        min_score = min((s for _, s in nodes_with_scores), default=0.0)
        bridge_score = min_score * 0.5

        bridges: dict[str, KGNode] = {}
        for node, _ in nodes_with_scores:
            if node.node_type != NodeType.COLUMN or not node.is_fk:
                continue
            for edge in self.graph.neighbors(node.node_id, edge_types=fk_edge_types):
                dst = all_nodes.get(edge.dst_id)
                if dst is None or dst.db_id != db_id:
                    continue
                target_table = dst.table_name
                if target_table not in retrieved_tables and target_table not in bridges:
                    table_nid = f"{db_id}.{target_table}"
                    tbl_node = all_nodes.get(table_nid)
                    if tbl_node is not None:
                        bridges[target_table] = tbl_node

        if not bridges:
            return nodes_with_scores

        result = list(nodes_with_scores)
        col_bridge_score = bridge_score * 0.8
        for tbl_node in bridges.values():
            result.append((tbl_node, bridge_score))
            for col_node in self.graph.column_nodes_for_table(db_id, tbl_node.table_name):
                result.append((col_node, col_bridge_score))

        logger.debug(
            "FK bridge: added %d bridge table(s) for db_id=%s: %s",
            len(bridges),
            db_id,
            list(bridges.keys()),
        )
        return result

    # ── Fix 3: Score gap pruning ───────────────────────────────────────────────

    def _apply_score_gap_pruning(
        self,
        nodes_with_scores: list[tuple[KGNode, float]],
        min_keep: int = 2,
        db_id: str | None = None,
    ) -> list[tuple[KGNode, float]]:
        """
        Remove low-confidence tail nodes via an elbow-cut heuristic.

        If two consecutive nodes have a score ratio (prev/curr) exceeding
        ``score_gap_ratio``, all nodes from the lower one onward are pruned.
        At least ``min_keep`` nodes are always retained.

        v3: Per-DB adaptive gap ratio — large DBs (≥10 tables) use a looser
        threshold to avoid pruning needed tables; small DBs (≤4 tables) use a
        tighter threshold to suppress noise.

        Example: scores [0.15, 0.14, 0.12, 0.06] with gap_ratio=3.0
          → 0.12 / 0.06 = 2.0  (no gap)
          → 0.15 / 0.14 = 1.07 (no gap)
          → all kept  (ratio never exceeds 3.0)

        Example: scores [0.30, 0.05, 0.04] with gap_ratio=3.0
          → 0.30 / 0.05 = 6.0  (gap!)  → prune from index 1 onward
          → keeps only [0.30]  (capped at min_keep=2: keeps [0.30, 0.05])
        """
        # v3: Adaptive gap ratio based on DB table count
        gap_ratio = self.score_gap_ratio
        if db_id is not None:
            table_count = sum(
                1 for n in self.graph.nodes_for_db(db_id)
                if n.node_type == NodeType.TABLE
            )
            if table_count >= 10:
                gap_ratio = max(self.score_gap_ratio, 4.5)  # looser for large DBs
            elif table_count <= 4:
                gap_ratio = min(self.score_gap_ratio, 2.0)  # tighter for small DBs
        if gap_ratio <= 0 or len(nodes_with_scores) <= min_keep:
            return nodes_with_scores

        pruned = [nodes_with_scores[0]]
        for i in range(1, len(nodes_with_scores)):
            prev_score = nodes_with_scores[i - 1][1]
            curr_score = nodes_with_scores[i][1]
            if prev_score > 0 and (prev_score / max(curr_score, 1e-9)) > gap_ratio:
                # Gap found — stop here, but ensure min_keep is satisfied
                if len(pruned) < min_keep:
                    pruned.extend(
                        nodes_with_scores[len(pruned):min_keep]
                    )
                break
            pruned.append(nodes_with_scores[i])

        return pruned

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

    Fix 4: Filters out NL stopwords (articles, auxiliaries, question words)
    to reduce spurious synonym boosts from generic English words like "what",
    "the", "are" that appear in many queries but carry no schema signal.
    """
    tokens: set[str] = set()
    for tok in re.split(r"\W+", query.lower()):
        if len(tok) >= 3 and tok not in _NL_STOPWORDS:
            tokens.add(tok)
    return tokens
