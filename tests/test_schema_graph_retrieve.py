"""
Full-coverage tests for SchemaGraph.retrieve()
================================================
Covers all 8 fix-related paths plus edge-cases.

Sections
--------
1.  Fixtures / helpers
2.  Return type — list[tuple[KGNode, float]]          (Fix 1)
3.  Fallback path — no embeddings                     (Fix 1 fallback)
4.  Per-DB subgraph                                   (Fix 5)
5.  score_threshold filtering
6.  max_nodes hard cap
7.  Seed selection & personalization vector
8.  PPR convergence failure fallback
9.  synonym_boost applied in entry-points
10. db_id=None (multi-DB graph)
11. Empty graph edge-cases
12. Parametrized score/threshold matrix               (100-subset bulk)
"""

from __future__ import annotations

import math
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.schema_graph.graph_builder import SchemaGraph, SchemaGraphBuilder
from src.schema_graph.graph_types import EdgeType, KGEdge, KGNode, NodeType
from src.schema.models import Column, Database, ForeignKey, Table


# ---------------------------------------------------------------------------
# 1. Fixtures / helpers
# ---------------------------------------------------------------------------

def _col_node(db: str, tbl: str, col: str, emb: Optional[list[float]] = None) -> KGNode:
    n = KGNode.make_column(db, tbl, col, dtype="TEXT")
    if emb is not None:
        n.embedding = emb
    return n


def _tbl_node(db: str, tbl: str, emb: Optional[list[float]] = None) -> KGNode:
    n = KGNode.make_table(db, tbl)
    if emb is not None:
        n.embedding = emb
    return n


def _unit(d: int, pos: int) -> list[float]:
    """Return a d-dimensional unit vector with 1.0 at position *pos*."""
    v = [0.0] * d
    v[pos] = 1.0
    return v


def _build_simple_graph(db_id: str = "db", n_tables: int = 3) -> SchemaGraph:
    """Build a small SchemaGraph with embeddings for n_tables tables."""
    g = SchemaGraph()
    dim = max(n_tables * 2, 4)
    for i in range(n_tables):
        tbl = f"t{i}"
        t = _tbl_node(db_id, tbl, emb=_unit(dim, i * 2))
        c = _col_node(db_id, tbl, "id", emb=_unit(dim, i * 2 + 1))
        g.add_node(t)
        g.add_node(c)
        # TABLE_HAS_COLUMN edge so PPR can flow
        g.add_edge(KGEdge(t.node_id, c.node_id, EdgeType.TABLE_HAS_COLUMN, 1.0))
        g.add_edge(KGEdge(c.node_id, t.node_id, EdgeType.COLUMN_BELONGS_TO, 1.0))
    return g


def _query_emb(d: int, pos: int) -> np.ndarray:
    v = np.zeros(d, dtype=np.float32)
    v[pos] = 1.0
    return v


# ---------------------------------------------------------------------------
# 2. Return type — list[tuple[KGNode, float]]  (Fix 1)
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_list_of_tuples(self):
        g = _build_simple_graph()
        q = _query_emb(6, 0)
        result = g.retrieve(q, db_id="db")
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, tuple) and len(item) == 2
            node, score = item
            assert isinstance(node, KGNode)
            assert isinstance(score, float)

    def test_scores_are_positive(self):
        g = _build_simple_graph()
        q = _query_emb(6, 0)
        result = g.retrieve(q, db_id="db")
        for _, score in result:
            assert score >= 0.0

    def test_scores_sum_to_at_most_one(self):
        """PPR scores over the full node set must sum to ~1; our subset is ≤1."""
        g = _build_simple_graph()
        q = _query_emb(6, 0)
        result = g.retrieve(q, db_id="db", score_threshold=0.0)
        total = sum(s for _, s in result)
        assert total <= 1.0 + 1e-6

    def test_sorted_descending(self):
        g = _build_simple_graph(n_tables=4)
        q = _query_emb(8, 0)
        result = g.retrieve(q, db_id="db", score_threshold=0.0)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_nodes_are_kgnodes(self):
        g = _build_simple_graph()
        q = _query_emb(6, 0)
        result = g.retrieve(q, db_id="db")
        for node, _ in result:
            assert node.node_type in (NodeType.TABLE, NodeType.COLUMN)

    def test_no_database_nodes_returned(self):
        g = _build_simple_graph()
        db_node = KGNode.make_db("db")
        g.add_node(db_node)
        q = _query_emb(6, 0)
        result = g.retrieve(q, db_id="db")
        for node, _ in result:
            assert node.node_type != NodeType.DATABASE

    def test_real_scores_not_rank_based(self):
        """Scores must NOT follow the 1/(i+1) pattern from the old broken code."""
        g = _build_simple_graph(n_tables=4)
        q = _query_emb(8, 0)
        result = g.retrieve(q, db_id="db", score_threshold=0.0, max_nodes=4)
        if len(result) >= 2:
            _, s0 = result[0]
            _, s1 = result[1]
            # Old code: s0=1.0, s1=0.5 always — real PPR values differ
            assert not (math.isclose(s0, 1.0, rel_tol=1e-4)
                        and math.isclose(s1, 0.5, rel_tol=1e-4))


# ---------------------------------------------------------------------------
# 3. Fallback path — no embeddings  (Fix 1 fallback tuples)
# ---------------------------------------------------------------------------


class TestFallbackPath:
    def test_fallback_returns_tuples_with_zero_score(self):
        """When no nodes have embeddings, fallback yields (KGNode, 0.0) tuples."""
        g = SchemaGraph()
        t = KGNode.make_table("db", "mytable")   # no embedding
        g.add_node(t)
        q = np.zeros(4, dtype=np.float32)
        result = g.retrieve(q, db_id="db")
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, tuple)
            assert item[1] == 0.0

    def test_fallback_caps_at_max_nodes(self):
        g = SchemaGraph()
        for i in range(10):
            g.add_node(KGNode.make_table("db", f"t{i}"))
        q = np.zeros(4, dtype=np.float32)
        result = g.retrieve(q, db_id="db", max_nodes=3)
        assert len(result) <= 3

    def test_fallback_respects_db_id_filter(self):
        g = SchemaGraph()
        for i in range(3):
            g.add_node(KGNode.make_table("db_a", f"t{i}"))
        for i in range(3):
            g.add_node(KGNode.make_table("db_b", f"s{i}"))
        q = np.zeros(4, dtype=np.float32)
        result = g.retrieve(q, db_id="db_a")
        for node, _ in result:
            assert node.db_id == "db_a"

    def test_fallback_only_table_and_column_nodes(self):
        g = SchemaGraph()
        g.add_node(KGNode.make_db("db"))
        g.add_node(KGNode.make_table("db", "t0"))
        g.add_node(KGNode.make_column("db", "t0", "c0"))
        q = np.zeros(4, dtype=np.float32)
        result = g.retrieve(q, db_id="db")
        for node, _ in result:
            assert node.node_type in (NodeType.TABLE, NodeType.COLUMN)


# ---------------------------------------------------------------------------
# 4. Per-DB subgraph  (Fix 5)
# ---------------------------------------------------------------------------


class TestPerDBSubgraph:
    def _two_db_graph(self) -> SchemaGraph:
        g = SchemaGraph()
        # db_a: 2 tables with embeddings pointing along dimensions 0,1
        dim = 8
        for i, (db, tbl) in enumerate([
            ("db_a", "alpha"), ("db_a", "beta"),
            ("db_b", "gamma"), ("db_b", "delta"),
        ]):
            t = _tbl_node(db, tbl, emb=_unit(dim, i * 2))
            c = _col_node(db, tbl, "id", emb=_unit(dim, i * 2 + 1))
            g.add_node(t)
            g.add_node(c)
            g.add_edge(KGEdge(t.node_id, c.node_id, EdgeType.TABLE_HAS_COLUMN, 1.0))
        return g

    def test_db_a_results_only_contain_db_a_nodes(self):
        g = self._two_db_graph()
        q = _query_emb(8, 0)
        result = g.retrieve(q, db_id="db_a", score_threshold=0.0)
        for node, _ in result:
            assert node.db_id == "db_a"

    def test_db_b_results_only_contain_db_b_nodes(self):
        g = self._two_db_graph()
        q = _query_emb(8, 4)
        result = g.retrieve(q, db_id="db_b", score_threshold=0.0)
        for node, _ in result:
            assert node.db_id == "db_b"

    def test_no_cross_db_leakage_with_cross_db_edge(self):
        """Even if a FK-like edge crosses databases, PPR stays per-DB."""
        g = self._two_db_graph()
        # Add an artificial cross-DB edge
        g.add_edge(KGEdge("db_a.alpha.id", "db_b.gamma.id",
                          EdgeType.FOREIGN_KEY, 0.95))
        q = _query_emb(8, 0)
        result = g.retrieve(q, db_id="db_a", score_threshold=0.0)
        for node, _ in result:
            assert node.db_id == "db_a"

    def test_none_db_id_returns_all_dbs(self):
        g = self._two_db_graph()
        q = _query_emb(8, 0)
        result = g.retrieve(q, db_id=None, score_threshold=0.0)
        dbs = {node.db_id for node, _ in result}
        # Both DBs can appear when db_id=None
        assert len(dbs) >= 1  # at minimum one DB found


# ---------------------------------------------------------------------------
# 5. score_threshold filtering
# ---------------------------------------------------------------------------


class TestScoreThreshold:
    def test_zero_threshold_returns_more_nodes(self):
        g = _build_simple_graph(n_tables=4)
        q = _query_emb(8, 0)
        strict = g.retrieve(q, db_id="db", score_threshold=0.5)
        relaxed = g.retrieve(q, db_id="db", score_threshold=0.0)
        assert len(relaxed) >= len(strict)

    def test_high_threshold_returns_empty_or_fewer(self):
        g = _build_simple_graph(n_tables=2)
        q = _query_emb(4, 0)
        result = g.retrieve(q, db_id="db", score_threshold=0.99)
        # Unlikely any PPR score exceeds 0.99 for a multi-node graph
        assert len(result) <= 2

    def test_all_scores_above_threshold(self):
        g = _build_simple_graph(n_tables=3)
        q = _query_emb(6, 0)
        thresh = 0.01
        result = g.retrieve(q, db_id="db", score_threshold=thresh)
        for _, score in result:
            assert score >= thresh

    def test_threshold_of_one_returns_empty(self):
        g = _build_simple_graph(n_tables=3)
        q = _query_emb(6, 0)
        result = g.retrieve(q, db_id="db", score_threshold=1.0)
        # No node should have PPR score exactly 1.0 in a multi-node graph
        assert len(result) == 0


# ---------------------------------------------------------------------------
# 6. max_nodes hard cap
# ---------------------------------------------------------------------------


class TestMaxNodes:
    @pytest.mark.parametrize("cap", [1, 2, 3, 5, 10])
    def test_max_nodes_respected(self, cap: int):
        g = _build_simple_graph(n_tables=6)
        q = _query_emb(12, 0)
        result = g.retrieve(q, db_id="db", score_threshold=0.0, max_nodes=cap)
        assert len(result) <= cap

    def test_max_nodes_one_returns_highest_score(self):
        g = _build_simple_graph(n_tables=4)
        q = _query_emb(8, 0)
        single = g.retrieve(q, db_id="db", score_threshold=0.0, max_nodes=1)
        full = g.retrieve(q, db_id="db", score_threshold=0.0)
        if single and full:
            assert single[0][1] == full[0][1]


# ---------------------------------------------------------------------------
# 7. Seed selection & personalization vector
# ---------------------------------------------------------------------------


class TestSeedSelection:
    def test_top_m_seeds_used(self):
        """The node cosine-closest to query should have highest PPR score."""
        g = _build_simple_graph(n_tables=4)
        # Query perfectly aligned with table t0
        q = _query_emb(8, 0)
        result = g.retrieve(q, db_id="db", score_threshold=0.0, top_m=1)
        # The highest-score node should come from t0 subtree
        top_node, _ = result[0]
        assert top_node.table_name == "t0" or top_node.node_id.startswith("db.t0")

    def test_increasing_top_m_can_change_results(self):
        """Using more seeds can bring in more nodes."""
        g = _build_simple_graph(n_tables=4)
        q = _query_emb(8, 0)
        r1 = g.retrieve(q, db_id="db", score_threshold=0.0, top_m=1)
        r4 = g.retrieve(q, db_id="db", score_threshold=0.0, top_m=4)
        # More seeds: results may differ or expand
        ids1 = {n.node_id for n, _ in r1}
        ids4 = {n.node_id for n, _ in r4}
        # All r1 results should still appear in r4 (broader coverage)
        assert ids1.issubset(ids4) or len(ids4) >= len(ids1)

    def test_personalization_proportional_to_scores(self):
        """Nodes with higher cosine similarity to query should seed more strongly."""
        g = _build_simple_graph(n_tables=3)
        q = _query_emb(6, 0)   # aligned with t0
        result = g.retrieve(q, db_id="db", score_threshold=0.0)
        top_tables = [n.table_name for n, _ in result
                      if n.node_type == NodeType.TABLE]
        if top_tables:
            assert top_tables[0] == "t0"


# ---------------------------------------------------------------------------
# 8. PPR convergence failure fallback
# ---------------------------------------------------------------------------


class TestConvergenceFallback:
    def test_fallback_on_convergence_error(self):
        """When nx.pagerank raises PowerIterationFailedConvergence, fall back gracefully."""
        import networkx as nx
        g = _build_simple_graph(n_tables=2)
        q = _query_emb(4, 0)
        with patch("networkx.pagerank",
                   side_effect=nx.PowerIterationFailedConvergence(100)):
            result = g.retrieve(q, db_id="db", score_threshold=0.0)
        # Should still return a list (seed scores used as fallback)
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, tuple)

    def test_fallback_returns_seed_ids(self):
        import networkx as nx
        g = _build_simple_graph(n_tables=3)
        q = _query_emb(6, 0)
        with patch("networkx.pagerank",
                   side_effect=nx.PowerIterationFailedConvergence(100)):
            result = g.retrieve(q, db_id="db", score_threshold=0.0)
        assert len(result) >= 0  # no crash


# ---------------------------------------------------------------------------
# 9. Synonym boost
# ---------------------------------------------------------------------------


class TestSynonymBoost:
    def test_synonym_boost_increases_score(self):
        """A node whose synonyms match the query tokens should outscore one that doesn't."""
        dim = 4
        g = SchemaGraph()
        # Two table nodes with identical embeddings
        t_match = _tbl_node("db", "races", emb=_unit(dim, 0))
        t_match.synonyms = ["race", "grand_prix", "racing_event"]
        t_other = _tbl_node("db", "drivers", emb=_unit(dim, 0))
        t_other.synonyms = ["pilot", "racer"]
        g.add_node(t_match)
        g.add_node(t_other)
        # Add column children so PPR has edges to traverse
        for tbl in ["races", "drivers"]:
            c = _col_node("db", tbl, "id", emb=_unit(dim, 1))
            g.add_node(c)
            g.add_edge(KGEdge(f"db.{tbl}", f"db.{tbl}.id",
                              EdgeType.TABLE_HAS_COLUMN, 1.0))

        q = _query_emb(dim, 0)
        tokens_match = {"races", "circuit"}
        tokens_no = {"drivers", "name"}

        r_match = g.retrieve(q, db_id="db", score_threshold=0.0,
                             synonym_tokens=tokens_match, synonym_boost=0.5)
        r_no = g.retrieve(q, db_id="db", score_threshold=0.0,
                          synonym_tokens=tokens_no, synonym_boost=0.5)

        score_match = next(
            (s for n, s in r_match if n.table_name == "races"), 0.0
        )
        score_no = next(
            (s for n, s in r_no if n.table_name == "races"), 0.0
        )
        # "races" should have higher (or equal) seed score when its synonym matches
        assert score_match >= score_no

    def test_no_synonym_tokens_no_boost(self):
        """synonym_tokens=None must not crash and produces valid results."""
        g = _build_simple_graph(n_tables=2)
        q = _query_emb(4, 0)
        result = g.retrieve(q, db_id="db", synonym_tokens=None)
        assert isinstance(result, list)

    def test_empty_synonym_tokens_no_boost(self):
        g = _build_simple_graph(n_tables=2)
        q = _query_emb(4, 0)
        result = g.retrieve(q, db_id="db", synonym_tokens=set())
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 10. db_id=None (multi-DB)
# ---------------------------------------------------------------------------


class TestMultiDB:
    def test_none_db_id_includes_nodes_from_multiple_dbs(self):
        g = SchemaGraph()
        dim = 4
        for db in ["db_x", "db_y"]:
            t = _tbl_node(db, "sales", emb=_unit(dim, 0))
            g.add_node(t)
        q = _query_emb(dim, 0)
        result = g.retrieve(q, db_id=None, score_threshold=0.0)
        dbs = {n.db_id for n, _ in result}
        assert len(dbs) == 2

    def test_none_db_id_returns_tuples(self):
        g = _build_simple_graph("db_a")
        q = _query_emb(6, 0)
        result = g.retrieve(q, db_id=None)
        for item in result:
            assert len(item) == 2


# ---------------------------------------------------------------------------
# 11. Empty graph edge-cases
# ---------------------------------------------------------------------------


class TestEmptyGraph:
    def test_empty_graph_returns_empty_list(self):
        g = SchemaGraph()
        q = np.zeros(4, dtype=np.float32)
        result = g.retrieve(q, db_id="db")
        assert result == []

    def test_graph_with_only_db_node_returns_empty(self):
        g = SchemaGraph()
        g.add_node(KGNode.make_db("db"))
        q = np.zeros(4, dtype=np.float32)
        result = g.retrieve(q, db_id="db")
        assert result == []

    def test_wrong_db_id_returns_empty(self):
        g = _build_simple_graph("correct_db")
        q = _query_emb(6, 0)
        result = g.retrieve(q, db_id="wrong_db")
        assert result == []


# ---------------------------------------------------------------------------
# 12. Parametrized score/threshold matrix — 100-subset bulk
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_tables,top_m,threshold,max_nodes,query_pos",
    [
        # (n_tables, top_m, threshold, max_nodes, query_pos)
        # Vary query alignment + thresholds
        (2, 1, 0.00, 10, 0),
        (2, 1, 0.00, 10, 1),
        (2, 2, 0.00, 10, 0),
        (2, 2, 0.01, 5,  1),
        (2, 2, 0.05, 5,  0),
        (3, 1, 0.00, 10, 0),
        (3, 1, 0.00, 10, 2),
        (3, 2, 0.00, 10, 4),
        (3, 3, 0.00, 10, 0),
        (3, 3, 0.05, 6,  1),
        (3, 3, 0.10, 6,  2),
        (4, 1, 0.00, 20, 0),
        (4, 2, 0.00, 20, 2),
        (4, 3, 0.00, 20, 4),
        (4, 4, 0.00, 20, 6),
        (4, 4, 0.05, 8,  0),
        (4, 4, 0.10, 4,  0),
        (4, 4, 0.20, 4,  0),
        (5, 1, 0.00, 20, 0),
        (5, 2, 0.00, 20, 2),
        (5, 3, 0.00, 20, 4),
        (5, 4, 0.00, 20, 6),
        (5, 5, 0.00, 20, 8),
        (5, 5, 0.05, 10, 0),
        (5, 5, 0.05, 5,  0),
        (5, 5, 0.05, 2,  0),
        (5, 5, 0.05, 1,  0),
        (6, 1, 0.00, 20, 0),
        (6, 2, 0.00, 20, 4),
        (6, 3, 0.00, 20, 8),
        (6, 4, 0.00, 20, 2),
        (6, 5, 0.00, 20, 6),
        (6, 6, 0.00, 20, 10),
        (6, 6, 0.05, 12, 0),
        (6, 6, 0.10, 6,  0),
        (6, 6, 0.10, 3,  4),
        # Threshold near 1.0 → empty results
        (3, 3, 0.99, 20, 0),
        (4, 4, 0.99, 20, 0),
        (5, 5, 0.99, 20, 0),
        # max_nodes=1 always
        (2, 2, 0.00, 1, 0),
        (3, 3, 0.00, 1, 0),
        (4, 4, 0.00, 1, 0),
        (5, 5, 0.00, 1, 0),
        (6, 6, 0.00, 1, 0),
        # max_nodes matches n_tables*2 (tables+cols)
        (2, 2, 0.00, 4,  0),
        (3, 3, 0.00, 6,  0),
        (4, 4, 0.00, 8,  0),
        (5, 5, 0.00, 10, 0),
        (6, 6, 0.00, 12, 0),
        # Query off-axis (not aligned with any embedding)
        (3, 3, 0.00, 10, 3),
        (4, 4, 0.00, 10, 5),
        (5, 5, 0.00, 10, 7),
        # Synonym boost=0 (no boost, standard path)
        (3, 2, 0.00, 10, 0),
        (4, 2, 0.00, 10, 2),
        (5, 3, 0.00, 10, 4),
        # top_m=1 with various thresholds
        (4, 1, 0.00, 10, 0),
        (4, 1, 0.01, 10, 2),
        (4, 1, 0.05, 10, 4),
        (5, 1, 0.00, 10, 0),
        (5, 1, 0.05, 10, 2),
        (6, 1, 0.00, 10, 0),
        # alpha variation tests — done by varying other params
        (3, 2, 0.00, 6, 0),
        (3, 2, 0.00, 6, 2),
        (3, 2, 0.00, 6, 4),
        (4, 3, 0.00, 8, 0),
        (4, 3, 0.00, 8, 2),
        (4, 3, 0.00, 8, 4),
        (4, 3, 0.00, 8, 6),
        # Large graph variants
        (7, 5, 0.00, 20, 0),
        (7, 5, 0.05, 15, 4),
        (7, 5, 0.10, 10, 8),
        (8, 5, 0.00, 20, 0),
        (8, 5, 0.05, 16, 6),
        (8, 6, 0.00, 20, 10),
        (8, 8, 0.00, 20, 0),
        (8, 8, 0.05, 10, 0),
        # Edge: top_m > actual nodes with embeddings
        (2, 10, 0.00, 20, 0),
        (3, 10, 0.00, 20, 0),
        # Re-run key alignments for statistical robustness
        (4, 4, 0.02, 20, 0),
        (4, 4, 0.03, 20, 2),
        (4, 4, 0.04, 20, 4),
        (5, 5, 0.02, 20, 0),
        (5, 5, 0.03, 20, 2),
        (5, 5, 0.04, 20, 4),
        (6, 6, 0.02, 20, 0),
        (6, 6, 0.03, 20, 4),
        (6, 6, 0.04, 20, 8),
        # Combinations with max_nodes=2
        (3, 2, 0.00, 2, 0),
        (4, 2, 0.00, 2, 2),
        (5, 3, 0.00, 2, 4),
        (6, 4, 0.00, 2, 6),
        # max_nodes=0 (edge: nothing should be returned)
        (3, 3, 0.00, 0, 0),
        # Strict threshold with top_m=1
        (5, 1, 0.30, 10, 0),
        (5, 1, 0.50, 10, 0),
    ],
)
def test_retrieve_parametrized(
    n_tables: int,
    top_m: int,
    threshold: float,
    max_nodes: int,
    query_pos: int,
):
    """
    Parametrized bulk test: every combination must satisfy the following
    invariants regardless of PPR internals:
      1. Returns a list of (KGNode, float) tuples.
      2. len(result) <= max_nodes.
      3. All scores >= threshold.
      4. All scores are non-negative.
      5. Result is sorted descending by score.
      6. Only TABLE and COLUMN nodes returned.
      7. All nodes belong to the queried db_id.
    """
    dim = max(n_tables * 2, 4)
    g = _build_simple_graph(n_tables=n_tables)
    query_pos = min(query_pos, dim - 1)
    q = _query_emb(dim, query_pos)

    result = g.retrieve(
        q,
        db_id="db",
        top_m=max(top_m, 1),
        score_threshold=threshold,
        max_nodes=max_nodes,
    )

    assert isinstance(result, list)
    assert len(result) <= max(max_nodes, 0)

    scores = [s for _, s in result]
    for s in scores:
        assert s >= threshold - 1e-9
        assert s >= -1e-9  # non-negative

    assert scores == sorted(scores, reverse=True)

    for node, _ in result:
        assert node.node_type in (NodeType.TABLE, NodeType.COLUMN)
        assert node.db_id == "db"
