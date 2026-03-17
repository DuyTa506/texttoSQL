"""
Schema Knowledge Graph — builder and container.

SchemaGraph
-----------
  The graph container backed by ``networkx.MultiDiGraph``.

  Holds KGNode / KGEdge objects as node / edge attributes, exposes graph-traversal
  retrieval (Personalized PageRank via ``nx.pagerank``), serialization, and
  convenience query methods.

  NetworkX gives us:
    • Battle-tested PPR via  nx.pagerank(G, personalization=seed_scores)
    • Rich graph algorithms available for future extensions
    • Standard Python data-science ecosystem — easy for collaborators to extend
    • MultiDiGraph supports multiple edge types between the same node pair
      (e.g. FOREIGN_KEY *and* EMBEDDING_SIMILAR between the same two columns)

  Node attributes stored as:  G.nodes[node_id]["data"] = KGNode
  Edge attributes stored as:  G[src][dst][edge_key]["data"] = KGEdge
                              G[src][dst][edge_key]["weight"] = KGEdge.weight
  (The ``weight`` attribute is read directly by ``nx.pagerank``.)

SchemaGraphBuilder
------------------
  Factory that constructs a SchemaGraph from a ``Database`` object.
  Runs Layer-1 (structural) edges by default.
  Layer-2 (semantic) and Layer-3 (statistical) edges are added later via
  dedicated builders (semantic_edges.py, statistical_edges.py) that mutate
  the graph in-place.

Usage
-----
  # Build structural graph only (fast, no LLM needed)
  builder = SchemaGraphBuilder()
  graph   = builder.build(db)

  # Later: enrich nodes with LLM descriptions (offline)
  # from src.schema_graph.node_enricher import NodeEnricher
  # enricher = NodeEnricher(llm_client)
  # enricher.enrich(graph)

  # Later: add semantic + statistical edges
  # from src.schema_graph.edge_builders.semantic_edges import build_semantic_edges
  # graph.add_edges(build_semantic_edges(graph))

  # Retrieve schema subgraph for a question
  subgraph_nodes = graph.retrieve(question_embedding, top_m=5, alpha=0.7)
  schema_str     = graph.to_schema_context(subgraph_nodes)

  # Save / load  (embeddings serialized as list[float] in JSON)
  graph.save("data/schema_graphs/spider.json")
  graph2 = SchemaGraph.load("data/schema_graphs/spider.json")
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

from src.schema.models import Database
from src.schema_graph.edge_builders.structural_edges import build_structural_edges
from src.schema_graph.graph_types import (
    STATISTICAL_EDGE_TYPES,
    STRUCTURAL_EDGE_TYPES,
    EdgeType,
    KGEdge,
    KGNode,
    NodeType,
    SEMANTIC_EDGE_TYPES,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SchemaGraph
# ---------------------------------------------------------------------------


class SchemaGraph:
    """
    Container for a Schema Knowledge Graph of one or more databases.

    Backed by a ``networkx.MultiDiGraph``.

    Node storage
    ~~~~~~~~~~~~
    Each node is registered as::

        G.add_node(node_id, data=KGNode(...))

    Attribute access::

        kg_node: KGNode = graph.G.nodes[node_id]["data"]

    Edge storage
    ~~~~~~~~~~~~
    Each edge is registered as::

        G.add_edge(src_id, dst_id,
                   key=edge_type.value,   # unique key within (src, dst) pair
                   data=KGEdge(...),
                   weight=edge.weight)    # nx.pagerank reads "weight" directly

    Attribute access::

        kg_edge: KGEdge = graph.G[src][dst][edge_key]["data"]
    """

    def __init__(self) -> None:
        self.G: nx.MultiDiGraph = nx.MultiDiGraph()
        # Optional ChromaDB collection for ANN entry-point lookup.
        # Set via attach_chroma() after loading the graph.
        self._chroma_collection = None

    # ── ChromaDB integration ───────────────────────────────────────────────────

    def attach_chroma(self, collection) -> None:
        """
        Attach a ChromaDB collection that holds pre-computed node embeddings.

        When attached, ``retrieve()`` uses ChromaDB's HNSW ANN index for the
        initial cosine entry-point scoring instead of a linear numpy scan over
        all nodes.  This is significantly faster for large schemas and avoids
        loading large float arrays into memory.

        Parameters
        ----------
        collection:
            A ``chromadb.Collection`` object (already opened, cosine space).
            Typically the ``"schema_graph_nodes"`` collection produced by
            ``NodeEnricher.embed_nodes(chroma_persist_dir=...)``.

        Note
        ----
        Call this once after ``SchemaGraph.load()`` at the start of inference:

            graph = SchemaGraph.load("spider_enriched.json")
            graph.attach_chroma(chroma_client.get_collection("schema_graph_nodes"))
        """
        self._chroma_collection = collection
        logger.debug(
            "SchemaGraph: ChromaDB collection '%s' attached for ANN entry-point search.",
            getattr(collection, "name", str(collection)),
        )

    # ── Mutation ──────────────────────────────────────────────────────────────

    def add_node(self, node: KGNode) -> None:
        """Add (or replace) a node.  If the node_id already exists the KGNode
        is updated in-place on the existing nx node."""
        self.G.add_node(node.node_id, data=node)

    def add_nodes(self, nodes: list[KGNode]) -> None:
        for n in nodes:
            self.add_node(n)

    def add_edge(self, edge: KGEdge) -> None:
        """Add a directed edge.

        The edge_type value is used as the MultiDiGraph *key* so that two edges
        of different types between the same (src, dst) pair are kept distinct.
        If an edge with the same (src, dst, edge_type) already exists it is
        overwritten (idempotent).
        """
        self.G.add_edge(
            edge.src_id,
            edge.dst_id,
            key=edge.edge_type.value,
            data=edge,
            weight=edge.weight,
        )

    def add_edges(self, edges: list[KGEdge]) -> None:
        for e in edges:
            self.add_edge(e)

    # ── Query helpers ─────────────────────────────────────────────────────────

    @property
    def nodes(self) -> dict[str, KGNode]:
        """Return a plain dict mapping node_id → KGNode (snapshot, not live view)."""
        return {nid: attrs["data"] for nid, attrs in self.G.nodes(data=True)
                if "data" in attrs}

    @property
    def edges(self) -> list[KGEdge]:
        """Return all edges as a list of KGEdge objects."""
        return [attrs["data"] for _, _, attrs in self.G.edges(data=True)
                if "data" in attrs]

    def get_node(self, node_id: str) -> KGNode | None:
        if node_id in self.G:
            return self.G.nodes[node_id].get("data")
        return None

    def neighbors(
        self,
        node_id: str,
        edge_types: set[EdgeType] | None = None,
    ) -> list[KGEdge]:
        """Return outgoing KGEdges from *node_id*, optionally filtered by type."""
        if node_id not in self.G:
            return []
        result: list[KGEdge] = []
        for dst, key_dict in self.G[node_id].items():
            for _key, attrs in key_dict.items():
                edge: KGEdge = attrs.get("data")
                if edge is None:
                    continue
                if edge_types is None or edge.edge_type in edge_types:
                    result.append(edge)
        return result

    def nodes_by_type(self, node_type: NodeType) -> list[KGNode]:
        return [
            attrs["data"]
            for _, attrs in self.G.nodes(data=True)
            if "data" in attrs and attrs["data"].node_type == node_type
        ]

    def nodes_for_db(self, db_id: str) -> list[KGNode]:
        return [
            attrs["data"]
            for _, attrs in self.G.nodes(data=True)
            if "data" in attrs and attrs["data"].db_id == db_id
        ]

    def column_nodes_for_table(self, db_id: str, table_name: str) -> list[KGNode]:
        prefix = f"{db_id}.{table_name}."
        return [
            attrs["data"]
            for nid, attrs in self.G.nodes(data=True)
            if "data" in attrs
            and attrs["data"].node_type == NodeType.COLUMN
            and nid.startswith(prefix)
        ]

    # ── Statistics ────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return a summary dict of node/edge counts by type."""
        node_counts: dict[str, int] = defaultdict(int)
        enriched_cols = 0
        total_cols = 0

        for _, attrs in self.G.nodes(data=True):
            node: KGNode = attrs.get("data")
            if node is None:
                continue
            node_counts[node.node_type.value] += 1
            if node.node_type == NodeType.COLUMN:
                total_cols += 1
                if node.description:
                    enriched_cols += 1

        edge_counts: dict[str, int] = defaultdict(int)
        for _, _, attrs in self.G.edges(data=True):
            edge: KGEdge = attrs.get("data")
            if edge:
                edge_counts[edge.edge_type.value] += 1

        return {
            "total_nodes": self.G.number_of_nodes(),
            "total_edges": self.G.number_of_edges(),
            "nodes_by_type": dict(node_counts),
            "edges_by_type": dict(edge_counts),
            "enriched_columns": enriched_cols,
            "total_columns": total_cols,
            "enrichment_ratio": enriched_cols / max(total_cols, 1),
        }

    # ── Retrieval (Personalized PageRank) ─────────────────────────────────────

    def retrieve(
        self,
        query_embedding: np.ndarray,
        *,
        db_id: str | None = None,
        top_m: int = 5,
        alpha: float = 0.7,
        max_iter: int = 100,
        score_threshold: float = 0.05,
        max_nodes: int = 20,
        synonym_tokens: set[str] | None = None,
        synonym_boost: float = 0.3,
        value_matched_nodes: dict[str, float] | None = None,
    ) -> list[tuple[KGNode, float]]:
        """
        Retrieve a relevant schema subgraph for the given query embedding via
        Personalized PageRank (PPR) using ``nx.pagerank``.

        Algorithm
        ---------
        1. Compute cosine similarity between *query_embedding* and all node
           embeddings.  Only COLUMN and TABLE nodes that have embeddings are
           candidates.
        2. Apply synonym exact-match boost (if *synonym_tokens* provided).
        3. Take the top-*top_m* nodes as PPR seeds; build a sparse
           ``personalization`` dict proportional to their cosine scores.
        4. Call ``nx.pagerank(G, alpha=alpha, personalization=...,
           weight='weight')`` — edge weights act as transition probabilities.
        5. Return all nodes with final PPR score > *score_threshold*, capped
           at *max_nodes*, sorted descending.

        Parameters
        ----------
        query_embedding:
            Dense query vector (same dimension as node embeddings).
        db_id:
            If provided, restrict retrieval to nodes from this database.
            PPR runs on a per-DB subgraph view (O(1) copy) to prevent
            probability-mass leakage across databases.
        top_m:
            Number of highest-similarity nodes used as PPR seeds.
        alpha:
            Damping factor (probability of following an edge vs. teleporting
            back to seeds).  0.7 → favours nodes close to seeds.
        max_iter:
            Maximum PPR power-iteration steps (passed to nx.pagerank).
        score_threshold:
            Nodes with final PPR score below this are excluded.
        max_nodes:
            Hard cap on returned nodes.
        synonym_tokens:
            Set of lowercase tokens from the question to match against node
            synonyms.
        synonym_boost:
            Score increment added to nodes whose synonyms intersect with
            *synonym_tokens*.
        value_matched_nodes:
            Optional mapping ``{node_id → boost_score}`` from ValueScanner
            matches (v3).  Boosts are merged into the entry-point scores
            before PPR seed selection, ensuring columns containing matched
            cell values are strongly seeded.

        Returns
        -------
        list[tuple[KGNode, float]]
            Pairs of (schema node, PPR score), sorted by score descending.
        """
        scores = self._get_entry_points(
            query_embedding,
            db_id=db_id,
            top_m=top_m,
            synonym_tokens=synonym_tokens,
            synonym_boost=synonym_boost,
            max_nodes=max_nodes,
        )
        # _get_entry_points returns early fallback tuples when no embeddings found
        if scores is None:
            fallback = [
                attrs["data"]
                for _, attrs in self.G.nodes(data=True)
                if "data" in attrs
                and attrs["data"].node_type in (NodeType.COLUMN, NodeType.TABLE)
                and (db_id is None or attrs["data"].db_id == db_id)
            ]
            return [(n, 0.0) for n in fallback[:max_nodes]]

        self._merge_value_boosts(scores, value_matched_nodes)
        personalization = self._build_personalization(scores, top_m)
        ppr_scores = self._run_ppr(personalization, db_id=db_id, alpha=alpha, max_iter=max_iter, fallback_scores=scores)
        return self._filter_and_sort(ppr_scores, db_id=db_id, score_threshold=score_threshold, max_nodes=max_nodes)

    # ── PPR phase helpers ──────────────────────────────────────────────────────

    def _get_entry_points(
        self,
        query_embedding: np.ndarray,
        *,
        db_id: str | None,
        top_m: int,
        synonym_tokens: set[str] | None,
        synonym_boost: float,
        max_nodes: int,
    ) -> dict[str, float] | None:
        """Step 1 — compute cosine entry-point scores via ChromaDB or linear scan.

        Returns a ``{node_id → score}`` dict, or ``None`` when no embedded nodes
        are found at all (caller should return name-only fallback).
        """
        if self._chroma_collection is not None:
            scores = self._chroma_entry_points(
                query_embedding,
                db_id=db_id,
                top_m=top_m,
                synonym_tokens=synonym_tokens,
                synonym_boost=synonym_boost,
            )
            if not scores:
                logger.warning(
                    "ChromaDB entry-point query returned no results%s — "
                    "falling back to linear scan.",
                    f" for db_id={db_id}" if db_id else "",
                )
                scores = self._linear_entry_points(
                    query_embedding,
                    db_id=db_id,
                    top_m=top_m,
                    synonym_tokens=synonym_tokens,
                    synonym_boost=synonym_boost,
                )
        else:
            scores = self._linear_entry_points(
                query_embedding,
                db_id=db_id,
                top_m=top_m,
                synonym_tokens=synonym_tokens,
                synonym_boost=synonym_boost,
            )

        if not scores:
            logger.warning(
                "No embedded nodes found%s — returning top-%d by name only.",
                f" for db_id={db_id}" if db_id else "",
                max_nodes,
            )
            return None

        return scores

    @staticmethod
    def _merge_value_boosts(
        scores: dict[str, float],
        value_matched_nodes: dict[str, float] | None,
    ) -> None:
        """Step 2b — merge ValueScanner cell-value boosts into entry-point scores in-place."""
        if not value_matched_nodes:
            return
        for nid, boost in value_matched_nodes.items():
            scores[nid] = min(scores.get(nid, 0.0) + boost, 1.0)

    def _build_personalization(
        self,
        scores: dict[str, float],
        top_m: int,
    ) -> dict[str, float]:
        """Step 3 — build the normalized PPR personalization vector from top-M seeds.

        v3: When a FK column is selected as a seed, its parent TABLE node is also
        injected at half the column's score.  This gives the PPR walk a direct
        table-level entry point instead of having to traverse 4 hops
        (column → COLUMN_BELONGS_TO → table → TABLE_HAS_COLUMN → column).
        Expected gain: +3–5 pp recall on multi-join queries.
        """
        sorted_seeds = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        seed_ids = [nid for nid, _ in sorted_seeds[:top_m]]
        raw: dict[str, float] = {nid: max(scores[nid], 1e-9) for nid in seed_ids}

        # v3: inject parent TABLE node for every FK-column seed
        for nid in seed_ids:
            node = self.nodes.get(nid)
            if node is not None and node.is_fk and node.table_name:
                tbl_nid = f"{node.db_id}.{node.table_name}"
                if tbl_nid not in raw and tbl_nid in self.nodes:
                    raw[tbl_nid] = raw[nid] * 0.5   # table seed at half the column's weight

        total = sum(raw.values())
        return {nid: s / total for nid, s in raw.items()}

    def _run_ppr(
        self,
        personalization: dict[str, float],
        *,
        db_id: str | None,
        alpha: float,
        max_iter: int,
        fallback_scores: dict[str, float],
    ) -> dict[str, float]:
        """Step 4 — run Personalized PageRank on the per-DB subgraph.

        Returns a ``{node_id → ppr_score}`` dict.
        Falls back to cosine seed scores if PPR fails to converge.
        """
        if db_id is not None:
            db_node_ids = [
                nid for nid, attrs in self.G.nodes(data=True)
                if "data" in attrs and attrs["data"].db_id == db_id
            ]
            run_graph = self.G.subgraph(db_node_ids)
        else:
            run_graph = self.G

        try:
            return nx.pagerank(
                run_graph,
                alpha=alpha,
                personalization=personalization,
                max_iter=max_iter,
                tol=1e-6,
                weight="weight",
            )
        except nx.PowerIterationFailedConvergence:
            logger.warning(
                "PPR did not converge (max_iter=%d); falling back to cosine seeds.",
                max_iter,
            )
            return {nid: fallback_scores.get(nid, 0.0) for nid in self.G.nodes}

    def _filter_and_sort(
        self,
        ppr_scores: dict[str, float],
        *,
        db_id: str | None,
        score_threshold: float,
        max_nodes: int,
    ) -> list[tuple[KGNode, float]]:
        """Step 5 — filter by threshold/db_id, sort descending, cap at max_nodes."""
        all_nodes = self.nodes
        result_ids = [
            nid for nid, s in ppr_scores.items()
            if s >= score_threshold
            and nid in all_nodes
            and all_nodes[nid].node_type in (NodeType.COLUMN, NodeType.TABLE)
            and (db_id is None or all_nodes[nid].db_id == db_id)
        ]
        result_ids.sort(key=lambda nid: ppr_scores[nid], reverse=True)
        result_ids = result_ids[:max_nodes]
        return [(all_nodes[nid], ppr_scores[nid]) for nid in result_ids]

    # ── Entry-point scoring helpers ────────────────────────────────────────────

    def _chroma_entry_points(
        self,
        query_embedding: np.ndarray,
        *,
        db_id: str | None,
        top_m: int,
        synonym_tokens: set,
        synonym_boost: float,
    ) -> dict[str, float]:
        """
        Use ChromaDB HNSW ANN to retrieve the top-M most similar nodes,
        then apply synonym boost.

        ChromaDB uses cosine distance (0 = identical, 2 = opposite).
        We convert to similarity: score = 1 - distance.

        Returns a dict {node_id → score} for the top-M seeds.
        Only nodes that also exist in the NetworkX graph are included
        (guards against stale ChromaDB entries after graph edits).
        """
        where = {"db_id": {"$eq": db_id}} if db_id else None

        try:
            results = self._chroma_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=max(top_m * 2, 10),  # fetch extra; filter to graph nodes
                where=where,
                include=["distances", "metadatas"],
            )
        except Exception as exc:
            logger.warning("ChromaDB query failed: %s", exc)
            return {}

        ids       = results["ids"][0] if results["ids"] else []
        distances = results["distances"][0] if results["distances"] else []

        if not ids:
            return {}

        # Build scores dict — only keep nodes present in the current graph
        scores: dict[str, float] = {}
        for nid, dist in zip(ids, distances):
            if nid in self.G.nodes:
                scores[nid] = 1.0 - float(dist)   # cosine distance → similarity

        # Synonym boost using KGNode.synonyms from the graph
        if synonym_tokens and scores:
            for nid in list(scores.keys()):
                node: KGNode = self.G.nodes[nid]["data"]
                if {s.lower() for s in node.synonyms} & synonym_tokens:
                    scores[nid] = min(1.0, scores[nid] + synonym_boost)

        # Return only top_m after boost re-ranking
        top_scores = dict(
            sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_m]
        )
        return top_scores

    def _linear_entry_points(
        self,
        query_embedding: np.ndarray,
        *,
        db_id: str | None,
        top_m: int,
        synonym_tokens: set,
        synonym_boost: float,
    ) -> dict[str, float]:
        """
        Linear O(N) cosine scan over all in-node embeddings.
        Used as fallback when no ChromaDB collection is attached, or as
        a compatibility path for graphs that were built without ChromaDB.

        Returns a dict {node_id → score} for all nodes that have embeddings.
        """
        candidates: list[KGNode] = [
            attrs["data"]
            for nid, attrs in self.G.nodes(data=True)
            if "data" in attrs
            and attrs["data"].embedding
            and attrs["data"].node_type in (NodeType.COLUMN, NodeType.TABLE)
            and (db_id is None or attrs["data"].db_id == db_id)
        ]

        if not candidates:
            return {}

        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        scores: dict[str, float] = {}
        for node in candidates:
            node_emb  = np.array(node.embedding, dtype=np.float32)
            node_norm = node_emb / (np.linalg.norm(node_emb) + 1e-10)
            scores[node.node_id] = float(np.dot(q_norm, node_norm))

        if synonym_tokens:
            for node in candidates:
                if {s.lower() for s in node.synonyms} & synonym_tokens:
                    scores[node.node_id] = min(1.0, scores[node.node_id] + synonym_boost)

        return scores

    # ── Schema context formatting ──────────────────────────────────────────────

    def to_schema_context(
        self,
        subgraph_nodes: list[KGNode],
        *,
        include_descriptions: bool = True,
        include_fk_summary: bool = True,
    ) -> str:
        """
        Format a list of KGNodes as a CREATE TABLE schema context string,
        analogous to SchemaFilter.filter_and_format() but driven by the graph.

        Parameters
        ----------
        subgraph_nodes:
            The nodes returned by ``retrieve()``.
        include_descriptions:
            If True and a node has an LLM description, append it as SQL comment.
        include_fk_summary:
            If True, append a ``Foreign Keys:`` section at the bottom.

        Returns
        -------
        str
            Formatted schema ready for an LLM prompt.
        """
        if not subgraph_nodes:
            return ""

        db_ids = sorted({n.db_id for n in subgraph_nodes})

        table_nodes: dict[str, KGNode] = {}
        col_nodes_by_table: dict[str, list[KGNode]] = defaultdict(list)

        for node in subgraph_nodes:
            if node.node_type == NodeType.TABLE:
                table_nodes[node.node_id] = node
            elif node.node_type == NodeType.COLUMN:
                table_key = f"{node.db_id}.{node.table_name}"
                col_nodes_by_table[table_key].append(node)

        # Ensure every column's parent table is present
        all_nodes = self.nodes
        for table_key, cols in col_nodes_by_table.items():
            if table_key not in table_nodes and table_key in all_nodes:
                table_nodes[table_key] = all_nodes[table_key]

        lines: list[str] = []
        if db_ids:
            lines.append(f"Database: {', '.join(db_ids)}")
            lines.append("")

        fk_lines: list[str] = []

        for table_key, table_node in sorted(table_nodes.items()):
            col_nodes = sorted(
                col_nodes_by_table.get(table_key, []),
                key=lambda c: c.column_name,
            )
            col_parts: list[str] = []
            for col in col_nodes:
                dtype_str = col.dtype or "TEXT"
                pk_str = " [PK]" if col.is_pk else ""
                fk_str = " [FK]" if col.is_fk else ""
                col_parts.append(f"{col.column_name} ({dtype_str}){pk_str}{fk_str}")

            cols_str = ", ".join(col_parts) if col_parts else "..."
            lines.append(f"CREATE TABLE {table_node.table_name} ({cols_str})")

            if include_descriptions and table_node.description:
                lines.append(f"  -- {table_node.description}")

            if include_descriptions:
                for col in col_nodes:
                    if col.description:
                        lines.append(f"  -- {col.column_name}: {col.description}")

            # FK edges via nx adjacency
            for col in col_nodes:
                for edge in self.neighbors(col.node_id,
                                           edge_types={EdgeType.FOREIGN_KEY}):
                    dst = all_nodes.get(edge.dst_id)
                    if dst and dst.node_type == NodeType.COLUMN:
                        lines.append(
                            f"  -- FK: {col.table_name}.{col.column_name}"
                            f" → {dst.table_name}.{dst.column_name}"
                        )
                        fk_lines.append(
                            f"  {col.table_name}.{col.column_name}"
                            f" = {dst.table_name}.{dst.column_name}"
                        )
            lines.append("")

        if include_fk_summary and fk_lines:
            seen: set[str] = set()
            unique_fk: list[str] = []
            for fl in fk_lines:
                if fl not in seen:
                    seen.add(fl)
                    unique_fk.append(fl)
            lines.append("Foreign Keys:")
            lines.extend(unique_fk)

        return "\n".join(lines).rstrip()

    # ── Serialization ─────────────────────────────────────────────────────────
    #
    # We do NOT use nx.node_link_data() because node attributes include
    # list[float] embeddings (numpy-derived) that need controlled serialization.
    # Our own format is a simple {"nodes": [...], "edges": [...]} JSON dict
    # using KGNode.to_dict() / KGEdge.to_dict() which handle the embedding
    # as list[float] explicitly.

    def save(self, path: str | Path) -> None:
        """Serialize the graph to a JSON file at *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(
            "Saved SchemaGraph (%d nodes, %d edges) to %s",
            self.G.number_of_nodes(),
            self.G.number_of_edges(),
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "SchemaGraph":
        """Deserialize a graph from a JSON file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        graph = cls()
        for nd in data.get("nodes", []):
            graph.add_node(KGNode.from_dict(nd))
        for ed in data.get("edges", []):
            graph.add_edge(KGEdge.from_dict(ed))
        logger.info(
            "Loaded SchemaGraph (%d nodes, %d edges) from %s",
            graph.G.number_of_nodes(),
            graph.G.number_of_edges(),
            path,
        )
        return graph

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"SchemaGraph("
            f"nodes={s['total_nodes']}, edges={s['total_edges']}, "
            f"enriched_columns={s['enriched_columns']}/{s['total_columns']})"
        )


# ---------------------------------------------------------------------------
# SchemaGraphBuilder
# ---------------------------------------------------------------------------


class SchemaGraphBuilder:
    """
    Factory that builds a :class:`SchemaGraph` from one or more
    :class:`~src.schema.models.Database` objects.

    By default only Layer-1 (structural) edges are added.
    Layer-2 and Layer-3 edges are added later by calling the dedicated
    builder functions and using ``graph.add_edges()``.

    Parameters
    ----------
    sample_values_limit:
        Maximum number of sample values stored per column node.
        Values come from ``Database.sample_values`` (pre-populated
        by the adapter or by a ValueScanner call).  Default: 20.
    """

    def __init__(self, sample_values_limit: int = 20) -> None:
        self.sample_values_limit = sample_values_limit

    # ── Public API ────────────────────────────────────────────────────────────

    def build(self, db: Database) -> SchemaGraph:
        """
        Build a SchemaGraph for a single *Database*.

        Steps:
          1. Create DATABASE node.
          2. Create TABLE nodes.
          3. Create COLUMN nodes (with dtype, is_pk, sample_values).
          4. Build Layer-1 structural edges (including is_fk flag mutation).

        Parameters
        ----------
        db:
            The database whose schema drives graph construction.

        Returns
        -------
        SchemaGraph
            A fully-structured graph with Layer-1 edges only.
            Ready for Layer-2/3 augmentation and optional LLM enrichment.
        """
        graph = SchemaGraph()
        nodes = self._build_nodes(db)
        graph.add_nodes(list(nodes.values()))

        edges = build_structural_edges(db, nodes)
        graph.add_edges(edges)

        st = graph.stats()
        logger.info(
            "Built structural graph for '%s': %d nodes (%d tables, %d columns), "
            "%d edges",
            db.db_id,
            st["total_nodes"],
            st["nodes_by_type"].get("table", 0),
            st["nodes_by_type"].get("column", 0),
            st["total_edges"],
        )
        return graph

    def build_many(self, databases: list[Database]) -> SchemaGraph:
        """
        Build a single SchemaGraph covering *all* databases in the list.

        All nodes and edges from each database are merged into one graph.
        This is useful when building a shared graph for an entire benchmark
        (e.g. all 206 Spider databases in one file).

        Parameters
        ----------
        databases:
            List of Database objects to merge.

        Returns
        -------
        SchemaGraph
            Combined graph for all databases.
        """
        merged = SchemaGraph()
        for db in databases:
            db_nodes = self._build_nodes(db)
            merged.add_nodes(list(db_nodes.values()))
            db_edges = build_structural_edges(db, db_nodes)
            merged.add_edges(db_edges)

        st = merged.stats()
        logger.info(
            "Built merged structural graph for %d databases: "
            "%d nodes (%d tables, %d columns), %d edges",
            len(databases),
            st["total_nodes"],
            st["nodes_by_type"].get("table", 0),
            st["nodes_by_type"].get("column", 0),
            st["total_edges"],
        )
        return merged

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_nodes(self, db: Database) -> dict[str, KGNode]:
        """
        Construct all KGNodes for *db* and return them as a dict.

        Returns
        -------
        dict[str, KGNode]
            Mapping node_id → KGNode.
        """
        nodes: dict[str, KGNode] = {}

        # DATABASE node
        db_node = KGNode.make_db(db.db_id)
        nodes[db_node.node_id] = db_node

        for table in db.tables:
            # TABLE node
            t_node = KGNode.make_table(db.db_id, table.name)
            nodes[t_node.node_id] = t_node

            for col in table.columns:
                # Sample values from Database.sample_values dict
                sv_key = f"{table.name}.{col.name}"
                raw_sv = db.sample_values.get(sv_key, [])
                sv = [str(v) for v in raw_sv][: self.sample_values_limit]

                col_node = KGNode.make_column(
                    db_id=db.db_id,
                    table_name=table.name,
                    column_name=col.name,
                    dtype=col.dtype,
                    is_pk=col.primary_key,
                    sample_values=sv,
                )
                # Propagate any pre-existing description from Column dataclass
                if col.description:
                    col_node.description = col.description

                nodes[col_node.node_id] = col_node

        return nodes
