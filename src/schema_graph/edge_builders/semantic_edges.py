"""
Layer 2 — Semantic Edge Builder.

Derives edges from name-level and description-level similarity between nodes.
No SQL parsing, no DB access, no LLM calls at build time.

Three edge types produced (all bidirectional — both directions added):

  LEXICAL_SIMILAR
    Between any two COLUMN (or TABLE) nodes in the same database whose
    tokenized identifiers share significant token overlap.
    Weight = Jaccard similarity of token bags.
    Always available — requires only node names, zero external dependencies.

  EMBEDDING_SIMILAR
    Between COLUMN (or TABLE) nodes whose pre-computed description embeddings
    are close in cosine space.
    Weight = cosine similarity (clipped to [0, 1]).
    Requires: nodes must have ``embedding`` populated (by NodeEnricher).
    Nodes without embeddings are silently skipped.

  SYNONYM_MATCH
    Between COLUMN (or TABLE) nodes whose LLM-generated synonym sets overlap.
    Weight = |intersection| / max(|syns_a|, |syns_b|) — normalised overlap.
    Requires: nodes must have ``synonyms`` populated (by NodeEnricher).
    Nodes without synonyms are silently skipped.

Design choices
--------------
* All three sub-builders operate **within a single database only** — cross-DB
  semantic edges are not meaningful for schema linking.
* ``build_semantic_edges(graph)`` processes every database found in the graph.
  Pass ``db_id=`` to restrict to one database (useful during incremental builds).
* Self-loops (src == dst) are never added.
* Duplicate edges (same src/dst/type) are not deduplicated here — the graph
  container allows them and PPR handles it gracefully via weight normalisation.
  In practice duplicates don't occur because we iterate over sorted pairs.
* The ``include_same_table_pairs`` flag (default True) controls whether columns
  within the same table can get LEXICAL_SIMILAR / EMBEDDING_SIMILAR edges.
  Setting it to False limits semantic edges to cross-table column pairs only.
  Keeping it True allows PPR to find related columns within a table
  (e.g. ``first_name`` ↔ ``last_name`` both tokenise to ['name']).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from itertools import combinations
from typing import Optional, TYPE_CHECKING

import numpy as np

from src.schema_graph.graph_types import EdgeType, KGEdge, KGNode, NodeType
from src.schema_graph.edge_builders.structural_edges import tokenize_name

if TYPE_CHECKING:
    from src.schema_graph.graph_builder import SchemaGraph

logger = logging.getLogger(__name__)


# ── Weight / threshold constants ──────────────────────────────────────────────

# Lexical
DEFAULT_LEXICAL_THRESHOLD: float = 0.4       # minimum Jaccard to emit an edge
W_LEXICAL_MAX: float = 0.7                   # cap weight so lexical never beats FK (0.95)

# Embedding
DEFAULT_EMBEDDING_THRESHOLD: float = 0.75   # minimum cosine to emit an edge
W_EMBEDDING_MAX: float = 0.9                 # strong signal, but still below structural

# Synonym
DEFAULT_SYNONYM_MIN_OVERLAP: int = 1         # at least 1 common synonym token
W_SYNONYM_MAX: float = 0.85                  # high precision signal


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_semantic_edges(
    graph: "SchemaGraph",
    *,
    db_id: Optional[str] = None,
    lexical_threshold: float = DEFAULT_LEXICAL_THRESHOLD,
    embedding_threshold: float = DEFAULT_EMBEDDING_THRESHOLD,
    synonym_min_overlap: int = DEFAULT_SYNONYM_MIN_OVERLAP,
    include_same_table_pairs: bool = True,
    node_types: tuple[NodeType, ...] = (NodeType.COLUMN, NodeType.TABLE),
) -> list[KGEdge]:
    """
    Build all Layer-2 (semantic) edges for the graph.

    Parameters
    ----------
    graph:
        The ``SchemaGraph`` whose nodes are inspected.
        The graph is NOT mutated — edges are returned for the caller to add
        via ``graph.add_edges()``.
    db_id:
        If given, restrict edge building to this database only.
        If None, process all databases found in the graph.
    lexical_threshold:
        Minimum Jaccard coefficient (token overlap) to create a
        ``LEXICAL_SIMILAR`` edge.  Range [0, 1]; default 0.4.
    embedding_threshold:
        Minimum cosine similarity to create an ``EMBEDDING_SIMILAR`` edge.
        Range [0, 1]; default 0.75.
        Nodes without embeddings are silently excluded.
    synonym_min_overlap:
        Minimum number of shared synonym tokens to create a
        ``SYNONYM_MATCH`` edge.  Default 1 (any shared synonym).
        Synonyms are tokenized the same way as column names (snake_case split).
    include_same_table_pairs:
        If True (default), column pairs within the same table are eligible
        for semantic edges.  Set False to limit edges to cross-table pairs.
    node_types:
        Which node types participate.  Default: COLUMN and TABLE.
        DATABASE nodes are excluded regardless.

    Returns
    -------
    list[KGEdge]
        All Layer-2 edges, in order: LEXICAL → EMBEDDING → SYNONYM.
        Both directions (A→B and B→A) are included for each pair.
    """
    # Determine which db_ids to process
    if db_id is not None:
        target_dbs = {db_id}
    else:
        target_dbs = {n.db_id for n in graph.nodes.values()
                      if n.node_type != NodeType.DATABASE}

    all_edges: list[KGEdge] = []

    for current_db in sorted(target_dbs):
        # Gather eligible nodes for this DB
        candidates = [
            n for n in graph.nodes.values()
            if n.db_id == current_db
            and n.node_type in node_types
            and n.node_type != NodeType.DATABASE
        ]

        if len(candidates) < 2:
            continue   # Nothing to compare

        # Build pairwise comparisons (upper triangle only — we add both directions)
        pairs: list[tuple[KGNode, KGNode]] = []
        for a, b in combinations(candidates, 2):
            if not include_same_table_pairs and a.table_name == b.table_name:
                continue
            pairs.append((a, b))

        lex_edges = _build_lexical_edges(pairs, lexical_threshold)
        emb_edges = _build_embedding_edges(pairs, embedding_threshold)
        syn_edges = _build_synonym_edges(pairs, synonym_min_overlap)

        all_edges.extend(lex_edges)
        all_edges.extend(emb_edges)
        all_edges.extend(syn_edges)

        logger.debug(
            "[%s] semantic edges: %d lexical, %d embedding, %d synonym",
            current_db, len(lex_edges), len(emb_edges), len(syn_edges),
        )

    logger.info(
        "build_semantic_edges: %d total edges across %d database(s)",
        len(all_edges), len(target_dbs),
    )
    return all_edges


# ---------------------------------------------------------------------------
# Sub-builders
# ---------------------------------------------------------------------------


def _build_lexical_edges(
    pairs: list[tuple[KGNode, KGNode]],
    threshold: float,
) -> list[KGEdge]:
    """
    LEXICAL_SIMILAR edges via Jaccard similarity on tokenised identifier names.

    Tokenisation: snake_case + camelCase split → lowercase token bag.
    Example:
      "hire_date"      → {'hire', 'date'}
      "employment_date" → {'employment', 'date'}
      Jaccard = |{'date'}| / |{'hire', 'date', 'employment'}| = 1/3 = 0.33  → below 0.4, no edge
      "start_date" → {'start', 'date'}
      Jaccard with "hire_date" = 1/3 → no edge
      "date_hired" → {'date', 'hired'}   (if 'hired'≈'hire' — but we don't stem here)

    Single-token identifiers like "id" or "name" are extremely common and
    would produce high Jaccard against every column named "*_id" or "*_name".
    We apply a ``_stopword_penalty`` to reduce weight for generic tokens.
    """
    edges: list[KGEdge] = []

    for a, b in pairs:
        # Use column_name for COLUMN nodes, table_name for TABLE nodes
        name_a = a.column_name if a.node_type == NodeType.COLUMN else a.table_name
        name_b = b.column_name if b.node_type == NodeType.COLUMN else b.table_name

        tokens_a = set(tokenize_name(name_a))
        tokens_b = set(tokenize_name(name_b))

        score = _jaccard(tokens_a, tokens_b)
        if score < threshold:
            continue

        # Penalise pairs where the only shared token is a schema stopword
        # (e.g. "id", "name", "type", "code") — those create noise
        shared = tokens_a & tokens_b
        if shared and shared.issubset(_SCHEMA_STOPWORDS):
            score *= 0.5   # halve the weight; keep edge if still >= threshold
            if score < threshold:
                continue

        weight = min(score, W_LEXICAL_MAX)

        meta = {"tokens_a": sorted(tokens_a), "tokens_b": sorted(tokens_b),
                "jaccard": round(score, 4)}
        edges.extend(_bidirectional(a.node_id, b.node_id, EdgeType.LEXICAL_SIMILAR,
                                    weight, meta))

    return edges


def _build_embedding_edges(
    pairs: list[tuple[KGNode, KGNode]],
    threshold: float,
) -> list[KGEdge]:
    """
    EMBEDDING_SIMILAR edges via cosine similarity of node description embeddings.

    Nodes without embeddings (empty list) are silently skipped.
    Embeddings are stored as ``list[float]`` for JSON compatibility; we convert
    to ``np.ndarray`` here for fast batch computation.

    This is the highest-quality semantic signal — available only after
    ``NodeEnricher`` has run offline.  When embeddings encode both the column
    name and the LLM-generated description + synonyms, this edge type captures
    deep semantic equivalence (e.g. "salary" ↔ "compensation", "wage").
    """
    edges: list[KGEdge] = []

    # Pre-convert to numpy arrays (skip nodes without embeddings)
    def _to_array(node: KGNode) -> Optional[np.ndarray]:
        if not node.embedding:
            return None
        arr = np.array(node.embedding, dtype=np.float32)
        norm = np.linalg.norm(arr)
        return arr / (norm + 1e-10) if norm > 0 else None

    for a, b in pairs:
        arr_a = _to_array(a)
        arr_b = _to_array(b)
        if arr_a is None or arr_b is None:
            continue

        cosine = float(np.dot(arr_a, arr_b))
        # Cosine of unit vectors is in [-1, 1]; clip to [0, 1]
        cosine = max(0.0, cosine)

        if cosine < threshold:
            continue

        weight = min(cosine, W_EMBEDDING_MAX)
        meta = {"cosine": round(cosine, 4)}
        edges.extend(_bidirectional(a.node_id, b.node_id, EdgeType.EMBEDDING_SIMILAR,
                                    weight, meta))

    return edges


def _build_synonym_edges(
    pairs: list[tuple[KGNode, KGNode]],
    min_overlap: int,
) -> list[KGEdge]:
    """
    SYNONYM_MATCH edges via overlap between LLM-generated synonym sets.

    Synonyms are tokenised (snake_case split) before comparison so that
    multi-word synonyms like "hiring date" match tokens from "hire_date".

    Weight = |shared tokens| / max(|tokens_a|, |tokens_b|)
    This normalisation means a node with many synonyms is not penalised for
    also sharing fewer tokens with another node.

    Only available after ``NodeEnricher`` has populated ``KGNode.synonyms``.
    Pairs where both nodes have empty synonym lists are skipped entirely.

    Example
    -------
    Node A synonyms: ["hired", "joining date", "start date", "employment date"]
    Node B synonyms: ["date joined", "entry date", "start date"]
    Shared tokens after tokenisation: {'start', 'date', 'join', ...}
    → High overlap → high weight SYNONYM_MATCH edge
    """
    edges: list[KGEdge] = []

    for a, b in pairs:
        if not a.synonyms and not b.synonyms:
            continue

        # Tokenise all synonyms and flatten into token bags
        tokens_a = _synonym_token_bag(a.synonyms)
        tokens_b = _synonym_token_bag(b.synonyms)

        if not tokens_a or not tokens_b:
            continue

        shared = tokens_a & tokens_b

        # Remove generic stopwords from shared count to avoid spurious matches
        shared -= _SCHEMA_STOPWORDS

        if len(shared) < min_overlap:
            continue

        overlap_score = len(shared) / max(len(tokens_a), len(tokens_b))
        weight = min(overlap_score, W_SYNONYM_MAX)

        meta = {"shared_tokens": sorted(shared), "overlap_score": round(overlap_score, 4)}
        edges.extend(_bidirectional(a.node_id, b.node_id, EdgeType.SYNONYM_MATCH,
                                    weight, meta))

    return edges


# ---------------------------------------------------------------------------
# Similarity utilities
# ---------------------------------------------------------------------------


def _jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity: |A ∩ B| / |A ∪ B|.  Returns 0.0 for empty inputs."""
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def _synonym_token_bag(synonyms: list[str]) -> set[str]:
    """
    Flatten a list of synonym strings into a single set of lowercase tokens.

    "hiring date" → {'hiring', 'date'}
    ["hired", "joining date"] → {'hired', 'joining', 'date'}
    """
    tokens: set[str] = set()
    for syn in synonyms:
        tokens.update(tokenize_name(syn))
    return tokens


def _bidirectional(
    id_a: str,
    id_b: str,
    edge_type: EdgeType,
    weight: float,
    metadata: dict,
) -> list[KGEdge]:
    """Return two directed edges A→B and B→A with the same type and weight."""
    return [
        KGEdge(src_id=id_a, dst_id=id_b, edge_type=edge_type,
               weight=weight, metadata=metadata),
        KGEdge(src_id=id_b, dst_id=id_a, edge_type=edge_type,
               weight=weight, metadata=metadata),
    ]


# ---------------------------------------------------------------------------
# Schema identifier stopwords
# ---------------------------------------------------------------------------

# Tokens so common in schema identifiers that sharing them alone gives
# little information about semantic relatedness.
# Shared "id" between "user_id" and "order_id" is structural (FK), not semantic.
# We reduce (not remove) their contribution rather than hard-filtering.
_SCHEMA_STOPWORDS: frozenset[str] = frozenset({
    # Generic identifiers
    "id", "ids", "key", "keys", "pk", "fk",
    # Generic name/value tokens
    "name", "names", "type", "types", "code", "codes",
    "value", "values", "val", "num", "no",
    # Generic temporal tokens
    "date", "time", "year", "month", "day", "at", "on",
    # Generic status/flag tokens
    "status", "flag", "is", "has",
    # Common abbreviations
    "desc", "info", "data", "detail", "details",
    # Numeric-ish
    "count", "total", "amount", "sum", "avg",
})
