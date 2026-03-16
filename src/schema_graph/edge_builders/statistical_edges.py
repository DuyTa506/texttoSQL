"""
Layer 3 — Statistical Edge Builder.

Derives edges from two independent statistical signals:

  A) SQL Co-occurrence (requires training data)
     Analyses a corpus of SQL queries to count how often schema elements
     appear together.  Three sub-types:

       CO_JOIN       table  ↔ table    — appear in same FROM / JOIN clause
       CO_PREDICATE  column ↔ column   — appear together in WHERE / HAVING
       CO_SELECT     column ↔ column   — appear together in SELECT list

     Weight = normalised co-occurrence frequency (PMI-like, see _pmi_weight).
     Edges are added only when the normalised weight exceeds a threshold.

  B) Value Overlap (requires SQLite DB files)
     For every pair of TEXT columns within the same database, computes the
     Jaccard similarity of their DISTINCT value sets.

       VALUE_OVERLAP  column ↔ column  — share cell values in the live DB

     Weight = Jaccard(values_a, values_b).
     Useful for finding hidden semantic links not expressed in the DDL
     (e.g. orders.ship_city ↔ customers.city both contain "New York").

Design notes
------------
* All edges are within-DB only.
* Both directions (A→B and B→A) are added for every pair.
* sqlparse is used for SQL parsing — already a project dependency.
* sqlite3 is used for value scanning — stdlib, no extra deps.
* The two builders (co-occurrence, value overlap) can be called independently
  via ``build_cooccurrence_edges`` and ``build_value_overlap_edges``, or
  together via the convenience ``build_statistical_edges``.
* SQL parsing is best-effort: unrecognised tokens are silently skipped rather
  than raising an exception, so malformed training queries don't break the build.
"""

from __future__ import annotations

import logging
import math
import re
import sqlite3
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import sqlparse
from sqlparse.sql import Identifier, IdentifierList, Where
from sqlparse.tokens import Keyword, DML, Punctuation

from src.schema_graph.graph_types import EdgeType, KGEdge, KGNode, NodeType
from src.schema_graph.edge_builders.structural_edges import tokenize_name

if TYPE_CHECKING:
    from src.schema_graph.graph_builder import SchemaGraph
    from src.schema.models import Example

logger = logging.getLogger(__name__)

# ── Thresholds and weight caps ────────────────────────────────────────────────

DEFAULT_JOIN_THRESHOLD: float = 0.10          # min normalised co-occurrence for CO_JOIN
DEFAULT_PREDICATE_THRESHOLD: float = 0.10     # min normalised co-occurrence for CO_PREDICATE
DEFAULT_SELECT_THRESHOLD: float = 0.10        # min normalised co-occurrence for CO_SELECT
DEFAULT_VALUE_OVERLAP_THRESHOLD: float = 0.20 # min Jaccard for VALUE_OVERLAP
DEFAULT_MAX_DISTINCT_VALUES: int = 500        # skip high-cardinality columns

W_CO_JOIN_MAX: float = 0.50                     # v3: reduced from 0.80 to limit noise
W_CO_PREDICATE_MAX: float = 0.75
W_CO_SELECT_MAX: float = 0.70
W_VALUE_OVERLAP_MAX: float = 0.85

# v3: Inferred FK from VALUE_OVERLAP (for DBs lacking explicit FK declarations)
W_INFERRED_FK: float = 0.80
INFERRED_FK_JACCARD_THRESHOLD: float = 0.50      # min Jaccard to infer FK
INFERRED_FK_MIN_SHARED: int = 3                   # min shared values to infer FK


# ---------------------------------------------------------------------------
# Public API — convenience wrapper
# ---------------------------------------------------------------------------


def build_statistical_edges(
    graph: "SchemaGraph",
    examples: list["Example"],
    *,
    db_dir: str = "",
    db_id: Optional[str] = None,
    join_threshold: float = DEFAULT_JOIN_THRESHOLD,
    predicate_threshold: float = DEFAULT_PREDICATE_THRESHOLD,
    select_threshold: float = DEFAULT_SELECT_THRESHOLD,
    value_overlap_threshold: float = DEFAULT_VALUE_OVERLAP_THRESHOLD,
    max_distinct_values: int = DEFAULT_MAX_DISTINCT_VALUES,
) -> list[KGEdge]:
    """
    Build all Layer-3 (statistical) edges for the graph.

    Parameters
    ----------
    graph:
        The ``SchemaGraph`` whose nodes are inspected.  Not mutated — caller
        adds returned edges via ``graph.add_edges()``.
    examples:
        Training examples (``Example`` objects with ``db_id`` and ``query``
        fields).  Used for co-occurrence counting.  Can be an empty list if
        only value-overlap edges are desired.
    db_dir:
        Root directory containing SQLite files, structured as
        ``{db_dir}/{db_id}/{db_id}.sqlite``.
        Required for VALUE_OVERLAP edges.  Pass ``""`` to skip.
    db_id:
        If given, restrict to a single database.
    join_threshold / predicate_threshold / select_threshold:
        Minimum normalised weight to emit the corresponding co-occurrence edge.
    value_overlap_threshold:
        Minimum Jaccard to emit a VALUE_OVERLAP edge.
    max_distinct_values:
        TEXT columns with more DISTINCT values than this are skipped during
        value-overlap scanning (high-cardinality columns are uninformative).

    Returns
    -------
    list[KGEdge]
        All Layer-3 edges (both directions per pair).
    """
    all_edges: list[KGEdge] = []

    # ── A: Co-occurrence edges ────────────────────────────────────────────────
    if examples:
        co_edges = build_cooccurrence_edges(
            graph, examples,
            db_id=db_id,
            join_threshold=join_threshold,
            predicate_threshold=predicate_threshold,
            select_threshold=select_threshold,
        )
        all_edges.extend(co_edges)

    # ── B: Value overlap edges ────────────────────────────────────────────────
    if db_dir:
        val_edges = build_value_overlap_edges(
            graph,
            db_dir=db_dir,
            db_id=db_id,
            threshold=value_overlap_threshold,
            max_distinct_values=max_distinct_values,
        )
        all_edges.extend(val_edges)

    logger.info(
        "build_statistical_edges: %d total edges (co-occurrence + value overlap)",
        len(all_edges),
    )
    return all_edges


# ---------------------------------------------------------------------------
# A: SQL Co-occurrence edges
# ---------------------------------------------------------------------------


def build_cooccurrence_edges(
    graph: "SchemaGraph",
    examples: list["Example"],
    *,
    db_id: Optional[str] = None,
    join_threshold: float = DEFAULT_JOIN_THRESHOLD,
    predicate_threshold: float = DEFAULT_PREDICATE_THRESHOLD,
    select_threshold: float = DEFAULT_SELECT_THRESHOLD,
) -> list[KGEdge]:
    """
    Build CO_JOIN, CO_PREDICATE and CO_SELECT edges from a SQL corpus.

    The weight of a co-occurrence edge between A and B is computed as a
    PMI-inspired normalised frequency:

        weight(A, B) = count(A, B) / (count(A) + count(B) - count(A, B))

    This is the Jaccard coefficient on query membership — the fraction of
    queries in which BOTH A and B appear, relative to queries in which
    EITHER appears.  It is symmetric and bounded to [0, 1].

    Parameters
    ----------
    graph:
        Graph providing node metadata (to resolve column identifiers to node_ids).
    examples:
        Training ``Example`` objects.  Only examples whose ``db_id`` matches
        a database in the graph are processed.
    db_id:
        Restrict to one database.  If None, process all.
    join_threshold / predicate_threshold / select_threshold:
        Minimum weight to emit an edge.
    """
    # Determine which db_ids to process
    if db_id is not None:
        target_dbs = {db_id}
    else:
        target_dbs = {n.db_id for n in graph.nodes.values()
                      if n.node_type != NodeType.DATABASE}

    # Build a lookup: db_id → {table_name → node_id, table.col → node_id}
    schema_lookup = _build_schema_lookup(graph, target_dbs)

    # Counters per DB
    # table_pair_counts[(db, t1, t2)] = set of query indices where both appear
    table_single:  dict[tuple[str, str], set[int]] = defaultdict(set)
    table_pairs:   dict[tuple[str, str, str], set[int]] = defaultdict(set)
    col_pred_single: dict[tuple[str, str], set[int]] = defaultdict(set)
    col_pred_pairs:  dict[tuple[str, str, str], set[int]] = defaultdict(set)
    col_sel_single:  dict[tuple[str, str], set[int]] = defaultdict(set)
    col_sel_pairs:   dict[tuple[str, str, str], set[int]] = defaultdict(set)

    for idx, ex in enumerate(examples):
        if ex.db_id not in target_dbs:
            continue
        db = ex.db_id

        try:
            tables, sel_cols, pred_cols = _parse_sql(ex.query, db, schema_lookup)
        except Exception:
            logger.debug("Failed to parse SQL for example %d (db=%s) — skipping", idx, db)
            continue

        # Table join co-occurrence
        for t in tables:
            table_single[(db, t)].add(idx)
        for t1, t2 in combinations(sorted(tables), 2):
            table_pairs[(db, t1, t2)].add(idx)

        # Predicate column co-occurrence
        for c in pred_cols:
            col_pred_single[(db, c)].add(idx)
        for c1, c2 in combinations(sorted(pred_cols), 2):
            col_pred_pairs[(db, c1, c2)].add(idx)

        # Select column co-occurrence
        for c in sel_cols:
            col_sel_single[(db, c)].add(idx)
        for c1, c2 in combinations(sorted(sel_cols), 2):
            col_sel_pairs[(db, c1, c2)].add(idx)

    edges: list[KGEdge] = []

    # ── CO_JOIN ───────────────────────────────────────────────────────────────
    for (db, t1, t2), pair_queries in table_pairs.items():
        w = _jaccard_weight(
            table_single[(db, t1)],
            table_single[(db, t2)],
            pair_queries,
        )
        if w < join_threshold:
            continue
        n1 = f"{db}.{t1}"
        n2 = f"{db}.{t2}"
        if n1 not in graph.nodes or n2 not in graph.nodes:
            continue
        weight = min(w, W_CO_JOIN_MAX)
        meta = {"count": len(pair_queries), "jaccard": round(w, 4)}
        edges.extend(_bidir(n1, n2, EdgeType.CO_JOIN, weight, meta))

    # ── CO_PREDICATE ──────────────────────────────────────────────────────────
    for (db, c1, c2), pair_queries in col_pred_pairs.items():
        w = _jaccard_weight(
            col_pred_single[(db, c1)],
            col_pred_single[(db, c2)],
            pair_queries,
        )
        if w < predicate_threshold:
            continue
        n1 = schema_lookup.get(db, {}).get(c1)
        n2 = schema_lookup.get(db, {}).get(c2)
        if not n1 or not n2 or n1 not in graph.nodes or n2 not in graph.nodes:
            continue
        weight = min(w, W_CO_PREDICATE_MAX)
        meta = {"count": len(pair_queries), "jaccard": round(w, 4)}
        edges.extend(_bidir(n1, n2, EdgeType.CO_PREDICATE, weight, meta))

    # ── CO_SELECT ─────────────────────────────────────────────────────────────
    for (db, c1, c2), pair_queries in col_sel_pairs.items():
        w = _jaccard_weight(
            col_sel_single[(db, c1)],
            col_sel_single[(db, c2)],
            pair_queries,
        )
        if w < select_threshold:
            continue
        n1 = schema_lookup.get(db, {}).get(c1)
        n2 = schema_lookup.get(db, {}).get(c2)
        if not n1 or not n2 or n1 not in graph.nodes or n2 not in graph.nodes:
            continue
        weight = min(w, W_CO_SELECT_MAX)
        meta = {"count": len(pair_queries), "jaccard": round(w, 4)}
        edges.extend(_bidir(n1, n2, EdgeType.CO_SELECT, weight, meta))

    logger.info(
        "build_cooccurrence_edges: %d CO_JOIN + %d CO_PREDICATE + %d CO_SELECT "
        "(from %d examples, %d databases)",
        sum(1 for e in edges if e.edge_type == EdgeType.CO_JOIN) // 2,
        sum(1 for e in edges if e.edge_type == EdgeType.CO_PREDICATE) // 2,
        sum(1 for e in edges if e.edge_type == EdgeType.CO_SELECT) // 2,
        len(examples),
        len(target_dbs),
    )
    return edges


# ---------------------------------------------------------------------------
# B: Value overlap edges
# ---------------------------------------------------------------------------


def build_value_overlap_edges(
    graph: "SchemaGraph",
    *,
    db_dir: str,
    db_id: Optional[str] = None,
    threshold: float = DEFAULT_VALUE_OVERLAP_THRESHOLD,
    max_distinct_values: int = DEFAULT_MAX_DISTINCT_VALUES,
) -> list[KGEdge]:
    """
    Build VALUE_OVERLAP edges by comparing DISTINCT cell values between column pairs.

    Only TEXT / VARCHAR columns are scanned — numeric columns rarely share
    meaningful cell values.  High-cardinality columns (> ``max_distinct_values``
    distinct entries) are skipped to avoid O(N²) comparisons on free-text data.

    Parameters
    ----------
    graph:
        Graph providing column node metadata.
    db_dir:
        Root directory containing SQLite files.  Layout expected:
        ``{db_dir}/{db_id}/{db_id}.sqlite``
    db_id:
        Restrict to one database.
    threshold:
        Minimum Jaccard similarity of value sets to emit an edge.
    max_distinct_values:
        Skip columns with more than this many DISTINCT values.
    """
    if db_id is not None:
        target_dbs = {db_id}
    else:
        target_dbs = {n.db_id for n in graph.nodes.values()
                      if n.node_type != NodeType.DATABASE}

    edges: list[KGEdge] = []

    for current_db in sorted(target_dbs):
        db_path = _resolve_db_path(db_dir, current_db)
        if db_path is None:
            logger.debug("SQLite file not found for db=%s — skipping value overlap", current_db)
            continue

        # Collect TEXT column nodes for this DB
        text_col_nodes = [
            n for n in graph.nodes.values()
            if n.db_id == current_db
            and n.node_type == NodeType.COLUMN
            and _is_text_dtype(n.dtype)
        ]

        if len(text_col_nodes) < 2:
            continue

        # Fetch DISTINCT values per column from SQLite
        col_values: dict[str, frozenset[str]] = {}
        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            for node in text_col_nodes:
                vals = _fetch_distinct_values(
                    conn, node.table_name, node.column_name, max_distinct_values
                )
                if vals:
                    col_values[node.node_id] = frozenset(v.strip().lower() for v in vals if v)
            conn.close()
        except Exception as exc:
            logger.warning("Error reading SQLite for db=%s: %s", current_db, exc)
            continue

        # Pairwise Jaccard on value sets
        node_ids = list(col_values.keys())
        # Track existing FK edges to avoid duplicating as INFERRED_FK
        existing_fk_pairs: set[frozenset[str]] = set()
        for _, _, attrs in graph.G.edges(data=True):
            edge_data = attrs.get("data")
            if edge_data and edge_data.edge_type in (EdgeType.FOREIGN_KEY, EdgeType.FOREIGN_KEY_REV):
                existing_fk_pairs.add(frozenset([edge_data.src_id, edge_data.dst_id]))

        # Track inferred table-FK pairs for deduplication
        inferred_table_fk_pairs: set[frozenset[str]] = set()

        for id_a, id_b in combinations(node_ids, 2):
            vals_a = col_values[id_a]
            vals_b = col_values[id_b]

            if not vals_a or not vals_b:
                continue

            j = _jaccard_sets(vals_a, vals_b)
            if j < threshold:
                continue

            # Skip trivial overlap: very small sets can produce high Jaccard
            # by chance (e.g. both have only "N/A" or "Unknown")
            shared = vals_a & vals_b
            if len(shared) < 2:
                continue

            weight = min(j, W_VALUE_OVERLAP_MAX)
            meta = {
                "jaccard": round(j, 4),
                "shared_count": len(shared),
                "sample_shared": sorted(shared)[:5],
            }
            edges.extend(_bidir(id_a, id_b, EdgeType.VALUE_OVERLAP, weight, meta))

            # v3: Emit INFERRED_FK edges for high-overlap column pairs
            if (j >= INFERRED_FK_JACCARD_THRESHOLD
                    and len(shared) >= INFERRED_FK_MIN_SHARED
                    and frozenset([id_a, id_b]) not in existing_fk_pairs):
                node_a = graph.nodes.get(id_a)
                node_b = graph.nodes.get(id_b)
                if node_a and node_b and node_a.table_name != node_b.table_name:
                    inferred_meta = {
                        "source": "value_overlap",
                        "jaccard": round(j, 4),
                        "shared_count": len(shared),
                    }
                    # Column-level INFERRED_FK
                    edges.extend(_bidir(id_a, id_b, EdgeType.INFERRED_FK, W_INFERRED_FK, inferred_meta))

                    # Table-level TABLE_FK (deduplicated)
                    tbl_a_id = f"{current_db}.{node_a.table_name}"
                    tbl_b_id = f"{current_db}.{node_b.table_name}"
                    tbl_pair = frozenset([tbl_a_id, tbl_b_id])
                    if tbl_pair not in inferred_table_fk_pairs:
                        inferred_table_fk_pairs.add(tbl_pair)
                        tbl_meta = {
                            "source": "value_overlap",
                            "jaccard": round(j, 4),
                            "shared_count": len(shared),
                            "inferred_from": [id_a, id_b],
                        }
                        edges.extend(_bidir(tbl_a_id, tbl_b_id, EdgeType.TABLE_FK, W_INFERRED_FK, tbl_meta))

    inferred_fk_count = sum(1 for e in edges if e.edge_type == EdgeType.INFERRED_FK) // 2
    inferred_table_fk_count = sum(
        1 for e in edges if e.edge_type == EdgeType.TABLE_FK
    ) // 2
    logger.info(
        "build_value_overlap_edges: %d VALUE_OVERLAP + %d INFERRED_FK + %d TABLE_FK "
        "edges across %d database(s)",
        (len(edges) - inferred_fk_count * 2 - inferred_table_fk_count * 2) // 2,
        inferred_fk_count,
        inferred_table_fk_count,
        len(target_dbs),
    )
    return edges


# ---------------------------------------------------------------------------
# SQL parsing helpers
# ---------------------------------------------------------------------------


def _parse_sql(
    sql: str,
    db: str,
    schema_lookup: dict[str, dict[str, str]],
) -> tuple[set[str], set[str], set[str]]:
    """
    Parse a SQL query and extract:
      - tables: table names appearing in FROM / JOIN
      - sel_cols: fully-qualified column keys appearing in SELECT
      - pred_cols: fully-qualified column keys appearing in WHERE / HAVING / ON

    All names are lower-cased.  Aliases are resolved where possible.
    Returns three sets of strings.

    The parser is best-effort — SQL with unusual formatting may miss some
    references.  This is acceptable: statistical co-occurrence is a soft signal
    and missed extractions simply reduce counts rather than producing errors.
    """
    sql_lower = sql.strip().lower()
    db_schema = schema_lookup.get(db, {})

    # ── Extract table names ────────────────────────────────────────────────────
    tables = _extract_tables(sql_lower)

    # Build alias → real table mapping from what we know in the schema
    alias_map: dict[str, str] = {}
    for t in tables:
        if t in {n.split(".")[0] for n in db_schema}:
            alias_map[t] = t
    # Simple alias pattern: "FROM table_name alias" or "JOIN table_name alias"
    alias_pattern = re.compile(
        r"(?:from|join)\s+(\w+)\s+(?:as\s+)?(\w+)(?:\s|,|$)", re.IGNORECASE
    )
    for m in alias_pattern.finditer(sql_lower):
        real_tbl, alias = m.group(1), m.group(2)
        if alias not in {"where", "on", "set", "inner", "outer", "left", "right", "join"}:
            alias_map[alias] = real_tbl

    # ── Extract column references ──────────────────────────────────────────────
    # Pattern: table.column or alias.column
    qualified = re.compile(r"\b(\w+)\.(\w+)\b")
    # Unqualified column: word that matches a known column name
    known_columns: dict[str, list[str]] = defaultdict(list)
    for key in db_schema:
        # key can be "table.col" or just "table"
        if "." in key:
            _, col = key.rsplit(".", 1)
            known_columns[col].append(key)

    sel_cols: set[str] = set()
    pred_cols: set[str] = set()

    # Determine rough clause boundaries by finding keyword positions
    select_end = _find_keyword_pos(sql_lower, ["from"])
    where_start = _find_keyword_pos(sql_lower, ["where", "having"])

    select_part = sql_lower[:select_end] if select_end else ""
    where_part = sql_lower[where_start:] if where_start else ""

    def _resolve(tbl_token: str, col_token: str) -> Optional[str]:
        real_tbl = alias_map.get(tbl_token, tbl_token)
        key = f"{real_tbl}.{col_token}"
        if key in db_schema:
            return key
        return None

    # SELECT columns
    for m in qualified.finditer(select_part):
        resolved = _resolve(m.group(1), m.group(2))
        if resolved:
            sel_cols.add(resolved)

    # WHERE / HAVING / ON columns — scan full query for predicate context
    for m in qualified.finditer(where_part):
        resolved = _resolve(m.group(1), m.group(2))
        if resolved:
            pred_cols.add(resolved)

    # Also catch unqualified columns in WHERE if they uniquely identify a table
    for m in re.finditer(r"\b(\w+)\b", where_part):
        col = m.group(1)
        if col in known_columns and len(known_columns[col]) == 1:
            pred_cols.add(known_columns[col][0])

    return tables, sel_cols, pred_cols


def _extract_tables(sql_lower: str) -> set[str]:
    """Extract table names from FROM and JOIN clauses using regex."""
    tables: set[str] = set()
    # Match: FROM table, FROM table alias, JOIN table, JOIN table alias
    pattern = re.compile(
        r"(?:from|join)\s+(\w+)(?:\s+(?:as\s+)?(\w+))?",
        re.IGNORECASE,
    )
    skip = {"where", "on", "set", "inner", "outer", "left", "right",
            "cross", "full", "natural", "join", "select", "with"}
    for m in pattern.finditer(sql_lower):
        name = m.group(1).lower()
        if name not in skip:
            tables.add(name)
    return tables


def _find_keyword_pos(sql: str, keywords: list[str]) -> Optional[int]:
    """Return the position of the first occurrence of any keyword (word-boundary)."""
    for kw in keywords:
        m = re.search(rf"\b{kw}\b", sql, re.IGNORECASE)
        if m:
            return m.start()
    return None


def _build_schema_lookup(
    graph: "SchemaGraph",
    db_ids: set[str],
) -> dict[str, dict[str, str]]:
    """
    Build a per-DB lookup: {db_id: {"table.col" → node_id, "table" → node_id}}

    Used during SQL parsing to resolve extracted names to graph node IDs.
    """
    lookup: dict[str, dict[str, str]] = defaultdict(dict)
    for node in graph.nodes.values():
        if node.db_id not in db_ids:
            continue
        if node.node_type == NodeType.TABLE:
            lookup[node.db_id][node.table_name.lower()] = node.node_id
        elif node.node_type == NodeType.COLUMN:
            key = f"{node.table_name.lower()}.{node.column_name.lower()}"
            lookup[node.db_id][key] = node.node_id
    return dict(lookup)


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------


def _resolve_db_path(db_dir: str, db_id: str) -> Optional[Path]:
    """
    Resolve ``{db_dir}/{db_id}/{db_id}.sqlite``.
    Falls back to ``{db_dir}/{db_id}.sqlite`` (flat layout).
    Returns None if neither exists.
    """
    if not db_dir:
        return None
    base = Path(db_dir)
    nested = base / db_id / f"{db_id}.sqlite"
    flat = base / f"{db_id}.sqlite"
    if nested.exists():
        return nested
    if flat.exists():
        return flat
    return None


def _fetch_distinct_values(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    max_values: int,
) -> list[str]:
    """
    Fetch up to *max_values* DISTINCT non-NULL values from *table.column*.
    Returns an empty list if the query fails (e.g. table doesn't exist yet).
    """
    try:
        cur = conn.cursor()
        # Use parameterised table/column names via format (sqlite3 doesn't
        # support ? for identifiers).  Table and column names come from the
        # graph (trusted schema data), not user input.
        cur.execute(
            f'SELECT DISTINCT "{column}" FROM "{table}" '  # noqa: S608
            f'WHERE "{column}" IS NOT NULL LIMIT {max_values}'
        )
        return [str(row[0]) for row in cur.fetchall()]
    except Exception:
        return []


def _is_text_dtype(dtype: str) -> bool:
    """Return True for TEXT-like column types; False for numeric / date types."""
    if not dtype:
        return True  # unknown dtype — include conservatively
    dt = dtype.upper()
    text_types = {"TEXT", "VARCHAR", "CHAR", "STRING", "NVARCHAR", "NCHAR", "CLOB"}
    numeric_types = {"INTEGER", "INT", "REAL", "FLOAT", "DOUBLE",
                     "NUMERIC", "DECIMAL", "BOOLEAN", "BLOB"}
    date_types = {"DATE", "DATETIME", "TIMESTAMP", "TIME", "YEAR"}
    if any(dt.startswith(t) for t in text_types):
        return True
    if any(dt.startswith(t) for t in numeric_types | date_types):
        return False
    return True  # default include


# ---------------------------------------------------------------------------
# Weight / similarity utilities
# ---------------------------------------------------------------------------


def _jaccard_weight(
    set_a: set[int],
    set_b: set[int],
    pair_set: set[int],
) -> float:
    """
    Jaccard coefficient on query membership sets.

    J(A, B) = |A ∩ B| / |A ∪ B|
            = |pair_set| / (|set_a| + |set_b| - |pair_set|)

    Using precomputed pair_set avoids recomputing the intersection.
    """
    union_size = len(set_a) + len(set_b) - len(pair_set)
    if union_size == 0:
        return 0.0
    return len(pair_set) / union_size


def _jaccard_sets(a: frozenset, b: frozenset) -> float:
    """Jaccard similarity between two frozensets."""
    if not a and not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


def _bidir(
    id_a: str,
    id_b: str,
    edge_type: EdgeType,
    weight: float,
    metadata: dict,
) -> list[KGEdge]:
    """Return A→B and B→A edges."""
    return [
        KGEdge(src_id=id_a, dst_id=id_b, edge_type=edge_type,
               weight=weight, metadata=metadata),
        KGEdge(src_id=id_b, dst_id=id_a, edge_type=edge_type,
               weight=weight, metadata=metadata),
    ]
