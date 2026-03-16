"""
Layer 1 — Structural Edge Builder.

Derives edges that are 100% deterministic from the DDL (CREATE TABLE statements
and FOREIGN KEY declarations). No LLM calls, no heuristics, zero cost.

Edges produced:
  DB_HAS_TABLE        db_node      → table_node      (ownership)
  TABLE_HAS_COLUMN    table_node   → column_node      (membership)
  COLUMN_BELONGS_TO   column_node  → table_node       (reverse membership)
  PRIMARY_KEY_OF      pk_col_node  → table_node       (PK annotation)
  FOREIGN_KEY         src_col_node → dst_col_node     (FK constraint, canonical direction)
  FOREIGN_KEY_REV     dst_col_node → src_col_node     (FK reverse, for bidirectional traversal)

All structural edges have weight = 1.0 except:
  FOREIGN_KEY / FOREIGN_KEY_REV  →  0.95
    (slightly lower so PPR mildly prefers direct table membership over cross-table jumps)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.schema_graph.graph_types import EdgeType, KGEdge, KGNode, NodeType

if TYPE_CHECKING:
    from src.schema.models import Database

logger = logging.getLogger(__name__)

# ── Edge weight constants ──────────────────────────────────────────────────────

W_OWNERSHIP = 1.0          # DB_HAS_TABLE
W_MEMBERSHIP = 1.0         # TABLE_HAS_COLUMN / COLUMN_BELONGS_TO
W_PRIMARY_KEY = 1.0        # PRIMARY_KEY_OF
W_FOREIGN_KEY = 0.95       # FOREIGN_KEY / FOREIGN_KEY_REV
W_TABLE_FK = 0.85          # TABLE_FK / TABLE_FK_REV (table-level FK shortcut, v3)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_structural_edges(
    db: "Database",
    nodes: dict[str, KGNode],
) -> list[KGEdge]:
    """
    Build all Layer-1 (structural) edges for *db* using the pre-built *nodes* dict.

    Parameters
    ----------
    db:
        The ``Database`` object whose schema drives edge construction.
    nodes:
        Mapping of ``node_id → KGNode`` already constructed for this database
        (typically produced by ``SchemaGraphBuilder._build_nodes()``).

    Returns
    -------
    list[KGEdge]
        All structural edges for the database, in deterministic order:
        ownership → membership → PK annotations → FK constraints.
    """
    edges: list[KGEdge] = []

    db_node_id = db.db_id

    # ── 1. DB → TABLE ownership ───────────────────────────────────────────────
    for table in db.tables:
        table_node_id = f"{db.db_id}.{table.name}"
        if table_node_id not in nodes:
            logger.warning("Table node missing: %s — skipping ownership edge", table_node_id)
            continue

        edges.append(KGEdge(
            src_id=db_node_id,
            dst_id=table_node_id,
            edge_type=EdgeType.DB_HAS_TABLE,
            weight=W_OWNERSHIP,
        ))

    # ── 2. TABLE ↔ COLUMN membership ──────────────────────────────────────────
    for table in db.tables:
        table_node_id = f"{db.db_id}.{table.name}"
        if table_node_id not in nodes:
            continue

        for col in table.columns:
            col_node_id = f"{db.db_id}.{table.name}.{col.name}"
            if col_node_id not in nodes:
                logger.warning(
                    "Column node missing: %s — skipping membership edge", col_node_id
                )
                continue

            # TABLE → COLUMN
            edges.append(KGEdge(
                src_id=table_node_id,
                dst_id=col_node_id,
                edge_type=EdgeType.TABLE_HAS_COLUMN,
                weight=W_MEMBERSHIP,
            ))

            # COLUMN → TABLE (reverse, for upward traversal)
            edges.append(KGEdge(
                src_id=col_node_id,
                dst_id=table_node_id,
                edge_type=EdgeType.COLUMN_BELONGS_TO,
                weight=W_MEMBERSHIP,
            ))

            # ── 3. PK annotation ──────────────────────────────────────────────
            if col.primary_key:
                edges.append(KGEdge(
                    src_id=col_node_id,
                    dst_id=table_node_id,
                    edge_type=EdgeType.PRIMARY_KEY_OF,
                    weight=W_PRIMARY_KEY,
                    metadata={"pk_column": col.name},
                ))

    # ── 4. FK constraints ─────────────────────────────────────────────────────
    fk_columns: set[str] = set()   # track which column nodes are part of a FK

    for fk in db.foreign_keys:
        src_col_id = f"{db.db_id}.{fk.from_table}.{fk.from_column}"
        dst_col_id = f"{db.db_id}.{fk.to_table}.{fk.to_column}"

        if src_col_id not in nodes:
            logger.warning(
                "FK source node missing: %s (FK: %s.%s → %s.%s) — skipping",
                src_col_id, fk.from_table, fk.from_column, fk.to_table, fk.to_column,
            )
            continue
        if dst_col_id not in nodes:
            logger.warning(
                "FK target node missing: %s (FK: %s.%s → %s.%s) — skipping",
                dst_col_id, fk.from_table, fk.from_column, fk.to_table, fk.to_column,
            )
            continue

        fk_metadata = {
            "from_table": fk.from_table,
            "from_column": fk.from_column,
            "to_table": fk.to_table,
            "to_column": fk.to_column,
        }

        # Canonical direction: FK source → FK target
        edges.append(KGEdge(
            src_id=src_col_id,
            dst_id=dst_col_id,
            edge_type=EdgeType.FOREIGN_KEY,
            weight=W_FOREIGN_KEY,
            metadata=fk_metadata,
        ))

        # Reverse direction: FK target → FK source (bidirectional traversal)
        edges.append(KGEdge(
            src_id=dst_col_id,
            dst_id=src_col_id,
            edge_type=EdgeType.FOREIGN_KEY_REV,
            weight=W_FOREIGN_KEY,
            metadata=fk_metadata,
        ))

        fk_columns.add(src_col_id)
        fk_columns.add(dst_col_id)

    # ── 5. Mark is_fk flag on column nodes (mutates nodes dict in-place) ──────
    for col_id in fk_columns:
        if col_id in nodes:
            nodes[col_id].is_fk = True

    # ── 5b. TABLE_FK shortcut edges (v3) ────────────────────────────────────
    # Add direct Table→Table edges for each FK relationship so PPR can hop
    # between tables in 1 step instead of the 4-hop Table→Col→Col→Table path.
    # Deduplicate: multiple FK columns between the same table pair → one edge.
    table_fk_pairs: dict[tuple[str, str], list[dict]] = {}
    for fk in db.foreign_keys:
        from_tbl_id = f"{db.db_id}.{fk.from_table}"
        to_tbl_id = f"{db.db_id}.{fk.to_table}"
        if from_tbl_id not in nodes or to_tbl_id not in nodes:
            continue
        pair_key = (from_tbl_id, to_tbl_id)
        if pair_key not in table_fk_pairs:
            table_fk_pairs[pair_key] = []
        table_fk_pairs[pair_key].append({
            "from_column": fk.from_column,
            "to_column": fk.to_column,
        })

    for (from_tbl_id, to_tbl_id), fk_col_list in table_fk_pairs.items():
        meta = {"fk_columns": fk_col_list}
        edges.append(KGEdge(
            src_id=from_tbl_id,
            dst_id=to_tbl_id,
            edge_type=EdgeType.TABLE_FK,
            weight=W_TABLE_FK,
            metadata=meta,
        ))
        edges.append(KGEdge(
            src_id=to_tbl_id,
            dst_id=from_tbl_id,
            edge_type=EdgeType.TABLE_FK_REV,
            weight=W_TABLE_FK,
            metadata=meta,
        ))

    logger.debug(
        "[%s] structural edges: %d ownership, %d membership, %d FK, %d TABLE_FK",
        db.db_id,
        sum(1 for e in edges if e.edge_type == EdgeType.DB_HAS_TABLE),
        sum(1 for e in edges if e.edge_type in (EdgeType.TABLE_HAS_COLUMN, EdgeType.COLUMN_BELONGS_TO)),
        sum(1 for e in edges if e.edge_type in (EdgeType.FOREIGN_KEY, EdgeType.FOREIGN_KEY_REV)),
        sum(1 for e in edges if e.edge_type in (EdgeType.TABLE_FK, EdgeType.TABLE_FK_REV)),
    )

    return edges


# ---------------------------------------------------------------------------
# Utility helpers (reused by semantic/statistical builders)
# ---------------------------------------------------------------------------


def tokenize_name(name: str) -> list[str]:
    """
    Split a snake_case or camelCase identifier into lowercase tokens.

    Examples
    --------
    >>> tokenize_name("hire_date")
    ['hire', 'date']
    >>> tokenize_name("employeeID")
    ['employee', 'i', 'd']   # camelCase split is coarse; snake_case is the common case
    >>> tokenize_name("dept_name")
    ['dept', 'name']
    """
    import re
    # Insert space before uppercase letters (camelCase)
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    # Split on underscores and whitespace
    tokens = re.split(r"[_\s]+", name.lower())
    return [t for t in tokens if t]


def column_full_id(db_id: str, table_name: str, column_name: str) -> str:
    """Canonical node_id for a column node."""
    return f"{db_id}.{table_name}.{column_name}"


def table_full_id(db_id: str, table_name: str) -> str:
    """Canonical node_id for a table node."""
    return f"{db_id}.{table_name}"
