"""
Core type definitions for the Schema Knowledge Graph.

KGNode    — a node in the graph (DATABASE | TABLE | COLUMN)
KGEdge    — a typed, weighted directed edge between two nodes
NodeType  — enum of node categories
EdgeType  — enum of edge categories, grouped by layer
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class NodeType(str, Enum):
    """Semantic category of a graph node."""

    DATABASE = "database"
    TABLE = "table"
    COLUMN = "column"


class EdgeType(str, Enum):
    """
    Typed edge categories, grouped by construction layer.

    Layer 1 — Structural (deterministic, from DDL)
      TABLE_HAS_COLUMN        table  → column   (membership)
      COLUMN_BELONGS_TO       column → table    (reverse membership)
      FOREIGN_KEY             column → column   (FK constraint)
      FOREIGN_KEY_REV         column → column   (reverse FK, bidirectional access)
      PRIMARY_KEY_OF          column → table    (PK annotation)
      DB_HAS_TABLE            db     → table    (ownership)

    Layer 2 — Semantic (derived from names / LLM descriptions)
      LEXICAL_SIMILAR         column ↔ column   (token overlap in snake-case names)
      EMBEDDING_SIMILAR       column ↔ column   (cosine similarity of descriptions)
      SYNONYM_MATCH           column ↔ column   (synonym set intersection)

    Layer 3 — Statistical (from training SQL + live DB)
      CO_JOIN                 table  ↔ table    (appear in same FROM/JOIN)
      CO_PREDICATE            column ↔ column   (appear in same WHERE clause)
      CO_SELECT               column ↔ column   (appear in same SELECT list)
      VALUE_OVERLAP           column ↔ column   (shared DISTINCT values in SQLite)
    """

    # ── Layer 1 ───────────────────────────────────────────────────────────────
    TABLE_HAS_COLUMN = "table_has_column"
    COLUMN_BELONGS_TO = "column_belongs_to"
    FOREIGN_KEY = "foreign_key"
    FOREIGN_KEY_REV = "foreign_key_rev"
    PRIMARY_KEY_OF = "primary_key_of"
    DB_HAS_TABLE = "db_has_table"
    TABLE_FK = "table_fk"                    # Table→Table shortcut for FK (v3)
    TABLE_FK_REV = "table_fk_rev"            # Reverse table-level FK shortcut (v3)
    INFERRED_FK = "inferred_fk"              # Column→Column inferred from value overlap (v3)
    INFERRED_FK_REV = "inferred_fk_rev"      # Reverse inferred FK (v3)

    # ── Layer 2 ───────────────────────────────────────────────────────────────
    LEXICAL_SIMILAR = "lexical_similar"
    EMBEDDING_SIMILAR = "embedding_similar"
    SYNONYM_MATCH = "synonym_match"

    # ── Layer 3 ───────────────────────────────────────────────────────────────
    CO_JOIN = "co_join"
    CO_PREDICATE = "co_predicate"
    CO_SELECT = "co_select"
    VALUE_OVERLAP = "value_overlap"


# Structural edge types (Layer 1) — used to filter by layer
STRUCTURAL_EDGE_TYPES: frozenset[EdgeType] = frozenset({
    EdgeType.TABLE_HAS_COLUMN,
    EdgeType.COLUMN_BELONGS_TO,
    EdgeType.FOREIGN_KEY,
    EdgeType.FOREIGN_KEY_REV,
    EdgeType.PRIMARY_KEY_OF,
    EdgeType.DB_HAS_TABLE,
    EdgeType.TABLE_FK,
    EdgeType.TABLE_FK_REV,
    EdgeType.INFERRED_FK,
    EdgeType.INFERRED_FK_REV,
})

SEMANTIC_EDGE_TYPES: frozenset[EdgeType] = frozenset({
    EdgeType.LEXICAL_SIMILAR,
    EdgeType.EMBEDDING_SIMILAR,
    EdgeType.SYNONYM_MATCH,
})

STATISTICAL_EDGE_TYPES: frozenset[EdgeType] = frozenset({
    EdgeType.CO_JOIN,
    EdgeType.CO_PREDICATE,
    EdgeType.CO_SELECT,
    EdgeType.VALUE_OVERLAP,
})


# ---------------------------------------------------------------------------
# KGNode
# ---------------------------------------------------------------------------


@dataclass
class KGNode:
    """
    A single node in the Schema Knowledge Graph.

    node_id is globally unique within a graph, formatted as:
      DATABASE  →  "{db_id}"
      TABLE     →  "{db_id}.{table_name}"
      COLUMN    →  "{db_id}.{table_name}.{column_name}"
    """

    node_id: str
    node_type: NodeType
    db_id: str

    # For TABLE and COLUMN nodes
    table_name: str = ""

    # For COLUMN nodes only
    column_name: str = ""
    dtype: str = ""
    is_pk: bool = False
    is_fk: bool = False           # True if this column appears in any FK constraint
    sample_values: list[str] = field(default_factory=list)

    # LLM-enriched fields (populated by NodeEnricher, optional at build time)
    description: str = ""
    synonyms: list[str] = field(default_factory=list)

    # Pre-computed embedding (populated by NodeEnricher after LLM step)
    # Stored as list[float] for JSON serializability; convert to np.ndarray on load
    embedding: list[float] = field(default_factory=list)

    # ── Convenience constructors ──────────────────────────────────────────────

    @classmethod
    def make_db(cls, db_id: str) -> "KGNode":
        return cls(node_id=db_id, node_type=NodeType.DATABASE, db_id=db_id)

    @classmethod
    def make_table(cls, db_id: str, table_name: str) -> "KGNode":
        return cls(
            node_id=f"{db_id}.{table_name}",
            node_type=NodeType.TABLE,
            db_id=db_id,
            table_name=table_name,
        )

    @classmethod
    def make_column(
        cls,
        db_id: str,
        table_name: str,
        column_name: str,
        dtype: str = "",
        is_pk: bool = False,
        sample_values: list[str] | None = None,
    ) -> "KGNode":
        return cls(
            node_id=f"{db_id}.{table_name}.{column_name}",
            node_type=NodeType.COLUMN,
            db_id=db_id,
            table_name=table_name,
            column_name=column_name,
            dtype=dtype,
            is_pk=is_pk,
            sample_values=sample_values or [],
        )

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "db_id": self.db_id,
            "table_name": self.table_name,
            "column_name": self.column_name,
            "dtype": self.dtype,
            "is_pk": self.is_pk,
            "is_fk": self.is_fk,
            "sample_values": self.sample_values,
            "description": self.description,
            "synonyms": self.synonyms,
            "embedding": self.embedding.tolist() if self.embedding is not None and hasattr(self.embedding, 'tolist') else self.embedding,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KGNode":
        return cls(
            node_id=d["node_id"],
            node_type=NodeType(d["node_type"]),
            db_id=d["db_id"],
            table_name=d.get("table_name", ""),
            column_name=d.get("column_name", ""),
            dtype=d.get("dtype", ""),
            is_pk=d.get("is_pk", False),
            is_fk=d.get("is_fk", False),
            sample_values=d.get("sample_values", []),
            description=d.get("description", ""),
            synonyms=d.get("synonyms", []),
            embedding=d.get("embedding", []),
        )

    def __repr__(self) -> str:
        extras = []
        if self.dtype:
            extras.append(self.dtype)
        if self.is_pk:
            extras.append("PK")
        if self.is_fk:
            extras.append("FK")
        suffix = f" [{', '.join(extras)}]" if extras else ""
        return f"KGNode({self.node_type.value}: {self.node_id}{suffix})"


# ---------------------------------------------------------------------------
# KGEdge
# ---------------------------------------------------------------------------


@dataclass
class KGEdge:
    """
    A typed, weighted directed edge between two nodes.

    Edges are stored directed (src → dst) but many logical relationships
    are bidirectional — the builder adds both directions when appropriate.

    weight is in [0.0, 1.0]:
      1.0  — certain structural relationship (TABLE_HAS_COLUMN, FOREIGN_KEY)
      0.7+ — strong semantic/statistical signal
      0.3+ — weak signal (lexical overlap only)
    """

    src_id: str
    dst_id: str
    edge_type: EdgeType
    weight: float = 1.0

    # Optional provenance metadata (not used in PPR, useful for debugging)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "src_id": self.src_id,
            "dst_id": self.dst_id,
            "edge_type": self.edge_type.value,
            "weight": self.weight,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KGEdge":
        return cls(
            src_id=d["src_id"],
            dst_id=d["dst_id"],
            edge_type=EdgeType(d["edge_type"]),
            weight=d.get("weight", 1.0),
            metadata=d.get("metadata", {}),
        )

    def __repr__(self) -> str:
        return (
            f"KGEdge({self.src_id} --[{self.edge_type.value}, w={self.weight:.2f}]--> {self.dst_id})"
        )
