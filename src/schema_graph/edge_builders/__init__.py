"""
edge_builders sub-package.

Each module adds one layer of edges to a SchemaGraph:
  structural_edges  — Layer 1: DDL-derived (always run at build time)
  semantic_edges    — Layer 2: name/description/synonym similarity
  statistical_edges — Layer 3: SQL co-occurrence + DB value overlap  (TODO)
"""

from src.schema_graph.edge_builders.structural_edges import (
    build_structural_edges,
    tokenize_name,
    column_full_id,
    table_full_id,
)
from src.schema_graph.edge_builders.semantic_edges import (
    build_semantic_edges,
)
from src.schema_graph.edge_builders.statistical_edges import (
    build_statistical_edges,
    build_cooccurrence_edges,
    build_value_overlap_edges,
)

__all__ = [
    "build_structural_edges",
    "build_semantic_edges",
    "build_statistical_edges",
    "build_cooccurrence_edges",
    "build_value_overlap_edges",
    "tokenize_name",
    "column_full_id",
    "table_full_id",
]
