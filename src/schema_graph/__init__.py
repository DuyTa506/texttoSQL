"""
Schema Knowledge Graph package.

Build a typed, multi-layer graph from a Database schema for use in
graph-guided schema linking during text-to-SQL inference.

Three layers of edges:
  Layer 1 — Structural  (DDL-derived: table↔column, FK, PK)
  Layer 2 — Semantic    (name similarity, description similarity, synonym overlap)
  Layer 3 — Statistical (SQL co-occurrence, DB value overlap)

Public API:
  from src.schema_graph import (
      SchemaGraph, SchemaGraphBuilder,
      KGNode, KGEdge, EdgeType, NodeType,
      build_structural_edges,
      build_semantic_edges,
      build_statistical_edges,
      NodeEnricher,
  )
"""

from src.schema_graph.graph_types import EdgeType, KGEdge, KGNode, NodeType
from src.schema_graph.graph_builder import SchemaGraph, SchemaGraphBuilder
from src.schema_graph.edge_builders.structural_edges import build_structural_edges
from src.schema_graph.edge_builders.semantic_edges import build_semantic_edges
from src.schema_graph.edge_builders.statistical_edges import (
    build_statistical_edges,
    build_cooccurrence_edges,
    build_value_overlap_edges,
)
from src.schema_graph.node_enricher import NodeEnricher
from src.schema_graph.graph_retriever import GraphRetriever

__all__ = [
    # Core containers
    "SchemaGraph",
    "SchemaGraphBuilder",
    # Node / edge types
    "KGNode",
    "KGEdge",
    "EdgeType",
    "NodeType",
    # Edge builders
    "build_structural_edges",
    "build_semantic_edges",
    "build_statistical_edges",
    "build_cooccurrence_edges",
    "build_value_overlap_edges",
    # Enrichment + retrieval
    "NodeEnricher",
    "GraphRetriever",
]

