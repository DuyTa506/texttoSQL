"""
Schema package — core data models and chunking utilities.

All public classes are re-exported here so the rest of the codebase
can use a single flat import instead of reaching into sub-modules:

    from src.schema import Database, Table, Column, SchemaChunk
"""

from .models import Column, Database, Example, ForeignKey, Table
from .schema_chunker import SchemaChunk, SchemaChunker
from .schema_indexer import SchemaIndexer

__all__ = [
    # models
    "Column",
    "Database",
    "Example",
    "ForeignKey",
    "Table",
    # chunker
    "SchemaChunk",
    "SchemaChunker",
    # indexer
    "SchemaIndexer",
]
