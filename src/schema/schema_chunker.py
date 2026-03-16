"""
Schema Chunker – decomposes database schemas into 4 types of semantic chunks.

Chunk types (following RASL paper):
  1. table   – table-level summary with all column names
  2. column  – individual column with type and description
  3. fk      – foreign-key relationship
  4. value   – sample values for a column
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .models import Database


@dataclass
class SchemaChunk:
    """A single semantic unit derived from a database schema."""
    db_id: str
    chunk_type: str        # table | column | fk | value
    table_name: str        # primary table this chunk refers to
    content: str           # text representation for embedding
    metadata: dict = None  # extra metadata for filtering

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SchemaChunker:
    """Decomposes a :class:`Database` into a flat list of :class:`SchemaChunk`."""

    def __init__(self, max_sample_values: int = 5):
        self.max_sample_values = max_sample_values

    def chunk(self, db: Database) -> list[SchemaChunk]:
        """Generate all chunks for a single database."""
        chunks: list[SchemaChunk] = []
        chunks.extend(self._table_chunks(db))
        chunks.extend(self._column_chunks(db))
        chunks.extend(self._fk_chunks(db))
        chunks.extend(self._value_chunks(db))
        return chunks

    def chunk_many(self, databases: list[Database]) -> list[SchemaChunk]:
        """Generate chunks for multiple databases."""
        all_chunks: list[SchemaChunk] = []
        for db in databases:
            all_chunks.extend(self.chunk(db))
        return all_chunks

    # ---- chunk generators ---------------------------------------------------

    def _table_chunks(self, db: Database) -> list[SchemaChunk]:
        chunks = []
        for table in db.tables:
            col_names = ", ".join(c.name for c in table.columns)
            content = f"Table: {table.name}. Columns: {col_names}"
            chunks.append(
                SchemaChunk(
                    db_id=db.db_id,
                    chunk_type="table",
                    table_name=table.name,
                    content=content,
                    metadata={"num_columns": len(table.columns)},
                )
            )
        return chunks

    def _column_chunks(self, db: Database) -> list[SchemaChunk]:
        chunks = []
        for table in db.tables:
            for col in table.columns:
                pk_marker = " [PK]" if col.primary_key else ""
                desc = f" Description: {col.description}" if col.description else ""
                content = f"Column: {table.name}.{col.name} ({col.dtype}){pk_marker}.{desc}"
                chunks.append(
                    SchemaChunk(
                        db_id=db.db_id,
                        chunk_type="column",
                        table_name=table.name,
                        content=content,
                        metadata={"column_name": col.name, "dtype": col.dtype},
                    )
                )
        return chunks

    def _fk_chunks(self, db: Database) -> list[SchemaChunk]:
        chunks = []
        for fk in db.foreign_keys:
            content = f"FK: {fk.from_table}.{fk.from_column} → {fk.to_table}.{fk.to_column}"
            chunks.append(
                SchemaChunk(
                    db_id=db.db_id,
                    chunk_type="fk",
                    table_name=fk.from_table,
                    content=content,
                    metadata={"to_table": fk.to_table},
                )
            )
        return chunks

    def _value_chunks(self, db: Database) -> list[SchemaChunk]:
        chunks = []
        for key, values in db.sample_values.items():
            parts = key.split(".", 1)
            table_name = parts[0] if len(parts) > 1 else ""
            col_name = parts[1] if len(parts) > 1 else parts[0]
            sample = values[: self.max_sample_values]
            content = f"Values of {key}: {', '.join(str(v) for v in sample)}"
            chunks.append(
                SchemaChunk(
                    db_id=db.db_id,
                    chunk_type="value",
                    table_name=table_name,
                    content=content,
                    metadata={"column_name": col_name, "num_values": len(sample)},
                )
            )
        return chunks
