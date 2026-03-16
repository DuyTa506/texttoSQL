"""
Standard data models for Text-to-SQL.

Defines the core dataclasses shared across the entire project:
``Column``, ``Table``, ``ForeignKey``, ``Database``, ``Example``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# =============================================================================
# Standard Data Models
# =============================================================================

@dataclass
class Column:
    """Represents a single column in a database table."""
    name: str
    dtype: str  # e.g. "INTEGER", "TEXT", "REAL"
    primary_key: bool = False
    description: str = ""


@dataclass
class ForeignKey:
    """Represents a foreign-key relationship between two columns."""
    from_table: str
    from_column: str
    to_table: str
    to_column: str


@dataclass
class Table:
    """Represents a database table with its columns."""
    name: str
    columns: list[Column] = field(default_factory=list)


@dataclass
class Database:
    """Represents a complete database schema (one DB from the benchmark)."""
    db_id: str
    tables: list[Table] = field(default_factory=list)
    foreign_keys: list[ForeignKey] = field(default_factory=list)
    sample_values: dict[str, list[str]] = field(default_factory=dict)
    db_path: str | None = None  # path to SQLite file if available

    # ---- convenience helpers ------------------------------------------------

    def get_table(self, name: str) -> Table | None:
        """Return table by name (case-insensitive)."""
        name_low = name.lower()
        for t in self.tables:
            if t.name.lower() == name_low:
                return t
        return None

    def get_fk_neighbours(self, table_name: str) -> list[str]:
        """Return table names that are FK-connected to *table_name*."""
        name_low = table_name.lower()
        neighbours: set[str] = set()
        for fk in self.foreign_keys:
            if fk.from_table.lower() == name_low:
                neighbours.add(fk.to_table)
            elif fk.to_table.lower() == name_low:
                neighbours.add(fk.from_table)
        return sorted(neighbours)


@dataclass
class Example:
    """A single (question, SQL) example from the benchmark."""
    db_id: str
    question: str
    query: str  # gold SQL
    lang: str = "en"
    difficulty: str = "unknown"
    evidence: str = ""  # domain hint (e.g. BIRD "evidence" field)
