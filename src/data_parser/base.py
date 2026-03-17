"""
Base parser interface for dataset loading.

Provides :class:`BaseParser` — an abstract base class that all dataset parsers
implement — plus a shared helper for parsing Spider-format ``tables.json``
files (used by both Spider and BIRD).
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path

from src.schema import Column, Database, Example, ForeignKey, Table

logger = logging.getLogger(__name__)


class BaseParser(ABC):
    """Abstract parser that loads a dataset into the standard format.

    Every concrete parser must implement :meth:`load`, which returns a
    ``(databases, examples)`` tuple.
    """

    @abstractmethod
    def load(self, path: str | Path) -> tuple[list[Database], list[Example]]:
        """Load databases and examples from *path*.

        Parameters
        ----------
        path:
            Root directory of the dataset (e.g. ``data/spider`` or
            ``datasets/data/bird/dev_20240627``).

        Returns
        -------
        databases:
            Schema information for every database in the dataset.
        examples:
            Question / SQL pairs, each referencing a ``db_id``.
        """
        ...

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_spider_format_schemas(
        tables_json: Path,
        db_dir: Path,
    ) -> list[Database]:
        """Parse a Spider-format ``tables.json`` into :class:`Database` objects.

        Both Spider 1.0 and BIRD use the same schema JSON layout
        (``table_names_original``, ``column_names_original``, ``column_types``,
        ``primary_keys``, ``foreign_keys``).  This shared helper avoids
        duplicating the parsing logic.

        Parameters
        ----------
        tables_json:
            Path to the JSON file containing schema definitions.
        db_dir:
            Directory that contains per-database SQLite files, laid out as
            ``{db_dir}/{db_id}/{db_id}.sqlite``.
        """
        with open(tables_json, "r", encoding="utf-8") as f:
            raw_dbs = json.load(f)

        databases: list[Database] = []
        for raw in raw_dbs:
            db_id: str = raw["db_id"]

            # Build columns grouped by table index
            # table_names_original is a flat list of strings: ["table1", "table2", ...]
            table_names: list[str] = raw.get("table_names_original", [])
            col_names = raw.get("column_names_original", [])
            col_types = raw.get("column_types", [])
            # Primary keys — may contain ints (single-col PK) or lists
            # (composite PK).  Flatten to a set of column indices.
            pk_raw = raw.get("primary_keys", [])
            pk_set: set[int] = set()
            for pk in pk_raw:
                if isinstance(pk, list):
                    pk_set.update(pk)
                else:
                    pk_set.add(pk)

            # Construct Table objects
            tables: list[Table] = [Table(name=t_name) for t_name in table_names]

            for col_idx, (table_idx, col_name) in enumerate(col_names):
                if table_idx < 0:
                    continue  # skip the special "*" column
                col_type = col_types[col_idx] if col_idx < len(col_types) else "TEXT"
                is_pk = col_idx in pk_set
                tables[table_idx].columns.append(
                    Column(name=col_name, dtype=col_type, primary_key=is_pk)
                )

            # Foreign keys
            foreign_keys: list[ForeignKey] = []
            for fk_from_idx, fk_to_idx in raw.get("foreign_keys", []):
                from_table_idx, from_col = col_names[fk_from_idx]
                to_table_idx, to_col = col_names[fk_to_idx]
                if from_table_idx >= 0 and to_table_idx >= 0:
                    foreign_keys.append(
                        ForeignKey(
                            from_table=table_names[from_table_idx],
                            from_column=from_col,
                            to_table=table_names[to_table_idx],
                            to_column=to_col,
                        )
                    )

            # DB path
            db_path = db_dir / db_id / f"{db_id}.sqlite"
            databases.append(
                Database(
                    db_id=db_id,
                    tables=tables,
                    foreign_keys=foreign_keys,
                    db_path=str(db_path) if db_path.exists() else None,
                )
            )

        return databases
