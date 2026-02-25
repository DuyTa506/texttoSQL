"""
Spider 1.0 dataset adapter.

Converts Spider 1.0 native format (tables.json + train_spider.json / dev.json)
into the standard internal representation.
"""

from __future__ import annotations

import json
from pathlib import Path

from .base_adapter import BaseAdapter, Column, Database, Example, ForeignKey, Table


class SpiderV1Adapter(BaseAdapter):
    """Adapter for Spider 1.0 dataset."""

    def load(self, path: str | Path) -> tuple[list[Database], list[Example]]:
        path = Path(path)
        databases = self._load_schemas(path / "tables.json", path / "database")
        examples = self._load_examples(path)
        return databases, examples

    # ---- schema loading -----------------------------------------------------

    def _load_schemas(self, tables_json: Path, db_dir: Path) -> list[Database]:
        """Parse ``tables.json`` into a list of :class:`Database`."""
        with open(tables_json, "r", encoding="utf-8") as f:
            raw_dbs = json.load(f)

        databases: list[Database] = []
        for raw in raw_dbs:
            db_id = raw["db_id"]

            # Build columns grouped by table index
            table_names: list[str] = [t[1] for t in raw.get("table_names_original", [])]
            col_names = raw.get("column_names_original", [])
            col_types = raw.get("column_types", [])
            pk_set = set(raw.get("primary_keys", []))

            # Construct Table objects
            tables: list[Table] = []
            for t_name in table_names:
                tables.append(Table(name=t_name))

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

    # ---- example loading ----------------------------------------------------

    def _load_examples(self, dataset_dir: Path) -> list[Example]:
        """Load train + dev examples from Spider 1.0."""
        examples: list[Example] = []

        for filename in ["train_spider.json", "train_others.json", "dev.json"]:
            filepath = dataset_dir / filename
            if not filepath.exists():
                continue
            with open(filepath, "r", encoding="utf-8") as f:
                raw_examples = json.load(f)
            for raw in raw_examples:
                examples.append(
                    Example(
                        db_id=raw["db_id"],
                        question=raw["question"],
                        query=raw.get("query", raw.get("sql", "")),
                        difficulty=raw.get("difficulty", "unknown") if isinstance(raw.get("difficulty"), str) else "unknown",
                    )
                )

        return examples
