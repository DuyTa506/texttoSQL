"""
BIRD dataset parser.

Converts BIRD native format into the standard internal representation.
Handles key differences from Spider:

- Schema file: ``dev_tables.json`` / ``train_tables.json`` (not ``tables.json``)
- DB directory: ``dev_databases/`` / ``train_databases/`` (not ``database/``)
- SQL key: ``"SQL"`` (uppercase, not ``"query"``)
- Extra field: ``"evidence"`` (domain hints) → :attr:`Example.evidence`
- Difficulty values: ``"simple"`` / ``"moderate"`` / ``"challenging"``

The parser auto-detects whether the path points to a **dev** or **train** split
based on the presence of ``dev_tables.json`` vs ``train_tables.json``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.schema import Database, Example

from .base import BaseParser

logger = logging.getLogger(__name__)


class BirdParser(BaseParser):
    """Parser for BIRD dataset.

    Expected directory layout (dev split)::

        {path}/
          dev_tables.json
          dev_databases/
            {db_id}/{db_id}.sqlite
          dev.json

    Expected directory layout (train split)::

        {path}/
          train_tables.json
          train_databases/
            {db_id}/{db_id}.sqlite
          train.json
    """

    def load(self, path: str | Path) -> tuple[list[Database], list[Example]]:
        path = Path(path)
        split = self._detect_split(path)
        logger.info("BirdParser: detected '%s' split in %s", split, path)

        # Resolve schema + DB paths based on split
        tables_json = path / f"{split}_tables.json"
        db_dir = path / f"{split}_databases"

        # Fallback: if split-specific files don't exist, try generic names
        if not tables_json.exists():
            tables_json = path / "tables.json"
        if not db_dir.exists():
            db_dir = path / "database"

        if not tables_json.exists():
            raise FileNotFoundError(
                f"Cannot find schema file. Tried '{split}_tables.json' and "
                f"'tables.json' in {path}"
            )

        databases = self._load_spider_format_schemas(tables_json, db_dir)
        examples = self._load_examples(path, split)
        logger.info(
            "BirdParser: loaded %d databases, %d examples from %s",
            len(databases), len(examples), path,
        )
        return databases, examples

    # ------------------------------------------------------------------

    @staticmethod
    def _detect_split(path: Path) -> str:
        """Auto-detect whether *path* is a dev or train split.

        Checks for ``dev_tables.json`` first (dev), then
        ``train_tables.json`` (train).  Falls back to ``"dev"``.
        """
        if (path / "dev_tables.json").exists():
            return "dev"
        if (path / "train_tables.json").exists():
            return "train"
        # Heuristic: check directory name
        name = path.name.lower()
        if "train" in name:
            return "train"
        return "dev"  # default fallback

    @staticmethod
    def _load_examples(dataset_dir: Path, split: str) -> list[Example]:
        """Load examples from BIRD JSON files.

        Tries ``{split}.json`` first, then falls back to common filenames.
        """
        examples: list[Example] = []

        # BIRD dev uses ``dev.json``; train uses ``train.json``
        candidates = [
            f"{split}.json",
            "dev.json",
            "train.json",
        ]

        loaded_file: str | None = None
        for filename in candidates:
            filepath = dataset_dir / filename
            if not filepath.exists():
                continue

            with open(filepath, "r", encoding="utf-8") as f:
                raw_examples = json.load(f)

            for raw in raw_examples:
                # BIRD uses uppercase "SQL" key — handle both cases
                gold_sql = raw.get("SQL", raw.get("sql", raw.get("query", "")))

                difficulty_raw = raw.get("difficulty", "unknown")
                difficulty = (
                    difficulty_raw if isinstance(difficulty_raw, str) else "unknown"
                )

                evidence = raw.get("evidence", "")

                examples.append(
                    Example(
                        db_id=raw["db_id"],
                        question=raw["question"],
                        query=gold_sql,
                        difficulty=difficulty,
                        evidence=evidence if isinstance(evidence, str) else "",
                    )
                )

            loaded_file = filename
            break  # load only the first matching file

        if loaded_file:
            logger.info("BirdParser: loaded %d examples from %s", len(examples), loaded_file)
        else:
            logger.warning(
                "BirdParser: no example file found in %s (tried %s)",
                dataset_dir, candidates,
            )

        return examples
