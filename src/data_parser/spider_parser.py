"""
Spider 1.0 dataset parser.

Converts Spider 1.0 native format (``tables.json`` + ``train_spider.json`` /
``dev.json``) into the standard internal representation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.schema import Database, Example

from .base import BaseParser

logger = logging.getLogger(__name__)


class SpiderParser(BaseParser):
    """Parser for Spider 1.0 dataset.

    Expected directory layout::

        {path}/
          tables.json
          database/
            {db_id}/{db_id}.sqlite
          train_spider.json   (optional)
          train_others.json   (optional)
          dev.json            (optional)
    """

    def load(self, path: str | Path) -> tuple[list[Database], list[Example]]:
        path = Path(path)
        databases = self._load_spider_format_schemas(
            tables_json=path / "tables.json",
            db_dir=path / "database",
        )
        examples = self._load_examples(path)
        logger.info(
            "SpiderParser: loaded %d databases, %d examples from %s",
            len(databases), len(examples), path,
        )
        return databases, examples

    # ------------------------------------------------------------------

    @staticmethod
    def _load_examples(dataset_dir: Path) -> list[Example]:
        """Load train + dev examples from Spider 1.0."""
        examples: list[Example] = []

        for filename in ("train_spider.json", "train_others.json", "dev.json"):
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
                        difficulty=(
                            raw.get("difficulty", "unknown")
                            if isinstance(raw.get("difficulty"), str)
                            else "unknown"
                        ),
                    )
                )

        return examples
