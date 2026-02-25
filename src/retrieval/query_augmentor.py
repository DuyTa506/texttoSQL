"""
Query Augmentor – enriches raw NL questions with schema-relevant hints.

Strategy: extract keywords and entity-like tokens that may correspond to
table / column names, then append them as "target slots" to the query.
"""

from __future__ import annotations

import re
from typing import Optional

from ..data.base_adapter import Database


class QueryAugmentor:
    """Augments a raw natural-language question with schema-relevant signals."""

    def __init__(self, strategy: str = "keyword"):
        """
        Parameters
        ----------
        strategy : str
            ``"keyword"`` – simple keyword extraction (default, no dependencies).
        """
        self.strategy = strategy

    def augment(
        self,
        question: str,
        db: Optional[Database] = None,
    ) -> str:
        """Return the augmented query string.

        Parameters
        ----------
        question : str
            Raw natural-language question.
        db : Database, optional
            Database schema — used to match tokens against known tables/columns.
        """
        if self.strategy == "keyword":
            return self._keyword_augment(question, db)
        return question

    # ---- strategies ---------------------------------------------------------

    def _keyword_augment(self, question: str, db: Optional[Database]) -> str:
        """Append table/column tokens found in the question."""
        if db is None:
            return question

        # Build lookup sets
        table_names = {t.name.lower() for t in db.tables}
        column_names: set[str] = set()
        for t in db.tables:
            for c in t.columns:
                column_names.add(c.name.lower())

        # Tokenise question (split on non-alphanumeric, keep underscores)
        tokens = set(re.findall(r"[a-zA-Z_]\w*", question.lower()))

        # Match against schema
        matched_tables = sorted(tokens & table_names)
        matched_columns = sorted(tokens & column_names)

        # Build target slots string
        slots: list[str] = []
        if matched_tables:
            slots.append("tables: " + ", ".join(matched_tables))
        if matched_columns:
            slots.append("columns: " + ", ".join(matched_columns))

        if slots:
            return question + " [" + "; ".join(slots) + "]"
        return question
