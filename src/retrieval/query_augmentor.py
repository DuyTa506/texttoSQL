"""
Query Augmentor – enriches raw NL questions with schema-relevant hints.

Strategies:
  "keyword"   – simple keyword extraction (default, no dependencies)
  "value"     – incorporates ValueScanner matches into the query
  "decompose" – returns multiple augmented queries (one per sub-question)
"""

from __future__ import annotations

import re
from typing import Optional, Union

from ..data.base_adapter import Database


class QueryAugmentor:
    """Augments a raw natural-language question with schema-relevant signals."""

    def __init__(self, strategy: str = "keyword"):
        """
        Parameters
        ----------
        strategy : str
            ``"keyword"``   – simple keyword extraction (default, no dependencies).
            ``"value"``     – keyword + ValueScanner cell-value hints.
            ``"decompose"`` – decompose complex questions + keyword augment each part.
        """
        self.strategy = strategy

    def augment(
        self,
        question: str,
        db: Optional[Database] = None,
        *,
        value_scanner=None,
        decomposer=None,
    ) -> Union[str, list[str]]:
        """Return the augmented query string (or list of strings for 'decompose').

        Parameters
        ----------
        question : str
            Raw natural-language question.
        db : Database, optional
            Database schema — used to match tokens against known tables/columns.
        value_scanner : ValueScanner, optional
            Pre-built ValueScanner instance (required for strategy='value').
        decomposer : QuestionDecomposer, optional
            Pre-built QuestionDecomposer instance (required for strategy='decompose').

        Returns
        -------
        str | list[str]
            Single augmented string for all strategies except 'decompose'.
            List of strings only when strategy='decompose' and question is complex.
        """
        if self.strategy == "keyword":
            return self._keyword_augment(question, db)

        if self.strategy == "value":
            return self._value_augment(question, db, value_scanner)

        if self.strategy == "decompose":
            return self._decompose_augment(question, db, decomposer)

        # Unknown strategy — return as-is
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

    def _value_augment(
        self,
        question: str,
        db: Optional[Database],
        value_scanner,
    ) -> str:
        """Keyword augment + ValueScanner cell-value hints.

        Format:
          question [tables: ...; columns: ...; values: table.col='value', ...]
        """
        # Start with keyword augmentation
        base = self._keyword_augment(question, db)

        if value_scanner is None or db is None:
            return base

        try:
            matches = value_scanner.scan(question, db)
        except Exception:
            return base

        if not matches:
            return base

        # Build value slots
        value_parts = [
            f"{m.table_name}.{m.column_name}='{m.matched_value}'"
            for m in matches
        ]
        value_slot = "values: " + ", ".join(value_parts)

        # Append value slot (or add a new bracket group)
        if base.endswith("]"):
            # Insert before closing bracket
            return base[:-1] + "; " + value_slot + "]"
        return base + " [" + value_slot + "]"

    def _decompose_augment(
        self,
        question: str,
        db: Optional[Database],
        decomposer,
    ) -> Union[str, list[str]]:
        """Decompose complex question into sub-questions, augment each with keywords.

        Returns list[str] when multiple sub-questions are produced,
        str otherwise (backward compatible).
        """
        if decomposer is None:
            return self._keyword_augment(question, db)

        sub_questions = decomposer.decompose(question)

        if len(sub_questions) <= 1:
            return self._keyword_augment(question, db)

        # Augment each sub-question independently
        augmented = [self._keyword_augment(sq, db) for sq in sub_questions]
        return augmented
