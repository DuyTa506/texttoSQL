"""
Value Scanner – scans actual DB cell values to match NL mentions before retrieval.

Critical for BIRD-style questions that reference specific values (e.g. "New York").
Runs a SQLite DISTINCT query per column and uses fuzzy string matching to find
cell-value mentions in the question.

Config flags:
  pre_retrieval.value_scan.enabled: true
  pre_retrieval.value_scan.max_values: 500
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from ..schema.models import Database

logger = logging.getLogger(__name__)

# Minimum similarity score (0–1) for a fuzzy match to be accepted
_DEFAULT_MIN_SCORE = 0.75
# Only scan TEXT / VARCHAR columns (skip numeric-only columns)
_TEXT_DTYPES = frozenset({
    "text", "varchar", "char", "string", "nvarchar", "name",
    "character", "mediumtext", "longtext", "clob",
})

# ---- candidate extraction constants ----------------------------------------

# Minimum length for a quoted string to be a useful candidate
_MIN_QUOTED_LEN = 2
# Regex: single- or double-quoted strings of at least _MIN_QUOTED_LEN chars
_QUOTED_STRING_RE = re.compile(r'["\']([^"\']{' + str(_MIN_QUOTED_LEN) + r',})["\']')
# Regex: runs of Title-case words (named entities like "New York")
_CAPITALIZED_PHRASE_RE = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')
# Regex: word tokens (letters + underscores + digits)
_WORD_TOKEN_RE = re.compile(r"[a-zA-Z_]\w*")
# Minimum token length to include as a candidate
_MIN_TOKEN_LEN = 3
# Common question stop-words that are rarely DB values
_STOP_WORDS = frozenset({
    "the", "and", "for", "are", "was", "were", "has", "have",
    "had", "been", "with", "that", "this", "from", "what",
    "which", "who", "how", "many", "all", "each", "any",
})


@dataclass
class ValueMatch:
    """A cell value in the DB that was matched to a token in the NL question."""

    table_name: str
    column_name: str
    matched_value: str
    score: float  # similarity in [0, 1]

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ValueMatch({self.table_name}.{self.column_name}"
            f"='{self.matched_value}', score={self.score:.2f})"
        )


class ValueScanner:
    """Scan DB cell values to identify NL question mentions.

    Parameters
    ----------
    max_values_per_col : int
        Maximum number of DISTINCT values to fetch per column (LIMIT).
    top_k : int
        Return at most this many ValueMatch objects.
    min_score : float
        Minimum similarity threshold for fuzzy matching.
    """

    def __init__(
        self,
        max_values_per_col: int = 500,
        top_k: int = 5,
        min_score: float = _DEFAULT_MIN_SCORE,
    ):
        self.max_values_per_col = max_values_per_col
        self.top_k = top_k
        self.min_score = min_score

    # ---- public API ---------------------------------------------------------

    def scan(self, question: str, db: Database) -> list[ValueMatch]:
        """Scan DB values to find matches for tokens in *question*.

        Parameters
        ----------
        question : str
            The natural-language question.
        db : Database
            Database schema + path to SQLite file.

        Returns
        -------
        list[ValueMatch]
            Top-k matches sorted by score descending.
        """
        if not db.db_path or not Path(db.db_path).exists():
            logger.debug(
                "ValueScanner: db_path not set or missing for '%s'", db.db_id
            )
            return []

        # Extract candidate tokens from the question (quoted strings first,
        # then multi-word n-grams, then single words)
        candidates = self._extract_candidates(question)
        if not candidates:
            return []

        matches: list[ValueMatch] = []

        try:
            with sqlite3.connect(db.db_path) as conn:
                conn.execute("PRAGMA read_uncommitted = true")
                cursor = conn.cursor()

                for table in db.tables:
                    for column in table.columns:
                        # Skip non-text columns to avoid scanning numeric/blob data
                        if not self._is_text_column(column.dtype):
                            continue

                        values = self._fetch_distinct_values(
                            cursor, table.name, column.name
                        )
                        for value in values:
                            if not value:
                                continue
                            for candidate in candidates:
                                score = self._similarity(candidate, value)
                                if score >= self.min_score:
                                    matches.append(
                                        ValueMatch(
                                            table_name=table.name,
                                            column_name=column.name,
                                            matched_value=value,
                                            score=score,
                                        )
                                    )
        except Exception as e:
            logger.warning("ValueScanner error for db '%s': %s", db.db_id, e)
            return []

        # Deduplicate: keep best score per (table, column, value) triple
        best: dict[tuple, ValueMatch] = {}
        for m in matches:
            key = (m.table_name, m.column_name, m.matched_value.lower())
            if key not in best or m.score > best[key].score:
                best[key] = m

        sorted_matches = sorted(best.values(), key=lambda x: x.score, reverse=True)
        return sorted_matches[: self.top_k]

    def to_schema_hints(self, matches: list[ValueMatch]) -> str:
        """Format ValueMatch list as SQL comment hints for the prompt.

        Returns a string like:
            -- students.name likely contains: 'Alice'
            -- enrollment.course likely contains: 'CS101'
        """
        if not matches:
            return ""
        lines = [
            f"-- {m.table_name}.{m.column_name} likely contains: '{m.matched_value}'"
            for m in matches
        ]
        return "\n".join(lines)

    # ---- internal helpers ---------------------------------------------------

    def _fetch_distinct_values(
        self,
        cursor: sqlite3.Cursor,
        table_name: str,
        column_name: str,
    ) -> list[str]:
        """Fetch DISTINCT non-null values from a column (up to max_values_per_col)."""
        try:
            # Quote identifiers to handle reserved words
            query = (
                f'SELECT DISTINCT "{column_name}" FROM "{table_name}" '
                f'WHERE "{column_name}" IS NOT NULL '
                f"LIMIT {self.max_values_per_col}"
            )
            cursor.execute(query)
            return [str(row[0]).strip() for row in cursor.fetchall() if row[0] is not None]
        except sqlite3.OperationalError as e:
            logger.debug(
                "ValueScanner: skip %s.%s — %s", table_name, column_name, e
            )
            return []

    @staticmethod
    def _is_text_column(dtype: str) -> bool:
        """Return True if the column type suggests textual content."""
        return dtype.lower().split("(")[0].strip() in _TEXT_DTYPES

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        """Fuzzy similarity between *a* (question token) and *b* (DB value).

        Uses SequenceMatcher for a simple, dependency-free metric.
        Falls back to exact/substring match for speed on short strings.
        """
        a_low = a.lower().strip()
        b_low = b.lower().strip()

        # Exact match → score 1.0
        if a_low == b_low:
            return 1.0

        # Substring match (question token is contained in DB value or vice versa)
        if a_low in b_low or b_low in a_low:
            # Score proportional to coverage
            shorter = min(len(a_low), len(b_low))
            longer = max(len(a_low), len(b_low))
            if longer == 0:
                return 0.0
            return shorter / longer

        # Sequence-based similarity
        return SequenceMatcher(None, a_low, b_low).ratio()

    @staticmethod
    def _extract_candidates(question: str) -> list[str]:
        """Extract candidate strings from the question to match against DB values.

        Priority order:
          1. Quoted strings (single or double quotes)
          2. Capitalized multi-word phrases (e.g. "New York")
          3. Individual tokens ≥ _MIN_TOKEN_LEN characters
        """
        candidates: list[str] = []

        # 1. Quoted strings
        candidates.extend(_QUOTED_STRING_RE.findall(question))

        # 2. Capitalized sequences (potential named entities)
        candidates.extend(_CAPITALIZED_PHRASE_RE.findall(question))

        # 3. Individual tokens (lowercase, len >= _MIN_TOKEN_LEN), skip stop-words
        for tok in _WORD_TOKEN_RE.findall(question):
            tok_low = tok.lower()
            if len(tok_low) >= _MIN_TOKEN_LEN and tok_low not in _STOP_WORDS:
                candidates.append(tok)

        # Deduplicate preserving order
        seen: set[str] = set()
        result: list[str] = []
        for c in candidates:
            key = c.lower()
            if key not in seen:
                seen.add(key)
                result.append(c)

        return result
