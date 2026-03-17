"""
NPMI Scorer – Normalized Pointwise Mutual Information for schema linking.

Measures statistical association between NL question tokens and schema
elements (table.column names) from training data.  Used as a 3rd retrieval
signal alongside BM25 and semantic search in the hybrid retrieval pipeline.

NPMI(token, col) = log(P(token,col) / P(token)*P(col)) / -log(P(token,col))

Values range from -1 (never co-occur) to +1 (always co-occur).
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from ...schema.schema_chunker import SchemaChunk

logger = logging.getLogger(__name__)

# Minimal English stopwords to filter out noise
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "shall", "should", "may", "might", "can", "could",
    "not", "and", "or", "but", "if", "of", "at", "by", "for",
    "with", "about", "to", "from", "in", "on", "it", "its",
    "this", "that", "these", "those", "i", "you", "he", "she",
    "we", "they", "me", "him", "her", "us", "them", "my", "your",
    "his", "our", "their", "what", "which", "who", "whom",
    "how", "many", "much", "each", "every", "all", "both",
    "few", "more", "most", "other", "some", "such", "no",
    "than", "too", "very", "just", "also", "so", "there",
    "then", "when", "where", "why", "as",
})

# Regex to extract table.column references from SQL
_SQL_TABLE_RE = re.compile(
    r"\bFROM\s+[`\"]?(\w+)[`\"]?"
    r"|\bJOIN\s+[`\"]?(\w+)[`\"]?"
    r"|\bINTO\s+[`\"]?(\w+)[`\"]?"
    r"|\bUPDATE\s+[`\"]?(\w+)[`\"]?",
    re.IGNORECASE,
)

_SQL_COLUMN_RE = re.compile(
    r"[`\"]?(\w+)[`\"]?\s*\.\s*[`\"]?(\w+)[`\"]?",
)


class NPMIScorer:
    """NPMI-based schema linking scorer.

    Builds a co-occurrence matrix from (question, SQL) training pairs
    and uses it to score schema chunks at inference time.
    """

    def __init__(self, min_count: int = 3):
        """
        Parameters
        ----------
        min_count : int
            Minimum co-occurrence count for a pair to be included.
            Pairs below this threshold are treated as NPMI = 0.
        """
        self.min_count = min_count
        # token → {schema_element → npmi_score}
        self.npmi_matrix: dict[str, dict[str, float]] = {}
        self._total_docs = 0

    # ========================================================================
    # Matrix Building
    # ========================================================================

    def build_matrix(
        self,
        examples: list[dict],
        data_format: str = "spider",
    ) -> None:
        """Build NPMI co-occurrence matrix from training data.

        Parameters
        ----------
        examples : list[dict]
            Training examples.  Expected keys depend on ``data_format``:

            - ``"spider"``:  ``{question, query}``
            - ``"omnisql"``: ``{input_seq, output_seq}``

        data_format : str
            ``"spider"`` or ``"omnisql"``.
        """
        # Step 1: Extract (nl_tokens, schema_elements) from each example
        pairs: list[tuple[set[str], set[str]]] = []

        for ex in examples:
            if data_format == "omnisql":
                question = self._extract_question_from_omnisql(ex.get("input_seq", ""))
                sql = self._extract_sql_from_omnisql(ex.get("output_seq", ""))
            else:
                question = ex.get("question", "")
                sql = ex.get("query", ex.get("sql", ""))

            nl_tokens = self._tokenize_question(question)
            schema_elements = self._extract_schema_refs(sql)

            if nl_tokens and schema_elements:
                pairs.append((nl_tokens, schema_elements))

        self._total_docs = len(pairs)
        logger.info("Building NPMI matrix from %d examples...", self._total_docs)

        if self._total_docs == 0:
            logger.warning("No valid examples found — NPMI matrix is empty.")
            return

        # Step 2: Count occurrences
        token_count: Counter = Counter()       # C(token)
        schema_count: Counter = Counter()      # C(schema_element)
        cooccur_count: Counter = Counter()     # C(token, schema_element)

        for nl_tokens, schema_elements in pairs:
            for t in nl_tokens:
                token_count[t] += 1
            for s in schema_elements:
                schema_count[s] += 1
            for t in nl_tokens:
                for s in schema_elements:
                    cooccur_count[(t, s)] += 1

        # Step 3: Compute NPMI
        N = self._total_docs
        self.npmi_matrix = {}

        for (token, schema_el), count in cooccur_count.items():
            if count < self.min_count:
                continue

            p_joint = count / N
            p_token = token_count[token] / N
            p_schema = schema_count[schema_el] / N

            # PMI = log(P(t,s) / P(t)*P(s))
            pmi = math.log(p_joint / (p_token * p_schema))

            # NPMI = PMI / -log(P(t,s))   — normalized to [-1, +1]
            neg_log_p_joint = -math.log(p_joint)
            if neg_log_p_joint == 0:
                continue  # skip degenerate case
            npmi = pmi / neg_log_p_joint

            if token not in self.npmi_matrix:
                self.npmi_matrix[token] = {}
            self.npmi_matrix[token][schema_el] = round(npmi, 4)

        total_pairs = sum(len(v) for v in self.npmi_matrix.values())
        logger.info(
            "NPMI matrix built: %d tokens, %d schema elements, %d pairs",
            len(self.npmi_matrix),
            len(schema_count),
            total_pairs,
        )

    # ========================================================================
    # Scoring
    # ========================================================================

    def score_chunks(
        self,
        question: str,
        chunks: list[SchemaChunk],
        *,
        db_id: str | None = None,
        top_k: int = 30,
    ) -> list[dict]:
        """Score schema chunks by NPMI relevance to question.

        For each chunk, compute max NPMI between any question token
        and the chunk's table/column names.

        Returns
        -------
        list[dict]
            Ranked results: ``{id, content, chunk, score, source}``.
        """
        nl_tokens = self._tokenize_question(question)

        scored: list[dict] = []
        for i, chunk in enumerate(chunks):
            if db_id and chunk.db_id != db_id:
                continue

            # Schema elements from this chunk
            chunk_elements = self._chunk_to_schema_elements(chunk)

            # Max NPMI across all (token, element) pairs
            max_npmi = 0.0
            for token in nl_tokens:
                token_scores = self.npmi_matrix.get(token, {})
                for elem in chunk_elements:
                    npmi_val = token_scores.get(elem, 0.0)
                    max_npmi = max(max_npmi, npmi_val)

            if max_npmi > 0:
                scored.append({
                    "id": f"{chunk.db_id}__{chunk.chunk_type}__{chunk.table_name}__{i}",
                    "content": chunk.content,
                    "chunk": chunk,
                    "score": max_npmi,
                    "source": "npmi",
                })

        # Sort by NPMI score descending, take top-k
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    # ========================================================================
    # Persistence
    # ========================================================================

    def save(self, path: str | Path) -> None:
        """Save NPMI matrix to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "min_count": self.min_count,
            "total_docs": self._total_docs,
            "matrix": self.npmi_matrix,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        total_pairs = sum(len(v) for v in self.npmi_matrix.values())
        logger.info("NPMI matrix saved to %s (%d pairs)", path, total_pairs)

    @classmethod
    def load(cls, path: str | Path) -> "NPMIScorer":
        """Load pre-built NPMI matrix from JSON file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        scorer = cls(min_count=data.get("min_count", 3))
        scorer._total_docs = data.get("total_docs", 0)
        scorer.npmi_matrix = data.get("matrix", {})

        total_pairs = sum(len(v) for v in scorer.npmi_matrix.values())
        logger.info("NPMI matrix loaded from %s (%d pairs)", path, total_pairs)
        return scorer

    # ========================================================================
    # Internal Helpers
    # ========================================================================

    @staticmethod
    def _tokenize_question(question: str) -> set[str]:
        """Tokenize NL question into a set of normalized tokens."""
        tokens = re.findall(r"[a-zA-Z_]\w*", question.lower())
        return {t for t in tokens if t not in _STOPWORDS and len(t) > 1}

    @staticmethod
    def _extract_schema_refs(sql: str) -> set[str]:
        """Extract table and table.column references from a SQL query.

        Returns lowered strings like ``"table_name"`` and ``"table.column"``.
        """
        refs: set[str] = set()

        # Extract table names from FROM/JOIN/INTO/UPDATE
        for match in _SQL_TABLE_RE.finditer(sql):
            table = next(g for g in match.groups() if g is not None)
            refs.add(table.lower())

        # Extract table.column references
        for match in _SQL_COLUMN_RE.finditer(sql):
            table, column = match.group(1).lower(), match.group(2).lower()
            refs.add(f"{table}.{column}")
            refs.add(table)

        return refs

    @staticmethod
    def _chunk_to_schema_elements(chunk: SchemaChunk) -> set[str]:
        """Extract schema element identifiers from a SchemaChunk."""
        elements: set[str] = set()

        if chunk.table_name:
            elements.add(chunk.table_name.lower())

        # For column chunks, add table.column
        col_name = (chunk.metadata or {}).get("column_name")
        if col_name and chunk.table_name:
            elements.add(f"{chunk.table_name.lower()}.{col_name.lower()}")

        return elements

    @staticmethod
    def _extract_question_from_omnisql(input_seq: str) -> str:
        """Quick extraction of question from OmniSQL input_seq."""
        q_start = input_seq.find("Question:\n")
        if q_start == -1:
            return input_seq[:200]
        q_content = q_start + len("Question:\n")
        q_end = input_seq.find("\nInstructions:", q_content)
        if q_end != -1:
            return input_seq[q_content:q_end].strip()
        return input_seq[q_content:q_content + 500].strip()

    @staticmethod
    def _extract_sql_from_omnisql(output_seq: str) -> str:
        """Quick extraction of SQL from OmniSQL output_seq."""
        match = re.search(r"```(?:sql)?\s*\n(.*?)```", output_seq, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # Fallback: look for SELECT
        select_match = re.search(r"(SELECT\s+.+)", output_seq, re.IGNORECASE | re.DOTALL)
        if select_match:
            return select_match.group(1).strip()
        return ""
