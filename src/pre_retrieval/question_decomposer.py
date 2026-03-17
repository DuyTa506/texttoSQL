"""
Question Decomposer – breaks complex multi-hop questions into sub-questions
for targeted retrieval.

Applied only when question complexity score exceeds a threshold.
Entirely rule-based and deterministic — no LLM call, zero latency.

Config flag:
  pre_retrieval.decomposition.enabled: false   (off by default, opt-in)
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Complexity heuristics
# ---------------------------------------------------------------------------

# Conjunctive / comparative keywords that suggest multi-hop structure
_COMPLEX_KEYWORDS = frozenset({
    "and", "or", "both", "compare", "comparison", "also", "additionally",
    "as well as", "along with", "together with", "versus", "vs",
    "difference between", "in addition", "furthermore",
})

# Relative clause / sub-question markers
_RELATIVE_MARKERS = re.compile(
    r"\b(who|which|that|where|whose|whom)\b",
    re.IGNORECASE,
)

# Aggregation patterns often co-occur with multi-hop reasoning
_AGGREGATION_RE = re.compile(
    r"\b(how many|how much|average|total|sum|count|maximum|minimum|most|least|"
    r"greatest|smallest|highest|lowest|percentage|ratio|proportion)\b",
    re.IGNORECASE,
)

# Split boundaries for divide-and-conquer decomposition
_CONJUNCTION_SPLIT_RE = re.compile(
    r"\s+(?:and|,\s*and|;\s*and|;\s*)\s+(?=[a-z])",
    re.IGNORECASE,
)

_COMPARISON_SPLIT_RE = re.compile(
    r"\b(?:compare|what is the difference between|versus|vs\.?)\b",
    re.IGNORECASE,
)


class QuestionDecomposer:
    """Rule-based question decomposer for divide-and-conquer retrieval.

    Parameters
    ----------
    complexity_threshold : float
        Minimum complexity score [0, 1] before decomposition is attempted.
        Lower → more aggressive decomposition.
    min_token_count : int
        Questions shorter than this are never decomposed (avoid splitting
        trivial questions on incidental conjunctions).
    """

    # ---- complexity scoring weights ----------------------------------------
    # Token-length contribution: score += min(n / _LEN_DIVISOR, _LEN_MAX_CONTRIB)
    _LEN_DIVISOR: float = 40.0
    _LEN_MAX_CONTRIB: float = 0.3
    # Per complex-keyword hit (at most one hit counts)
    _KEYWORD_CONTRIB: float = 0.15
    # Per relative-clause marker (capped at _REL_MAX_CONTRIB)
    _REL_PER_MATCH: float = 0.1
    _REL_MAX_CONTRIB: float = 0.2
    # Multiple aggregation functions (≥2 hits vs. exactly 1)
    _AGG_MULTI_CONTRIB: float = 0.2
    _AGG_SINGLE_CONTRIB: float = 0.05
    # Per comma in the question (capped at _COMMA_MAX_CONTRIB)
    _COMMA_PER_COUNT: float = 0.05
    _COMMA_MAX_CONTRIB: float = 0.15

    def __init__(
        self,
        complexity_threshold: float = 0.6,
        min_token_count: int = 12,
    ):
        self.complexity_threshold = complexity_threshold
        self.min_token_count = min_token_count

    # ---- public API ---------------------------------------------------------

    def is_complex(self, question: str) -> bool:
        """Return True if the question is complex enough to benefit from decomposition."""
        score = self._complexity_score(question)
        return score >= self.complexity_threshold

    def decompose(self, question: str) -> list[str]:
        """Decompose *question* into sub-questions.

        Returns a list with a single element (the original question) if the
        question is judged to be simple or cannot be meaningfully split.

        Parameters
        ----------
        question : str
            The natural-language question.

        Returns
        -------
        list[str]
            One or more sub-questions.
        """
        if not self.is_complex(question):
            return [question]

        sub_questions = self._split(question)

        # Filter out trivial sub-questions (< 4 tokens)
        sub_questions = [s.strip() for s in sub_questions if len(s.split()) >= 4]

        if len(sub_questions) < 2:
            return [question]

        return sub_questions

    # ---- complexity scoring -------------------------------------------------

    def _complexity_score(self, question: str) -> float:
        """Heuristic complexity score in [0, 1]."""
        tokens = question.split()
        n = len(tokens)

        if n < self.min_token_count:
            return 0.0

        score = 0.0
        question_lower = question.lower()

        # Token length signal: longer questions tend to be more complex
        score += min(n / self._LEN_DIVISOR, self._LEN_MAX_CONTRIB)

        # Presence of complex keywords
        for kw in _COMPLEX_KEYWORDS:
            if kw in question_lower:
                score += self._KEYWORD_CONTRIB
                break

        # Relative clause markers
        rel_matches = len(_RELATIVE_MARKERS.findall(question))
        score += min(rel_matches * self._REL_PER_MATCH, self._REL_MAX_CONTRIB)

        # Multiple aggregation functions
        agg_matches = len(_AGGREGATION_RE.findall(question))
        if agg_matches >= 2:
            score += self._AGG_MULTI_CONTRIB
        elif agg_matches == 1:
            score += self._AGG_SINGLE_CONTRIB

        # Comma + clause structure
        comma_count = question.count(",")
        score += min(comma_count * self._COMMA_PER_COUNT, self._COMMA_MAX_CONTRIB)

        return min(score, 1.0)

    # ---- splitting strategies -----------------------------------------------

    def _split(self, question: str) -> list[str]:
        """Try multiple splitting strategies, return the first that works."""
        # Strategy 1: Comparison split ("compare X and Y", "X versus Y")
        parts = self._comparison_split(question)
        if len(parts) >= 2:
            return parts

        # Strategy 2: Conjunction split on "and" between clauses
        parts = self._conjunction_split(question)
        if len(parts) >= 2:
            return parts

        # Strategy 3: Relative clause split
        parts = self._relative_clause_split(question)
        if len(parts) >= 2:
            return parts

        return [question]

    def _comparison_split(self, question: str) -> list[str]:
        """Split comparison questions like 'Compare X and Y' into [X, Y]."""
        match = _COMPARISON_SPLIT_RE.search(question)
        if not match:
            return [question]

        # After the comparison keyword, look for "and" as the separator
        after = question[match.end():].strip()
        and_match = re.search(r"\band\b", after, re.IGNORECASE)
        if not and_match:
            return [question]

        part_a = after[: and_match.start()].strip()
        part_b = after[and_match.end():].strip()

        # Reconstruct as independent sub-questions
        results = []
        if part_a:
            results.append(f"What is {part_a}?")
        if part_b:
            results.append(f"What is {part_b}?")
        return results if len(results) >= 2 else [question]

    def _conjunction_split(self, question: str) -> list[str]:
        """Split on conjunctions that separate independent clauses."""
        parts = _CONJUNCTION_SPLIT_RE.split(question)
        if len(parts) < 2:
            return [question]

        # Clean up: ensure each part ends with a question mark
        result = []
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            # First part: keep original question structure
            if i == 0:
                result.append(part if part.endswith("?") else part)
            else:
                # Subsequent parts: often verb phrases — prepend a stub
                result.append(part if part.endswith("?") else part)
        return result

    def _relative_clause_split(self, question: str) -> list[str]:
        """Split on relative clause markers (who, which, that, where)."""
        # Find a relative clause that introduces a constraint
        match = re.search(
            r"\b(who|which|that)\b\s+(?:is|are|was|were|has|have|had|"
            r"works?|lives?|comes?|comes?)\b",
            question,
            re.IGNORECASE,
        )
        if not match:
            return [question]

        main_clause = question[: match.start()].strip()
        sub_clause = question[match.start():].strip()

        if not main_clause or not sub_clause:
            return [question]

        return [main_clause, sub_clause]
