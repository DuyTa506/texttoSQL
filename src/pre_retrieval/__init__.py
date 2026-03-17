"""
Pre-Retrieval package — Phase 1 pipeline components.

Prepares and enriches the natural-language question before schema linking:

  ValueScanner       — fuzzy-matches NL tokens against DB cell values
  QuestionDecomposer — splits complex questions into simpler sub-questions
  QueryAugmentor     — enriches the query with schema/value hints
"""

from .value_scanner import ValueMatch, ValueScanner
from .question_decomposer import QuestionDecomposer
from .query_augmentor import QueryAugmentor

__all__ = [
    "ValueMatch",
    "ValueScanner",
    "QuestionDecomposer",
    "QueryAugmentor",
]
