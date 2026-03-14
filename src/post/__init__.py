"""
src/post – post-generation components for the Text-to-SQL pipeline.

Exports:
  SQLExecutor       – execute SQL and classify errors
  ExecutionResult   – result dataclass from SQLExecutor
  ErrorType         – enum of error categories
  RetryLoop         – post-generation retry loop
  RetryConfig       – config dataclass for RetryLoop
  CandidateSelector – select best SQL from multiple candidates
"""

from .candidate_selector import CandidateSelector
from .retry_loop import RetryConfig, RetryLoop
from .sql_executor import ErrorType, ExecutionResult, SQLExecutor

__all__ = [
    "SQLExecutor",
    "ExecutionResult",
    "ErrorType",
    "RetryLoop",
    "RetryConfig",
    "CandidateSelector",
]
