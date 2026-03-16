"""
Retry Loop – core post-generation loop that retries SQL generation on errors.

Flow:
  1. Execute initial SQL → inspect error type
  2. Build correction prompt with error-specific hints
  3. Re-generate via SQLInference
  4. Repeat up to max_retries times
  5. Return final result with retry metadata

Config flags:
  post_generation.retry_loop.enabled: false   (off by default)
  post_generation.retry_loop.max_retries: 3
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .sql_executor import ErrorType, ExecutionResult, SQLExecutor

if TYPE_CHECKING:
    from ..generation.inference import SQLInference

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Maximum characters of an error message shown in the retry log line
_ERROR_MSG_PREVIEW_LEN = 120


@dataclass
class RetryConfig:
    """Configuration for the RetryLoop."""

    max_retries: int = 3
    enabled: bool = True


# ---------------------------------------------------------------------------
# Error-type → correction hint mapping
# ---------------------------------------------------------------------------

_ERROR_HINTS: dict[ErrorType, str] = {
    ErrorType.SYNTAX_ERROR: (
        "The SQL has a syntax error. "
        "Check parentheses, keywords, quotes, and clause ordering."
    ),
    ErrorType.NO_SUCH_TABLE: (
        "A table referenced in the SQL does not exist. "
        "Use only tables listed in the schema above."
    ),
    ErrorType.NO_SUCH_COLUMN: (
        "A column referenced in the SQL does not exist. "
        "Check the column names carefully against the schema above."
    ),
    ErrorType.WRONG_RESULT: (
        "The SQL executes but returns incorrect results. "
        "Re-examine the JOIN conditions, WHERE filters, and aggregation logic."
    ),
    ErrorType.EMPTY_RESULT: (
        "The SQL returns no rows. "
        "Check whether your WHERE conditions are too restrictive "
        "or whether JOINs are eliminating rows unexpectedly."
    ),
    ErrorType.EXECUTION_ERROR: (
        "The SQL caused a runtime error. "
        "Check for type mismatches, division by zero, or unsupported operations."
    ),
}

# Correction prompt template
_CORRECTION_TEMPLATE = """\
Given the schema and question below, the SQL query you previously generated has an error. Fix it.

Schema:
{schema_context}

Question: {question}

Wrong SQL:
```sql
{wrong_sql}
```

Error type: {error_type}
Error message: {error_message}
Hint: {hint}

Generate the corrected SQL inside ```sql ... ```."""


class RetryLoop:
    """Post-generation retry loop: run SQL → inspect error → re-prompt → repeat.

    Parameters
    ----------
    executor : SQLExecutor
        Executor instance for running SQL against SQLite.
    inference : SQLInference
        Inference engine for re-generating corrected SQL.
    config : RetryConfig, optional
        Retry parameters (max_retries, enabled).
    """

    def __init__(
        self,
        executor: SQLExecutor,
        inference: "SQLInference",
        config: RetryConfig | None = None,
    ):
        self.executor = executor
        self.inference = inference
        self.config = config or RetryConfig()

    # ---- public API ---------------------------------------------------------

    def run(
        self,
        question: str,
        schema_context: str,
        initial_result: dict,
        db_id: str,
        gold_sql: str | None = None,
    ) -> dict:
        """Run the retry loop starting from *initial_result*.

        Parameters
        ----------
        question : str
            The original NL question.
        schema_context : str
            Formatted schema string (from SchemaFilter).
        initial_result : dict
            Dict from SQLInference.generate(): keys ``sql``, ``reasoning``, ``raw_output``.
        db_id : str
            Database identifier for execution.
        gold_sql : str, optional
            Gold SQL for WRONG_RESULT detection.

        Returns
        -------
        dict
            The *initial_result* dict extended with:
              - ``retry_count`` (int)
              - ``retry_history`` (list of {sql, error_type, error_message})
              - ``correction_applied`` (bool)
        """
        result = dict(initial_result)
        result.setdefault("retry_count", 0)
        result.setdefault("retry_history", [])
        result.setdefault("correction_applied", False)

        if not self.config.enabled:
            return result

        current_sql = result.get("sql", "")

        for attempt in range(self.config.max_retries):
            exec_result = self.executor.execute(current_sql, db_id, gold_sql=gold_sql)

            if exec_result.success:
                # Already correct — stop early
                logger.debug(
                    "RetryLoop: SQL succeeded on attempt %d for db_id='%s'",
                    attempt,
                    db_id,
                )
                break

            # Record this attempt's failure
            result["retry_history"].append({
                "sql": current_sql,
                "error_type": exec_result.error_type.value,
                "error_message": exec_result.error_message,
            })

            logger.info(
                "RetryLoop attempt %d/%d — error=%s: %s",
                attempt + 1,
                self.config.max_retries,
                exec_result.error_type.value,
                exec_result.error_message[:_ERROR_MSG_PREVIEW_LEN],
            )

            # Build correction prompt and re-generate
            correction_prompt = self._build_correction_prompt(
                question, schema_context, current_sql, exec_result
            )

            try:
                new_result = self.inference.generate(correction_prompt, "")
                new_sql = new_result.get("sql", "").strip()
            except Exception as e:
                logger.warning("RetryLoop: inference error on attempt %d: %s", attempt + 1, e)
                break

            if not new_sql or new_sql == current_sql:
                logger.debug(
                    "RetryLoop: no change on attempt %d, stopping.", attempt + 1
                )
                break

            # Update current state
            current_sql = new_sql
            result["sql"] = new_sql
            result["reasoning"] = new_result.get("reasoning", result.get("reasoning", ""))
            result["raw_output"] = new_result.get("raw_output", result.get("raw_output", ""))
            result["retry_count"] = attempt + 1
            result["correction_applied"] = True

        return result

    # ---- prompt building ----------------------------------------------------

    def _build_correction_prompt(
        self,
        question: str,
        schema_context: str,
        wrong_sql: str,
        error: ExecutionResult,
    ) -> str:
        """Build a correction prompt for the inference model.

        Parameters
        ----------
        question : str
            Original NL question.
        schema_context : str
            Formatted schema context.
        wrong_sql : str
            The SQL that failed.
        error : ExecutionResult
            Execution result with classified error type + message.

        Returns
        -------
        str
            Complete correction prompt ready for SQLInference.generate().
        """
        hint = _ERROR_HINTS.get(error.error_type, "Review the SQL carefully.")

        # Enrich hint with extracted entity from error message when possible
        if error.error_type == ErrorType.NO_SUCH_TABLE:
            table_match = re.search(r"no such table:\s*(\S+)", error.error_message, re.IGNORECASE)
            if table_match:
                hint = (
                    f"Table `{table_match.group(1)}` does not exist. "
                    "Use only tables from the schema above."
                )
        elif error.error_type == ErrorType.NO_SUCH_COLUMN:
            col_match = re.search(r"no such column:\s*(\S+)", error.error_message, re.IGNORECASE)
            if col_match:
                hint = (
                    f"Column `{col_match.group(1)}` does not exist. "
                    "Check column names in the schema above."
                )

        return _CORRECTION_TEMPLATE.format(
            schema_context=schema_context,
            question=question,
            wrong_sql=wrong_sql,
            error_type=error.error_type.value,
            error_message=error.error_message or "(no details)",
            hint=hint,
        )
