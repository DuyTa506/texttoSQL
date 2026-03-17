"""
SQL Executor – executes SQL against SQLite and classifies errors precisely.

Reuses the same sqlite3 pattern as training/reward.py and src/evaluation/metrics.py
but adds fine-grained error classification to power the RetryLoop correction prompts.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from src.shared.sqlite_utils import safe_execute

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    SUCCESS = "success"
    SYNTAX_ERROR = "syntax_error"           # sqlite3.OperationalError on parse
    NO_SUCH_TABLE = "no_such_table"         # "no such table: X"
    NO_SUCH_COLUMN = "no_such_column"       # "no such column: X"
    WRONG_RESULT = "wrong_result"           # executed but result != gold
    EMPTY_RESULT = "empty_result"           # executed but returned 0 rows
    EXECUTION_ERROR = "execution_error"     # other runtime errors


@dataclass
class ExecutionResult:
    """Result of executing a SQL query against a SQLite database."""

    sql: str
    error_type: ErrorType
    error_message: str = ""
    result_rows: list = field(default_factory=list)
    row_count: int = 0

    @property
    def success(self) -> bool:
        return self.error_type == ErrorType.SUCCESS

    @property
    def failed(self) -> bool:
        return self.error_type != ErrorType.SUCCESS


# ---------------------------------------------------------------------------
# Error message pattern matchers
# ---------------------------------------------------------------------------

_NO_SUCH_TABLE_RE = re.compile(r"no such table:\s*(\S+)", re.IGNORECASE)
_NO_SUCH_COLUMN_RE = re.compile(r"no such column:\s*(\S+)", re.IGNORECASE)
_SYNTAX_MARKERS = (
    "syntax error",
    "incomplete input",
    "unrecognized token",
    "near ",
)


def _classify_error(msg: str) -> tuple[ErrorType, str]:
    """Classify any execution error message string into an ErrorType."""
    msg_lower = msg.lower()

    if _NO_SUCH_TABLE_RE.search(msg):
        return ErrorType.NO_SUCH_TABLE, msg

    if _NO_SUCH_COLUMN_RE.search(msg):
        return ErrorType.NO_SUCH_COLUMN, msg

    if any(marker in msg_lower for marker in _SYNTAX_MARKERS):
        return ErrorType.SYNTAX_ERROR, msg

    return ErrorType.EXECUTION_ERROR, msg


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class SQLExecutor:
    """Execute SQL against SQLite databases and classify outcomes.

    Parameters
    ----------
    db_dir : str
        Base directory containing per-database subdirectories.
        Each DB lives at ``{db_dir}/{db_id}/{db_id}.sqlite``.
    """

    def __init__(self, db_dir: str):
        self.db_dir = Path(db_dir)

    # ---- public API ---------------------------------------------------------

    def execute(
        self,
        sql: str,
        db_id: str,
        gold_sql: str | None = None,
    ) -> ExecutionResult:
        """Execute *sql* against *db_id* and return a classified result.

        Parameters
        ----------
        sql : str
            Predicted SQL query to execute.
        db_id : str
            Database identifier (subfolder name under ``db_dir``).
        gold_sql : str, optional
            Gold SQL — if provided the result set is compared and the
            execution result is classified as SUCCESS or WRONG_RESULT.

        Returns
        -------
        ExecutionResult
            Fully populated result with error_type, error_message, rows.
        """
        if not sql or not sql.strip():
            return ExecutionResult(
                sql=sql,
                error_type=ErrorType.SYNTAX_ERROR,
                error_message="Empty SQL string",
            )

        db_path = self._resolve_db_path(db_id)
        if db_path is None:
            return ExecutionResult(
                sql=sql,
                error_type=ErrorType.EXECUTION_ERROR,
                error_message=f"Database file not found for db_id='{db_id}'",
            )

        rows, exec_err = safe_execute(db_path, sql)

        if exec_err is not None:
            error_type, error_msg = _classify_error(exec_err)
            return ExecutionResult(
                sql=sql,
                error_type=error_type,
                error_message=error_msg,
            )

        if len(rows) == 0:
            # Check if gold also returns empty (then it's a match)
            if gold_sql is not None:
                gold_result = self._run_gold(gold_sql, db_path)
                if gold_result is not None and len(gold_result) == 0:
                    return ExecutionResult(
                        sql=sql,
                        error_type=ErrorType.SUCCESS,
                        result_rows=rows,
                        row_count=0,
                    )
            return ExecutionResult(
                sql=sql,
                error_type=ErrorType.EMPTY_RESULT,
                result_rows=rows,
                row_count=0,
            )

        # If gold provided: compare result sets
        if gold_sql is not None:
            gold_rows = self._run_gold(gold_sql, db_path)
            if gold_rows is not None:
                pred_set = set(map(tuple, rows))
                gold_set = set(map(tuple, gold_rows))
                if pred_set == gold_set:
                    return ExecutionResult(
                        sql=sql,
                        error_type=ErrorType.SUCCESS,
                        result_rows=rows,
                        row_count=len(rows),
                    )
                else:
                    return ExecutionResult(
                        sql=sql,
                        error_type=ErrorType.WRONG_RESULT,
                        error_message=(
                            f"Result mismatch: got {len(rows)} rows, "
                            f"gold has {len(gold_rows)} rows"
                        ),
                        result_rows=rows,
                        row_count=len(rows),
                    )

        # No gold provided: execution succeeded
        return ExecutionResult(
            sql=sql,
            error_type=ErrorType.SUCCESS,
            result_rows=rows,
            row_count=len(rows),
        )

    # ---- helpers ------------------------------------------------------------

    def _resolve_db_path(self, db_id: str) -> Path | None:
        """Return path to the SQLite file for *db_id*, or None if not found."""
        candidates = [
            self.db_dir / db_id / f"{db_id}.sqlite",
            self.db_dir / f"{db_id}.sqlite",
            self.db_dir / db_id / f"{db_id}.db",
        ]
        for p in candidates:
            if p.exists():
                return p
        logger.warning("SQLite file not found for db_id='%s' in '%s'", db_id, self.db_dir)
        return None

    @staticmethod
    def _run_gold(gold_sql: str, db_path: Path) -> list | None:
        """Execute gold SQL and return rows, or None on error."""
        rows, err = safe_execute(db_path, gold_sql)
        if err:
            logger.debug("Gold SQL execution error: %s", err)
            return None
        return rows
