"""
Shared SQLite utilities.

Provides a single ``safe_execute()`` helper that encapsulates the
connect → PRAGMA → execute → fetchall → close pattern used across:
  - src/evaluation/metrics.py   (execution_accuracy)
  - src/post/sql_executor.py    (SQLExecutor.execute, SQLExecutor._run_gold)

Usage
-----
    from src.shared.sqlite_utils import safe_execute

    rows, err = safe_execute(db_path, sql)
    if err:
        ...handle error...
    else:
        ...use rows...
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


def safe_execute(
    db_path: str | Path,
    sql: str,
) -> tuple[list, str | None]:
    """Execute *sql* against the SQLite database at *db_path*.

    Opens a read-only connection (PRAGMA read_uncommitted), executes the
    query, fetches all results, and closes the connection — all in a
    context manager to guarantee cleanup.

    Parameters
    ----------
    db_path : str | Path
        Path to the ``.sqlite`` or ``.db`` file.
    sql : str
        SQL query to execute.

    Returns
    -------
    tuple[list, str | None]
        ``(rows, None)`` on success — *rows* is a ``list`` of tuples.
        ``([], error_message)`` on any exception.
    """
    try:
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("PRAGMA read_uncommitted = true")
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
        return rows, None
    except Exception as exc:
        return [], str(exc)
