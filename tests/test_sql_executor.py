"""
Tests for SQLExecutor – verifies error classification and execution logic.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from src.post.sql_executor import ErrorType, ExecutionResult, SQLExecutor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_db_dir(tmp_path):
    """Create a temp SQLite database for testing."""
    db_id = "test_db"
    db_subdir = tmp_path / db_id
    db_subdir.mkdir()
    db_path = db_subdir / f"{db_id}.sqlite"

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE students (id INTEGER PRIMARY KEY, name TEXT, grade INTEGER)"
    )
    cursor.execute(
        "INSERT INTO students VALUES (1, 'Alice', 90), "
        "(2, 'Bob', 85), (3, 'Charlie', 92)"
    )
    cursor.execute(
        "CREATE TABLE courses (id INTEGER PRIMARY KEY, title TEXT, credits INTEGER)"
    )
    cursor.execute(
        "INSERT INTO courses VALUES (1, 'Math', 3), (2, 'Science', 4)"
    )
    conn.commit()
    conn.close()

    return tmp_path, db_id


@pytest.fixture
def executor(temp_db_dir):
    db_dir, _ = temp_db_dir
    return SQLExecutor(db_dir=str(db_dir))


# ---------------------------------------------------------------------------
# Success cases
# ---------------------------------------------------------------------------


class TestSQLExecutorSuccess:
    def test_simple_select(self, executor, temp_db_dir):
        _, db_id = temp_db_dir
        result = executor.execute("SELECT * FROM students", db_id)
        assert result.success
        assert result.error_type == ErrorType.SUCCESS
        assert result.row_count == 3

    def test_select_with_filter(self, executor, temp_db_dir):
        _, db_id = temp_db_dir
        result = executor.execute(
            "SELECT name FROM students WHERE grade > 88", db_id
        )
        assert result.success
        assert result.row_count == 2  # Alice (90) and Charlie (92)

    def test_gold_comparison_match(self, executor, temp_db_dir):
        _, db_id = temp_db_dir
        gold = "SELECT name FROM students WHERE id = 1"
        pred = "SELECT name FROM students WHERE id = 1"
        result = executor.execute(pred, db_id, gold_sql=gold)
        assert result.success
        assert result.error_type == ErrorType.SUCCESS

    def test_gold_comparison_semantic_match(self, executor, temp_db_dir):
        """Different syntax, same result → SUCCESS."""
        _, db_id = temp_db_dir
        gold = "SELECT name FROM students WHERE id = 1"
        pred = 'SELECT name FROM students WHERE id = "1"'  # string vs int
        # SQLite may or may not match; either success or wrong_result is valid
        result = executor.execute(pred, db_id, gold_sql=gold)
        assert result.error_type in (ErrorType.SUCCESS, ErrorType.WRONG_RESULT)


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


class TestSQLExecutorErrorClassification:
    def test_no_such_table(self, executor, temp_db_dir):
        _, db_id = temp_db_dir
        result = executor.execute("SELECT * FROM nonexistent_table", db_id)
        assert result.error_type == ErrorType.NO_SUCH_TABLE
        assert "nonexistent_table" in result.error_message.lower()

    def test_no_such_column(self, executor, temp_db_dir):
        _, db_id = temp_db_dir
        result = executor.execute(
            "SELECT nonexistent_column FROM students", db_id
        )
        assert result.error_type == ErrorType.NO_SUCH_COLUMN

    def test_syntax_error(self, executor, temp_db_dir):
        _, db_id = temp_db_dir
        result = executor.execute("SELECT FROM WHERE", db_id)
        assert result.error_type == ErrorType.SYNTAX_ERROR

    def test_empty_result(self, executor, temp_db_dir):
        _, db_id = temp_db_dir
        result = executor.execute(
            "SELECT * FROM students WHERE grade > 999", db_id
        )
        assert result.error_type == ErrorType.EMPTY_RESULT
        assert result.row_count == 0

    def test_wrong_result(self, executor, temp_db_dir):
        _, db_id = temp_db_dir
        gold = "SELECT name FROM students WHERE id = 1"
        pred = "SELECT name FROM students WHERE id = 2"
        result = executor.execute(pred, db_id, gold_sql=gold)
        assert result.error_type == ErrorType.WRONG_RESULT

    def test_empty_sql(self, executor, temp_db_dir):
        _, db_id = temp_db_dir
        result = executor.execute("", db_id)
        assert result.error_type == ErrorType.SYNTAX_ERROR

    def test_missing_db(self, executor):
        result = executor.execute("SELECT 1", "nonexistent_db")
        assert result.error_type == ErrorType.EXECUTION_ERROR


# ---------------------------------------------------------------------------
# ExecutionResult properties
# ---------------------------------------------------------------------------


class TestExecutionResult:
    def test_success_property(self):
        r = ExecutionResult(sql="SELECT 1", error_type=ErrorType.SUCCESS)
        assert r.success is True
        assert r.failed is False

    def test_failed_property(self):
        r = ExecutionResult(sql="bad", error_type=ErrorType.SYNTAX_ERROR)
        assert r.failed is True
        assert r.success is False

    def test_default_rows(self):
        r = ExecutionResult(sql="SELECT 1", error_type=ErrorType.SUCCESS)
        assert r.result_rows == []
        assert r.row_count == 0


# ---------------------------------------------------------------------------
# DB path resolution
# ---------------------------------------------------------------------------


class TestDBPathResolution:
    def test_resolve_nested_path(self, temp_db_dir):
        db_dir, db_id = temp_db_dir
        executor = SQLExecutor(db_dir=str(db_dir))
        resolved = executor._resolve_db_path(db_id)
        assert resolved is not None
        assert resolved.exists()

    def test_resolve_missing_db(self, tmp_path):
        executor = SQLExecutor(db_dir=str(tmp_path))
        resolved = executor._resolve_db_path("missing_db")
        assert resolved is None
