"""
Tests for RetryLoop – verifies retry logic, prompt building, and early exit.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from src.post.retry_loop import RetryConfig, RetryLoop
from src.post.sql_executor import ErrorType, ExecutionResult


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def make_exec_result(error_type, error_message=""):
    return ExecutionResult(
        sql="SELECT 1",
        error_type=error_type,
        error_message=error_message,
    )


def make_inference_result(sql):
    return {"sql": sql, "reasoning": "test reasoning", "raw_output": sql}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRetryConfig:
    def test_defaults(self):
        cfg = RetryConfig()
        assert cfg.max_retries == 3
        assert cfg.enabled is True

    def test_custom(self):
        cfg = RetryConfig(max_retries=5, enabled=False)
        assert cfg.max_retries == 5
        assert cfg.enabled is False


class TestRetryLoopDisabled:
    def test_disabled_returns_initial_result(self):
        executor = MagicMock()
        inference = MagicMock()
        config = RetryConfig(enabled=False)
        loop = RetryLoop(executor, inference, config)

        initial = make_inference_result("SELECT * FROM t")
        result = loop.run("q", "schema", initial, "db_id")

        assert result["sql"] == "SELECT * FROM t"
        assert result["retry_count"] == 0
        assert result["correction_applied"] is False
        executor.execute.assert_not_called()


class TestRetryLoopEarlyExit:
    def test_early_exit_on_success(self):
        executor = MagicMock()
        executor.execute.return_value = make_exec_result(ErrorType.SUCCESS)
        inference = MagicMock()
        loop = RetryLoop(executor, inference, RetryConfig(max_retries=3))

        initial = make_inference_result("SELECT * FROM students")
        result = loop.run("q", "schema", initial, "db_id")

        assert result["retry_count"] == 0
        assert result["correction_applied"] is False
        # Execute called once (initial check)
        executor.execute.assert_called_once()
        inference.generate.assert_not_called()


class TestRetryLoopRetries:
    def test_retries_on_error(self):
        executor = MagicMock()
        # First call: syntax error; second call: success
        executor.execute.side_effect = [
            make_exec_result(ErrorType.SYNTAX_ERROR, "syntax error"),
            make_exec_result(ErrorType.SUCCESS),
        ]

        inference = MagicMock()
        corrected_sql = "SELECT name FROM students"
        inference.generate.return_value = make_inference_result(corrected_sql)

        loop = RetryLoop(executor, inference, RetryConfig(max_retries=3))
        initial = make_inference_result("SELECT FROM WHERE")
        result = loop.run("q", "schema", initial, "db_id")

        assert result["sql"] == corrected_sql
        assert result["retry_count"] == 1
        assert result["correction_applied"] is True
        assert len(result["retry_history"]) == 1
        assert result["retry_history"][0]["error_type"] == "syntax_error"

    def test_max_retries_respected(self):
        executor = MagicMock()
        # Always fails
        executor.execute.return_value = make_exec_result(
            ErrorType.WRONG_RESULT, "mismatch"
        )

        inference = MagicMock()
        inference.generate.return_value = make_inference_result("SELECT * FROM t2")

        loop = RetryLoop(executor, inference, RetryConfig(max_retries=2))
        initial = make_inference_result("SELECT * FROM t1")
        result = loop.run("q", "schema", initial, "db_id")

        # Should have retried up to max_retries
        assert result["retry_count"] <= 2
        # Execute should be called: 1 initial + up to max_retries times
        assert executor.execute.call_count <= 3

    def test_no_change_stops_loop(self):
        executor = MagicMock()
        executor.execute.return_value = make_exec_result(
            ErrorType.SYNTAX_ERROR, "syntax error"
        )

        inference = MagicMock()
        # Returns the SAME sql as initial (no change)
        initial_sql = "SELECT FROM WHERE"
        inference.generate.return_value = make_inference_result(initial_sql)

        loop = RetryLoop(executor, inference, RetryConfig(max_retries=3))
        initial = make_inference_result(initial_sql)
        result = loop.run("q", "schema", initial, "db_id")

        # Should stop after first retry (no change detected)
        assert inference.generate.call_count == 1


class TestCorrectionPrompt:
    def test_syntax_error_hint(self):
        executor = MagicMock()
        inference = MagicMock()
        loop = RetryLoop(executor, inference)

        error = ExecutionResult(
            sql="bad sql",
            error_type=ErrorType.SYNTAX_ERROR,
            error_message="near 'syntax': syntax error",
        )
        prompt = loop._build_correction_prompt("q", "schema", "bad sql", error)

        assert "syntax error" in prompt.lower()
        assert "bad sql" in prompt
        assert "schema" in prompt
        assert "q" in prompt

    def test_no_such_table_hint_uses_table_name(self):
        executor = MagicMock()
        inference = MagicMock()
        loop = RetryLoop(executor, inference)

        error = ExecutionResult(
            sql="SELECT * FROM fake_table",
            error_type=ErrorType.NO_SUCH_TABLE,
            error_message="no such table: fake_table",
        )
        prompt = loop._build_correction_prompt("q", "schema", "SELECT * FROM fake_table", error)
        assert "fake_table" in prompt

    def test_no_such_column_hint_uses_column_name(self):
        executor = MagicMock()
        inference = MagicMock()
        loop = RetryLoop(executor, inference)

        error = ExecutionResult(
            sql="SELECT bad_col FROM students",
            error_type=ErrorType.NO_SUCH_COLUMN,
            error_message="no such column: bad_col",
        )
        prompt = loop._build_correction_prompt(
            "q", "schema", "SELECT bad_col FROM students", error
        )
        assert "bad_col" in prompt

    def test_template_contains_required_fields(self):
        executor = MagicMock()
        inference = MagicMock()
        loop = RetryLoop(executor, inference)

        error = ExecutionResult(
            sql="SELECT * FROM t",
            error_type=ErrorType.EMPTY_RESULT,
            error_message="returned 0 rows",
        )
        prompt = loop._build_correction_prompt(
            "Who is Alice?", "Database: test\nCREATE TABLE t", "SELECT * FROM t", error
        )
        # Required fields in the template
        assert "Who is Alice?" in prompt
        assert "CREATE TABLE t" in prompt
        assert "SELECT * FROM t" in prompt
        assert "empty_result" in prompt
