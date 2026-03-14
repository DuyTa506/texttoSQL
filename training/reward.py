"""
Reward Functions – multi-signal rewards for RL-based SQL training (GRPO/DPO).

Qwen3 Thinking Mode: model outputs <think>reasoning</think> then SQL.
Two usage patterns supported:
  1. GRPO reward_funcs (list of callables) → in rl_trainer.py
  2. RewardFunction class (composite score) → for evaluation & DPO

Reward signals:
  1. Execution Accuracy — compare SQL execution results against gold
  2. Format Check        — SQL syntax validity (sqlparse)
  3. Schema Faithfulness — only reference provided schema elements
  4. Reasoning Quality   — <think> block mentions relevant tables/columns
"""

from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path

import sqlparse

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Parse helpers for Qwen3 output format
# ──────────────────────────────────────────────────────────────

THINK_RE = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL)
SQL_BLOCK_RE = re.compile(r"```sql\s*(.+?)\s*```", flags=re.DOTALL)

FULL_FORMAT_RE = re.compile(
    r"</think>.*?```sql\s*(.+?)\s*```[\s]*(?:<\|endoftext\|>)?[\s]*$",
    flags=re.MULTILINE | re.DOTALL,
)



def extract_thinking(text: str) -> str:
    """Extract content from <think>...</think> block."""
    match = THINK_RE.search(text)
    return match.group(1).strip() if match else ""


def extract_sql(text: str) -> str:
    """Extract SQL from ```sql...``` block, fallback to 'SQL:' prefix."""
    match = SQL_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()
    # Fallback: look for SQL: prefix (legacy format)
    for line in reversed(text.strip().split("\n")):
        if line.strip().upper().startswith("SQL:"):
            return line.strip()[4:].strip()
    return ""


# ──────────────────────────────────────────────────────────────
# RewardFunction class (composite score for evaluation & DPO)
# ──────────────────────────────────────────────────────────────

class RewardFunction:
    """Multi-signal reward function for RL training and evaluation.

    Computes weighted sum of 4 reward components.
    Works with both Qwen3 <think> format and legacy CoT format.
    """

    def __init__(
        self,
        weight_execution: float = 0.5,
        weight_format: float = 0.2,
        weight_schema: float = 0.15,
        weight_reasoning: float = 0.15,
    ):
        self.w_exec = weight_execution
        self.w_format = weight_format
        self.w_schema = weight_schema
        self.w_reason = weight_reasoning

    def compute(
        self,
        predicted_sql: str,
        gold_sql: str,
        db_path: str,
        schema_context: str,
        cot_output: str = "",
    ) -> float:
        """Compute composite reward score in [0, 1]."""
        # If predicted_sql is empty, try extracting from cot_output
        if not predicted_sql.strip() and cot_output:
            predicted_sql = extract_sql(cot_output)

        r_exec = self.execution_accuracy(predicted_sql, gold_sql, db_path)
        r_format = self.format_check(predicted_sql)
        r_schema = self.schema_faithfulness(predicted_sql, schema_context)

        # Extract thinking block for reasoning quality
        thinking = extract_thinking(cot_output) if cot_output else ""
        r_reason = self.reasoning_quality(thinking or cot_output, schema_context)

        total = (
            self.w_exec * r_exec
            + self.w_format * r_format
            + self.w_schema * r_schema
            + self.w_reason * r_reason
        )
        return min(max(total, 0.0), 1.0)

    @staticmethod
    def execution_accuracy(predicted_sql: str, gold_sql: str, db_path: str) -> float:
        """Compare execution results. Returns 1.0 if match, partial credit for overlap."""
        if not predicted_sql.strip() or not Path(db_path).exists():
            return 0.0
        try:
            conn = sqlite3.connect(db_path)
            conn.execute("PRAGMA read_uncommitted = true")
            cursor = conn.cursor()
            cursor.execute(gold_sql)
            gold_results = set(map(tuple, cursor.fetchall()))
            cursor.execute(predicted_sql)
            pred_results = set(map(tuple, cursor.fetchall()))
            conn.close()
            if gold_results == pred_results:
                return 1.0
            if pred_results and gold_results:
                overlap = len(gold_results & pred_results) / max(len(gold_results), 1)
                return overlap * 0.5
            return 0.0
        except Exception as e:
            logger.debug("Execution error: %s", e)
            return 0.0

    @staticmethod
    def format_check(sql: str) -> float:
        """Check SQL syntax validity. 1.0=valid SELECT, 0.5=valid other, 0.0=invalid."""
        sql = sql.strip()
        if not sql:
            return 0.0
        try:
            parsed = sqlparse.parse(sql)
            if not parsed:
                return 0.0
            stmt = parsed[0]
            if stmt.get_type() == "SELECT":
                return 1.0
            elif stmt.get_type() is not None:
                return 0.5
            return 0.3
        except Exception:
            return 0.0

    @staticmethod
    def schema_faithfulness(sql: str, schema_context: str) -> float:
        """Check that SQL only references tables/columns from the schema."""
        if not sql.strip() or not schema_context.strip():
            return 0.0
        sql_upper = sql.upper()
        schema_upper = schema_context.upper()
        identifiers = set(re.findall(r'\b([A-Z_]\w*)\b', sql_upper))
        sql_keywords = {
            "SELECT", "FROM", "WHERE", "AND", "OR", "JOIN", "ON", "LEFT",
            "RIGHT", "INNER", "OUTER", "GROUP", "BY", "ORDER", "ASC",
            "DESC", "HAVING", "LIMIT", "DISTINCT", "AS", "IN", "NOT",
            "NULL", "IS", "LIKE", "BETWEEN", "CASE", "WHEN", "THEN",
            "ELSE", "END", "COUNT", "SUM", "AVG", "MAX", "MIN", "CAST",
            "UNION", "ALL", "EXISTS", "INSERT", "UPDATE", "DELETE", "INTO",
            "VALUES", "SET", "CREATE", "TABLE", "DROP", "ALTER", "INDEX",
            "TRUE", "FALSE", "OFFSET", "EXCEPT", "INTERSECT",
        }
        identifiers -= sql_keywords
        if not identifiers:
            return 1.0
        found = sum(1 for ident in identifiers if ident in schema_upper)
        return found / len(identifiers)

    @staticmethod
    def reasoning_quality(reasoning_text: str, schema_context: str) -> float:
        """Evaluate whether reasoning mentions relevant schema elements.

        Works with both <think> block content and legacy CoT text.
        """
        if not reasoning_text.strip():
            return 0.3
        table_names = set(re.findall(
            r'(?:CREATE TABLE|TABLE)\s+(\w+)', schema_context, re.IGNORECASE,
        ))
        if not table_names:
            return 0.5
        reasoning_upper = reasoning_text.upper()
        mentioned = sum(1 for t in table_names if t.upper() in reasoning_upper)
        return min(mentioned / max(len(table_names), 1), 1.0)


# ──────────────────────────────────────────────────────────────
# Correction Reward Functions (for multitask GRPO)
# These are used when task_type == "correction" in rl_trainer.py
# ──────────────────────────────────────────────────────────────


def check_correction_improvement(completions, answer, wrong_sql=None, **kwargs):
    """Reward for actually improving the wrong SQL.

    Scoring:
      +4.0  corrected SQL is different from wrong AND matches gold (answer)
      +2.0  corrected SQL is different from wrong (changed something)
      -1.0  corrected SQL is identical to wrong SQL (no change)
    """
    scores = []
    for i, completion in enumerate(completions):
        response = completion[0]["content"]

        # Extract predicted SQL from response
        match = SQL_BLOCK_RE.search(response)
        if match is None:
            match = FULL_FORMAT_RE.search(response)
        predicted = match.group(1).strip() if match else ""

        if not predicted:
            scores.append(-1.0)
            continue

        # Get wrong_sql for this sample
        if wrong_sql is not None:
            if isinstance(wrong_sql, list):
                bad_sql = wrong_sql[i] if i < len(wrong_sql) else ""
            else:
                bad_sql = str(wrong_sql)
        else:
            bad_sql = ""

        gold = answer[i] if isinstance(answer, list) else str(answer)

        is_different = predicted.strip() != bad_sql.strip()
        is_correct = (
            predicted.strip() == gold.strip()
            or predicted.strip().lower() == gold.strip().lower()
        )

        if is_different and is_correct:
            scores.append(4.0)
        elif is_different:
            scores.append(2.0)
        else:
            scores.append(-1.0)

    return scores


def check_error_addressed(completions, error_type=None, **kwargs):
    """Reward for specifically addressing the error category.

    +1.5 if the specific error type is no longer present in the corrected SQL:
      - no_such_table: no "no such table" pattern in the SQL
      - syntax_error:  SQL parses cleanly via sqlparse
      - no_such_column: no "no such column" pattern (structural check)
    Other error types: +0.5 credit for non-empty SQL (best-effort).
    """
    scores = []
    for i, completion in enumerate(completions):
        response = completion[0]["content"]

        match = SQL_BLOCK_RE.search(response)
        if match is None:
            match = FULL_FORMAT_RE.search(response)
        predicted = match.group(1).strip() if match else ""

        if not predicted:
            scores.append(0.0)
            continue

        # Resolve error_type for this sample
        if error_type is not None:
            if isinstance(error_type, list):
                etype = error_type[i] if i < len(error_type) else ""
            else:
                etype = str(error_type)
        else:
            etype = ""

        score = 0.0

        if etype == "syntax_error":
            # Check if SQL parses cleanly
            try:
                parsed = sqlparse.parse(predicted)
                if parsed and parsed[0].get_type() is not None:
                    score = 1.5
                else:
                    score = 0.5
            except Exception:
                score = 0.0

        elif etype == "no_such_table":
            # Heuristic: check SQL doesn't reference obviously wrong table names
            # (we can't run SQLite here, so use structural check)
            if predicted.upper().startswith("SELECT"):
                score = 1.0  # partial credit — it's a SELECT, error may be fixed
            else:
                score = 0.3

        elif etype == "no_such_column":
            # SQL is syntactically valid SELECT
            if predicted.upper().startswith("SELECT"):
                score = 1.0
            else:
                score = 0.3

        elif etype in ("wrong_result", "empty_result", "execution_error"):
            # Can't verify without executing — give partial credit for non-trivial SQL
            if len(predicted.split()) > 3 and predicted.upper().startswith("SELECT"):
                score = 0.5
            else:
                score = 0.0

        else:
            # Unknown error type — give small credit for any SQL
            score = 0.3 if predicted else 0.0

        scores.append(score)

    return scores
