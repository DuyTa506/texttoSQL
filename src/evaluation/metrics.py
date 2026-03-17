"""
Evaluation Metrics for Text-to-SQL.

  - Execution Accuracy (EX): compare SQL execution results
  - Exact Match (EM): compare normalized SQL strings
  - Schema Recall / Precision: measure retrieval quality
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
import sqlparse

from src.shared.sqlite_utils import safe_execute

logger = logging.getLogger(__name__)


# =============================================================================
# Execution Accuracy (EX)
# =============================================================================

def execution_accuracy(
    predicted_sql: str,
    gold_sql: str,
    db_path: str,
) -> bool:
    """Return True if predicted and gold SQL produce the same result sets."""
    if not predicted_sql.strip() or not Path(db_path).exists():
        return False
    try:
        gold_rows, err = safe_execute(db_path, gold_sql)
        if err:
            logger.debug("EX gold SQL error: %s", err)
            return False
        pred_rows, err = safe_execute(db_path, predicted_sql)
        if err:
            logger.debug("EX predicted SQL error: %s", err)
            return False
        return set(map(tuple, gold_rows)) == set(map(tuple, pred_rows))
    except Exception as e:
        logger.debug("EX evaluation error: %s", e)
        return False


# =============================================================================
# Exact Match (EM)
# =============================================================================

def _normalize_sql(sql: str) -> str:
    """Normalize SQL for exact-match comparison."""
    sql = sql.strip().rstrip(";").strip()
    sql = sqlparse.format(sql, keyword_case="upper", strip_comments=True)
    # Collapse whitespace
    sql = re.sub(r"\s+", " ", sql).strip()
    return sql


def exact_match(predicted_sql: str, gold_sql: str) -> bool:
    """Return True if normalized SQL strings match."""
    return _normalize_sql(predicted_sql) == _normalize_sql(gold_sql)


# =============================================================================
# Schema Recall / Precision
# =============================================================================

def _extract_tables_from_sql(sql: str) -> set[str]:
    """Extract table names referenced in SQL (heuristic)."""
    # Match tables after FROM or JOIN keywords
    pattern = r"(?:FROM|JOIN)\s+(\w+)"
    return {m.lower() for m in re.findall(pattern, sql, re.IGNORECASE)}


def schema_recall(
    retrieved_tables: set[str],
    gold_sql: str,
) -> float:
    """Fraction of gold-referenced tables that were retrieved."""
    gold_tables = _extract_tables_from_sql(gold_sql)
    if not gold_tables:
        return 1.0
    retrieved_lower = {t.lower() for t in retrieved_tables}
    found = gold_tables & retrieved_lower
    return len(found) / len(gold_tables)


def schema_precision(
    retrieved_tables: set[str],
    gold_sql: str,
) -> float:
    """Fraction of retrieved tables that are actually used in gold SQL."""
    gold_tables = _extract_tables_from_sql(gold_sql)
    if not retrieved_tables:
        return 0.0
    retrieved_lower = {t.lower() for t in retrieved_tables}
    relevant = retrieved_lower & gold_tables
    return len(relevant) / len(retrieved_lower)


# =============================================================================
# Aggregate metrics
# =============================================================================

def compute_metrics(
    results: list[dict],
) -> dict[str, float]:
    """Compute aggregate metrics from a list of evaluation results.

    Each dict in *results* should have:
      ``ex`` (bool), ``em`` (bool), ``recall`` (float), ``precision`` (float).

    Optional fields (added by Phase 4):
      ``retry_count`` (int)       — number of retries for this example
      ``correction_applied`` (bool) — whether a correction changed the SQL
    """
    n = len(results) or 1
    metrics = {
        "execution_accuracy": sum(r.get("ex", False) for r in results) / n,
        "exact_match": sum(r.get("em", False) for r in results) / n,
        "schema_recall": sum(r.get("recall", 0.0) for r in results) / n,
        "schema_precision": sum(r.get("precision", 0.0) for r in results) / n,
        "total_examples": len(results),
    }

    # Optional Phase 4 retry metrics (only included if any result has retry_count)
    if any("retry_count" in r for r in results):
        metrics["avg_retry_count"] = (
            sum(r.get("retry_count", 0) for r in results) / n
        )
        metrics["examples_retried"] = sum(
            1 for r in results if r.get("retry_count", 0) > 0
        )
        metrics["corrections_applied"] = sum(
            1 for r in results if r.get("correction_applied", False)
        )

    return metrics
