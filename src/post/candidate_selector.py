"""
Candidate Selector – selects the best SQL from multiple candidates using
execution consistency (majority vote on result sets).

Used when generation.n_candidates > 1.

Strategy: "execution_consistency" (default)
  1. Execute all candidates
  2. Cluster by result-set hash (frozenset of result tuples)
  3. Pick the candidate from the largest cluster
  4. Tie-break: pick the shortest SQL (Occam's razor)
  5. Falls back to the first candidate if all executions fail
"""

from __future__ import annotations

import hashlib
import logging
from .sql_executor import ErrorType, SQLExecutor

logger = logging.getLogger(__name__)


class CandidateSelector:
    """Select the best SQL from a list of candidates.

    Parameters
    ----------
    executor : SQLExecutor
        Executor instance for running SQL against SQLite.
    strategy : str
        Selection strategy. Currently only ``"execution_consistency"`` is
        implemented (majority vote on result sets).
    """

    def __init__(
        self,
        executor: SQLExecutor,
        strategy: str = "execution_consistency",
    ):
        self.executor = executor
        self.strategy = strategy

    # ---- public API ---------------------------------------------------------

    def select(
        self,
        candidates: list[dict],
        db_id: str,
    ) -> dict:
        """Select the best candidate from *candidates*.

        Parameters
        ----------
        candidates : list[dict]
            Each element is a dict from SQLInference.generate() with key ``sql``.
        db_id : str
            Database identifier for execution.

        Returns
        -------
        dict
            The selected candidate dict, extended with:
              - ``selected_by`` (str): selection strategy used
              - ``n_candidates`` (int): total candidates evaluated
              - ``cluster_size`` (int): size of the winning cluster
        """
        if not candidates:
            raise ValueError("candidates list is empty")

        if len(candidates) == 1:
            result = dict(candidates[0])
            result.update(selected_by=self.strategy, n_candidates=1, cluster_size=1)
            return result

        if self.strategy == "execution_consistency":
            return self._execution_consistency_select(candidates, db_id)

        # Fallback: return first candidate
        logger.warning(
            "CandidateSelector: unknown strategy '%s', returning first candidate.",
            self.strategy,
        )
        result = dict(candidates[0])
        result.update(
            selected_by="fallback",
            n_candidates=len(candidates),
            cluster_size=1,
        )
        return result

    # ---- strategies ---------------------------------------------------------

    def _execution_consistency_select(
        self,
        candidates: list[dict],
        db_id: str,
    ) -> dict:
        """Majority vote on result-set content."""
        # Execute all candidates and group by result-set hash
        clusters: dict[str, list[int]] = {}  # hash → list of candidate indices
        hashes: list[str | None] = []

        for i, candidate in enumerate(candidates):
            sql = candidate.get("sql", "")
            exec_result = self.executor.execute(sql, db_id)

            if exec_result.failed or exec_result.error_type == ErrorType.EMPTY_RESULT:
                hashes.append(None)
                continue

            # Hash the result set (order-independent)
            result_hash = self._hash_result_set(exec_result.result_rows)
            hashes.append(result_hash)
            clusters.setdefault(result_hash, []).append(i)
            logger.debug(
                "CandidateSelector: candidate %d hash=%s rows=%d",
                i,
                result_hash[:8],
                exec_result.row_count,
            )

        # Pick the largest cluster
        if clusters:
            best_hash = max(clusters, key=lambda h: len(clusters[h]))
            winning_indices = clusters[best_hash]
            cluster_size = len(winning_indices)

            # Tie-break within cluster: shortest SQL
            best_idx = min(
                winning_indices,
                key=lambda i: len(candidates[i].get("sql", "")),
            )
            logger.info(
                "CandidateSelector: selected candidate %d (cluster=%d/%d)",
                best_idx,
                cluster_size,
                len(candidates),
            )
        else:
            # All candidates failed to execute — fall back to first
            best_idx = 0
            cluster_size = 0
            logger.warning(
                "CandidateSelector: all %d candidates failed execution, "
                "returning first.",
                len(candidates),
            )

        result = dict(candidates[best_idx])
        result.update(
            selected_by=self.strategy,
            n_candidates=len(candidates),
            cluster_size=cluster_size,
        )
        return result

    # ---- helpers ------------------------------------------------------------

    @staticmethod
    def _hash_result_set(rows: list) -> str:
        """Produce an order-independent hash of a SQL result set."""
        # Convert rows to a frozenset of tuples for order independence
        # then hash the canonical string representation
        row_set = frozenset(tuple(str(cell) for cell in row) for row in rows)
        canonical = repr(sorted(row_set))
        return hashlib.sha256(canonical.encode()).hexdigest()
