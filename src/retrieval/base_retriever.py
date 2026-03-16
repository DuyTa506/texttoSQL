"""
BaseRetriever — abstract interface for all Phase 2 schema-linking retrievers.

Both concrete retrievers implement this contract:
  - HybridRetriever  (BM25 + ChromaDB semantic + NPMI → RRF)
  - GraphRetriever   (Personalized PageRank over SchemaGraph)

This allows ``run_pipeline.py`` and any evaluation script to call
``retriever.retrieve(query, db_id=db_id, value_matches=matches)``
without knowing which backend is active.

Return format
-------------
Both ``retrieve`` and ``retrieve_multi`` return ``list[dict]`` where
each dict has the keys:
  id       : str              — unique chunk / node identifier
  content  : str              — human-readable schema text
  chunk    : SchemaChunk | _SyntheticChunk | None
  score    : float            — retrieval score (higher = more relevant)
  source   : str              — e.g. "bm25", "semantic", "graph_ppr", "rrf"
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseRetriever(ABC):
    """Abstract base class for schema-linking retrievers (Phase 2)."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        *,
        db_id: str | None = None,
        value_matches: list | None = None,
    ) -> list[dict]:
        """Retrieve relevant schema chunks / nodes for a single query.

        Parameters
        ----------
        query:
            Natural language question.
        db_id:
            Restrict results to this database.  Pass ``None`` to search
            across all indexed databases (not recommended for large corpora).
        value_matches:
            Optional list of ``ValueMatch`` objects from ``ValueScanner``.
            Retrievers that support value hints (e.g. ``GraphRetriever``)
            use these to boost relevant columns in their scoring.
            Retrievers that do not support them silently ignore the argument.

        Returns
        -------
        list[dict]
            Ranked results, each a dict with keys:
            ``id``, ``content``, ``chunk``, ``score``, ``source``.
        """

    @abstractmethod
    def retrieve_multi(
        self,
        queries: list[str],
        *,
        db_id: str | None = None,
        value_matches: list | None = None,
    ) -> list[dict]:
        """Retrieve and merge results for multiple sub-queries.

        Used by the decompose strategy when ``QuestionDecomposer`` splits a
        complex question into sub-queries.  Each sub-query is retrieved
        independently and the results are fused (typically via RRF).

        Parameters
        ----------
        queries:
            Sub-queries from ``QuestionDecomposer`` / ``QueryAugmentor``.
        db_id:
            Restrict results to this database.
        value_matches:
            Same as in ``retrieve()``.

        Returns
        -------
        list[dict]
            Merged, deduplicated results sorted by fused score.
        """
