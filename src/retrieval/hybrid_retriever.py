"""
Hybrid Retriever – combines BM25 sparse retrieval with ChromaDB dense
retrieval, merged via Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import logging
from rank_bm25 import BM25Okapi

from ..schema.schema_chunker import SchemaChunk
from ..schema.schema_indexer import SchemaIndexer
from .npmi_scorer import NPMIScorer

logger = logging.getLogger(__name__)


class HybridRetriever:
    """BM25 + semantic search → RRF merge."""

    def __init__(
        self,
        indexer: SchemaIndexer,
        chunks: list[SchemaChunk],
        bm25_top_k: int = 30,
        semantic_top_k: int = 30,
        rrf_k: int = 60,
        npmi_scorer: NPMIScorer | None = None,
        npmi_top_k: int = 30,
    ):
        """
        Parameters
        ----------
        indexer : SchemaIndexer
            Initialised ChromaDB indexer for dense retrieval.
        chunks : list[SchemaChunk]
            All chunks (needed for BM25 corpus building).
        bm25_top_k, semantic_top_k : int
            How many results to pull from each retriever before fusion.
        rrf_k : int
            RRF constant (default 60, standard value from the original paper).
        npmi_scorer : NPMIScorer, optional
            Pre-built NPMI scorer for statistical schema linking.
        npmi_top_k : int
            How many NPMI results before fusion.
        """
        self.indexer = indexer
        self.chunks = chunks
        self.bm25_top_k = bm25_top_k
        self.semantic_top_k = semantic_top_k
        self.rrf_k = rrf_k
        self.npmi_scorer = npmi_scorer
        self.npmi_top_k = npmi_top_k

        # Build BM25 index
        self._chunk_by_id: dict[str, SchemaChunk] = {}
        self._bm25_ids: list[str] = []
        corpus: list[list[str]] = []
        for i, chunk in enumerate(chunks):
            cid = f"{chunk.db_id}__{chunk.chunk_type}__{chunk.table_name}__{i}"
            self._chunk_by_id[cid] = chunk
            self._bm25_ids.append(cid)
            corpus.append(chunk.content.lower().split())

        self._bm25 = BM25Okapi(corpus)
        logger.info("BM25 index built with %d documents", len(corpus))

    # ---- public API ---------------------------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        db_id: str | None = None,
    ) -> list[dict]:
        """Run hybrid retrieval and return merged results.

        Returns list of dicts: {id, content, chunk, score, source}.
        """
        bm25_results = self._bm25_search(query, db_id=db_id)
        semantic_results = self._semantic_search(query, db_id=db_id)

        result_lists = [bm25_results, semantic_results]

        # Add NPMI signal if available
        if self.npmi_scorer is not None:
            npmi_results = self._npmi_search(query, db_id=db_id)
            result_lists.append(npmi_results)

        merged = self._rrf_merge(*result_lists)
        return merged

    # ---- BM25 ---------------------------------------------------------------

    def _bm25_search(
        self, query: str, *, db_id: str | None = None
    ) -> list[dict]:
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        # Rank and filter
        scored_ids = list(zip(self._bm25_ids, scores))
        if db_id:
            scored_ids = [(cid, s) for cid, s in scored_ids if cid.startswith(f"{db_id}__")]
        scored_ids.sort(key=lambda x: x[1], reverse=True)
        top = scored_ids[: self.bm25_top_k]

        results = []
        for cid, score in top:
            chunk = self._chunk_by_id.get(cid)
            if chunk:
                results.append(
                    {
                        "id": cid,
                        "content": chunk.content,
                        "chunk": chunk,
                        "score": float(score),
                        "source": "bm25",
                    }
                )
        return results

    # ---- Semantic -----------------------------------------------------------

    def _semantic_search(
        self, query: str, *, db_id: str | None = None
    ) -> list[dict]:
        raw = self.indexer.query(query, top_k=self.semantic_top_k, db_id=db_id)
        results = []
        for item in raw:
            chunk = self._chunk_by_id.get(item["id"])
            results.append(
                {
                    "id": item["id"],
                    "content": item["content"],
                    "chunk": chunk,
                    "score": 1.0 - item.get("distance", 0.0),  # cosine → similarity
                    "source": "semantic",
                }
            )
        return results

    # ---- RRF merge ----------------------------------------------------------

    @staticmethod
    def _rrf_merge(
        *result_lists: list[dict],
        k: int | None = None,
    ) -> list[dict]:
        """Reciprocal Rank Fusion: ``score = Σ 1/(k + rank_i)``."""
        k = k or 60
        rrf_scores: dict[str, float] = {}
        best_item: dict[str, dict] = {}

        for results in result_lists:
            for rank, item in enumerate(results, start=1):
                cid = item["id"]
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
                if cid not in best_item:
                    best_item[cid] = item

        # Sort by fused score
        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

        merged = []
        for cid in sorted_ids:
            entry = {**best_item[cid]}
            entry["score"] = rrf_scores[cid]
            entry["source"] = "rrf"
            merged.append(entry)

        return merged

    def retrieve_multi(
        self,
        queries: list[str],
        *,
        db_id: str | None = None,
    ) -> list[dict]:
        """Run hybrid retrieval for multiple queries and merge results via RRF.

        Used by the decompose strategy when a question is split into sub-queries.
        Each sub-query is retrieved independently, then all result lists are merged
        via a final RRF pass, and duplicates are removed by content.

        Parameters
        ----------
        queries : list[str]
            Sub-queries produced by QuestionDecomposer / QueryAugmentor.
        db_id : str, optional
            Filter results to this database.
    
        Returns
        -------
        list[dict]
            Merged, deduplicated, RRF-ranked results.
        """
        if not queries:
            return []

        if len(queries) == 1:
            return self.retrieve(queries[0], db_id=db_id)

        # Retrieve for each sub-query
        per_query_results: list[list[dict]] = [
            self.retrieve(q, db_id=db_id) for q in queries
        ]

        # Merge all result lists via one final RRF pass
        merged = self._rrf_merge(*per_query_results, k=self.rrf_k)

        # Deduplicate by content (keep highest-scored occurrence)
        seen_content: set[str] = set()
        deduped: list[dict] = []
        for item in merged:
            content = item.get("content", "")
            if content not in seen_content:
                seen_content.add(content)
                deduped.append(item)

        return deduped

    # ---- NPMI ---------------------------------------------------------------

    def _npmi_search(
        self, query: str, *, db_id: str | None = None
    ) -> list[dict]:
        """Score chunks using NPMI co-occurrence statistics."""
        # Filter chunks by db_id
        if db_id:
            target_chunks = [c for c in self.chunks if c.db_id == db_id]
        else:
            target_chunks = self.chunks

        return self.npmi_scorer.score_chunks(
            query, target_chunks, db_id=db_id, top_k=self.npmi_top_k
        )
