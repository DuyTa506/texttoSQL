"""
Bidirectional Schema Linker – expands retrieved schema via FK relationships.

Implements the table-first → FK-expansion strategy from RSL-SQL:
  1. Start with top-N retrieved tables
  2. For each table, follow FK edges to add related tables
  3. Merge and deduplicate
"""

from __future__ import annotations

import logging
from typing import Optional

from ..data.base_adapter import Database
from ..data.schema_chunker import SchemaChunk

logger = logging.getLogger(__name__)


class BidirectionalLinker:
    """Expand retrieved schema elements via foreign-key graph traversal."""

    def __init__(self, max_expansion_depth: int = 1):
        """
        Parameters
        ----------
        max_expansion_depth : int
            Number of FK hops to follow.  1 = direct neighbours only.
        """
        self.max_expansion_depth = max_expansion_depth

    def expand(
        self,
        retrieved_chunks: list[dict],
        db: Database,
        all_chunks: list[SchemaChunk],
    ) -> list[dict]:
        """Expand retrieved results by adding FK-related tables.

        Parameters
        ----------
        retrieved_chunks : list[dict]
            Results from HybridRetriever (must have ``chunk`` key).
        db : Database
            Full database schema (used for FK graph).
        all_chunks : list[SchemaChunk]
            Complete chunk list for this db_id (used to pull extra chunks).

        Returns
        -------
        list[dict]
            Original + expanded chunks, deduplicated.
        """
        # Collect already-retrieved table names
        seen_tables: set[str] = set()
        for item in retrieved_chunks:
            chunk: Optional[SchemaChunk] = item.get("chunk")
            if chunk and chunk.table_name:
                seen_tables.add(chunk.table_name.lower())

        # BFS through FK graph
        tables_to_add: set[str] = set()
        frontier = set(seen_tables)

        for _depth in range(self.max_expansion_depth):
            next_frontier: set[str] = set()
            for t_name in frontier:
                neighbours = db.get_fk_neighbours(t_name)
                for nb in neighbours:
                    nb_low = nb.lower()
                    if nb_low not in seen_tables and nb_low not in tables_to_add:
                        tables_to_add.add(nb_low)
                        next_frontier.add(nb_low)
            frontier = next_frontier
            if not frontier:
                break

        if tables_to_add:
            logger.info(
                "FK expansion added %d new tables: %s",
                len(tables_to_add),
                tables_to_add,
            )

        # Pull chunks for newly discovered tables
        expansion_chunks = []
        for chunk in all_chunks:
            if (
                chunk.db_id == db.db_id
                and chunk.table_name.lower() in tables_to_add
            ):
                expansion_chunks.append(
                    {
                        "id": f"{chunk.db_id}__{chunk.chunk_type}__{chunk.table_name}__fk_expand",
                        "content": chunk.content,
                        "chunk": chunk,
                        "score": 0.0,  # no relevance score for expanded
                        "source": "fk_expansion",
                    }
                )

        # Merge: original first, then expansions (deduplicated by content)
        seen_content: set[str] = {item["content"] for item in retrieved_chunks}
        merged = list(retrieved_chunks)
        for item in expansion_chunks:
            if item["content"] not in seen_content:
                seen_content.add(item["content"])
                merged.append(item)

        return merged
