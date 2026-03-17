"""
Schema Filter – selects top-K schema elements and formats them for the LLM prompt.
"""

from __future__ import annotations

from ...schema.models import Database
from ...schema.schema_chunker import SchemaChunk


class SchemaFilter:
    """Filter and format retrieved schema chunks for LLM consumption."""

    def __init__(self, top_k: int = 15):
        self.top_k = top_k

    def filter_and_format(
        self,
        retrieved_chunks: list[dict],
        db: Database,
        *,
        top_k: int | None = None,
        value_hints: str | None = None,
    ) -> str:
        """Select top-K chunks and produce a structured schema string.

        Parameters
        ----------
        retrieved_chunks : list[dict]
            Merged results from HybridRetriever + BidirectionalLinker.
        db : Database
            Full database schema (for supplementary info).
        top_k : int, optional
            Override instance-level top_k.
        value_hints : str, optional
            SQL comment hints from ValueScanner (appended after FK summary).
            Example: "-- students.name likely contains: 'Alice'"

        Returns
        -------
        str
            Formatted schema string ready for LLM prompt.
        """
        sorted_chunks = self._select_top_chunks(retrieved_chunks, top_k or self.top_k)
        table_chunks = self._group_by_table(sorted_chunks)

        lines: list[str] = [f"Database: {db.db_id}", ""]
        for table_name, chunks in table_chunks.items():
            lines.extend(self._format_table_block(table_name, chunks, sorted_chunks, db))
            lines.append("")

        lines.extend(self._format_fk_summary(db, table_chunks))
        lines.extend(self._inject_value_hints(value_hints))

        return "\n".join(lines)

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _select_top_chunks(
        retrieved_chunks: list[dict],
        k: int,
    ) -> list[dict]:
        """Sort by score descending and return the top-k items."""
        return sorted(
            retrieved_chunks,
            key=lambda x: x.get("score", 0.0),
            reverse=True,
        )[:k]

    @staticmethod
    def _group_by_table(
        sorted_chunks: list[dict],
    ) -> dict[str, list[SchemaChunk]]:
        """Group retrieved chunks by table name, preserving score order."""
        table_chunks: dict[str, list[SchemaChunk]] = {}
        for item in sorted_chunks:
            chunk: SchemaChunk | None = item.get("chunk")
            if chunk is None:
                continue
            tname = chunk.table_name or "__other__"
            table_chunks.setdefault(tname, []).append(chunk)
        return table_chunks

    @staticmethod
    def _collect_col_scores(
        table_name: str,
        sorted_chunks: list[dict],
    ) -> dict[str, float]:
        """Return per-column scores for *table_name* from GraphRetriever chunks.

        Only chunk dicts that carry ``chunk_type == "column"`` and a column name
        are counted.  HybridRetriever chunks have no column-level entries, so the
        returned dict will be empty — the caller falls back to showing all columns.
        """
        col_scores: dict[str, float] = {}
        for item in sorted_chunks:
            c = item.get("chunk")
            if (
                c is not None
                and getattr(c, "table_name", None) == table_name
                and getattr(c, "chunk_type", None) == "column"
                and getattr(c, "column_name", None)
            ):
                col_scores[c.column_name] = item.get("score", 0.0)
        return col_scores

    @staticmethod
    def _cols_to_show(
        table_name: str,
        table_obj,
        col_scores: dict[str, float],
        fk_cols: set[str],
    ) -> list:
        """Decide which columns to show.

        GraphRetriever path (col_scores non-empty): only scored + PK + FK cols.
        HybridRetriever path (col_scores empty): all columns (backward-compat).
        """
        if col_scores:
            return [
                c for c in table_obj.columns
                if c.name in col_scores or c.primary_key or c.name in fk_cols
            ]
        return table_obj.columns

    def _format_table_block(
        self,
        table_name: str,
        chunks: list[SchemaChunk],
        sorted_chunks: list[dict],
        db: Database,
    ) -> list[str]:
        """Render a single table as CREATE TABLE + optional FK comment lines."""
        lines: list[str] = []
        table_obj = db.get_table(table_name)

        if table_obj:
            col_scores = self._collect_col_scores(table_name, sorted_chunks)

            fk_cols: set[str] = set()
            for fk in db.foreign_keys:
                if fk.from_table == table_name:
                    fk_cols.add(fk.from_column)
                if fk.to_table == table_name:
                    fk_cols.add(fk.to_column)

            visible_cols = self._cols_to_show(table_name, table_obj, col_scores, fk_cols)
            cols_str = ", ".join(
                f"{c.name} ({c.dtype})" + (" [PK]" if c.primary_key else "")
                for c in visible_cols
            )
            lines.append(f"CREATE TABLE {table_name} ({cols_str})")
        else:
            lines.append(f"TABLE {table_name}")

        for chunk in chunks:
            if chunk.chunk_type == "fk":
                lines.append(f"  -- {chunk.content}")

        return lines

    @staticmethod
    def _format_fk_summary(
        db: Database,
        table_chunks: dict[str, list[SchemaChunk]],
    ) -> list[str]:
        """Render the FK summary block for all retrieved tables."""
        if not db.foreign_keys:
            return []
        retrieved_tables = {tn.lower() for tn in table_chunks}
        fk_lines = [
            f"  {fk.from_table}.{fk.from_column} = {fk.to_table}.{fk.to_column}"
            for fk in db.foreign_keys
            if fk.from_table.lower() in retrieved_tables
            or fk.to_table.lower() in retrieved_tables
        ]
        if not fk_lines:
            return []
        return ["Foreign Keys:"] + fk_lines

    @staticmethod
    def _inject_value_hints(value_hints: str | None) -> list[str]:
        """Return value-hint comment lines, or empty list if none."""
        if not value_hints or not value_hints.strip():
            return []
        return [
            "",
            "-- Value hints (likely cell values for this question):",
            value_hints.strip(),
        ]

    def get_retrieved_tables(self, retrieved_chunks: list[dict]) -> set[str]:
        """Return the set of table names in the retrieved chunks."""
        tables: set[str] = set()
        for item in retrieved_chunks:
            chunk: SchemaChunk | None = item.get("chunk")
            if chunk and chunk.table_name:
                tables.add(chunk.table_name)
        return tables
