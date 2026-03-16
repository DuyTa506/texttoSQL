"""
Schema Filter – selects top-K schema elements and formats them for the LLM prompt.
"""

from __future__ import annotations

from ..schema.models import Database
from ..schema.schema_chunker import SchemaChunk


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
        k = top_k or self.top_k

        # Sort by score descending, take top-k
        sorted_chunks = sorted(
            retrieved_chunks,
            key=lambda x: x.get("score", 0.0),
            reverse=True,
        )[:k]

        # Group by table
        table_chunks: dict[str, list[SchemaChunk]] = {}
        for item in sorted_chunks:
            chunk: SchemaChunk | None = item.get("chunk")
            if chunk is None:
                continue
            tname = chunk.table_name or "__other__"
            table_chunks.setdefault(tname, []).append(chunk)

        # Format output
        lines: list[str] = [f"Database: {db.db_id}"]
        lines.append("")

        for table_name, chunks in table_chunks.items():
            # Get table object for column details
            table_obj = db.get_table(table_name)
            if table_obj:
                # Fix 7: Collect per-column scores from retrieved chunks.
                # GraphRetriever emits individual column nodes as scored dicts;
                # if any column scores are present for this table, show only
                # scored columns + PK/FK columns (precision improvement).
                # HybridRetriever path has no column-level scores → show all
                # columns unchanged (fully backward-compatible).
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

                # FK columns of this table (always include regardless of score)
                fk_cols: set[str] = set()
                for fk in db.foreign_keys:
                    if fk.from_table == table_name:
                        fk_cols.add(fk.from_column)
                    if fk.to_table == table_name:
                        fk_cols.add(fk.to_column)

                if col_scores:
                    # GraphRetriever path: only show scored columns + PK + FK
                    cols_to_show = [
                        c for c in table_obj.columns
                        if c.name in col_scores or c.primary_key or c.name in fk_cols
                    ]
                else:
                    # HybridRetriever path: show all columns (unchanged behaviour)
                    cols_to_show = table_obj.columns

                cols_str = ", ".join(
                    f"{c.name} ({c.dtype})" + (" [PK]" if c.primary_key else "")
                    for c in cols_to_show
                )
                lines.append(f"CREATE TABLE {table_name} ({cols_str})")
            else:
                lines.append(f"TABLE {table_name}")

            # Add FK info if present
            for chunk in chunks:
                if chunk.chunk_type == "fk":
                    lines.append(f"  -- {chunk.content}")

            lines.append("")

        # Add FK summary at the end
        if db.foreign_keys:
            fk_lines = []
            # Only include FKs involving retrieved tables
            retrieved_tables = {tn.lower() for tn in table_chunks}
            for fk in db.foreign_keys:
                if (
                    fk.from_table.lower() in retrieved_tables
                    or fk.to_table.lower() in retrieved_tables
                ):
                    fk_lines.append(
                        f"  {fk.from_table}.{fk.from_column} = {fk.to_table}.{fk.to_column}"
                    )
            if fk_lines:
                lines.append("Foreign Keys:")
                lines.extend(fk_lines)

        # Inject value hints from ValueScanner (if provided)
        if value_hints and value_hints.strip():
            lines.append("")
            lines.append("-- Value hints (likely cell values for this question):")
            lines.append(value_hints.strip())

        return "\n".join(lines)

    def get_retrieved_tables(self, retrieved_chunks: list[dict]) -> set[str]:
        """Return the set of table names in the retrieved chunks."""
        tables: set[str] = set()
        for item in retrieved_chunks:
            chunk: SchemaChunk | None = item.get("chunk")
            if chunk and chunk.table_name:
                tables.add(chunk.table_name)
        return tables
