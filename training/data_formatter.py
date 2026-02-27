"""
Data Formatter – prepares training data for Qwen3 SFT and GRPO stages.

Qwen3 Thinking Mode format:
  SFT:  instruction + response (with <think> block for reasoning)
  GRPO: prompt + answer (gold SQL for reward comparison)

Can be used independently from training/ folder.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

# Import from src if available, otherwise work standalone
try:
    from src.data.base_adapter import Database, Example
except ImportError:
    Database = None
    Example = None

logger = logging.getLogger(__name__)


@dataclass
class TrainingSample:
    """A single formatted training example."""
    db_id: str
    question: str
    schema_context: str
    gold_sql: str
    schema_tables: str = ""      # comma-separated table names for GRPO reward
    reasoning: str = ""
    difficulty: str = "unknown"

    def to_sft_dict(self) -> dict:
        """Format as instruction-response pair for SFT with Qwen3.

        Response uses ```sql``` block format (Qwen3 fills <think> automatically).
        """
        instruction = (
            f"Given the database schema below, generate SQL to answer the question.\n\n"
            f"Schema:\n{self.schema_context}\n\n"
            f"Question: {self.question}"
        )
        # SQL in code block format — Qwen3 will prepend <think> reasoning
        response = f"```sql\n{self.gold_sql}\n```"
        return {"instruction": instruction, "response": response}

    def to_sft_thinking_dict(self) -> dict:
        """Format SFT with explicit reasoning in <think> block.

        For training examples where we provide the reasoning trace.
        """
        instruction = (
            f"Given the database schema below, generate SQL to answer the question.\n\n"
            f"Schema:\n{self.schema_context}\n\n"
            f"Question: {self.question}"
        )
        # Include reasoning in <think> block + SQL in code block
        thinking = self.reasoning if self.reasoning else self._auto_reasoning()
        response = f"<think>\n{thinking}\n</think>\n\n```sql\n{self.gold_sql}\n```"
        return {"instruction": instruction, "response": response}

    def to_grpo_dict(self) -> dict:
        """Format for GRPO training.

        GRPO needs: prompt (user message) + answer (gold SQL for reward).
        Optional: schema_tables for schema_faithfulness reward.
        """
        prompt = (
            f"Given the database schema below, generate SQL to answer the question.\n\n"
            f"Schema:\n{self.schema_context}\n\n"
            f"Question: {self.question}"
        )
        result = {
            "prompt": prompt,
            "answer": self.gold_sql,
        }
        if self.schema_tables:
            result["schema_tables"] = self.schema_tables
        return result

    def to_dpo_dict(self, chosen: str, rejected: str) -> dict:
        """Format as DPO preference pair."""
        prompt = (
            f"Given the database schema below, generate SQL to answer the question.\n\n"
            f"Schema:\n{self.schema_context}\n\n"
            f"Question: {self.question}"
        )
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

    def _auto_reasoning(self) -> str:
        """Generate minimal reasoning trace from gold SQL."""
        sql_upper = self.gold_sql.upper()
        steps = []
        # Extract table mentions from SQL
        tables = self.schema_tables.split(",") if self.schema_tables else []
        if tables:
            steps.append(f"Tables needed: {', '.join(t.strip() for t in tables)}")
        if "JOIN" in sql_upper:
            steps.append("Need JOINs to connect tables via foreign keys")
        if "WHERE" in sql_upper:
            steps.append("Apply WHERE conditions to filter results")
        if "GROUP BY" in sql_upper:
            steps.append("Group results for aggregation")
        if "ORDER BY" in sql_upper:
            steps.append("Order the results")
        if "HAVING" in sql_upper:
            steps.append("Filter aggregated results with HAVING")
        if "LIMIT" in sql_upper:
            steps.append("Limit output rows")
        return "\n".join(steps) if steps else "Direct query, no complex reasoning needed."


class DataFormatter:
    """Formats examples into training data for Qwen3 SFT and GRPO stages."""

    def format_examples(
        self,
        examples: list,
        databases: dict,
        schema_contexts: dict,
        kd_reasoning_dict: dict | None = None,
    ) -> list[TrainingSample]:
        """Convert raw examples into training samples.
        
        If kd_reasoning_dict is provided, it uses the high-quality LLM-generated 
        reasoning trace instead of the rule-based pseudo-reasoning fallback.
        """
        samples: list[TrainingSample] = []
        for ex in examples:
            stable_hash = hashlib.md5(ex.question.encode("utf-8")).hexdigest()[:8]
            key = f"{ex.db_id}__{stable_hash}"
            
            schema_ctx = schema_contexts.get(key, "")
            db = databases.get(ex.db_id)
            if not schema_ctx and db:
                schema_ctx = self._basic_schema(db)

            # Collect table names for schema faithfulness reward
            table_names = ""
            if db:
                sql_upper = ex.query.upper()
                mentioned = [t.name for t in db.tables if t.name.upper() in sql_upper]
                table_names = ",".join(mentioned)

            # Use KD reasoning if available, else fallback to auto pseudo-reasoning
            if kd_reasoning_dict and key in kd_reasoning_dict:
                reasoning = kd_reasoning_dict[key]
            else:
                reasoning = self._generate_reasoning(ex, db)
                
            samples.append(
                TrainingSample(
                    db_id=ex.db_id,
                    question=ex.question,
                    schema_context=schema_ctx,
                    gold_sql=ex.query,
                    schema_tables=table_names,
                    reasoning=reasoning,
                    difficulty=getattr(ex, "difficulty", "unknown"),
                )
            )
        return samples

    def save_jsonl(
        self,
        samples: list[TrainingSample],
        output_path: str | Path,
        *,
        format: str = "sft",
        include_thinking: bool = True,
    ) -> int:
        """Save formatted samples to JSONL file.

        Args:
            samples: List of TrainingSample objects.
            output_path: Output JSONL file path.
            format: "sft" | "grpo" | "dpo"
            include_thinking: If True, use thinking format for SFT (explicit reasoning).
                              If False, use plain format (model generates reasoning itself).
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                if format == "sft":
                    if include_thinking:
                        record = sample.to_sft_thinking_dict()
                    else:
                        record = sample.to_sft_dict()
                elif format == "grpo":
                    record = sample.to_grpo_dict()
                elif format == "dpo":
                    continue  # DPO requires chosen/rejected, handle separately
                else:
                    logger.warning("Unknown format: %s", format)
                    continue
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

        logger.info("Saved %d samples to %s (format=%s)", count, output_path, format)
        return count

    @staticmethod
    def _basic_schema(db) -> str:
        """Generate a simple schema string as fallback."""
        lines = [f"Database: {db.db_id}", ""]
        for table in db.tables:
            cols = ", ".join(
                f"{c.name} ({c.dtype})" + (" [PK]" if c.primary_key else "")
                for c in table.columns
            )
            lines.append(f"CREATE TABLE {table.name} ({cols})")
        if db.foreign_keys:
            lines.append("")
            lines.append("Foreign Keys:")
            for fk in db.foreign_keys:
                lines.append(f"  {fk.from_table}.{fk.from_column} = {fk.to_table}.{fk.to_column}")
        return "\n".join(lines)

    @staticmethod
    def _generate_reasoning(ex, db) -> str:
        """Generate a CoT reasoning trace from the gold SQL.

        This trace goes inside the <think> block during SFT.
        """
        if db is None:
            return ""
        sql_upper = ex.query.upper()
        mentioned_tables = [t.name for t in db.tables if t.name.upper() in sql_upper]
        steps = []
        if mentioned_tables:
            steps.append(f"1. Identify relevant tables: {', '.join(mentioned_tables)}")
        if "JOIN" in sql_upper:
            steps.append("2. Tables need to be joined using foreign keys")
        if "WHERE" in sql_upper:
            steps.append("3. Apply filtering conditions")
        if "GROUP BY" in sql_upper:
            steps.append("4. Results need aggregation with GROUP BY")
        if "ORDER BY" in sql_upper:
            steps.append("5. Results should be ordered")
        if "HAVING" in sql_upper:
            steps.append("6. Apply post-aggregation filter with HAVING")
        return "\n".join(steps)


# =============================================================================
# OmniSQL Multi-Dataset Formatter
# =============================================================================

class OmniSQLFormatter:
    """Convert OmniSQL processed data ({input_seq, output_seq}) to TrainingSample.

    Handles the multi-dataset format from OmniSQL's process_dataset.py:
      - train_synsql.json  (~2.5M samples)
      - train_spider.json  (~7K samples)
      - train_bird.json    (~9.4K samples)

    The output_seq contains CoT reasoning (markdown steps) followed by SQL
    in a ```sql``` code block.  This formatter splits them apart for
    Qwen3 think/no-think mode training.
    """

    # Regex to extract SQL from markdown ```sql ... ``` or ``` ... ``` blocks
    _SQL_BLOCK_RE = re.compile(
        r"```(?:sql)?\s*\n(.*?)```",
        re.DOTALL | re.IGNORECASE,
    )

    def load_and_merge(
        self,
        data_paths: list[str | Path],
        max_samples_per_file: dict[str, int] | None = None,
    ) -> list[TrainingSample]:
        """Load multiple OmniSQL processed JSON files and convert to TrainingSamples.

        Parameters
        ----------
        data_paths : list[str | Path]
            Paths to processed JSON files (e.g. train_synsql.json, train_spider.json).
        max_samples_per_file : dict[str, int], optional
            Cap per filename, e.g. ``{"train_synsql.json": 200_000}``.
            Files not listed have no cap.

        Returns
        -------
        list[TrainingSample]
            Merged and converted training samples from all files.
        """
        max_samples_per_file = max_samples_per_file or {}
        all_samples: list[TrainingSample] = []

        for fpath in data_paths:
            fpath = Path(fpath)
            fname = fpath.name
            cap = max_samples_per_file.get(fname)

            logger.info("Loading OmniSQL data from: %s (cap=%s)", fpath, cap)

            samples = self._load_single_file(fpath, max_samples=cap)
            all_samples.extend(samples)
            logger.info("  → loaded %d samples (total so far: %d)", len(samples), len(all_samples))

        logger.info("Total OmniSQL samples loaded: %d", len(all_samples))
        return all_samples

    def _load_single_file(
        self,
        path: Path,
        max_samples: int | None = None,
    ) -> list[TrainingSample]:
        """Load a single OmniSQL processed JSON file.

        Uses streaming JSON parsing (ijson) for large files like
        train_synsql.json (~23GB).
        """
        samples: list[TrainingSample] = []

        try:
            import ijson  # streaming JSON parser for large files

            with open(path, "r", encoding="utf-8") as f:
                for obj in ijson.items(f, "item"):
                    input_seq = obj.get("input_seq", "")
                    output_seq = obj.get("output_seq", "")

                    schema_context, question = self._parse_input_seq(input_seq)
                    reasoning, sql = self._parse_output_seq(output_seq)

                    samples.append(
                        TrainingSample(
                            db_id="omnisql",
                            question=question,
                            schema_context=schema_context,
                            gold_sql=sql,
                            reasoning=reasoning,
                            difficulty="unknown",
                        )
                    )

                    if max_samples and len(samples) >= max_samples:
                        break

        except ImportError:
            # Fallback: standard json (will use lots of RAM for large files)
            logger.warning("ijson not installed, falling back to json.load() — may be slow for large files")
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if max_samples:
                data = data[:max_samples]

            for obj in data:
                input_seq = obj.get("input_seq", "")
                output_seq = obj.get("output_seq", "")

                schema_context, question = self._parse_input_seq(input_seq)
                reasoning, sql = self._parse_output_seq(output_seq)

                samples.append(
                    TrainingSample(
                        db_id="omnisql",
                        question=question,
                        schema_context=schema_context,
                        gold_sql=sql,
                        reasoning=reasoning,
                        difficulty="unknown",
                    )
                )

        return samples

    @staticmethod
    def _parse_input_seq(input_seq: str) -> tuple[str, str]:
        """Extract (schema_context, question) from OmniSQL input_seq.

        OmniSQL input_seq follows this template::

            Task Overview: ...
            Database Engine: SQLite
            Database Schema:
            {schema DDLs}
            This schema describes ...
            Question:
            {question}
            Instructions: ...

        Returns
        -------
        tuple[str, str]
            (schema_context, question)
        """
        schema_context = ""
        question = ""

        # Extract schema: between "Database Schema:\n" and "This schema describes"
        schema_start = input_seq.find("Database Schema:\n")
        schema_end = input_seq.find("This schema describes")

        if schema_start != -1 and schema_end != -1:
            schema_context = input_seq[schema_start + len("Database Schema:\n"):schema_end].strip()
        elif schema_start != -1:
            # Fallback: take until "Question:"
            q_marker = input_seq.find("Question:\n", schema_start)
            if q_marker != -1:
                schema_context = input_seq[schema_start + len("Database Schema:\n"):q_marker].strip()

        # Extract question: between "Question:\n" and "Instructions:\n"
        q_start = input_seq.find("Question:\n")
        q_end = input_seq.find("\nInstructions:")

        if q_start != -1:
            q_content_start = q_start + len("Question:\n")
            if q_end != -1 and q_end > q_content_start:
                question = input_seq[q_content_start:q_end].strip()
            else:
                # Fallback: take rest until "Output Format" or end
                of_marker = input_seq.find("\nOutput Format:", q_content_start)
                if of_marker != -1:
                    question = input_seq[q_content_start:of_marker].strip()
                else:
                    question = input_seq[q_content_start:].strip()

        return schema_context, question

    @classmethod
    def _parse_output_seq(cls, output_seq: str) -> tuple[str, str]:
        """Extract (reasoning, sql) from OmniSQL output_seq.

        The output_seq contains CoT reasoning followed by SQL in a
        ```sql ... ``` markdown code block.

        Returns
        -------
        tuple[str, str]
            (reasoning, sql) — reasoning is everything before the LAST
            sql code block; sql is the content of the LAST code block.
        """
        # Find all SQL code blocks, take the last one (the final answer)
        matches = list(cls._SQL_BLOCK_RE.finditer(output_seq))

        if not matches:
            # No code block found — try to extract SELECT statement
            select_match = re.search(r"(SELECT\s+.+)", output_seq, re.IGNORECASE | re.DOTALL)
            if select_match:
                sql = select_match.group(1).strip()
                reasoning = output_seq[:select_match.start()].strip()
                return reasoning, sql
            # Nothing found — entire output is the response
            return "", output_seq.strip()

        last_match = matches[-1]
        sql = last_match.group(1).strip()

        # Reasoning = everything before the last code block
        reasoning = output_seq[:last_match.start()].strip()

        # Clean up trailing "---" or "### Final" etc. from reasoning
        reasoning = re.sub(r"\n---\s*$", "", reasoning).strip()

        return reasoning, sql
