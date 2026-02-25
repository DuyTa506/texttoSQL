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
