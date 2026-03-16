"""
Benchmark – end-to-end evaluation pipeline.

  Question → RAG (schema linking) → SLM (SQL generation) → Execute → Compare
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table as RichTable
from tqdm import tqdm

from ..schema.models import Database, Example
from ..evaluation.metrics import (
    compute_metrics,
    exact_match,
    execution_accuracy,
    schema_precision,
    schema_recall,
)
from ..generation.inference import SQLInference
from ..retrieval.schema_filter import SchemaFilter

logger = logging.getLogger(__name__)
console = Console()


class Benchmark:
    """End-to-end evaluation pipeline for the text-to-SQL system."""

    def __init__(
        self,
        inference_engine: SQLInference,
        schema_filter: SchemaFilter,
        databases: dict[str, Database],
        db_dir: str,
    ):
        self.inference = inference_engine
        self.schema_filter = schema_filter
        self.databases = databases
        self.db_dir = db_dir

    def run(
        self,
        examples: list[Example],
        retrieved_schemas: dict[str, list[dict]],
        *,
        output_dir: Optional[str] = None,
        subset: Optional[int] = None,
    ) -> dict:
        """Run evaluation on a set of examples.

        Parameters
        ----------
        examples : list[Example]
            Test examples with gold SQL.
        retrieved_schemas : dict[str, list[dict]]
            Pre-computed retrieval results per example key.
        output_dir : str, optional
            Save detailed results to this directory.
        subset : int, optional
            Evaluate only the first N examples.

        Returns
        -------
        dict
            Aggregate metrics.
        """
        if subset:
            examples = examples[:subset]

        results: list[dict] = []

        for ex in tqdm(examples, desc="Evaluating"):
            db = self.databases.get(ex.db_id)
            if db is None:
                logger.warning("DB not found: %s", ex.db_id)
                continue

            # Get schema context
            key = f"{ex.db_id}__{hash(ex.question) % 10**8}"
            retrieved = retrieved_schemas.get(key, [])
            schema_ctx = self.schema_filter.filter_and_format(retrieved, db)

            # Generate SQL
            gen_result = self.inference.generate(ex.question, schema_ctx)
            predicted_sql = gen_result["sql"]

            # DB path
            db_path = db.db_path or str(
                Path(self.db_dir) / ex.db_id / f"{ex.db_id}.sqlite"
            )

            # Compute metrics
            retrieved_tables = self.schema_filter.get_retrieved_tables(retrieved)
            ex_result = execution_accuracy(predicted_sql, ex.query, db_path)
            em_result = exact_match(predicted_sql, ex.query)
            recall = schema_recall(retrieved_tables, ex.query)
            precision = schema_precision(retrieved_tables, ex.query)

            result = {
                "db_id": ex.db_id,
                "question": ex.question,
                "gold_sql": ex.query,
                "predicted_sql": predicted_sql,
                "reasoning": gen_result.get("reasoning", ""),
                "difficulty": ex.difficulty,
                "ex": ex_result,
                "em": em_result,
                "recall": recall,
                "precision": precision,
            }
            results.append(result)

        # Aggregate
        metrics = compute_metrics(results)

        # Display
        self._display_results(metrics, results)

        # Save
        if output_dir:
            self._save_results(output_dir, metrics, results)

        return metrics

    def _display_results(self, metrics: dict, results: list[dict]):
        """Display results in a rich table."""
        table = RichTable(title="Text-to-SQL Evaluation Results")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        for key, value in metrics.items():
            if key == "total_examples":
                table.add_row(key, str(int(value)))
            else:
                table.add_row(key, f"{value:.4f}")

        console.print(table)

        # Per-difficulty breakdown
        difficulties = set(r.get("difficulty", "unknown") for r in results)
        if len(difficulties) > 1:
            diff_table = RichTable(title="Per-Difficulty Breakdown")
            diff_table.add_column("Difficulty")
            diff_table.add_column("Count", justify="right")
            diff_table.add_column("EX", justify="right")
            diff_table.add_column("EM", justify="right")

            for diff in sorted(difficulties):
                subset = [r for r in results if r.get("difficulty") == diff]
                n = len(subset) or 1
                ex_acc = sum(r["ex"] for r in subset) / n
                em_acc = sum(r["em"] for r in subset) / n
                diff_table.add_row(diff, str(len(subset)), f"{ex_acc:.4f}", f"{em_acc:.4f}")

            console.print(diff_table)

    @staticmethod
    def _save_results(output_dir: str, metrics: dict, results: list[dict]):
        """Save results to JSON files."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        with open(out / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        with open(out / "details.jsonl", "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

        logger.info("Results saved to %s", out)
