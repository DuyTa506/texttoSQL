"""
Benchmark – end-to-end evaluation pipeline.

  Question → RAG (schema linking) → SLM (SQL generation) → Execute → Compare

Classes
-------
BenchmarkReporter
    Handles result display (Rich console tables) and persistence (JSON files).
    Fully independent of the retrieval / generation stack — swap or subclass
    without touching the runner logic.

Benchmark
    Orchestrates the evaluation loop.  Delegates all display and saving to an
    injected ``BenchmarkReporter`` instance.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from rich.console import Console
from rich.table import Table as RichTable
from tqdm import tqdm

from src.schema import Database, Example
from ..evaluation.metrics import (
    compute_metrics,
    exact_match,
    execution_accuracy,
    schema_precision,
    schema_recall,
)
from ..generation.inference import SQLInference
from ..retrieval.utils.schema_filter import SchemaFilter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reporter — display + persistence (no dependency on retrieval/generation)
# ---------------------------------------------------------------------------

class BenchmarkReporter:
    """Renders evaluation results to the console and saves them to disk.

    Completely decoupled from the retrieval and generation stack, so it can
    be subclassed or swapped independently of the runner logic.

    Parameters
    ----------
    console : Console, optional
        Rich ``Console`` instance.  A default one is created if not provided.
    """

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def display(self, metrics: dict, results: list[dict]) -> None:
        """Print a Rich summary table + optional per-difficulty breakdown."""
        self._display_aggregate(metrics)
        self._display_per_difficulty(results)

    def save(self, output_dir: str, metrics: dict, results: list[dict]) -> None:
        """Write ``metrics.json`` and ``details.jsonl`` to *output_dir*."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        with open(out / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        with open(out / "details.jsonl", "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

        logger.info("Results saved to %s", out)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _display_aggregate(self, metrics: dict) -> None:
        table = RichTable(title="Text-to-SQL Evaluation Results")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        for key, value in metrics.items():
            if key == "total_examples":
                table.add_row(key, str(int(value)))
            else:
                table.add_row(key, f"{value:.4f}")
        self.console.print(table)

    def _display_per_difficulty(self, results: list[dict]) -> None:
        difficulties = set(r.get("difficulty", "unknown") for r in results)
        if len(difficulties) <= 1:
            return
        diff_table = RichTable(title="Per-Difficulty Breakdown")
        diff_table.add_column("Difficulty")
        diff_table.add_column("Count", justify="right")
        diff_table.add_column("EX", justify="right")
        diff_table.add_column("EM", justify="right")
        for diff in sorted(difficulties):
            subset = [r for r in results if r.get("difficulty") == diff]
            n = len(subset) or 1
            diff_table.add_row(
                diff,
                str(len(subset)),
                f"{sum(r['ex'] for r in subset) / n:.4f}",
                f"{sum(r['em'] for r in subset) / n:.4f}",
            )
        self.console.print(diff_table)


# ---------------------------------------------------------------------------
# Benchmark runner — orchestration only
# ---------------------------------------------------------------------------

class Benchmark:
    """End-to-end evaluation pipeline for the text-to-SQL system.

    Parameters
    ----------
    inference_engine : SQLInference
    schema_filter : SchemaFilter
    databases : dict[str, Database]
    db_dir : str
    reporter : BenchmarkReporter, optional
        Handles display and persistence.  A default instance is created
        if not provided, preserving the original zero-arg behavior.
    """

    def __init__(
        self,
        inference_engine: SQLInference,
        schema_filter: SchemaFilter,
        databases: dict[str, Database],
        db_dir: str,
        reporter: BenchmarkReporter | None = None,
    ):
        self.inference = inference_engine
        self.schema_filter = schema_filter
        self.databases = databases
        self.db_dir = db_dir
        self.reporter = reporter or BenchmarkReporter()

    def run(
        self,
        examples: list[Example],
        retrieved_schemas: dict[str, list[dict]],
        *,
        output_dir: str | None = None,
        subset: int | None = None,
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

            key = f"{ex.db_id}__{hash(ex.question) % 10**8}"
            retrieved = retrieved_schemas.get(key, [])
            schema_ctx = self.schema_filter.filter_and_format(retrieved, db)

            gen_result = self.inference.generate(ex.question, schema_ctx)
            predicted_sql = gen_result["sql"]

            db_path = db.db_path or str(
                Path(self.db_dir) / ex.db_id / f"{ex.db_id}.sqlite"
            )

            retrieved_tables = self.schema_filter.get_retrieved_tables(retrieved)
            results.append({
                "db_id":         ex.db_id,
                "question":      ex.question,
                "gold_sql":      ex.query,
                "predicted_sql": predicted_sql,
                "reasoning":     gen_result.get("reasoning", ""),
                "difficulty":    ex.difficulty,
                "ex":            execution_accuracy(predicted_sql, ex.query, db_path),
                "em":            exact_match(predicted_sql, ex.query),
                "recall":        schema_recall(retrieved_tables, ex.query),
                "precision":     schema_precision(retrieved_tables, ex.query),
            })

        metrics = compute_metrics(results)
        self.reporter.display(metrics, results)

        if output_dir:
            self.reporter.save(output_dir, metrics, results)

        return metrics
