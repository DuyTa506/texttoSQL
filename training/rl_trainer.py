"""
RL Trainer (Unsloth + Qwen3) – Stage 2 of the 2-stage SLM training pipeline.

Follows the Qwen3 GRPO notebook pattern:
  1. Pre-finetune: Short SFT to teach output format (optional)
  2. GRPO training: Reward-based optimization with vLLM sampling

Qwen3's built-in <think>...</think> is leveraged for reasoning.
Custom reward functions: format matching + SQL execution + schema check.

Supports:
  - GRPO (primary) — trl.GRPOTrainer + GRPOConfig + vLLM
  - DPO (ablation) — trl.DPOTrainer with Unsloth PatchDPOTrainer()

Docs:
  - https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune#grpo-with-qwen3
  - Qwen3 (4B) GRPO notebook pattern
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
from pathlib import Path

from .config import RLConfig

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# SQL Format Patterns for Qwen3 Thinking Mode
# ──────────────────────────────────────────────────────────────

# Expected output format:
#   <think>reasoning about tables, joins, conditions</think>
#   ```sql
#   SELECT ... FROM ... WHERE ...
#   ```

SQL_BLOCK_RE = re.compile(
    r"```sql\s*(.+?)\s*```",
    flags=re.MULTILINE | re.DOTALL,
)

THINK_BLOCK_RE = re.compile(
    r"<think>(.+?)</think>",
    flags=re.MULTILINE | re.DOTALL,
)

# Full format: thinking block + SQL block
FULL_FORMAT_RE = re.compile(
    r"</think>.*?```sql\s*(.+?)\s*```[\s]*(?:<\|endoftext\|>)?[\s]*$",
    flags=re.MULTILINE | re.DOTALL,
)


# ──────────────────────────────────────────────────────────────
# GRPO Reward Functions (following Qwen3 GRPO notebook pattern)
# Each function signature: (completions, **kwargs) -> list[float]
# ──────────────────────────────────────────────────────────────

def match_sql_format_exactly(completions, **kwargs):
    """Reward 3.0 if output follows exact format: <think>...</think> ```sql...```."""
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        if FULL_FORMAT_RE.search(response) is not None:
            scores.append(3.0)
        else:
            scores.append(0.0)
    return scores


def match_sql_format_approximately(completions, **kwargs):
    """Partial credit for each format element present.

    Also penalizes trivially short <think> blocks to prevent thinking collapse
    (HES-SQL finding: GRPO can degrade thinking capability without this guard).
    """
    scores = []
    for completion in completions:
        score = 0.0
        response = completion[0]["content"]

        # Check for thinking block
        if response.count("</think>") == 1:
            score += 0.5
        elif response.count("</think>") > 1:
            score -= 1.0  # penalize multiple blocks

        # Check for SQL code block
        if response.count("```sql") == 1:
            score += 0.5
        elif response.count("```sql") > 1:
            score -= 1.0

        # Check for closing code block
        closing_count = response.count("```") - response.count("```sql")
        if closing_count == 1:
            score += 0.5
        elif closing_count > 1:
            score -= 0.5

        # Penalize trivially short thinking (thinking collapse prevention)
        think_match = THINK_BLOCK_RE.search(response)
        if think_match:
            think_len = len(think_match.group(1).strip().split())
            if think_len < 5:
                score -= 0.5  # near-empty thinking block

        scores.append(score)
    return scores


def check_sql_execution(completions, answer, **kwargs):
    """Check if extracted SQL matches gold answer.

    Uses string matching (execution checking requires DB connection).
    5.0 for exact match, 3.5 for match after stripping whitespace.
    """
    scores = []
    for completion, gold in zip(completions, answer):
        response = completion[0]["content"]
        match = FULL_FORMAT_RE.search(response)

        if match is None:
            # Try fallback: just SQL block without thinking
            match = SQL_BLOCK_RE.search(response)

        if match is None:
            scores.append(-2.0)
            continue

        predicted = match.group(1).strip()
        gold_stripped = gold.strip()

        if predicted == gold_stripped:
            scores.append(5.0)
        elif predicted.lower() == gold_stripped.lower():
            scores.append(3.5)
        else:
            # Partial credit: check if key tables/columns appear
            gold_tokens = set(gold_stripped.lower().split())
            pred_tokens = set(predicted.lower().split())
            overlap = len(gold_tokens & pred_tokens) / max(len(gold_tokens), 1)
            if overlap > 0.7:
                scores.append(1.5)
            elif overlap > 0.4:
                scores.append(0.5)
            else:
                scores.append(-1.5)

    return scores


def check_sql_execution_real(completions, answer, db_path=None, **kwargs):
    """Check SQL execution accuracy against real SQLite database.

    Replaces string-matching with actual DB execution when db_path is available.
    Falls back to check_sql_execution() if no db_path provided.

    Scores: 5.0 exact result match, overlap*3.0 partial, -2.0 no SQL found.
    """
    if db_path is None:
        return check_sql_execution(completions, answer, **kwargs)

    scores = []
    for completion, gold, dbp in zip(completions, answer, db_path):
        response = completion[0]["content"]

        match = FULL_FORMAT_RE.search(response)
        if match is None:
            match = SQL_BLOCK_RE.search(response)
        if match is None:
            scores.append(-2.0)
            continue

        predicted = match.group(1).strip()
        gold_stripped = gold.strip()

        if not dbp or not Path(dbp).exists():
            # DB not accessible, fall back to string comparison
            if predicted == gold_stripped:
                scores.append(5.0)
            elif predicted.lower() == gold_stripped.lower():
                scores.append(3.5)
            else:
                gold_tokens = set(gold_stripped.lower().split())
                pred_tokens = set(predicted.lower().split())
                overlap = len(gold_tokens & pred_tokens) / max(len(gold_tokens), 1)
                scores.append(overlap * 3.0 if overlap > 0.3 else -1.0)
            continue

        try:
            conn = sqlite3.connect(f"file:{dbp}?mode=ro", uri=True, timeout=5)
            conn.execute("PRAGMA query_only = true")
            cursor = conn.cursor()
            cursor.execute(gold_stripped)
            gold_results = set(map(tuple, cursor.fetchall()))
            cursor.execute(predicted)
            pred_results = set(map(tuple, cursor.fetchall()))
            conn.close()

            if gold_results == pred_results:
                scores.append(5.0)
            elif pred_results and gold_results:
                overlap = len(gold_results & pred_results) / max(len(gold_results), 1)
                scores.append(overlap * 3.0)
            else:
                scores.append(0.0)
        except Exception:
            scores.append(-1.0)

    return scores


def check_schema_faithfulness(completions, schema_tables=None, **kwargs):
    """Check if SQL references only tables/columns from the given schema.

    2.0 for using only valid schema elements, -1.0 for hallucinating.
    """
    if schema_tables is None:
        return [0.0] * len(completions)

    scores = []
    for completion, tables in zip(completions, schema_tables):
        response = completion[0]["content"]
        match = SQL_BLOCK_RE.search(response)

        if match is None:
            scores.append(0.0)
            continue

        sql = match.group(1).lower()
        valid_tables = set(t.lower() for t in tables.split(",")) if isinstance(tables, str) else set()

        # Simple check: are referenced tables in the schema?
        from_matches = re.findall(r'\bfrom\s+(\w+)', sql)
        join_matches = re.findall(r'\bjoin\s+(\w+)', sql)
        referenced = set(from_matches + join_matches)

        if not referenced:
            scores.append(0.0)
        elif referenced.issubset(valid_tables):
            scores.append(2.0)
        else:
            invalid_count = len(referenced - valid_tables)
            scores.append(-0.5 * invalid_count)

    return scores


def check_column_set_matching(completions, answer, **kwargs):
    """Dense structural reward: partial credit for correct tables/columns even if full SQL wrong.

    Inspired by SQL-ASTRA's Column-Set Matching Reward (CSMR).
    Converts binary EX feedback into [0, 2.0] signal based on structural overlap.
    Critical for small models that need dense gradients (Think2SQL finding).

    Scores: [0, 1.0] for table overlap + [0, 1.0] for column/identifier overlap = max 2.0.
    """
    scores = []
    for completion, gold in zip(completions, answer):
        response = completion[0]["content"]

        match = FULL_FORMAT_RE.search(response)
        if match is None:
            match = SQL_BLOCK_RE.search(response)
        if match is None:
            scores.append(0.0)
            continue

        predicted = match.group(1).strip().lower()
        gold_lower = gold.strip().lower()

        # Table overlap: extract tables after FROM and JOIN
        pred_tables = set(re.findall(r'\b(?:from|join)\s+(\w+)', predicted))
        gold_tables = set(re.findall(r'\b(?:from|join)\s+(\w+)', gold_lower))

        if gold_tables:
            table_jaccard = len(pred_tables & gold_tables) / max(len(pred_tables | gold_tables), 1)
        else:
            table_jaccard = 1.0  # no tables expected, neutral

        # Identifier overlap: all word tokens excluding SQL keywords
        _SQL_KEYWORDS = frozenset({
            "select", "from", "where", "and", "or", "join", "on", "left", "right",
            "inner", "outer", "group", "by", "order", "asc", "desc", "having",
            "limit", "distinct", "as", "in", "not", "null", "is", "like",
            "between", "case", "when", "then", "else", "end", "count", "sum",
            "avg", "max", "min", "cast", "union", "all", "exists", "with",
            "insert", "update", "delete", "into", "values", "set", "create",
            "table", "drop", "alter", "true", "false", "offset", "except",
            "intersect", "cross", "full", "natural",
        })
        pred_ids = set(re.findall(r'\b[a-z_]\w*\b', predicted)) - _SQL_KEYWORDS
        gold_ids = set(re.findall(r'\b[a-z_]\w*\b', gold_lower)) - _SQL_KEYWORDS

        if gold_ids:
            id_jaccard = len(pred_ids & gold_ids) / max(len(pred_ids | gold_ids), 1)
        else:
            id_jaccard = 1.0

        scores.append(table_jaccard * 1.0 + id_jaccard * 1.0)

    return scores


def check_error_correction_success(completions, answer, db_path=None, task_type=None, schema_tables=None, **kwargs):
    """Bonus reward specifically for error_correction task type.

    Returns 0.0 for standard text2sql tasks (neutral, no interference).
    For error_correction tasks, rewards:
      - Execution match against gold (+4.0)
      - SQL executes without error (+1.0)
      - Dense semantic reasoning quality:
          - Diagnosis signal: mentions error/wrong/incorrect/missing (+0.5)
          - Fix signal: mentions fix/correct/change/replace/use (+0.5)
          - Schema grounding: mentions actual table names from schema (+0.5)
          - Length bonus (+0.3 for >100 words, +0.1 for >30 words)
    Max: 1.0 + 4.0 + 0.5 + 0.5 + 0.5 + 0.3 = 6.8
    """
    if task_type is None:
        return [0.0] * len(completions)

    scores = []
    for i, (completion, gold, tt) in enumerate(zip(completions, answer, task_type)):
        if tt != "error_correction":
            scores.append(0.0)
            continue

        response = completion[0]["content"]

        match = FULL_FORMAT_RE.search(response)
        if match is None:
            match = SQL_BLOCK_RE.search(response)
        if match is None:
            scores.append(-2.0)
            continue

        predicted = match.group(1).strip()
        gold_stripped = gold.strip()
        score = 0.0

        # ── Execution correctness ────────────────────────────────────────
        dbp = db_path[i] if isinstance(db_path, (list, tuple)) else db_path
        if dbp and Path(dbp).exists():
            try:
                conn = sqlite3.connect(f"file:{dbp}?mode=ro", uri=True, timeout=5)
                conn.execute("PRAGMA query_only = true")
                cursor = conn.cursor()

                # Does it execute at all?
                cursor.execute(predicted)
                cursor.fetchall()
                score += 1.0  # valid SQL that executes

                # Does it match gold?
                cursor.execute(gold_stripped)
                gold_results = set(map(tuple, cursor.fetchall()))
                cursor.execute(predicted)
                pred_results = set(map(tuple, cursor.fetchall()))
                conn.close()

                if gold_results == pred_results:
                    score += 4.0
            except Exception:
                # SQL failed to execute — no execution bonus
                pass
        else:
            # Fallback: string comparison
            if predicted.lower() == gold_stripped.lower():
                score += 5.0

        # ── Semantic reasoning quality ───────────────────────────────────
        think_match = THINK_BLOCK_RE.search(response)
        if think_match:
            think_text = think_match.group(1).strip().lower()
            reasoning_len = len(think_text.split())

            # Diagnosis signal — does the reasoning identify the error?
            has_diagnosis = any(kw in think_text for kw in [
                "error", "wrong", "incorrect", "missing", "instead of",
                "should be", "not exist", "no such", "ambiguous",
            ])
            # Fix signal — does the reasoning propose a correction?
            has_fix = any(kw in think_text for kw in [
                "fix", "correct", "change", "replace", "use",
                "add", "remove", "join", "where",
            ])
            # Schema grounding — does reasoning reference actual tables?
            tables_str = ""
            if schema_tables is not None:
                tables_str = schema_tables[i] if isinstance(schema_tables, (list, tuple)) else schema_tables
            table_names = [t.strip().lower() for t in tables_str.split(",") if t.strip()]
            has_grounding = any(t in think_text for t in table_names) if table_names else False

            score += 0.5 if has_diagnosis else 0.0
            score += 0.5 if has_fix else 0.0
            score += 0.5 if has_grounding else 0.0
            # Length bonus (tiebreaker)
            score += 0.3 if reasoning_len > 100 else (0.1 if reasoning_len > 30 else 0.0)

        scores.append(score)

    return scores


# ──────────────────────────────────────────────────────────────
# GRPO Trainer
# ──────────────────────────────────────────────────────────────

class GRPOTrainerUnsloth:
    """Stage 2: GRPO training with Qwen3 thinking mode + vLLM sampling.

    Follows 2-phase approach from Qwen3 GRPO notebook:
      Phase A: Pre-finetune (optional SFT) - teach output format
      Phase B: GRPO reward optimization - improve SQL quality

    Reward functions are separate callables (not a single monolithic function).
    """

    def __init__(self, config: RLConfig):
        self.config = config
        self.model = None
        self.tokenizer = None

    def setup(self):
        """Load SFT checkpoint via Unsloth and attach LoRA for RL stage."""
        import torch
        from unsloth import FastLanguageModel

        # Ensure each torchrun worker sticks to its local GPU before loading weights.
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
            logger.info("Using CUDA device local_rank=%d", local_rank)

        logger.info("Loading SFT model via Unsloth: %s", self.config.sft_model_path)

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.sft_model_path,
            max_seq_length=self.config.max_seq_length,
            dtype=None,
            load_in_4bit=self.config.load_in_4bit,
            device_map=None,  # required for torchrun/DDP
        )

        # Attach LoRA for RL fine-tuning
        lora = self.config.lora
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=lora.r,
            target_modules=lora.target_modules,
            lora_alpha=lora.lora_alpha,
            lora_dropout=lora.lora_dropout,
            bias=lora.bias,
            use_gradient_checkpointing=lora.use_gradient_checkpointing,
            random_state=lora.random_state,
            use_rslora=lora.use_rslora,
            max_seq_length=self.config.max_seq_length,
        )

    def train(self):
        """Run GRPO training with vLLM sampling (Qwen3 pattern)."""
        from datasets import load_dataset
        from trl import GRPOConfig, GRPOTrainer
        from unsloth import is_bfloat16_supported

        if self.model is None:
            self.setup()

        dataset = load_dataset("json", data_files={"train": self.config.train_data_path})

        # vLLM sampling params for generation (following Qwen3 notebook)
        from vllm import SamplingParams
        vllm_sampling_params = SamplingParams(
            min_p=self.config.vllm_min_p,
            top_p=self.config.vllm_top_p,
            top_k=self.config.vllm_top_k,
            seed=self.config.vllm_seed,
            stop=[self.tokenizer.eos_token],
            include_stop_str_in_output=True,
        )

        # GRPOConfig (replaces TrainingArguments for GRPO)
        grpo_config = GRPOConfig(
            vllm_sampling_params=vllm_sampling_params,
            temperature=self.config.grpo_temperature,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type="linear",
            optim=self.config.optim,
            logging_steps=self.config.logging_steps,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            num_generations=self.config.grpo_num_generations,
            max_prompt_length=self.config.max_prompt_length,
            max_completion_length=self.config.max_completion_length,
            max_steps=100,  # adjust for full training
            save_steps=100,
            output_dir=self.config.output_dir,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            report_to="none",
        )

        # Multi-signal reward functions.
        # trl.GRPOTrainer auto-forwards extra dataset columns (db_path, task_type,
        # schema_tables) as **kwargs to each reward function.
        reward_funcs = [
            match_sql_format_exactly,        # max 3.0  — format correctness
            match_sql_format_approximately,  # max 1.5  — partial format + collapse guard
            check_sql_execution_real,        # max 5.0  — real DB execution accuracy
            check_column_set_matching,       # max 2.0  — dense structural signal (SQL-ASTRA)
            check_schema_faithfulness,       # max 2.0  — schema compliance
            check_error_correction_success,  # max 6.8  — error correction bonus
        ]

        trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            reward_funcs=reward_funcs,
            args=grpo_config,
            train_dataset=dataset["train"],
        )

        logger.info(
            "Starting GRPO (Qwen3, N=%d, vLLM)...",
            self.config.grpo_num_generations,
        )
        trainer.train()

        output_path = Path(self.config.output_dir) / "final"
        self.model.save_pretrained(str(output_path))
        self.tokenizer.save_pretrained(str(output_path))
        logger.info("GRPO model saved to %s", output_path)
        return trainer

    @staticmethod
    def extract_sql(text: str) -> str:
        """Extract SQL from Qwen3 thinking output.

        Tries full format first (<think>...</think> ```sql...```)
        then falls back to just ```sql...```.
        """
        match = FULL_FORMAT_RE.search(text)
        if match:
            return match.group(1).strip()
        match = SQL_BLOCK_RE.search(text)
        if match:
            return match.group(1).strip()
        # Last resort: look for SQL: prefix
        for line in reversed(text.strip().split("\n")):
            if line.strip().upper().startswith("SQL:"):
                return line.strip()[4:].strip()
        return ""


# ──────────────────────────────────────────────────────────────
# DPO Trainer (ablation alternative)
# ──────────────────────────────────────────────────────────────

class DPOTrainerUnsloth:
    """Stage 2 alternative: DPO with Unsloth acceleration.

    Follows Unsloth DPO pattern:
      PatchDPOTrainer()
      model = FastLanguageModel.from_pretrained(...)
      model = FastLanguageModel.get_peft_model(...)
      trainer = DPOTrainer(model=model, ref_model=None, ...)
    """

    def __init__(self, config: RLConfig):
        self.config = config
        self.model = None
        self.tokenizer = None

    def setup(self):
        """Load SFT checkpoint with Unsloth."""
        import torch
        from unsloth import FastLanguageModel, PatchDPOTrainer

        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
            logger.info("Using CUDA device local_rank=%d", local_rank)

        # Patch DPOTrainer for Unsloth compatibility
        PatchDPOTrainer()

        logger.info("Loading SFT model for DPO: %s", self.config.sft_model_path)

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.sft_model_path,
            max_seq_length=self.config.max_seq_length,
            dtype=None,
            load_in_4bit=self.config.load_in_4bit,
            device_map=None,  # required for torchrun/DDP
        )

        lora = self.config.lora
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=lora.r,
            target_modules=lora.target_modules,
            lora_alpha=lora.lora_alpha,
            lora_dropout=lora.lora_dropout,
            bias=lora.bias,
            use_gradient_checkpointing=lora.use_gradient_checkpointing,
            random_state=lora.random_state,
            max_seq_length=self.config.max_seq_length,
        )

    def train(self):
        """Run DPO training with Unsloth."""
        from datasets import load_dataset
        from transformers import TrainingArguments
        from trl import DPOTrainer
        from unsloth import is_bfloat16_supported

        if self.model is None:
            self.setup()

        # DPO dataset expects: prompt, chosen, rejected
        dataset = load_dataset("json", data_files={"train": self.config.train_data_path})

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            warmup_ratio=0.1,
            num_train_epochs=self.config.num_epochs,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=self.config.logging_steps,
            save_strategy=self.config.save_strategy,
            optim=self.config.optim,
            seed=self.config.lora.random_state,
            report_to="none",
        )

        # ref_model=None → Unsloth handles reference model internally
        dpo_trainer = DPOTrainer(
            model=self.model,
            ref_model=None,
            args=training_args,
            beta=self.config.dpo_beta,
            train_dataset=dataset["train"],
            tokenizer=self.tokenizer,
            max_length=self.config.dpo_max_length,
            max_prompt_length=self.config.dpo_max_prompt_length,
        )

        logger.info("Starting DPO training (beta=%.2f, Unsloth)...", self.config.dpo_beta)
        dpo_trainer.train()

        output_path = Path(self.config.output_dir) / "final"
        self.model.save_pretrained(str(output_path))
        self.tokenizer.save_pretrained(str(output_path))
        logger.info("DPO model saved to %s", output_path)
        return dpo_trainer
