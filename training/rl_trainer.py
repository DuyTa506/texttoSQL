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
import re
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
    """Partial credit for each format element present."""
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
        from unsloth import FastLanguageModel

        logger.info("Loading SFT model via Unsloth: %s", self.config.sft_model_path)

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.sft_model_path,
            max_seq_length=self.config.max_seq_length,
            dtype=None,
            load_in_4bit=self.config.load_in_4bit,
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

        # Multi-signal reward functions (each called independently by GRPO)
        reward_funcs = [
            match_sql_format_exactly,
            match_sql_format_approximately,
            check_sql_execution,
            check_schema_faithfulness,
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
        from unsloth import FastLanguageModel, PatchDPOTrainer

        # Patch DPOTrainer for Unsloth compatibility
        PatchDPOTrainer()

        logger.info("Loading SFT model for DPO: %s", self.config.sft_model_path)

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.sft_model_path,
            max_seq_length=self.config.max_seq_length,
            dtype=None,
            load_in_4bit=self.config.load_in_4bit,
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
