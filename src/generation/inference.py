"""
Inference – generates SQL from a question using the trained SLM.

Loads the final model (SFT + RL merged), takes a filtered schema from the RAG
pipeline, and generates CoT reasoning + SQL query.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


class SQLInference:
    """Inference engine for the fine-tuned text-to-SQL model."""

    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._model = None
        self._tokenizer = None

    def _load(self):
        """Lazy-load model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        logger.info("Loading model from: %s", self.model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self._model.eval()

    @property
    def model(self):
        if self._model is None:
            self._load()
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._load()
        return self._tokenizer

    def generate(self, question: str, schema_context: str) -> dict:
        """Generate SQL from question + schema.

        Returns
        -------
        dict
            Keys: ``sql``, ``reasoning``, ``raw_output``.
        """
        prompt = self._build_prompt(question, schema_context)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "temperature": max(self.temperature, 1e-7),
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }

        with __import__("torch").no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        # Decode only new tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        raw_output = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        sql = self._extract_sql(raw_output)
        reasoning = self._extract_reasoning(raw_output)

        return {
            "sql": sql,
            "reasoning": reasoning,
            "raw_output": raw_output,
        }

    def generate_batch(
        self,
        questions: list[str],
        schema_contexts: list[str],
    ) -> list[dict]:
        """Generate SQL for a batch of questions."""
        return [
            self.generate(q, sc)
            for q, sc in zip(questions, schema_contexts)
        ]

    # ---- prompt formatting ---------------------------------------------------

    @staticmethod
    def _build_prompt(question: str, schema_context: str) -> str:
        return (
            f"<|im_start|>user\n"
            f"Question: {question}\n"
            f"Schema:\n{schema_context}\n\n"
            f"Think step-by-step, then generate SQL:<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    @staticmethod
    def _extract_sql(text: str) -> str:
        """Extract SQL from model output."""
        # Try explicit "SQL:" marker
        for line in reversed(text.strip().split("\n")):
            stripped = line.strip()
            if stripped.upper().startswith("SQL:"):
                return stripped[4:].strip()

        # Try to find SELECT statement
        match = re.search(r'(SELECT\s+.+)', text, re.IGNORECASE | re.DOTALL)
        if match:
            sql = match.group(1).strip()
            # Remove trailing explanation
            sql = re.split(r'\n\n|\n(?=[A-Z])', sql)[0]
            return sql.strip()

        # Fallback: last non-empty line
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        return lines[-1] if lines else ""

    @staticmethod
    def _extract_reasoning(text: str) -> str:
        """Extract reasoning part (everything before 'SQL:' marker)."""
        parts = re.split(r'(?i)\bSQL\s*:', text, maxsplit=1)
        if len(parts) > 1:
            return parts[0].strip()
        return ""
