"""
SFT Trainer (Unsloth + Qwen3) – Stage 1 of the 2-stage SLM training pipeline.

Qwen3-4B has built-in thinking mode (<think>...</think>).
SFT teaches domain knowledge (Text-to-SQL), NOT reasoning format.
Data mix: 75% thinking + 25% direct (preserves reasoning ability).

API Reference:
  - FastLanguageModel.from_pretrained()  → load base model
  - FastLanguageModel.get_peft_model()   → attach LoRA adapters
  - SFTTrainer (from trl)                → training loop
  - tokenizer.apply_chat_template(enable_thinking=True/False)

Docs: https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune
"""

from __future__ import annotations

import logging
from pathlib import Path

from .config import SFTConfig

logger = logging.getLogger(__name__)


class SFTTrainerUnsloth:
    """Unsloth-powered SFT trainer for Qwen3 text-to-SQL fine-tuning.

    Key Qwen3 differences from Qwen2.5:
      - Built-in <think>...</think> reasoning blocks
      - enable_thinking param in apply_chat_template
      - 75/25 thinking/direct data mix to preserve reasoning
    """

    def __init__(self, config: SFTConfig):
        self.config = config
        self.model = None
        self.tokenizer = None

    def setup(self):
        """Load model with Unsloth and attach LoRA adapters."""
        from unsloth import FastLanguageModel

        logger.info("Loading Qwen3 model via Unsloth: %s", self.config.base_model)
        logger.info("Thinking mode: %s", self.config.enable_thinking)

        # Step 1: Load base model
        # NOTE: No device_map — DDP handles multi-GPU via torchrun
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.base_model,
            max_seq_length=self.config.max_seq_length,
            dtype=None,  # auto-detect (float16 / bfloat16)
            load_in_4bit=self.config.load_in_4bit,
        )

        # Step 2: Attach LoRA adapters
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
            loftq_config=lora.loftq_config,
            max_seq_length=self.config.max_seq_length,
        )

        # Log param counts
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info("Trainable: %d / %d (%.2f%%)", trainable, total, trainable / total * 100)

    def _format_thinking_example(self, instruction: str, response: str) -> str:
        """Format as thinking example — model generates <think> block.

        Qwen3 will produce:
          <think>reasoning about tables and SQL logic</think>
          ```sql
          SELECT ...
          ```
        """
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=False,
            enable_thinking=True,
        )

    def _format_direct_example(self, instruction: str, response: str) -> str:
        """Format as direct answer — no thinking block.

        For simple queries where reasoning is unnecessary.
        Uses <think>\n\n</think> prefix per Qwen3 non-thinking convention.
        """
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )

    def train(self):
        """Run SFT training with Unsloth optimizations."""
        import random
        from datasets import load_dataset
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from unsloth import is_bfloat16_supported

        if self.model is None:
            self.setup()

        # --- Data Loading ---
        if self.config.data_source == "omnisql":
            dataset = self._load_omnisql_data()
        else:
            # Original custom/spider path
            data_files = {"train": self.config.train_data_path}
            if self.config.eval_data_path:
                data_files["validation"] = self.config.eval_data_path
            dataset = load_dataset("json", data_files=data_files)

        # Format function – 75/25 thinking/direct mix
        thinking_ratio = self.config.thinking_data_ratio

        def format_prompt(examples):
            """Format data with Qwen3 thinking mode.

            75% of examples use enable_thinking=True (model sees reasoning).
            25% use enable_thinking=False (direct SQL output).
            This preserves Qwen3's reasoning ability per Unsloth recommendation.
            """
            texts = []
            for instruction, response in zip(examples["instruction"], examples["response"]):
                use_thinking = random.random() < thinking_ratio
                if use_thinking and self.config.enable_thinking:
                    text = self._format_thinking_example(instruction, response)
                else:
                    text = self._format_direct_example(instruction, response)
                texts.append(text)
            return {"text": texts}

        dataset = dataset.map(format_prompt, batched=True)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_train_epochs=self.config.num_epochs,
            max_steps=self.config.max_steps if self.config.max_steps > 0 else -1,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            save_strategy=self.config.save_strategy,
            eval_strategy="epoch" if (self.config.eval_data_path or "validation" in dataset) else "no",
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            optim=self.config.optim,
            seed=self.config.lora.random_state,
            report_to="none",
        )

        # SFTTrainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("validation"),
            args=training_args,
            max_seq_length=self.config.max_seq_length,
            dataset_text_field="text",
        )

        logger.info("Starting SFT training (Qwen3 + Unsloth, thinking=%s)...",
                     self.config.enable_thinking)
        trainer.train()

        # Save LoRA adapter
        output_path = Path(self.config.output_dir) / "final"
        self.model.save_pretrained(str(output_path))
        self.tokenizer.save_pretrained(str(output_path))
        logger.info("LoRA adapter saved to %s", output_path)

        return trainer

    def _load_omnisql_data(self):
        """Load pre-prepared OmniSQL data from prepare_sft_data.py output.

        The data is already formatted with <think> tags and think/no-think
        mix, so we load it directly as {instruction, response} pairs.

        If omnisql_data_paths contains raw OmniSQL JSONs, falls back to
        on-the-fly conversion via OmniSQLFormatter.
        """
        from datasets import load_dataset, Dataset

        paths = self.config.omnisql_data_paths

        # Check if paths point to prepared JSONL files
        prepared_paths = [p for p in paths if p.endswith(".jsonl")]
        raw_paths = [p for p in paths if p.endswith(".json")]

        if prepared_paths:
            # Load pre-prepared JSONL (recommended path)
            logger.info("Loading pre-prepared JSONL data: %s", prepared_paths)
            data_files = {"train": prepared_paths[0]}
            if len(prepared_paths) > 1:
                data_files["validation"] = prepared_paths[1]
            dataset = load_dataset("json", data_files=data_files)
            logger.info("OmniSQL prepared dataset loaded: %d train examples",
                         len(dataset["train"]))
            return dataset

        elif raw_paths:
            # Fallback: on-the-fly conversion from raw OmniSQL JSONs
            logger.info("Loading raw OmniSQL JSONs (on-the-fly conversion): %s", raw_paths)
            from .data_formatter import OmniSQLFormatter

            formatter = OmniSQLFormatter()
            samples = formatter.load_and_merge(
                data_paths=raw_paths,
                max_samples_per_file=self.config.omnisql_max_samples,
            )

            logger.info("Converting %d OmniSQL samples to SFT format...", len(samples))
            records = [sample.to_sft_thinking_dict() for sample in samples]

            dataset = Dataset.from_list(records)
            logger.info("OmniSQL dataset ready: %d examples", len(dataset))
            return {"train": dataset}

        else:
            raise ValueError(f"No valid data paths found in omnisql_data_paths: {paths}")

    def export_gguf(self, quantization: str = "q8_0"):
        """Export model to GGUF format for Ollama/llama.cpp deployment."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call setup() or train() first.")

        output_path = Path(self.config.output_dir) / "gguf"
        logger.info("Exporting to GGUF (%s)...", quantization)
        self.model.save_pretrained_gguf(
            str(output_path),
            self.tokenizer,
            quantization_method=quantization,
        )
        logger.info("GGUF exported to %s", output_path)
