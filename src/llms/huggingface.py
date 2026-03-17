"""
HuggingFace local model LLM provider.

Loads any HuggingFace / Unsloth checkpoint with AutoModelForCausalLM.
Uses ChatML format (<|im_start|> / <|im_end|>) which Qwen3 and most
instruction-tuned models expect.

Use this for:
  - Evaluating SFT / RL fine-tuned checkpoints on GPU hardware
  - Offline inference without API costs
  - Any HuggingFace-compatible model (Qwen, LLaMA, Mistral, etc.)
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

from .base import BaseLLM

logger = logging.getLogger(__name__)


class HuggingFaceLLM(BaseLLM):
    """
    Local HuggingFace model provider via AutoModelForCausalLM.

    Parameters
    ----------
    model_name:
        Path to local checkpoint directory or HuggingFace model ID.
    default_temperature:
        Temperature used when the caller does not specify one.
        0.0 = greedy / deterministic.
    default_max_tokens:
        Max new tokens generated per call.
    load_in_4bit:
        Load model in 4-bit quantization (QLoRA / bitsandbytes).
        Reduces VRAM usage significantly; requires ``bitsandbytes`` package.
    trust_remote_code:
        Pass ``trust_remote_code=True`` to from_pretrained().
        Required for some models (e.g. Qwen3).
    """

    def __init__(
        self,
        model_name: str,
        default_temperature: float = 0.0,
        default_max_tokens: int = 512,
        load_in_4bit: bool = False,
        trust_remote_code: bool = True,
    ):
        self.model_name = model_name
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.load_in_4bit = load_in_4bit
        self.trust_remote_code = trust_remote_code

        self._model = None
        self._tokenizer = None

    # ── Lazy load ─────────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Lazy-load model and tokenizer on first generate() call."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading HuggingFace model: %s", self.model_name)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )

        load_kwargs: dict = {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            except ImportError:
                logger.warning(
                    "bitsandbytes not installed — falling back to bfloat16 (no 4-bit)."
                )

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **load_kwargs
        )
        self._model.eval()
        logger.info("HuggingFace model loaded.")

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

    # ── Core ──────────────────────────────────────────────────────────────────

    def generate(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> str:
        import torch

        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens

        prompt = _build_chatml(user_prompt, system_prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        gen_kwargs: dict = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "temperature": max(temperature, 1e-7),
            "pad_token_id": (
                self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            ),
            **kwargs,
        }

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    def stream(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> Iterator[str]:
        """
        Token-by-token streaming via HuggingFace TextIteratorStreamer.

        Falls back to a single-chunk yield if ``transformers`` streamer is
        not available (older versions).
        """
        try:
            import torch
            from transformers import TextIteratorStreamer
            import threading

            temperature = (
                temperature if temperature is not None else self.default_temperature
            )
            max_tokens = (
                max_tokens if max_tokens is not None else self.default_max_tokens
            )

            prompt = _build_chatml(user_prompt, system_prompt)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )
            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "do_sample": temperature > 0,
                "temperature": max(temperature, 1e-7),
                "pad_token_id": (
                    self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                ),
                "streamer": streamer,
                **kwargs,
            }

            thread = threading.Thread(
                target=self.model.generate,
                kwargs={**inputs, **gen_kwargs},
                daemon=True,
            )
            thread.start()

            for token in streamer:
                yield token

        except Exception:
            # Fallback: yield complete response
            yield self.generate(user_prompt, system_prompt, temperature, max_tokens)

    def get_model_info(self) -> dict[str, Any]:
        return {
            "provider": "huggingface",
            "model_name": self.model_name,
            "load_in_4bit": self.load_in_4bit,
            "default_temperature": self.default_temperature,
            "default_max_tokens": self.default_max_tokens,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

from src.llms.chat_format import DEFAULT_SYSTEM_PROMPT as _DEFAULT_HF_SYSTEM_PROMPT
from src.llms.chat_format import build_chatml as _build_chatml
