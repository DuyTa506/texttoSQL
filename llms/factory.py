"""
LLM factory — centralized provider construction.

Only two providers are needed for this project:

  "openai"      OpenAI Chat Completions API + every compatible endpoint
                (Together.ai, Groq, Fireworks, local vLLM, LM Studio …)

  "huggingface" Local HuggingFace / Unsloth checkpoint via AutoModelForCausalLM

Every other provider (Ollama, Anthropic, Cohere …) can be accessed through
the "openai" provider by setting ``base_url`` to the compatible endpoint.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from .base import BaseLLM
from .huggingface import HuggingFaceLLM
from .openai import OpenAILLM

logger = logging.getLogger(__name__)

# Registry — extend here if a truly incompatible provider is added later
_PROVIDERS: Dict[str, type] = {
    "openai": OpenAILLM,
    "huggingface": HuggingFaceLLM,
    # Aliases
    "hf": HuggingFaceLLM,
    "local": HuggingFaceLLM,
}


class LLMFactory:
    """
    Factory for creating LLM instances from config dicts or explicit kwargs.

    Typical usage (from YAML config):

        config = {
            "provider": "openai",
            "model_name": "gpt-4o-mini",
            "api_key": "sk-...",        # or set OPENAI_API_KEY env var
        }
        llm = LLMFactory.from_config(config)

        # OpenAI-compatible endpoint (Together.ai, Groq, local vLLM, …)
        config = {
            "provider": "openai",
            "model_name": "meta-llama/Llama-3-8B-Instruct",
            "base_url": "https://api.together.xyz/v1",
            "api_key": "...",
        }

        # Local HuggingFace checkpoint
        config = {
            "provider": "huggingface",
            "model_name": "./checkpoints/sft/final",
            "load_in_4bit": True,
        }
    """

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> Optional[BaseLLM]:
        """
        Build an LLM from a configuration dictionary.

        Special keys consumed by the factory (not forwarded to the provider):
          ``enabled``  — if False, returns None immediately.
          ``provider`` — selects the provider class.

        All remaining keys are forwarded as kwargs to the provider constructor.

        Returns
        -------
        BaseLLM | None
            Configured LLM instance, or None if ``enabled`` is False.
        """
        if not config.get("enabled", True):
            logger.info("LLM disabled (enabled=false in config)")
            return None

        provider = (config.get("provider") or "openai").lower()
        kwargs = {k: v for k, v in config.items() if k not in ("enabled", "provider")}

        return cls.create(provider, **kwargs)

    @classmethod
    def create(cls, provider: str, **kwargs) -> BaseLLM:
        """
        Create an LLM instance by provider name.

        Parameters
        ----------
        provider:
            "openai" | "huggingface" | "hf" | "local"
        **kwargs:
            Forwarded to the provider constructor.

        Raises
        ------
        ValueError
            If the provider is not registered.
        ImportError
            If the provider's required package is not installed.
        """
        provider = provider.lower()
        if provider not in _PROVIDERS:
            available = ", ".join(sorted(set(_PROVIDERS.keys())))
            raise ValueError(
                f"Unknown LLM provider '{provider}'. "
                f"Available: {available}"
            )

        llm_class = _PROVIDERS[provider]
        logger.info("Creating %s LLM (model=%s)", provider, kwargs.get("model_name", "?"))
        instance = llm_class(**kwargs)
        logger.info("%s LLM ready.", provider)
        return instance

    # ── Convenience constructors ──────────────────────────────────────────────

    @classmethod
    def openai(
        cls,
        model_name: str = "gpt-4o-mini",
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        **kwargs,
    ) -> OpenAILLM:
        """
        Create an OpenAI (or compatible) LLM.

        Examples
        --------
        Standard OpenAI:
            llm = LLMFactory.openai("gpt-4o-mini")

        Groq (OpenAI-compatible):
            llm = LLMFactory.openai(
                "llama3-8b-8192",
                api_key=os.environ["GROQ_API_KEY"],
                base_url="https://api.groq.com/openai/v1",
            )

        Local vLLM server:
            llm = LLMFactory.openai(
                "Qwen/Qwen3-4B",
                api_key="token",
                base_url="http://localhost:8000/v1",
            )
        """
        return OpenAILLM(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            **kwargs,
        )

    @classmethod
    def huggingface(
        cls,
        model_name: str,
        *,
        load_in_4bit: bool = False,
        **kwargs,
    ) -> HuggingFaceLLM:
        """
        Create a local HuggingFace model LLM.

        Examples
        --------
        SFT checkpoint (fp16, needs 16 GB VRAM):
            llm = LLMFactory.huggingface("./checkpoints/sft/final")

        QLoRA (4-bit, fits in 8 GB VRAM):
            llm = LLMFactory.huggingface(
                "./checkpoints/sft/final",
                load_in_4bit=True,
            )
        """
        return HuggingFaceLLM(
            model_name=model_name,
            load_in_4bit=load_in_4bit,
            **kwargs,
        )

    @classmethod
    def available_providers(cls) -> list[str]:
        """Return list of registered provider names (deduplicated)."""
        return sorted(set(_PROVIDERS.keys()))

    @classmethod
    def register(cls, name: str, llm_class: type) -> None:
        """
        Register a custom provider.

        Parameters
        ----------
        name:
            Provider name string (e.g. "anthropic").
        llm_class:
            Class that inherits from BaseLLM.
        """
        if not issubclass(llm_class, BaseLLM):
            raise TypeError(f"{llm_class} must be a subclass of BaseLLM")
        _PROVIDERS[name.lower()] = llm_class
        logger.info("Registered custom LLM provider: %s", name)
