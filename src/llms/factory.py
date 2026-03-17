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
from dataclasses import dataclass, field
from typing import Any

from .base import BaseLLM
from .huggingface import HuggingFaceLLM
from .openai import OpenAILLM

logger = logging.getLogger(__name__)

# Registry — extend here if a truly incompatible provider is added later
_PROVIDERS: dict[str, type] = {
    "openai": OpenAILLM,
    "huggingface": HuggingFaceLLM,
    # Aliases
    "hf": HuggingFaceLLM,
    "local": HuggingFaceLLM,
}


# ---------------------------------------------------------------------------
# Typed config
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    """Typed, validated configuration for a single LLM provider.

    Replaces raw ``dict`` parsing in ``LLMFactory.from_config()`` with an
    explicit, IDE-friendly dataclass.  Extra provider kwargs (e.g.
    ``load_in_4bit``, ``organization``) are collected in ``extra``.

    Parameters
    ----------
    provider : str
        Provider name — ``"openai"``, ``"huggingface"``, ``"hf"``, ``"local"``.
    model_name : str
        Model identifier (OpenAI model string or local checkpoint path).
    enabled : bool
        When False, ``LLMFactory.from_config()`` returns None immediately.
    extra : dict
        Any additional kwargs forwarded verbatim to the provider constructor
        (e.g. ``api_key``, ``base_url``, ``load_in_4bit``).
    """

    provider: str = "openai"
    model_name: str = "gpt-4o-mini"
    enabled: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> LLMConfig:
        """Parse a raw config dictionary into a typed ``LLMConfig``.

        Special top-level keys ``provider``, ``model_name``, and ``enabled``
        are extracted; everything else goes into ``extra`` and is forwarded
        to the provider constructor.

        Parameters
        ----------
        config : dict
            Typically loaded directly from ``configs/default.yaml`` or passed
            inline in tests.

        Returns
        -------
        LLMConfig
        """
        config = dict(config)  # shallow copy — don't mutate caller's dict
        return cls(
            provider=str(config.pop("provider", "openai")).lower(),
            model_name=str(config.pop("model_name", "gpt-4o-mini")),
            enabled=bool(config.pop("enabled", True)),
            extra=config,  # remaining keys forwarded as provider kwargs
        )

    def to_kwargs(self) -> dict[str, Any]:
        """Return the kwargs dict to pass to the provider constructor."""
        return {"model_name": self.model_name, **self.extra}



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
    def from_config(cls, config: dict[str, Any]) -> BaseLLM | None:
        """
        Build an LLM from a configuration dictionary.

        Parses *config* into a typed ``LLMConfig`` first, then delegates
        to ``LLMFactory.create()``.  Returns None when ``enabled`` is False.

        Parameters
        ----------
        config : dict
            Must contain ``"provider"`` and ``"model_name"``.  Optional key
            ``"enabled": false`` short-circuits and returns None.
            All other keys are forwarded to the provider constructor.

        Returns
        -------
        BaseLLM | None
        """
        llm_cfg = LLMConfig.from_dict(config)
        if not llm_cfg.enabled:
            logger.info("LLM disabled (enabled=false in config)")
            return None
        return cls.create(llm_cfg.provider, **llm_cfg.to_kwargs())

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
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
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
