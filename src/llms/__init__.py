"""
LLM provider submodule.

Two providers are supported:

  OpenAILLM       — OpenAI Chat Completions API + every compatible endpoint
                    (Together.ai, Groq, Fireworks, Azure, local vLLM, LM Studio …)

  HuggingFaceLLM  — Local HuggingFace / Unsloth checkpoint via AutoModelForCausalLM

Use LLMFactory to construct instances from config dicts:

    from src.llms import LLMFactory

    llm = LLMFactory.openai("gpt-4o-mini")
    llm = LLMFactory.huggingface("./checkpoints/sft/final", load_in_4bit=True)
    llm = LLMFactory.from_config({"provider": "openai", "model_name": "gpt-4o-mini"})
"""

from .base import BaseLLM
from .factory import LLMFactory
from .huggingface import HuggingFaceLLM
from .openai import OpenAILLM

__all__ = [
    "BaseLLM",
    "LLMFactory",
    "OpenAILLM",
    "HuggingFaceLLM",
]
