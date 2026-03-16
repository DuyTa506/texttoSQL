"""
OpenAI LLM provider.

Covers every OpenAI-compatible endpoint via ``base_url``:
  - OpenAI API          (default, no base_url needed)
  - Azure OpenAI        base_url="https://<resource>.openai.azure.com/..."
  - Together.ai         base_url="https://api.together.xyz/v1"
  - Groq                base_url="https://api.groq.com/openai/v1"
  - Fireworks           base_url="https://api.fireworks.ai/inference/v1"
  - Local vLLM server   base_url="http://localhost:8000/v1"
  - Local LM Studio     base_url="http://localhost:1234/v1"

Any provider with an OpenAI-compatible Chat Completions endpoint works here —
you do NOT need separate provider classes for them.
"""

from __future__ import annotations

import logging
import os
from typing import Any, AsyncIterator, Iterator

from .base import BaseLLM

logger = logging.getLogger(__name__)

# Default system prompt for text-to-SQL tasks
_SQL_SYSTEM_PROMPT = (
    "You are an expert SQL assistant. "
    "Given a database schema and a natural-language question, "
    "generate a correct SQL query. "
    "Always wrap your final SQL in a ```sql ... ``` code block."
)


class OpenAILLM(BaseLLM):
    """
    LLM provider backed by the OpenAI Chat Completions API
    (or any OpenAI-compatible endpoint).

    Parameters
    ----------
    model_name:
        Model identifier, e.g. "gpt-4o-mini", "gpt-4o",
        "meta-llama/Llama-3-8B-Instruct" (for Together/Fireworks),
        "qwen/qwen3-4b" (for local vLLM).
    api_key:
        API key.  Falls back to ``OPENAI_API_KEY`` env var when ``None``.
    organization:
        Optional OpenAI organization ID.
    base_url:
        Override the API base URL for compatible endpoints.
        Leave ``None`` for the standard OpenAI API.
    default_temperature:
        Temperature used when the caller does not specify one.
    default_max_tokens:
        Token cap used when the caller does not specify one.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: str | None = None,
        organization: str | None = None,
        base_url: str | None = None,
        default_temperature: float = 0.0,
        default_max_tokens: int | None = 512,
    ):
        try:
            from openai import AsyncOpenAI, OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for OpenAILLM. "
                "Install it with:  pip install openai"
            )

        self.model_name = model_name
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "OpenAI API key not found. "
                "Pass api_key=..., set generation.api_key in config, "
                "or export OPENAI_API_KEY=<your-key>."
            )

        client_kwargs: dict = {"api_key": resolved_key}
        if organization:
            client_kwargs["organization"] = organization
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = OpenAI(**client_kwargs)
        self._async_client = AsyncOpenAI(**client_kwargs)

        logger.info(
            "OpenAILLM ready (model=%s%s)",
            model_name,
            f", base_url={base_url}" if base_url else "",
        )

    # ── Core ──────────────────────────────────────────────────────────────────

    def generate(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> str:
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens

        messages = self._build_messages(user_prompt, system_prompt)

        try:
            completion = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            answer = completion.choices[0].message.content or ""
            logger.debug("OpenAI generated %d chars", len(answer))
            return answer
        except Exception as exc:
            logger.error("OpenAI API error: %s", exc)
            raise

    def stream(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> Iterator[str]:
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens

        messages = self._build_messages(user_prompt, system_prompt)

        try:
            stream = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as exc:
            logger.error("OpenAI stream error: %s", exc)
            raise

    # ── Native async ──────────────────────────────────────────────────────────

    async def agenerate(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> str:
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens

        messages = self._build_messages(user_prompt, system_prompt)

        try:
            completion = await self._async_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            return completion.choices[0].message.content or ""
        except Exception as exc:
            logger.error("OpenAI async API error: %s", exc)
            raise

    async def astream(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens

        messages = self._build_messages(user_prompt, system_prompt)

        try:
            stream = await self._async_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as exc:
            logger.error("OpenAI async stream error: %s", exc)
            raise

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_messages(
        self, user_prompt: str, system_prompt: str | None
    ) -> list[dict]:
        return [
            {"role": "system", "content": system_prompt or _SQL_SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]

    def get_model_info(self) -> dict[str, Any]:
        return {
            "provider": "openai",
            "model_name": self.model_name,
            "default_temperature": self.default_temperature,
            "default_max_tokens": self.default_max_tokens,
        }
