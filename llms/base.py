"""
Base class for all LLM provider implementations.

Every provider must implement:
  generate(user_prompt, system_prompt, temperature, max_tokens) -> str

Streaming and async are optional overrides — the default implementations
wrap the sync generate() so providers that don't need them still work.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, Iterator, Optional

logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""

    # ── Required ──────────────────────────────────────────────────────────────

    @abstractmethod
    def generate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Generate a complete response.

        Parameters
        ----------
        user_prompt:
            The full user message (question, context, instructions already formatted).
        system_prompt:
            Optional system-level instruction.  When ``None`` the provider uses
            its own default (usually a SQL-assistant instruction).
        temperature:
            Sampling temperature.  0.0 = greedy / deterministic.
        max_tokens:
            Hard cap on generated tokens.  ``None`` = provider default.
        **kwargs:
            Provider-specific pass-through (top_p, stop sequences, etc.).

        Returns
        -------
        str
            The raw response text from the model.
        """

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return a dict with at least ``provider`` and ``model_name`` keys."""

    # ── Optional: streaming (default wraps generate) ──────────────────────────

    def stream(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Iterator[str]:
        """
        Stream response token by token.

        Default implementation yields the complete response in one chunk.
        Override for true streaming support.
        """
        yield self.generate(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    # ── Optional: async (default wraps sync in thread executor) ──────────────

    async def agenerate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Async generate.  Default wraps sync ``generate()`` in a thread executor.
        Override for a native async implementation.
        """
        return await asyncio.to_thread(
            self.generate,
            user_prompt,
            system_prompt,
            temperature,
            max_tokens,
            **kwargs,
        )

    async def astream(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Async stream.  Default wraps sync ``stream()``.
        Override for a native async streaming implementation.
        """
        for token in self.stream(
            user_prompt, system_prompt, temperature, max_tokens, **kwargs
        ):
            yield token
