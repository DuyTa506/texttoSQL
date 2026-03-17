"""
ChatML formatting utilities for Qwen3 and compatible models.

A single canonical implementation consumed by:
  - src/generation/inference.py  (_messages_to_chatml)
  - src/llms/huggingface.py      (_build_chatml)

Format
------
    <|im_start|>system
    {system_content}<|im_end|>
    <|im_start|>user
    {user_content}<|im_end|>
    <|im_start|>assistant
"""

from __future__ import annotations

# Default system prompt shared by both the inference module and the HF LLM wrapper.
DEFAULT_SYSTEM_PROMPT = (
    "You are an expert SQL assistant. "
    "Given a database schema and a natural-language question, "
    "generate a correct SQL query. "
    "Always wrap your final SQL in a ```sql ... ``` code block."
)


def messages_to_chatml(messages: list[dict]) -> str:
    """Convert an OpenAI-style messages list to a ChatML string.

    Parameters
    ----------
    messages : list[dict]
        List of ``{"role": ..., "content": ...}`` dicts.
        Typical roles: ``"system"``, ``"user"``, ``"assistant"``.

    Returns
    -------
    str
        ChatML-formatted prompt ending with ``<|im_start|>assistant\\n``
        ready for the model to continue.
    """
    parts = [
        f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>"
        for msg in messages
    ]
    parts.append("<|im_start|>assistant")
    return "\n".join(parts)


def build_chatml(
    user_prompt: str,
    system_prompt: str | None = None,
) -> str:
    """Build a two-turn (system + user) ChatML prompt string.

    Convenience wrapper around ``messages_to_chatml`` for the common case
    of a single system message followed by a single user message.

    Parameters
    ----------
    user_prompt : str
        The user message content.
    system_prompt : str, optional
        System instruction.  Defaults to ``DEFAULT_SYSTEM_PROMPT``.

    Returns
    -------
    str
        ChatML-formatted prompt ready for generation.
    """
    return messages_to_chatml([
        {"role": "system",  "content": system_prompt or DEFAULT_SYSTEM_PROMPT},
        {"role": "user",    "content": user_prompt},
    ])
