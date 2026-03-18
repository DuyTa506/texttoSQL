"""
Tests for async/batch inference capabilities.

Tests:
  - _OpenAIBackend.async_complete() with mocked AsyncOpenAI client
  - Semaphore concurrency limit in SQLInference.agenerate_batch()
  - _LocalBackend.batch_complete() returns correct count
  - SQLInference.agenerate() async generation
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.inference import (
    SQLInference,
    _OpenAIBackend,
    _LocalBackend,
    _build_messages,
    _parse_output,
)


# =============================================================================
# _OpenAIBackend.async_complete() tests
# =============================================================================


class TestOpenAIBackendAsyncComplete:
    """Test the async_complete() method on _OpenAIBackend."""

    def _make_backend(self):
        backend = _OpenAIBackend(
            model="gpt-4o-mini",
            api_key="test-key-123",
        )
        # Pre-set compat strategy so it doesn't try sync discovery
        backend._compat_strategy = 0
        return backend

    @pytest.mark.asyncio
    async def test_async_complete_returns_content(self):
        """async_complete should return the content from the API response."""
        backend = self._make_backend()

        # Mock the async client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "SELECT * FROM users"

        mock_async_client = AsyncMock()
        mock_async_client.chat.completions.create = AsyncMock(return_value=mock_response)
        backend._async_client = mock_async_client

        messages = [{"role": "user", "content": "Hello"}]
        result = await backend.async_complete(messages, temperature=0.0)

        assert result == "SELECT * FROM users"
        mock_async_client.chat.completions.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_async_complete_uses_cached_strategy(self):
        """async_complete should use the cached compat strategy."""
        backend = self._make_backend()
        backend._compat_strategy = 2  # strategy 2: max_tokens + temperature

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "SELECT 1"

        mock_async_client = AsyncMock()
        mock_async_client.chat.completions.create = AsyncMock(return_value=mock_response)
        backend._async_client = mock_async_client

        messages = [{"role": "user", "content": "test"}]
        await backend.async_complete(messages, temperature=0.5)

        # Verify strategy 2 kwargs were used
        call_kwargs = mock_async_client.chat.completions.create.call_args
        assert "max_tokens" in call_kwargs.kwargs
        assert "temperature" in call_kwargs.kwargs
        assert call_kwargs.kwargs["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_async_complete_empty_content(self):
        """async_complete should return empty string when content is None."""
        backend = self._make_backend()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None

        mock_async_client = AsyncMock()
        mock_async_client.chat.completions.create = AsyncMock(return_value=mock_response)
        backend._async_client = mock_async_client

        messages = [{"role": "user", "content": "test"}]
        result = await backend.async_complete(messages)
        assert result == ""


# =============================================================================
# SQLInference.agenerate() tests
# =============================================================================


class TestSQLInferenceAgenerate:
    """Test the agenerate() async method."""

    @pytest.mark.asyncio
    async def test_agenerate_with_async_backend(self):
        """agenerate should call backend.async_complete when available."""
        mock_backend = MagicMock()
        mock_backend.async_complete = AsyncMock(
            return_value="```sql\nSELECT id FROM users\n```"
        )

        inference = SQLInference(mock_backend, temperature=0.0)
        result = await inference.agenerate("What users?", "CREATE TABLE users (id INT)")

        assert result["sql"] == "SELECT id FROM users"
        mock_backend.async_complete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_agenerate_fallback_to_thread(self):
        """agenerate should fall back to asyncio.to_thread for backends without async."""
        mock_backend = MagicMock()
        # No async_complete attribute
        del mock_backend.async_complete
        mock_backend.complete = MagicMock(
            return_value="```sql\nSELECT * FROM orders\n```"
        )

        inference = SQLInference(mock_backend, temperature=0.0)
        result = await inference.agenerate("List orders", "CREATE TABLE orders (id INT)")

        assert result["sql"] == "SELECT * FROM orders"
        mock_backend.complete.assert_called_once()


# =============================================================================
# SQLInference.agenerate_batch() tests
# =============================================================================


class TestSQLInferenceAgenerateBatch:
    """Test the agenerate_batch() async batch method."""

    @pytest.mark.asyncio
    async def test_batch_returns_correct_count(self):
        """agenerate_batch should return one result per input item."""
        call_count = 0

        async def mock_async_complete(messages, temperature=0.0):
            nonlocal call_count
            call_count += 1
            return f"```sql\nSELECT {call_count}\n```"

        mock_backend = MagicMock(spec=[])  # spec=[] prevents auto-attribute creation
        mock_backend.async_complete = mock_async_complete

        inference = SQLInference(mock_backend, temperature=0.0)

        items = [
            ("q1", "schema1"),
            ("q2", "schema2"),
            ("q3", "schema3"),
        ]
        results = await inference.agenerate_batch(items, concurrency=2)

        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results)
        assert all("sql" in r for r in results)

    @pytest.mark.asyncio
    async def test_batch_concurrency_limit(self):
        """agenerate_batch should respect the semaphore concurrency limit."""
        max_concurrent = 0
        current_concurrent = 0

        async def mock_async_complete(messages, temperature=0.0):
            nonlocal max_concurrent, current_concurrent
            current_concurrent += 1
            max_concurrent = max(max_concurrent, current_concurrent)
            await asyncio.sleep(0.05)  # Simulate API latency
            current_concurrent -= 1
            return "```sql\nSELECT 1\n```"

        mock_backend = MagicMock(spec=[])
        mock_backend.async_complete = mock_async_complete

        inference = SQLInference(mock_backend, temperature=0.0)

        items = [("q", "s") for _ in range(10)]
        results = await inference.agenerate_batch(items, concurrency=3)

        assert len(results) == 10
        assert max_concurrent <= 3, f"Max concurrent was {max_concurrent}, expected <= 3"

    @pytest.mark.asyncio
    async def test_batch_uses_local_batch_complete(self):
        """agenerate_batch should use batch_complete for local backends."""
        mock_backend = MagicMock()
        mock_backend.batch_complete = MagicMock(
            return_value=[
                "```sql\nSELECT 1\n```",
                "```sql\nSELECT 2\n```",
            ]
        )
        # Remove async_complete so it doesn't take the async path
        if hasattr(mock_backend, 'async_complete'):
            del mock_backend.async_complete

        inference = SQLInference(mock_backend, temperature=0.0)

        items = [("q1", "s1"), ("q2", "s2")]
        results = await inference.agenerate_batch(items, concurrency=5)

        assert len(results) == 2
        mock_backend.batch_complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_rate_limit_retry(self):
        """agenerate_batch should retry on rate limit errors."""
        call_count = 0

        async def mock_async_complete(messages, temperature=0.0):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Rate limit exceeded (429)")
            return "```sql\nSELECT 1\n```"

        mock_backend = MagicMock(spec=[])
        mock_backend.async_complete = mock_async_complete

        inference = SQLInference(mock_backend, temperature=0.0)

        items = [("q1", "s1")]
        results = await inference.agenerate_batch(items, concurrency=5)

        assert len(results) == 1
        assert results[0]["sql"] == "SELECT 1"
        assert call_count == 2  # First call failed, second succeeded


# =============================================================================
# _LocalBackend.batch_complete() tests
# =============================================================================


class TestLocalBackendBatchComplete:
    """Test _LocalBackend.batch_complete() with mocked model."""

    def test_batch_complete_returns_correct_count(self):
        """batch_complete should return one string per input message list."""
        torch = pytest.importorskip("torch")

        backend = _LocalBackend(model_path="/fake/path")

        # Mock tokenizer and model
        mock_tokenizer = MagicMock()
        mock_tokenizer.padding_side = "right"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 0

        # Mock tokenizer __call__ (for batched encoding)
        mock_inputs = MagicMock()
        mock_inputs.__getitem__ = MagicMock(return_value=MagicMock(shape=(2, 10)))
        mock_inputs.to = MagicMock(return_value=mock_inputs)
        mock_tokenizer.return_value = mock_inputs

        # Mock decode to return different SQL for each
        mock_tokenizer.decode = MagicMock(
            side_effect=["SELECT 1", "SELECT 2"]
        )

        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        # generate returns tensor of shape (batch, seq_len)
        mock_model.generate = MagicMock(
            return_value=torch.zeros(2, 20, dtype=torch.long)
        )

        backend._tokenizer = mock_tokenizer
        backend._model = mock_model

        messages_list = [
            [{"role": "user", "content": "q1"}],
            [{"role": "user", "content": "q2"}],
        ]

        results = backend.batch_complete(messages_list, temperature=0.0)

        assert len(results) == 2
        assert results[0] == "SELECT 1"
        assert results[1] == "SELECT 2"

    def test_batch_complete_restores_tokenizer_state(self):
        """batch_complete should restore original tokenizer padding settings."""
        torch = pytest.importorskip("torch")

        backend = _LocalBackend(model_path="/fake/path")

        mock_tokenizer = MagicMock()
        mock_tokenizer.padding_side = "right"
        mock_tokenizer.pad_token_id = 42
        mock_tokenizer.eos_token_id = 0

        mock_inputs = MagicMock()
        mock_inputs.__getitem__ = MagicMock(return_value=MagicMock(shape=(1, 5)))
        mock_inputs.to = MagicMock(return_value=mock_inputs)
        mock_tokenizer.return_value = mock_inputs
        mock_tokenizer.decode = MagicMock(return_value="SELECT 1")

        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        mock_model.generate = MagicMock(
            return_value=torch.zeros(1, 15, dtype=torch.long)
        )

        backend._tokenizer = mock_tokenizer
        backend._model = mock_model

        messages_list = [[{"role": "user", "content": "q1"}]]
        backend.batch_complete(messages_list)

        # Check that tokenizer state was restored
        assert mock_tokenizer.padding_side == "right"
        assert mock_tokenizer.pad_token_id == 42


# =============================================================================
# Helper function tests
# =============================================================================


class TestHelpers:
    """Test shared helper functions."""

    def test_build_messages_standard(self):
        messages = _build_messages("What users?", "CREATE TABLE users (id INT)")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "Question: What users?" in messages[1]["content"]

    def test_build_messages_empty_schema(self):
        """When schema_context is empty, question is passed as-is."""
        messages = _build_messages("Full prompt here", "")
        assert messages[1]["content"] == "Full prompt here"

    def test_parse_output_sql_block(self):
        raw = "Here is the SQL:\n```sql\nSELECT * FROM users\n```"
        result = _parse_output(raw)
        assert result["sql"] == "SELECT * FROM users"

    def test_parse_output_no_block(self):
        raw = "SELECT id FROM users WHERE name = 'Alice'"
        result = _parse_output(raw)
        assert "SELECT" in result["sql"]
