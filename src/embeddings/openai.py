"""
OpenAI embedding provider.

Uses the ``openai`` SDK (already a core dependency) to call the OpenAI
Embeddings API.  Works with any OpenAI-compatible endpoint via ``base_url``.

Default model: ``text-embedding-3-large`` (3072-dim)

Usage:

    from src.embeddings import OpenAIEmbeddingModel

    model = OpenAIEmbeddingModel()                                   # default
    model = OpenAIEmbeddingModel("text-embedding-3-small")           # smaller/cheaper
    model = OpenAIEmbeddingModel(api_key="sk-...")                   # explicit key

    vectors = model.embed(["What are the top 5 products?"])          # list of texts
    vec     = model.embed_one("What are the top 5 products?")        # single text
    dim     = model.get_embedding_dimension()                        # 3072
"""

from __future__ import annotations

import logging
import os
from typing import Union

from .base import BaseEmbeddingModel

logger = logging.getLogger(__name__)

# Known dimensions for OpenAI embedding models
_KNOWN_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
}

_DEFAULT_MODEL = "text-embedding-3-large"
_DEFAULT_BATCH = 2048  # OpenAI allows up to 2048 inputs per request


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    """
    OpenAI Embeddings API provider.

    Parameters
    ----------
    model:
        OpenAI embedding model name.
        Defaults to ``text-embedding-3-large`` (3072-dim).
    api_key:
        OpenAI API key.  Falls back to ``OPENAI_API_KEY`` env var.
    base_url:
        Custom base URL for OpenAI-compatible endpoints (e.g. Azure,
        Together.ai, local vLLM).  ``None`` uses the default OpenAI URL.
    batch_size:
        Maximum number of texts per API call.  The OpenAI API supports
        up to 2048 inputs per request.
    dimensions:
        Override embedding dimensionality (for models that support it,
        e.g. ``text-embedding-3-*`` with Matryoshka truncation).
        ``None`` uses the model's native dimensionality.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        batch_size: int = _DEFAULT_BATCH,
        dimensions: int | None = None,
    ):
        self.model = model
        self.batch_size = batch_size
        self._dimensions_override = dimensions

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "OpenAI API key required.  Pass api_key= or set the "
                "OPENAI_API_KEY environment variable."
            )

        try:
            import openai
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for OpenAIEmbeddingModel. "
                "Install it with:  pip install openai"
            )

        kwargs: dict = {"api_key": resolved_key}
        if base_url:
            kwargs["base_url"] = base_url

        self._client = openai.OpenAI(**kwargs)

        # Resolve dimension
        if dimensions is not None:
            self._dimension = dimensions
        elif model in _KNOWN_DIMENSIONS:
            self._dimension = _KNOWN_DIMENSIONS[model]
        else:
            # Unknown model -- probe with a tiny request
            self._dimension = self._probe_dimension()

        logger.info(
            "OpenAIEmbeddingModel ready (model=%s, dim=%d)",
            model,
            self._dimension,
        )

    # ------------------------------------------------------------------
    # BaseEmbeddingModel interface
    # ------------------------------------------------------------------

    def embed(self, texts: str | list[str]) -> list[list[float]]:
        """
        Embed one or more texts via the OpenAI Embeddings API.

        Parameters
        ----------
        texts:
            Single string or list of strings.

        Returns
        -------
        list[list[float]]
            One float vector per input text, in the same order.
        """
        if isinstance(texts, str):
            texts = [texts]

        all_vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]

            kwargs: dict = {"input": batch, "model": self.model}
            if self._dimensions_override is not None:
                kwargs["dimensions"] = self._dimensions_override

            response = self._client.embeddings.create(**kwargs)

            # Sort by index to guarantee order (API returns sorted, but be safe)
            sorted_data = sorted(response.data, key=lambda d: d.index)
            all_vectors.extend([d.embedding for d in sorted_data])

        return all_vectors

    def get_embedding_dimension(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        return self._dimension

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _probe_dimension(self) -> int:
        """Send a tiny request to discover the model's embedding dimension."""
        response = self._client.embeddings.create(
            input=["dimension probe"],
            model=self.model,
        )
        dim = len(response.data[0].embedding)
        logger.debug("Probed embedding dimension for '%s': %d", self.model, dim)
        return dim
