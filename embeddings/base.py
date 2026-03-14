"""
Base class for embedding model implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Union


class BaseEmbeddingModel(ABC):
    """
    Abstract base for embedding providers.

    All implementations must support:
      embed(texts)  →  list of float vectors

    The interface accepts either a single string or a list of strings so
    callers do not need to special-case single inputs.
    """

    @abstractmethod
    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Embed one or more texts into dense float vectors.

        Parameters
        ----------
        texts:
            A single string or a list of strings to embed.

        Returns
        -------
        list[list[float]]
            One float vector per input text, in the same order.
            All vectors have the same dimensionality.
        """

    def embed_one(self, text: str) -> List[float]:
        """
        Convenience wrapper: embed a single string and return the vector.

        Parameters
        ----------
        text:
            Text to embed.

        Returns
        -------
        list[float]
            Single embedding vector.
        """
        return self.embed([text])[0]

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Return the dimensionality of the embedding vectors."""
