"""
Embedding model submodule.

Providers:
  OpenAIEmbeddingModel       — OpenAI Embeddings API (no torch required)
  HuggingFaceEmbeddingModel  — sentence-transformers (requires torch)

Default for pipeline: ``OpenAIEmbeddingModel`` with ``text-embedding-3-large``
  (3072-dim, set via OPENAI_API_KEY env var or .env file).

Usage:

    from src.embeddings import OpenAIEmbeddingModel

    model = OpenAIEmbeddingModel()                                  # default
    model = OpenAIEmbeddingModel("text-embedding-3-small")          # cheaper

    # Local (requires torch + sentence-transformers):
    from src.embeddings import HuggingFaceEmbeddingModel
    model = HuggingFaceEmbeddingModel()                             # default
    model = HuggingFaceEmbeddingModel("BAAI/bge-m3")                # swap model

    vectors = model.embed(["What are the top 5 products?"])         # list of texts
    vec     = model.embed_one("What are the top 5 products?")       # single text
"""

from .base import BaseEmbeddingModel
from .openai import OpenAIEmbeddingModel

__all__ = [
    "BaseEmbeddingModel",
    "OpenAIEmbeddingModel",
]

# HuggingFaceEmbeddingModel is lazy-imported to avoid pulling in torch
# when only OpenAI embeddings are needed.


def __getattr__(name: str):
    if name == "HuggingFaceEmbeddingModel":
        from .huggingface import HuggingFaceEmbeddingModel

        return HuggingFaceEmbeddingModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
