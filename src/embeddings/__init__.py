"""
Embedding model submodule.

Current provider:
  HuggingFaceEmbeddingModel  — sentence-transformers (no extra deps)

Default model: ``paraphrase-multilingual-mpnet-base-v2`` (768-dim)
  matches the existing SchemaIndexer so no re-indexing is needed.

Usage:

    from embeddings import HuggingFaceEmbeddingModel

    model = HuggingFaceEmbeddingModel()                           # default
    model = HuggingFaceEmbeddingModel("BAAI/bge-m3")             # swap model
    model = HuggingFaceEmbeddingModel(device="cuda")             # explicit device

    vectors = model.embed(["What are the top 5 products?"])      # list of texts
    vec     = model.embed_one("What are the top 5 products?")    # single text
"""

from .base import BaseEmbeddingModel
from .huggingface import HuggingFaceEmbeddingModel

__all__ = [
    "BaseEmbeddingModel",
    "HuggingFaceEmbeddingModel",
]
