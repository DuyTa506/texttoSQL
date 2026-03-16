"""
HuggingFace sentence-transformers embedding provider.

Uses the ``sentence-transformers`` library (already a core dependency)
— no langchain, no extra installs required.

Default model matches the existing SchemaIndexer:
  ``paraphrase-multilingual-mpnet-base-v2``  (768-dim, multilingual)

Any sentence-transformers compatible model works as a drop-in replacement:
  - ``all-MiniLM-L6-v2``               (384-dim, English, very fast)
  - ``all-mpnet-base-v2``              (768-dim, English, high quality)
  - ``intfloat/multilingual-e5-large`` (1024-dim, multilingual, high quality)
  - ``BAAI/bge-m3``                    (1024-dim, multilingual, strong)
"""

from __future__ import annotations

import logging
from typing import List, Optional, Union

from .base import BaseEmbeddingModel

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "paraphrase-multilingual-mpnet-base-v2"
_DEFAULT_BATCH  = 128


class HuggingFaceEmbeddingModel(BaseEmbeddingModel):
    """
    Sentence-transformers embedding model.

    Parameters
    ----------
    model_name:
        HuggingFace model ID or local path.
        Defaults to ``paraphrase-multilingual-mpnet-base-v2`` to match
        the existing ``SchemaIndexer`` embedding model.
    device:
        ``"cuda"``, ``"cpu"``, or ``None`` (auto-detect).
    batch_size:
        Number of texts embedded per forward pass.
    normalize_embeddings:
        Whether to L2-normalize output vectors (recommended for cosine similarity).
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: Optional[str] = None,
        batch_size: int = _DEFAULT_BATCH,
        normalize_embeddings: bool = True,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for HuggingFaceEmbeddingModel. "
                "Install it with:  pip install sentence-transformers"
            )

        # Auto-detect device if not specified
        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        self.device = device
        self._model = SentenceTransformer(model_name, device=device)

        logger.info(
            "HuggingFaceEmbeddingModel ready (model=%s, device=%s, dim=%d)",
            model_name,
            device,
            self.get_embedding_dimension(),
        )

    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Embed one or more texts.

        Parameters
        ----------
        texts:
            Single string or list of strings.

        Returns
        -------
        list[list[float]]
            One L2-normalised (optional) float vector per input text.
        """
        if isinstance(texts, str):
            texts = [texts]

        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vectors]

    def get_embedding_dimension(self) -> int:
        """Return the output dimensionality of the model."""
        return self._model.get_sentence_embedding_dimension()
