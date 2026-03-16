"""
Schema Indexer -- embeds schema chunks and indexes them into ChromaDB.

Accepts any ``BaseEmbeddingModel`` (OpenAI, HuggingFace, etc.) as the
embedding encoder.  When no encoder is provided, falls back to loading
``SentenceTransformer`` locally (requires torch + sentence-transformers).

ChromaDB runs in embedded mode -- no Docker needed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import chromadb
from chromadb.config import Settings
from tqdm import tqdm

from .schema_chunker import SchemaChunk

if TYPE_CHECKING:
    from ..embeddings.base import BaseEmbeddingModel

logger = logging.getLogger(__name__)


class SchemaIndexer:
    """Embeds :class:`SchemaChunk` objects and indexes them in ChromaDB.

    Parameters
    ----------
    embedding_model:
        Model name passed to the fallback ``SentenceTransformer`` loader.
        Ignored when *encoder* is provided.
    persist_dir:
        Path to the ChromaDB persistent storage directory.
    collection_name:
        Name of the ChromaDB collection to use.
    batch_size:
        Number of chunks embedded per batch.
    encoder:
        Pre-built embedding model implementing ``BaseEmbeddingModel``.
        When provided the indexer uses it directly and never touches
        ``SentenceTransformer``.  This is the recommended path when you
        want to avoid pulling in torch.
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        persist_dir: str = "./data/chroma_db",
        collection_name: str = "schema_chunks",
        batch_size: int = 128,
        encoder: "BaseEmbeddingModel | None" = None,
    ):
        self.embedding_model_name = embedding_model
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.batch_size = batch_size

        # External encoder takes priority over lazy SentenceTransformer
        self._external_encoder: "BaseEmbeddingModel | None" = encoder

        # Lazy-loaded (only used when no external encoder is provided)
        self._st_encoder = None  # SentenceTransformer instance
        self._client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None

    # ---- lazy initialisation ------------------------------------------------

    def _get_st_encoder(self):
        """Lazy-load a SentenceTransformer (fallback when no external encoder)."""
        if self._st_encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required when no external encoder "
                    "is provided.  Either pass encoder= to SchemaIndexer or "
                    "install sentence-transformers:  pip install sentence-transformers"
                )
            logger.info("Loading SentenceTransformer: %s", self.embedding_model_name)
            self._st_encoder = SentenceTransformer(self.embedding_model_name)
        return self._st_encoder

    @property
    def client(self) -> chromadb.ClientAPI:
        if self._client is None:
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
        return self._client

    @property
    def collection(self) -> chromadb.Collection:
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    # ---- internal embedding helpers -----------------------------------------

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using whichever encoder is available."""
        if self._external_encoder is not None:
            return self._external_encoder.embed(texts)
        # Fallback to SentenceTransformer
        encoder = self._get_st_encoder()
        vectors = encoder.encode(texts, show_progress_bar=False)
        return [v.tolist() for v in vectors]

    # ---- Backward-compat property: encoder ----------------------------------
    # Code that accessed `indexer.encoder` directly (e.g. HybridRetriever)
    # should still work.  Returns the SentenceTransformer when available, or
    # a thin wrapper when using an external BaseEmbeddingModel.

    @property
    def encoder(self):
        """Return a SentenceTransformer-like encoder for backward compat.

        If an external ``BaseEmbeddingModel`` is active, returns a lightweight
        wrapper that exposes ``.encode()`` so existing callers (e.g.
        ``HybridRetriever``) keep working.
        """
        if self._external_encoder is not None:
            return _STCompatWrapper(self._external_encoder)
        return self._get_st_encoder()

    # ---- public API ---------------------------------------------------------

    def index(self, chunks: list[SchemaChunk], *, reset: bool = False) -> int:
        """Embed and index *chunks* into ChromaDB.

        Parameters
        ----------
        chunks : list[SchemaChunk]
            Chunks to index.
        reset : bool
            If True, delete existing collection before indexing.

        Returns
        -------
        int
            Number of indexed chunks.
        """
        if reset:
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass  # collection may not exist yet
            self._collection = None  # force re-creation
            # Eagerly re-create so the UUID is fresh before any upserts
            _ = self.collection

        # Prepare data
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{chunk.db_id}__{chunk.chunk_type}__{chunk.table_name}__{i}"
            ids.append(chunk_id)
            documents.append(chunk.content)
            metadatas.append(
                {
                    "db_id": chunk.db_id,
                    "chunk_type": chunk.chunk_type,
                    "table_name": chunk.table_name,
                    **(chunk.metadata or {}),
                }
            )

        # Batch embed + upsert
        total = 0
        for start in tqdm(range(0, len(documents), self.batch_size), desc="Indexing"):
            end = start + self.batch_size
            batch_docs = documents[start:end]
            batch_ids = ids[start:end]
            batch_meta = metadatas[start:end]

            embeddings = self._embed_texts(batch_docs)

            self.collection.upsert(
                ids=batch_ids,
                embeddings=embeddings,
                documents=batch_docs,
                metadatas=batch_meta,
            )
            total += len(batch_ids)

        logger.info("Indexed %d chunks into collection '%s'", total, self.collection_name)
        return total

    def query(
        self,
        query_text: str,
        *,
        top_k: int = 10,
        db_id: str | None = None,
        chunk_type: str | None = None,
    ) -> list[dict]:
        """Query the index for similar chunks.

        Returns list of dicts with keys: id, content, metadata, distance.
        """
        where_clause: dict | None = None
        conditions = []
        if db_id:
            conditions.append({"db_id": db_id})
        if chunk_type:
            conditions.append({"chunk_type": chunk_type})

        if len(conditions) == 1:
            where_clause = conditions[0]
        elif len(conditions) > 1:
            where_clause = {"$and": conditions}

        query_embedding = self._embed_texts([query_text])

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=where_clause,
        )

        # Flatten ChromaDB's nested response
        output = []
        if results and results["ids"]:
            for i, doc_id in enumerate(results["ids"][0]):
                output.append(
                    {
                        "id": doc_id,
                        "content": results["documents"][0][i] if results["documents"] else "",
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": results["distances"][0][i] if results["distances"] else 0.0,
                    }
                )
        return output


class _STCompatWrapper:
    """Thin wrapper so callers expecting ``.encode()`` keep working."""

    def __init__(self, model: "BaseEmbeddingModel"):
        self._model = model

    def encode(self, texts: str | list[str], **kwargs):
        """Mimic SentenceTransformer.encode() return value (list of lists)."""
        import numpy as np

        if isinstance(texts, str):
            texts = [texts]
        vectors = self._model.embed(texts)
        return np.array(vectors)
