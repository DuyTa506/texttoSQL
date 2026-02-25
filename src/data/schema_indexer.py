"""
Schema Indexer – embeds schema chunks and indexes them into ChromaDB.

Uses sentence-transformers for embedding and ChromaDB for local vector storage.
No Docker needed – ChromaDB runs in embedded mode.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .schema_chunker import SchemaChunk

logger = logging.getLogger(__name__)


class SchemaIndexer:
    """Embeds :class:`SchemaChunk` objects and indexes them in ChromaDB."""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        persist_dir: str = "./data/chroma_db",
        collection_name: str = "schema_chunks",
        batch_size: int = 128,
    ):
        self.embedding_model_name = embedding_model
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.batch_size = batch_size

        # Lazy-loaded
        self._encoder: Optional[SentenceTransformer] = None
        self._client: Optional[chromadb.ClientAPI] = None
        self._collection: Optional[chromadb.Collection] = None

    # ---- lazy initialisation ------------------------------------------------

    @property
    def encoder(self) -> SentenceTransformer:
        if self._encoder is None:
            logger.info("Loading embedding model: %s", self.embedding_model_name)
            self._encoder = SentenceTransformer(self.embedding_model_name)
        return self._encoder

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
            self.client.delete_collection(self.collection_name)
            self._collection = None  # force re-creation

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

            embeddings = self.encoder.encode(batch_docs, show_progress_bar=False).tolist()

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
        db_id: Optional[str] = None,
        chunk_type: Optional[str] = None,
    ) -> list[dict]:
        """Query the index for similar chunks.

        Returns list of dicts with keys: id, content, metadata, distance.
        """
        where_clause: Optional[dict] = None
        conditions = []
        if db_id:
            conditions.append({"db_id": db_id})
        if chunk_type:
            conditions.append({"chunk_type": chunk_type})

        if len(conditions) == 1:
            where_clause = conditions[0]
        elif len(conditions) > 1:
            where_clause = {"$and": conditions}

        query_embedding = self.encoder.encode([query_text]).tolist()

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
