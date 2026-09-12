"""
chroma_store.py

ChromaDB persistent vector store wrapper. Stores DocumentChunk text and
full metadata alongside precomputed embeddings (generated via
app/embeddings/embedder.py). This module never computes embeddings itself
— it only persists and queries vectors that are passed in, which keeps it
trivially testable without a real embedding model.

Per project scope, embeddings/vectors live exclusively here — never in
SQLite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.embeddings.exceptions import EmbeddingError
from app.models.document import DocumentChunk
from app.utils.config_loader import get_project_root, get_settings
from app.utils.logging_setup import get_logger
from app.utils.perf import perf_span

logger = get_logger(__name__)

_COLLECTION_NAME = "contract_chunks"
_client_instance: Any = None


def _get_persist_dir() -> Path:
    """Resolve the absolute ChromaDB persistence directory from settings.yaml."""
    settings = get_settings()
    persist_dir_rel = settings["paths"]["chroma_persist_dir"]
    return get_project_root() / persist_dir_rel


def get_client() -> "chromadb.ClientAPI":
    """
    Return a cached, persistent ChromaDB client. Telemetry is disabled to
    avoid any outbound network calls in offline/sandboxed environments.

    Returns:
        A chromadb PersistentClient instance.
    """
    global _client_instance
    if _client_instance is not None:
        return _client_instance

    persist_dir = _get_persist_dir()
    persist_dir.mkdir(parents=True, exist_ok=True)

    _client_instance = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    logger.info("ChromaDB persistent client initialized at %s.", persist_dir)
    return _client_instance


def reset_client_cache() -> None:
    """Clear the cached ChromaDB client. Used by tests to point at a fresh temp store."""
    global _client_instance
    _client_instance = None


def get_or_create_collection() -> "chromadb.Collection":
    """
    Return the single shared collection used for all contract chunks across
    all documents (documents are distinguished via the document_id metadata
    field, not via separate collections).

    Returns:
        The ChromaDB collection for contract chunks.
    """
    client = get_client()
    return client.get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
    """
    Store a batch of document chunks and their precomputed embeddings in
    ChromaDB, with full metadata for citation purposes.

    Args:
        chunks: The DocumentChunk objects to store.
        embeddings: Precomputed embedding vectors, same order/length as chunks.

    Raises:
        EmbeddingError: If chunks and embeddings have mismatched lengths,
            or if chunks is empty while embeddings is not (or vice versa).
    """
    if len(chunks) != len(embeddings):
        raise EmbeddingError(
            f"Mismatched lengths: {len(chunks)} chunk(s) vs {len(embeddings)} embedding(s)."
        )
    if not chunks:
        logger.warning("add_chunks called with an empty chunk list; nothing to store.")
        return

    with perf_span("chroma_add_chunks", extra_info=f"count={len(chunks)}"):
        collection = get_or_create_collection()
        collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "document_id": c.document_id,
                    "contract_name": c.contract_name,
                    "page_number": c.page_number,
                    "chunk_index": c.chunk_index,
                    "char_start": c.char_start,
                    "char_end": c.char_end,
                }
                for c in chunks
            ],
        )
    logger.info("Stored %d chunk(s) in ChromaDB collection '%s'.", len(chunks), _COLLECTION_NAME)


def query_similar(
    query_embedding: list[float],
    top_k: int,
    document_id: str | None = None,
) -> list[dict]:
    """
    Query ChromaDB for the top-K most similar chunks to a query embedding.

    Args:
        query_embedding: The query's embedding vector.
        top_k: Number of results to return.
        document_id: Optional — if supplied, restricts results to chunks
            belonging to this document. Used by the risk module to avoid
            retrieving evidence from other indexed contracts.

    Returns:
        List of result dicts, each containing: chunk_id, text, metadata,
        and distance (cosine distance; lower is more similar).
    """
    with perf_span("chroma_vector_retrieval", extra_info=f"top_k={top_k} doc_id={document_id}"):
        collection = get_or_create_collection()
        if collection.count() == 0:
            return []

        where_filter = {"document_id": document_id} if document_id else None

        raw = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            where=where_filter,
        )

        results: list[dict] = []
        ids = raw.get("ids", [[]])[0]
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            results.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "metadata": metadata,
                    "distance": distance,
                }
            )
        return results


def delete_document_chunks(document_id: str) -> None:
    """
    Delete all chunks belonging to a given document from ChromaDB. Used to
    keep ChromaDB in sync if a document is removed via the SQLite repository.

    Args:
        document_id: The document whose chunks should be removed.
    """
    collection = get_or_create_collection()
    collection.delete(where={"document_id": document_id})
    logger.info("Deleted all chunks for document_id=%s from ChromaDB.", document_id)


def collection_count() -> int:
    """Return the total number of chunks currently stored in ChromaDB."""
    return get_or_create_collection().count()


def get_chunks_by_document(document_id: str) -> list[dict]:
    """
    Fetch all chunks belonging to a single document, ordered by chunk_index.
    Used by the structured field extraction pipeline (app/extraction) to
    iterate over a document's chunks without re-reading SQLite/re-chunking.

    Args:
        document_id: The document whose chunks should be retrieved.

    Returns:
        List of dicts, each with: chunk_id, text, metadata. Empty list if
        the document has no chunks indexed.
    """
    collection = get_or_create_collection()
    if collection.count() == 0:
        return []

    data = collection.get(where={"document_id": document_id}, include=["documents", "metadatas"])
    results = [
        {"chunk_id": chunk_id, "text": text, "metadata": metadata}
        for chunk_id, text, metadata in zip(data["ids"], data["documents"], data["metadatas"])
    ]
    results.sort(key=lambda r: r["metadata"].get("chunk_index", 0))
    return results
