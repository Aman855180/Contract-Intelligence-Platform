"""
pipeline.py

Orchestrates the full chunking -> embedding -> ChromaDB storage pipeline
for a single parsed document. This is the single entry point Phase 11's
Streamlit UI (and Phase 10's evaluation harness) should call after a
document has been ingested and stored in SQLite (Phase 2).
"""

from __future__ import annotations

from app.chunking.chunker import chunk_parsed_document
from app.embeddings.chroma_store import add_chunks
from app.embeddings.embedder import embed_texts
from app.models.document import DocumentChunk, ParsedDocument
from app.utils.logging_setup import get_logger
from app.utils.perf import perf_span

logger = get_logger(__name__)


def index_document(parsed_document: ParsedDocument) -> list[DocumentChunk]:
    """
    Chunk a parsed document, embed every chunk, and persist the chunks +
    embeddings to ChromaDB.

    Args:
        parsed_document: The parsed document to chunk and index. Should
            already be stored in SQLite via app.database.repository before
            calling this (this function does not touch SQLite).

    Returns:
        The list of DocumentChunk objects that were created and indexed.
        Returns an empty list (with a warning logged) if the document
        produced zero chunks.
    """
    with perf_span("document_indexing", extra_info=f"doc_id={parsed_document.metadata.document_id}"):
        with perf_span("document_chunking"):
            chunks = chunk_parsed_document(parsed_document)
        if not chunks:
            logger.warning(
                "Document '%s' (id=%s) produced zero chunks; nothing to index.",
                parsed_document.metadata.contract_name,
                parsed_document.metadata.document_id,
            )
            return []

        embeddings = embed_texts([c.text for c in chunks])
        add_chunks(chunks, embeddings)

        logger.info(
            "Indexed document '%s' (id=%s): %d chunk(s) embedded and stored.",
            parsed_document.metadata.contract_name,
            parsed_document.metadata.document_id,
            len(chunks),
        )
        return chunks
