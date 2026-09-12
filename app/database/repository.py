"""
repository.py

Repository layer providing typed CRUD operations over the documents and
document_pages tables. This is the only module in the codebase that should
contain raw SQL — all other modules interact with documents via these
functions and the Pydantic models in app/models/document.py.
"""

from __future__ import annotations

import sqlite3

from app.database.exceptions import DocumentNotFoundError, DuplicateDocumentError
from app.database.sqlite_db import get_connection
from app.models.document import DocumentMetadata, DocumentType, PageText, ParsedDocument
from app.utils.logging_setup import get_logger

logger = get_logger(__name__)


def insert_parsed_document(parsed_document: ParsedDocument) -> None:
    """
    Persist a ParsedDocument's metadata and per-page text to SQLite in a
    single transaction.

    Args:
        parsed_document: The validated ParsedDocument to store.

    Raises:
        DuplicateDocumentError: If a document with the same document_id
            already exists.
    """
    meta = parsed_document.metadata
    try:
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO documents
                    (document_id, contract_name, document_type, upload_date,
                     page_count, file_size_bytes, source_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    meta.document_id,
                    meta.contract_name,
                    meta.document_type.value,
                    meta.upload_date.isoformat(),
                    meta.page_count,
                    meta.file_size_bytes,
                    meta.source_path,
                ),
            )
            conn.executemany(
                "INSERT INTO document_pages (document_id, page_number, text) VALUES (?, ?, ?)",
                [(meta.document_id, page.page_number, page.text) for page in parsed_document.pages],
            )
    except sqlite3.IntegrityError as exc:
        logger.error("Duplicate document_id on insert: %s", meta.document_id)
        raise DuplicateDocumentError(f"Document with id '{meta.document_id}' already exists.") from exc

    logger.info("Stored document '%s' (id=%s) with %d page(s) in SQLite.", meta.contract_name, meta.document_id, len(parsed_document.pages))


def get_document_metadata(document_id: str) -> DocumentMetadata:
    """
    Retrieve a single document's metadata by ID.

    Args:
        document_id: The document's unique identifier.

    Returns:
        The DocumentMetadata for the requested document.

    Raises:
        DocumentNotFoundError: If no document with this ID exists.
    """
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE document_id = ?", (document_id,)
        ).fetchone()

    if row is None:
        raise DocumentNotFoundError(f"No document found with id '{document_id}'.")

    return _row_to_metadata(row)


def get_document_pages(document_id: str) -> list[PageText]:
    """
    Retrieve all pages for a document, ordered by page_number.

    Args:
        document_id: The document's unique identifier.

    Returns:
        List of PageText objects in page order.

    Raises:
        DocumentNotFoundError: If no document with this ID exists.
    """
    # Ensure the parent document exists for a clear error message.
    get_document_metadata(document_id)

    with get_connection() as conn:
        rows = conn.execute(
            "SELECT page_number, text FROM document_pages WHERE document_id = ? ORDER BY page_number ASC",
            (document_id,),
        ).fetchall()

    return [PageText(page_number=row["page_number"], text=row["text"]) for row in rows]


def get_parsed_document(document_id: str) -> ParsedDocument:
    """
    Retrieve the full ParsedDocument (metadata + pages) for a document.

    Args:
        document_id: The document's unique identifier.

    Returns:
        The reconstructed ParsedDocument.

    Raises:
        DocumentNotFoundError: If no document with this ID exists.
    """
    metadata = get_document_metadata(document_id)
    pages = get_document_pages(document_id)
    return ParsedDocument(metadata=metadata, pages=pages)


def list_documents() -> list[DocumentMetadata]:
    """
    List metadata for all documents currently stored, most recently
    uploaded first.

    Returns:
        List of DocumentMetadata objects.
    """
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM documents ORDER BY upload_date DESC").fetchall()
    return [_row_to_metadata(row) for row in rows]


def document_exists(document_id: str) -> bool:
    """Return True if a document with the given ID exists in the database."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM documents WHERE document_id = ? LIMIT 1", (document_id,)
        ).fetchone()
    return row is not None


def delete_document(document_id: str) -> None:
    """
    Delete a document and its pages (cascades via foreign key) from SQLite.

    Args:
        document_id: The document's unique identifier.

    Raises:
        DocumentNotFoundError: If no document with this ID exists.
    """
    if not document_exists(document_id):
        raise DocumentNotFoundError(f"No document found with id '{document_id}'.")

    with get_connection() as conn:
        conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
    logger.info("Deleted document id=%s from SQLite.", document_id)


def _row_to_metadata(row: sqlite3.Row) -> DocumentMetadata:
    """Convert a sqlite3.Row from the documents table into a DocumentMetadata."""
    return DocumentMetadata(
        document_id=row["document_id"],
        contract_name=row["contract_name"],
        document_type=DocumentType(row["document_type"]),
        upload_date=row["upload_date"],
        page_count=row["page_count"],
        file_size_bytes=row["file_size_bytes"],
        source_path=row["source_path"],
    )
