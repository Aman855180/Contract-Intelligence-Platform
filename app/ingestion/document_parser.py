"""
document_parser.py

Top-level ingestion entry point. Validates a file, dispatches to the
correct format-specific parser, and assembles the final ParsedDocument
(metadata + per-page text) consumed by app/chunking and app/database.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from app.ingestion.docx_parser import parse_docx
from app.ingestion.exceptions import EmptyDocumentError
from app.ingestion.pdf_parser import parse_pdf
from app.ingestion.validation import validate_and_classify
from app.models.document import DocumentMetadata, DocumentType, ParsedDocument
from app.utils.logging_setup import get_logger
from app.utils.perf import perf_span

logger = get_logger(__name__)


def ingest_document(file_path: Path, contract_name: str | None = None) -> ParsedDocument:
    """
    Validate, parse, and assemble a ParsedDocument from a contract file on
    disk. This is the single entry point the rest of the application
    (database layer, chunking pipeline, Streamlit UI) should call.

    Args:
        file_path: Path to the uploaded PDF or DOCX file on disk.
        contract_name: Optional human-readable name override. Defaults to
            the file's stem (filename without extension).

    Returns:
        A fully validated ParsedDocument ready for SQLite storage and
        downstream chunking.

    Raises:
        UnsupportedFileTypeError: If the file extension isn't allowed.
        FileTooLargeError: If the file exceeds the configured size limit.
        DocumentParsingError: If the underlying parser fails.
        EmptyDocumentError: If parsing succeeds but yields no text at all.
    """
    with perf_span("document_ingestion_request", extra_info=f"file={file_path.name}"):
        document_type = validate_and_classify(file_path)

        if document_type == DocumentType.PDF:
            pages = parse_pdf(file_path)
        elif document_type == DocumentType.DOCX:
            pages = parse_docx(file_path)
        else:
            # Unreachable given validate_and_classify's contract, but kept explicit
            # rather than silently falling through.
            raise EmptyDocumentError(f"Unhandled document type: {document_type}")

        total_text_length = sum(len(p.text) for p in pages)
        if total_text_length == 0:
            logger.error("Document '%s' parsed with zero extractable text (possibly scanned/image-only).", file_path.name)
            raise EmptyDocumentError(
                f"No extractable text found in '{file_path.name}'. Scanned/image-only documents "
                "are out of scope for this project (no OCR support)."
            )

        document_id = uuid.uuid4().hex
        resolved_name = contract_name.strip() if contract_name and contract_name.strip() else file_path.stem

        metadata = DocumentMetadata(
            document_id=document_id,
            contract_name=resolved_name,
            document_type=document_type,
            page_count=len(pages),
            file_size_bytes=file_path.stat().st_size,
            source_path=str(file_path),
        )

        parsed_document = ParsedDocument(metadata=metadata, pages=pages)
        logger.info(
            "Ingested document '%s' (id=%s, type=%s, pages=%d).",
            resolved_name,
            document_id,
            document_type.value,
            len(pages),
        )
        return parsed_document
