"""
pdf_parser.py

PDF text extraction using PyMuPDF (fitz), preserving per-page boundaries.
Page numbers in the output are 1-indexed to match how a human would
reference a page in a contract (matters for citation accuracy throughout
the rest of the platform).
"""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

from app.ingestion.exceptions import DocumentParsingError
from app.models.document import PageText
from app.utils.logging_setup import get_logger
from app.utils.perf import perf_span

logger = get_logger(__name__)


def parse_pdf(file_path: Path) -> list[PageText]:
    """
    Extract per-page text from a PDF file using PyMuPDF.

    Args:
        file_path: Path to the PDF file on disk.

    Returns:
        List of PageText objects, one per page, 1-indexed page_number.
        Pages with no extractable text (e.g. scanned/image-only pages) are
        still included with empty string text — OCR is explicitly out of
        scope for this project, so such pages simply yield no chunks later.

    Raises:
        DocumentParsingError: If PyMuPDF fails to open or read the file.
    """
    with perf_span("document_loading", extra_info=f"file={file_path.name}"):
        try:
            doc = fitz.open(str(file_path))
        except Exception as exc:  # PyMuPDF raises various internal exception types
            logger.error("Failed to open PDF file %s: %s", file_path.name, exc)
            raise DocumentParsingError(f"Could not open PDF file '{file_path.name}': {exc}") from exc

    pages: list[PageText] = []
    with perf_span("document_parsing", extra_info=f"file={file_path.name} pages={doc.page_count}"):
        try:
            for page_index in range(doc.page_count):
                page = doc.load_page(page_index)
                text = page.get_text("text") or ""
                pages.append(PageText(page_number=page_index + 1, text=text.strip()))
        except Exception as exc:
            logger.error("Failed while extracting text from PDF %s: %s", file_path.name, exc)
            raise DocumentParsingError(f"Failed extracting text from '{file_path.name}': {exc}") from exc
        finally:
            doc.close()

    logger.info("Parsed PDF '%s': %d page(s).", file_path.name, len(pages))
    return pages
