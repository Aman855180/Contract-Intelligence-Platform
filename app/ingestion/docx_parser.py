"""
docx_parser.py

DOCX text extraction using python-docx. Word documents have no inherent
pagination at the XML level (page layout is computed by the rendering
engine, not stored), so true page numbers cannot be recovered without a
layout engine. As a deliberate, documented simplification for this
project's scope:

  - We detect explicit hard page breaks (<w:br w:type="page"/>) inserted by
    the document author and use them to increment a running page counter.
  - Paragraphs before the first page break are page 1; each subsequent hard
    page break starts a new page.
  - Documents with no hard page breaks are treated as a single page
    (page_number=1), which is correct for most short contract templates.

This is noted explicitly in docs/DESIGN_DECISIONS.md (Phase 12) as a known
limitation versus true PDF pagination.
"""

from __future__ import annotations

from pathlib import Path

import docx
from docx.oxml.ns import qn

from app.ingestion.exceptions import DocumentParsingError
from app.models.document import PageText
from app.utils.logging_setup import get_logger

logger = get_logger(__name__)


def _paragraph_has_page_break(paragraph: "docx.text.paragraph.Paragraph") -> bool:
    """
    Check whether a paragraph contains an explicit hard page break run.

    Args:
        paragraph: A python-docx Paragraph object.

    Returns:
        True if any run in the paragraph contains a <w:br w:type="page"/>.
    """
    for run in paragraph.runs:
        for br in run._element.findall(qn("w:br")):
            if br.get(qn("w:type")) == "page":
                return True
    return False


def parse_docx(file_path: Path) -> list[PageText]:
    """
    Extract text from a DOCX file, grouped into pseudo-pages based on
    explicit hard page breaks (see module docstring for the documented
    limitation).

    Args:
        file_path: Path to the DOCX file on disk.

    Returns:
        List of PageText objects with 1-indexed page_number.

    Raises:
        DocumentParsingError: If python-docx fails to open or read the file.
    """
    try:
        document = docx.Document(str(file_path))
    except Exception as exc:
        logger.error("Failed to open DOCX file %s: %s", file_path.name, exc)
        raise DocumentParsingError(f"Could not open DOCX file '{file_path.name}': {exc}") from exc

    pages: list[PageText] = []
    current_page_number = 1
    current_page_lines: list[str] = []

    try:
        # Also include text inside tables, in document order is not trivial with
        # python-docx's flat body iteration, so we walk body children directly.
        for element in document.element.body.iterchildren():
            if element.tag == qn("w:p"):
                paragraph = docx.text.paragraph.Paragraph(element, document)
                if paragraph.text.strip():
                    current_page_lines.append(paragraph.text)
                if _paragraph_has_page_break(paragraph):
                    pages.append(
                        PageText(page_number=current_page_number, text="\n".join(current_page_lines).strip())
                    )
                    current_page_number += 1
                    current_page_lines = []
            elif element.tag == qn("w:tbl"):
                table = docx.table.Table(element, document)
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                    if row_text:
                        current_page_lines.append(row_text)

        # Flush the final (or only) page.
        pages.append(PageText(page_number=current_page_number, text="\n".join(current_page_lines).strip()))

    except Exception as exc:
        logger.error("Failed while extracting text from DOCX %s: %s", file_path.name, exc)
        raise DocumentParsingError(f"Failed extracting text from '{file_path.name}': {exc}") from exc

    logger.info(
        "Parsed DOCX '%s': %d pseudo-page(s) (based on hard page breaks).",
        file_path.name,
        len(pages),
    )
    return pages
