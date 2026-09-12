"""
chunker.py

Intelligent overlapping text chunking. Chunks are produced independently
per page (never spanning a page boundary) so that every chunk carries an
unambiguous page_number for citation purposes downstream. chunk_size and
chunk_overlap are always read from config/retrieval.yaml — never hardcoded.

Chunk boundaries are snapped to the nearest whitespace where possible so
chunks don't split words mid-token, which both improves embedding quality
and keeps citation text spans readable.
"""

from __future__ import annotations

import uuid

from app.chunking.exceptions import ChunkingError
from app.models.document import DocumentChunk, ParsedDocument
from app.utils.config_loader import get_retrieval_config
from app.utils.logging_setup import get_logger

logger = get_logger(__name__)

# How far (in characters) we're willing to look for a whitespace boundary
# before falling back to a hard cut. Keeps chunk sizes close to the
# configured target while avoiding mid-word splits.
_BOUNDARY_SEARCH_WINDOW = 50


def _find_chunk_end(text: str, start: int, target_end: int) -> int:
    """
    Determine the actual end offset for a chunk starting at `start`, trying
    to land on a whitespace boundary near `target_end` rather than cutting
    a word in half.

    Args:
        text: Full text being chunked.
        start: Start offset of the current chunk.
        target_end: The ideal end offset (start + chunk_size), clipped to len(text).

    Returns:
        The chosen end offset, guaranteed to be > start.
    """
    text_len = len(text)
    if target_end >= text_len:
        return text_len

    # If the character right at target_end is already whitespace, that's a clean cut.
    if text[target_end].isspace():
        return target_end

    # Search backwards from target_end for the nearest whitespace.
    search_floor = max(start + 1, target_end - _BOUNDARY_SEARCH_WINDOW)
    for i in range(target_end, search_floor, -1):
        if text[i].isspace():
            return i

    # No good boundary found nearby; fall back to a hard cut at target_end.
    return target_end


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[tuple[str, int, int]]:
    """
    Split a single block of text into overlapping chunks.

    Args:
        text: The text to chunk (typically one page's text).
        chunk_size: Target chunk size in characters.
        chunk_overlap: Number of characters of overlap between consecutive chunks.

    Returns:
        List of (chunk_text, char_start, char_end) tuples in order.

    Raises:
        ChunkingError: If chunk_overlap >= chunk_size (would never make progress).
    """
    if chunk_overlap >= chunk_size:
        raise ChunkingError(
            f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})."
        )

    stripped = text.strip()
    if not stripped:
        return []

    # Re-locate the stripped text's offset within the original so char_start/char_end
    # remain accurate against the original page text.
    leading_ws = len(text) - len(text.lstrip())
    text_len = len(text)

    chunks: list[tuple[str, int, int]] = []
    start = leading_ws
    while start < text_len:
        target_end = min(start + chunk_size, text_len)
        end = _find_chunk_end(text, start, target_end)
        chunk_text = text[start:end].strip()

        if chunk_text:
            # Recompute exact start/end to match the stripped chunk_text within text[start:end].
            local_offset = text[start:end].find(chunk_text)
            actual_start = start + local_offset
            actual_end = actual_start + len(chunk_text)
            chunks.append((chunk_text, actual_start, actual_end))

        if end >= text_len:
            break

        next_start = end - chunk_overlap
        # Guarantee forward progress even in pathological inputs.
        start = next_start if next_start > start else start + 1

    return chunks


def chunk_parsed_document(parsed_document: ParsedDocument) -> list[DocumentChunk]:
    """
    Chunk every page of a ParsedDocument into overlapping DocumentChunk
    objects, using chunk_size/chunk_overlap from config/retrieval.yaml.
    Chunks never span page boundaries, preserving accurate page citations.

    Args:
        parsed_document: The parsed document (metadata + per-page text).

    Returns:
        List of DocumentChunk objects across all pages, in document order.
        Pages with no text (e.g. blank pages) simply contribute no chunks.
    """
    retrieval_cfg = get_retrieval_config()
    chunk_size = retrieval_cfg["chunking"]["chunk_size"]
    chunk_overlap = retrieval_cfg["chunking"]["chunk_overlap"]

    document_chunks: list[DocumentChunk] = []
    chunk_index = 0

    for page in parsed_document.pages:
        raw_chunks = _chunk_text(page.text, chunk_size, chunk_overlap)
        for chunk_text, char_start, char_end in raw_chunks:
            document_chunks.append(
                DocumentChunk(
                    chunk_id=uuid.uuid4().hex,
                    document_id=parsed_document.metadata.document_id,
                    contract_name=parsed_document.metadata.contract_name,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    char_start=char_start,
                    char_end=char_end,
                )
            )
            chunk_index += 1

    logger.info(
        "Chunked document '%s' (id=%s) into %d chunk(s) across %d page(s).",
        parsed_document.metadata.contract_name,
        parsed_document.metadata.document_id,
        len(document_chunks),
        len(parsed_document.pages),
    )
    return document_chunks
