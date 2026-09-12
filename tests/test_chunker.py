"""
test_chunker.py

Unit tests for app/chunking/chunker.py. Verifies chunk_size/overlap
adherence, page-boundary preservation, offset accuracy, and edge cases.
"""

from __future__ import annotations

import pytest

from app.chunking.chunker import _chunk_text, chunk_parsed_document
from app.chunking.exceptions import ChunkingError
from app.models.document import DocumentMetadata, DocumentType, PageText, ParsedDocument


class TestChunkTextLowLevel:
    def test_short_text_yields_single_chunk(self) -> None:
        text = "This is a short clause about payment terms."
        chunks = _chunk_text(text, chunk_size=800, chunk_overlap=100)
        assert len(chunks) == 1
        assert chunks[0][0] == text

    def test_empty_text_yields_no_chunks(self) -> None:
        assert _chunk_text("   ", chunk_size=800, chunk_overlap=100) == []
        assert _chunk_text("", chunk_size=800, chunk_overlap=100) == []

    def test_long_text_produces_multiple_chunks_within_size_bound(self) -> None:
        # Build text well beyond chunk_size so we get multiple chunks.
        word = "clause "
        text = word * 300  # ~2100 characters
        chunks = _chunk_text(text, chunk_size=800, chunk_overlap=100)
        assert len(chunks) > 1
        for chunk_text, start, end in chunks:
            # Small slack allowed since boundaries snap to whitespace, never mid-word.
            assert len(chunk_text) <= 800 + 10
            assert text[start:end] == chunk_text

    def test_overlap_produces_shared_content_between_consecutive_chunks(self) -> None:
        word = "termination "
        text = word * 200
        chunks = _chunk_text(text, chunk_size=800, chunk_overlap=100)
        assert len(chunks) >= 2
        first_text, _, first_end = chunks[0]
        second_text, second_start, _ = chunks[1]
        # The overlap region should mean chunk 2 starts before chunk 1 ends.
        assert second_start < first_end

    def test_offsets_are_accurate_against_source_text(self) -> None:
        text = "Alpha bravo charlie delta echo foxtrot golf hotel india juliet."
        chunks = _chunk_text(text, chunk_size=20, chunk_overlap=5)
        for chunk_text, start, end in chunks:
            assert text[start:end] == chunk_text

    def test_overlap_must_be_smaller_than_chunk_size(self) -> None:
        with pytest.raises(ChunkingError):
            _chunk_text("some text here", chunk_size=100, chunk_overlap=100)
        with pytest.raises(ChunkingError):
            _chunk_text("some text here", chunk_size=100, chunk_overlap=150)

    def test_no_mid_word_split_on_clean_boundary_text(self) -> None:
        # Construct text where a clean word boundary should exist near chunk_size.
        text = ("word " * 50).strip()  # 5 chars per "word ", clean spaces throughout
        chunks = _chunk_text(text, chunk_size=50, chunk_overlap=10)
        for chunk_text, _, _ in chunks:
            # No chunk should start or end with a partial "wor" / "ord" fragment;
            # every chunk should be composed of whole "word" tokens.
            tokens = chunk_text.split()
            assert all(t == "word" for t in tokens)


class TestChunkParsedDocument:
    @pytest.fixture()
    def two_page_document(self) -> ParsedDocument:
        metadata = DocumentMetadata(
            document_id="doc-abc",
            contract_name="Test Contract",
            document_type=DocumentType.PDF,
            page_count=2,
            file_size_bytes=1000,
            source_path="x.pdf",
        )
        long_text = "Payment terms shall apply. " * 100  # long enough to span multiple chunks
        pages = [
            PageText(page_number=1, text=long_text),
            PageText(page_number=2, text="Short page two content about termination."),
        ]
        return ParsedDocument(metadata=metadata, pages=pages)

    def test_chunks_never_span_page_boundaries(self, two_page_document: ParsedDocument) -> None:
        chunks = chunk_parsed_document(two_page_document)
        page_numbers = {c.page_number for c in chunks}
        assert page_numbers == {1, 2}

        # Every chunk's text must be a substring of its own page's text only.
        page_text_by_number = {p.page_number: p.text for p in two_page_document.pages}
        for chunk in chunks:
            assert chunk.text in page_text_by_number[chunk.page_number]

    def test_chunk_index_increments_sequentially_across_pages(self, two_page_document: ParsedDocument) -> None:
        chunks = chunk_parsed_document(two_page_document)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_metadata_matches_document(self, two_page_document: ParsedDocument) -> None:
        chunks = chunk_parsed_document(two_page_document)
        for chunk in chunks:
            assert chunk.document_id == "doc-abc"
            assert chunk.contract_name == "Test Contract"

    def test_blank_page_yields_no_chunks_for_that_page(self) -> None:
        metadata = DocumentMetadata(
            document_id="doc-blank",
            contract_name="Blank Page Contract",
            document_type=DocumentType.PDF,
            page_count=2,
            file_size_bytes=500,
            source_path="x.pdf",
        )
        pages = [
            PageText(page_number=1, text="Real content about confidentiality obligations."),
            PageText(page_number=2, text="   "),
        ]
        parsed = ParsedDocument(metadata=metadata, pages=pages)
        chunks = chunk_parsed_document(parsed)
        assert all(c.page_number == 1 for c in chunks)
