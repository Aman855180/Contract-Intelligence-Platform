"""
test_bm25_index.py

Tests for app/retriever/bm25_index.py. Uses a real, isolated ChromaDB
instance (chunks + precomputed fake embeddings) as the corpus source, and
exercises the actual rank_bm25 library — no mocking of BM25 itself.
"""

from __future__ import annotations

import uuid

import pytest

from app.embeddings import chroma_store
from app.models.document import DocumentChunk
from app.retriever import bm25_index


def _make_chunk(document_id: str, text: str, page_number: int = 1, chunk_index: int = 0) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=uuid.uuid4().hex,
        document_id=document_id,
        contract_name=f"Contract {document_id}",
        page_number=page_number,
        chunk_index=chunk_index,
        text=text,
        char_start=0,
        char_end=len(text),
    )


@pytest.fixture()
def isolated_corpus(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Isolated ChromaDB + BM25 index cache for each test."""
    from app.utils import config_loader

    config_loader.get_settings.cache_clear()
    settings = config_loader.get_settings()
    settings["paths"]["chroma_persist_dir"] = str(tmp_path / f"chroma_{uuid.uuid4().hex}")

    chroma_store.reset_client_cache()
    bm25_index.reset_index_cache()
    yield
    chroma_store.reset_client_cache()
    bm25_index.reset_index_cache()
    config_loader.get_settings.cache_clear()


class TestBM25Search:
    def test_search_ranks_relevant_chunk_first(self, isolated_corpus) -> None:
        chunk_payment = _make_chunk("doc-1", "Payment shall be due within thirty days of invoice receipt.")
        chunk_termination = _make_chunk(
            "doc-1", "Either party may terminate this agreement with sixty days written notice.", page_number=2, chunk_index=1
        )
        chunk_confidentiality = _make_chunk(
            "doc-1", "Each party agrees to keep confidential information secret.", page_number=3, chunk_index=2
        )
        chroma_store.add_chunks(
            [chunk_payment, chunk_termination, chunk_confidentiality],
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
        )

        results = bm25_index.search_bm25("payment invoice days", top_k=3)

        assert len(results) >= 1
        assert results[0]["chunk_id"] == chunk_payment.chunk_id
        assert results[0]["bm25_score"] > 0

    def test_search_excludes_zero_score_chunks(self, isolated_corpus) -> None:
        chunk_a = _make_chunk("doc-1", "Confidentiality obligations survive termination.")
        chunk_b = _make_chunk("doc-1", "Liability is capped at total fees paid.", chunk_index=1)
        chunk_c = _make_chunk("doc-1", "Renewal extends the agreement automatically each year.", chunk_index=2)
        chroma_store.add_chunks(
            [chunk_a, chunk_b, chunk_c], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )

        # Query has zero token overlap with chunk_b's or chunk_c's text.
        results = bm25_index.search_bm25("confidentiality obligations survive", top_k=5)
        chunk_ids = {r["chunk_id"] for r in results}
        assert chunk_a.chunk_id in chunk_ids
        assert chunk_b.chunk_id not in chunk_ids
        assert chunk_c.chunk_id not in chunk_ids

    def test_search_respects_top_k(self, isolated_corpus) -> None:
        chunks = [_make_chunk("doc-1", f"Clause about payment number {i}.", chunk_index=i) for i in range(10)]
        embeddings = [[float(i), 0.0] for i in range(10)]
        chroma_store.add_chunks(chunks, embeddings)

        results = bm25_index.search_bm25("payment clause", top_k=4)
        assert len(results) <= 4

    def test_empty_corpus_returns_empty_list(self, isolated_corpus) -> None:
        results = bm25_index.search_bm25("any query", top_k=5)
        assert results == []

    def test_empty_query_returns_empty_list(self, isolated_corpus) -> None:
        chunk = _make_chunk("doc-1", "Some clause text about payment.")
        chroma_store.add_chunks([chunk], [[1.0, 0.0]])
        results = bm25_index.search_bm25("   ", top_k=5)
        assert results == []

    def test_index_rebuilds_when_new_chunks_added(self, isolated_corpus) -> None:
        chunk_a = _make_chunk("doc-1", "Original clause about payment terms and invoicing.")
        decoys = [
            _make_chunk("doc-1", "Termination requires sixty days written notice.", chunk_index=1),
            _make_chunk("doc-1", "Confidentiality survives for five years after expiry.", chunk_index=2),
            _make_chunk("doc-1", "Liability is limited to direct damages only.", chunk_index=3),
            _make_chunk("doc-1", "Renewal extends the agreement automatically each year.", chunk_index=4),
        ]
        chroma_store.add_chunks(
            [chunk_a, *decoys],
            [[1.0, 0.0, 0.0, 0.0, 0.0]] + [[0.0] * i + [1.0] + [0.0] * (4 - i) for i in range(1, 5)],
        )

        first_results = bm25_index.search_bm25("payment invoicing", top_k=5)
        assert len(first_results) == 1
        assert first_results[0]["chunk_id"] == chunk_a.chunk_id

        chunk_b = _make_chunk("doc-1", "New clause about payment scheduling and invoicing windows.", chunk_index=5)
        chroma_store.add_chunks([chunk_b], [[0.5, 0.5, 0.0, 0.0, 0.0]])

        second_results = bm25_index.search_bm25("payment invoicing", top_k=5)
        second_ids = {r["chunk_id"] for r in second_results}
        assert second_ids == {chunk_a.chunk_id, chunk_b.chunk_id}

    def test_result_metadata_matches_stored_chunk(self, isolated_corpus) -> None:
        chunk = _make_chunk("doc-7", "Renewal terms automatically extend the agreement.", page_number=4)
        decoy1 = _make_chunk("doc-7", "Payment is due within thirty days of invoice.", page_number=1, chunk_index=1)
        decoy2 = _make_chunk("doc-7", "Liability is limited to direct damages only.", page_number=2, chunk_index=2)
        chroma_store.add_chunks(
            [chunk, decoy1, decoy2], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )

        results = bm25_index.search_bm25("renewal terms extend", top_k=1)
        assert results[0]["metadata"]["document_id"] == "doc-7"
        assert results[0]["metadata"]["page_number"] == 4
        assert results[0]["text"] == chunk.text
