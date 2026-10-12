"""
test_chroma_store.py

Tests for app/embeddings/chroma_store.py against a REAL ChromaDB
PersistentClient pointed at a temporary directory. Embeddings are
precomputed simple vectors (no real embedding model needed, since
chroma_store never generates embeddings itself), so these tests exercise
actual ChromaDB read/write/query behavior rather than mocks.
"""

from __future__ import annotations

import uuid

import pytest

from app.embeddings import chroma_store
from app.embeddings.exceptions import EmbeddingError
from app.models.document import DocumentChunk


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
def isolated_chroma(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Point ChromaDB at a fresh temp directory and reset the cached client for each test."""
    from app.utils import config_loader

    config_loader.get_settings.cache_clear()
    settings = config_loader.get_settings()
    settings["paths"]["chroma_persist_dir"] = str(tmp_path / f"chroma_{uuid.uuid4().hex}")

    chroma_store.reset_client_cache()
    yield
    chroma_store.reset_client_cache()
    config_loader.get_settings.cache_clear()


class TestAddAndQueryChunks:
    def test_add_chunks_then_query_returns_most_similar(self, isolated_chroma) -> None:
        chunk_a = _make_chunk("doc-1", "Payment shall be due within 30 days of invoice.")
        chunk_b = _make_chunk("doc-1", "Either party may terminate with 60 days notice.", page_number=2, chunk_index=1)

        # Embedding A is close to the query; embedding B is far from it.
        embedding_a = [1.0, 0.0, 0.0]
        embedding_b = [0.0, 1.0, 0.0]
        query_embedding = [0.9, 0.1, 0.0]

        chroma_store.add_chunks([chunk_a, chunk_b], [embedding_a, embedding_b])
        results = chroma_store.query_similar(query_embedding, top_k=2)

        assert len(results) == 2
        # The closer embedding (chunk_a) should rank first.
        assert results[0]["chunk_id"] == chunk_a.chunk_id
        assert results[0]["metadata"]["page_number"] == 1
        assert results[0]["metadata"]["contract_name"] == chunk_a.contract_name

    def test_query_respects_top_k(self, isolated_chroma) -> None:
        chunks = [_make_chunk("doc-1", f"Clause number {i}.", chunk_index=i) for i in range(5)]
        embeddings = [[float(i), 0.0, 0.0] for i in range(5)]
        chroma_store.add_chunks(chunks, embeddings)

        results = chroma_store.query_similar([0.0, 0.0, 0.0], top_k=3)
        assert len(results) == 3

    def test_query_similar_batch_multiple_queries(self, isolated_chroma) -> None:
        chunk_a = _make_chunk("doc-1", "Payment clause.", chunk_index=0)
        chunk_b = _make_chunk("doc-1", "Termination clause.", chunk_index=1)
        embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        chroma_store.add_chunks([chunk_a, chunk_b], embeddings)

        query_vecs = [[0.9, 0.1, 0.0], [0.1, 0.9, 0.0]]
        batch_results = chroma_store.query_similar_batch(query_vecs, top_k=1)
        assert len(batch_results) == 2
        assert len(batch_results[0]) == 1
        assert len(batch_results[1]) == 1
        assert batch_results[0][0]["chunk_id"] == chunk_a.chunk_id
        assert batch_results[1][0]["chunk_id"] == chunk_b.chunk_id

    def test_query_similar_batch_empty_input(self, isolated_chroma) -> None:
        assert chroma_store.query_similar_batch([], top_k=3) == []

    def test_query_on_empty_collection_returns_empty_list(self, isolated_chroma) -> None:
        results = chroma_store.query_similar([1.0, 0.0, 0.0], top_k=5)
        assert results == []

    def test_mismatched_lengths_raises(self, isolated_chroma) -> None:
        chunk = _make_chunk("doc-1", "Some clause text.")
        with pytest.raises(EmbeddingError):
            chroma_store.add_chunks([chunk], [[1.0, 0.0], [0.0, 1.0]])

    def test_add_empty_chunk_list_is_noop(self, isolated_chroma) -> None:
        chroma_store.add_chunks([], [])
        assert chroma_store.collection_count() == 0


class TestDeleteDocumentChunks:
    def test_delete_removes_only_target_document_chunks(self, isolated_chroma) -> None:
        chunk_doc1 = _make_chunk("doc-1", "Doc 1 clause text.")
        chunk_doc2 = _make_chunk("doc-2", "Doc 2 clause text.")
        chroma_store.add_chunks([chunk_doc1, chunk_doc2], [[1.0, 0.0], [0.0, 1.0]])

        assert chroma_store.collection_count() == 2

        chroma_store.delete_document_chunks("doc-1")

        assert chroma_store.collection_count() == 1
        remaining = chroma_store.query_similar([0.0, 1.0], top_k=5)
        assert remaining[0]["metadata"]["document_id"] == "doc-2"


class TestCollectionPersistence:
    def test_data_persists_across_client_resets(self, isolated_chroma) -> None:
        chunk = _make_chunk("doc-persist", "Persisted clause text.")
        chroma_store.add_chunks([chunk], [[0.5, 0.5, 0.0]])

        # Simulate a fresh process by resetting the cached client (same persist dir).
        chroma_store.reset_client_cache()

        assert chroma_store.collection_count() == 1


class TestGetChunksByDocument:
    def test_returns_only_chunks_for_target_document_in_chunk_order(self, isolated_chroma) -> None:
        doc1_chunks = [
            _make_chunk("doc-1", "First chunk of doc 1.", chunk_index=0),
            _make_chunk("doc-1", "Second chunk of doc 1.", chunk_index=1),
        ]
        doc2_chunk = _make_chunk("doc-2", "Only chunk of doc 2.", chunk_index=0)
        # Insert out of order to verify sorting by chunk_index works.
        chroma_store.add_chunks(
            [doc1_chunks[1], doc2_chunk, doc1_chunks[0]],
            [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]],
        )

        results = chroma_store.get_chunks_by_document("doc-1")

        assert len(results) == 2
        assert results[0]["text"] == "First chunk of doc 1."
        assert results[1]["text"] == "Second chunk of doc 1."
        assert all(r["metadata"]["document_id"] == "doc-1" for r in results)

    def test_returns_empty_list_for_unknown_document(self, isolated_chroma) -> None:
        chunk = _make_chunk("doc-1", "Some text.")
        chroma_store.add_chunks([chunk], [[1.0, 0.0]])

        assert chroma_store.get_chunks_by_document("doc-unknown") == []

    def test_returns_empty_list_for_empty_collection(self, isolated_chroma) -> None:
        assert chroma_store.get_chunks_by_document("doc-1") == []
