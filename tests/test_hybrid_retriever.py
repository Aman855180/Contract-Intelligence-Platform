"""
test_hybrid_retriever.py

Integration tests for app/retriever/hybrid_retriever.py: real ChromaDB +
real BM25 index, with a stubbed (deterministic, no-network) embedding
model. Verifies the full embed -> vector search -> BM25 search -> RRF
fusion -> RetrievalResult pipeline, including top_k truncation and the
above_threshold flag used by Phase 5's Q&A module.
"""

from __future__ import annotations

import math
import uuid

import pytest

from app.chunking.pipeline import index_document
from app.embeddings import chroma_store, embedder
from app.models.document import DocumentMetadata, DocumentType, PageText, ParsedDocument
from app.retriever import bm25_index
from app.retriever.hybrid_retriever import hybrid_retrieve


def _stable_hash(token: str) -> int:
    """Deterministic token hash (unlike Python's randomized built-in hash())."""
    import hashlib

    return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)


class StubEmbeddingModel:
    """
    Deterministic fake model where similarity is driven by shared words,
    so that semantically related queries/chunks actually score higher,
    making above_threshold assertions meaningful.
    """

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def _vectorize(self, text: str) -> list[float]:
        tokens = text.lower().split()
        vec = [0.0] * self.dim
        for token in tokens:
            idx = _stable_hash(token) % self.dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def encode(self, texts, normalize_embeddings: bool = True, batch_size: int = 16):
        return [self._vectorize(t) for t in texts]


@pytest.fixture()
def isolated_retrieval_env(tmp_path, monkeypatch: pytest.MonkeyPatch):
    from app.utils import config_loader

    config_loader.get_settings.cache_clear()
    config_loader.get_retrieval_config.cache_clear()
    settings = config_loader.get_settings()
    settings["paths"]["chroma_persist_dir"] = str(tmp_path / f"chroma_{uuid.uuid4().hex}")

    chroma_store.reset_client_cache()
    bm25_index.reset_index_cache()
    embedder.reset_model_cache()
    monkeypatch.setattr(embedder, "_load_model", lambda: StubEmbeddingModel())

    yield

    chroma_store.reset_client_cache()
    bm25_index.reset_index_cache()
    embedder.reset_model_cache()
    config_loader.get_settings.cache_clear()
    config_loader.get_retrieval_config.cache_clear()


def _index_sample_contract(document_id: str = "doc-1") -> None:
    metadata = DocumentMetadata(
        document_id=document_id,
        contract_name="Sample Services Agreement",
        document_type=DocumentType.PDF,
        page_count=3,
        file_size_bytes=5000,
        source_path="x.pdf",
    )
    pages = [
        PageText(page_number=1, text="Payment shall be due within thirty days of invoice receipt by the client."),
        PageText(page_number=2, text="Either party may terminate this agreement with sixty days written notice."),
        PageText(page_number=3, text="Each party agrees to keep all confidential information strictly secret."),
    ]
    parsed = ParsedDocument(metadata=metadata, pages=pages)
    index_document(parsed)


class TestHybridRetrieve:
    def test_returns_relevant_chunk_for_payment_query(self, isolated_retrieval_env) -> None:
        _index_sample_contract()

        result = hybrid_retrieve("What is the payment term?")

        assert result.query == "What is the payment term?"
        assert len(result.chunks) > 0
        top_chunk = result.chunks[0]
        assert "payment" in top_chunk.text.lower() or "invoice" in top_chunk.text.lower()
        assert top_chunk.page_number == 1
        assert top_chunk.contract_name == "Sample Services Agreement"

    def test_above_threshold_true_for_well_matched_query(self, isolated_retrieval_env) -> None:
        _index_sample_contract()
        result = hybrid_retrieve("payment invoice thirty days client")
        assert result.above_threshold is True

    def test_respects_configured_top_k(self, isolated_retrieval_env) -> None:
        _index_sample_contract()
        from app.utils.config_loader import get_retrieval_config

        top_k = get_retrieval_config()["retrieval"]["top_k"]
        result = hybrid_retrieve("agreement party clause")
        assert len(result.chunks) <= top_k

    def test_empty_query_returns_empty_result(self, isolated_retrieval_env) -> None:
        _index_sample_contract()
        result = hybrid_retrieve("   ")
        assert result.chunks == []
        assert result.above_threshold is False

    def test_empty_corpus_returns_empty_result(self, isolated_retrieval_env) -> None:
        result = hybrid_retrieve("What is the payment term?")
        assert result.chunks == []
        assert result.above_threshold is False

    def test_fused_chunks_carry_both_score_types_when_double_matched(self, isolated_retrieval_env) -> None:
        _index_sample_contract()
        result = hybrid_retrieve("termination notice party agreement")
        # At least one returned chunk should have been found by both vector
        # search and BM25, given the strong lexical + semantic overlap.
        assert any(c.vector_score is not None and c.bm25_score is not None for c in result.chunks)

    def test_chunks_from_multiple_documents_are_distinguishable(self, isolated_retrieval_env) -> None:
        _index_sample_contract("doc-1")
        metadata2 = DocumentMetadata(
            document_id="doc-2",
            contract_name="Second Agreement",
            document_type=DocumentType.PDF,
            page_count=1,
            file_size_bytes=1000,
            source_path="y.pdf",
        )
        parsed2 = ParsedDocument(
            metadata=metadata2,
            pages=[PageText(page_number=1, text="Confidentiality survives termination for two years in this agreement.")],
        )
        index_document(parsed2)

        result = hybrid_retrieve("confidentiality survives termination")
        document_ids = {c.document_id for c in result.chunks}
        # Both documents discuss confidentiality/termination, so both may appear;
        # what matters is that document_id correctly distinguishes them.
        assert document_ids.issubset({"doc-1", "doc-2"})
        assert len(document_ids) >= 1
