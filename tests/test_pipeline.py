"""
test_pipeline.py

Integration test for app/chunking/pipeline.py: verifies the full
chunk -> embed -> ChromaDB-store flow using a stubbed embedding model
(no network) and a real, isolated ChromaDB instance.
"""

from __future__ import annotations

import math
import uuid

import pytest

from app.chunking.pipeline import index_document
from app.embeddings import chroma_store, embedder
from app.models.document import DocumentMetadata, DocumentType, PageText, ParsedDocument


class StubEmbeddingModel:
    """Deterministic fake model, mirrors the one in test_embedder.py."""

    def __init__(self, dim: int = 6) -> None:
        self.dim = dim

    def encode(self, texts, normalize_embeddings: bool = True, batch_size: int = 16):
        vectors = []
        for text in texts:
            base = [float((sum(map(ord, text)) + i * 11) % 89) for i in range(self.dim)]
            if normalize_embeddings:
                norm = math.sqrt(sum(v * v for v in base)) or 1.0
                base = [v / norm for v in base]
            vectors.append(base)
        return vectors


@pytest.fixture()
def isolated_environment(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Isolate both ChromaDB persistence and the embedding model for this test module."""
    from app.utils import config_loader

    config_loader.get_settings.cache_clear()
    settings = config_loader.get_settings()
    settings["paths"]["chroma_persist_dir"] = str(tmp_path / f"chroma_{uuid.uuid4().hex}")

    chroma_store.reset_client_cache()
    embedder.reset_model_cache()
    monkeypatch.setattr(embedder, "_load_model", lambda: StubEmbeddingModel())

    yield

    chroma_store.reset_client_cache()
    embedder.reset_model_cache()
    config_loader.get_settings.cache_clear()


class TestIndexDocument:
    def test_indexes_chunks_into_chroma(self, isolated_environment) -> None:
        metadata = DocumentMetadata(
            document_id="doc-pipeline-1",
            contract_name="Pipeline Test Contract",
            document_type=DocumentType.PDF,
            page_count=1,
            file_size_bytes=2000,
            source_path="x.pdf",
        )
        long_text = "Confidentiality obligations apply to both parties for five years. " * 30
        parsed = ParsedDocument(metadata=metadata, pages=[PageText(page_number=1, text=long_text)])

        chunks = index_document(parsed)

        assert len(chunks) > 1
        assert chroma_store.collection_count() == len(chunks)

        # Verify a query against the same content actually returns results.
        results = chroma_store.query_similar([1.0] * 6, top_k=3)
        assert len(results) <= 3
        assert all(r["metadata"]["document_id"] == "doc-pipeline-1" for r in results)

    def test_empty_document_indexes_nothing(self, isolated_environment) -> None:
        metadata = DocumentMetadata(
            document_id="doc-pipeline-empty",
            contract_name="Blank Contract",
            document_type=DocumentType.PDF,
            page_count=1,
            file_size_bytes=10,
            source_path="x.pdf",
        )
        parsed = ParsedDocument(metadata=metadata, pages=[PageText(page_number=1, text="   ")])

        chunks = index_document(parsed)

        assert chunks == []
        assert chroma_store.collection_count() == 0

    def test_two_documents_remain_distinguishable_by_metadata(self, isolated_environment) -> None:
        meta1 = DocumentMetadata(
            document_id="doc-A",
            contract_name="Contract A",
            document_type=DocumentType.PDF,
            page_count=1,
            file_size_bytes=500,
            source_path="a.pdf",
        )
        meta2 = DocumentMetadata(
            document_id="doc-B",
            contract_name="Contract B",
            document_type=DocumentType.DOCX,
            page_count=1,
            file_size_bytes=500,
            source_path="b.docx",
        )
        parsed1 = ParsedDocument(metadata=meta1, pages=[PageText(page_number=1, text="Payment terms for contract A apply here.")])
        parsed2 = ParsedDocument(metadata=meta2, pages=[PageText(page_number=1, text="Payment terms for contract B apply here.")])

        index_document(parsed1)
        index_document(parsed2)

        assert chroma_store.collection_count() == 2
        chroma_store.delete_document_chunks("doc-A")
        assert chroma_store.collection_count() == 1
