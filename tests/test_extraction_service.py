"""
test_extraction_service.py

Integration tests for app/extraction/extraction_service.py: real SQLite
(Phase 2) + real ChromaDB indexing (Phase 3, stubbed embedder) feeding the
real keyword classifier and date extractor, with a stubbed NER pipeline
for the obligation extractor (no network).
"""

from __future__ import annotations

import math
import uuid

import pytest

from app.chunking.pipeline import index_document
from app.database import repository
from app.database.sqlite_db import init_db
from app.embeddings import chroma_store, embedder
from app.extraction import extraction_service, ner_pipeline
from app.extraction.exceptions import ExtractionError
from app.models.document import DocumentMetadata, DocumentType, PageText, ParsedDocument
from app.models.extraction import FieldType
from app.retriever import bm25_index


class StubEmbeddingModel:
    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    def encode(self, texts, normalize_embeddings: bool = True, batch_size: int = 16):
        vectors = []
        for text in texts:
            base = [float((sum(map(ord, text)) + i * 13) % 71) for i in range(self.dim)]
            if normalize_embeddings:
                norm = math.sqrt(sum(v * v for v in base)) or 1.0
                base = [v / norm for v in base]
            vectors.append(base)
        return vectors


class StubNERPipeline:
    """Returns an ORG entity whenever 'Acme Corp' appears in the sentence."""

    def __call__(self, text: str) -> list[dict]:
        if "Acme Corp" in text:
            return [{"word": "Acme Corp", "entity_group": "ORG", "score": 0.9}]
        return []


@pytest.fixture()
def isolated_extraction_env(tmp_path, monkeypatch: pytest.MonkeyPatch):
    from app.utils import config_loader

    config_loader.get_settings.cache_clear()
    config_loader.get_retrieval_config.cache_clear()
    config_loader.get_extraction_keywords.cache_clear()
    settings = config_loader.get_settings()
    settings["paths"]["chroma_persist_dir"] = str(tmp_path / f"chroma_{uuid.uuid4().hex}")
    settings["paths"]["sqlite_db_path"] = f"data/test_extraction_{uuid.uuid4().hex}.db"

    chroma_store.reset_client_cache()
    bm25_index.reset_index_cache()
    embedder.reset_model_cache()
    ner_pipeline.reset_pipeline_cache()
    monkeypatch.setattr(embedder, "_load_model", lambda: StubEmbeddingModel())
    monkeypatch.setattr(ner_pipeline, "_load_ner_pipeline", lambda: StubNERPipeline())

    init_db()

    yield

    chroma_store.reset_client_cache()
    bm25_index.reset_index_cache()
    embedder.reset_model_cache()
    ner_pipeline.reset_pipeline_cache()
    config_loader.get_settings.cache_clear()
    config_loader.get_retrieval_config.cache_clear()
    config_loader.get_extraction_keywords.cache_clear()


def _ingest_and_index_sample_contract(document_id: str = "doc-1") -> None:
    metadata = DocumentMetadata(
        document_id=document_id,
        contract_name="Sample Services Agreement",
        document_type=DocumentType.PDF,
        page_count=3,
        file_size_bytes=5000,
        source_path="x.pdf",
    )
    pages = [
        PageText(
            page_number=1,
            text="Payment shall be due within thirty days of invoice receipt. "
            "Acme Corp shall pay all undisputed invoices in full.",
        ),
        PageText(
            page_number=2,
            text="Either party may terminate this agreement with sixty days written notice. "
            "This agreement is effective as of January 1, 2025.",
        ),
        PageText(
            page_number=3,
            text="IN WITNESS WHEREOF, the parties have executed this agreement as of the date "
            "first written above.",
        ),
    ]
    parsed = ParsedDocument(metadata=metadata, pages=pages)
    repository.insert_parsed_document(parsed)
    index_document(parsed)


class TestExtractFieldsFromDocument:
    def test_extracts_multiple_field_types_across_pages(self, isolated_extraction_env) -> None:
        _ingest_and_index_sample_contract()

        result = extraction_service.extract_fields_from_document("doc-1")

        assert result.document_id == "doc-1"
        assert result.contract_name == "Sample Services Agreement"

        field_types_found = {f.field_type for f in result.fields}
        assert FieldType.PAYMENT_TERMS in field_types_found
        assert FieldType.TERMINATION_CLAUSE in field_types_found
        assert FieldType.SIGNATURE in field_types_found
        assert FieldType.OBLIGATIONS in field_types_found
        assert FieldType.KEY_DATES in field_types_found

    def test_obligation_field_enriched_with_ner_party(self, isolated_extraction_env) -> None:
        _ingest_and_index_sample_contract()

        result = extraction_service.extract_fields_from_document("doc-1")
        obligation_fields = result.get_fields_by_type(FieldType.OBLIGATIONS)

        acme_obligation = next((f for f in obligation_fields if "Acme Corp" in f.value_text), None)
        assert acme_obligation is not None
        assert acme_obligation.confidence == pytest.approx(0.9)  # NER entity score, not pattern-only fallback

    def test_key_dates_field_captures_extracted_date(self, isolated_extraction_env) -> None:
        _ingest_and_index_sample_contract()

        result = extraction_service.extract_fields_from_document("doc-1")
        date_fields = result.get_fields_by_type(FieldType.KEY_DATES)

        assert any("January 1, 2025" in f.value_text for f in date_fields)
        assert all(f.page_number == 2 for f in date_fields if "January 1, 2025" in f.value_text)

    def test_all_fields_carry_correct_citation_metadata(self, isolated_extraction_env) -> None:
        _ingest_and_index_sample_contract()

        result = extraction_service.extract_fields_from_document("doc-1")
        for field in result.fields:
            assert field.document_id == "doc-1"
            assert field.contract_name == "Sample Services Agreement"
            assert 1 <= field.page_number <= 3
            assert 0.0 < field.confidence <= 1.0

    def test_unindexed_document_raises_extraction_error(self, isolated_extraction_env) -> None:
        with pytest.raises(ExtractionError):
            extraction_service.extract_fields_from_document("doc-does-not-exist")

    def test_falls_back_to_chunk_metadata_when_not_in_sqlite(self, isolated_extraction_env) -> None:
        # Index directly via ChromaDB without inserting into SQLite, to exercise the fallback path.
        metadata = DocumentMetadata(
            document_id="doc-orphan",
            contract_name="Orphan Contract",
            document_type=DocumentType.PDF,
            page_count=1,
            file_size_bytes=500,
            source_path="x.pdf",
        )
        parsed = ParsedDocument(
            metadata=metadata,
            pages=[PageText(page_number=1, text="Payment shall be due within thirty days.")],
        )
        index_document(parsed)  # no repository.insert_parsed_document call

        result = extraction_service.extract_fields_from_document("doc-orphan")
        assert result.contract_name == "Orphan Contract"
