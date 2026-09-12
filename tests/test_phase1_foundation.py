"""
test_phase1_foundation.py

Smoke tests for Phase 1: config loading, logging setup, and Pydantic schema
construction/validation. These do not test business logic (none exists yet)
but verify the foundation every later phase depends on is sound.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from app.models.comparison import (
    ContractComparisonResult,
    FieldInconsistency,
    MismatchDetectionMethod,
)
from app.models.document import (
    DocumentChunk,
    DocumentMetadata,
    DocumentType,
    PageText,
    ParsedDocument,
)
from app.models.extraction import ContractExtractionResult, ExtractedField, FieldType
from app.models.report import (
    KeyParty,
    PaymentTermsSummary,
    ReportSchema,
)
from app.models.retrieval import GroundednessVerdict, QAAnswer, RetrievalResult, RetrievedChunk
from app.models.risk import RiskAssessment, RiskCategory, RiskFlag, RiskFlagSource
from app.utils.config_loader import get_retrieval_config, get_settings
from app.utils.logging_setup import get_logger


class TestConfigLoader:
    def test_settings_loads_and_has_required_sections(self) -> None:
        settings = get_settings()
        for key in ("app", "paths", "logging", "llm", "embeddings", "groundedness", "ner", "risk_scoring"):
            assert key in settings

    def test_retrieval_config_has_fixed_hyperparameters(self) -> None:
        retrieval_cfg = get_retrieval_config()
        assert retrieval_cfg["chunking"]["chunk_size"] == 800
        assert retrieval_cfg["chunking"]["chunk_overlap"] == 100
        assert retrieval_cfg["retrieval"]["top_k"] == 5

    def test_settings_is_cached_singleton(self) -> None:
        assert get_settings() is get_settings()


class TestLoggingSetup:
    def test_get_logger_returns_namespaced_logger(self) -> None:
        logger = get_logger(__name__)
        assert logger.name.startswith("contract_platform.")

    def test_logger_can_log_without_raising(self) -> None:
        logger = get_logger(__name__)
        logger.info("Phase 1 smoke test log message.")


class TestDocumentSchemas:
    def test_document_metadata_valid(self) -> None:
        meta = DocumentMetadata(
            document_id="doc-1",
            contract_name="Sample Contract A",
            document_type=DocumentType.PDF,
            page_count=3,
            file_size_bytes=1024,
            source_path="data/raw_documents/sample_a.pdf",
        )
        assert meta.contract_name == "Sample Contract A"

    def test_document_metadata_rejects_empty_name(self) -> None:
        with pytest.raises(ValidationError):
            DocumentMetadata(
                document_id="doc-1",
                contract_name="   ",
                document_type=DocumentType.PDF,
                page_count=1,
                file_size_bytes=100,
                source_path="x.pdf",
            )

    def test_parsed_document_requires_at_least_one_page(self) -> None:
        meta = DocumentMetadata(
            document_id="doc-2",
            contract_name="Sample Contract B",
            document_type=DocumentType.DOCX,
            page_count=1,
            file_size_bytes=500,
            source_path="data/raw_documents/sample_b.docx",
        )
        with pytest.raises(ValidationError):
            ParsedDocument(metadata=meta, pages=[])

        parsed = ParsedDocument(metadata=meta, pages=[PageText(page_number=1, text="Hello world")])
        assert len(parsed.pages) == 1

    def test_document_chunk_char_end_validation(self) -> None:
        with pytest.raises(ValidationError):
            DocumentChunk(
                chunk_id="c1",
                document_id="doc-1",
                contract_name="Sample Contract A",
                page_number=1,
                chunk_index=0,
                text="some text",
                char_start=100,
                char_end=50,
            )


class TestRetrievalSchemas:
    def test_retrieved_chunk_and_result(self) -> None:
        chunk = RetrievedChunk(
            chunk_id="c1",
            document_id="doc-1",
            contract_name="Sample Contract A",
            page_number=2,
            text="Payment shall be due within 30 days.",
            vector_score=0.82,
            bm25_score=5.1,
            fused_score=0.91,
        )
        result = RetrievalResult(query="What is the payment term?", chunks=[chunk], above_threshold=True)
        assert result.above_threshold is True
        assert result.chunks[0].page_number == 2

    def test_qa_answer_not_found_case(self) -> None:
        answer = QAAnswer(
            question="What is the governing law?",
            answer="NOT_FOUND_IN_CONTRACT",
            found_in_contract=False,
        )
        assert answer.groundedness is None

    def test_groundedness_verdict_bounds(self) -> None:
        with pytest.raises(ValidationError):
            GroundednessVerdict(
                claim="x",
                source_passage="y",
                entailment_score=1.5,  # out of bounds
                contradiction_score=0.1,
                neutral_score=0.1,
                is_grounded=True,
            )


class TestExtractionSchemas:
    def test_extraction_result_filtering(self) -> None:
        field1 = ExtractedField(
            field_type=FieldType.PAYMENT_TERMS,
            value_text="Net 30 days",
            document_id="doc-1",
            contract_name="Sample Contract A",
            page_number=2,
            confidence=0.95,
        )
        field2 = ExtractedField(
            field_type=FieldType.TERMINATION_CLAUSE,
            value_text="Either party may terminate with 60 days notice.",
            document_id="doc-1",
            contract_name="Sample Contract A",
            page_number=4,
            confidence=0.88,
        )
        result = ContractExtractionResult(document_id="doc-1", contract_name="Sample Contract A", fields=[field1, field2])
        payment_fields = result.get_fields_by_type(FieldType.PAYMENT_TERMS)
        assert len(payment_fields) == 1
        assert payment_fields[0].value_text == "Net 30 days"


class TestComparisonSchemas:
    def test_comparison_result_construction(self) -> None:
        inconsistency = FieldInconsistency(
            field_type=FieldType.PAYMENT_TERMS,
            detection_method=MismatchDetectionMethod.RULE_BASED_NUMERIC_DATE,
            contract_a_name="Contract A",
            contract_a_page=2,
            contract_a_text="Net 30 days",
            contract_b_name="Contract B",
            contract_b_page=3,
            contract_b_text="Net 60 days",
            explanation="Payment term day count differs: 30 vs 60.",
            confidence=1.0,
        )
        result = ContractComparisonResult(
            document_id_a="doc-1",
            document_id_b="doc-2",
            contract_a_name="Contract A",
            contract_b_name="Contract B",
            inconsistencies=[inconsistency],
            fields_compared=7,
        )
        assert result.inconsistencies[0].detection_method == MismatchDetectionMethod.RULE_BASED_NUMERIC_DATE


class TestRiskSchemas:
    def test_risk_assessment_triggered_flags(self) -> None:
        flag1 = RiskFlag(
            category=RiskCategory.MISSING_SIGNATURE_FIELD,
            source=RiskFlagSource.RULE_BASED,
            triggered=True,
            weight=25,
            explanation="No signature block detected in document.",
        )
        flag2 = RiskFlag(
            category=RiskCategory.AMBIGUOUS_PAYMENT_TERMS,
            source=RiskFlagSource.LLM_CLASSIFIED,
            triggered=False,
            weight=10,
            explanation="Payment terms are clearly specified.",
            source_contract="Contract A",
            source_page=2,
            supporting_text="Net 30 days from invoice date.",
        )
        assessment = RiskAssessment(
            document_id="doc-1", contract_name="Contract A", flags=[flag1, flag2], risk_score=25
        )
        triggered = assessment.triggered_flags()
        assert len(triggered) == 1
        assert triggered[0].category == RiskCategory.MISSING_SIGNATURE_FIELD

    def test_risk_score_out_of_bounds_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RiskAssessment(document_id="doc-1", contract_name="Contract A", flags=[], risk_score=150)


class TestReportSchema:
    def test_report_schema_valid_minimal(self) -> None:
        report = ReportSchema(
            document_id="doc-1",
            contract_name="Contract A",
            overview="This is a services agreement between two parties.",
            key_parties=[KeyParty(name="Acme Corp", role="Client", source_contract="Contract A", source_page=1)],
            payment_terms=PaymentTermsSummary(
                summary="Net 30 days from invoice.", source_contract="Contract A", source_page=2
            ),
        )
        assert report.key_parties[0].name == "Acme Corp"

    def test_report_schema_rejects_invalid_severity(self) -> None:
        with pytest.raises(ValidationError):
            ReportSchema(
                document_id="doc-1",
                contract_name="Contract A",
                overview="Overview text.",
                payment_terms=PaymentTermsSummary(summary="x", source_contract="Contract A", source_page=1),
                risk_flags=[
                    {
                        "category": "high_liability_exposure",
                        "severity": "extreme",  # invalid, must be low|medium|high
                        "explanation": "x",
                        "source_contract": "Contract A",
                        "source_page": 1,
                    }
                ],
            )
