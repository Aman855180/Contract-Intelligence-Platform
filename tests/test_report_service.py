"""
test_report_service.py

Integration tests for app/reports/report_service.py. Stubs LLM generation,
retrieval, and risk/extraction auto-calls to verify the orchestration
contract: correct pre-computed result reuse, correct auto-computation when
not provided, and correct ReportResult shape.
"""

from __future__ import annotations

import json

import pytest

from app.models.extraction import ContractExtractionResult, ExtractedField, FieldType
from app.models.risk import RiskAssessment, RiskCategory, RiskFlag, RiskFlagSource
from app.reports import report_generator, report_service


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_extraction(document_id: str = "doc-1") -> ContractExtractionResult:
    return ContractExtractionResult(
        document_id=document_id,
        contract_name="Service Agreement",
        fields=[
            ExtractedField(field_type=FieldType.PAYMENT_TERMS, value_text="Net 30.",
                           document_id=document_id, contract_name="Service Agreement",
                           page_number=2, confidence=0.9),
        ],
    )


def _make_assessment(document_id: str = "doc-1") -> RiskAssessment:
    return RiskAssessment(
        document_id=document_id,
        contract_name="Service Agreement",
        flags=[
            RiskFlag(category=RiskCategory.MISSING_SIGNATURE_FIELD,
                     source=RiskFlagSource.RULE_BASED, triggered=True,
                     weight=25, explanation="No signature detected."),
        ],
        risk_score=25,
    )


def _valid_llm_json() -> str:
    return json.dumps({
        "overview": "A services agreement between two parties.",
        "key_parties": [
            {"name": "Acme", "role": "Client",
             "source_contract": "Service Agreement", "source_page": 1}
        ],
        "important_dates": [],
        "payment_terms": {"summary": "Net 30.", "source_contract": "Service Agreement", "source_page": 2},
        "risk_flags": [
            {"category": "missing_signature_field", "severity": "high",
             "explanation": "No signature block.", "source_contract": "Service Agreement", "source_page": 1}
        ],
        "key_clauses": [],
        "recommendations": ["Add signatures."],
    })


@pytest.fixture(autouse=True)
def stub_retrieval_and_llm(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(report_generator, "embed_query", lambda q: [0.1] * 8)
    monkeypatch.setattr(report_generator, "query_similar", lambda vec, top_k, document_id: [])
    monkeypatch.setattr(report_generator, "generate", lambda p: _valid_llm_json())


class TestGenerateContractReport:
    def test_returns_report_result_with_both_outputs(self) -> None:
        result = report_service.generate_contract_report(
            "doc-1",
            extraction=_make_extraction(),
            assessment=_make_assessment(),
        )
        assert result.report is not None
        assert isinstance(result.markdown, str)
        assert len(result.markdown) > 50

    def test_markdown_contains_contract_name(self) -> None:
        result = report_service.generate_contract_report(
            "doc-1",
            extraction=_make_extraction(),
            assessment=_make_assessment(),
        )
        assert "Service Agreement" in result.markdown

    def test_markdown_contains_all_section_headers(self) -> None:
        result = report_service.generate_contract_report(
            "doc-1",
            extraction=_make_extraction(),
            assessment=_make_assessment(),
        )
        for header in ["## Overview", "## Key Parties", "## Payment Terms",
                       "## Risk Flags", "## Recommendations"]:
            assert header in result.markdown

    def test_report_schema_document_id_matches(self) -> None:
        result = report_service.generate_contract_report(
            "doc-1",
            extraction=_make_extraction(),
            assessment=_make_assessment(),
        )
        assert result.report.document_id == "doc-1"

    def test_auto_extraction_called_when_not_provided(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called = {"flag": False}

        def fake_extract(document_id: str):
            called["flag"] = True
            return _make_extraction(document_id)

        monkeypatch.setattr(report_service, "extract_fields_from_document", fake_extract)
        monkeypatch.setattr(report_service, "assess_risk", lambda doc_id, extraction: _make_assessment(doc_id))

        report_service.generate_contract_report("doc-1")
        assert called["flag"] is True

    def test_auto_assessment_called_when_not_provided(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called = {"flag": False}

        def fake_assess(document_id: str, extraction=None):
            called["flag"] = True
            return _make_assessment(document_id)

        monkeypatch.setattr(report_service, "assess_risk", fake_assess)

        report_service.generate_contract_report(
            "doc-1",
            extraction=_make_extraction(),
            assessment=None,  # trigger auto-assessment
        )
        assert called["flag"] is True

    def test_pre_computed_extraction_not_re_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called = {"flag": False}

        def fail_if_called(document_id: str):
            called["flag"] = True
            raise AssertionError("Should not re-extract when extraction is pre-computed.")

        monkeypatch.setattr(report_service, "extract_fields_from_document", fail_if_called)

        report_service.generate_contract_report(
            "doc-1",
            extraction=_make_extraction(),
            assessment=_make_assessment(),
        )
        assert called["flag"] is False
