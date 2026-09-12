"""
test_risk_service.py

Integration tests for app/risk/risk_service.py. Uses a pre-built
ContractExtractionResult to skip auto-extraction, stubs the LLM
classifier, and verifies the full orchestration: rule flags + LLM flags
combined, score computed, RiskAssessment populated and valid.
"""

from __future__ import annotations

import json

import pytest

from app.models.extraction import ContractExtractionResult, ExtractedField, FieldType
from app.models.risk import RiskAssessment, RiskCategory, RiskFlagSource
from app.risk import llm_classifier, risk_service
from app.risk.exceptions import LLMClassificationError


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_full_extraction(document_id: str = "doc-1") -> ContractExtractionResult:
    """Extraction result that has all three rule-checked clause types present."""
    fields = [
        ExtractedField(field_type=ft, value_text=f"{ft.value} text.", document_id=document_id,
                       contract_name="Test Contract", page_number=i + 1, confidence=0.9)
        for i, ft in enumerate([
            FieldType.SIGNATURE, FieldType.TERMINATION_CLAUSE, FieldType.CONFIDENTIALITY_CLAUSE,
        ])
    ]
    return ContractExtractionResult(
        document_id=document_id, contract_name="Test Contract", fields=fields
    )


def _make_minimal_extraction(document_id: str = "doc-1") -> ContractExtractionResult:
    """Extraction result with NO clauses — all three rule flags should trigger."""
    return ContractExtractionResult(
        document_id=document_id, contract_name="Sparse Contract", fields=[]
    )


def _stub_llm_flags(monkeypatch: pytest.MonkeyPatch, *, indemnity: bool = False,
                    liability: bool = False, payment: bool = False) -> None:
    """Monkeypatch classify_llm_risk_flags to return controlled flags."""
    from app.models.risk import RiskFlag

    def _fake_classify(document_id: str, contract_name: str):
        return [
            RiskFlag(category=RiskCategory.UNLIMITED_INDEMNITY_LANGUAGE,
                     source=RiskFlagSource.LLM_CLASSIFIED, triggered=indemnity,
                     weight=20, explanation="stub"),
            RiskFlag(category=RiskCategory.HIGH_LIABILITY_EXPOSURE,
                     source=RiskFlagSource.LLM_CLASSIFIED, triggered=liability,
                     weight=15, explanation="stub"),
            RiskFlag(category=RiskCategory.AMBIGUOUS_PAYMENT_TERMS,
                     source=RiskFlagSource.LLM_CLASSIFIED, triggered=payment,
                     weight=10, explanation="stub"),
        ]

    monkeypatch.setattr(risk_service, "classify_llm_risk_flags", _fake_classify)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAssessRisk:
    def test_returns_risk_assessment_with_six_flags(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_llm_flags(monkeypatch)
        result = risk_service.assess_risk("doc-1", extraction=_make_full_extraction())
        assert isinstance(result, RiskAssessment)
        # 3 rule-based + 3 LLM-classified
        assert len(result.flags) == 6

    def test_all_clauses_present_no_rule_flags_triggered(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_llm_flags(monkeypatch)
        result = risk_service.assess_risk("doc-1", extraction=_make_full_extraction())
        rule_flags = [f for f in result.flags if f.source == RiskFlagSource.RULE_BASED]
        assert all(not f.triggered for f in rule_flags)

    def test_no_clauses_all_rule_flags_triggered(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_llm_flags(monkeypatch)
        result = risk_service.assess_risk("doc-1", extraction=_make_minimal_extraction())
        rule_flags = [f for f in result.flags if f.source == RiskFlagSource.RULE_BASED]
        assert all(f.triggered for f in rule_flags)

    def test_llm_triggered_flags_reflected_in_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_llm_flags(monkeypatch, indemnity=True, liability=True)
        result = risk_service.assess_risk("doc-1", extraction=_make_full_extraction())
        llm_flags = [f for f in result.flags if f.source == RiskFlagSource.LLM_CLASSIFIED]
        by_cat = {f.category: f for f in llm_flags}
        assert by_cat[RiskCategory.UNLIMITED_INDEMNITY_LANGUAGE].triggered is True
        assert by_cat[RiskCategory.HIGH_LIABILITY_EXPOSURE].triggered is True
        assert by_cat[RiskCategory.AMBIGUOUS_PAYMENT_TERMS].triggered is False

    def test_score_matches_sum_of_triggered_weights(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # All clauses present (no rule flags) + indemnity triggered (weight 20)
        _stub_llm_flags(monkeypatch, indemnity=True)
        result = risk_service.assess_risk("doc-1", extraction=_make_full_extraction())
        assert result.risk_score == 20

    def test_score_sums_rule_and_llm_weights(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Missing termination (20) + missing confidentiality (15) + liability triggered (15)
        extraction = ContractExtractionResult(
            document_id="doc-1", contract_name="Contract",
            fields=[ExtractedField(field_type=FieldType.SIGNATURE, value_text="sig",
                                   document_id="doc-1", contract_name="Contract",
                                   page_number=1, confidence=0.9)]
        )
        _stub_llm_flags(monkeypatch, liability=True)
        result = risk_service.assess_risk("doc-1", extraction=extraction)
        # Missing termination=20, missing confidentiality=15, high_liability=15 → 50
        assert result.risk_score == 50

    def test_score_clipped_at_100(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_llm_flags(monkeypatch, indemnity=True, liability=True, payment=True)
        # No clauses: all 3 rule flags (25+20+15=60) + all 3 LLM flags (20+15+10=45) = 105 → 100
        result = risk_service.assess_risk("doc-1", extraction=_make_minimal_extraction())
        assert result.risk_score == 100

    def test_risk_assessment_schema_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_llm_flags(monkeypatch)
        result = risk_service.assess_risk("doc-1", extraction=_make_full_extraction())
        assert result.document_id == "doc-1"
        assert result.contract_name == "Test Contract"
        assert 0 <= result.risk_score <= 100

    def test_triggered_flags_helper_returns_only_triggered(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_llm_flags(monkeypatch, indemnity=True)
        result = risk_service.assess_risk("doc-1", extraction=_make_full_extraction())
        triggered = result.triggered_flags()
        assert all(f.triggered for f in triggered)
        assert any(f.category == RiskCategory.UNLIMITED_INDEMNITY_LANGUAGE for f in triggered)

    def test_llm_classification_error_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            risk_service, "classify_llm_risk_flags",
            lambda doc_id, name: (_ for _ in ()).throw(
                LLMClassificationError("LLM failed after all retries.")
            )
        )
        with pytest.raises(LLMClassificationError):
            risk_service.assess_risk("doc-1", extraction=_make_full_extraction())

    def test_flag_sources_correctly_assigned(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_llm_flags(monkeypatch)
        result = risk_service.assess_risk("doc-1", extraction=_make_full_extraction())
        rule_sources = {f.source for f in result.flags if f.source == RiskFlagSource.RULE_BASED}
        llm_sources = {f.source for f in result.flags if f.source == RiskFlagSource.LLM_CLASSIFIED}
        assert rule_sources == {RiskFlagSource.RULE_BASED}
        assert llm_sources == {RiskFlagSource.LLM_CLASSIFIED}
