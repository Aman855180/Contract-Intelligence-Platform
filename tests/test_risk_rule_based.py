"""
test_risk_rule_based.py

Unit tests for app/risk/rule_based_flags.py.
Pure logic — no I/O, no LLM.
"""

from __future__ import annotations

import pytest

from app.models.extraction import ContractExtractionResult, ExtractedField, FieldType
from app.models.risk import RiskCategory, RiskFlagSource
from app.risk.rule_based_flags import detect_rule_based_flags


def _make_result(field_types: list[FieldType], document_id: str = "doc-1") -> ContractExtractionResult:
    fields = [
        ExtractedField(
            field_type=ft,
            value_text=f"Sample {ft.value} text.",
            document_id=document_id,
            contract_name="Test Contract",
            page_number=1,
            confidence=0.9,
        )
        for ft in field_types
    ]
    return ContractExtractionResult(
        document_id=document_id,
        contract_name="Test Contract",
        fields=fields,
    )


class TestDetectRuleBasedFlags:
    def test_always_returns_exactly_three_flags(self) -> None:
        result = detect_rule_based_flags(_make_result([]))
        assert len(result) == 3

    def test_missing_signature_triggers(self) -> None:
        extraction = _make_result([])  # no fields at all
        flags = detect_rule_based_flags(extraction)
        sig_flag = next(f for f in flags if f.category == RiskCategory.MISSING_SIGNATURE_FIELD)
        assert sig_flag.triggered is True
        assert sig_flag.source == RiskFlagSource.RULE_BASED

    def test_present_signature_does_not_trigger(self) -> None:
        extraction = _make_result([FieldType.SIGNATURE])
        flags = detect_rule_based_flags(extraction)
        sig_flag = next(f for f in flags if f.category == RiskCategory.MISSING_SIGNATURE_FIELD)
        assert sig_flag.triggered is False

    def test_missing_termination_clause_triggers(self) -> None:
        extraction = _make_result([FieldType.SIGNATURE, FieldType.CONFIDENTIALITY_CLAUSE])
        flags = detect_rule_based_flags(extraction)
        term_flag = next(f for f in flags if f.category == RiskCategory.MISSING_TERMINATION_CLAUSE)
        assert term_flag.triggered is True

    def test_present_termination_clause_does_not_trigger(self) -> None:
        extraction = _make_result([FieldType.TERMINATION_CLAUSE])
        flags = detect_rule_based_flags(extraction)
        term_flag = next(f for f in flags if f.category == RiskCategory.MISSING_TERMINATION_CLAUSE)
        assert term_flag.triggered is False

    def test_missing_confidentiality_clause_triggers(self) -> None:
        extraction = _make_result([FieldType.SIGNATURE, FieldType.TERMINATION_CLAUSE])
        flags = detect_rule_based_flags(extraction)
        conf_flag = next(f for f in flags if f.category == RiskCategory.MISSING_CONFIDENTIALITY_CLAUSE)
        assert conf_flag.triggered is True

    def test_all_present_no_flags_triggered(self) -> None:
        extraction = _make_result([
            FieldType.SIGNATURE,
            FieldType.TERMINATION_CLAUSE,
            FieldType.CONFIDENTIALITY_CLAUSE,
        ])
        flags = detect_rule_based_flags(extraction)
        assert all(not f.triggered for f in flags)

    def test_all_absent_all_flags_triggered(self) -> None:
        extraction = _make_result([])
        flags = detect_rule_based_flags(extraction)
        assert all(f.triggered for f in flags)

    def test_weights_are_positive_and_from_config(self) -> None:
        flags = detect_rule_based_flags(_make_result([]))
        for flag in flags:
            assert flag.weight > 0

    def test_triggered_flag_has_meaningful_explanation(self) -> None:
        flags = detect_rule_based_flags(_make_result([]))
        for flag in flags:
            assert flag.triggered is True
            assert len(flag.explanation) > 10

    def test_non_triggered_flag_has_explanation(self) -> None:
        extraction = _make_result([
            FieldType.SIGNATURE,
            FieldType.TERMINATION_CLAUSE,
            FieldType.CONFIDENTIALITY_CLAUSE,
        ])
        flags = detect_rule_based_flags(extraction)
        for flag in flags:
            assert len(flag.explanation) > 5

    def test_categories_cover_all_three_rule_based_types(self) -> None:
        flags = detect_rule_based_flags(_make_result([]))
        categories = {f.category for f in flags}
        assert RiskCategory.MISSING_SIGNATURE_FIELD in categories
        assert RiskCategory.MISSING_TERMINATION_CLAUSE in categories
        assert RiskCategory.MISSING_CONFIDENTIALITY_CLAUSE in categories
