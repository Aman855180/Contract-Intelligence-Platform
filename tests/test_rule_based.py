"""
test_rule_based.py

Unit tests for app/comparison/rule_based.py.
Pure logic — no I/O, no LLM, no NLI.
"""

from __future__ import annotations

import pytest

from app.comparison.rule_based import check_rule_based_mismatch
from app.models.comparison import MismatchDetectionMethod
from app.models.extraction import ExtractedField, FieldType


def _make_field(
    field_type: FieldType,
    value_text: str,
    contract_name: str = "Contract A",
    page_number: int = 1,
    document_id: str = "doc-1",
) -> ExtractedField:
    return ExtractedField(
        field_type=field_type,
        value_text=value_text,
        document_id=document_id,
        contract_name=contract_name,
        page_number=page_number,
        confidence=0.9,
    )


class TestPaymentTerms:
    def test_numeric_mismatch_flagged(self) -> None:
        fa = _make_field(FieldType.PAYMENT_TERMS, "Payment due within 30 days of invoice.")
        fb = _make_field(FieldType.PAYMENT_TERMS, "Payment due within 60 days of invoice.", "Contract B")
        result = check_rule_based_mismatch(fa, fb)
        assert result is not None
        assert result.detection_method == MismatchDetectionMethod.RULE_BASED_NUMERIC_DATE
        assert result.confidence == 1.0
        assert "30" in result.explanation or "60" in result.explanation

    def test_identical_numbers_not_flagged(self) -> None:
        fa = _make_field(FieldType.PAYMENT_TERMS, "Net 30 days from invoice date.")
        fb = _make_field(FieldType.PAYMENT_TERMS, "Payment within 30 days.", "Contract B")
        assert check_rule_based_mismatch(fa, fb) is None

    def test_one_side_no_numbers_not_flagged(self) -> None:
        # "thirty" is prose, not a digit — can't compare numerically so no flag
        fa = _make_field(FieldType.PAYMENT_TERMS, "Payment due within thirty days.")
        fb = _make_field(FieldType.PAYMENT_TERMS, "Payment due within 30 days.", "Contract B")
        assert check_rule_based_mismatch(fa, fb) is None

    def test_flagged_inconsistency_carries_correct_citations(self) -> None:
        fa = _make_field(FieldType.PAYMENT_TERMS, "Net 30.", "Contract Alpha", page_number=3)
        fb = _make_field(FieldType.PAYMENT_TERMS, "Net 90.", "Contract Beta", page_number=5, document_id="doc-2")
        result = check_rule_based_mismatch(fa, fb)
        assert result.contract_a_name == "Contract Alpha"
        assert result.contract_a_page == 3
        assert result.contract_b_name == "Contract Beta"
        assert result.contract_b_page == 5


class TestRenewalTerms:
    def test_different_renewal_durations_flagged(self) -> None:
        fa = _make_field(FieldType.RENEWAL_TERMS, "Automatically renews for 1 year.")
        fb = _make_field(FieldType.RENEWAL_TERMS, "Automatically renews for 3 years.", "Contract B")
        result = check_rule_based_mismatch(fa, fb)
        assert result is not None
        assert result.detection_method == MismatchDetectionMethod.RULE_BASED_NUMERIC_DATE

    def test_same_renewal_duration_not_flagged(self) -> None:
        fa = _make_field(FieldType.RENEWAL_TERMS, "Renews annually for 1 year periods.")
        fb = _make_field(FieldType.RENEWAL_TERMS, "Auto-renews each 1 year term.", "Contract B")
        assert check_rule_based_mismatch(fa, fb) is None


class TestKeyDates:
    def test_different_dates_flagged(self) -> None:
        fa = _make_field(FieldType.KEY_DATES, "Effective January 1, 2024.")
        fb = _make_field(FieldType.KEY_DATES, "Effective March 1, 2025.", "Contract B")
        result = check_rule_based_mismatch(fa, fb)
        assert result is not None
        assert result.detection_method == MismatchDetectionMethod.RULE_BASED_NUMERIC_DATE

    def test_identical_dates_not_flagged(self) -> None:
        fa = _make_field(FieldType.KEY_DATES, "Effective January 1, 2024.")
        fb = _make_field(FieldType.KEY_DATES, "Effective date: January 1, 2024.", "Contract B")
        assert check_rule_based_mismatch(fa, fb) is None

    def test_one_side_no_dates_not_flagged(self) -> None:
        fa = _make_field(FieldType.KEY_DATES, "Effective upon execution.")
        fb = _make_field(FieldType.KEY_DATES, "Effective January 1, 2024.", "Contract B")
        assert check_rule_based_mismatch(fa, fb) is None


class TestNonRuleBasedFieldTypes:
    def test_termination_clause_returns_none(self) -> None:
        # TERMINATION_CLAUSE is prose-only, handled by NLI — rule checker must return None
        fa = _make_field(FieldType.TERMINATION_CLAUSE, "Terminate with 30 days notice.")
        fb = _make_field(FieldType.TERMINATION_CLAUSE, "Terminate with 90 days notice.", "Contract B")
        # Even though the numbers differ, TERMINATION_CLAUSE is not in RULE_BASED_FIELD_TYPES
        assert check_rule_based_mismatch(fa, fb) is None

    def test_mismatched_field_types_raises(self) -> None:
        fa = _make_field(FieldType.PAYMENT_TERMS, "Net 30.")
        fb = _make_field(FieldType.TERMINATION_CLAUSE, "60 days notice.", "Contract B")
        with pytest.raises(ValueError):
            check_rule_based_mismatch(fa, fb)
