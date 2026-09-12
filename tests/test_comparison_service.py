"""
test_comparison_service.py

Integration tests for app/comparison/comparison_service.py, exercising
the full rule-based + NLI pipeline. The NLI check_contradiction function
is stubbed so tests run without network access, while the rule-based
detector and field aligner run for real.
"""

from __future__ import annotations

import pytest

from app.comparison import comparison_service
from app.comparison.exceptions import ComparisonError
from app.groundedness import nli_checker
from app.models.comparison import MismatchDetectionMethod
from app.models.extraction import ContractExtractionResult, ExtractedField, FieldType


def _make_field(
    field_type: FieldType,
    value_text: str,
    document_id: str = "doc-1",
    contract_name: str = "Contract A",
    page_number: int = 1,
    confidence: float = 0.9,
) -> ExtractedField:
    return ExtractedField(
        field_type=field_type,
        value_text=value_text,
        document_id=document_id,
        contract_name=contract_name,
        page_number=page_number,
        confidence=confidence,
    )


def _make_result(
    fields: list[ExtractedField],
    document_id: str = "doc-1",
    contract_name: str = "Contract A",
) -> ContractExtractionResult:
    return ContractExtractionResult(
        document_id=document_id,
        contract_name=contract_name,
        fields=fields,
    )


@pytest.fixture(autouse=True)
def reset_nli_cache():
    nli_checker.reset_model_cache()
    yield
    nli_checker.reset_model_cache()


class TestCompareContracts:
    def test_rule_based_numeric_mismatch_detected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Stub NLI to always return 0 contradiction so only rule-based fires.
        monkeypatch.setattr(comparison_service, "check_nli_contradiction", lambda fa, fb: None)

        result_a = _make_result([_make_field(FieldType.PAYMENT_TERMS, "Payment due within 30 days.")])
        result_b = _make_result(
            [_make_field(FieldType.PAYMENT_TERMS, "Payment due within 60 days.", document_id="doc-2", contract_name="Contract B")],
            document_id="doc-2", contract_name="Contract B",
        )

        result = comparison_service.compare_contracts(result_a, result_b)

        assert result.fields_compared == 1
        assert len(result.inconsistencies) == 1
        assert result.inconsistencies[0].detection_method == MismatchDetectionMethod.RULE_BASED_NUMERIC_DATE

    def test_nli_contradiction_detected_for_clause_text_field(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from app.models.comparison import FieldInconsistency

        nli_inconsistency = FieldInconsistency(
            field_type=FieldType.TERMINATION_CLAUSE,
            detection_method=MismatchDetectionMethod.NLI_SEMANTIC_CONTRADICTION,
            contract_a_name="Contract A", contract_a_page=2,
            contract_a_text="Vendor may terminate immediately for any breach.",
            contract_b_name="Contract B", contract_b_page=3,
            contract_b_text="Termination requires 30-day cure period before taking effect.",
            explanation="NLI contradiction detected (score=0.82).",
            confidence=0.82,
        )
        monkeypatch.setattr(comparison_service, "check_nli_contradiction", lambda fa, fb: nli_inconsistency)

        result_a = _make_result([_make_field(FieldType.TERMINATION_CLAUSE, "Vendor may terminate immediately for any breach.")])
        result_b = _make_result(
            [_make_field(FieldType.TERMINATION_CLAUSE, "Termination requires 30-day cure period.", document_id="doc-2", contract_name="Contract B")],
            document_id="doc-2", contract_name="Contract B",
        )

        result = comparison_service.compare_contracts(result_a, result_b)

        assert len(result.inconsistencies) == 1
        assert result.inconsistencies[0].detection_method == MismatchDetectionMethod.NLI_SEMANTIC_CONTRADICTION
        assert result.inconsistencies[0].confidence == pytest.approx(0.82)

    def test_rule_flagged_field_does_not_also_trigger_nli(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A field already flagged by rule-based should NOT also generate an NLI flag."""
        nli_call_log: list[tuple] = []

        def logging_nli(fa, fb):
            nli_call_log.append((fa.field_type,))
            return None

        monkeypatch.setattr(comparison_service, "check_nli_contradiction", logging_nli)

        result_a = _make_result([_make_field(FieldType.PAYMENT_TERMS, "Net 30 days.")])
        result_b = _make_result(
            [_make_field(FieldType.PAYMENT_TERMS, "Net 90 days.", document_id="doc-2", contract_name="Contract B")],
            document_id="doc-2", contract_name="Contract B",
        )

        result = comparison_service.compare_contracts(result_a, result_b)

        # Rule-based flagged it → NLI should still run (per our design: NLI always
        # runs as second pass for rule-based fields when rule found no mismatch,
        # and NOT run when rule DID find a mismatch — but here rule DID find one
        # so NLI should NOT be called).
        assert len(result.inconsistencies) == 1
        assert result.inconsistencies[0].detection_method == MismatchDetectionMethod.RULE_BASED_NUMERIC_DATE
        nli_called_for_payment = any(ft == FieldType.PAYMENT_TERMS for ft, in nli_call_log)
        assert not nli_called_for_payment

    def test_nli_runs_on_rule_field_when_no_numeric_mismatch_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When rule-based finds no mismatch on a PAYMENT_TERMS field, NLI should still run."""
        nli_calls: list[FieldType] = []

        def logging_nli(fa, fb):
            nli_calls.append(fa.field_type)
            return None

        monkeypatch.setattr(comparison_service, "check_nli_contradiction", logging_nli)

        # Both use "thirty" (prose) so rule-based can't compare numerically.
        result_a = _make_result([_make_field(FieldType.PAYMENT_TERMS, "Fees payable within thirty days.")])
        result_b = _make_result(
            [_make_field(FieldType.PAYMENT_TERMS, "Fees are non-refundable and payable immediately.", document_id="doc-2", contract_name="Contract B")],
            document_id="doc-2", contract_name="Contract B",
        )

        comparison_service.compare_contracts(result_a, result_b)

        assert FieldType.PAYMENT_TERMS in nli_calls

    def test_consistent_contracts_produce_no_inconsistencies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(comparison_service, "check_nli_contradiction", lambda fa, fb: None)

        result_a = _make_result([_make_field(FieldType.PAYMENT_TERMS, "Net 30 days from invoice.")])
        result_b = _make_result(
            [_make_field(FieldType.PAYMENT_TERMS, "Payment due within 30 days.", document_id="doc-2", contract_name="Contract B")],
            document_id="doc-2", contract_name="Contract B",
        )

        result = comparison_service.compare_contracts(result_a, result_b)
        assert result.inconsistencies == []
        assert result.fields_compared == 1

    def test_multiple_field_types_compared_and_counted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(comparison_service, "check_nli_contradiction", lambda fa, fb: None)

        result_a = _make_result([
            _make_field(FieldType.PAYMENT_TERMS, "Net 30."),
            _make_field(FieldType.TERMINATION_CLAUSE, "60 days notice.", page_number=2),
            _make_field(FieldType.KEY_DATES, "Effective January 1, 2025.", page_number=3),
        ])
        result_b = _make_result(
            [
                _make_field(FieldType.PAYMENT_TERMS, "Net 30.", document_id="doc-2", contract_name="Contract B"),
                _make_field(FieldType.TERMINATION_CLAUSE, "90 days notice.", document_id="doc-2", contract_name="Contract B", page_number=2),
                _make_field(FieldType.KEY_DATES, "Effective January 1, 2025.", document_id="doc-2", contract_name="Contract B", page_number=3),
            ],
            document_id="doc-2", contract_name="Contract B",
        )

        result = comparison_service.compare_contracts(result_a, result_b)
        assert result.fields_compared == 3

    def test_empty_extraction_a_raises_comparison_error(self) -> None:
        result_a = _make_result([])
        result_b = _make_result(
            [_make_field(FieldType.PAYMENT_TERMS, "Net 30.", document_id="doc-2", contract_name="Contract B")],
            document_id="doc-2", contract_name="Contract B",
        )
        with pytest.raises(ComparisonError):
            comparison_service.compare_contracts(result_a, result_b)

    def test_result_schema_is_fully_populated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(comparison_service, "check_nli_contradiction", lambda fa, fb: None)

        result_a = _make_result([_make_field(FieldType.PAYMENT_TERMS, "Net 30.")])
        result_b = _make_result(
            [_make_field(FieldType.PAYMENT_TERMS, "Net 60.", document_id="doc-2", contract_name="Contract B")],
            document_id="doc-2", contract_name="Contract B",
        )

        result = comparison_service.compare_contracts(result_a, result_b)

        assert result.document_id_a == "doc-1"
        assert result.document_id_b == "doc-2"
        assert result.contract_a_name == "Contract A"
        assert result.contract_b_name == "Contract B"
        assert result.fields_compared >= 0
