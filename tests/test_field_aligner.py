"""
test_field_aligner.py

Unit tests for app/comparison/field_aligner.py.
"""

from __future__ import annotations

from app.comparison.field_aligner import align_fields
from app.models.extraction import ContractExtractionResult, ExtractedField, FieldType


def _make_field(
    field_type: FieldType,
    value_text: str,
    document_id: str = "doc-1",
    contract_name: str = "Contract A",
    confidence: float = 0.9,
    page_number: int = 1,
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


class TestAlignFields:
    def test_shared_field_types_produce_one_pair_each(self) -> None:
        result_a = _make_result(
            [
                _make_field(FieldType.PAYMENT_TERMS, "Net 30."),
                _make_field(FieldType.TERMINATION_CLAUSE, "60 days notice."),
            ]
        )
        result_b = _make_result(
            [
                _make_field(FieldType.PAYMENT_TERMS, "Net 60.", document_id="doc-2", contract_name="Contract B"),
                _make_field(FieldType.TERMINATION_CLAUSE, "90 days notice.", document_id="doc-2", contract_name="Contract B"),
            ],
            document_id="doc-2",
            contract_name="Contract B",
        )
        pairs = align_fields(result_a, result_b)
        field_types = {fa.field_type for fa, _ in pairs}
        assert field_types == {FieldType.PAYMENT_TERMS, FieldType.TERMINATION_CLAUSE}
        assert len(pairs) == 2

    def test_field_present_in_only_one_contract_omitted(self) -> None:
        result_a = _make_result([_make_field(FieldType.PAYMENT_TERMS, "Net 30.")])
        result_b = _make_result(
            [_make_field(FieldType.CONFIDENTIALITY_CLAUSE, "Keep it secret.", document_id="doc-2", contract_name="Contract B")],
            document_id="doc-2",
            contract_name="Contract B",
        )
        pairs = align_fields(result_a, result_b)
        assert pairs == []

    def test_highest_confidence_representative_selected_per_type(self) -> None:
        # Contract A has two PAYMENT_TERMS fields — only the higher-confidence one should be selected.
        low_conf = _make_field(FieldType.PAYMENT_TERMS, "Low confidence clause.", confidence=0.6)
        high_conf = _make_field(FieldType.PAYMENT_TERMS, "High confidence clause.", confidence=0.95)
        result_a = _make_result([low_conf, high_conf])
        result_b = _make_result(
            [_make_field(FieldType.PAYMENT_TERMS, "Net 60.", document_id="doc-2", contract_name="Contract B")],
            document_id="doc-2",
            contract_name="Contract B",
        )
        pairs = align_fields(result_a, result_b)
        assert len(pairs) == 1
        selected_a, _ = pairs[0]
        assert selected_a.value_text == "High confidence clause."

    def test_empty_extraction_results_produce_no_pairs(self) -> None:
        result_a = _make_result([])
        result_b = _make_result([], document_id="doc-2", contract_name="Contract B")
        assert align_fields(result_a, result_b) == []

    def test_pair_carries_correct_contract_attribution(self) -> None:
        result_a = _make_result(
            [_make_field(FieldType.TERMINATION_CLAUSE, "Terminate with 30 days notice.", contract_name="Contract Alpha", page_number=4)]
        )
        result_b = _make_result(
            [_make_field(FieldType.TERMINATION_CLAUSE, "Terminate with 90 days notice.", document_id="doc-2", contract_name="Contract Beta", page_number=7)],
            document_id="doc-2",
            contract_name="Contract Beta",
        )
        pairs = align_fields(result_a, result_b)
        fa, fb = pairs[0]
        assert fa.contract_name == "Contract Alpha"
        assert fa.page_number == 4
        assert fb.contract_name == "Contract Beta"
        assert fb.page_number == 7
