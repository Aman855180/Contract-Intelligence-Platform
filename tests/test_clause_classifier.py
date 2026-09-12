"""
test_clause_classifier.py

Unit tests for app/extraction/clause_classifier.py against the real
config/extraction_keywords.yaml keyword lists.
"""

from __future__ import annotations

from app.extraction.clause_classifier import classify_chunk
from app.models.extraction import FieldType


class TestClassifyChunk:
    def test_payment_terms_detected(self) -> None:
        matches = classify_chunk("Payment shall be due within thirty days of invoice receipt.")
        field_types = {ft for ft, _ in matches}
        assert FieldType.PAYMENT_TERMS in field_types

    def test_termination_clause_detected(self) -> None:
        matches = classify_chunk("Either party may terminate this agreement with sixty days notice.")
        field_types = {ft for ft, _ in matches}
        assert FieldType.TERMINATION_CLAUSE in field_types

    def test_confidentiality_clause_detected(self) -> None:
        matches = classify_chunk("Each party agrees to keep all confidential information secret.")
        field_types = {ft for ft, _ in matches}
        assert FieldType.CONFIDENTIALITY_CLAUSE in field_types

    def test_liability_terms_detected(self) -> None:
        matches = classify_chunk("Liability under this agreement shall not exceed total fees paid.")
        field_types = {ft for ft, _ in matches}
        assert FieldType.LIABILITY_TERMS in field_types

    def test_renewal_terms_detected(self) -> None:
        matches = classify_chunk("This agreement shall automatically renew for successive one-year terms.")
        field_types = {ft for ft, _ in matches}
        assert FieldType.RENEWAL_TERMS in field_types

    def test_signature_detected(self) -> None:
        matches = classify_chunk("IN WITNESS WHEREOF, the parties have executed this agreement.")
        field_types = {ft for ft, _ in matches}
        assert FieldType.SIGNATURE in field_types

    def test_no_match_returns_empty_list(self) -> None:
        matches = classify_chunk("The weather today is sunny with a chance of rain.")
        assert matches == []

    def test_multiple_field_types_can_match_one_chunk(self) -> None:
        text = (
            "Payment shall be due within thirty days. Either party may terminate "
            "this agreement with sixty days written notice."
        )
        matches = classify_chunk(text)
        field_types = {ft for ft, _ in matches}
        assert FieldType.PAYMENT_TERMS in field_types
        assert FieldType.TERMINATION_CLAUSE in field_types

    def test_more_keyword_matches_yield_higher_confidence(self) -> None:
        weak_text = "Payment is due."
        strong_text = "Payment shall be due, payable via invoice, net 30, as compensation for services."

        weak_matches = dict(classify_chunk(weak_text))
        strong_matches = dict(classify_chunk(strong_text))

        assert strong_matches[FieldType.PAYMENT_TERMS] > weak_matches[FieldType.PAYMENT_TERMS]

    def test_confidence_always_within_bounds(self) -> None:
        text = (
            "Payment payable invoice due within net 30 net 60 net 90 compensation amount due "
            "fees shall be paid"
        )
        matches = classify_chunk(text)
        for _, confidence in matches:
            assert 0.0 < confidence <= 1.0

    def test_case_insensitive_matching(self) -> None:
        matches_lower = classify_chunk("payment shall be due within thirty days.")
        matches_upper = classify_chunk("PAYMENT SHALL BE DUE WITHIN THIRTY DAYS.")
        assert {ft for ft, _ in matches_lower} == {ft for ft, _ in matches_upper}

    def test_obligations_and_key_dates_not_classified_here(self) -> None:
        # obligations and key_dates have their own dedicated extraction modules;
        # the clause classifier should never emit them.
        text = "The Contractor shall deliver the report by January 1, 2025."
        matches = classify_chunk(text)
        field_types = {ft for ft, _ in matches}
        assert FieldType.OBLIGATIONS not in field_types
        assert FieldType.KEY_DATES not in field_types
