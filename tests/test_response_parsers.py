"""
test_response_parsers.py

Unit tests for app/llm/response_parsers.py.
"""

from __future__ import annotations

import pytest

from app.llm.exceptions import LLMResponseParsingError
from app.llm.response_parsers import parse_comparison_response, parse_qa_response


class TestParseQAResponse:
    def test_parses_well_formed_response(self) -> None:
        raw = (
            "ANSWER: Payment is due within 30 days of invoice receipt.\n"
            "SOURCE_CONTRACT: Sample Services Agreement\n"
            "SOURCE_PAGE: 2\n"
            "SUPPORTING_TEXT: Payment shall be due within thirty (30) days of invoice."
        )
        parsed = parse_qa_response(raw)
        assert parsed["answer"] == "Payment is due within 30 days of invoice receipt."
        assert parsed["source_contract"] == "Sample Services Agreement"
        assert parsed["source_page"] == 2
        assert "thirty (30) days" in parsed["supporting_text"]

    def test_parses_not_found_response(self) -> None:
        raw = (
            "ANSWER: NOT_FOUND_IN_CONTRACT\n"
            "SOURCE_CONTRACT: \n"
            "SOURCE_PAGE: \n"
            "SUPPORTING_TEXT: "
        )
        parsed = parse_qa_response(raw)
        assert parsed["answer"] == "NOT_FOUND_IN_CONTRACT"
        assert parsed["source_contract"] is None
        assert parsed["source_page"] is None
        assert parsed["supporting_text"] is None

    def test_handles_multiline_answer(self) -> None:
        raw = (
            "ANSWER: The liability cap is limited to total fees paid\n"
            "in the preceding twelve months.\n"
            "SOURCE_CONTRACT: MSA\n"
            "SOURCE_PAGE: 5\n"
            "SUPPORTING_TEXT: Liability shall not exceed total fees paid."
        )
        parsed = parse_qa_response(raw)
        assert "liability cap" in parsed["answer"].lower()
        assert "twelve months" in parsed["answer"]

    def test_extracts_numeric_page_from_noisy_text(self) -> None:
        raw = (
            "ANSWER: Some answer.\n"
            "SOURCE_CONTRACT: Contract X\n"
            "SOURCE_PAGE: page 7\n"
            "SUPPORTING_TEXT: Some supporting text."
        )
        parsed = parse_qa_response(raw)
        assert parsed["source_page"] == 7

    def test_malformed_response_raises(self) -> None:
        with pytest.raises(LLMResponseParsingError):
            parse_qa_response("This is just a free-text response with no structure at all.")

    def test_missing_field_raises(self) -> None:
        raw = "ANSWER: Some answer.\nSOURCE_CONTRACT: Contract X"  # missing SOURCE_PAGE/SUPPORTING_TEXT
        with pytest.raises(LLMResponseParsingError):
            parse_qa_response(raw)


class TestParseComparisonResponse:
    def test_parses_contradictory_true(self) -> None:
        raw = (
            "IS_CONTRADICTORY: true\n"
            "EXPLANATION: Contract A requires 30 days notice while Contract B requires 90 days, "
            "which is a direct conflict."
        )
        parsed = parse_comparison_response(raw)
        assert parsed["is_contradictory"] is True
        assert "30 days" in parsed["explanation"]

    def test_parses_contradictory_false(self) -> None:
        raw = "IS_CONTRADICTORY: false\nEXPLANATION: Both clauses express the same obligation differently."
        parsed = parse_comparison_response(raw)
        assert parsed["is_contradictory"] is False

    def test_case_insensitive_boolean(self) -> None:
        raw = "IS_CONTRADICTORY: TRUE\nEXPLANATION: Conflicting terms."
        parsed = parse_comparison_response(raw)
        assert parsed["is_contradictory"] is True

    def test_malformed_comparison_response_raises(self) -> None:
        with pytest.raises(LLMResponseParsingError):
            parse_comparison_response("Not in the expected format at all.")
