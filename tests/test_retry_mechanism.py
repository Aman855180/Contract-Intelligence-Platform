"""
test_retry_mechanism.py

Unit tests verifying the LLM retry mechanism, safe limits, structured output,
fail-fast error handling, and secure sanitized retry logging.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.llm.exceptions import GeminiConnectionError, LLMPermanentError
from app.models.extraction import ContractExtractionResult, ExtractedField, FieldType
from app.models.risk import RiskAssessment, RiskCategory, RiskFlag, RiskFlagSource
from app.reports import report_generator
from app.reports.exceptions import ReportValidationError
from app.reports.report_generator import (
    LLM_REPORT_OUTPUT_SCHEMA,
    _LLMReportOutput,
    generate_report,
)
from app.risk import llm_classifier
from app.risk.exceptions import LLMClassificationError


def _make_extraction() -> ContractExtractionResult:
    return ContractExtractionResult(
        document_id="doc-test-1",
        contract_name="Retry Test Contract",
        fields=[
            ExtractedField(
                field_type=FieldType.PAYMENT_TERMS,
                value_text="Net 30 days upon invoice.",
                document_id="doc-test-1",
                contract_name="Retry Test Contract",
                page_number=1,
                confidence=0.95,
            )
        ],
    )


def _make_assessment() -> RiskAssessment:
    return RiskAssessment(
        document_id="doc-test-1",
        contract_name="Retry Test Contract",
        flags=[
            RiskFlag(
                category=RiskCategory.UNLIMITED_INDEMNITY_LANGUAGE,
                source=RiskFlagSource.LLM_CLASSIFIED,
                triggered=False,
                weight=25,
                explanation="No indemnity clause found.",
            )
        ],
        risk_score=0,
    )


def _valid_report_json() -> str:
    return json.dumps({
        "overview": "Overview of agreement between Company A and Company B.",
        "key_parties": [
            {"name": "Company A", "role": "Client", "source_contract": "Test", "source_page": 1}
        ],
        "important_dates": [
            {"label": "Effective Date", "date": "2025-01-01", "source_contract": "Test", "source_page": 1}
        ],
        "payment_terms": {
            "summary": "Net 30 days.", "source_contract": "Test", "source_page": 1
        },
        "key_clauses": [
            {"clause_type": "termination", "summary": "30 days notice.", "source_contract": "Test", "source_page": 1}
        ],
        "recommendations": ["Review notice period."],
    })


@pytest.fixture(autouse=True)
def stub_retrieval(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(report_generator, "embed_queries", lambda q: [[0.1] * 8 for _ in q])
    monkeypatch.setattr(report_generator, "query_similar_batch", lambda vecs, top_k, document_id: [[] for _ in vecs])
    monkeypatch.setattr(llm_classifier, "embed_queries", lambda q: [[0.1] * 8 for _ in q])
    monkeypatch.setattr(llm_classifier, "query_similar_batch", lambda vecs, top_k, document_id: [[] for _ in vecs])


class TestStructuredOutputIntegration:
    """Verify structured output schema is passed to avoid free-form parsing failures."""

    def test_report_generation_passes_structured_schema(self, monkeypatch: pytest.MonkeyPatch) -> None:
        received_schema = []

        def mock_generate(prompt: str, attempt: int | None = None, response_schema: Any | None = None) -> str:
            received_schema.append(response_schema)
            return _valid_report_json()

        monkeypatch.setattr(report_generator, "_safe_generate", mock_generate)
        report = generate_report(_make_extraction(), _make_assessment())

        assert report is not None
        assert len(received_schema) == 1
        assert received_schema[0] == LLM_REPORT_OUTPUT_SCHEMA
        assert "overview" in received_schema[0]["properties"]
        assert "key_parties" in received_schema[0]["properties"]


class TestFailFastPermanentErrors:
    """Verify non-retryable errors fail immediately without looping."""

    def test_auth_error_fails_immediately_in_report_generator(self, monkeypatch: pytest.MonkeyPatch) -> None:
        call_count = 0

        def mock_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise GeminiConnectionError("Gemini API error: 401 Unauthorized - invalid API key")

        monkeypatch.setattr(report_generator, "generate", mock_generate)

        with pytest.raises(GeminiConnectionError):
            generate_report(_make_extraction(), _make_assessment())

        # Must fail fast on attempt 1 without spinning through max_retries
        assert call_count == 1

    def test_auth_error_fails_immediately_in_risk_classifier(self, monkeypatch: pytest.MonkeyPatch) -> None:
        call_count = 0

        def mock_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise GeminiConnectionError("Gemini API error: 401 Unauthorized - invalid API key")

        monkeypatch.setattr(llm_classifier, "generate", mock_generate)

        with pytest.raises(GeminiConnectionError):
            llm_classifier.classify_llm_risk_flags("doc-1", "Test Contract")

        assert call_count == 1


class TestRetryLimitEnforcement:
    """Verify bounded retry loop respects configured max_retries."""

    def test_report_generator_respects_max_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        call_count = 0

        def mock_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return "invalid json string"

        monkeypatch.setattr(report_generator, "generate", mock_generate)
        monkeypatch.setattr(report_generator, "get_settings", lambda: {"llm": {"max_json_retries": 2}})

        with pytest.raises(ReportValidationError):
            generate_report(_make_extraction(), _make_assessment())

        # max_json_retries=2 means exactly 3 attempts (1 initial + 2 retries)
        assert call_count == 3

    def test_risk_classifier_respects_max_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        call_count = 0

        def mock_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return "invalid json string"

        monkeypatch.setattr(llm_classifier, "generate", mock_generate)
        monkeypatch.setattr(llm_classifier, "get_settings", lambda: {"llm": {"max_json_retries": 2}})

        with pytest.raises(LLMClassificationError):
            llm_classifier.classify_llm_risk_flags("doc-1", "Test Contract")

        assert call_count == 3


class TestSecureSanitizedRetryLogging:
    """Verify retry logs record attempt and reason without exposing secrets."""

    def test_report_retry_logging_sanitizes_reason(self, monkeypatch: pytest.MonkeyPatch) -> None:
        attempts = iter([
            "bad json with api_key=mock-secret-never-log",
            _valid_report_json(),
        ])

        monkeypatch.setattr(report_generator, "generate", lambda *args, **kwargs: next(attempts))

        records: list[logging.LogRecord] = []
        class TestHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        handler = TestHandler()
        logger = logging.getLogger("contract_platform")
        logger.addHandler(handler)
        try:
            report = generate_report(_make_extraction(), _make_assessment())
        finally:
            logger.removeHandler(handler)

        assert report is not None
        retry_logs = [r.getMessage() for r in records if "[LLM_RETRY]" in r.getMessage()]
        assert len(retry_logs) == 1
        assert "operation=report_generation" in retry_logs[0]
        assert "attempt=1/" in retry_logs[0]
        full_text = " ".join(r.getMessage() for r in records)
        assert "mock-secret-never-log" not in full_text


class TestGeminiProviderErrorSanitization:
    """Verify GeminiProvider sanitizes sensitive tokens and classifies permanent errors."""

    def test_gemini_sanitizes_api_key_in_errors(self) -> None:
        from app.llm.gemini_provider import _sanitize_error_message
        raw_error = "API call failed: https://generativelanguage.googleapis.com/v1beta/models?api_key=mock-secret-never-log"
        sanitized = _sanitize_error_message(raw_error)
        assert "mock-secret-never-log" not in sanitized
        assert "[REDACTED_SECRET]" in sanitized

    def test_gemini_raises_permanent_error_on_auth_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from app.llm.exceptions import GeminiPermanentError
        from app.llm.gemini_provider import GeminiProvider

        monkeypatch.setenv("TEST_KEY_ENV", "dummy_val")
        provider = GeminiProvider(settings={
            "llm": {"gemini": {"api_key_env": "TEST_KEY_ENV"}}
        })

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception(
            "API_KEY_INVALID: User not authorized"
        )

        # Patch the SDK client actually imported by GeminiProvider, so this
        # test cannot make a live request and is independent of import order.
        monkeypatch.setattr("google.genai.Client", lambda **kwargs: mock_client)

        with pytest.raises(GeminiPermanentError) as exc_info:
            provider.generate("Test prompt")
        assert "permanent" in str(exc_info.value).lower()
        mock_client.models.generate_content.assert_called_once()
