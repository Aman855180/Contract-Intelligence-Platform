"""
test_report_generator.py

Tests for app/reports/report_generator.py. Stubs embed_query, query_similar,
and generate — exercises JSON parsing, Pydantic validation, retry logic,
context assembly helpers, and the document_id/contract_name injection.
"""

from __future__ import annotations

import json

import pytest

from app.models.extraction import ContractExtractionResult, ExtractedField, FieldType
from app.models.risk import RiskAssessment, RiskCategory, RiskFlag, RiskFlagSource
from app.reports import report_generator
from app.reports.exceptions import ReportValidationError


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_extraction(document_id: str = "doc-1", contract_name: str = "Test Contract") -> ContractExtractionResult:
    return ContractExtractionResult(
        document_id=document_id,
        contract_name=contract_name,
        fields=[
            ExtractedField(field_type=FieldType.PAYMENT_TERMS, value_text="Net 30 days.",
                           document_id=document_id, contract_name=contract_name,
                           page_number=2, confidence=0.9),
        ],
    )


def _make_assessment(document_id: str = "doc-1", contract_name: str = "Test Contract") -> RiskAssessment:
    return RiskAssessment(
        document_id=document_id,
        contract_name=contract_name,
        flags=[
            RiskFlag(category=RiskCategory.MISSING_SIGNATURE_FIELD,
                     source=RiskFlagSource.RULE_BASED, triggered=True, weight=25,
                     explanation="No signature block detected."),
        ],
        risk_score=25,
    )


def _valid_report_json(contract_name: str = "Test Contract") -> str:
    return json.dumps({
        "overview": "This is a services agreement between Acme Corp and Globex Inc.",
        "key_parties": [
            {"name": "Acme Corp", "role": "Client",
             "source_contract": contract_name, "source_page": 1},
        ],
        "important_dates": [
            {"label": "Effective Date", "date": "January 1, 2025",
             "source_contract": contract_name, "source_page": 1},
        ],
        "payment_terms": {
            "summary": "Net 30 days from invoice.",
            "source_contract": contract_name, "source_page": 2,
        },
        "risk_flags": [
            {"category": "missing_signature_field", "severity": "high",
             "explanation": "No signature block detected.",
             "source_contract": contract_name, "source_page": 1},
        ],
        "key_clauses": [
            {"clause_type": "termination", "summary": "60 days notice required.",
             "source_contract": contract_name, "source_page": 3},
        ],
        "recommendations": ["Add signature block.", "Clarify payment schedule."],
    })


@pytest.fixture(autouse=True)
def stub_retrieval(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(report_generator, "embed_query", lambda q: [0.1] * 8)
    monkeypatch.setattr(report_generator, "query_similar", lambda vec, top_k, document_id: [])


# ---------------------------------------------------------------------------
# Core generation tests
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def test_valid_response_returns_report_schema(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(report_generator, "generate", lambda p: _valid_report_json())
        report = report_generator.generate_report(_make_extraction(), _make_assessment())
        assert report.contract_name == "Test Contract"
        assert report.document_id == "doc-1"
        assert report.overview != ""

    def test_document_id_and_contract_name_injected_not_from_llm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # LLM output that tries to set these (they'd be ignored anyway since
        # _LLMReportOutput doesn't have those fields)
        monkeypatch.setattr(report_generator, "generate", lambda p: _valid_report_json())
        report = report_generator.generate_report(
            _make_extraction(document_id="my-doc-42", contract_name="My Contract"),
            _make_assessment(document_id="my-doc-42", contract_name="My Contract"),
        )
        assert report.document_id == "my-doc-42"
        assert report.contract_name == "My Contract"

    def test_key_parties_populated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(report_generator, "generate", lambda p: _valid_report_json())
        report = report_generator.generate_report(_make_extraction(), _make_assessment())
        assert len(report.key_parties) == 1
        assert report.key_parties[0].name == "Acme Corp"

    def test_risk_flags_populated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(report_generator, "generate", lambda p: _valid_report_json())
        report = report_generator.generate_report(_make_extraction(), _make_assessment())
        assert len(report.risk_flags) == 1
        assert report.risk_flags[0].severity == "high"

    def test_strips_json_markdown_fences(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fenced = f"```json\n{_valid_report_json()}\n```"
        monkeypatch.setattr(report_generator, "generate", lambda p: fenced)
        report = report_generator.generate_report(_make_extraction(), _make_assessment())
        assert report.overview != ""

    def test_recommendations_list_preserved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(report_generator, "generate", lambda p: _valid_report_json())
        report = report_generator.generate_report(_make_extraction(), _make_assessment())
        assert len(report.recommendations) == 2
        assert "signature" in report.recommendations[0].lower()


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestReportValidation:
    def test_missing_required_field_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        bad = json.loads(_valid_report_json())
        del bad["overview"]
        monkeypatch.setattr(report_generator, "generate", lambda p: json.dumps(bad))
        with pytest.raises(ReportValidationError):
            report_generator.generate_report(_make_extraction(), _make_assessment())

    def test_incorrect_field_type_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        bad = json.loads(_valid_report_json())
        bad["key_parties"][0]["role"] = 7
        monkeypatch.setattr(report_generator, "generate", lambda p: json.dumps(bad))
        with pytest.raises(ReportValidationError):
            report_generator.generate_report(_make_extraction(), _make_assessment())

    def test_invalid_severity_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        bad = json.loads(_valid_report_json())
        bad["risk_flags"][0]["severity"] = "critical"  # not in low|medium|high
        monkeypatch.setattr(report_generator, "generate", lambda p: json.dumps(bad))
        with pytest.raises(ReportValidationError):
            report_generator.generate_report(_make_extraction(), _make_assessment())

    def test_empty_overview_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        bad = json.loads(_valid_report_json())
        bad["overview"] = ""
        monkeypatch.setattr(report_generator, "generate", lambda p: json.dumps(bad))
        with pytest.raises(ReportValidationError):
            report_generator.generate_report(_make_extraction(), _make_assessment())

    def test_empty_party_name_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        bad = json.loads(_valid_report_json())
        bad["key_parties"][0]["name"] = "   "
        monkeypatch.setattr(report_generator, "generate", lambda p: json.dumps(bad))
        with pytest.raises(ReportValidationError):
            report_generator.generate_report(_make_extraction(), _make_assessment())

    def test_missing_payment_terms_key_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        bad = json.loads(_valid_report_json())
        del bad["payment_terms"]
        monkeypatch.setattr(report_generator, "generate", lambda p: json.dumps(bad))
        with pytest.raises(ReportValidationError):
            report_generator.generate_report(_make_extraction(), _make_assessment())


# ---------------------------------------------------------------------------
# Retry tests
# ---------------------------------------------------------------------------

class TestRetryLogic:
    def test_retries_on_invalid_json_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        responses = iter(["not json !!!", _valid_report_json()])
        monkeypatch.setattr(report_generator, "generate", lambda p: next(responses))
        report = report_generator.generate_report(_make_extraction(), _make_assessment())
        assert report is not None

    def test_retries_twice_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        responses = iter(["bad", "also bad", _valid_report_json()])
        monkeypatch.setattr(report_generator, "generate", lambda p: next(responses))
        report = report_generator.generate_report(_make_extraction(), _make_assessment())
        assert report.document_id == "doc-1"

    def test_exhausted_retries_raises_report_validation_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(report_generator, "generate", lambda p: "not json at all")
        with pytest.raises(ReportValidationError):
            report_generator.generate_report(_make_extraction(), _make_assessment())

    def test_retry_prompt_contains_error_context(self, monkeypatch: pytest.MonkeyPatch) -> None:
        prompts: list[str] = []

        def capture(p: str) -> str:
            prompts.append(p)
            return "bad json"

        monkeypatch.setattr(report_generator, "generate", capture)
        with pytest.raises(ReportValidationError):
            report_generator.generate_report(_make_extraction(), _make_assessment())

        assert len(prompts) >= 2
        assert "invalid" in prompts[1].lower() or "error" in prompts[1].lower()

    def test_prompt_declares_report_contract_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        prompts: list[str] = []

        def capture(prompt: str, **kwargs) -> str:
            prompts.append(prompt)
            return _valid_report_json()

        monkeypatch.setattr(report_generator, "generate", capture)
        report_generator.generate_report(_make_extraction(), _make_assessment())
        prompt = prompts[0]
        for field in ("overview", "key_parties", "role", "important_dates", "payment_terms",
                      "clause_type", "summary", "source_contract", "source_page", "recommendations"):
            assert field in prompt
        assert "Do not add fields such as contract_name" in prompt

    def test_gemini_schema_matches_required_pydantic_fields(self) -> None:
        expected = set(report_generator._LLMReportOutput.model_fields)
        schema = report_generator.LLM_REPORT_OUTPUT_SCHEMA
        assert set(schema["properties"]) == expected
        assert set(schema["required"]) == {
            name for name, field in report_generator._LLMReportOutput.model_fields.items()
            if field.is_required()
        }


# ---------------------------------------------------------------------------
# Context assembly helpers
# ---------------------------------------------------------------------------

class TestContextHelpers:
    def test_format_extracted_fields_includes_field_type(self) -> None:
        extraction = _make_extraction()
        text = report_generator._format_extracted_fields(extraction)
        assert "Payment Terms" in text
        assert "Net 30 days." in text

    def test_format_extracted_fields_empty_returns_placeholder(self) -> None:
        extraction = ContractExtractionResult(
            document_id="doc-1", contract_name="Empty", fields=[]
        )
        text = report_generator._format_extracted_fields(extraction)
        assert "No structured fields extracted." in text

    def test_format_triggered_risk_flags_only_triggered(self) -> None:
        assessment = RiskAssessment(
            document_id="doc-1", contract_name="Test",
            flags=[
                RiskFlag(category=RiskCategory.MISSING_SIGNATURE_FIELD,
                         source=RiskFlagSource.RULE_BASED, triggered=True,
                         weight=25, explanation="No sig."),
                RiskFlag(category=RiskCategory.AMBIGUOUS_PAYMENT_TERMS,
                         source=RiskFlagSource.LLM_CLASSIFIED, triggered=False,
                         weight=10, explanation="Payment clear."),
            ],
            risk_score=25,
        )
        text = report_generator._format_triggered_risk_flags(assessment)
        assert "missing_signature_field" in text
        assert "ambiguous_payment_terms" not in text

    def test_format_triggered_risk_flags_no_triggered_returns_empty_array(self) -> None:
        assessment = RiskAssessment(
            document_id="doc-1", contract_name="Test",
            flags=[
                RiskFlag(category=RiskCategory.MISSING_SIGNATURE_FIELD,
                         source=RiskFlagSource.RULE_BASED, triggered=False,
                         weight=25, explanation="Sig present."),
            ],
            risk_score=0,
        )
        text = report_generator._format_triggered_risk_flags(assessment)
        assert text == "[]"

    def test_severity_mapping_from_weight(self) -> None:
        assessment = RiskAssessment(
            document_id="doc-1", contract_name="Test",
            flags=[
                RiskFlag(category=RiskCategory.MISSING_SIGNATURE_FIELD,
                         source=RiskFlagSource.RULE_BASED, triggered=True,
                         weight=25, explanation="No sig."),  # weight>=20 -> high
                RiskFlag(category=RiskCategory.MISSING_CONFIDENTIALITY_CLAUSE,
                         source=RiskFlagSource.RULE_BASED, triggered=True,
                         weight=15, explanation="No conf."),  # weight>=15 -> medium
                RiskFlag(category=RiskCategory.AMBIGUOUS_PAYMENT_TERMS,
                         source=RiskFlagSource.LLM_CLASSIFIED, triggered=True,
                         weight=10, explanation="Ambiguous."),  # weight<15 -> low
            ],
            risk_score=50,
        )
        text = report_generator._format_triggered_risk_flags(assessment)
        parsed = json.loads(text)
        by_cat = {item["category"]: item["severity"] for item in parsed}
        assert by_cat["missing_signature_field"] == "high"
        assert by_cat["missing_confidentiality_clause"] == "medium"
        assert by_cat["ambiguous_payment_terms"] == "low"

    def test_generate_fallback_to_ollama_on_gemini_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from unittest.mock import MagicMock
        from app.llm.exceptions import GeminiConnectionError

        mock_gemini = MagicMock()
        mock_gemini.provider_name = "gemini"
        mock_gemini.generate.side_effect = GeminiConnectionError("Network down")

        mock_ollama = MagicMock()
        mock_ollama.provider_name = "ollama"
        mock_ollama.generate.return_value = _valid_report_json()

        def mock_get_report_provider(settings=None):
            return mock_gemini

        def mock_get_llm_provider(settings=None, provider_type=None):
            if provider_type == "ollama":
                return mock_ollama
            return mock_gemini

        monkeypatch.setattr(report_generator, "get_report_provider", mock_get_report_provider)
        monkeypatch.setattr(report_generator, "get_llm_provider", mock_get_llm_provider)
        monkeypatch.setattr(report_generator, "get_settings", lambda: {"llm": {"fallback_to_ollama": True}})
        monkeypatch.delenv("LLM_FALLBACK_TO_OLLAMA", raising=False)

        result = report_generator.generate("test prompt")
        assert result == _valid_report_json()
        mock_gemini.generate.assert_called_once()
        mock_ollama.generate.assert_called_once()

    def test_generate_propagates_gemini_error_when_fallback_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from unittest.mock import MagicMock
        from app.llm.exceptions import GeminiConnectionError

        gemini = MagicMock()
        gemini.provider_name = "gemini"
        gemini.generate.side_effect = GeminiConnectionError("Gemini request failed")
        ollama = MagicMock()
        monkeypatch.setattr(report_generator, "get_report_provider", lambda settings=None: gemini)
        monkeypatch.setattr(report_generator, "get_llm_provider", lambda settings=None, provider_type=None: ollama)
        monkeypatch.setattr(report_generator, "get_settings", lambda: {"llm": {"fallback_to_ollama": True}})
        monkeypatch.setenv("LLM_FALLBACK_TO_OLLAMA", "false")

        with pytest.raises(GeminiConnectionError, match="Gemini request failed"):
            report_generator.generate("test prompt")
        gemini.generate.assert_called_once()
        ollama.generate.assert_not_called()


# ---------------------------------------------------------------------------
# Deterministic risk flag assembly tests (Option A)
# ---------------------------------------------------------------------------

class TestDeterministicRiskFlags:
    def test_llm_json_without_risk_flags_populates_from_assessment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        json_no_risk = json.loads(_valid_report_json())
        del json_no_risk["risk_flags"]  # simulate model omitting risk_flags per new prompt
        monkeypatch.setattr(report_generator, "generate", lambda p: json.dumps(json_no_risk))

        assessment = _make_assessment()
        report = report_generator.generate_report(_make_extraction(), assessment)
        assert len(report.risk_flags) == 1
        assert report.risk_flags[0].category == "missing_signature_field"
        assert report.risk_flags[0].severity == "high"
        assert report.risk_flags[0].explanation == "No signature block detected."

    def test_empty_assessment_produces_empty_report_risk_flags(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(report_generator, "generate", lambda p: _valid_report_json())
        empty_assessment = RiskAssessment(
            document_id="doc-1",
            contract_name="Test Contract",
            flags=[
                RiskFlag(category=RiskCategory.MISSING_SIGNATURE_FIELD,
                         source=RiskFlagSource.RULE_BASED, triggered=False, weight=25,
                         explanation="Signature present."),
            ],
            risk_score=0,
        )
        report = report_generator.generate_report(_make_extraction(), empty_assessment)
        assert report.risk_flags == []

    def test_multiple_triggered_flags_correctly_mapped(self) -> None:
        assessment = RiskAssessment(
            document_id="doc-1",
            contract_name="Test Contract",
            flags=[
                RiskFlag(category=RiskCategory.UNLIMITED_INDEMNITY_LANGUAGE,
                         source=RiskFlagSource.LLM_CLASSIFIED, triggered=True, weight=25,
                         explanation="Uncapped indemnity.", source_page=2),
                RiskFlag(category=RiskCategory.MISSING_TERMINATION_CLAUSE,
                         source=RiskFlagSource.RULE_BASED, triggered=True, weight=15,
                         explanation="No termination clause.", source_page=1),
                RiskFlag(category=RiskCategory.AMBIGUOUS_PAYMENT_TERMS,
                         source=RiskFlagSource.LLM_CLASSIFIED, triggered=True, weight=10,
                         explanation="No milestone dates.", source_page=3),
                RiskFlag(category=RiskCategory.MISSING_CONFIDENTIALITY_CLAUSE,
                         source=RiskFlagSource.RULE_BASED, triggered=False, weight=15,
                         explanation="Confidentiality clause present."),
            ],
            risk_score=50,
        )
        report_flags = report_generator._build_report_risk_flags(assessment)
        assert len(report_flags) == 3
        by_cat = {f.category: f for f in report_flags}
        assert by_cat["unlimited_indemnity_language"].severity == "high"
        assert by_cat["unlimited_indemnity_language"].source_page == 2
        assert by_cat["missing_termination_clause"].severity == "medium"
        assert by_cat["ambiguous_payment_terms"].severity == "low"
