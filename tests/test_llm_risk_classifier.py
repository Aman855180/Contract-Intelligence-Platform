"""
test_llm_risk_classifier.py

Tests for app/risk/llm_classifier.py. Stubs out embed_query, query_similar,
and generate so no network or Ollama server is needed. Exercises JSON
parsing, Pydantic validation, retry-on-failure, and the mapping from
validated LLM output to RiskFlag objects.
"""

from __future__ import annotations

import json

import pytest

from app.models.risk import RiskCategory, RiskFlagSource
from app.risk import llm_classifier
from app.risk.exceptions import LLMClassificationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_llm_json(
    indemnity_triggered: bool = False,
    liability_triggered: bool = False,
    payment_triggered: bool = False,
) -> str:
    return json.dumps({
        "flags": [
            {
                "category": "unlimited_indemnity_language",
                "triggered": indemnity_triggered,
                "explanation": "Indemnity clause has no cap.",
                "source_contract": "Contract A",
                "source_page": 3,
                "supporting_text": "Indemnify against all claims.",
            },
            {
                "category": "high_liability_exposure",
                "triggered": liability_triggered,
                "explanation": "Liability terms expose disproportionate risk.",
                "source_contract": "Contract A",
                "source_page": 4,
                "supporting_text": "Liable for all consequential damages.",
            },
            {
                "category": "ambiguous_payment_terms",
                "triggered": payment_triggered,
                "explanation": "Payment schedule not clearly specified.",
                "source_contract": "Contract A",
                "source_page": 2,
                "supporting_text": "Payment due on reasonable notice.",
            },
        ]
    })


@pytest.fixture(autouse=True)
def stub_retrieval(monkeypatch: pytest.MonkeyPatch):
    """Stub embedding + ChromaDB calls so no vector store is needed."""
    monkeypatch.setattr(llm_classifier, "embed_query", lambda q: [0.1] * 8)
    monkeypatch.setattr(llm_classifier, "query_similar", lambda vec, top_k, document_id: [])


# ---------------------------------------------------------------------------
# Parsing and validation tests
# ---------------------------------------------------------------------------

class TestClassifyLLMRiskFlags:
    def test_valid_response_returns_three_flags(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(llm_classifier, "generate", lambda prompt: _valid_llm_json())
        flags = llm_classifier.classify_llm_risk_flags("doc-1", "Test Contract")
        assert len(flags) == 3

    def test_all_flags_have_llm_classified_source(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(llm_classifier, "generate", lambda prompt: _valid_llm_json())
        flags = llm_classifier.classify_llm_risk_flags("doc-1", "Test Contract")
        assert all(f.source == RiskFlagSource.LLM_CLASSIFIED for f in flags)

    def test_triggered_flags_correctly_mapped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            llm_classifier, "generate",
            lambda prompt: _valid_llm_json(indemnity_triggered=True, payment_triggered=True)
        )
        flags = llm_classifier.classify_llm_risk_flags("doc-1", "Test Contract")
        by_cat = {f.category: f for f in flags}
        assert by_cat[RiskCategory.UNLIMITED_INDEMNITY_LANGUAGE].triggered is True
        assert by_cat[RiskCategory.HIGH_LIABILITY_EXPOSURE].triggered is False
        assert by_cat[RiskCategory.AMBIGUOUS_PAYMENT_TERMS].triggered is True

    def test_weights_assigned_from_config_not_llm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(llm_classifier, "generate", lambda prompt: _valid_llm_json(indemnity_triggered=True))
        flags = llm_classifier.classify_llm_risk_flags("doc-1", "Test Contract")
        by_cat = {f.category: f for f in flags}
        # Weight for unlimited_indemnity_language is 20 per settings.yaml
        assert by_cat[RiskCategory.UNLIMITED_INDEMNITY_LANGUAGE].weight == 20
        assert by_cat[RiskCategory.HIGH_LIABILITY_EXPOSURE].weight == 15
        assert by_cat[RiskCategory.AMBIGUOUS_PAYMENT_TERMS].weight == 10

    def test_citation_fields_preserved_from_llm_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(llm_classifier, "generate", lambda prompt: _valid_llm_json(indemnity_triggered=True))
        flags = llm_classifier.classify_llm_risk_flags("doc-1", "Test Contract")
        by_cat = {f.category: f for f in flags}
        indemnity = by_cat[RiskCategory.UNLIMITED_INDEMNITY_LANGUAGE]
        assert indemnity.source_contract == "Contract A"
        assert indemnity.source_page == 3
        assert "Indemnify" in indemnity.supporting_text

    def test_strips_json_markdown_fences(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fenced = f"```json\n{_valid_llm_json()}\n```"
        monkeypatch.setattr(llm_classifier, "generate", lambda prompt: fenced)
        flags = llm_classifier.classify_llm_risk_flags("doc-1", "Test Contract")
        assert len(flags) == 3

    def test_strips_plain_code_fences(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fenced = f"```\n{_valid_llm_json()}\n```"
        monkeypatch.setattr(llm_classifier, "generate", lambda prompt: fenced)
        flags = llm_classifier.classify_llm_risk_flags("doc-1", "Test Contract")
        assert len(flags) == 3


# ---------------------------------------------------------------------------
# Retry logic tests
# ---------------------------------------------------------------------------

class TestRetryOnValidationFailure:
    def test_retries_on_invalid_json_and_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        responses = iter(["this is not json", _valid_llm_json()])
        monkeypatch.setattr(llm_classifier, "generate", lambda prompt: next(responses))
        flags = llm_classifier.classify_llm_risk_flags("doc-1", "Test Contract")
        assert len(flags) == 3

    def test_retries_on_pydantic_failure_and_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        bad_json = json.dumps({"flags": [{"category": "unknown_category", "triggered": True, "explanation": "x"}]})
        responses = iter([bad_json, bad_json, _valid_llm_json()])
        monkeypatch.setattr(llm_classifier, "generate", lambda prompt: next(responses))
        flags = llm_classifier.classify_llm_risk_flags("doc-1", "Test Contract")
        assert len(flags) == 3

    def test_exhausted_retries_raises_llm_classification_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(llm_classifier, "generate", lambda prompt: "not json at all !!!")
        with pytest.raises(LLMClassificationError):
            llm_classifier.classify_llm_risk_flags("doc-1", "Test Contract")

    def test_retry_prompt_includes_previous_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        prompts_received: list[str] = []

        def capture_and_fail(prompt: str) -> str:
            prompts_received.append(prompt)
            return "bad json"

        monkeypatch.setattr(llm_classifier, "generate", capture_and_fail)
        with pytest.raises(LLMClassificationError):
            llm_classifier.classify_llm_risk_flags("doc-1", "Test Contract")

        # After the first failed attempt, subsequent prompts must include error context.
        assert len(prompts_received) >= 2
        assert "invalid" in prompts_received[1].lower() or "error" in prompts_received[1].lower()

    def test_wrong_number_of_flags_returns_valid_subset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # LLM returns only 2 flags instead of 3.
        # Full _LLMRiskOutput validation rejects the response (wrong count),
        # but permissive per-flag parsing salvages the 2 individually-valid flags.
        two_flags = json.dumps({
            "flags": [
                {"category": "unlimited_indemnity_language", "triggered": False, "explanation": "No unlimited indemnity found."},
                {"category": "high_liability_exposure", "triggered": False, "explanation": "Liability is capped at fees paid."},
            ]
        })
        monkeypatch.setattr(llm_classifier, "generate", lambda prompt: two_flags)
        flags = llm_classifier.classify_llm_risk_flags("doc-1", "Test Contract")
        assert len(flags) == 2
        categories = {f.category for f in flags}
        assert RiskCategory.UNLIMITED_INDEMNITY_LANGUAGE in categories
        assert RiskCategory.HIGH_LIABILITY_EXPOSURE in categories

    def test_missing_explanation_discards_invalid_returns_valid_flags(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Pydantic validator (unchanged) still rejects the empty-explanation flag.
        # Permissive fallback discards that flag and returns the other two.
        bad = json.dumps({
            "flags": [
                {"category": "unlimited_indemnity_language", "triggered": True, "explanation": ""},  # invalid
                {"category": "high_liability_exposure", "triggered": False, "explanation": "ok"},
                {"category": "ambiguous_payment_terms", "triggered": False, "explanation": "ok"},
            ]
        })
        monkeypatch.setattr(llm_classifier, "generate", lambda prompt: bad)
        flags = llm_classifier.classify_llm_risk_flags("doc-1", "Test Contract")
        # 2 valid flags returned; the empty-explanation flag is silently discarded
        assert len(flags) == 2
        cats = {f.category for f in flags}
        assert RiskCategory.HIGH_LIABILITY_EXPOSURE in cats
        assert RiskCategory.AMBIGUOUS_PAYMENT_TERMS in cats
        assert RiskCategory.UNLIMITED_INDEMNITY_LANGUAGE not in cats

    def test_generate_fallback_to_ollama_on_gemini_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from unittest.mock import MagicMock
        from app.llm.exceptions import GeminiConnectionError

        mock_gemini = MagicMock()
        mock_gemini.provider_name = "gemini"
        mock_gemini.generate.side_effect = GeminiConnectionError("Network down")

        mock_ollama = MagicMock()
        mock_ollama.provider_name = "ollama"
        mock_ollama.generate.return_value = _valid_llm_json()

        def mock_get_risk_provider(settings=None):
            return mock_gemini

        def mock_get_llm_provider(settings=None, provider_type=None):
            if provider_type == "ollama":
                return mock_ollama
            return mock_gemini

        monkeypatch.setattr(llm_classifier, "get_risk_provider", mock_get_risk_provider)
        monkeypatch.setattr(llm_classifier, "get_llm_provider", mock_get_llm_provider)

        result = llm_classifier.generate("test prompt")
        assert result == _valid_llm_json()
        mock_gemini.generate.assert_called_once()
        mock_ollama.generate.assert_called_once()

    def test_generate_does_not_fallback_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from unittest.mock import MagicMock
        from app.llm.exceptions import GeminiConnectionError

        gemini = MagicMock()
        gemini.provider_name = "gemini"
        gemini.generate.side_effect = GeminiConnectionError("Gemini unavailable")
        monkeypatch.setenv("LLM_FALLBACK_TO_OLLAMA", "false")
        monkeypatch.setattr(llm_classifier, "get_risk_provider", lambda settings=None: gemini)
        monkeypatch.setattr(
            llm_classifier,
            "get_llm_provider",
            lambda *args, **kwargs: pytest.fail("Ollama fallback must remain disabled"),
        )

        with pytest.raises(GeminiConnectionError, match="Gemini unavailable"):
            llm_classifier.generate("test prompt")


# ---------------------------------------------------------------------------
# Tests for Structured Risk Response Handling
# ---------------------------------------------------------------------------

class TestStructuredRiskResponseHandling:
    def test_valid_risk_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Valid response matching schema succeeds immediately on attempt 1."""
        valid_json = _valid_llm_json()
        calls: list[dict] = []

        def capture_generate(prompt: str, attempt: int | None = None, response_schema: Any | None = None) -> str:
            calls.append({"prompt": prompt, "schema": response_schema})
            return valid_json

        monkeypatch.setattr(llm_classifier, "generate", capture_generate)
        flags = llm_classifier.classify_llm_risk_flags("doc-1", "Test Contract")

        assert len(flags) == 3
        assert len(calls) == 1
        assert calls[0]["schema"] == llm_classifier.LLMRiskOutputSchema

    def test_empty_risk_response_triggers_retry_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Empty model response triggers validation error and retry before succeeding."""
        responses = iter(["", _valid_llm_json()])
        attempts_seen: list[int] = []

        def mock_generate(prompt: str, attempt: int | None = None, response_schema: Any | None = None) -> str:
            attempts_seen.append(len(attempts_seen) + 1)
            return next(responses)

        monkeypatch.setattr(llm_classifier, "generate", mock_generate)
        flags = llm_classifier.classify_llm_risk_flags("doc-1", "Test Contract")

        assert len(flags) == 3
        assert len(attempts_seen) == 2  # Attempt 1 failed on empty, attempt 2 succeeded

    def test_multiple_risk_flags_triggered(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Multiple risk flags triggered simultaneously are mapped and weighted correctly."""
        multi_triggered_json = _valid_llm_json(
            indemnity_triggered=True,
            liability_triggered=True,
            payment_triggered=True,
        )
        monkeypatch.setattr(llm_classifier, "generate", lambda prompt, **kwargs: multi_triggered_json)
        flags = llm_classifier.classify_llm_risk_flags("doc-1", "Test Contract")

        assert len(flags) == 3
        triggered_cats = {f.category for f in flags if f.triggered}
        assert triggered_cats == {
            RiskCategory.UNLIMITED_INDEMNITY_LANGUAGE,
            RiskCategory.HIGH_LIABILITY_EXPOSURE,
            RiskCategory.AMBIGUOUS_PAYMENT_TERMS,
        }
        total_weight = sum(f.weight for f in flags if f.triggered)
        assert total_weight == 20 + 15 + 10  # 45

    def test_malformed_model_response_handled_by_permissive_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Malformed/unexpected flag structure is salvaged by permissive per-flag fallback."""
        malformed_json = json.dumps({
            "flags": [
                {
                    "category": "unlimited_indemnity_language",
                    "triggered": True,
                    "explanation": "Valid indemnity flag.",
                },
                {
                    "category": "corrupted_nonexistent_category",
                    "triggered": "not_a_bool",  # malformed type
                    "explanation": 12345,
                },
            ]
        })
        monkeypatch.setattr(llm_classifier, "generate", lambda prompt, **kwargs: malformed_json)
        flags = llm_classifier.classify_llm_risk_flags("doc-1", "Test Contract")

        # The valid flag is salvaged, the malformed one discarded
        assert len(flags) == 1
        assert flags[0].category == RiskCategory.UNLIMITED_INDEMNITY_LANGUAGE
        assert flags[0].triggered is True
