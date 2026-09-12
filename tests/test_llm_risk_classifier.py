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
