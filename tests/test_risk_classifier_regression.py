"""
test_risk_classifier_regression.py

Regression tests for the runtime bug:

    flags.1.explanation
    Value error, explanation must not be empty

Scenario: Ollama returns a valid three-flag JSON response but one flag has
an empty explanation string (""). The old code raised LLMClassificationError
for the whole request. The fixed code:
  1. Detects the validation error.
  2. Retries once with a repair prompt that names the empty-explanation issue.
  3. If the retry still contains an invalid flag, discards ONLY that flag and
     returns the remaining valid ones.
  4. Never crashes the Risk Analysis page.

Every test in this file is derived from the exact production failure path.
"""

from __future__ import annotations

import json

import pytest

from app.models.risk import RiskCategory, RiskFlagSource
from app.risk import llm_classifier
from app.risk.exceptions import LLMClassificationError


# ---------------------------------------------------------------------------
# Helpers — build the exact Ollama responses that triggered the bug
# ---------------------------------------------------------------------------

def _three_flags_one_empty_explanation(empty_index: int = 1) -> str:
    """Three-flag response where one flag has explanation='' (the runtime bug)."""
    categories = [
        "unlimited_indemnity_language",
        "high_liability_exposure",
        "ambiguous_payment_terms",
    ]
    explanations = [
        "No uncapped indemnity language was detected in the retrieved passages.",
        "Liability exposure appears proportionate.",
        "Payment terms specify net 30 days from invoice.",
    ]
    explanations[empty_index] = ""   # ← the bug
    flags = [
        {
            "category": categories[i],
            "triggered": False,
            "explanation": explanations[i],
            "source_contract": "Acme Services Agreement",
            "source_page": 2,
            "supporting_text": "",
        }
        for i in range(3)
    ]
    return json.dumps({"flags": flags})


def _three_flags_all_valid() -> str:
    """Three-flag response where every explanation is non-empty (the fixed response)."""
    return json.dumps({
        "flags": [
            {
                "category": "unlimited_indemnity_language",
                "triggered": False,
                "explanation": "No unlimited indemnity language was found in the retrieved passages.",
                "source_contract": "Acme Services Agreement",
                "source_page": 2,
                "supporting_text": "",
            },
            {
                "category": "high_liability_exposure",
                "triggered": False,
                "explanation": "Liability is capped at total fees paid.",
                "source_contract": "Acme Services Agreement",
                "source_page": 3,
                "supporting_text": "liability shall not exceed total fees paid.",
            },
            {
                "category": "ambiguous_payment_terms",
                "triggered": False,
                "explanation": "Payment terms clearly state net 30 days from invoice.",
                "source_contract": "Acme Services Agreement",
                "source_page": 1,
                "supporting_text": "within 30 days of invoice receipt.",
            },
        ]
    })


@pytest.fixture(autouse=True)
def stub_retrieval(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(llm_classifier, "embed_query", lambda q: [0.1] * 8)
    monkeypatch.setattr(llm_classifier, "query_similar", lambda vec, top_k, document_id: [])


# ---------------------------------------------------------------------------
# Regression test 1 — exact production failure path
# ---------------------------------------------------------------------------

class TestEmptyExplanationBug:
    def test_flags_1_empty_explanation_is_discarded_not_crashed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Reproduces the exact runtime error:
            flags.1.explanation — Value error, explanation must not be empty

        The response returned by Ollama has flags[1].explanation = "".
        Expected: flags[1] is discarded; the other two flags are returned.
        The Risk page must not crash (no LLMClassificationError raised).
        """
        # All generate() calls return the same buggy response (both initial
        # attempt and every retry exhaust with the same empty explanation).
        monkeypatch.setattr(
            llm_classifier, "generate",
            lambda prompt: _three_flags_one_empty_explanation(empty_index=1),
        )

        flags = llm_classifier.classify_llm_risk_flags("doc-acme", "Acme Services Agreement")

        # Two valid flags returned, not three.
        assert len(flags) == 2, (
            f"Expected 2 valid flags after discarding the empty-explanation flag, got {len(flags)}"
        )
        categories = {f.category for f in flags}
        # flags[0] and flags[2] survive.
        assert RiskCategory.UNLIMITED_INDEMNITY_LANGUAGE in categories
        assert RiskCategory.AMBIGUOUS_PAYMENT_TERMS in categories
        # The invalid flag (flags[1]) is discarded.
        assert RiskCategory.HIGH_LIABILITY_EXPOSURE not in categories

    def test_flags_1_empty_explanation_all_returned_flags_have_llm_source(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            llm_classifier, "generate",
            lambda prompt: _three_flags_one_empty_explanation(empty_index=1),
        )
        flags = llm_classifier.classify_llm_risk_flags("doc-acme", "Acme Services Agreement")
        assert all(f.source == RiskFlagSource.LLM_CLASSIFIED for f in flags)

    def test_flags_1_empty_explanation_returned_flags_have_valid_explanations(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every flag in the returned list must pass the non-empty explanation validator."""
        monkeypatch.setattr(
            llm_classifier, "generate",
            lambda prompt: _three_flags_one_empty_explanation(empty_index=1),
        )
        flags = llm_classifier.classify_llm_risk_flags("doc-acme", "Acme Services Agreement")
        for flag in flags:
            assert flag.explanation and flag.explanation.strip(), (
                f"Returned flag {flag.category} has empty explanation — validator bypass detected"
            )

    def test_empty_explanation_on_first_flag_discards_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Bug is not position-specific — flags[0] empty explanation should also be handled."""
        monkeypatch.setattr(
            llm_classifier, "generate",
            lambda prompt: _three_flags_one_empty_explanation(empty_index=0),
        )
        flags = llm_classifier.classify_llm_risk_flags("doc-acme", "Acme Services Agreement")
        assert len(flags) == 2
        cats = {f.category for f in flags}
        assert RiskCategory.HIGH_LIABILITY_EXPOSURE in cats
        assert RiskCategory.AMBIGUOUS_PAYMENT_TERMS in cats
        assert RiskCategory.UNLIMITED_INDEMNITY_LANGUAGE not in cats

    def test_empty_explanation_on_last_flag_discards_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            llm_classifier, "generate",
            lambda prompt: _three_flags_one_empty_explanation(empty_index=2),
        )
        flags = llm_classifier.classify_llm_risk_flags("doc-acme", "Acme Services Agreement")
        assert len(flags) == 2
        cats = {f.category for f in flags}
        assert RiskCategory.UNLIMITED_INDEMNITY_LANGUAGE in cats
        assert RiskCategory.HIGH_LIABILITY_EXPOSURE in cats
        assert RiskCategory.AMBIGUOUS_PAYMENT_TERMS not in cats


# ---------------------------------------------------------------------------
# Regression test 2 — retry succeeds, returning all three flags
# ---------------------------------------------------------------------------

class TestSuccessfulRetryAfterEmptyExplanation:
    def test_retry_returns_all_three_flags_when_fixed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        First attempt has the bug (empty explanation); retry returns a valid
        response. All three flags should be returned from the successful retry.
        """
        responses = iter([
            _three_flags_one_empty_explanation(empty_index=1),  # fails validation
            _three_flags_all_valid(),                            # succeeds
        ])
        monkeypatch.setattr(llm_classifier, "generate", lambda prompt: next(responses))

        flags = llm_classifier.classify_llm_risk_flags("doc-acme", "Acme Services Agreement")

        assert len(flags) == 3
        cats = {f.category for f in flags}
        assert cats == {
            RiskCategory.UNLIMITED_INDEMNITY_LANGUAGE,
            RiskCategory.HIGH_LIABILITY_EXPOSURE,
            RiskCategory.AMBIGUOUS_PAYMENT_TERMS,
        }


# ---------------------------------------------------------------------------
# Regression test 3 — repair prompt targets the empty-explanation error
# ---------------------------------------------------------------------------

class TestRepairPromptContent:
    def test_retry_prompt_mentions_empty_explanation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        When the validation error is an empty explanation, the retry prompt
        must include targeted language about non-empty explanations so the
        LLM can self-correct.
        """
        prompts_received: list[str] = []

        def capture(prompt: str) -> str:
            prompts_received.append(prompt)
            return _three_flags_one_empty_explanation(empty_index=1)

        monkeypatch.setattr(llm_classifier, "generate", capture)

        # Exhaust retries so we observe all prompts sent.
        llm_classifier.classify_llm_risk_flags("doc-acme", "Acme Services Agreement")

        assert len(prompts_received) >= 2, "Expected at least one retry prompt"

        repair_prompt = prompts_received[1]
        # The repair prompt must call out the empty-explanation problem specifically.
        assert any(
            term in repair_prompt.lower()
            for term in ("empty", "explanation", "non-empty")
        ), (
            "Repair prompt does not mention the empty-explanation problem. "
            f"Repair prompt was:\n{repair_prompt[:500]}"
        )

    def test_pydantic_validator_itself_is_unchanged(self) -> None:
        """
        The _LLMFlagItem validator must still reject an empty explanation.
        This verifies the validator was not weakened as part of the fix.
        """
        from pydantic import ValidationError as PydanticValidationError

        with pytest.raises(PydanticValidationError) as exc_info:
            llm_classifier._LLMFlagItem(
                category=RiskCategory.HIGH_LIABILITY_EXPOSURE,
                triggered=False,
                explanation="",   # must be rejected
            )

        errors = exc_info.value.errors()
        assert any(
            "explanation" in str(e.get("loc", "")) and "empty" in e.get("msg", "").lower()
            for e in errors
        ), f"Expected empty-explanation error, got: {errors}"


# ---------------------------------------------------------------------------
# Regression test 4 — all flags invalid → LLMClassificationError raised
# ---------------------------------------------------------------------------

class TestAllFlagsInvalidRaisesError:
    def test_all_empty_explanations_raises_after_retries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        If every single flag has an invalid explanation, permissive parsing
        yields zero valid flags and LLMClassificationError must be raised.
        """
        all_empty = json.dumps({
            "flags": [
                {"category": "unlimited_indemnity_language",
                 "triggered": False, "explanation": ""},
                {"category": "high_liability_exposure",
                 "triggered": False, "explanation": "   "},  # whitespace-only
                {"category": "ambiguous_payment_terms",
                 "triggered": False, "explanation": ""},
            ]
        })
        monkeypatch.setattr(llm_classifier, "generate", lambda prompt: all_empty)

        with pytest.raises(LLMClassificationError):
            llm_classifier.classify_llm_risk_flags("doc-acme", "Acme Services Agreement")
