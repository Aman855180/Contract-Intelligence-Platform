"""
test_nli_checker.py

Tests for app/groundedness/nli_checker.py. The real DeBERTa MNLI model
requires downloading weights from huggingface.co, unreachable in this
sandboxed environment, so tests monkeypatch the model loader with a
deterministic stub to verify the wrapper's contract: threshold logic,
label-agnostic score extraction, and error propagation.
"""

from __future__ import annotations

import pytest

from app.groundedness import nli_checker
from app.groundedness.exceptions import GroundednessError


class StubNLIModel:
    """Stub returning caller-controlled (entailment, neutral, contradiction) scores."""

    def __init__(self, entailment: float, neutral: float, contradiction: float) -> None:
        self._scores = {"entailment": entailment, "neutral": neutral, "contradiction": contradiction}
        self.last_call: tuple[str, str] | None = None

    def classify(self, premise: str, hypothesis: str) -> dict[str, float]:
        self.last_call = (premise, hypothesis)
        return self._scores


@pytest.fixture(autouse=True)
def reset_nli_cache():
    nli_checker.reset_model_cache()
    yield
    nli_checker.reset_model_cache()


class TestCheckGroundedness:
    def test_high_entailment_is_grounded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stub = StubNLIModel(entailment=0.92, neutral=0.05, contradiction=0.03)
        monkeypatch.setattr(nli_checker, "_load_nli_model", lambda: stub)

        verdict = nli_checker.check_groundedness(
            claim="Payment is due within 30 days.",
            source_passage="Payment shall be due within thirty (30) days of invoice.",
        )

        assert verdict.is_grounded is True
        assert verdict.entailment_score == 0.92
        assert verdict.contradiction_score == 0.03

    def test_low_entailment_is_not_grounded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stub = StubNLIModel(entailment=0.1, neutral=0.6, contradiction=0.3)
        monkeypatch.setattr(nli_checker, "_load_nli_model", lambda: stub)

        verdict = nli_checker.check_groundedness(
            claim="Liability is unlimited.",
            source_passage="Payment shall be due within thirty (30) days of invoice.",
        )

        assert verdict.is_grounded is False

    def test_threshold_boundary_uses_configured_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from app.utils import config_loader

        config_loader.get_settings.cache_clear()
        settings = config_loader.get_settings()
        settings["groundedness"]["entailment_threshold"] = 0.7

        stub = StubNLIModel(entailment=0.7, neutral=0.2, contradiction=0.1)
        monkeypatch.setattr(nli_checker, "_load_nli_model", lambda: stub)

        verdict = nli_checker.check_groundedness(claim="x", source_passage="y")
        assert verdict.is_grounded is True  # exactly at threshold counts as grounded

        config_loader.get_settings.cache_clear()

    def test_premise_is_source_passage_hypothesis_is_claim(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stub = StubNLIModel(entailment=0.5, neutral=0.3, contradiction=0.2)
        monkeypatch.setattr(nli_checker, "_load_nli_model", lambda: stub)

        nli_checker.check_groundedness(claim="The claim text.", source_passage="The source passage text.")

        assert stub.last_call == ("The source passage text.", "The claim text.")

    def test_model_load_failure_raises_groundedness_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def raise_error():
            raise GroundednessError("simulated load failure")

        monkeypatch.setattr(nli_checker, "_load_nli_model", raise_error)
        with pytest.raises(GroundednessError):
            nli_checker.check_groundedness(claim="x", source_passage="y")

    def test_classification_failure_raises_groundedness_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FailingModel:
            def classify(self, premise: str, hypothesis: str) -> dict[str, float]:
                raise RuntimeError("inference crashed")

        monkeypatch.setattr(nli_checker, "_load_nli_model", lambda: FailingModel())
        with pytest.raises(GroundednessError):
            nli_checker.check_groundedness(claim="x", source_passage="y")


class TestModelCaching:
    def test_load_nli_model_caches_singleton(self, monkeypatch: pytest.MonkeyPatch) -> None:
        load_count = {"n": 0}
        stub = StubNLIModel(entailment=0.8, neutral=0.1, contradiction=0.1)

        class _CountingFactory:
            def __call__(self, model_name: str, **kwargs):
                load_count["n"] += 1
                return stub

        monkeypatch.setattr(nli_checker, "_HuggingFaceNLIModel", _CountingFactory())

        nli_checker._load_nli_model()
        nli_checker._load_nli_model()
        nli_checker._load_nli_model()

        assert load_count["n"] == 1


class TestCheckContradiction:
    def test_high_contradiction_score_returned(self, monkeypatch: pytest.MonkeyPatch) -> None:
        call_log: list[tuple[str, str]] = []

        class DirectionalStub:
            def classify(self, premise: str, hypothesis: str) -> dict[str, float]:
                call_log.append((premise, hypothesis))
                # Simulate strong A->B contradiction
                if premise == "text_a":
                    return {"entailment": 0.05, "neutral": 0.10, "contradiction": 0.85}
                return {"entailment": 0.50, "neutral": 0.30, "contradiction": 0.20}

        monkeypatch.setattr(nli_checker, "_load_nli_model", lambda: DirectionalStub())

        score = nli_checker.check_contradiction("text_a", "text_b")

        # Returns the MAX of both directions — should pick the 0.85 from A->B
        assert score == pytest.approx(0.85)
        # Both directions must have been checked
        assert len(call_log) == 2

    def test_bidirectional_both_directions_checked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        call_log: list[tuple[str, str]] = []

        class LoggingStub:
            def classify(self, premise: str, hypothesis: str) -> dict[str, float]:
                call_log.append((premise, hypothesis))
                return {"entailment": 0.3, "neutral": 0.4, "contradiction": 0.3}

        monkeypatch.setattr(nli_checker, "_load_nli_model", lambda: LoggingStub())
        nli_checker.check_contradiction("clause_a", "clause_b")

        premises = [p for p, _ in call_log]
        assert "clause_a" in premises
        assert "clause_b" in premises

    def test_returns_max_of_both_directions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # B->A direction has the higher contradiction score
        class AsymmetricStub:
            def classify(self, premise: str, hypothesis: str) -> dict[str, float]:
                if premise == "b":
                    return {"entailment": 0.1, "neutral": 0.1, "contradiction": 0.8}
                return {"entailment": 0.5, "neutral": 0.3, "contradiction": 0.2}

        monkeypatch.setattr(nli_checker, "_load_nli_model", lambda: AsymmetricStub())
        score = nli_checker.check_contradiction("a", "b")
        assert score == pytest.approx(0.8)
