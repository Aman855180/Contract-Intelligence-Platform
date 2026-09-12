"""
nli_checker.py

NLI-based groundedness verification: checks whether a generated claim is
actually entailed by its cited source passage, using a dedicated Natural
Language Inference model (MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli per
config). This is explicitly NOT an LLM self-reported confidence score —
it's an independent classifier scoring (passage, claim) as
entailment/neutral/contradiction.

The model is loaded lazily and cached as a singleton, mirroring the
pattern in app/embeddings/embedder.py, so this module can be imported and
exercised in tests (via a monkeypatched _load_nli_model) without requiring
network access to download model weights.
"""

from __future__ import annotations

from typing import Any, Protocol

from app.groundedness.exceptions import GroundednessError
from app.models.retrieval import GroundednessVerdict
from app.utils.config_loader import get_settings
from app.utils.logging_setup import get_logger

logger = get_logger(__name__)

_model_instance: Any = None


class NLIModel(Protocol):
    """Structural type for whatever _load_nli_model() returns."""

    def classify(self, premise: str, hypothesis: str) -> dict[str, float]:
        """Return a dict mapping lowercase label names ('entailment',
        'neutral', 'contradiction') to probabilities summing to ~1.0."""
        ...


class _HuggingFaceNLIModel:
    """Wraps a transformers AutoModelForSequenceClassification NLI checkpoint."""

    def __init__(self, model_name: str) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        # Build a case-insensitive label -> index map so we don't hardcode an
        # assumed label ordering (different NLI checkpoints order labels
        # differently in their config).
        self.label_to_index: dict[str, int] = {
            label.lower(): idx for idx, label in self.model.config.id2label.items()
        }

    def classify(self, premise: str, hypothesis: str) -> dict[str, float]:
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
        with self._torch.no_grad():
            logits = self.model(**inputs).logits[0]
            probs = self._torch.softmax(logits, dim=-1).tolist()

        result: dict[str, float] = {}
        for label, idx in self.label_to_index.items():
            result[label] = probs[idx]
        return result


def _load_nli_model() -> NLIModel:
    """
    Load (or return the cached) NLI model for groundedness checking.

    Returns:
        A loaded NLI model implementing the .classify() protocol.

    Raises:
        GroundednessError: If the model fails to load.
    """
    global _model_instance
    if _model_instance is not None:
        return _model_instance

    settings = get_settings()
    model_name = settings.get("groundedness", {}).get(
        "nli_model_name", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    )

    try:
        logger.info("Loading NLI groundedness model '%s'...", model_name)
        _model_instance = _HuggingFaceNLIModel(model_name)
        logger.info("NLI model '%s' loaded successfully.", model_name)
    except Exception as exc:
        logger.error("Failed to load NLI model '%s': %s", model_name, exc)
        raise GroundednessError(f"Could not load NLI model '{model_name}': {exc}") from exc

    return _model_instance


def reset_model_cache() -> None:
    """Clear the cached NLI model instance. Used by tests to inject stub models."""
    global _model_instance
    _model_instance = None


def check_groundedness(claim: str, source_passage: str) -> GroundednessVerdict:
    """
    Check whether a generated claim is entailed by its cited source passage,
    using the configured NLI model (premise=source_passage,
    hypothesis=claim).

    Args:
        claim: The generated claim/answer text to verify.
        source_passage: The cited source text the claim should be grounded in.

    Returns:
        A GroundednessVerdict with entailment/contradiction/neutral scores
        and an is_grounded flag based on groundedness.entailment_threshold
        in settings.yaml.

    Raises:
        GroundednessError: If the NLI model fails to load or run inference.
    """
    settings = get_settings()
    threshold = settings.get("groundedness", {}).get("entailment_threshold", 0.5)

    model = _load_nli_model()
    try:
        scores = model.classify(premise=source_passage, hypothesis=claim)
    except Exception as exc:
        logger.error("NLI classification failed for claim '%s': %s", claim, exc)
        raise GroundednessError(f"NLI classification failed: {exc}") from exc

    entailment_score = scores.get("entailment", 0.0)
    contradiction_score = scores.get("contradiction", 0.0)
    neutral_score = scores.get("neutral", 0.0)

    return GroundednessVerdict(
        claim=claim,
        source_passage=source_passage,
        entailment_score=entailment_score,
        contradiction_score=contradiction_score,
        neutral_score=neutral_score,
        is_grounded=entailment_score >= threshold,
    )


def check_contradiction(text_a: str, text_b: str) -> float:
    """
    Check whether two clause texts semantically contradict each other,
    using the configured NLI model. Used by Phase 7's contract comparison
    for clause pairs that aren't simple rule-based numeric/date mismatches
    (e.g. "Acme shall maintain insurance" vs "Acme is not required to
    maintain insurance").

    NLI entailment/contradiction is directional (premise -> hypothesis), but
    semantic contradiction between two clauses is conceptually symmetric, so
    this runs the check in both directions and returns the higher
    contradiction score — a contradiction found in either direction counts.

    Args:
        text_a: The first clause text (e.g. from Contract A).
        text_b: The second clause text (e.g. from Contract B).

    Returns:
        The maximum contradiction probability (0.0-1.0) across both
        (text_a -> text_b) and (text_b -> text_a) directions.

    Raises:
        GroundednessError: If the NLI model fails to load or run inference.
    """
    model = _load_nli_model()
    try:
        scores_a_to_b = model.classify(premise=text_a, hypothesis=text_b)
        scores_b_to_a = model.classify(premise=text_b, hypothesis=text_a)
    except Exception as exc:
        logger.error("NLI contradiction check failed: %s", exc)
        raise GroundednessError(f"NLI contradiction check failed: {exc}") from exc

    return max(
        scores_a_to_b.get("contradiction", 0.0),
        scores_b_to_a.get("contradiction", 0.0),
    )
