"""
nli_based.py

NLI-based semantic contradiction detection for clause text pairs. Used for
field types whose values are free prose that cannot be reliably compared
via simple numeric/date rules (TERMINATION_CLAUSE, CONFIDENTIALITY_CLAUSE,
LIABILITY_TERMS, OBLIGATIONS) and also as a second-pass fallback for
PAYMENT_TERMS / RENEWAL_TERMS when the rule-based checker found no numeric
mismatch but the clauses might still conflict semantically.

Calls app.groundedness.nli_checker.check_contradiction (bidirectional NLI),
and returns a FieldInconsistency when the contradiction score clears the
threshold in settings.yaml (groundedness.contradiction_threshold).
"""

from __future__ import annotations

from app.groundedness.nli_checker import check_contradiction
from app.models.comparison import FieldInconsistency, MismatchDetectionMethod
from app.models.extraction import ExtractedField
from app.utils.config_loader import get_settings
from app.utils.logging_setup import get_logger

logger = get_logger(__name__)


def check_nli_contradiction(
    field_a: ExtractedField,
    field_b: ExtractedField,
) -> FieldInconsistency | None:
    """
    Run bidirectional NLI contradiction detection on a pair of aligned
    clause-text fields. Returns a FieldInconsistency when the contradiction
    probability clears groundedness.contradiction_threshold in settings.yaml,
    otherwise None.

    Args:
        field_a: Extracted field from Contract A.
        field_b: Extracted field from Contract B (same field_type as field_a).

    Returns:
        A FieldInconsistency with detection_method=NLI_SEMANTIC_CONTRADICTION,
        or None if no semantic contradiction was detected above threshold.
    """
    settings = get_settings()
    threshold: float = settings.get("groundedness", {}).get("contradiction_threshold", 0.5)

    contradiction_score = check_contradiction(field_a.value_text, field_b.value_text)

    logger.debug(
        "NLI contradiction score for %s: %.3f (threshold=%.2f)",
        field_a.field_type.value,
        contradiction_score,
        threshold,
    )

    if contradiction_score < threshold:
        return None

    return FieldInconsistency(
        field_type=field_a.field_type,
        detection_method=MismatchDetectionMethod.NLI_SEMANTIC_CONTRADICTION,
        contract_a_name=field_a.contract_name,
        contract_a_page=field_a.page_number,
        contract_a_text=field_a.value_text,
        contract_b_name=field_b.contract_name,
        contract_b_page=field_b.page_number,
        contract_b_text=field_b.value_text,
        explanation=(
            f"NLI semantic contradiction detected on {field_a.field_type.value} "
            f"(score={contradiction_score:.3f}, threshold={threshold:.2f})."
        ),
        confidence=contradiction_score,
    )
