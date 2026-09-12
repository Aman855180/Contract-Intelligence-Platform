"""
rule_based_flags.py

Deterministic rule-based risk flags based solely on the presence or
absence of extracted structured fields from Phase 6. No LLM, no NLI.

Covers exactly the three rule-based categories from the project spec:
  - missing_signature_field
  - missing_termination_clause
  - missing_confidentiality_clause

Each flag's weight is read from config/settings.yaml (risk_scoring.weights)
so it is never hardcoded here.
"""

from __future__ import annotations

from app.models.extraction import ContractExtractionResult, FieldType
from app.models.risk import RiskCategory, RiskFlag, RiskFlagSource
from app.utils.config_loader import get_settings
from app.utils.logging_setup import get_logger

logger = get_logger(__name__)

# Maps each rule-based RiskCategory to the FieldType whose *absence* triggers it.
_ABSENCE_RULES: list[tuple[RiskCategory, FieldType]] = [
    (RiskCategory.MISSING_SIGNATURE_FIELD, FieldType.SIGNATURE),
    (RiskCategory.MISSING_TERMINATION_CLAUSE, FieldType.TERMINATION_CLAUSE),
    (RiskCategory.MISSING_CONFIDENTIALITY_CLAUSE, FieldType.CONFIDENTIALITY_CLAUSE),
]


def _get_weight(category: RiskCategory) -> int:
    """Look up this category's weight from settings.yaml."""
    weights: dict[str, int] = get_settings().get("risk_scoring", {}).get("weights", {})
    return weights.get(category.value, 0)


def detect_rule_based_flags(extraction: ContractExtractionResult) -> list[RiskFlag]:
    """
    Evaluate the three deterministic clause-absence rules against the
    extraction result for a single contract.

    Args:
        extraction: The structured field extraction result for the contract.

    Returns:
        A list of three RiskFlag objects (one per rule-based category),
        each with triggered=True/False and the corresponding weight from
        settings.yaml. Always returns exactly three flags — one per category —
        regardless of how many are triggered, so downstream consumers
        receive a complete, predictable set.
    """
    flags: list[RiskFlag] = []

    for category, field_type in _ABSENCE_RULES:
        present = bool(extraction.get_fields_by_type(field_type))
        triggered = not present
        weight = _get_weight(category)

        if triggered:
            explanation = (
                f"No {field_type.value.replace('_', ' ')} was detected in the contract. "
                "This is a structural risk flag that requires manual review."
            )
        else:
            explanation = (
                f"{field_type.value.replace('_', ' ').title()} is present in the contract."
            )

        flags.append(
            RiskFlag(
                category=category,
                source=RiskFlagSource.RULE_BASED,
                triggered=triggered,
                weight=weight,
                explanation=explanation,
            )
        )
        logger.debug(
            "Rule-based flag '%s': triggered=%s (weight=%d)",
            category.value, triggered, weight,
        )

    return flags
