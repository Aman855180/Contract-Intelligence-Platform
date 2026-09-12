"""
rule_based.py

Deterministic (no LLM, no NLI) mismatch detection for structured fields
whose values can be compared via numeric or date parsing. Returns a
FieldInconsistency when a mismatch is found, or None when the values are
consistent or cannot be reliably compared.

This module covers two sub-strategies:
  1. Numeric extraction — finds all numbers in each text and compares their
     sets (e.g. "30 days" vs "60 days" is a clear mismatch; "30" vs "thirty"
     cannot be compared numerically so falls through to NLI).
  2. Date extraction — reuses app/extraction/date_extractor.py and checks
     whether the canonical date sets differ.

Covered field types: PAYMENT_TERMS, RENEWAL_TERMS, KEY_DATES. The remaining
clause-text fields (TERMINATION_CLAUSE, CONFIDENTIALITY_CLAUSE,
LIABILITY_TERMS, OBLIGATIONS) contain free prose and are handled in
nli_based.py.
"""

from __future__ import annotations

import re

from app.extraction.date_extractor import extract_dates
from app.models.comparison import FieldInconsistency, MismatchDetectionMethod
from app.models.extraction import ExtractedField, FieldType
from app.utils.logging_setup import get_logger

logger = get_logger(__name__)

# Field types where rule-based numeric/date comparison applies.
RULE_BASED_FIELD_TYPES: frozenset[FieldType] = frozenset(
    {FieldType.PAYMENT_TERMS, FieldType.RENEWAL_TERMS, FieldType.KEY_DATES}
)

_NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")


def _extract_numbers(text: str) -> frozenset[float]:
    """Extract all numeric literals from text as a frozenset of floats."""
    return frozenset(float(m) for m in _NUMBER_PATTERN.findall(text))


def _extract_date_set(text: str) -> frozenset[str]:
    """Extract normalised (lowercased) date strings from text."""
    return frozenset(d.lower() for d in extract_dates(text))


def _build_explanation(field_type: FieldType, values_a: frozenset, values_b: frozenset) -> str:
    """Build a human-readable explanation string for a rule-based mismatch."""
    only_in_a = sorted(str(v) for v in values_a - values_b)
    only_in_b = sorted(str(v) for v in values_b - values_a)
    return (
        f"Rule-based {field_type.value} mismatch: "
        f"values only in Contract A: {only_in_a}; "
        f"values only in Contract B: {only_in_b}."
    )


def check_rule_based_mismatch(
    field_a: ExtractedField,
    field_b: ExtractedField,
) -> FieldInconsistency | None:
    """
    Apply deterministic rule-based comparison to a pair of aligned fields.
    Returns a FieldInconsistency if a clear numeric or date mismatch is
    found, otherwise None (meaning the field pair should be passed to the
    NLI checker or treated as consistent).

    Args:
        field_a: The extracted field from Contract A.
        field_b: The extracted field from Contract B. Must have the same
            field_type as field_a.

    Returns:
        A FieldInconsistency with detection_method=RULE_BASED_NUMERIC_DATE,
        or None if no rule-based mismatch was detected.
    """
    if field_a.field_type != field_b.field_type:
        raise ValueError(
            f"field_type mismatch: {field_a.field_type} vs {field_b.field_type}. "
            "Only aligned same-type fields should be compared."
        )

    field_type = field_a.field_type

    # --- KEY_DATES: compare extracted date sets ---
    if field_type == FieldType.KEY_DATES:
        dates_a = _extract_date_set(field_a.value_text)
        dates_b = _extract_date_set(field_b.value_text)
        # Only flag if both sides have dates and they differ.
        if dates_a and dates_b and dates_a != dates_b:
            logger.debug(
                "Rule-based date mismatch on %s: %s vs %s",
                field_type.value, dates_a, dates_b,
            )
            return FieldInconsistency(
                field_type=field_type,
                detection_method=MismatchDetectionMethod.RULE_BASED_NUMERIC_DATE,
                contract_a_name=field_a.contract_name,
                contract_a_page=field_a.page_number,
                contract_a_text=field_a.value_text,
                contract_b_name=field_b.contract_name,
                contract_b_page=field_b.page_number,
                contract_b_text=field_b.value_text,
                explanation=_build_explanation(field_type, dates_a, dates_b),
                confidence=1.0,
            )
        return None

    # --- PAYMENT_TERMS / RENEWAL_TERMS: compare extracted numeric sets ---
    if field_type in (FieldType.PAYMENT_TERMS, FieldType.RENEWAL_TERMS):
        nums_a = _extract_numbers(field_a.value_text)
        nums_b = _extract_numbers(field_b.value_text)
        # Only flag when both sides have numbers and they differ — avoids
        # false positives when one side uses prose ("thirty") and the other
        # uses digits ("30").
        if nums_a and nums_b and nums_a != nums_b:
            logger.debug(
                "Rule-based numeric mismatch on %s: %s vs %s",
                field_type.value, nums_a, nums_b,
            )
            return FieldInconsistency(
                field_type=field_type,
                detection_method=MismatchDetectionMethod.RULE_BASED_NUMERIC_DATE,
                contract_a_name=field_a.contract_name,
                contract_a_page=field_a.page_number,
                contract_a_text=field_a.value_text,
                contract_b_name=field_b.contract_name,
                contract_b_page=field_b.page_number,
                contract_b_text=field_b.value_text,
                explanation=_build_explanation(field_type, nums_a, nums_b),
                confidence=1.0,
            )
        return None

    # Field type not handled by rule-based logic → caller should use NLI.
    return None
