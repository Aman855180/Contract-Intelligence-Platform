"""
field_aligner.py

Aligns extracted fields between two contracts for pairwise comparison.
Each contract may have multiple ExtractedField objects for the same
field_type (e.g. several chunks all matching PAYMENT_TERMS). Rather than
comparing every permutation (O(n²) NLI calls), we select the single
highest-confidence representative per field_type per contract and compare
those pairs — one aligned pair per field_type present in both contracts.

Field types absent from either contract are skipped (not flagged as a
mismatch here; missing-clause detection is handled separately in the
risk module in Phase 8).
"""

from __future__ import annotations

from app.models.extraction import ContractExtractionResult, ExtractedField, FieldType
from app.utils.logging_setup import get_logger

logger = get_logger(__name__)


def _best_field(fields: list[ExtractedField]) -> ExtractedField:
    """Return the highest-confidence field from a non-empty list."""
    return max(fields, key=lambda f: f.confidence)


def align_fields(
    result_a: ContractExtractionResult,
    result_b: ContractExtractionResult,
) -> list[tuple[ExtractedField, ExtractedField]]:
    """
    Produce a list of (field_from_A, field_from_B) pairs for every
    FieldType that appears in both extraction results.

    Args:
        result_a: Extraction result for Contract A.
        result_b: Extraction result for Contract B.

    Returns:
        List of aligned field pairs (one per shared field_type).
        Field types present in only one contract are omitted from the
        returned list — they are not flagged here.
    """
    shared_types = set()
    for field_type in FieldType:
        has_in_a = bool(result_a.get_fields_by_type(field_type))
        has_in_b = bool(result_b.get_fields_by_type(field_type))
        if has_in_a and has_in_b:
            shared_types.add(field_type)

    pairs: list[tuple[ExtractedField, ExtractedField]] = []
    for field_type in sorted(shared_types, key=lambda ft: ft.value):
        best_a = _best_field(result_a.get_fields_by_type(field_type))
        best_b = _best_field(result_b.get_fields_by_type(field_type))
        pairs.append((best_a, best_b))
        logger.debug(
            "Aligned field '%s': [%s p%d] vs [%s p%d]",
            field_type.value,
            best_a.contract_name, best_a.page_number,
            best_b.contract_name, best_b.page_number,
        )

    return pairs
