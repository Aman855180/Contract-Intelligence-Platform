"""
comparison_service.py

Orchestrates the full contract-to-contract inconsistency detection pipeline:
  1. Align fields between two ContractExtractionResults (field_aligner.py).
  2. For each aligned pair, run rule-based detection first (fast, free,
     deterministic — numeric/date fields).
  3. For field types NOT handled by rule-based logic, OR when rule-based
     found no mismatch on a clause-text field, run NLI contradiction
     detection (nli_based.py).
  4. Deduplicate: if a pair was already flagged by rule-based, do NOT also
     run NLI on it (avoid double-counting the same inconsistency).
  5. Return a Pydantic-validated ContractComparisonResult.

Decision boundary: rule-based first, NLI only when rules don't apply.
This keeps NLI calls bounded to fields that genuinely need semantic
reasoning, which matters for latency and interpretability.
"""

from __future__ import annotations

from app.comparison.exceptions import ComparisonError
from app.comparison.field_aligner import align_fields
from app.comparison.nli_based import check_nli_contradiction
from app.comparison.rule_based import RULE_BASED_FIELD_TYPES, check_rule_based_mismatch
from app.models.comparison import ContractComparisonResult, FieldInconsistency
from app.models.extraction import ContractExtractionResult, FieldType
from app.utils.logging_setup import get_logger

logger = get_logger(__name__)

# Field types for which NLI is the primary (or only) detection mechanism.
# OBLIGATIONS is skipped — comparing individual obligation sentences
# across contracts produces very high false-positive rates (almost any
# two obligation sentences sound like they "contradict" at the NLI level)
# without a deeper semantic alignment pass that's out of scope here.
_NLI_FIELD_TYPES: frozenset[FieldType] = frozenset(
    {
        FieldType.TERMINATION_CLAUSE,
        FieldType.CONFIDENTIALITY_CLAUSE,
        FieldType.LIABILITY_TERMS,
    }
)


def compare_contracts(
    extraction_a: ContractExtractionResult,
    extraction_b: ContractExtractionResult,
) -> ContractComparisonResult:
    """
    Compare two contract extraction results and return a structured
    inconsistency report.

    Args:
        extraction_a: Structured field extraction result for Contract A.
        extraction_b: Structured field extraction result for Contract B.

    Returns:
        A ContractComparisonResult listing all detected inconsistencies.

    Raises:
        ComparisonError: If either contract has no extractable fields.
    """
    if not extraction_a.fields:
        raise ComparisonError(
            f"Contract A ('{extraction_a.contract_name}') has no extracted fields to compare."
        )
    if not extraction_b.fields:
        raise ComparisonError(
            f"Contract B ('{extraction_b.contract_name}') has no extracted fields to compare."
        )

    aligned_pairs = align_fields(extraction_a, extraction_b)
    inconsistencies: list[FieldInconsistency] = []

    for field_a, field_b in aligned_pairs:
        field_type = field_a.field_type

        # Skip field types where neither rule-based nor NLI applies cleanly.
        if field_type not in RULE_BASED_FIELD_TYPES and field_type not in _NLI_FIELD_TYPES:
            logger.debug("Skipping comparison for field type '%s' (not in scope).", field_type.value)
            continue

        rule_result: FieldInconsistency | None = None
        if field_type in RULE_BASED_FIELD_TYPES:
            rule_result = check_rule_based_mismatch(field_a, field_b)
            if rule_result is not None:
                inconsistencies.append(rule_result)
                logger.info(
                    "Rule-based mismatch on '%s': %s p%d vs %s p%d",
                    field_type.value,
                    field_a.contract_name, field_a.page_number,
                    field_b.contract_name, field_b.page_number,
                )

        # Run NLI for:
        #   (a) clause-text field types always handled by NLI, OR
        #   (b) rule-based fields where the rule found no numeric/date mismatch
        #       (still possible the clauses contradict semantically, e.g. one
        #       says payment is due "net 30" but also "subject to dispute
        #       withholding", and the other says "unconditional net 30").
        should_run_nli = (field_type in _NLI_FIELD_TYPES) or (
            field_type in RULE_BASED_FIELD_TYPES and rule_result is None
        )
        if should_run_nli:
            nli_result = check_nli_contradiction(field_a, field_b)
            if nli_result is not None:
                inconsistencies.append(nli_result)
                logger.info(
                    "NLI contradiction on '%s': score=%.3f",
                    field_type.value,
                    nli_result.confidence,
                )

    result = ContractComparisonResult(
        document_id_a=extraction_a.document_id,
        document_id_b=extraction_b.document_id,
        contract_a_name=extraction_a.contract_name,
        contract_b_name=extraction_b.contract_name,
        inconsistencies=inconsistencies,
        fields_compared=len(aligned_pairs),
    )

    logger.info(
        "Comparison '%s' vs '%s': %d field(s) compared, %d inconsistenc(ies) detected.",
        extraction_a.contract_name,
        extraction_b.contract_name,
        result.fields_compared,
        len(result.inconsistencies),
    )
    return result
