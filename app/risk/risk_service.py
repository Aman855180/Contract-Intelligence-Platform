"""
risk_service.py

Orchestrates the full hybrid risk detection pipeline for a single
contract document:

  1. Run rule-based clause-absence checks against the extraction result
     (app/risk/rule_based_flags.py) — deterministic, no LLM.
  2. Run LLM-based semantic risk classification with Pydantic-validated
     structured output and retry (app/risk/llm_classifier.py).
  3. Combine all flags and compute the deterministic risk score
     (app/risk/score_calculator.py).
  4. Return a fully validated RiskAssessment.

This is the single entry point Phase 11's Streamlit UI calls for the
risk detection feature and Phase 10's evaluation harness uses for metrics.
"""

from __future__ import annotations

from app.extraction.extraction_service import extract_fields_from_document
from app.models.extraction import ContractExtractionResult
from app.models.risk import RiskAssessment
from app.risk.llm_classifier import classify_llm_risk_flags
from app.risk.rule_based_flags import detect_rule_based_flags
from app.risk.score_calculator import calculate_risk_score
from app.utils.logging_setup import get_logger
from app.utils.perf import perf_span

logger = get_logger(__name__)


def assess_risk(
    document_id: str,
    extraction: ContractExtractionResult | None = None,
) -> RiskAssessment:
    """
    Perform the full hybrid risk assessment for a single contract.

    Args:
        document_id: The document to assess. Must already be ingested
            (SQLite) and indexed (ChromaDB).
        extraction: Optional pre-computed ContractExtractionResult. If
            None, extraction is run automatically. Providing it avoids
            a redundant extraction pass when the caller already has it
            (e.g. Phase 10 evaluation harness, which runs extraction
            once and reuses it for both comparison and risk).

    Returns:
        A fully populated RiskAssessment with rule-based flags,
        LLM-classified flags, and the deterministic risk score.
    """
    with perf_span("risk_assessment_request", extra_info=f"doc_id={document_id}"):
        if extraction is None:
            extraction = extract_fields_from_document(document_id)
            logger.info("Auto-extracted fields for risk assessment of '%s'.", extraction.contract_name)

        # Step 1 — Rule-based: deterministic clause-presence checks.
        with perf_span("risk_rule_based_detection"):
            rule_flags = detect_rule_based_flags(extraction)

        # Step 2 — LLM-classified: semantic risk with validated structured output.
        llm_flags = classify_llm_risk_flags(document_id, extraction.contract_name)

        # Step 3 — Combine and score.
        all_flags = rule_flags + llm_flags
        with perf_span("risk_score_calculation"):
            risk_score = calculate_risk_score(all_flags)

        assessment = RiskAssessment(
            document_id=document_id,
            contract_name=extraction.contract_name,
            flags=all_flags,
            risk_score=risk_score,
        )

        logger.info(
            "Risk assessment complete for '%s': score=%d, triggered flags=%d/%d.",
            extraction.contract_name,
            risk_score,
            len(assessment.triggered_flags()),
            len(all_flags),
        )
        return assessment
