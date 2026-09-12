"""
report_service.py

Top-level orchestrator for the executive summary pipeline. Accepts a
document_id, runs extraction and risk assessment if not pre-computed,
generates and validates the JSON report, then renders it to Markdown.

Returns both the validated ReportSchema (canonical JSON form) and the
rendered Markdown string — callers choose which they need.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.extraction.extraction_service import extract_fields_from_document
from app.models.extraction import ContractExtractionResult
from app.models.report import ReportSchema
from app.models.risk import RiskAssessment
from app.reports.markdown_renderer import render_markdown
from app.reports.report_generator import generate_report
from app.risk.risk_service import assess_risk
from app.utils.logging_setup import get_logger
from app.utils.perf import perf_span

logger = get_logger(__name__)


@dataclass
class ReportResult:
    """
    Container for both output forms of a generated report, so the caller
    decides how to use them without the service dictating format.
    """
    report: ReportSchema           # Pydantic-validated JSON representation
    markdown: str                  # Rendered Markdown (separate step, no LLM)


def generate_contract_report(
    document_id: str,
    extraction: ContractExtractionResult | None = None,
    assessment: RiskAssessment | None = None,
) -> ReportResult:
    """
    Generate a full executive summary report for a single contract document.

    Args:
        document_id: The contract to report on. Must be ingested and indexed.
        extraction: Optional pre-computed extraction result. Auto-computed if
            None (saves a re-extraction pass when the caller already has it).
        assessment: Optional pre-computed risk assessment. Auto-computed if
            None (saves a re-assessment pass when the caller already has it).

    Returns:
        A ReportResult containing the validated ReportSchema and the
        rendered Markdown string.

    Raises:
        ExtractionError: If extraction fails (document not indexed).
        ReportValidationError: If the LLM fails to produce valid JSON
            after all retries.
    """
    with perf_span("report_generation_request", extra_info=f"doc_id={document_id}"):
        if extraction is None:
            extraction = extract_fields_from_document(document_id)
            logger.info("Auto-extracted fields for report on '%s'.", extraction.contract_name)

        if assessment is None:
            assessment = assess_risk(document_id, extraction=extraction)
            logger.info("Auto-assessed risk for report on '%s'.", extraction.contract_name)

        report = generate_report(extraction, assessment)

        # Markdown rendering is a pure transformation — no LLM, no I/O.
        with perf_span("markdown_rendering"):
            markdown = render_markdown(report)

        logger.info(
            "Report complete for '%s': %d parties, %d dates, %d risk flags, %d clauses.",
            report.contract_name,
            len(report.key_parties),
            len(report.important_dates),
            len(report.risk_flags),
            len(report.key_clauses),
        )
        return ReportResult(report=report, markdown=markdown)
