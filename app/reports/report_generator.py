"""
report_generator.py

Generates a Pydantic-validated ReportSchema from a contract's extraction
result and risk assessment.

Pipeline (per spec — never deviates from this order):
  LLM → raw JSON text → json.loads() → _LLMReportOutput(**parsed)
  → ReportSchema(document_id=..., contract_name=..., **validated)

The LLM never generates document_id or contract_name — those are injected
by this module after validation. On Pydantic or JSON failure the module
retries up to llm.max_json_retries times, injecting the error back into
the prompt so the LLM can self-correct.

Context assembly for the LLM:
  1. Extracted structured fields (from ContractExtractionResult) — formatted
     as a readable text block, not raw Pydantic JSON, to fit in the prompt.
  2. Retrieved passages — up to top_k document-scoped chunks retrieved with
     targeted queries covering parties, dates, payment, and clauses.
  3. Pre-computed risk flags — only triggered flags are forwarded; the LLM
     reproduces them verbatim (it does not re-derive risk).
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ValidationError, field_validator

from app.embeddings.chroma_store import query_similar
from app.embeddings.embedder import embed_query
from app.llm.ollama_client import generate
from app.llm.prompt_loader import load_prompt
from app.models.extraction import ContractExtractionResult
from app.models.report import (
    ImportantDate,
    KeyClause,
    KeyParty,
    PaymentTermsSummary,
    ReportRiskFlag,
    ReportSchema,
)
from app.models.risk import RiskAssessment, RiskFlag
from app.reports.exceptions import ReportValidationError
from app.utils.config_loader import get_retrieval_config, get_settings
from app.utils.logging_setup import get_logger
from app.utils.perf import llm_attempt_scope, perf_span

logger = get_logger(__name__)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

# Queries used to retrieve a broad cross-section of the contract's content
# for the report's context block.
_REPORT_RETRIEVAL_QUERIES = [
    "parties involved agreement between",
    "effective date start expiry renewal term",
    "payment fees compensation invoice",
    "termination confidentiality liability obligations",
]


# ---------------------------------------------------------------------------
# Internal Pydantic model for the raw LLM output (no document_id/contract_name)
# ---------------------------------------------------------------------------

class _LLMKeyParty(BaseModel):
    name: str
    role: str
    source_contract: str
    source_page: int

    @field_validator("name", "role")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("field must not be empty")
        return v.strip()


class _LLMImportantDate(BaseModel):
    label: str
    date: str
    source_contract: str
    source_page: int


class _LLMPaymentTerms(BaseModel):
    summary: str
    source_contract: str
    source_page: int

    @field_validator("summary")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("summary must not be empty")
        return v.strip()


class _LLMRiskFlag(BaseModel):
    category: str
    severity: str
    explanation: str
    source_contract: str
    source_page: int

    @field_validator("severity")
    @classmethod
    def valid_severity(cls, v: str) -> str:
        if v not in ("low", "medium", "high"):
            raise ValueError(f"severity must be low|medium|high, got '{v}'")
        return v

    @field_validator("explanation")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("explanation must not be empty")
        return v.strip()


class _LLMKeyClause(BaseModel):
    clause_type: str
    summary: str
    source_contract: str
    source_page: int


class _LLMReportOutput(BaseModel):
    """Validated shape of the LLM's raw JSON output (subset of ReportSchema)."""

    overview: str
    key_parties: list[_LLMKeyParty]
    important_dates: list[_LLMImportantDate]
    payment_terms: _LLMPaymentTerms
    risk_flags: list[_LLMRiskFlag]
    key_clauses: list[_LLMKeyClause]
    recommendations: list[str]

    @field_validator("overview")
    @classmethod
    def overview_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("overview must not be empty")
        return v.strip()


# ---------------------------------------------------------------------------
# Context assembly helpers
# ---------------------------------------------------------------------------

def _format_extracted_fields(extraction: ContractExtractionResult) -> str:
    """Format extraction result as a readable text block for the LLM prompt."""
    if not extraction.fields:
        return "No structured fields extracted."

    from app.models.extraction import FieldType
    lines: list[str] = []
    for field_type in FieldType:
        fields = extraction.get_fields_by_type(field_type)
        if not fields:
            continue
        best = max(fields, key=lambda f: f.confidence)
        lines.append(
            f"{field_type.value.replace('_', ' ').title()} "
            f"[{best.contract_name}, p{best.page_number}, conf={best.confidence:.2f}]:\n"
            f"  {best.value_text[:300]}"
        )
    return "\n\n".join(lines) if lines else "No structured fields extracted."


def _format_triggered_risk_flags(assessment: RiskAssessment) -> str:
    """
    Format only triggered risk flags as a JSON-like block for the LLM prompt.
    Severity is derived deterministically from the flag's weight so the LLM
    does not need to invent it.
    """

    def _severity(weight: int) -> str:
        if weight >= 20:
            return "high"
        if weight >= 15:
            return "medium"
        return "low"

    triggered = assessment.triggered_flags()
    if not triggered:
        return "[]"

    items: list[dict[str, Any]] = []
    for flag in triggered:
        items.append({
            "category": flag.category.value,
            "severity": _severity(flag.weight),
            "explanation": flag.explanation,
            "source_contract": flag.source_contract or assessment.contract_name,
            "source_page": flag.source_page or 1,
        })
    return json.dumps(items, indent=2)


def _retrieve_report_context(document_id: str) -> str:
    """
    Retrieve a broad set of passages from the target document for the report
    context block (avoids pulling passages from other indexed documents).
    """
    with perf_span("report_context_retrieval"):
        retrieval_cfg = get_retrieval_config()
        top_k = retrieval_cfg["retrieval"]["top_k"]

        seen: set[str] = set()
        blocks: list[str] = []
        for query in _REPORT_RETRIEVAL_QUERIES:
            vec = embed_query(query)
            for r in query_similar(vec, top_k=top_k, document_id=document_id):
                if r["chunk_id"] not in seen:
                    seen.add(r["chunk_id"])
                    m = r["metadata"]
                    blocks.append(f"[{m['contract_name']}, page {m['page_number']}]\n{r['text']}")

        return "\n\n".join(blocks) if blocks else "No passages retrieved."


def _parse_llm_json(raw: str) -> dict:
    stripped = raw.strip()
    fence = _JSON_FENCE_RE.search(stripped)
    if fence:
        stripped = fence.group(1).strip()
    return json.loads(stripped)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_report(
    extraction: ContractExtractionResult,
    assessment: RiskAssessment,
) -> ReportSchema:
    """
    Generate a Pydantic-validated ReportSchema from extraction + risk data.
    LLM output is always validated before being trusted; invalid responses
    trigger a retry with error feedback, up to llm.max_json_retries attempts.

    Args:
        extraction: Structured field extraction for the contract.
        assessment: Completed risk assessment for the contract.

    Returns:
        A fully validated ReportSchema — the canonical JSON representation.
        Markdown rendering is done separately by markdown_renderer.py.

    Raises:
        ReportValidationError: If the LLM fails after all retries.
    """
    settings = get_settings()
    max_retries: int = settings.get("llm", {}).get("max_json_retries", 3)

    extracted_fields_text = _format_extracted_fields(extraction)
    retrieved_context = _retrieve_report_context(extraction.document_id)
    risk_flags_text = _format_triggered_risk_flags(assessment)

    base_prompt = load_prompt(
        "summary_prompt.txt",
        contract_name=extraction.contract_name,
        extracted_fields=extracted_fields_text,
        retrieved_context=retrieved_context,
        risk_flags=risk_flags_text,
    )

    prompt = base_prompt
    last_error = ""

    with perf_span("report_llm_generation"):
        for attempt in range(1, max_retries + 2):
            try:
                with llm_attempt_scope(attempt):
                    raw = generate(prompt)
                parsed = _parse_llm_json(raw)
                validated = _LLMReportOutput(**parsed)

                # Build the full ReportSchema — document_id and contract_name are
                # injected here, not from LLM output.
                report = ReportSchema(
                    document_id=extraction.document_id,
                    contract_name=extraction.contract_name,
                    overview=validated.overview,
                    key_parties=[
                        KeyParty(**p.model_dump()) for p in validated.key_parties
                    ],
                    important_dates=[
                        ImportantDate(**d.model_dump()) for d in validated.important_dates
                    ],
                    payment_terms=PaymentTermsSummary(**validated.payment_terms.model_dump()),
                    risk_flags=[
                        ReportRiskFlag(**f.model_dump()) for f in validated.risk_flags
                    ],
                    key_clauses=[
                        KeyClause(**c.model_dump()) for c in validated.key_clauses
                    ],
                    recommendations=validated.recommendations,
                )

                logger.info(
                    "Report generated for '%s' on attempt %d/%d.",
                    extraction.contract_name, attempt, max_retries + 1,
                )
                return report

            except (json.JSONDecodeError, ValidationError, KeyError, TypeError) as exc:
                last_error = str(exc)
                logger.warning(
                    "Report generation attempt %d/%d failed for '%s': %s",
                    attempt, max_retries + 1, extraction.contract_name, last_error,
                )
                if attempt <= max_retries:
                    prompt = (
                        base_prompt
                        + f"\n\nYour previous response was invalid. Error: {last_error}\n"
                        "Return ONLY the JSON object. No markdown fences, no extra text. "
                        "Ensure every field matches the schema exactly."
                    )

    raise ReportValidationError(
        f"Report generation failed for '{extraction.contract_name}' after "
        f"{max_retries + 1} attempt(s). Last error: {last_error}"
    )
