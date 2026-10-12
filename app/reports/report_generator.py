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
  3. Pre-computed risk flags — triggered flags are provided as background
     context to inform recommendations; ReportRiskFlag objects are assembled
     deterministically in Python (zero risk hallucination).
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.embeddings.chroma_store import query_similar, query_similar_batch
from app.embeddings.embedder import embed_queries, embed_query

_real_embed_query = embed_query
_real_query_similar = query_similar
from app.llm.prompt_loader import load_prompt
from app.llm.provider import get_llm_provider, get_report_provider
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
from app.utils.runtime_config import get_runtime_bool

logger = get_logger(__name__)


def _fallback_to_ollama_enabled(llm_cfg: dict[str, Any]) -> bool:
    """Resolve fallback override, preserving the YAML default for local use."""
    return get_runtime_bool(
        "LLM_FALLBACK_TO_OLLAMA", bool(llm_cfg.get("fallback_to_ollama", False))
    )


def generate(
    prompt: str,
    attempt: int | None = None,
    response_schema: Any | None = None,
) -> str:
    """
    Generate contract report response using the configured report LLM provider,
    with automatic fallback to Ollama if Gemini fails and fallback is enabled.
    """
    settings = get_settings()
    llm_cfg = settings.get("llm", {})
    fallback_enabled = _fallback_to_ollama_enabled(llm_cfg)
    report_provider = get_report_provider(settings)

    try:
        return report_provider.generate(
            prompt,
            attempt=attempt,
            response_schema=response_schema,
        )
    except Exception as exc:
        if fallback_enabled and report_provider.provider_name != "ollama":
            # Exception text may include SDK request details or credentials.
            logger.warning(
                "Report provider '%s' failed (%s). Falling back to Ollama.",
                report_provider.provider_name,
                type(exc).__name__,
            )
            ollama_provider = get_llm_provider(settings, provider_type="ollama")
            return ollama_provider.generate(
                prompt,
                attempt=attempt,
                response_schema=response_schema,
            )
        raise


def _safe_generate(
    prompt: str,
    attempt: int | None = None,
    response_schema: Any | None = None,
) -> str:
    """Invoke generate(), falling back to 1-arg signature if mock does not accept kwargs."""
    try:
        return generate(prompt, attempt=attempt, response_schema=response_schema)
    except TypeError as exc:
        err_msg = str(exc)
        if "response_schema" in err_msg or "unexpected keyword argument" in err_msg:
            return generate(prompt)
        raise


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

# Queries used to retrieve a broad cross-section of the contract's content
# for the report's context block.
_REPORT_RETRIEVAL_QUERIES = [
    "parties involved agreement between",
    "effective date start expiry renewal term",
    "payment fees compensation invoice",
    "termination confidentiality liability obligations",
]


# Explicit JSON schema for structured LLM response (Gemini structured output)
LLM_REPORT_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "overview": {"type": "STRING"},
        "key_parties": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "role": {"type": "STRING"},
                    "source_contract": {"type": "STRING"},
                    "source_page": {"type": "INTEGER"},
                },
                "required": ["name", "role", "source_contract", "source_page"],
            },
        },
        "important_dates": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "label": {"type": "STRING"},
                    "date": {"type": "STRING"},
                    "source_contract": {"type": "STRING"},
                    "source_page": {"type": "INTEGER"},
                },
                "required": ["label", "date", "source_contract", "source_page"],
            },
        },
        "payment_terms": {
            "type": "OBJECT",
            "properties": {
                "summary": {"type": "STRING"},
                "source_contract": {"type": "STRING"},
                "source_page": {"type": "INTEGER"},
            },
            "required": ["summary", "source_contract", "source_page"],
        },
        "key_clauses": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "clause_type": {"type": "STRING"},
                    "summary": {"type": "STRING"},
                    "source_contract": {"type": "STRING"},
                    "source_page": {"type": "INTEGER"},
                },
                "required": ["clause_type", "summary", "source_contract", "source_page"],
            },
        },
        "recommendations": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
        # Optional in _LLMReportOutput; the canonical risk flags are assembled
        # deterministically from the risk assessment after validation.
        "risk_flags": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "category": {"type": "STRING"},
                    "severity": {"type": "STRING", "enum": ["low", "medium", "high"]},
                    "explanation": {"type": "STRING"},
                    "source_contract": {"type": "STRING"},
                    "source_page": {"type": "INTEGER"},
                },
                "required": ["category", "severity", "explanation", "source_contract", "source_page"],
            },
        },
    },
    "required": [
        "overview",
        "key_parties",
        "important_dates",
        "payment_terms",
        "key_clauses",
        "recommendations",
    ],
}


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
    key_clauses: list[_LLMKeyClause]
    recommendations: list[str]
    risk_flags: list[_LLMRiskFlag] = Field(default_factory=list)

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
    """Format extraction result as a concise summary block for the LLM prompt."""
    if not extraction.fields:
        return "No structured fields extracted."

    from app.models.extraction import FieldType
    lines: list[str] = []
    for field_type in FieldType:
        fields = extraction.get_fields_by_type(field_type)
        if not fields:
            continue
        best = max(fields, key=lambda f: f.confidence)
        snippet = " ".join(best.value_text[:120].split())
        lines.append(
            f"- {field_type.value.replace('_', ' ').title()} "
            f"[{best.contract_name}, p{best.page_number}, conf={best.confidence:.2f}]: {snippet}"
        )
    return "\n".join(lines) if lines else "No structured fields extracted."


def _format_triggered_risk_flags(assessment: RiskAssessment) -> str:
    """
    Format only triggered risk flags as a concise JSON block for the LLM prompt.
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
    return json.dumps(items)


def _build_report_risk_flags(assessment: RiskAssessment) -> list[ReportRiskFlag]:
    """
    Construct ReportRiskFlag objects deterministically from the pre-computed
    assessment. Severity is derived deterministically from the flag's weight
    (weight >= 20 -> high, weight >= 15 -> medium, else low).
    """
    def _severity(weight: int) -> str:
        if weight >= 20:
            return "high"
        if weight >= 15:
            return "medium"
        return "low"

    return [
        ReportRiskFlag(
            category=flag.category.value if hasattr(flag.category, "value") else str(flag.category),
            severity=_severity(flag.weight),
            explanation=flag.explanation,
            source_contract=flag.source_contract or assessment.contract_name,
            source_page=flag.source_page or 1,
        )
        for flag in assessment.triggered_flags()
    ]


def _retrieve_report_context(document_id: str) -> str:
    """
    Retrieve a broad set of passages from the target document for the report
    context block (avoids pulling passages from other indexed documents).
    """
    with perf_span("report_context_retrieval"):
        retrieval_cfg = get_retrieval_config()
        # Override global top_k to reduce context bloat for multi-query retrieval
        top_k = 3

        seen: set[str] = set()
        blocks: list[str] = []

        if embed_query is not _real_embed_query:
            query_vecs = [embed_query(q) for q in _REPORT_RETRIEVAL_QUERIES]
        else:
            query_vecs = embed_queries(_REPORT_RETRIEVAL_QUERIES)

        if query_similar is not _real_query_similar:
            all_results = [query_similar(v, top_k=top_k, document_id=document_id) for v in query_vecs]
        else:
            all_results = query_similar_batch(query_vecs, top_k=top_k, document_id=document_id)

        for results in all_results:
            for r in results:
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

    with perf_span("report_llm_generation", extra_info=f"contract={extraction.contract_name}"):
        for attempt in range(1, max_retries + 2):
            try:
                with llm_attempt_scope(attempt):
                    raw = _safe_generate(
                        prompt,
                        attempt=attempt,
                        response_schema=LLM_REPORT_OUTPUT_SCHEMA,
                    )
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
                    risk_flags=_build_report_risk_flags(assessment),
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
                if isinstance(exc, ValidationError):
                    # Pydantic's default string includes input values, which can
                    # contain contract text. Log only locations and error types.
                    details = ",".join(
                        f"{'.'.join(str(part) for part in item['loc'])}:{item['type']}"
                        for item in exc.errors(include_input=False)
                    )
                elif isinstance(exc, json.JSONDecodeError):
                    details = f"line={exc.lineno} column={exc.colno}"
                else:
                    details = type(exc).__name__
                logger.warning(
                    "[LLM_RETRY] operation=report_generation attempt=%d/%d failure=validation type=%s details=%s",
                    attempt,
                    max_retries + 1,
                    type(exc).__name__,
                    details[:300],
                )
                if attempt <= max_retries:
                    logger.info(
                        "[PERF] operation=llm_retry_scheduled attempt=%d next_attempt=%d reason=report_validation_failure",
                        attempt,
                        attempt + 1,
                    )
                    prompt = (
                        base_prompt
                        + f"\n\nYour previous response was invalid. Error: {last_error[:300]}\n"
                        "Return ONLY the JSON object. No markdown fences, no extra text. "
                        "Ensure every field matches the schema exactly."
                    )

    raise ReportValidationError(
        f"Report generation failed for '{extraction.contract_name}' after "
        f"{max_retries + 1} attempt(s). Last error: {last_error}"
    )
