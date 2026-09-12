"""
llm_classifier.py

LLM-classified risk flag detection for the three semantic risk categories:
  - unlimited_indemnity_language
  - high_liability_exposure
  - ambiguous_payment_terms

Validation and failure-handling strategy:
  1. Call the LLM and attempt full Pydantic validation (_LLMRiskOutput).
  2. On failure, retry up to llm.max_json_retries times with a repair prompt
     that includes the exact validation error (and a targeted hint when the
     error is an empty explanation field).
  3. If all full-validation attempts are exhausted, attempt permissive
     per-flag parsing (_parse_flags_permissive): validate each _LLMFlagItem
     individually, DISCARD only the invalid ones, and return the valid subset.
     This satisfies the requirement "preserve all valid flags" even when a
     single flag has an empty explanation.
  4. Only raise LLMClassificationError when permissive parsing also yields
     zero valid flags (e.g. the response was not JSON at all).
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, ValidationError, field_validator

from app.embeddings.chroma_store import query_similar
from app.embeddings.embedder import embed_query
from app.llm.ollama_client import generate
from app.llm.prompt_loader import load_prompt
from app.models.risk import RiskCategory, RiskFlag, RiskFlagSource
from app.risk.exceptions import LLMClassificationError
from app.utils.config_loader import get_retrieval_config, get_settings
from app.utils.logging_setup import get_logger
from app.utils.perf import llm_attempt_scope, perf_span

logger = get_logger(__name__)

LLM_CATEGORIES: list[RiskCategory] = [
    RiskCategory.UNLIMITED_INDEMNITY_LANGUAGE,
    RiskCategory.HIGH_LIABILITY_EXPOSURE,
    RiskCategory.AMBIGUOUS_PAYMENT_TERMS,
]

_RETRIEVAL_QUERIES = [
    "indemnify indemnification unlimited liability hold harmless",
    "liability cap damages limitation exposure financial risk",
    "payment terms ambiguous vague unclear due date amount",
]

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


# ---------------------------------------------------------------------------
# Pydantic schema for individual flag items and the full response
# ---------------------------------------------------------------------------

class _LLMFlagItem(BaseModel):
    category: RiskCategory
    triggered: bool
    explanation: str
    source_contract: str | None = None
    source_page: int | None = None
    supporting_text: str | None = None

    @field_validator("explanation")
    @classmethod
    def explanation_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("explanation must not be empty")
        return v.strip()


class _LLMRiskOutput(BaseModel):
    flags: list[_LLMFlagItem]

    @field_validator("flags")
    @classmethod
    def must_have_exactly_three_flags(cls, flags: list[_LLMFlagItem]) -> list[_LLMFlagItem]:
        if len(flags) != 3:
            raise ValueError(f"Expected exactly 3 flag objects, got {len(flags)}.")
        categories_found = {f.category for f in flags}
        expected = set(LLM_CATEGORIES)
        if categories_found != expected:
            raise ValueError(
                f"Flag categories mismatch. Expected {expected}, got {categories_found}."
            )
        return flags


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_weight(category: RiskCategory) -> int:
    weights: dict[str, int] = get_settings().get("risk_scoring", {}).get("weights", {})
    return weights.get(category.value, 0)


def _retrieve_risk_context(document_id: str) -> str:
    with perf_span("risk_context_retrieval", extra_info=f"doc_id={document_id}"):
        retrieval_cfg = get_retrieval_config()
        top_k = retrieval_cfg["retrieval"]["top_k"]
        seen: set[str] = set()
        blocks: list[str] = []
        for query in _RETRIEVAL_QUERIES:
            query_vec = embed_query(query)
            for r in query_similar(query_vec, top_k=top_k, document_id=document_id):
                if r["chunk_id"] not in seen:
                    seen.add(r["chunk_id"])
                    m = r["metadata"]
                    blocks.append(f"[{m['contract_name']}, page {m['page_number']}]\n{r['text']}")
        return "\n\n".join(blocks) if blocks else "No relevant passages retrieved."


def _parse_llm_json(raw_text: str) -> dict:
    stripped = raw_text.strip()
    fence_match = _JSON_FENCE_RE.search(stripped)
    if fence_match:
        stripped = fence_match.group(1).strip()
    return json.loads(stripped)


def _build_risk_flags(validated: _LLMRiskOutput) -> list[RiskFlag]:
    """Convert a fully-validated _LLMRiskOutput into RiskFlag objects."""
    return [
        RiskFlag(
            category=item.category,
            source=RiskFlagSource.LLM_CLASSIFIED,
            triggered=item.triggered,
            weight=_get_weight(item.category),
            explanation=item.explanation,
            source_contract=item.source_contract,
            source_page=item.source_page,
            supporting_text=item.supporting_text,
        )
        for item in validated.flags
    ]


def _parse_flags_permissive(
    parsed_dict: dict,
    contract_name: str,
) -> tuple[list[RiskFlag], list[str]]:
    """
    Validate each flag item individually, preserving those that pass
    _LLMFlagItem validation and discarding those that fail.

    This is the last-resort fallback invoked after all full-response
    validation attempts are exhausted. It satisfies the requirement:
    "discard only the invalid flags instead of failing the whole request."

    Args:
        parsed_dict: The already-parsed (json.loads'd) LLM response dict.
        contract_name: Used for warning log messages only.

    Returns:
        (valid_flags, error_messages) — valid_flags may be empty.
    """
    raw_flags = parsed_dict.get("flags", [])
    valid_flags: list[RiskFlag] = []
    errors: list[str] = []

    for i, item_dict in enumerate(raw_flags):
        try:
            item = _LLMFlagItem(**item_dict)
            valid_flags.append(
                RiskFlag(
                    category=item.category,
                    source=RiskFlagSource.LLM_CLASSIFIED,
                    triggered=item.triggered,
                    weight=_get_weight(item.category),
                    explanation=item.explanation,
                    source_contract=item.source_contract,
                    source_page=item.source_page,
                    supporting_text=item.supporting_text,
                )
            )
        except (ValidationError, TypeError, ValueError) as exc:
            msg = f"flags[{i}]: {exc}"
            errors.append(msg)
            logger.warning(
                "Discarding invalid flag[%d] for '%s': %s", i, contract_name, msg
            )

    return valid_flags, errors


def _repair_prompt(base_prompt: str, last_error: str) -> str:
    """
    Build a retry prompt that includes the validation error and, when the
    error is a missing/empty explanation, adds a targeted reminder.
    """
    targeted_hint = ""
    if "explanation" in last_error and "empty" in last_error:
        targeted_hint = (
            "\nCRITICAL: At least one flag had an empty 'explanation' field. "
            "Every flag MUST have a non-empty explanation string describing "
            "why the category was or was not triggered. "
            "Never use an empty string \"\" for explanation.\n"
        )
    return (
        base_prompt
        + f"\n\nYour previous response was invalid. Error: {last_error[:400]}\n"
        + targeted_hint
        + "Return ONLY the JSON object. No markdown fences, no extra text. "
        "Ensure every flag has a non-empty explanation."
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def classify_llm_risk_flags(document_id: str, contract_name: str) -> list[RiskFlag]:
    """
    Run LLM-based risk classification for a single document.

    Returns at least one RiskFlag when the LLM produces any parseable,
    individually-valid flag — even if the full response fails schema
    validation. Raises LLMClassificationError only when permissive
    per-flag parsing also yields zero valid flags.

    Args:
        document_id: The document to classify risk for.
        contract_name: The document's contract name (for logging).

    Returns:
        List of RiskFlag objects with source=LLM_CLASSIFIED. May contain
        fewer than three flags if some were individually invalid and
        discarded after permissive fallback.

    Raises:
        LLMClassificationError: If zero valid flags can be recovered from
            any response after all retries and permissive parsing.
    """
    with perf_span("risk_llm_classification", extra_info=f"doc_id={document_id}"):
        settings = get_settings()
        max_retries: int = settings.get("llm", {}).get("max_json_retries", 3)

        retrieved_context = _retrieve_risk_context(document_id)
        base_prompt = load_prompt("risk_prompt.txt", retrieved_context=retrieved_context)

        prompt = base_prompt
        last_error: str = ""
        last_raw: str = ""

        for attempt in range(1, max_retries + 2):
            try:
                with llm_attempt_scope(attempt):
                    last_raw = generate(prompt)
                parsed_dict = _parse_llm_json(last_raw)
                validated = _LLMRiskOutput(**parsed_dict)

                flags = _build_risk_flags(validated)
                logger.info(
                    "LLM risk classification succeeded for '%s' on attempt %d/%d.",
                    contract_name, attempt, max_retries + 1,
                )
                return flags

            except (json.JSONDecodeError, ValidationError, KeyError, TypeError) as exc:
                last_error = str(exc)
                logger.warning(
                    "LLM risk classification attempt %d/%d failed for '%s': %s",
                    attempt, max_retries + 1, contract_name, last_error[:200],
                )
                if attempt <= max_retries:
                    logger.info(
                        "[PERF] operation=llm_retry_scheduled attempt=%d next_attempt=%d reason=validation_failure",
                        attempt,
                        attempt + 1,
                    )
                    prompt = _repair_prompt(base_prompt, last_error)

        # ---------------------------------------------------------------------------
        # All full-validation attempts exhausted.
        # Attempt permissive per-flag parsing to salvage individually-valid flags.
        # ---------------------------------------------------------------------------
        if last_raw:
            try:
                parsed_dict = _parse_llm_json(last_raw)
                valid_flags, discarded_errors = _parse_flags_permissive(parsed_dict, contract_name)
                if valid_flags:
                    logger.warning(
                        "Returning %d valid flag(s) for '%s' after discarding %d invalid "
                        "flag(s) via permissive parsing. Discarded: %s",
                        len(valid_flags),
                        contract_name,
                        len(discarded_errors),
                        discarded_errors,
                    )
                    return valid_flags
            except Exception as parse_exc:
                logger.error(
                    "Permissive flag parsing also failed for '%s': %s",
                    contract_name, parse_exc,
                )

        raise LLMClassificationError(
            f"LLM risk classification failed for '{contract_name}' after "
            f"{max_retries + 1} attempt(s) and permissive parsing. "
            f"Last error: {last_error}"
        )
