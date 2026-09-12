"""
response_parsers.py

Parsers for the strict, line-prefixed text formats the LLM is instructed
to produce in prompts/qa_prompt.txt and prompts/comparison_prompt.txt
(JSON-based prompts like risk_prompt.txt and summary_prompt.txt are parsed
directly via Pydantic in their respective modules instead).
"""

from __future__ import annotations

import re

from app.llm.exceptions import LLMResponseParsingError

_QA_FIELD_PATTERN = re.compile(
    r"ANSWER:\s*(?P<answer>.*?)\s*"
    r"SOURCE_CONTRACT:\s*(?P<source_contract>.*?)\s*"
    r"SOURCE_PAGE:\s*(?P<source_page>.*?)\s*"
    r"SUPPORTING_TEXT:\s*(?P<supporting_text>.*)",
    re.DOTALL,
)

_COMPARISON_FIELD_PATTERN = re.compile(
    r"IS_CONTRADICTORY:\s*(?P<is_contradictory>true|false)\s*"
    r"EXPLANATION:\s*(?P<explanation>.*)",
    re.DOTALL | re.IGNORECASE,
)


def parse_qa_response(raw_text: str) -> dict[str, str | int | None]:
    """
    Parse the LLM's raw text response to the QA prompt into structured fields.

    Args:
        raw_text: The raw text returned by the LLM, expected to follow the
            ANSWER/SOURCE_CONTRACT/SOURCE_PAGE/SUPPORTING_TEXT format defined
            in prompts/qa_prompt.txt.

    Returns:
        Dict with keys: answer (str), source_contract (str | None),
        source_page (int | None), supporting_text (str | None).

    Raises:
        LLMResponseParsingError: If the response doesn't match the expected format.
    """
    match = _QA_FIELD_PATTERN.search(raw_text)
    if match is None:
        raise LLMResponseParsingError(
            f"LLM response did not match the expected QA format. Raw response: {raw_text!r}"
        )

    answer = match.group("answer").strip()
    source_contract = match.group("source_contract").strip()
    source_page_raw = match.group("source_page").strip()
    supporting_text = match.group("supporting_text").strip()

    if answer == "NOT_FOUND_IN_CONTRACT":
        return {"answer": answer, "source_contract": None, "source_page": None, "supporting_text": None}

    source_page: int | None
    try:
        source_page = int(re.sub(r"[^\d]", "", source_page_raw)) if source_page_raw else None
    except ValueError:
        source_page = None

    return {
        "answer": answer,
        "source_contract": source_contract or None,
        "source_page": source_page,
        "supporting_text": supporting_text or None,
    }


def parse_comparison_response(raw_text: str) -> dict[str, bool | str]:
    """
    Parse the LLM's raw text response to the comparison prompt into
    structured fields.

    Args:
        raw_text: The raw text returned by the LLM, expected to follow the
            IS_CONTRADICTORY/EXPLANATION format defined in
            prompts/comparison_prompt.txt.

    Returns:
        Dict with keys: is_contradictory (bool), explanation (str).

    Raises:
        LLMResponseParsingError: If the response doesn't match the expected format.
    """
    match = _COMPARISON_FIELD_PATTERN.search(raw_text)
    if match is None:
        raise LLMResponseParsingError(
            f"LLM response did not match the expected comparison format. Raw response: {raw_text!r}"
        )

    is_contradictory = match.group("is_contradictory").strip().lower() == "true"
    explanation = match.group("explanation").strip()

    return {"is_contradictory": is_contradictory, "explanation": explanation}
