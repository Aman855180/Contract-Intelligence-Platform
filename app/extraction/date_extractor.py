"""
date_extractor.py

Regex-based date extraction for the key_dates field. Standard NER label
sets (including CoNLL-2003, used by the dslim/bert-base-NER pipeline in
app/extraction/ner_pipeline.py) do not include a DATE entity type, so date
extraction is handled here via deterministic pattern matching instead —
honest about what general-purpose NER can and can't do out of the box.

Covers common contract date formats: "January 1, 2024", "1st of March,
2024", "01/15/2024", "2024-01-15".
"""

from __future__ import annotations

import re

_MONTH_NAMES = (
    "January|February|March|April|May|June|July|August|September|October|November|December"
)

_DATE_PATTERNS = [
    # "January 1, 2024" / "January 1st, 2024"
    re.compile(rf"\b(?:{_MONTH_NAMES})\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b", re.IGNORECASE),
    # "1st of March, 2024" / "15 March 2024"
    re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:{_MONTH_NAMES})\s*,?\s+\d{{4}}\b", re.IGNORECASE),
    # "01/15/2024" or "1/15/24" or "15-01-2024"
    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    # ISO format "2024-01-15"
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
]


def extract_dates(text: str) -> list[str]:
    """
    Extract all date-like substrings from a block of text.

    Args:
        text: The text to scan for dates (typically a single chunk).

    Returns:
        List of distinct matched date strings, in the order first
        encountered (duplicates removed, original casing preserved).
    """
    seen: set[str] = set()
    results: list[str] = []

    for pattern in _DATE_PATTERNS:
        for match in pattern.finditer(text):
            matched_text = match.group(0).strip()
            normalized = matched_text.lower()
            if normalized not in seen:
                seen.add(normalized)
                results.append(matched_text)

    return results
