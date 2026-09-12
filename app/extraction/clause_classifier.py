"""
clause_classifier.py

Deterministic, explainable clause-type classification via curated keyword
matching (patterns in config/extraction_keywords.yaml). See the design
note in config/settings.yaml's ner section for why this approach was
chosen over a token-classification NER model: no off-the-shelf model
predicts contract clause TYPES, and fine-tuning one is out of scope.

Covers: payment_terms, termination_clause, confidentiality_clause,
liability_terms, renewal_terms, signature. (key_dates is handled by
app/extraction/date_extractor.py; obligations by
app/extraction/obligation_extractor.py.)
"""

from __future__ import annotations

from app.models.extraction import FieldType
from app.utils.config_loader import get_extraction_keywords
from app.utils.logging_setup import get_logger

logger = get_logger(__name__)

# Field types handled by this keyword classifier (excludes key_dates and
# obligations, which have their own dedicated extraction logic).
_CLAUSE_FIELD_TYPES = (
    FieldType.PAYMENT_TERMS,
    FieldType.TERMINATION_CLAUSE,
    FieldType.CONFIDENTIALITY_CLAUSE,
    FieldType.LIABILITY_TERMS,
    FieldType.RENEWAL_TERMS,
    FieldType.SIGNATURE,
)

# Confidence scaling: each additional distinct matched keyword increases
# confidence, capped at 1.0. A single match yields a moderate-confidence
# 0.6, since one keyword alone is a weaker signal than several co-occurring.
_BASE_CONFIDENCE = 0.6
_CONFIDENCE_PER_EXTRA_MATCH = 0.15


def _score_matches(text_lower: str, keywords: list[str]) -> float:
    """
    Compute a confidence score for how strongly a chunk of text matches a
    field type's keyword list.

    Args:
        text_lower: The chunk text, already lowercased.
        keywords: The keyword/phrase patterns for one field type.

    Returns:
        0.0 if no keywords matched; otherwise a confidence in (0.0, 1.0].
    """
    match_count = sum(1 for kw in keywords if kw in text_lower)
    if match_count == 0:
        return 0.0
    confidence = _BASE_CONFIDENCE + _CONFIDENCE_PER_EXTRA_MATCH * (match_count - 1)
    return min(confidence, 1.0)


def classify_chunk(text: str) -> list[tuple[FieldType, float]]:
    """
    Classify a chunk of text against the clause-type keyword taxonomy. A
    single chunk may match multiple field types (e.g. a clause mentioning
    both payment and termination).

    Args:
        text: The chunk text to classify.

    Returns:
        List of (FieldType, confidence) tuples for every field type with at
        least one keyword match, confidence in (0.0, 1.0]. Empty list if no
        field type matched.
    """
    keywords_cfg = get_extraction_keywords()
    text_lower = text.lower()

    matches: list[tuple[FieldType, float]] = []
    for field_type in _CLAUSE_FIELD_TYPES:
        keywords = keywords_cfg.get(field_type.value, [])
        score = _score_matches(text_lower, keywords)
        if score > 0.0:
            matches.append((field_type, score))

    return matches
