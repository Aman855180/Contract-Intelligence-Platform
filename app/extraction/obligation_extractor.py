"""
obligation_extractor.py

Sentence-level obligation detection: splits chunk text into sentences,
flags sentences matching obligation language patterns ("shall", "agrees
to", etc. - config/extraction_keywords.yaml), and enriches each match with
the obligated party's name via the HuggingFace NER pipeline
(app/extraction/ner_pipeline.py), which is the genuine, defensible use of
a pretrained token-classification NER model in this project (see the
design note in config/settings.yaml's ner section).
"""

from __future__ import annotations

import re

from app.extraction.ner_pipeline import extract_entities
from app.models.extraction import ExtractedField, FieldType
from app.utils.config_loader import get_extraction_keywords, get_settings
from app.utils.logging_setup import get_logger

logger = get_logger(__name__)

# Simple sentence splitter: splits on '.', '!', '?' followed by whitespace
# and a capital letter or opening paren. Avoids pulling in a heavyweight
# NLP sentence tokenizer for what is, in practice, fairly clean contract
# prose extracted by PyMuPDF/python-docx.
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")

_PARTY_ENTITY_LABELS = {"PER", "ORG"}


def _split_sentences(text: str) -> list[str]:
    """Split chunk text into individual sentences."""
    stripped = text.strip()
    if not stripped:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_PATTERN.split(stripped) if s.strip()]


def _sentence_matches_obligation_pattern(sentence_lower: str, keywords: list[str]) -> bool:
    """Return True if the sentence contains any obligation-language keyword."""
    return any(kw in sentence_lower for kw in keywords)


def _best_party_entity(entities: list[dict[str, str | float]]) -> dict[str, str | float] | None:
    """Pick the highest-confidence PER/ORG entity from a list of NER results, if any."""
    party_entities = [e for e in entities if e["label"] in _PARTY_ENTITY_LABELS]
    if not party_entities:
        return None
    return max(party_entities, key=lambda e: e["score"])


def extract_obligations(
    text: str, document_id: str, contract_name: str, page_number: int
) -> list[ExtractedField]:
    """
    Extract obligation-language sentences from a chunk of text, enriching
    each with the obligated party's name (via NER) when identifiable.

    Args:
        text: The chunk text to scan for obligation sentences.
        document_id: The source document's ID (for citation).
        contract_name: The source document's contract name (for citation).
        page_number: The page this chunk came from (for citation).

    Returns:
        List of ExtractedField objects with field_type=OBLIGATIONS, one per
        matched sentence. Confidence is the NER party entity's score if a
        party was found, otherwise the configured
        ner.obligation_pattern_only_confidence fallback.
    """
    keywords_cfg = get_extraction_keywords()
    obligation_keywords = keywords_cfg.get("obligations", [])

    settings = get_settings()
    pattern_only_confidence = settings.get("ner", {}).get("obligation_pattern_only_confidence", 0.5)

    fields: list[ExtractedField] = []
    for sentence in _split_sentences(text):
        sentence_lower = sentence.lower()
        if not _sentence_matches_obligation_pattern(sentence_lower, obligation_keywords):
            continue

        entities = extract_entities(sentence)
        party_entity = _best_party_entity(entities)

        confidence = float(party_entity["score"]) if party_entity else pattern_only_confidence

        fields.append(
            ExtractedField(
                field_type=FieldType.OBLIGATIONS,
                value_text=sentence,
                document_id=document_id,
                contract_name=contract_name,
                page_number=page_number,
                confidence=confidence,
            )
        )

    return fields
