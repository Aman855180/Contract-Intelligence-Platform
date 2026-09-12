"""
extraction_service.py

Orchestrates structured field extraction for a single contract document:
iterates its indexed chunks (from ChromaDB) and runs the three extraction
mechanisms - keyword-based clause classification, regex date extraction,
and NER-enriched obligation detection - assembling the results into a
ContractExtractionResult. This is the single entry point Phase 7
(comparison) and Phase 8 (risk detection) build on.
"""

from __future__ import annotations

from app.database.exceptions import DocumentNotFoundError
from app.database.repository import get_document_metadata
from app.embeddings.chroma_store import get_chunks_by_document
from app.extraction.clause_classifier import classify_chunk
from app.extraction.date_extractor import extract_dates
from app.extraction.exceptions import ExtractionError
from app.extraction.obligation_extractor import extract_obligations
from app.models.extraction import ContractExtractionResult, ExtractedField, FieldType
from app.utils.config_loader import get_settings
from app.utils.logging_setup import get_logger

logger = get_logger(__name__)


def extract_fields_from_document(document_id: str) -> ContractExtractionResult:
    """
    Run the full structured field extraction pipeline over a single
    document's indexed chunks.

    Args:
        document_id: The document to extract fields from. Must already be
            ingested (SQLite, Phase 2) and indexed (ChromaDB, Phase 3).

    Returns:
        A ContractExtractionResult containing every extracted field across
        all chunks, in chunk order.

    Raises:
        ExtractionError: If the document has no indexed chunks to extract from.
    """
    chunks = get_chunks_by_document(document_id)
    if not chunks:
        raise ExtractionError(
            f"No indexed chunks found for document_id='{document_id}'. "
            "Has the document been ingested and indexed (Phases 2-3)?"
        )

    try:
        contract_name = get_document_metadata(document_id).contract_name
    except DocumentNotFoundError:
        # Fall back to chunk metadata if SQLite doesn't have this document
        # for some reason (e.g. it was indexed directly in a test/script).
        logger.warning(
            "Document '%s' not found in SQLite metadata; falling back to ChromaDB chunk metadata for contract_name.",
            document_id,
        )
        contract_name = chunks[0]["metadata"]["contract_name"]

    settings = get_settings()
    date_confidence = settings.get("ner", {}).get("date_regex_confidence", 0.9)

    fields: list[ExtractedField] = []

    for chunk in chunks:
        text = chunk["text"]
        page_number = chunk["metadata"]["page_number"]

        for field_type, confidence in classify_chunk(text):
            fields.append(
                ExtractedField(
                    field_type=field_type,
                    value_text=text,
                    document_id=document_id,
                    contract_name=contract_name,
                    page_number=page_number,
                    confidence=confidence,
                )
            )

        fields.extend(extract_obligations(text, document_id, contract_name, page_number))

        for date_str in extract_dates(text):
            fields.append(
                ExtractedField(
                    field_type=FieldType.KEY_DATES,
                    value_text=date_str,
                    document_id=document_id,
                    contract_name=contract_name,
                    page_number=page_number,
                    confidence=date_confidence,
                )
            )

    logger.info(
        "Extracted %d structured field(s) from document '%s' (id=%s) across %d chunk(s).",
        len(fields),
        contract_name,
        document_id,
        len(chunks),
    )

    return ContractExtractionResult(document_id=document_id, contract_name=contract_name, fields=fields)
