"""
extraction.py

Pydantic schemas for structured clause/field extraction (NER pipeline output)
used by both the comparison and risk-detection modules.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class FieldType(str, Enum):
    """Structured contract fields extracted per project scope."""

    PAYMENT_TERMS = "payment_terms"
    TERMINATION_CLAUSE = "termination_clause"
    CONFIDENTIALITY_CLAUSE = "confidentiality_clause"
    LIABILITY_TERMS = "liability_terms"
    RENEWAL_TERMS = "renewal_terms"
    KEY_DATES = "key_dates"
    OBLIGATIONS = "obligations"
    SIGNATURE = "signature"


class ExtractedField(BaseModel):
    """A single extracted structured field with its source citation."""

    field_type: FieldType
    value_text: str = Field(..., description="Exact extracted span of text for this field.")
    document_id: str
    contract_name: str
    page_number: int = Field(..., ge=1)
    confidence: float = Field(..., ge=0.0, le=1.0, description="NER pipeline confidence score.")


class ContractExtractionResult(BaseModel):
    """All structured fields extracted from a single contract document."""

    document_id: str
    contract_name: str
    fields: list[ExtractedField] = Field(default_factory=list)

    def get_fields_by_type(self, field_type: FieldType) -> list[ExtractedField]:
        """Convenience accessor used by comparison/risk modules."""
        return [f for f in self.fields if f.field_type == field_type]
