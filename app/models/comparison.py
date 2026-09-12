"""
comparison.py

Pydantic schemas for contract-to-contract comparison and inconsistency
detection (Phase 7). Captures both rule-based and NLI-based mismatch types.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from app.models.extraction import FieldType


class MismatchDetectionMethod(str, Enum):
    """Which detection mechanism flagged this inconsistency."""

    RULE_BASED_NUMERIC_DATE = "rule_based_numeric_date"
    NLI_SEMANTIC_CONTRADICTION = "nli_semantic_contradiction"


class FieldInconsistency(BaseModel):
    """A single flagged inconsistency between two contracts for one field."""

    field_type: FieldType
    detection_method: MismatchDetectionMethod
    contract_a_name: str
    contract_a_page: int = Field(..., ge=1)
    contract_a_text: str
    contract_b_name: str
    contract_b_page: int = Field(..., ge=1)
    contract_b_text: str
    explanation: str = Field(..., description="Why this was flagged, e.g. rule diff or NLI contradiction score.")
    confidence: float = Field(..., ge=0.0, le=1.0)


class ContractComparisonResult(BaseModel):
    """Full comparison result between exactly two contracts."""

    document_id_a: str
    document_id_b: str
    contract_a_name: str
    contract_b_name: str
    inconsistencies: list[FieldInconsistency] = Field(default_factory=list)
    fields_compared: int = Field(..., ge=0, description="Total number of aligned field pairs evaluated.")
