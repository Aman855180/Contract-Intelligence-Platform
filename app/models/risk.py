"""
risk.py

Pydantic schemas for the hybrid (rule-based + LLM-classified) risk detection
pipeline (Phase 8), including the deterministic risk score breakdown.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class RiskFlagSource(str, Enum):
    """Whether a risk flag was determined deterministically or via LLM classification."""

    RULE_BASED = "rule_based"
    LLM_CLASSIFIED = "llm_classified"


class RiskCategory(str, Enum):
    """Fixed risk taxonomy. The LLM must classify against this schema only."""

    MISSING_SIGNATURE_FIELD = "missing_signature_field"
    MISSING_TERMINATION_CLAUSE = "missing_termination_clause"
    MISSING_CONFIDENTIALITY_CLAUSE = "missing_confidentiality_clause"
    UNLIMITED_INDEMNITY_LANGUAGE = "unlimited_indemnity_language"
    HIGH_LIABILITY_EXPOSURE = "high_liability_exposure"
    AMBIGUOUS_PAYMENT_TERMS = "ambiguous_payment_terms"


class RiskFlag(BaseModel):
    """A single triggered (or evaluated) risk flag with citation."""

    category: RiskCategory
    source: RiskFlagSource
    triggered: bool
    weight: int = Field(..., ge=0, le=100, description="Weight from config/settings.yaml risk_scoring.weights.")
    explanation: str
    source_contract: str | None = None
    source_page: int | None = Field(default=None, ge=1)
    supporting_text: str | None = None


class RiskAssessment(BaseModel):
    """Full risk assessment for a single contract, including final score."""

    document_id: str
    contract_name: str
    flags: list[RiskFlag] = Field(default_factory=list)
    risk_score: int = Field(..., ge=0, le=100, description="Deterministic weighted sum, clipped to [0, 100].")

    def triggered_flags(self) -> list[RiskFlag]:
        """Convenience accessor returning only the flags that were triggered."""
        return [f for f in self.flags if f.triggered]
