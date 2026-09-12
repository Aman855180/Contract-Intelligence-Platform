"""
report.py

Pydantic schemas for the executive summary / report generation pipeline
(Phase 9). The LLM output is validated against ReportSchema BEFORE any
Markdown rendering occurs — JSON is the canonical internal representation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CitedItem(BaseModel):
    """Base shape shared by every report element that must carry a citation."""

    source_contract: str
    source_page: int = Field(..., ge=1)


class KeyParty(CitedItem):
    name: str
    role: str


class ImportantDate(CitedItem):
    label: str
    date: str


class PaymentTermsSummary(CitedItem):
    summary: str


class ReportRiskFlag(CitedItem):
    category: str
    severity: str = Field(..., pattern="^(low|medium|high)$")
    explanation: str


class KeyClause(CitedItem):
    clause_type: str
    summary: str


class ReportSchema(BaseModel):
    """
    Canonical, schema-validated representation of the executive summary
    report. app/reports performs: LLM -> JSON -> ReportSchema(**json) ->
    Markdown rendering. The LLM never generates Markdown directly.
    """

    document_id: str
    contract_name: str
    overview: str = Field(..., min_length=1)
    key_parties: list[KeyParty] = Field(default_factory=list)
    important_dates: list[ImportantDate] = Field(default_factory=list)
    payment_terms: PaymentTermsSummary
    risk_flags: list[ReportRiskFlag] = Field(default_factory=list)
    key_clauses: list[KeyClause] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
