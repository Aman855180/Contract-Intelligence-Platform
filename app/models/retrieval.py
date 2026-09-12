"""
retrieval.py

Pydantic schemas for hybrid retrieval results and grounded Q&A responses.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """A single chunk returned by the hybrid retriever, with fusion score."""

    chunk_id: str
    document_id: str
    contract_name: str
    page_number: int = Field(..., ge=1)
    text: str
    vector_score: float | None = Field(
        default=None, description="Raw cosine similarity score from ChromaDB, if retrieved by vector search."
    )
    bm25_score: float | None = Field(
        default=None, description="Raw BM25 score, if retrieved by keyword search."
    )
    fused_score: float = Field(..., description="Combined reciprocal-rank-fusion score used for final ranking.")


class RetrievalResult(BaseModel):
    """Full output of the hybrid retriever for a single query."""

    query: str
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    above_threshold: bool = Field(
        ..., description="Whether at least one chunk met the configured similarity_threshold."
    )


class GroundednessVerdict(BaseModel):
    """
    Output of the NLI-based groundedness check comparing a generated claim
    against its cited source passage. This is NOT an LLM self-reported score.
    """

    claim: str
    source_passage: str
    entailment_score: float = Field(..., ge=0.0, le=1.0)
    contradiction_score: float = Field(..., ge=0.0, le=1.0)
    neutral_score: float = Field(..., ge=0.0, le=1.0)
    is_grounded: bool = Field(..., description="True if entailment_score >= groundedness.entailment_threshold.")


class QAAnswer(BaseModel):
    """Structured, citation-bearing answer to a user's contract question."""

    question: str
    answer: str
    found_in_contract: bool = Field(
        ..., description="False if no supporting evidence was retrieved above the similarity threshold."
    )
    source_contract: str | None = None
    source_page: int | None = Field(default=None, ge=1)
    supporting_text: str | None = None
    groundedness: GroundednessVerdict | None = Field(
        default=None, description="Populated only when found_in_contract is True."
    )
