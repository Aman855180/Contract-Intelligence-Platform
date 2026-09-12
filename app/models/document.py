"""
document.py

Pydantic schemas for raw document ingestion and chunking.
These models define the canonical data shapes used between the ingestion,
chunking, embedding, and database layers.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class DocumentType(str, Enum):
    """Supported contract document formats."""

    PDF = "pdf"
    DOCX = "docx"


class DocumentMetadata(BaseModel):
    """
    Metadata describing an uploaded contract document.
    Persisted in SQLite (app/database) — never paired with embeddings there.
    """

    document_id: str = Field(..., description="Stable unique identifier (UUID4 hex string).")
    contract_name: str = Field(..., min_length=1, description="Human-readable contract/file name.")
    document_type: DocumentType
    upload_date: datetime = Field(default_factory=lambda: datetime.now(UTC))
    page_count: int = Field(..., ge=1)
    file_size_bytes: int = Field(..., ge=0)
    source_path: str = Field(..., description="Path on disk where the raw file is stored.")

    @field_validator("contract_name")
    @classmethod
    def strip_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("contract_name cannot be empty after stripping whitespace.")
        return v


class PageText(BaseModel):
    """Extracted raw text for a single page of a document."""

    page_number: int = Field(..., ge=1)
    text: str = Field(..., description="Raw extracted text for this page.")


class ParsedDocument(BaseModel):
    """
    Full result of parsing a single uploaded document: metadata plus the
    per-page extracted text. This is the output of app/ingestion and the
    input to app/chunking.
    """

    metadata: DocumentMetadata
    pages: list[PageText] = Field(..., min_length=1)


class DocumentChunk(BaseModel):
    """
    A single overlapping text chunk produced by app/chunking, ready to be
    embedded and stored in ChromaDB. chunk_size/chunk_overlap come from
    config/retrieval.yaml at chunk-creation time, not from this schema.
    """

    chunk_id: str = Field(..., description="Stable unique identifier (UUID4 hex string).")
    document_id: str = Field(..., description="Foreign key to DocumentMetadata.document_id.")
    contract_name: str
    page_number: int = Field(..., ge=1)
    chunk_index: int = Field(..., ge=0, description="Ordinal position of this chunk within the document.")
    text: str = Field(..., min_length=1)
    char_start: int = Field(..., ge=0, description="Start offset of this chunk within the page text.")
    char_end: int = Field(..., ge=0, description="End offset of this chunk within the page text.")

    @field_validator("char_end")
    @classmethod
    def end_after_start(cls, v: int, info) -> int:
        char_start = info.data.get("char_start")
        if char_start is not None and v < char_start:
            raise ValueError("char_end must be >= char_start.")
        return v
