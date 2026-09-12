"""
exceptions.py

Custom exception types for the document ingestion module. Explicit exception
types (rather than bare Exception/ValueError) make failure modes easy to
catch and handle distinctly in the Streamlit frontend (Phase 11).
"""

from __future__ import annotations


class IngestionError(Exception):
    """Base class for all ingestion-related errors."""


class UnsupportedFileTypeError(IngestionError):
    """Raised when an uploaded file's extension is not in the allowed list."""


class FileTooLargeError(IngestionError):
    """Raised when an uploaded file exceeds the configured max file size."""


class EmptyDocumentError(IngestionError):
    """Raised when a document parses successfully but contains no extractable text."""


class DocumentParsingError(IngestionError):
    """Raised when the underlying parsing library (PyMuPDF/python-docx) fails."""
