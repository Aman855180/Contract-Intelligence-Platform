"""
exceptions.py

Custom exception types for the database access layer.
"""

from __future__ import annotations


class DatabaseError(Exception):
    """Base class for all database-related errors."""


class DocumentNotFoundError(DatabaseError):
    """Raised when a requested document_id does not exist in the database."""


class DuplicateDocumentError(DatabaseError):
    """Raised when attempting to insert a document_id that already exists."""
