"""
exceptions.py

Custom exception types for the chunking module.
"""

from __future__ import annotations


class ChunkingError(Exception):
    """Raised when a document cannot be chunked (e.g. invalid configuration)."""
