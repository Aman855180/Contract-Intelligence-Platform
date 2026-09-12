"""
exceptions.py

Custom exception types for the embeddings module.
"""

from __future__ import annotations


class EmbeddingError(Exception):
    """Raised when embedding generation fails (e.g. model load failure)."""
