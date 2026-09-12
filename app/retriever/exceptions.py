"""
exceptions.py

Custom exception types for the hybrid retrieval module.
"""

from __future__ import annotations


class RetrievalError(Exception):
    """Raised when hybrid retrieval fails (e.g. embedding or index failure)."""
