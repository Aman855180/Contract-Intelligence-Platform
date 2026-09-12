"""
exceptions.py

Custom exception types for the groundedness (NLI) module.
"""

from __future__ import annotations


class GroundednessError(Exception):
    """Raised when NLI-based groundedness checking fails (e.g. model load failure)."""
