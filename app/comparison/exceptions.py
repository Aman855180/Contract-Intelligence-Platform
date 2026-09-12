"""
exceptions.py

Custom exception types for the contract comparison module.
"""

from __future__ import annotations


class ComparisonError(Exception):
    """Base class for all comparison-related errors."""


class InsufficientFieldsError(ComparisonError):
    """Raised when a contract lacks any extractable fields to compare."""
