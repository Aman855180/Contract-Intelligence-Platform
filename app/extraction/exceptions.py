"""
exceptions.py

Custom exception types for the structured field extraction module.
"""

from __future__ import annotations


class ExtractionError(Exception):
    """Base class for all extraction-related errors."""


class NERPipelineError(ExtractionError):
    """Raised when the HuggingFace NER pipeline fails to load or run inference."""
