"""
exceptions.py

Custom exception types for the report generation module.
"""

from __future__ import annotations


class ReportGenerationError(Exception):
    """Base class for all report generation errors."""


class ReportValidationError(ReportGenerationError):
    """
    Raised when the LLM fails to produce a Pydantic-valid report JSON after
    all configured retries are exhausted.
    """
