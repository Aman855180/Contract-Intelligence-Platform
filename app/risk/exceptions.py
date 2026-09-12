"""
exceptions.py

Custom exception types for the risk detection module.
"""

from __future__ import annotations


class RiskDetectionError(Exception):
    """Base class for all risk-detection errors."""


class LLMClassificationError(RiskDetectionError):
    """
    Raised when the LLM fails to produce valid structured risk output after
    all configured retries are exhausted.
    """
