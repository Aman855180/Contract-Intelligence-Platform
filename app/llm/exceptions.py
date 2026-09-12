"""
exceptions.py

Custom exception types for the LLM module (Ollama client, prompt loading,
response parsing).
"""

from __future__ import annotations


class LLMError(Exception):
    """Base class for all LLM-related errors."""


class OllamaConnectionError(LLMError):
    """Raised when the Ollama server cannot be reached or times out."""


class LLMResponseParsingError(LLMError):
    """Raised when the LLM's raw text response doesn't match the expected format."""


class PromptNotFoundError(LLMError):
    """Raised when a requested prompt template file does not exist."""
