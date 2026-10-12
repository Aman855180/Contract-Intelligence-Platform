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


class GeminiConnectionError(LLMError):
    """Raised when the Gemini API cannot be reached, times out, or fails authentication."""


class LLMProviderError(LLMError):
    """Raised when an unknown or misconfigured LLM provider is requested."""


class LLMPermanentError(LLMError):
    """Raised for non-retryable errors such as invalid credentials, bad requests, or missing models."""


class GeminiPermanentError(GeminiConnectionError, LLMPermanentError):
    """Raised for permanent non-retryable Gemini errors (invalid key, bad arguments)."""


class LLMResponseParsingError(LLMError):
    """Raised when the LLM's raw text response doesn't match the expected format."""


class PromptNotFoundError(LLMError):
    """Raised when a requested prompt template file does not exist."""
