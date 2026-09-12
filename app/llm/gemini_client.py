"""
gemini_client.py

Gemini API client wrapper for backward compatibility.
Delegates to GeminiProvider.
"""

from __future__ import annotations

from app.llm.gemini_provider import GeminiProvider


def generate_gemini(prompt: str, attempt: int = 1) -> str:
    """
    Generate text using GeminiProvider (kept for backward compatibility).
    """
    provider = GeminiProvider()
    return provider.generate(prompt, attempt=attempt)