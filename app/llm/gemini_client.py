"""
gemini_client.py

Gemini API client using google-generativeai SDK.
Only imported when settings.yaml has provider: gemini.
"""

from __future__ import annotations

import os

from app.llm.exceptions import OllamaConnectionError
from app.utils.config_loader import get_settings
from app.utils.logging_setup import get_logger

logger = get_logger(__name__)


def generate_gemini(prompt: str) -> str:
    try:
        import google.generativeai as genai
    except ImportError:
        raise OllamaConnectionError(
            "google-generativeai not installed. Run: pip install google-generativeai"
        )

    settings = get_settings()
    llm_cfg = settings.get("llm", {})
    api_key_env = llm_cfg.get("gemini_api_key_env", "[REDACTED]")
    api_key = os.getenv(api_key_env)

    if not api_key:
        raise OllamaConnectionError(
            f"Gemini API key not found. Set the {api_key_env} environment variable."
        )

    model_name = llm_cfg.get("model_name", "gemini-3.5-flash")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as exc:
        logger.error("Gemini API call failed: %s", exc)
        raise OllamaConnectionError(f"Gemini API error: {exc}") from exc