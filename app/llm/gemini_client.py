"""
gemini_client.py

Gemini API client using google-generativeai SDK.
Only imported when settings.yaml has provider: gemini.
"""

from __future__ import annotations

import os
import time

from app.llm.exceptions import OllamaConnectionError
from app.utils.config_loader import get_settings
from app.utils.logging_setup import get_logger
from app.utils.perf import log_llm_perf

logger = get_logger(__name__)


def generate_gemini(prompt: str, attempt: int = 1) -> str:
    try:
        import google.generativeai as genai
    except ImportError:
        raise OllamaConnectionError(
            "google-generativeai not installed. Run: pip install google-generativeai"
        )

    settings = get_settings()
    llm_cfg = settings.get("llm", {})
    api_key_env = llm_cfg.get("gemini_api_key_env", "GEMINI_API_KEY")
    api_key = os.getenv(api_key_env)

    if not api_key:
        raise OllamaConnectionError(
            f"Gemini API key not found. Set the {api_key_env} environment variable."
        )

    model_name = llm_cfg.get("model_name", "gemini-3.5-flash")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    prompt_chars = len(prompt)
    logger.info(
        "[PERF] operation=llm_request_start provider=gemini model=%s prompt_chars=%d attempt=%d",
        model_name,
        prompt_chars,
        attempt,
    )
    t0 = time.perf_counter()

    try:
        response = model.generate_content(prompt)
        res_text = response.text
    except Exception as exc:
        duration = time.perf_counter() - t0
        logger.error(
            "[PERF] operation=llm_request_failed duration=%.4fs provider=gemini model=%s attempt=%d",
            duration,
            model_name,
            attempt,
        )
        logger.error("Gemini API call failed: %s", exc)
        raise OllamaConnectionError(f"Gemini API error: {exc}") from exc

    duration = time.perf_counter() - t0
    response_chars = len(res_text)

    logger.info(
        "[PERF] operation=llm_request_end duration=%.4fs provider=gemini model=%s attempt=%d",
        duration,
        model_name,
        attempt,
    )
    log_llm_perf(
        provider="gemini",
        model=model_name,
        duration=duration,
        prompt_chars=prompt_chars,
        response_chars=response_chars,
        attempt=attempt,
        log_logger=logger,
    )

    return res_text