"""
gemini_provider.py

Gemini implementation of the LLMProvider interface.
Supports Google Gemini models via the official Google GenAI SDK
with fallback to google.generativeai if google-genai is not yet installed.
"""

from __future__ import annotations

import os
import time
from typing import Any

from app.llm.exceptions import GeminiConnectionError
from app.llm.provider import LLMProvider
from app.utils.config_loader import get_settings
from app.utils.logging_setup import get_logger
from app.utils.perf import get_current_llm_attempt, log_llm_perf

logger = get_logger(__name__)


class GeminiProvider(LLMProvider):
    """
    LLMProvider implementation for Google Gemini models.
    """

    def __init__(self, settings: dict[str, Any] | None = None) -> None:
        cfg = settings if settings is not None else get_settings()
        llm_cfg = cfg.get("llm", {})
        gemini_cfg = llm_cfg.get("gemini", {})

        self._api_key_env: str = (
            gemini_cfg.get("api_key_env")
            or llm_cfg.get("gemini_api_key_env", "GEMINI_API_KEY")
        )
        self._model_name: str = (
            gemini_cfg.get("model_name")
            or llm_cfg.get("gemini_model_name")
            or "gemini-2.5-flash"
        )
        self._temperature: float = float(llm_cfg.get("temperature", 0.0))
        self._timeout_seconds: int = int(
            gemini_cfg.get("request_timeout_seconds")
            or gemini_cfg.get("timeout_seconds")
            or llm_cfg.get("request_timeout_seconds", 30)
        )

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str, attempt: int | None = None) -> str:
        """
        Generate text response from Google Gemini.
        """
        if attempt is None:
            attempt = get_current_llm_attempt()

        api_key = os.getenv(self._api_key_env)
        if not api_key:
            raise GeminiConnectionError(
                f"Gemini API key not found. Set the {self._api_key_env} environment variable."
            )

        prompt_chars = len(prompt)
        logger.info(
            "[PERF] operation=llm_request_start provider=gemini model=%s prompt_chars=%d attempt=%d",
            self._model_name,
            prompt_chars,
            attempt,
        )
        t0 = time.perf_counter()

        try:
            # Try modern google-genai SDK first
            try:
                from google import genai
                from google.genai import types

                client = genai.Client(api_key=api_key)
                config = types.GenerateContentConfig(temperature=self._temperature)
                response = client.models.generate_content(
                    model=self._model_name,
                    contents=prompt,
                    config=config,
                )
                res_text = response.text or ""

            except ImportError:
                # Fallback to google-generativeai
                import google.generativeai as genai_legacy

                genai_legacy.configure(api_key=api_key)
                model = genai_legacy.GenerativeModel(self._model_name)
                response = model.generate_content(
                    prompt,
                    generation_config={"temperature": self._temperature},
                )
                res_text = response.text or ""

        except Exception as exc:
            duration = time.perf_counter() - t0
            logger.error(
                "[PERF] operation=llm_request_failed duration=%.4fs provider=gemini model=%s attempt=%d",
                duration,
                self._model_name,
                attempt,
            )
            logger.error("Gemini API call failed: %s", exc)
            raise GeminiConnectionError(f"Gemini API error: {exc}") from exc

        duration = time.perf_counter() - t0
        response_chars = len(res_text)

        logger.info(
            "[PERF] operation=llm_request_end duration=%.4fs provider=gemini model=%s attempt=%d",
            duration,
            self._model_name,
            attempt,
        )
        log_llm_perf(
            provider="gemini",
            model=self._model_name,
            duration=duration,
            prompt_chars=prompt_chars,
            response_chars=response_chars,
            attempt=attempt,
            log_logger=logger,
        )

        return res_text
