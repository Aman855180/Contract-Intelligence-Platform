"""
gemini_provider.py

Gemini implementation of the LLMProvider interface.
Supports Google Gemini models via the official Google GenAI SDK
with fallback to google.generativeai if google-genai is not yet installed.
"""

from __future__ import annotations

import re
import time
from typing import Any

from app.llm.exceptions import GeminiConnectionError, GeminiPermanentError
from app.llm.provider import LLMProvider
from app.utils.config_loader import get_settings
from app.utils.logging_setup import get_logger
from app.utils.perf import get_current_llm_attempt, log_llm_perf
from app.utils.runtime_config import get_runtime_setting

logger = get_logger(__name__)

_SENSITIVE_PATTERNS = (
    # API keys may use any provider-specific format, so redact the complete
    # value after a recognized key label rather than relying on a key prefix.
    re.compile(r"(?i)((?:[?&]|\b)(?:key|api[_-]?key|gemini_api_key|google_api_key|x-goog-api-key)\s*[:=]\s*)[^&\s,;\"'<>]+"),
    re.compile(r"(?i)(authorization\s*[:=]\s*)(?:bearer\s+)?[^\s,;\"'<>]+"),
    re.compile(r"AIzaSy[A-Za-z0-9_\-]+", re.IGNORECASE),
)


def _sanitize_error_message(msg: str, secrets: tuple[str, ...] = ()) -> str:
    """Scrub API keys and authorization values before logging or surfacing errors."""
    for secret in secrets:
        if secret:
            msg = msg.replace(secret, "[REDACTED_SECRET]")
    for pattern in _SENSITIVE_PATTERNS:
        replacement = r"\1[REDACTED_SECRET]" if pattern.pattern.startswith("(?i)(") else "[REDACTED_SECRET]"
        msg = pattern.sub(replacement, msg)
    return msg


def _safe_sdk_diagnostics(exc: Exception, api_key: str, prompt: str) -> tuple[str, str, str]:
    """Extract bounded SDK error fields without logging request/credential data."""
    status_value = None
    for attr in ("status_code", "code", "status"):
        value = getattr(exc, attr, None)
        if value is not None and isinstance(value, (str, int)):
            status_value = value
            break
    if status_value is None:
        response = getattr(exc, "response", None)
        value = getattr(response, "status_code", None) if response is not None else None
        if isinstance(value, (str, int)):
            status_value = value

    error_code = getattr(exc, "error_code", None)
    if error_code is None:
        error_code = getattr(exc, "code", None)

    def safe_scalar(value: Any) -> str:
        if not isinstance(value, (str, int)):
            return "unknown"
        value = _sanitize_error_message(str(value), (api_key,))
        value = re.sub(r"[^A-Za-z0-9_. -]", "", value)[:80]
        return value or "unknown"

    # SDK APIError exposes `message`; do not stringify the exception because
    # that can include URLs, headers, or serialized request details.
    message = getattr(exc, "message", None)
    if not isinstance(message, str):
        message = "No safe structured error message available."
    if prompt:
        message = message.replace(prompt, "[REDACTED_REQUEST_CONTENT]")
    message = _sanitize_error_message(message, (api_key,))
    message = re.sub(r"(?i)https?://\S+", "[REDACTED_URL]", message)
    message = re.sub(r"(?i)(authorization\s*[:=]\s*)(?:bearer\s+)?[^\s,;]+", r"\1[REDACTED_SECRET]", message)
    message = re.sub(r"\s+", " ", message).strip()[:240]
    return safe_scalar(status_value), safe_scalar(error_code), message


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

    def generate(
        self,
        prompt: str,
        attempt: int | None = None,
        response_schema: Any | None = None,
    ) -> str:
        """
        Generate text response from Google Gemini with optional structured JSON schema.
        """
        if attempt is None:
            attempt = get_current_llm_attempt()

        api_key = get_runtime_setting(self._api_key_env)
        if not api_key:
            raise GeminiConnectionError(
                f"Gemini API key not found. Set the {self._api_key_env} environment variable."
            )

        prompt_chars = len(prompt)
        logger.info(
            "[PERF] operation=llm_request_start provider=gemini model=%s prompt_chars=%d attempt=%d structured_output=%s",
            self._model_name,
            prompt_chars,
            attempt,
            response_schema is not None,
        )
        t0 = time.perf_counter()

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            config_kwargs: dict[str, Any] = {"temperature": self._temperature}
            if response_schema is not None:
                config_kwargs["response_mime_type"] = "application/json"
                config_kwargs["response_schema"] = response_schema

            config = types.GenerateContentConfig(**config_kwargs)
            response = client.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=config,
            )
            res_text = response.text or ""

        except Exception as exc:
            duration = time.perf_counter() - t0
            status, error_code, safe_message = _safe_sdk_diagnostics(exc, api_key, prompt)
            logger.error(
                "[PERF] operation=llm_request_failed duration=%.4fs provider=gemini model=%s attempt=%d structured_output=%s",
                duration,
                self._model_name,
                attempt,
                response_schema is not None,
            )
            logger.error(
                "Gemini SDK request failed type=%s http_or_status=%s error_code=%s safe_message=%s",
                type(exc).__name__, status, error_code, safe_message,
            )

            # Classify from the private exception text, but never log or expose it.
            lower_msg = str(exc).lower()
            if any(term in lower_msg for term in ("api_key_invalid", "unauthenticated", "permission_denied", "not_found")):
                logger.warning("[LLM_PERMANENT_ERROR] Non-retryable Gemini error detected.")
                raise GeminiPermanentError("Gemini API error (permanent).") from None

            if any(term in lower_msg for term in ("quota", "429", "resource_exhausted")):
                logger.warning("[LLM_QUOTA_EXHAUSTED] Gemini request quota exceeded.")
                raise GeminiConnectionError("Gemini API error (quota exhausted).") from None

            raise GeminiConnectionError("Gemini API error.") from None

        duration = time.perf_counter() - t0
        response_chars = len(res_text)

        prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) if hasattr(response, "usage_metadata") and response.usage_metadata else 0
        output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) if hasattr(response, "usage_metadata") and response.usage_metadata else 0
        total_tokens = getattr(response.usage_metadata, "total_token_count", 0) if hasattr(response, "usage_metadata") and response.usage_metadata else 0

        logger.info(
            "[PERF] operation=llm_request_end duration=%.4fs provider=gemini model=%s attempt=%d prompt_tokens=%d output_tokens=%d total_tokens=%d",
            duration,
            self._model_name,
            attempt,
            prompt_tokens,
            output_tokens,
            total_tokens
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
