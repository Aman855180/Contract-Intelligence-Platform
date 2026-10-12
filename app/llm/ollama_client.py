"""
ollama_client.py

Thin HTTP wrapper around the local Ollama server's /api/generate endpoint.
Configuration (model name, base URL, temperature, timeout) is read from
settings.yaml's llm section.

The actual HTTP call is isolated in a single function (_call_ollama_api)
so tests can monkeypatch just that function to verify prompt construction
and response parsing without requiring a running Ollama server.
"""

from __future__ import annotations

import time
from typing import Any

import requests

from app.llm.exceptions import OllamaConnectionError
from app.utils.config_loader import get_settings
from app.utils.logging_setup import get_logger
from app.utils.perf import get_current_llm_attempt, log_llm_perf

logger = get_logger(__name__)


def _call_ollama_api(prompt: str, attempt: int = 1) -> str:
    """
    Send a single prompt to the configured Ollama /api/generate endpoint
    and return the raw generated text.

    Args:
        prompt: The fully formatted prompt string.
        attempt: Attempt counter for retried calls.

    Returns:
        The raw text response from the LLM.

    Raises:
        OllamaConnectionError: If the request fails (connection refused,
            timeout, non-200 status, or malformed JSON response).
    """
    settings = get_settings()
    llm_cfg = settings.get("llm", {})
    ollama_cfg = llm_cfg.get("ollama", {})
    base_url = ollama_cfg.get("base_url") or llm_cfg.get("base_url", "http://localhost:11434")
    model_name = ollama_cfg.get("model_name") or llm_cfg.get("model_name", "llama3.1:8b")
    temperature = llm_cfg.get("temperature", 0.0)
    max_tokens = llm_cfg.get("max_tokens", 2048)
    timeout_seconds = ollama_cfg.get("request_timeout_seconds") or llm_cfg.get("request_timeout_seconds", 180)

    payload: dict[str, Any] = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    prompt_chars = len(prompt)
    logger.info(
        "[PERF] operation=llm_request_start provider=ollama model=%s prompt_chars=%d attempt=%d",
        model_name,
        prompt_chars,
        attempt,
    )
    t0 = time.perf_counter()

    try:
        response = requests.post(f"{base_url}/api/generate", json=payload, timeout=timeout_seconds)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.Timeout as exc:
        duration = time.perf_counter() - t0
        logger.error(
            "[PERF] operation=llm_request_timeout duration=%.4fs provider=ollama model=%s attempt=%d",
            duration,
            model_name,
            attempt,
        )
        logger.error("Ollama request timed out after %ds at %s", timeout_seconds, base_url)
        raise OllamaConnectionError(
            f"OLLAMA_TIMEOUT: Ollama is running but did not respond within "
            f"{timeout_seconds}s. Increase llm.request_timeout_seconds in "
            f"config/settings.yaml and try again."
        ) from exc
    except requests.exceptions.ConnectionError as exc:
        duration = time.perf_counter() - t0
        logger.error(
            "[PERF] operation=llm_request_connection_error duration=%.4fs provider=ollama model=%s attempt=%d",
            duration,
            model_name,
            attempt,
        )
        logger.error("Cannot connect to Ollama at %s: %s", base_url, exc)
        raise OllamaConnectionError(
            f"OLLAMA_UNREACHABLE: Cannot connect to Ollama at {base_url}. "
            f"Start it with: ollama serve"
        ) from exc
    except requests.exceptions.RequestException as exc:
        duration = time.perf_counter() - t0
        logger.error(
            "[PERF] operation=llm_request_failed duration=%.4fs provider=ollama model=%s attempt=%d",
            duration,
            model_name,
            attempt,
        )
        logger.error("Ollama request failed at %s: %s", base_url, exc)
        raise OllamaConnectionError(f"Ollama request failed: {exc}") from exc
    except ValueError as exc:  # JSON decoding failure
        duration = time.perf_counter() - t0
        logger.error(
            "[PERF] operation=llm_request_invalid_json duration=%.4fs provider=ollama model=%s attempt=%d",
            duration,
            model_name,
            attempt,
        )
        logger.error("Ollama returned a non-JSON response: %s", exc)
        raise OllamaConnectionError(f"Ollama returned an invalid JSON response: {exc}") from exc

    duration = time.perf_counter() - t0

    if "response" not in data:
        logger.error("Ollama response missing 'response' field: %s", data)
        raise OllamaConnectionError(f"Unexpected Ollama response shape (missing 'response' field): {data}")

    # Parse output
    res_text = data.get("response", "")
    response_chars = len(res_text)

    prompt_tokens = data.get("prompt_eval_count", 0)
    output_tokens = data.get("eval_count", 0)

    logger.info(
        "[PERF] operation=llm_request_end duration=%.4fs provider=ollama model=%s attempt=%d prompt_tokens=%d output_tokens=%d total_tokens=%d",
        duration,
        model_name,
        attempt,
        prompt_tokens,
        output_tokens,
        prompt_tokens + output_tokens,
    )
    log_llm_perf(
        provider="ollama",
        model=model_name,
        duration=duration,
        prompt_chars=prompt_chars,
        response_chars=response_chars,
        attempt=attempt,
        log_logger=logger,
    )

    return res_text


def generate(prompt: str, attempt: int | None = None) -> str:
    """
    Generate text using the configured LLM provider (kept for backward compatibility).
    """
    if attempt is None:
        attempt = get_current_llm_attempt()
    from app.llm.provider import get_llm_provider
    return get_llm_provider().generate(prompt, attempt=attempt)