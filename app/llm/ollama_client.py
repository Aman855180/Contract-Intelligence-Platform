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

from typing import Any

import requests

from app.llm.exceptions import OllamaConnectionError
from app.utils.config_loader import get_settings
from app.utils.logging_setup import get_logger

logger = get_logger(__name__)


def _call_ollama_api(prompt: str) -> str:
    """
    Send a single prompt to the configured Ollama /api/generate endpoint
    and return the raw generated text.

    Args:
        prompt: The fully formatted prompt string.

    Returns:
        The raw text response from the LLM.

    Raises:
        OllamaConnectionError: If the request fails (connection refused,
            timeout, non-200 status, or malformed JSON response).
    """
    settings = get_settings()
    llm_cfg = settings.get("llm", {})
    base_url = llm_cfg.get("base_url", "http://localhost:11434")
    model_name = llm_cfg.get("model_name", "llama3.1:8b")
    temperature = llm_cfg.get("temperature", 0.0)
    max_tokens = llm_cfg.get("max_tokens", 2048)
    timeout_seconds = llm_cfg.get("request_timeout_seconds", 120)

    payload: dict[str, Any] = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    try:
        response = requests.post(f"{base_url}/api/generate", json=payload, timeout=timeout_seconds)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.Timeout as exc:
        logger.error("Ollama request timed out after %ds at %s", timeout_seconds, base_url)
        raise OllamaConnectionError(
            f"OLLAMA_TIMEOUT: Ollama is running but did not respond within "
            f"{timeout_seconds}s. Increase llm.request_timeout_seconds in "
            f"config/settings.yaml and try again."
        ) from exc
    except requests.exceptions.ConnectionError as exc:
        logger.error("Cannot connect to Ollama at %s: %s", base_url, exc)
        raise OllamaConnectionError(
            f"OLLAMA_UNREACHABLE: Cannot connect to Ollama at {base_url}. "
            f"Start it with: ollama serve"
        ) from exc
    except requests.exceptions.RequestException as exc:
        logger.error("Ollama request failed at %s: %s", base_url, exc)
        raise OllamaConnectionError(f"Ollama request failed: {exc}") from exc
    except ValueError as exc:  # JSON decoding failure
        logger.error("Ollama returned a non-JSON response: %s", exc)
        raise OllamaConnectionError(f"Ollama returned an invalid JSON response: {exc}") from exc

    if "response" not in data:
        logger.error("Ollama response missing 'response' field: %s", data)
        raise OllamaConnectionError(f"Unexpected Ollama response shape (missing 'response' field): {data}")

    return data["response"]


def generate(prompt: str) -> str:
    settings = get_settings()
    provider = settings.get("llm", {}).get("provider", "ollama")

    if provider == "gemini":
        from app.llm.gemini_client import generate_gemini
        return generate_gemini(prompt)

    # Default: Ollama
    return _call_ollama_api(prompt)