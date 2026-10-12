"""
ollama_provider.py

Ollama implementation of the LLMProvider interface.
Sends requests to the local Ollama server's /api/generate endpoint.
"""

from __future__ import annotations

from typing import Any

from app.llm.exceptions import OllamaConnectionError
from app.llm.provider import LLMProvider
from app.utils.config_loader import get_settings
from app.utils.logging_setup import get_logger
from app.utils.perf import get_current_llm_attempt

logger = get_logger(__name__)


class OllamaProvider(LLMProvider):
    """
    LLMProvider implementation for local Ollama instances.
    """

    def __init__(self, settings: dict[str, Any] | None = None) -> None:
        cfg = settings if settings is not None else get_settings()
        llm_cfg = cfg.get("llm", {})
        ollama_cfg = llm_cfg.get("ollama", {})

        self._model_name: str = (
            ollama_cfg.get("model_name")
            or llm_cfg.get("model_name", "llama3.1:8b")
        )
        self._base_url: str = (
            ollama_cfg.get("base_url")
            or llm_cfg.get("base_url")
            or llm_cfg.get("ollama_base_url", "http://localhost:11434")
        )
        self._temperature: float = float(llm_cfg.get("temperature", 0.0))
        self._max_tokens: int = int(llm_cfg.get("max_tokens", 2048))
        self._timeout_seconds: int = int(
            ollama_cfg.get("request_timeout_seconds")
            or llm_cfg.get("request_timeout_seconds", 300)
        )

    @property
    def provider_name(self) -> str:
        return "ollama"

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
        Generate text response via Ollama HTTP API.
        Delegates to app.llm.ollama_client._call_ollama_api so existing test mocks remain active.
        """
        if attempt is None:
            attempt = get_current_llm_attempt()

        from app.llm.ollama_client import _call_ollama_api
        return _call_ollama_api(prompt, attempt=attempt)
