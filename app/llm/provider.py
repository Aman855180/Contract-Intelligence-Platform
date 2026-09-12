"""
provider.py

Defines the common LLMProvider interface and factory for instantiating
the configured provider (OllamaProvider, GeminiProvider).

Application callers (risk classification, report generation, grounded Q&A)
depend strictly on this interface rather than directly coupling to Ollama
or Gemini implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.llm.exceptions import LLMProviderError
from app.utils.config_loader import get_settings
from app.utils.logging_setup import get_logger

logger = get_logger(__name__)

_cached_provider: LLMProvider | None = None


class LLMProvider(ABC):
    """
    Abstract base class for all LLM providers.
    """

    @abstractmethod
    def generate(self, prompt: str, attempt: int | None = None) -> str:
        """
        Generate a text response from the underlying LLM model for the given prompt.

        Args:
            prompt: Formatted prompt text.
            attempt: Attempt counter for retried calls (used for metrics & logging).

        Returns:
            The raw text response from the LLM.
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the unique provider identifier (e.g. 'ollama', 'gemini')."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the active model name for this provider."""
        ...


def get_llm_provider(settings: dict[str, Any] | None = None) -> LLMProvider:
    """
    Return the active LLMProvider instance based on settings.yaml.
    Caches the provider instance as a singleton unless explicit settings are passed.

    Args:
        settings: Optional custom settings dict. If None, loaded from settings.yaml.

    Returns:
        An instantiated LLMProvider (OllamaProvider or GeminiProvider).

    Raises:
        LLMProviderError: If the configured provider is unsupported.
    """
    global _cached_provider
    if _cached_provider is not None and settings is None:
        return _cached_provider

    cfg = settings if settings is not None else get_settings()
    llm_cfg = cfg.get("llm", {})
    provider_type = llm_cfg.get("provider", "ollama").strip().lower()

    provider: LLMProvider
    if provider_type == "ollama":
        from app.llm.ollama_provider import OllamaProvider
        provider = OllamaProvider(cfg)
    elif provider_type == "gemini":
        from app.llm.gemini_provider import GeminiProvider
        provider = GeminiProvider(cfg)
    else:
        raise LLMProviderError(
            f"Unsupported LLM provider '{provider_type}'. "
            "Supported providers: 'ollama', 'gemini'."
        )

    if settings is None:
        _cached_provider = provider

    logger.info("Initialized LLM provider '%s' (model='%s').", provider.provider_name, provider.model_name)
    return provider


def reset_provider_cache() -> None:
    """
    Clear the cached LLMProvider instance.
    Used by tests when testing provider switching or mocking configurations.
    """
    global _cached_provider
    _cached_provider = None


def generate(prompt: str, attempt: int | None = None) -> str:
    """
    Top-level convenience function for generating text using the active LLMProvider.

    Args:
        prompt: Formatted prompt text.
        attempt: Attempt counter for retried calls.

    Returns:
        The raw text response from the LLM.
    """
    return get_llm_provider().generate(prompt, attempt=attempt)
