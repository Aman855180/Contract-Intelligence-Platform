"""
provider.py

Defines the common LLMProvider interface and factory for instantiating
the configured provider (OllamaProvider, GeminiProvider).

Application callers (risk classification, report generation, grounded Q&A)
depend strictly on this interface rather than directly coupling to a provider
implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.llm.exceptions import LLMProviderError
from app.utils.config_loader import get_settings
from app.utils.logging_setup import get_logger
from app.utils.runtime_config import get_runtime_setting

logger = get_logger(__name__)

_cached_provider: LLMProvider | None = None
_provider_cache: dict[str, LLMProvider] = {}


class LLMProvider(ABC):
    """
    Abstract base class for all LLM providers.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        attempt: int | None = None,
        response_schema: Any | None = None,
    ) -> str:
        """
        Generate a text response from the underlying LLM model for the given prompt.

        Args:
            prompt: Formatted prompt text.
            attempt: Attempt counter for retried calls (used for metrics & logging).
            response_schema: Optional JSON schema / TypedDict / structure for structured outputs.

        Returns:
            The raw text response from the LLM.
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the unique provider identifier (e.g. 'ollama' or 'gemini')."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the active model name for this provider."""
        ...


def get_llm_provider(
    settings: dict[str, Any] | None = None,
    provider_type: str | None = None,
) -> LLMProvider:
    """
    Return an instantiated LLMProvider based on configuration or explicit provider_type.
    Caches provider instances per provider_type unless custom settings are passed.

    Args:
        settings: Optional custom settings dict. If None, loaded from settings.yaml.
        provider_type: Optional provider name ('ollama' or 'gemini'). If None,
            resolved from LLM_PROVIDER or settings['llm']['provider'] (default: 'ollama').

    Returns:
        An instantiated LLMProvider for the selected provider.

    Raises:
        LLMProviderError: If the configured provider is unsupported.
    """
    global _cached_provider, _provider_cache

    cfg = settings if settings is not None else get_settings()
    llm_cfg = cfg.get("llm", {})

    target_type = (
        provider_type
        if provider_type is not None
        else get_runtime_setting("LLM_PROVIDER") or llm_cfg.get("provider", "ollama")
    ).strip().lower()

    if settings is None and target_type in _provider_cache:
        return _provider_cache[target_type]

    provider: LLMProvider
    if target_type == "ollama":
        from app.llm.ollama_provider import OllamaProvider
        provider = OllamaProvider(cfg)
    elif target_type == "gemini":
        from app.llm.gemini_provider import GeminiProvider
        provider = GeminiProvider(cfg)
    else:
        raise LLMProviderError(
            f"Unsupported LLM provider '{target_type}'. "
            "Supported providers: 'ollama', 'gemini'."
        )

    if settings is None:
        _provider_cache[target_type] = provider
        _cached_provider = provider

    logger.info("Initialized LLM provider '%s' (model='%s').", provider.provider_name, provider.model_name)
    return provider


def get_risk_provider(settings: dict[str, Any] | None = None) -> LLMProvider:
    """
    Return the LLMProvider configured specifically for risk assessment.
    Resolves 'llm.risk_provider' (defaulting to 'llm.provider' or 'ollama').

    Args:
        settings: Optional custom settings dict.

    Returns:
        The configured LLMProvider for risk assessment.
    """
    cfg = settings if settings is not None else get_settings()
    llm_cfg = cfg.get("llm", {})
    risk_provider_type = (
        get_runtime_setting("LLM_PROVIDER")
        or llm_cfg.get("risk_provider")
        or llm_cfg.get("workflows", {}).get("risk")
        or llm_cfg.get("provider", "ollama")
    ).strip().lower()
    return get_llm_provider(settings=settings, provider_type=risk_provider_type)


def get_report_provider(settings: dict[str, Any] | None = None) -> LLMProvider:
    """
    Return the LLMProvider configured specifically for report generation.
    Resolves 'llm.report_provider' (defaulting to 'llm.provider' or 'ollama').

    Args:
        settings: Optional custom settings dict.

    Returns:
        The configured LLMProvider for report generation.
    """
    cfg = settings if settings is not None else get_settings()
    llm_cfg = cfg.get("llm", {})
    report_provider_type = (
        get_runtime_setting("LLM_PROVIDER")
        or llm_cfg.get("report_provider")
        or llm_cfg.get("workflows", {}).get("reports")
        or llm_cfg.get("provider", "ollama")
    ).strip().lower()
    return get_llm_provider(settings=settings, provider_type=report_provider_type)


def get_qa_provider(settings: dict[str, Any] | None = None) -> LLMProvider:
    """
    Return the LLMProvider configured specifically for Q&A.
    Resolves 'llm.qa_provider' (defaulting to 'llm.provider' or 'ollama').

    Args:
        settings: Optional custom settings dict.

    Returns:
        The configured LLMProvider for Q&A.
    """
    cfg = settings if settings is not None else get_settings()
    llm_cfg = cfg.get("llm", {})
    qa_provider_type = (
        get_runtime_setting("LLM_PROVIDER")
        or llm_cfg.get("qa_provider")
        or llm_cfg.get("workflows", {}).get("qa")
        or llm_cfg.get("provider", "ollama")
    ).strip().lower()
    return get_llm_provider(settings=settings, provider_type=qa_provider_type)


def reset_provider_cache() -> None:
    """
    Clear the cached LLMProvider instances.
    Used by tests when testing provider switching or mocking configurations.
    """
    global _cached_provider, _provider_cache
    _cached_provider = None
    _provider_cache.clear()


def generate(
    prompt: str,
    attempt: int | None = None,
    provider_type: str | None = None,
    response_schema: Any | None = None,
) -> str:
    """
    Top-level convenience function for generating text using the active LLMProvider.

    Args:
        prompt: Formatted prompt text.
        attempt: Attempt counter for retried calls.
        provider_type: Optional provider override ('ollama' or 'gemini').
        response_schema: Optional schema for structured output.

    Returns:
        The raw text response from the LLM.
    """
    return get_llm_provider(provider_type=provider_type).generate(
        prompt,
        attempt=attempt,
        response_schema=response_schema,
    )
