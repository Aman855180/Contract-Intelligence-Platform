"""
test_llm_provider.py

Unit tests for LLMProvider abstraction, factory, OllamaProvider,
and GeminiProvider.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
import requests

from app.llm.exceptions import GeminiConnectionError, LLMProviderError, OllamaConnectionError
from app.llm.gemini_provider import GeminiProvider
from app.llm.ollama_provider import OllamaProvider
from app.llm.provider import LLMProvider, generate, get_llm_provider, reset_provider_cache


@pytest.fixture(autouse=True)
def clean_provider_cache():
    """Ensure provider cache is cleared before and after each test."""
    reset_provider_cache()
    yield
    reset_provider_cache()


class TestProviderFactory:
    def test_default_provider_is_ollama(self) -> None:
        provider = get_llm_provider()
        assert isinstance(provider, LLMProvider)
        assert isinstance(provider, OllamaProvider)
        assert provider.provider_name == "ollama"
        assert provider.model_name == "llama3.1:8b"

    def test_factory_returns_gemini_provider_when_configured(self) -> None:
        custom_settings = {
            "llm": {
                "provider": "gemini",
                "gemini": {
                    "model_name": "gemini-2.5-flash",
                    "api_key_env": "TEST_GEMINI_KEY",
                },
            }
        }
        provider = get_llm_provider(settings=custom_settings)
        assert isinstance(provider, GeminiProvider)
        assert provider.provider_name == "gemini"
        assert provider.model_name == "gemini-2.5-flash"

    def test_factory_raises_for_unsupported_provider(self) -> None:
        custom_settings = {"llm": {"provider": "unsupported_ai"}}
        with pytest.raises(LLMProviderError) as exc_info:
            get_llm_provider(settings=custom_settings)
        assert "Unsupported LLM provider 'unsupported_ai'" in str(exc_info.value)

    def test_singleton_caching_and_reset(self) -> None:
        p1 = get_llm_provider()
        p2 = get_llm_provider()
        assert p1 is p2

        reset_provider_cache()
        p3 = get_llm_provider()
        assert p3 is not p1


class TestOllamaProvider:
    def test_ollama_provider_generate_success(self) -> None:
        provider = OllamaProvider()
        with patch("app.llm.ollama_client.requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"response": "Provider response test."}
            mock_post.return_value = mock_resp

            result = provider.generate("Summarize contract.")
            assert result == "Provider response test."

    def test_ollama_provider_connection_error(self) -> None:
        provider = OllamaProvider()
        with patch("app.llm.ollama_client.requests.post") as mock_post:
            mock_post.side_effect = requests.exceptions.ConnectionError("Offline")
            with pytest.raises(OllamaConnectionError):
                provider.generate("Test prompt")


class TestGeminiProvider:
    def test_missing_api_key_raises_connection_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("TEST_KEY_ENV", raising=False)
        custom_settings = {
            "llm": {
                "provider": "gemini",
                "gemini": {
                    "api_key_env": "TEST_KEY_ENV",
                    "model_name": "gemini-2.5-flash",
                },
            }
        }
        provider = GeminiProvider(settings=custom_settings)
        with pytest.raises(GeminiConnectionError) as exc_info:
            provider.generate("Test prompt")
        assert "Gemini API key not found" in str(exc_info.value)

    def test_gemini_generate_success_mock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_KEY_ENV", "dummy-test-key")
        custom_settings = {
            "llm": {
                "provider": "gemini",
                "gemini": {
                    "api_key_env": "TEST_KEY_ENV",
                    "model_name": "gemini-2.5-flash",
                },
            }
        }
        provider = GeminiProvider(settings=custom_settings)

        # Mock genai response
        mock_response = MagicMock()
        mock_response.text = "Generated from Gemini mock."

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        # Test modern or fallback mock
        with patch("app.llm.gemini_provider.time.perf_counter", side_effect=[100.0, 100.5]):
            with patch("google.generativeai.GenerativeModel") as mock_model_cls:
                mock_model_instance = MagicMock()
                mock_model_instance.generate_content.return_value = mock_response
                mock_model_cls.return_value = mock_model_instance

                result = provider.generate("Test prompt")
                assert "Gemini mock" in result

    def test_gemini_api_error_raises_connection_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_KEY_ENV", "dummy-test-key")
        custom_settings = {
            "llm": {
                "provider": "gemini",
                "gemini": {
                    "api_key_env": "TEST_KEY_ENV",
                    "model_name": "gemini-2.5-flash",
                },
            }
        }
        provider = GeminiProvider(settings=custom_settings)

        with patch("google.generativeai.GenerativeModel") as mock_model_cls:
            mock_model_instance = MagicMock()
            mock_model_instance.generate_content.side_effect = Exception("API quota exceeded")
            mock_model_cls.return_value = mock_model_instance

            with pytest.raises(GeminiConnectionError) as exc_info:
                provider.generate("Test prompt")
            assert "Gemini API error" in str(exc_info.value)


class TestTopLevelGenerate:
    def test_generate_delegates_to_active_provider(self) -> None:
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.generate.return_value = "Delegated response"

        with patch("app.llm.provider.get_llm_provider", return_value=mock_provider):
            res = generate("Hello prompt", attempt=2)
            assert res == "Delegated response"
            mock_provider.generate.assert_called_once_with("Hello prompt", attempt=2)
