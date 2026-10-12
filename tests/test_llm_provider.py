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

    def test_streamlit_secret_can_select_gemini_for_all_workflows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.llm.provider import get_qa_provider, get_report_provider, get_risk_provider
        from app.utils import runtime_config

        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.setattr(
            runtime_config,
            "_streamlit_secret",
            lambda name: "gemini" if name == "LLM_PROVIDER" else None,
        )

        assert isinstance(get_qa_provider(), GeminiProvider)
        assert isinstance(get_risk_provider(), GeminiProvider)
        assert isinstance(get_report_provider(), GeminiProvider)

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

    def test_get_risk_provider_returns_gemini_when_configured(self) -> None:
        custom_settings = {
            "llm": {
                "provider": "ollama",
                "risk_provider": "gemini",
                "gemini": {
                    "model_name": "gemini-3.6-flash",
                    "api_key_env": "TEST_GEMINI_KEY",
                },
            }
        }
        from app.llm.provider import get_risk_provider
        provider = get_risk_provider(settings=custom_settings)
        assert isinstance(provider, GeminiProvider)
        assert provider.provider_name == "gemini"

    def test_get_risk_provider_falls_back_to_default_provider(self) -> None:
        custom_settings = {
            "llm": {
                "provider": "ollama",
            }
        }
        from app.llm.provider import get_risk_provider
        provider = get_risk_provider(settings=custom_settings)
        assert isinstance(provider, OllamaProvider)
        assert provider.provider_name == "ollama"

    def test_get_report_provider_returns_gemini_when_configured(self) -> None:
        custom_settings = {
            "llm": {
                "provider": "ollama",
                "report_provider": "gemini",
                "gemini": {
                    "model_name": "gemini-3.6-flash",
                    "api_key_env": "TEST_GEMINI_KEY",
                },
            }
        }
        from app.llm.provider import get_report_provider
        provider = get_report_provider(settings=custom_settings)
        assert isinstance(provider, GeminiProvider)
        assert provider.provider_name == "gemini"

    def test_get_report_provider_falls_back_to_default_provider(self) -> None:
        custom_settings = {
            "llm": {
                "provider": "ollama",
            }
        }
        from app.llm.provider import get_report_provider
        provider = get_report_provider(settings=custom_settings)
        assert isinstance(provider, OllamaProvider)
        assert provider.provider_name == "ollama"

    def test_explicit_provider_type_routing(self) -> None:
        custom_settings = {
            "llm": {
                "provider": "ollama",
                "gemini": {
                    "model_name": "gemini-3.6-flash",
                    "api_key_env": "TEST_GEMINI_KEY",
                },
            }
        }
        ollama = get_llm_provider(settings=custom_settings, provider_type="ollama")
        gemini = get_llm_provider(settings=custom_settings, provider_type="gemini")
        assert isinstance(ollama, OllamaProvider)
        assert isinstance(gemini, GeminiProvider)


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
    def test_streamlit_secret_supplies_api_key_when_environment_is_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.llm import gemini_provider

        monkeypatch.delenv("TEST_KEY_ENV", raising=False)
        monkeypatch.setattr(
            gemini_provider,
            "get_runtime_setting",
            lambda name: "test-secret" if name == "TEST_KEY_ENV" else None,
        )
        provider = GeminiProvider(settings={
            "llm": {"gemini": {"api_key_env": "TEST_KEY_ENV", "model_name": "gemini-3.6-flash"}}
        })
        response = MagicMock(text="ok")
        client = MagicMock()
        client.models.generate_content.return_value = response

        with patch("google.genai.Client", return_value=client) as client_factory:
            assert provider.generate("Reply ok") == "ok"

        client_factory.assert_called_once_with(api_key="test-secret")

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

        # GeminiProvider uses the google.genai SDK.
        with patch("app.llm.gemini_provider.time.perf_counter", side_effect=[100.0, 100.5]):
            mock_client.models.generate_content.return_value = mock_response
            with patch("google.genai.Client", return_value=mock_client):
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

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("API quota exceeded")
        with patch("google.genai.Client", return_value=mock_client):

            with pytest.raises(GeminiConnectionError) as exc_info:
                provider.generate("Test prompt")
            assert "Gemini API error" in str(exc_info.value)

    def test_gemini_error_logs_safe_structured_diagnostics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        secret = "test-secret-never-log"
        monkeypatch.setenv("TEST_KEY_ENV", secret)
        provider = GeminiProvider(settings={
            "llm": {"gemini": {"api_key_env": "TEST_KEY_ENV", "model_name": "gemini-3.6-flash"}}
        })

        class FakeApiError(Exception):
            code = 400
            message = f"Bad request for key={secret}"

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = FakeApiError("untrusted raw text")
        with patch("google.genai.Client", return_value=mock_client), patch(
            "app.llm.gemini_provider.logger.error"
        ) as log_error:
            with pytest.raises(GeminiConnectionError):
                provider.generate("private contract prompt")

        logs = " ".join(str(part) for call in log_error.call_args_list for part in call.args)
        assert "FakeApiError" in logs
        assert "http_or_status=%s" in logs and " 400 " in logs
        assert "Bad request for key=[REDACTED_SECRET]" in logs
        assert secret not in logs
        assert "private contract prompt" not in logs

    def test_gemini_generate_with_response_schema(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_KEY_ENV", "dummy-test-key")
        custom_settings = {
            "llm": {
                "provider": "gemini",
                "gemini": {
                    "api_key_env": "TEST_KEY_ENV",
                    "model_name": "gemini-3.6-flash",
                },
            }
        }
        provider = GeminiProvider(settings=custom_settings)

        mock_response = MagicMock()
        mock_response.text = '{"flags": []}'

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        with patch("google.genai.Client", return_value=mock_client):
            dummy_schema = {"type": "object"}
            result = provider.generate("Test prompt", response_schema=dummy_schema)
            assert result == '{"flags": []}'
            config = mock_client.models.generate_content.call_args.kwargs["config"]
            assert config.response_mime_type == "application/json"
            assert config.response_schema == dummy_schema


class TestTopLevelGenerate:
    def test_generate_delegates_to_active_provider(self) -> None:
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.generate.return_value = "Delegated response"

        with patch("app.llm.provider.get_llm_provider", return_value=mock_provider):
            res = generate("Hello prompt", attempt=2)
            assert res == "Delegated response"
            mock_provider.generate.assert_called_once_with(
                "Hello prompt",
                attempt=2,
                response_schema=None,
            )

    def test_generate_delegates_with_response_schema(self) -> None:
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.generate.return_value = '{"flags": []}'

        with patch("app.llm.provider.get_llm_provider", return_value=mock_provider):
            schema = {"type": "object"}
            res = generate("Hello prompt", response_schema=schema)
            assert res == '{"flags": []}'
            mock_provider.generate.assert_called_once_with(
                "Hello prompt",
                attempt=None,
                response_schema=schema,
            )
