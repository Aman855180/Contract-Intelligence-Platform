from __future__ import annotations

import pytest

from app.utils import runtime_config


def test_environment_value_precedes_streamlit_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setattr(runtime_config, "_streamlit_secret", lambda _name: "ollama")

    assert runtime_config.get_runtime_setting("LLM_PROVIDER") == "gemini"


def test_streamlit_secret_used_when_environment_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(runtime_config, "_streamlit_secret", lambda _name: "test-key")

    assert runtime_config.get_runtime_setting("GEMINI_API_KEY") == "test-key"


def test_missing_secret_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(runtime_config, "_streamlit_secret", lambda _name: None)

    assert runtime_config.get_runtime_setting("GEMINI_API_KEY") is None


@pytest.mark.parametrize("value", ["false", "0", "no", "off"])
def test_runtime_bool_can_disable_fallback_from_secret(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.delenv("LLM_FALLBACK_TO_OLLAMA", raising=False)
    monkeypatch.setattr(runtime_config, "_streamlit_secret", lambda _name: value)

    assert runtime_config.get_runtime_bool("LLM_FALLBACK_TO_OLLAMA", True) is False


def test_runtime_bool_preserves_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_FALLBACK_TO_OLLAMA", raising=False)
    monkeypatch.setattr(runtime_config, "_streamlit_secret", lambda _name: None)

    assert runtime_config.get_runtime_bool("LLM_FALLBACK_TO_OLLAMA", True) is True
