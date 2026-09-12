"""
test_ollama_client.py

Tests for app/llm/ollama_client.py. Mocks requests.post so these tests run
without a real Ollama server (not available in this sandboxed environment),
while still exercising the actual request construction and response
parsing logic.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from app.llm import ollama_client
from app.llm.exceptions import OllamaConnectionError


def _mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data
    if status_code >= 400:
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(f"{status_code} error")
    else:
        mock_resp.raise_for_status.return_value = None
    return mock_resp


class TestGenerate:
    def test_successful_generation_returns_text(self) -> None:
        with patch("app.llm.ollama_client.requests.post") as mock_post:
            mock_post.return_value = _mock_response({"response": "The payment term is 30 days.", "done": True})
            result = ollama_client.generate("What is the payment term?")
            assert result == "The payment term is 30 days."

    def test_request_uses_configured_model_and_temperature(self) -> None:
        with patch("app.llm.ollama_client.requests.post") as mock_post:
            mock_post.return_value = _mock_response({"response": "answer"})
            ollama_client.generate("some prompt")

            _, kwargs = mock_post.call_args
            payload = kwargs["json"]
            assert payload["model"] == "llama3.1:8b"
            assert payload["options"]["temperature"] == 0.0
            assert payload["stream"] is False
            assert payload["prompt"] == "some prompt"

    def test_request_posts_to_configured_base_url(self) -> None:
        with patch("app.llm.ollama_client.requests.post") as mock_post:
            mock_post.return_value = _mock_response({"response": "answer"})
            ollama_client.generate("some prompt")

            args, _ = mock_post.call_args
            assert args[0] == "http://localhost:11434/api/generate"

    def test_connection_error_raises_ollama_connection_error(self) -> None:
        with patch("app.llm.ollama_client.requests.post") as mock_post:
            mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")
            with pytest.raises(OllamaConnectionError):
                ollama_client.generate("some prompt")

    def test_timeout_raises_ollama_connection_error(self) -> None:
        with patch("app.llm.ollama_client.requests.post") as mock_post:
            mock_post.side_effect = requests.exceptions.Timeout("Request timed out")
            with pytest.raises(OllamaConnectionError):
                ollama_client.generate("some prompt")

    def test_http_error_status_raises_ollama_connection_error(self) -> None:
        with patch("app.llm.ollama_client.requests.post") as mock_post:
            mock_post.return_value = _mock_response({}, status_code=500)
            with pytest.raises(OllamaConnectionError):
                ollama_client.generate("some prompt")

    def test_missing_response_field_raises(self) -> None:
        with patch("app.llm.ollama_client.requests.post") as mock_post:
            mock_post.return_value = _mock_response({"done": True})  # no "response" key
            with pytest.raises(OllamaConnectionError):
                ollama_client.generate("some prompt")

    def test_invalid_json_raises(self) -> None:
        with patch("app.llm.ollama_client.requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.return_value = None
            mock_resp.json.side_effect = ValueError("not valid json")
            mock_post.return_value = mock_resp
            with pytest.raises(OllamaConnectionError):
                ollama_client.generate("some prompt")
