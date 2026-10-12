"""
test_prompt_loader.py

Tests for app/llm/prompt_loader.py against the real prompt template files
under prompts/.
"""

from __future__ import annotations

import pytest

from app.llm import prompt_loader
from app.llm.exceptions import PromptNotFoundError


@pytest.fixture(autouse=True)
def clear_prompt_cache():
    prompt_loader._read_prompt_file.cache_clear()
    yield
    prompt_loader._read_prompt_file.cache_clear()


class TestLoadPrompt:
    def test_loads_and_formats_qa_prompt(self) -> None:
        result = prompt_loader.load_prompt(
            "qa_prompt.txt", retrieved_context="[Contract A, page 2]\nPayment due in 30 days.", question="What is the payment term?"
        )
        assert "What is the payment term?" in result
        assert "Payment due in 30 days." in result
        assert "{retrieved_context}" not in result
        assert "{question}" not in result

    def test_loads_and_formats_risk_prompt_without_duplicate_json_schema(self) -> None:
        result = prompt_loader.load_prompt("risk_prompt.txt", retrieved_context="Some passage text.")
        assert "unlimited_indemnity_language" in result
        assert "high_liability_exposure" in result
        assert "ambiguous_payment_terms" in result
        assert "Some passage text." in result
        assert "{retrieved_context}" not in result

    def test_loads_summary_prompt(self) -> None:
        result = prompt_loader.load_prompt(
            "summary_prompt.txt",
            contract_name="Test Contract",
            extracted_fields="Payment terms: Net 30",
            retrieved_context="Some context.",
            risk_flags="[]",
        )
        assert "Payment terms: Net 30" in result
        assert "overview: string" in result
        assert "important_dates: array" in result
        assert "no more than 3 key clauses" in result
        assert "no more than 2 actionable recommendations" in result

    def test_missing_prompt_file_raises(self) -> None:
        with pytest.raises(PromptNotFoundError):
            prompt_loader.load_prompt("does_not_exist.txt")

    def test_missing_placeholder_raises(self) -> None:
        with pytest.raises(PromptNotFoundError):
            prompt_loader.load_prompt("qa_prompt.txt", question="only one of two required placeholders")

    def test_file_contents_are_cached(self) -> None:
        first = prompt_loader._read_prompt_file("qa_prompt.txt")
        second = prompt_loader._read_prompt_file("qa_prompt.txt")
        assert first is second  # same cached string object
