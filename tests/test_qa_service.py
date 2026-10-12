"""
test_qa_service.py

Integration tests for app/llm/qa_service.py: real hybrid retrieval (real
ChromaDB + real BM25) against a stubbed embedder, stubbed Ollama
generate(), and stubbed NLI groundedness check - verifying the full
retrieve -> prompt -> parse -> groundedness-verify -> QAAnswer pipeline.
"""

from __future__ import annotations

import hashlib
import math
import uuid

import pytest

from app.chunking.pipeline import index_document
from app.embeddings import chroma_store, embedder
from app.groundedness import nli_checker
from app.llm import qa_service
from app.models.document import DocumentMetadata, DocumentType, PageText, ParsedDocument
from app.models.retrieval import GroundednessVerdict
from app.retriever import bm25_index


def _stable_hash(token: str) -> int:
    """Deterministic token hash (unlike Python's randomized built-in hash())."""
    return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)


class StubEmbeddingModel:
    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def _vectorize(self, text: str) -> list[float]:
        tokens = text.lower().split()
        vec = [0.0] * self.dim
        for token in tokens:
            idx = _stable_hash(token) % self.dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def encode(self, texts, normalize_embeddings: bool = True, batch_size: int = 16):
        return [self._vectorize(t) for t in texts]


@pytest.fixture()
def isolated_qa_env(tmp_path, monkeypatch: pytest.MonkeyPatch):
    from app.utils import config_loader

    config_loader.get_settings.cache_clear()
    config_loader.get_retrieval_config.cache_clear()
    settings = config_loader.get_settings()
    settings["paths"]["chroma_persist_dir"] = str(tmp_path / f"chroma_{uuid.uuid4().hex}")

    chroma_store.reset_client_cache()
    bm25_index.reset_index_cache()
    embedder.reset_model_cache()
    nli_checker.reset_model_cache()
    monkeypatch.setattr(embedder, "_load_model", lambda: StubEmbeddingModel())

    yield

    chroma_store.reset_client_cache()
    bm25_index.reset_index_cache()
    embedder.reset_model_cache()
    nli_checker.reset_model_cache()
    config_loader.get_settings.cache_clear()
    config_loader.get_retrieval_config.cache_clear()


def _index_sample_contract() -> None:
    metadata = DocumentMetadata(
        document_id="doc-1",
        contract_name="Sample Services Agreement",
        document_type=DocumentType.PDF,
        page_count=2,
        file_size_bytes=5000,
        source_path="x.pdf",
    )
    pages = [
        PageText(page_number=1, text="Payment shall be due within thirty days of invoice receipt by the client."),
        PageText(page_number=2, text="Either party may terminate this agreement with sixty days written notice."),
    ]
    parsed = ParsedDocument(metadata=metadata, pages=pages)
    index_document(parsed)


def _stub_qa_provider(monkeypatch: pytest.MonkeyPatch, response) -> None:
    class StubProvider:
        provider_name = "test"

        def generate(self, prompt: str, attempt: int | None = None) -> str:
            return response(prompt) if callable(response) else response

    monkeypatch.setattr(qa_service, "get_qa_provider", lambda settings: StubProvider())


class TestAnswerQuestion:
    def test_gemini_failure_does_not_fallback_when_disabled(
        self, isolated_qa_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.llm.exceptions import GeminiConnectionError

        _index_sample_contract()

        class FailingGeminiProvider:
            provider_name = "gemini"

            def generate(self, prompt: str, attempt: int | None = None) -> str:
                raise GeminiConnectionError("Gemini unavailable")

        monkeypatch.setenv("LLM_FALLBACK_TO_OLLAMA", "false")
        monkeypatch.setattr(qa_service, "get_qa_provider", lambda settings: FailingGeminiProvider())
        monkeypatch.setattr(
            qa_service,
            "get_llm_provider",
            lambda *args, **kwargs: pytest.fail("Ollama fallback must remain disabled"),
        )

        with pytest.raises(GeminiConnectionError, match="Gemini unavailable"):
            qa_service.answer_question("What is the payment term?")

    def test_returns_grounded_answer_with_citation(self, isolated_qa_env, monkeypatch: pytest.MonkeyPatch) -> None:
        _index_sample_contract()

        llm_response = (
            "ANSWER: Payment is due within 30 days of invoice receipt.\n"
            "SOURCE_CONTRACT: Sample Services Agreement\n"
            "SOURCE_PAGE: 1\n"
            "SUPPORTING_TEXT: Payment shall be due within thirty days of invoice receipt by the client."
        )
        _stub_qa_provider(monkeypatch, llm_response)

        fake_verdict = GroundednessVerdict(
            claim="Payment is due within 30 days of invoice receipt.",
            source_passage="Payment shall be due within thirty days of invoice receipt by the client.",
            entailment_score=0.95,
            contradiction_score=0.02,
            neutral_score=0.03,
            is_grounded=True,
        )
        monkeypatch.setattr(qa_service, "check_groundedness", lambda claim, source_passage: fake_verdict)

        answer = qa_service.answer_question("What is the payment term?")

        assert answer.found_in_contract is True
        assert answer.source_contract == "Sample Services Agreement"
        assert answer.source_page == 1
        assert answer.groundedness is not None
        assert answer.groundedness.is_grounded is True

    def test_no_evidence_retrieved_skips_llm_call(self, isolated_qa_env, monkeypatch: pytest.MonkeyPatch) -> None:
        called = {"flag": False}

        def fail_if_called(prompt: str) -> str:
            called["flag"] = True
            raise AssertionError("LLM should not be called when no evidence is retrieved.")

        _stub_qa_provider(monkeypatch, fail_if_called)

        answer = qa_service.answer_question("What is the payment term?")

        assert answer.found_in_contract is False
        assert answer.answer == "NOT_FOUND_IN_CONTRACT"
        assert called["flag"] is False

    def test_llm_reports_not_found_skips_groundedness_check(self, isolated_qa_env, monkeypatch: pytest.MonkeyPatch) -> None:
        _index_sample_contract()

        llm_response = "ANSWER: NOT_FOUND_IN_CONTRACT\nSOURCE_CONTRACT: \nSOURCE_PAGE: \nSUPPORTING_TEXT: "
        _stub_qa_provider(monkeypatch, llm_response)

        called = {"flag": False}

        def fail_if_called(claim: str, source_passage: str):
            called["flag"] = True
            raise AssertionError("Groundedness check should not run when LLM reports NOT_FOUND.")

        monkeypatch.setattr(qa_service, "check_groundedness", fail_if_called)

        answer = qa_service.answer_question("What is the governing law?")

        assert answer.found_in_contract is False
        assert called["flag"] is False

    def test_low_groundedness_score_is_surfaced_not_hidden(self, isolated_qa_env, monkeypatch: pytest.MonkeyPatch) -> None:
        _index_sample_contract()

        llm_response = (
            "ANSWER: Liability is completely unlimited under this agreement.\n"
            "SOURCE_CONTRACT: Sample Services Agreement\n"
            "SOURCE_PAGE: 1\n"
            "SUPPORTING_TEXT: Payment shall be due within thirty days of invoice receipt by the client."
        )
        _stub_qa_provider(monkeypatch, llm_response)

        ungrounded_verdict = GroundednessVerdict(
            claim="Liability is completely unlimited under this agreement.",
            source_passage="Payment shall be due within thirty days of invoice receipt by the client.",
            entailment_score=0.05,
            contradiction_score=0.10,
            neutral_score=0.85,
            is_grounded=False,
        )
        monkeypatch.setattr(qa_service, "check_groundedness", lambda claim, source_passage: ungrounded_verdict)

        answer = qa_service.answer_question("Is liability limited?")

        assert answer.found_in_contract is True
        assert answer.groundedness.is_grounded is False

    def test_malformed_llm_response_propagates_parsing_error(self, isolated_qa_env, monkeypatch: pytest.MonkeyPatch) -> None:
        from app.llm.exceptions import LLMResponseParsingError

        _index_sample_contract()
        _stub_qa_provider(monkeypatch, "completely unstructured garbage text")

        with pytest.raises(LLMResponseParsingError):
            qa_service.answer_question("What is the payment term?")

    def test_groundedness_check_uses_supporting_text_when_present(self, isolated_qa_env, monkeypatch: pytest.MonkeyPatch) -> None:
        _index_sample_contract()

        llm_response = (
            "ANSWER: Termination requires 60 days notice.\n"
            "SOURCE_CONTRACT: Sample Services Agreement\n"
            "SOURCE_PAGE: 2\n"
            "SUPPORTING_TEXT: Either party may terminate this agreement with sixty days written notice."
        )
        _stub_qa_provider(monkeypatch, llm_response)

        captured = {}

        def capture_groundedness(claim: str, source_passage: str):
            captured["claim"] = claim
            captured["source_passage"] = source_passage
            return GroundednessVerdict(
                claim=claim,
                source_passage=source_passage,
                entailment_score=0.9,
                contradiction_score=0.05,
                neutral_score=0.05,
                is_grounded=True,
            )

        monkeypatch.setattr(qa_service, "check_groundedness", capture_groundedness)

        qa_service.answer_question("What is the termination notice period?")

        assert captured["source_passage"] == "Either party may terminate this agreement with sixty days written notice."
        assert captured["claim"] == "Termination requires 60 days notice."
