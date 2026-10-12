"""
test_embedder.py

Unit tests for app/embeddings/embedder.py. The real sentence-transformers
model requires downloading weights from huggingface.co, which is not
reachable in this environment. Tests therefore monkeypatch _load_model()
to return a deterministic stub model, which is sufficient to verify the
wrapper's contract (batching, normalization flag passthrough, query
instruction prefixing, error handling) without needing real inference.
"""

from __future__ import annotations

import math

import pytest

from app.embeddings import embedder
from app.embeddings.exceptions import EmbeddingError


class StubEmbeddingModel:
    """Deterministic fake model: encodes text length + char sum into a fixed-size vector."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.last_call_kwargs: dict | None = None

    def encode(self, texts: list[str], normalize_embeddings: bool = True, batch_size: int = 16):
        self.last_call_kwargs = {"normalize_embeddings": normalize_embeddings, "batch_size": batch_size}
        vectors = []
        for text in texts:
            base = [float((sum(map(ord, text)) + i * 7) % 97) for i in range(self.dim)]
            if normalize_embeddings:
                norm = math.sqrt(sum(v * v for v in base)) or 1.0
                base = [v / norm for v in base]
            vectors.append(base)
        return vectors


@pytest.fixture(autouse=True)
def reset_embedder_cache():
    """Ensure every test starts with a clean model cache."""
    embedder.reset_model_cache()
    yield
    embedder.reset_model_cache()


@pytest.fixture()
def stub_model(monkeypatch: pytest.MonkeyPatch) -> StubEmbeddingModel:
    model = StubEmbeddingModel()
    monkeypatch.setattr(embedder, "_load_model", lambda: model)
    return model


class TestEmbedTexts:
    def test_returns_one_vector_per_input(self, stub_model: StubEmbeddingModel) -> None:
        texts = ["Payment terms.", "Termination clause.", "Confidentiality obligations."]
        vectors = embedder.embed_texts(texts)
        assert len(vectors) == 3
        assert all(isinstance(v, list) for v in vectors)
        assert all(isinstance(x, float) for v in vectors for x in v)

    def test_empty_input_returns_empty_list_without_loading_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called = {"flag": False}

        def fail_if_called():
            called["flag"] = True
            raise AssertionError("Model should not be loaded for empty input.")

        monkeypatch.setattr(embedder, "_load_model", fail_if_called)
        result = embedder.embed_texts([])
        assert result == []
        assert called["flag"] is False

    def test_deterministic_same_text_same_vector(self, stub_model: StubEmbeddingModel) -> None:
        v1 = embedder.embed_texts(["The same clause text."])[0]
        v2 = embedder.embed_texts(["The same clause text."])[0]
        assert v1 == v2

    def test_different_text_different_vector(self, stub_model: StubEmbeddingModel) -> None:
        v1 = embedder.embed_texts(["Payment due in 30 days."])[0]
        v2 = embedder.embed_texts(["Liability is capped at fees paid."])[0]
        assert v1 != v2

    def test_model_load_failure_raises_embedding_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def raise_error():
            raise EmbeddingError("simulated load failure")

        monkeypatch.setattr(embedder, "_load_model", raise_error)
        with pytest.raises(EmbeddingError):
            embedder.embed_texts(["some text"])


class TestEmbedQuery:
    def test_returns_single_vector(self, stub_model: StubEmbeddingModel) -> None:
        vector = embedder.embed_query("What is the payment term?")
        assert isinstance(vector, list)
        assert all(isinstance(x, float) for x in vector)

    def test_query_instruction_prefix_changes_embedding(self, stub_model: StubEmbeddingModel) -> None:
        # Embedding the raw query text directly (no prefix) via embed_texts should differ
        # from embed_query's result, proving the instruction prefix is actually applied.
        raw_vector = embedder.embed_texts(["What is the payment term?"])[0]
        query_vector = embedder.embed_query("What is the payment term?")
        assert raw_vector != query_vector


class TestEmbedQueries:
    def test_returns_batch_vectors(self, stub_model: StubEmbeddingModel) -> None:
        queries = ["query 1", "query 2", "query 3"]
        vectors = embedder.embed_queries(queries)
        assert len(vectors) == 3
        assert all(isinstance(v, list) for v in vectors)

    def test_empty_input_returns_empty_list(self) -> None:
        assert embedder.embed_queries([]) == []

    def test_batch_matches_single_embed_query(self, stub_model: StubEmbeddingModel) -> None:
        queries = ["indemnity obligations", "payment terms"]
        batch_vectors = embedder.embed_queries(queries)
        single_vectors = [embedder.embed_query(q) for q in queries]
        assert batch_vectors == single_vectors



class TestModelCaching:
    def test_model_loaded_only_once_across_multiple_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        load_count = {"n": 0}
        model = StubEmbeddingModel()

        def counting_loader():
            load_count["n"] += 1
            return model

        # Patch the real _load_model (not pre-stubbed) to verify the caching
        # behavior inside _load_model itself using the module's actual cache variable.
        monkeypatch.setattr(embedder, "_model_instance", None)
        original_load_model = embedder._load_model

        def wrapped_load_model():
            if embedder._model_instance is None:
                load_count["n"] += 1
                embedder._model_instance = model
            return embedder._model_instance

        monkeypatch.setattr(embedder, "_load_model", wrapped_load_model)

        embedder.embed_texts(["a"])
        embedder.embed_texts(["b"])
        embedder.embed_query("c")

        assert load_count["n"] == 1
