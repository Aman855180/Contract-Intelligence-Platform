"""
test_ner_pipeline.py

Tests for app/extraction/ner_pipeline.py. The real dslim/bert-base-NER
model requires downloading weights from huggingface.co, unreachable in
this sandboxed environment, so tests monkeypatch _load_ner_pipeline with a
deterministic stub callable to verify the wrapper's contract (result
shape normalization, empty-text short-circuit, caching, error handling).
"""

from __future__ import annotations

import pytest

from app.extraction import ner_pipeline
from app.extraction.exceptions import NERPipelineError


class StubNERPipeline:
    """Stub mimicking a transformers NER pipeline's callable interface."""

    def __init__(self, fixed_entities: list[dict] | None = None) -> None:
        self.fixed_entities = fixed_entities or []
        self.call_count = 0
        self.last_text: str | None = None

    def __call__(self, text: str) -> list[dict]:
        self.call_count += 1
        self.last_text = text
        return self.fixed_entities


@pytest.fixture(autouse=True)
def reset_ner_cache():
    ner_pipeline.reset_pipeline_cache()
    yield
    ner_pipeline.reset_pipeline_cache()


class TestExtractEntities:
    def test_returns_normalized_entity_dicts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        raw = [{"word": "Acme Corp", "entity_group": "ORG", "score": 0.987}]
        stub = StubNERPipeline(fixed_entities=raw)
        monkeypatch.setattr(ner_pipeline, "_load_ner_pipeline", lambda: stub)

        entities = ner_pipeline.extract_entities("Acme Corp shall deliver the goods.")

        assert entities == [{"text": "Acme Corp", "label": "ORG", "score": pytest.approx(0.987)}]

    def test_handles_entity_key_fallback_when_no_entity_group(self, monkeypatch: pytest.MonkeyPatch) -> None:
        raw = [{"word": "John Smith", "entity": "PER", "score": 0.91}]
        stub = StubNERPipeline(fixed_entities=raw)
        monkeypatch.setattr(ner_pipeline, "_load_ner_pipeline", lambda: stub)

        entities = ner_pipeline.extract_entities("John Smith signed the agreement.")
        assert entities[0]["label"] == "PER"

    def test_empty_text_returns_empty_list_without_loading_pipeline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called = {"flag": False}

        def fail_if_called():
            called["flag"] = True
            raise AssertionError("Pipeline should not load for empty text.")

        monkeypatch.setattr(ner_pipeline, "_load_ner_pipeline", fail_if_called)
        assert ner_pipeline.extract_entities("") == []
        assert ner_pipeline.extract_entities("   ") == []
        assert called["flag"] is False

    def test_no_entities_found_returns_empty_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stub = StubNERPipeline(fixed_entities=[])
        monkeypatch.setattr(ner_pipeline, "_load_ner_pipeline", lambda: stub)

        assert ner_pipeline.extract_entities("The weather is nice today.") == []

    def test_pipeline_load_failure_raises_ner_pipeline_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def raise_error():
            raise NERPipelineError("simulated load failure")

        monkeypatch.setattr(ner_pipeline, "_load_ner_pipeline", raise_error)
        with pytest.raises(NERPipelineError):
            ner_pipeline.extract_entities("Some text.")

    def test_inference_failure_raises_ner_pipeline_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FailingPipeline:
            def __call__(self, text: str) -> list[dict]:
                raise RuntimeError("inference crashed")

        monkeypatch.setattr(ner_pipeline, "_load_ner_pipeline", lambda: FailingPipeline())
        with pytest.raises(NERPipelineError):
            ner_pipeline.extract_entities("Some text.")

    def test_passes_exact_text_to_pipeline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stub = StubNERPipeline(fixed_entities=[])
        monkeypatch.setattr(ner_pipeline, "_load_ner_pipeline", lambda: stub)

        ner_pipeline.extract_entities("Acme Corp shall pay Globex Inc.")
        assert stub.last_text == "Acme Corp shall pay Globex Inc."


class TestPipelineCaching:
    def test_load_ner_pipeline_caches_singleton(self, monkeypatch: pytest.MonkeyPatch) -> None:
        load_count = {"n": 0}
        stub = StubNERPipeline()

        class _CountingPipelineFactory:
            def __call__(self, task: str, model: str, aggregation_strategy: str):
                load_count["n"] += 1
                return stub

        # Patch the transformers.pipeline factory the real _load_ner_pipeline imports.
        import sys
        import types

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.pipeline = _CountingPipelineFactory()
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

        ner_pipeline._load_ner_pipeline()
        ner_pipeline._load_ner_pipeline()
        ner_pipeline._load_ner_pipeline()

        assert load_count["n"] == 1
