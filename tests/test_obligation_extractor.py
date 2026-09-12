"""
test_obligation_extractor.py

Tests for app/extraction/obligation_extractor.py. The NER pipeline is
stubbed (app.extraction.ner_pipeline.extract_entities) so these tests run
without network access while exercising real sentence splitting and
obligation-pattern matching logic.
"""

from __future__ import annotations

import pytest

from app.extraction import obligation_extractor
from app.models.extraction import FieldType


class TestExtractObligations:
    def test_detects_obligation_sentence_with_party_entity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_extract_entities(sentence: str):
            if "Acme Corp" in sentence:
                return [{"text": "Acme Corp", "label": "ORG", "score": 0.93}]
            return []

        monkeypatch.setattr(obligation_extractor, "extract_entities", fake_extract_entities)

        text = "Acme Corp shall deliver the goods within fourteen days."
        fields = obligation_extractor.extract_obligations(text, "doc-1", "Sample Contract", 1)

        assert len(fields) == 1
        assert fields[0].field_type == FieldType.OBLIGATIONS
        assert fields[0].confidence == pytest.approx(0.93)
        assert "Acme Corp" in fields[0].value_text

    def test_pattern_only_match_uses_fallback_confidence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(obligation_extractor, "extract_entities", lambda sentence: [])

        text = "The recipient agrees to maintain strict confidentiality of all materials."
        fields = obligation_extractor.extract_obligations(text, "doc-1", "Sample Contract", 1)

        assert len(fields) == 1
        assert fields[0].confidence == pytest.approx(0.5)  # ner.obligation_pattern_only_confidence default

    def test_non_obligation_sentences_are_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(obligation_extractor, "extract_entities", lambda sentence: [])

        text = "The weather was pleasant. The meeting room had a nice view."
        fields = obligation_extractor.extract_obligations(text, "doc-1", "Sample Contract", 1)
        assert fields == []

    def test_multiple_obligation_sentences_each_extracted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_extract_entities(sentence: str):
            if "Buyer" in sentence:
                return [{"text": "Buyer", "label": "PER", "score": 0.7}]
            if "Seller" in sentence:
                return [{"text": "Seller", "label": "PER", "score": 0.8}]
            return []

        monkeypatch.setattr(obligation_extractor, "extract_entities", fake_extract_entities)

        text = "Buyer shall pay all invoices within 30 days. Seller agrees to deliver goods on time."
        fields = obligation_extractor.extract_obligations(text, "doc-1", "Sample Contract", 2)

        assert len(fields) == 2
        confidences = sorted(f.confidence for f in fields)
        assert confidences == [pytest.approx(0.7), pytest.approx(0.8)]

    def test_non_party_entities_ignored_for_confidence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Only a LOC entity found (not PER/ORG) -> should fall back to pattern-only confidence.
        monkeypatch.setattr(
            obligation_extractor,
            "extract_entities",
            lambda sentence: [{"text": "New York", "label": "LOC", "score": 0.99}],
        )

        text = "The Contractor shall perform the work in New York."
        fields = obligation_extractor.extract_obligations(text, "doc-1", "Sample Contract", 1)

        assert len(fields) == 1
        assert fields[0].confidence == pytest.approx(0.5)

    def test_best_party_entity_chosen_when_multiple_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            obligation_extractor,
            "extract_entities",
            lambda sentence: [
                {"text": "Globex", "label": "ORG", "score": 0.6},
                {"text": "Acme Corp", "label": "ORG", "score": 0.95},
            ],
        )

        text = "Acme Corp shall notify Globex of any material breach."
        fields = obligation_extractor.extract_obligations(text, "doc-1", "Sample Contract", 1)

        assert fields[0].confidence == pytest.approx(0.95)

    def test_field_citations_match_inputs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(obligation_extractor, "extract_entities", lambda sentence: [])

        text = "The Contractor must provide weekly status reports."
        fields = obligation_extractor.extract_obligations(text, "doc-42", "Test Contract", 7)

        assert fields[0].document_id == "doc-42"
        assert fields[0].contract_name == "Test Contract"
        assert fields[0].page_number == 7

    def test_empty_text_returns_empty_list(self) -> None:
        assert obligation_extractor.extract_obligations("", "doc-1", "Contract", 1) == []
