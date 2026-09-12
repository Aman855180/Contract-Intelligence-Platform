"""
ner_pipeline.py

Wrapper around a HuggingFace transformers NER pipeline (dslim/bert-base-NER
by default, per config), used to identify named entities (PER/ORG/LOC/MISC)
within sentences - primarily to find the obligated party in
app/extraction/obligation_extractor.py.

The pipeline is loaded lazily and cached as a module-level singleton,
mirroring the pattern in app/embeddings/embedder.py and
app/groundedness/nli_checker.py, so this module is importable and testable
without requiring network access to download model weights unless NER is
actually invoked.
"""

from __future__ import annotations

from typing import Any, Protocol

from app.extraction.exceptions import NERPipelineError
from app.utils.config_loader import get_settings
from app.utils.logging_setup import get_logger

logger = get_logger(__name__)

_pipeline_instance: Any = None


class NERPipelineProtocol(Protocol):
    """Structural type for whatever _load_ner_pipeline() returns."""

    def __call__(self, text: str) -> list[dict]: ...


def _load_ner_pipeline() -> NERPipelineProtocol:
    """
    Load (or return the cached) HuggingFace NER pipeline.

    Returns:
        A callable transformers NER pipeline.

    Raises:
        NERPipelineError: If the pipeline fails to load.
    """
    global _pipeline_instance
    if _pipeline_instance is not None:
        return _pipeline_instance

    settings = get_settings()
    ner_cfg = settings.get("ner", {})
    model_name = ner_cfg.get("model_name", "dslim/bert-base-NER")
    aggregation_strategy = ner_cfg.get("aggregation_strategy", "simple")

    try:
        from transformers import pipeline

        logger.info("Loading NER pipeline '%s' (aggregation_strategy=%s)...", model_name, aggregation_strategy)
        _pipeline_instance = pipeline(
            "ner", model=model_name, aggregation_strategy=aggregation_strategy
        )
        logger.info("NER pipeline '%s' loaded successfully.", model_name)
    except Exception as exc:
        logger.error("Failed to load NER pipeline '%s': %s", model_name, exc)
        raise NERPipelineError(f"Could not load NER pipeline '{model_name}': {exc}") from exc

    return _pipeline_instance


def reset_pipeline_cache() -> None:
    """Clear the cached NER pipeline instance. Used by tests to inject stub pipelines."""
    global _pipeline_instance
    _pipeline_instance = None


def extract_entities(text: str) -> list[dict[str, str | float]]:
    """
    Run NER inference on a piece of text and return its detected entities.

    Args:
        text: The text to extract entities from (typically a single sentence).

    Returns:
        List of dicts, each with keys: text (str, the entity span),
        label (str, e.g. "PER"/"ORG"/"LOC"/"MISC"), score (float, 0-1
        confidence). Empty list if no entities found.

    Raises:
        NERPipelineError: If the pipeline fails to load or run inference.
    """
    if not text or not text.strip():
        return []

    ner_pipeline = _load_ner_pipeline()
    try:
        raw_entities = ner_pipeline(text)
    except Exception as exc:
        logger.error("NER inference failed for text '%s...': %s", text[:50], exc)
        raise NERPipelineError(f"NER inference failed: {exc}") from exc

    entities: list[dict[str, str | float]] = []
    for entity in raw_entities:
        entities.append(
            {
                "text": entity.get("word", ""),
                "label": entity.get("entity_group", entity.get("entity", "")),
                "score": float(entity.get("score", 0.0)),
            }
        )
    return entities
