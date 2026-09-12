"""
embedder.py

Wrapper around sentence-transformers for generating embeddings with
BAAI/bge-base-en-v1.5 (configured in settings.yaml). The model is loaded
lazily and cached as a module-level singleton so it is only loaded once
per process, regardless of how many times embed_texts()/embed_query() are
called.

The actual `SentenceTransformer` import happens inside _load_model() rather
than at module import time, so this module can be imported (and the rest of
the embeddings/chroma_store code exercised in tests) without requiring the
sentence-transformers package or network access to download model weights
unless embedding generation is actually invoked.
"""

from __future__ import annotations

from typing import Any, Protocol

from app.embeddings.exceptions import EmbeddingError
from app.utils.config_loader import get_settings
from app.utils.logging_setup import get_logger
from app.utils.perf import perf_span

logger = get_logger(__name__)

_model_instance: Any = None


class EmbeddingModel(Protocol):
    """Structural type for whatever _load_model() returns.

    Both the real sentence_transformers.SentenceTransformer and any test
    stub satisfy this protocol, which is all embed_texts()/embed_query()
    rely on.
    """

    def encode(self, texts: list[str], normalize_embeddings: bool, batch_size: int) -> Any: ...


def _load_model() -> EmbeddingModel:
    """
    Load (or return the cached) sentence-transformers embedding model.

    Returns:
        A loaded SentenceTransformer instance.

    Raises:
        EmbeddingError: If the underlying library/model fails to load
            (e.g. package not installed, network unavailable for first-time
            download, or an invalid model name in config).
    """
    global _model_instance
    if _model_instance is not None:
        return _model_instance

    settings = get_settings()
    embeddings_cfg = settings.get("embeddings", {})
    model_name = embeddings_cfg.get("model_name", "BAAI/bge-base-en-v1.5")
    device = embeddings_cfg.get("device", "cpu")

    with perf_span("embedding_model_loading", extra_info=f"model={model_name} device={device}"):
        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model '%s' on device '%s'...", model_name, device)
            _model_instance = SentenceTransformer(model_name, device=device)
            logger.info("Embedding model '%s' loaded successfully.", model_name)
        except Exception as exc:
            logger.error("Failed to load embedding model '%s': %s", model_name, exc)
            raise EmbeddingError(f"Could not load embedding model '{model_name}': {exc}") from exc

    return _model_instance


def reset_model_cache() -> None:
    """Clear the cached model instance. Used by tests to inject stub models."""
    global _model_instance
    _model_instance = None


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a batch of document chunk texts (passages, not
    queries — no instruction prefix is applied).

    Args:
        texts: List of chunk text strings to embed.

    Returns:
        List of embedding vectors (one per input text, same order).

    Raises:
        EmbeddingError: If the model fails to load or encoding fails.
    """
    if not texts:
        return []

    settings = get_settings()
    embeddings_cfg = settings.get("embeddings", {})
    normalize = embeddings_cfg.get("normalize_embeddings", True)
    batch_size = embeddings_cfg.get("batch_size", 16)

    model = _load_model()
    with perf_span("embed_texts", extra_info=f"count={len(texts)}"):
        try:
            vectors = model.encode(texts, normalize_embeddings=normalize, batch_size=batch_size)
        except Exception as exc:
            logger.error("Failed to embed %d text(s): %s", len(texts), exc)
            raise EmbeddingError(f"Embedding generation failed: {exc}") from exc

    return [list(map(float, vector)) for vector in vectors]


def embed_query(query: str) -> list[float]:
    """
    Generate an embedding for a single user query, applying the BGE
    asymmetric-retrieval query instruction prefix from config.

    Args:
        query: The raw user question text.

    Returns:
        The query's embedding vector.

    Raises:
        EmbeddingError: If the model fails to load or encoding fails.
    """
    settings = get_settings()
    embeddings_cfg = settings.get("embeddings", {})
    instruction = embeddings_cfg.get("query_instruction", "")
    normalize = embeddings_cfg.get("normalize_embeddings", True)

    model = _load_model()
    prefixed_query = f"{instruction}{query}"
    with perf_span("embed_query", extra_info=f"query_chars={len(query)}"):
        try:
            vector = model.encode([prefixed_query], normalize_embeddings=normalize, batch_size=1)[0]
        except Exception as exc:
            logger.error("Failed to embed query '%s': %s", query, exc)
            raise EmbeddingError(f"Query embedding generation failed: {exc}") from exc

    return list(map(float, vector))
