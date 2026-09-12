"""
bm25_index.py

BM25 keyword index over the same chunk corpus stored in ChromaDB. Rather
than maintaining a separate copy of chunk text, this module reads chunk
text + metadata directly from the ChromaDB collection (app/embeddings)
and builds an in-memory rank_bm25 index on top of it.

The index is cached in memory and automatically rebuilt whenever the
underlying chunk count changes (i.e. after new documents are indexed),
since rank_bm25 does not support incremental updates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

from app.embeddings.chroma_store import get_or_create_collection
from app.retriever.exceptions import RetrievalError
from app.utils.logging_setup import get_logger

logger = get_logger(__name__)

_TOKEN_PATTERN = re.compile(r"\w+")


@dataclass
class _BM25IndexState:
    """In-memory cached BM25 index plus the parallel chunk metadata it was built from."""

    bm25: BM25Okapi
    chunk_ids: list[str]
    texts: list[str]
    metadatas: list[dict]
    built_from_count: int = field(default=0)


_state: _BM25IndexState | None = None


def _tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric-token tokenizer used consistently for indexing and querying."""
    return _TOKEN_PATTERN.findall(text.lower())


def _fetch_all_chunks() -> tuple[list[str], list[str], list[dict]]:
    """
    Fetch every chunk's id, text, and metadata currently stored in ChromaDB.

    Returns:
        Tuple of (chunk_ids, texts, metadatas), all aligned by index.
    """
    collection = get_or_create_collection()
    if collection.count() == 0:
        return [], [], []

    data = collection.get(include=["documents", "metadatas"])
    return data["ids"], data["documents"], data["metadatas"]


def reset_index_cache() -> None:
    """Clear the cached BM25 index, forcing a rebuild on next search. Used by tests."""
    global _state
    _state = None


def _ensure_index_built() -> _BM25IndexState | None:
    """
    Build (or rebuild, if the chunk count has changed since last build) the
    in-memory BM25 index from the current ChromaDB contents.

    Returns:
        The current _BM25IndexState, or None if there are no chunks indexed yet.
    """
    global _state

    chunk_ids, texts, metadatas = _fetch_all_chunks()
    current_count = len(chunk_ids)

    if current_count == 0:
        _state = None
        return None

    if _state is not None and _state.built_from_count == current_count:
        return _state

    try:
        tokenized_corpus = [_tokenize(t) for t in texts]
        bm25 = BM25Okapi(tokenized_corpus)
    except Exception as exc:
        logger.error("Failed to build BM25 index: %s", exc)
        raise RetrievalError(f"Failed to build BM25 index: {exc}") from exc

    _state = _BM25IndexState(
        bm25=bm25, chunk_ids=chunk_ids, texts=texts, metadatas=metadatas, built_from_count=current_count
    )
    logger.info("Built BM25 index over %d chunk(s).", current_count)
    return _state


def search_bm25(query: str, top_k: int) -> list[dict]:
    """
    Search the BM25 index for the top-K most relevant chunks to a query.

    Args:
        query: The raw user query text.
        top_k: Number of results to return.

    Returns:
        List of result dicts (chunk_id, text, metadata, bm25_score), sorted
        descending by score. Empty list if no chunks are indexed yet.
    """
    state = _ensure_index_built()
    if state is None:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    scores = state.bm25.get_scores(query_tokens)

    # Pair each chunk with its score, then take the top_k descending.
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results: list[dict] = []
    for i in ranked_indices:
        if scores[i] <= 0:
            # A BM25 score of 0 means no query terms matched at all; don't
            # surface irrelevant chunks just to pad out top_k.
            continue
        results.append(
            {
                "chunk_id": state.chunk_ids[i],
                "text": state.texts[i],
                "metadata": state.metadatas[i],
                "bm25_score": float(scores[i]),
            }
        )
    return results
