"""
fusion.py

Reciprocal Rank Fusion (RRF) for combining vector-search and BM25 result
lists into a single ranked list, per config/retrieval.yaml's
fusion.rrf_k constant. RRF is rank-based (not score-based), which sidesteps
the problem of vector cosine similarity and BM25 scores living on
incomparable scales.

Standard RRF formula: score(chunk) = sum over each source ranking it
appears in of 1 / (rrf_k + rank), where rank is 0-indexed position in that
source's list.
"""

from __future__ import annotations

from app.utils.logging_setup import get_logger

logger = get_logger(__name__)


def reciprocal_rank_fusion(
    vector_results: list[dict], bm25_results: list[dict], rrf_k: int
) -> list[dict]:
    """
    Fuse vector-search and BM25 result lists via reciprocal rank fusion.

    Args:
        vector_results: Results from chroma_store.query_similar(), each a
            dict with chunk_id, text, metadata, distance.
        bm25_results: Results from bm25_index.search_bm25(), each a dict
            with chunk_id, text, metadata, bm25_score.
        rrf_k: The RRF constant (1 / (rrf_k + rank)); higher values reduce
            the influence of rank differences toward the top of each list.

    Returns:
        List of fused result dicts sorted descending by fused_score, each
        containing: chunk_id, text, metadata, fused_score, vector_score
        (cosine similarity, or None if not found by vector search),
        bm25_score (or None if not found by BM25).
    """
    fused: dict[str, dict] = {}

    for rank, result in enumerate(vector_results):
        chunk_id = result["chunk_id"]
        # ChromaDB with hnsw:space="cosine" returns cosine *distance*;
        # convert to similarity for a more intuitive 0-1 "higher is better" score.
        vector_similarity = 1.0 - result["distance"]
        entry = fused.setdefault(
            chunk_id,
            {
                "chunk_id": chunk_id,
                "text": result["text"],
                "metadata": result["metadata"],
                "fused_score": 0.0,
                "vector_score": None,
                "bm25_score": None,
            },
        )
        entry["vector_score"] = vector_similarity
        entry["fused_score"] += 1.0 / (rrf_k + rank + 1)

    for rank, result in enumerate(bm25_results):
        chunk_id = result["chunk_id"]
        entry = fused.setdefault(
            chunk_id,
            {
                "chunk_id": chunk_id,
                "text": result["text"],
                "metadata": result["metadata"],
                "fused_score": 0.0,
                "vector_score": None,
                "bm25_score": None,
            },
        )
        entry["bm25_score"] = result["bm25_score"]
        entry["fused_score"] += 1.0 / (rrf_k + rank + 1)

    fused_list = sorted(fused.values(), key=lambda e: e["fused_score"], reverse=True)
    logger.debug(
        "RRF fused %d vector result(s) and %d BM25 result(s) into %d unique chunk(s).",
        len(vector_results),
        len(bm25_results),
        len(fused_list),
    )
    return fused_list
