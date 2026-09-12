"""
hybrid_retriever.py

Top-level hybrid retrieval entry point: embeds the query, runs vector
search (ChromaDB) and BM25 keyword search in parallel candidate pools,
fuses them via reciprocal rank fusion, and returns the configured top_k
chunks as a validated RetrievalResult. This is the single function the
Q&A module (Phase 5) and comparison/risk modules call.
"""

from __future__ import annotations

from app.embeddings.chroma_store import query_similar
from app.embeddings.embedder import embed_query
from app.models.retrieval import RetrievalResult, RetrievedChunk
from app.retriever.bm25_index import search_bm25
from app.retriever.fusion import reciprocal_rank_fusion
from app.utils.config_loader import get_retrieval_config
from app.utils.logging_setup import get_logger

logger = get_logger(__name__)


def hybrid_retrieve(query: str) -> RetrievalResult:
    """
    Run hybrid (vector + BM25) retrieval for a user query and return the
    fused top-K chunks.

    Args:
        query: The raw user question or search text.

    Returns:
        A RetrievalResult containing the top_k fused chunks (per
        retrieval.top_k in retrieval.yaml) and an above_threshold flag
        indicating whether any returned chunk met the configured
        similarity_threshold via vector search. Q&A (Phase 5) uses this
        flag to decide whether to answer at all.
    """
    retrieval_cfg = get_retrieval_config()["retrieval"]
    top_k = retrieval_cfg["top_k"]
    vector_top_k = retrieval_cfg["vector_top_k"]
    bm25_top_k = retrieval_cfg["bm25_top_k"]
    similarity_threshold = retrieval_cfg["similarity_threshold"]
    rrf_k = get_retrieval_config()["fusion"]["rrf_k"]

    if not query or not query.strip():
        logger.warning("hybrid_retrieve called with an empty query.")
        return RetrievalResult(query=query, chunks=[], above_threshold=False)

    query_embedding = embed_query(query)
    vector_raw = query_similar(query_embedding, top_k=vector_top_k)
    bm25_raw = search_bm25(query, top_k=bm25_top_k)

    fused = reciprocal_rank_fusion(vector_raw, bm25_raw, rrf_k)
    top_fused = fused[:top_k]

    above_threshold = any(
        entry["vector_score"] is not None and entry["vector_score"] >= similarity_threshold
        for entry in top_fused
    )

    chunks = [
        RetrievedChunk(
            chunk_id=entry["chunk_id"],
            document_id=entry["metadata"]["document_id"],
            contract_name=entry["metadata"]["contract_name"],
            page_number=entry["metadata"]["page_number"],
            text=entry["text"],
            vector_score=entry["vector_score"],
            bm25_score=entry["bm25_score"],
            fused_score=entry["fused_score"],
        )
        for entry in top_fused
    ]

    logger.info(
        "Hybrid retrieval for query '%s': %d vector candidate(s), %d BM25 candidate(s), "
        "%d fused result(s) returned (above_threshold=%s).",
        query,
        len(vector_raw),
        len(bm25_raw),
        len(chunks),
        above_threshold,
    )

    return RetrievalResult(query=query, chunks=chunks, above_threshold=above_threshold)
