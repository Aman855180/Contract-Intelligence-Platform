"""
test_fusion.py

Unit tests for app/retriever/fusion.py (reciprocal rank fusion). Pure
logic tests with hand-crafted vector/BM25 result lists — no I/O needed.
"""

from __future__ import annotations

from app.retriever.fusion import reciprocal_rank_fusion


def _vec_result(chunk_id: str, distance: float, text: str = "text", doc_id: str = "doc-1") -> dict:
    return {
        "chunk_id": chunk_id,
        "text": text,
        "metadata": {"document_id": doc_id, "contract_name": "Contract", "page_number": 1},
        "distance": distance,
    }


def _bm25_result(chunk_id: str, score: float, text: str = "text", doc_id: str = "doc-1") -> dict:
    return {
        "chunk_id": chunk_id,
        "text": text,
        "metadata": {"document_id": doc_id, "contract_name": "Contract", "page_number": 1},
        "bm25_score": score,
    }


class TestReciprocalRankFusion:
    def test_chunk_in_both_lists_outranks_chunk_in_one_list(self) -> None:
        vector_results = [_vec_result("a", distance=0.1), _vec_result("b", distance=0.2)]
        bm25_results = [_bm25_result("a", score=5.0), _bm25_result("c", score=4.0)]

        fused = reciprocal_rank_fusion(vector_results, bm25_results, rrf_k=60)

        # "a" appears in both lists at rank 0, so it should be ranked first overall.
        assert fused[0]["chunk_id"] == "a"
        assert fused[0]["vector_score"] is not None
        assert fused[0]["bm25_score"] is not None

    def test_fused_score_matches_rrf_formula(self) -> None:
        vector_results = [_vec_result("a", distance=0.1)]
        bm25_results = [_bm25_result("a", score=10.0)]
        rrf_k = 60

        fused = reciprocal_rank_fusion(vector_results, bm25_results, rrf_k=rrf_k)

        expected_score = 1.0 / (rrf_k + 0 + 1) + 1.0 / (rrf_k + 0 + 1)
        assert abs(fused[0]["fused_score"] - expected_score) < 1e-9

    def test_chunk_only_in_vector_results_has_none_bm25_score(self) -> None:
        vector_results = [_vec_result("a", distance=0.05)]
        bm25_results: list[dict] = []

        fused = reciprocal_rank_fusion(vector_results, bm25_results, rrf_k=60)

        assert fused[0]["chunk_id"] == "a"
        assert fused[0]["bm25_score"] is None
        assert fused[0]["vector_score"] == 0.95  # 1 - distance

    def test_chunk_only_in_bm25_results_has_none_vector_score(self) -> None:
        vector_results: list[dict] = []
        bm25_results = [_bm25_result("a", score=3.5)]

        fused = reciprocal_rank_fusion(vector_results, bm25_results, rrf_k=60)

        assert fused[0]["chunk_id"] == "a"
        assert fused[0]["vector_score"] is None
        assert fused[0]["bm25_score"] == 3.5

    def test_both_empty_returns_empty_list(self) -> None:
        assert reciprocal_rank_fusion([], [], rrf_k=60) == []

    def test_results_sorted_descending_by_fused_score(self) -> None:
        vector_results = [_vec_result("a", distance=0.1), _vec_result("b", distance=0.5), _vec_result("c", distance=0.9)]
        bm25_results: list[dict] = []

        fused = reciprocal_rank_fusion(vector_results, bm25_results, rrf_k=60)

        scores = [entry["fused_score"] for entry in fused]
        assert scores == sorted(scores, reverse=True)
        assert [entry["chunk_id"] for entry in fused] == ["a", "b", "c"]

    def test_lower_rrf_k_amplifies_rank_position_effect(self) -> None:
        # With a small rrf_k, the gap between rank 0 and rank 1 contributions is larger.
        vector_results = [_vec_result("a", distance=0.1), _vec_result("b", distance=0.2)]
        bm25_results: list[dict] = []

        fused_small_k = reciprocal_rank_fusion(vector_results, bm25_results, rrf_k=1)
        fused_large_k = reciprocal_rank_fusion(vector_results, bm25_results, rrf_k=1000)

        gap_small_k = fused_small_k[0]["fused_score"] - fused_small_k[1]["fused_score"]
        gap_large_k = fused_large_k[0]["fused_score"] - fused_large_k[1]["fused_score"]
        assert gap_small_k > gap_large_k
