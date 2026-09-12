"""
test_eval_harness.py

Unit tests for evaluation/eval_harness.py. Uses synthetic inputs so these
tests are deterministic and require no model downloads or live pipeline.
"""

from __future__ import annotations

import json

import pytest

from app.models.comparison import ContractComparisonResult, FieldInconsistency, MismatchDetectionMethod
from app.models.extraction import FieldType
from app.models.risk import RiskAssessment, RiskCategory, RiskFlag, RiskFlagSource
from evaluation.eval_harness import (
    inconsistency_detection_metrics,
    retrieval_recall_at_k,
    risk_detection_metrics,
    schema_validation_rate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_comparison_result(pair_id: str, detected_field_types: list[str]) -> tuple[str, ContractComparisonResult]:
    inconsistencies = [
        FieldInconsistency(
            field_type=FieldType(ft),
            detection_method=MismatchDetectionMethod.RULE_BASED_NUMERIC_DATE,
            contract_a_name="A", contract_a_page=1, contract_a_text="x",
            contract_b_name="B", contract_b_page=1, contract_b_text="y",
            explanation="mismatch", confidence=1.0,
        )
        for ft in detected_field_types
    ]
    return pair_id, ContractComparisonResult(
        document_id_a="doc-1", document_id_b="doc-2",
        contract_a_name="A", contract_b_name="B",
        inconsistencies=inconsistencies,
        fields_compared=5,
    )


def _make_risk_assessment(doc_id: str, triggered_cats: list[str]) -> RiskAssessment:
    all_cats = [
        RiskCategory.MISSING_SIGNATURE_FIELD,
        RiskCategory.MISSING_TERMINATION_CLAUSE,
        RiskCategory.MISSING_CONFIDENTIALITY_CLAUSE,
    ]
    flags = [
        RiskFlag(
            category=cat,
            source=RiskFlagSource.RULE_BASED,
            triggered=cat.value in triggered_cats,
            weight=20,
            explanation="test",
        )
        for cat in all_cats
    ]
    score = sum(f.weight for f in flags if f.triggered)
    return RiskAssessment(document_id=doc_id, contract_name="C", flags=flags, risk_score=min(score, 100))


# ---------------------------------------------------------------------------
# Inconsistency detection metrics
# ---------------------------------------------------------------------------

class TestInconsistencyDetectionMetrics:
    def test_all_correct_detections_yields_perfect_scores(self) -> None:
        _, r1 = _make_comparison_result("p1", ["payment_terms"])
        _, r2 = _make_comparison_result("p2", ["key_dates"])
        comparison_results = {"p1": r1, "p2": r2}
        ground_truth = [
            {"pair_id": "p1", "expected_inconsistencies": [{"field_type": "payment_terms"}]},
            {"pair_id": "p2", "expected_inconsistencies": [{"field_type": "key_dates"}]},
        ]
        m = inconsistency_detection_metrics(comparison_results, ground_truth)
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0
        assert m["tp"] == 2
        assert m["fp"] == 0
        assert m["fn"] == 0

    def test_false_positive_reduces_precision(self) -> None:
        _, r1 = _make_comparison_result("p1", ["payment_terms", "key_dates"])  # extra FP
        comparison_results = {"p1": r1}
        ground_truth = [
            {"pair_id": "p1", "expected_inconsistencies": [{"field_type": "payment_terms"}]},
        ]
        m = inconsistency_detection_metrics(comparison_results, ground_truth)
        assert m["fp"] == 1
        assert m["precision"] < 1.0

    def test_missed_inconsistency_reduces_recall(self) -> None:
        _, r1 = _make_comparison_result("p1", [])  # detected nothing
        comparison_results = {"p1": r1}
        ground_truth = [
            {"pair_id": "p1", "expected_inconsistencies": [{"field_type": "payment_terms"}]},
        ]
        m = inconsistency_detection_metrics(comparison_results, ground_truth)
        assert m["fn"] == 1
        assert m["recall"] == 0.0

    def test_true_negative_pair_contributes_no_tp_fp_fn(self) -> None:
        _, r1 = _make_comparison_result("p_tn", [])
        comparison_results = {"p_tn": r1}
        ground_truth = [{"pair_id": "p_tn", "expected_inconsistencies": []}]
        m = inconsistency_detection_metrics(comparison_results, ground_truth)
        assert m["tp"] == 0
        assert m["fp"] == 0
        assert m["fn"] == 0

    def test_f1_formula_correct(self) -> None:
        _, r = _make_comparison_result("p1", ["payment_terms"])
        m = inconsistency_detection_metrics(
            {"p1": r},
            [{"pair_id": "p1", "expected_inconsistencies": [{"field_type": "payment_terms"}]}],
        )
        expected_f1 = 2 * m["precision"] * m["recall"] / (m["precision"] + m["recall"])
        assert abs(m["f1"] - expected_f1) < 1e-6


# ---------------------------------------------------------------------------
# BM25 retrieval recall
# ---------------------------------------------------------------------------

class TestRetrievalRecallAtK:
    def test_all_found_at_rank_1(self) -> None:
        results = [
            {"qa_id": f"q{i}", "source_page": 1, "found_at_rank": 1, "bm25_result_count": 10}
            for i in range(5)
        ]
        m = retrieval_recall_at_k(results)
        assert m["recall@1"] == 1.0
        assert m["recall@3"] == 1.0
        assert m["recall@5"] == 1.0
        assert m["mean_reciprocal_rank"] == 1.0

    def test_found_at_rank_6_not_in_top5(self) -> None:
        results = [{"qa_id": "q1", "source_page": 1, "found_at_rank": 6, "bm25_result_count": 20}]
        m = retrieval_recall_at_k(results)
        assert m["recall@5"] == 0.0

    def test_mixed_ranks(self) -> None:
        results = [
            {"qa_id": "q1", "found_at_rank": 1, "source_page": 1, "bm25_result_count": 10},
            {"qa_id": "q2", "found_at_rank": 4, "source_page": 1, "bm25_result_count": 10},
            {"qa_id": "q3", "found_at_rank": None, "source_page": 1, "bm25_result_count": 10},
            {"qa_id": "q4", "found_at_rank": 6, "source_page": 1, "bm25_result_count": 10},
        ]
        m = retrieval_recall_at_k(results)
        assert m["recall@1"] == pytest.approx(0.25)
        assert m["recall@3"] == pytest.approx(0.25)
        assert m["recall@5"] == pytest.approx(0.50)
        assert m["n_questions"] == 4

    def test_empty_results_returns_zeros(self) -> None:
        m = retrieval_recall_at_k([])
        assert m["recall@5"] == 0
        assert m["n_questions"] == 0


# ---------------------------------------------------------------------------
# Risk detection metrics
# ---------------------------------------------------------------------------

class TestRiskDetectionMetrics:
    def test_perfect_detection(self) -> None:
        risk_results = {
            "doc-1": _make_risk_assessment("doc-1", ["missing_signature_field"]),
            "doc-2": _make_risk_assessment("doc-2", []),
        }
        ground_truth = [
            {"contract_id": "doc-1", "expected_triggered_rule_flags": ["missing_signature_field"]},
            {"contract_id": "doc-2", "expected_triggered_rule_flags": []},
        ]
        m = risk_detection_metrics(risk_results, ground_truth)
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0

    def test_missed_flag_reduces_recall(self) -> None:
        risk_results = {"doc-1": _make_risk_assessment("doc-1", [])}  # missed the flag
        ground_truth = [
            {"contract_id": "doc-1", "expected_triggered_rule_flags": ["missing_signature_field"]}
        ]
        m = risk_detection_metrics(risk_results, ground_truth)
        assert m["fn"] >= 1
        assert m["recall"] < 1.0


# ---------------------------------------------------------------------------
# Schema validation rate
# ---------------------------------------------------------------------------

class TestSchemaValidationRate:
    def _valid_risk_json(self) -> str:
        return json.dumps({"flags": [
            {"category": "unlimited_indemnity_language", "triggered": False, "explanation": "ok",
             "source_contract": "C", "source_page": 1, "supporting_text": "x"},
            {"category": "high_liability_exposure", "triggered": False, "explanation": "ok",
             "source_contract": "C", "source_page": 1, "supporting_text": "x"},
            {"category": "ambiguous_payment_terms", "triggered": False, "explanation": "ok",
             "source_contract": "C", "source_page": 1, "supporting_text": "x"},
        ]})

    def test_all_valid_responses_pass(self) -> None:
        corpus = [
            {"response_id": f"r{i}", "schema": "risk",
             "json_str": self._valid_risk_json(), "expected_valid": True}
            for i in range(5)
        ]
        m = schema_validation_rate(corpus)
        assert m["first_attempt_pass_rate"] == 1.0

    def test_invalid_json_correctly_rejected(self) -> None:
        corpus = [
            {"response_id": "bad", "schema": "risk",
             "json_str": "not json", "expected_valid": False}
        ]
        m = schema_validation_rate(corpus)
        assert m["correct_rejection_rate"] == 1.0

    def test_overall_accuracy_computed_correctly(self) -> None:
        corpus = [
            {"response_id": "v1", "schema": "risk",
             "json_str": self._valid_risk_json(), "expected_valid": True},
            {"response_id": "i1", "schema": "risk",
             "json_str": "bad", "expected_valid": False},
        ]
        m = schema_validation_rate(corpus)
        assert m["overall_accuracy"] == 1.0

    def test_unexpected_failure_reduces_accuracy(self) -> None:
        corpus = [
            {"response_id": "v1", "schema": "risk",
             "json_str": self._valid_risk_json(), "expected_valid": True},
            {"response_id": "v2", "schema": "risk",
             "json_str": "bad json but expected valid", "expected_valid": True},
        ]
        m = schema_validation_rate(corpus)
        assert m["first_attempt_pass_rate"] == pytest.approx(0.5)
        assert m["overall_accuracy"] == pytest.approx(0.5)
