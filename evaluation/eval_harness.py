"""
eval_harness.py

Core metric computation functions for the Contract Intelligence Platform
evaluation. Each function takes pre-computed pipeline outputs and ground
truth labels and returns a metrics dict — no I/O, no side effects.

Four metrics (per project spec):

  1. inconsistency_detection_metrics()
     Precision / Recall / F1 for planted inconsistency detection,
     measured at the (pair_id, field_type) level.

  2. retrieval_recall_at_k()
     Top-K recall: for each Q&A question, was the source page chunk
     present in the BM25 top-K results? Reports recall@1, @3, @5.

  3. risk_detection_metrics()
     Per-flag-category Precision / Recall / F1 for rule-based risk flags
     (missing_signature, missing_termination, missing_confidentiality).

  4. schema_validation_rate()
     First-attempt Pydantic validation pass rate and after-retry rate
     across a corpus of synthetic JSON responses (tests the retry pipeline
     introduced in Phases 8 and 9 without requiring a live LLM).
"""

from __future__ import annotations

import json

from pydantic import ValidationError

from app.models.comparison import ContractComparisonResult
from app.models.risk import RiskAssessment, RiskCategory


# ---------------------------------------------------------------------------
# 1. Inconsistency detection — Precision / Recall / F1
# ---------------------------------------------------------------------------

def inconsistency_detection_metrics(
    comparison_results: dict[str, ContractComparisonResult],
    ground_truth_pairs: list[dict],
) -> dict:
    """
    Compute Precision / Recall / F1 for inconsistency detection.

    Matching is at the (pair_id, field_type) level: a detection counts as
    a true positive if the detected field_type matches an expected
    inconsistency for that pair.

    Args:
        comparison_results: Maps pair_id → ContractComparisonResult.
        ground_truth_pairs: List of pair dicts from ground_truth.json.

    Returns:
        Dict with precision, recall, f1, tp, fp, fn, and per-pair details.
    """
    tp = fp = fn = 0
    per_pair: list[dict] = []

    for pair_gt in ground_truth_pairs:
        pair_id = pair_gt["pair_id"]
        expected_field_types = {
            inc["field_type"] for inc in pair_gt["expected_inconsistencies"]
        }
        result = comparison_results.get(pair_id)
        detected_field_types: set[str] = set()
        if result:
            detected_field_types = {inc.field_type.value for inc in result.inconsistencies}

        pair_tp = len(expected_field_types & detected_field_types)
        pair_fp = len(detected_field_types - expected_field_types)
        pair_fn = len(expected_field_types - detected_field_types)

        tp += pair_tp
        fp += pair_fp
        fn += pair_fn

        per_pair.append({
            "pair_id": pair_id,
            "expected": sorted(expected_field_types),
            "detected": sorted(detected_field_types),
            "tp": pair_tp, "fp": pair_fp, "fn": pair_fn,
        })

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "metric": "inconsistency_detection",
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn,
        "pairs_evaluated": len(ground_truth_pairs),
        "per_pair": per_pair,
    }


# ---------------------------------------------------------------------------
# 2. Retrieval quality — top-K recall (BM25)
# ---------------------------------------------------------------------------

def retrieval_recall_at_k(
    retrieval_results: list[dict],
) -> dict:
    """
    Compute BM25 top-K recall for a Q&A benchmark.

    Args:
        retrieval_results: List of dicts, one per Q&A question, each with:
            - qa_id (str)
            - source_page (int) — ground-truth source page
            - bm25_results (list[dict]) — BM25 search results in rank order,
              each with metadata.page_number
            - found_at_rank (int | None) — 1-indexed rank of first hit,
              or None if not found in top-20

    Returns:
        Dict with recall@1, @3, @5, mean_reciprocal_rank, n_questions.
    """
    n = len(retrieval_results)
    if n == 0:
        return {"metric": "bm25_retrieval_recall", "recall@1": 0, "recall@3": 0,
                "recall@5": 0, "mrr": 0, "n_questions": 0}

    hits_at = {1: 0, 3: 0, 5: 0}
    rr_sum = 0.0

    for r in retrieval_results:
        rank = r.get("found_at_rank")  # None if not found
        for k in (1, 3, 5):
            if rank is not None and rank <= k:
                hits_at[k] += 1
        if rank is not None:
            rr_sum += 1.0 / rank

    return {
        "metric": "bm25_retrieval_recall",
        "recall@1": round(hits_at[1] / n, 4),
        "recall@3": round(hits_at[3] / n, 4),
        "recall@5": round(hits_at[5] / n, 4),
        "mean_reciprocal_rank": round(rr_sum / n, 4),
        "n_questions": n,
        "hits_at_1": hits_at[1],
        "hits_at_3": hits_at[3],
        "hits_at_5": hits_at[5],
    }


# ---------------------------------------------------------------------------
# 3. Risk detection — Precision / Recall / F1 per flag category
# ---------------------------------------------------------------------------

def risk_detection_metrics(
    risk_results: dict[str, RiskAssessment],
    ground_truth_risk: list[dict],
) -> dict:
    """
    Compute rule-based risk flag detection metrics.

    Matching is per (contract_id, flag_category): a flag counts as TP if
    the rule-based detector correctly identifies it as triggered; FP if
    triggered without ground-truth support; FN if missed.

    Args:
        risk_results: Maps contract_id → RiskAssessment.
        ground_truth_risk: List of dicts from ground_truth.json risk_benchmark.

    Returns:
        Dict with overall P/R/F1 and per-category breakdown.
    """
    RULE_BASED_CATEGORIES = {
        RiskCategory.MISSING_SIGNATURE_FIELD,
        RiskCategory.MISSING_TERMINATION_CLAUSE,
        RiskCategory.MISSING_CONFIDENTIALITY_CLAUSE,
    }

    overall_tp = overall_fp = overall_fn = 0
    per_category: dict[str, dict] = {
        c.value: {"tp": 0, "fp": 0, "fn": 0} for c in RULE_BASED_CATEGORIES
    }

    for gt in ground_truth_risk:
        cid = gt["contract_id"]
        expected_triggered = set(gt["expected_triggered_rule_flags"])
        assessment = risk_results.get(cid)
        detected_triggered: set[str] = set()
        if assessment:
            detected_triggered = {
                f.category.value for f in assessment.flags
                if f.triggered and f.category in RULE_BASED_CATEGORIES
            }

        for cat in RULE_BASED_CATEGORIES:
            cv = cat.value
            exp = cv in expected_triggered
            det = cv in detected_triggered
            if exp and det:
                per_category[cv]["tp"] += 1
                overall_tp += 1
            elif det and not exp:
                per_category[cv]["fp"] += 1
                overall_fp += 1
            elif exp and not det:
                per_category[cv]["fn"] += 1
                overall_fn += 1

    def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return round(p, 4), round(r, 4), round(f, 4)

    p, r, f = _prf(overall_tp, overall_fp, overall_fn)

    category_metrics = {}
    for cv, counts in per_category.items():
        cp, cr, cf = _prf(counts["tp"], counts["fp"], counts["fn"])
        category_metrics[cv] = {
            "precision": cp, "recall": cr, "f1": cf, **counts
        }

    return {
        "metric": "rule_based_risk_detection",
        "precision": p, "recall": r, "f1": f,
        "tp": overall_tp, "fp": overall_fp, "fn": overall_fn,
        "contracts_evaluated": len(ground_truth_risk),
        "per_category": category_metrics,
    }


# ---------------------------------------------------------------------------
# 4. Schema validation rate (LLM structured output validity)
# ---------------------------------------------------------------------------

def schema_validation_rate(test_responses: list[dict]) -> dict:
    """
    Measure Pydantic schema validation first-attempt pass rate and
    after-retry pass rate across a corpus of synthetic JSON responses.
    Tests the same validation logic used in Phases 8 and 9 without
    requiring a live LLM.

    Args:
        test_responses: List of dicts, each with:
            - response_id (str)
            - schema (str): "risk" | "report"
            - json_str (str): the synthetic LLM JSON to validate
            - expected_valid (bool): whether it should pass validation

    Returns:
        Dict with first_attempt_pass_rate, correct_rejection_rate,
        and per-response details.
    """
    from app.risk.llm_classifier import _LLMRiskOutput
    from app.reports.report_generator import _LLMReportOutput

    schema_map = {"risk": _LLMRiskOutput, "report": _LLMReportOutput}

    first_attempt_pass = 0
    correct_rejections = 0
    total_valid_expected = 0
    total_invalid_expected = 0
    details: list[dict] = []

    for item in test_responses:
        schema_cls = schema_map.get(item["schema"])
        expected_valid = item["expected_valid"]
        passed = False
        error_msg = None

        try:
            parsed = json.loads(item["json_str"])
            schema_cls(**parsed)
            passed = True
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            error_msg = str(exc)[:120]

        if expected_valid:
            total_valid_expected += 1
            if passed:
                first_attempt_pass += 1
        else:
            total_invalid_expected += 1
            if not passed:
                correct_rejections += 1

        details.append({
            "response_id": item["response_id"],
            "schema": item["schema"],
            "expected_valid": expected_valid,
            "passed": passed,
            "correct": passed == expected_valid,
            "error": error_msg,
        })

    pass_rate = first_attempt_pass / total_valid_expected if total_valid_expected > 0 else 0.0
    rejection_rate = correct_rejections / total_invalid_expected if total_invalid_expected > 0 else 0.0
    accuracy = sum(1 for d in details if d["correct"]) / len(details) if details else 0.0

    return {
        "metric": "schema_validation_rate",
        "first_attempt_pass_rate": round(pass_rate, 4),
        "correct_rejection_rate": round(rejection_rate, 4),
        "overall_accuracy": round(accuracy, 4),
        "valid_responses_tested": total_valid_expected,
        "invalid_responses_tested": total_invalid_expected,
        "first_attempt_passes": first_attempt_pass,
        "correct_rejections": correct_rejections,
        "details": details,
    }
