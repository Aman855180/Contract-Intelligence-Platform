"""
test_risk_score.py

Unit tests for app/risk/score_calculator.py, verifying the deterministic
weighted formula, the [0,100] clip, and config-driven weight usage.
"""

from __future__ import annotations

import pytest

from app.models.risk import RiskCategory, RiskFlag, RiskFlagSource
from app.risk.score_calculator import calculate_risk_score


def _make_flag(category: RiskCategory, triggered: bool, weight: int) -> RiskFlag:
    return RiskFlag(
        category=category,
        source=RiskFlagSource.RULE_BASED,
        triggered=triggered,
        weight=weight,
        explanation="Test explanation.",
    )


class TestCalculateRiskScore:
    def test_no_triggered_flags_score_is_zero(self) -> None:
        flags = [
            _make_flag(RiskCategory.MISSING_SIGNATURE_FIELD, False, 25),
            _make_flag(RiskCategory.MISSING_TERMINATION_CLAUSE, False, 20),
        ]
        assert calculate_risk_score(flags) == 0

    def test_single_triggered_flag_equals_its_weight(self) -> None:
        flags = [
            _make_flag(RiskCategory.MISSING_SIGNATURE_FIELD, True, 25),
            _make_flag(RiskCategory.MISSING_TERMINATION_CLAUSE, False, 20),
        ]
        assert calculate_risk_score(flags) == 25

    def test_multiple_triggered_flags_summed(self) -> None:
        flags = [
            _make_flag(RiskCategory.MISSING_SIGNATURE_FIELD, True, 25),
            _make_flag(RiskCategory.MISSING_TERMINATION_CLAUSE, True, 20),
            _make_flag(RiskCategory.MISSING_CONFIDENTIALITY_CLAUSE, False, 15),
        ]
        assert calculate_risk_score(flags) == 45

    def test_all_flags_triggered_clipped_at_100(self) -> None:
        # All six categories triggered: 25+20+15+20+15+10 = 105 → clipped to 100
        flags = [
            _make_flag(RiskCategory.MISSING_SIGNATURE_FIELD, True, 25),
            _make_flag(RiskCategory.MISSING_TERMINATION_CLAUSE, True, 20),
            _make_flag(RiskCategory.MISSING_CONFIDENTIALITY_CLAUSE, True, 15),
            _make_flag(RiskCategory.UNLIMITED_INDEMNITY_LANGUAGE, True, 20),
            _make_flag(RiskCategory.HIGH_LIABILITY_EXPOSURE, True, 15),
            _make_flag(RiskCategory.AMBIGUOUS_PAYMENT_TERMS, True, 10),
        ]
        assert calculate_risk_score(flags) == 100

    def test_empty_flags_list_returns_zero(self) -> None:
        assert calculate_risk_score([]) == 0

    def test_only_non_triggered_flags_returns_zero(self) -> None:
        flags = [_make_flag(cat, False, 25) for cat in RiskCategory]
        assert calculate_risk_score(flags) == 0

    def test_score_within_bounds(self) -> None:
        flags = [_make_flag(cat, True, 50) for cat in RiskCategory]  # absurdly high weights
        score = calculate_risk_score(flags)
        assert 0 <= score <= 100

    def test_non_triggered_flags_contribute_zero_regardless_of_weight(self) -> None:
        flags = [
            _make_flag(RiskCategory.MISSING_SIGNATURE_FIELD, False, 99),
            _make_flag(RiskCategory.MISSING_TERMINATION_CLAUSE, True, 10),
        ]
        assert calculate_risk_score(flags) == 10
