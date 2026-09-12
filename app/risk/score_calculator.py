"""
score_calculator.py

Deterministic risk score formula for the Contract Intelligence Platform.

FORMULA (documented here and in docs/DESIGN_DECISIONS.md):
  risk_score = min(sum(weight_i for each triggered flag_i), max_score)

Where:
  - weight_i comes from config/settings.yaml (risk_scoring.weights)
  - max_score comes from config/settings.yaml (risk_scoring.max_score = 100)
  - The LLM never produces or influences the score — it only classifies
    triggered/not-triggered; the score is computed deterministically here.

Weights (from settings.yaml at time of writing):
  missing_signature_field:      25
  missing_termination_clause:   20
  missing_confidentiality_clause: 15
  unlimited_indemnity_language: 20
  high_liability_exposure:      15
  ambiguous_payment_terms:      10
  ─────────────────────────────────
  Max possible (all triggered): 105 → clipped to 100
"""

from __future__ import annotations

from app.models.risk import RiskFlag
from app.utils.config_loader import get_settings
from app.utils.logging_setup import get_logger

logger = get_logger(__name__)


def calculate_risk_score(flags: list[RiskFlag]) -> int:
    """
    Compute the deterministic risk score from a list of evaluated flags.

    Args:
        flags: All evaluated RiskFlag objects (both triggered and not
            triggered). Non-triggered flags contribute 0 to the score.

    Returns:
        An integer risk score in [0, 100].
    """
    settings = get_settings()
    max_score: int = settings.get("risk_scoring", {}).get("max_score", 100)

    raw_score = sum(flag.weight for flag in flags if flag.triggered)
    score = min(raw_score, max_score)

    triggered_summary = [
        f"{f.category.value}(+{f.weight})" for f in flags if f.triggered
    ]
    logger.info(
        "Risk score calculated: %d (raw=%d, max=%d). Triggered: %s",
        score, raw_score, max_score,
        triggered_summary if triggered_summary else "none",
    )
    return score
