"""
risk_page.py

Risk Analysis page. Runs the hybrid (rule-based + LLM-classified) risk
detection pipeline and displays the deterministic risk score alongside a
breakdown of every triggered flag with its source and citation.
"""

from __future__ import annotations

import streamlit as st

from app.extraction.extraction_service import extract_fields_from_document
from app.frontend.components import document_selector, risk_score_gauge, severity_badge, show_operation_error
from app.frontend.state import get_documents
from app.models.risk import RiskCategory, RiskFlagSource
from app.risk.exceptions import LLMClassificationError
from app.risk.llm_classifier import classify_llm_risk_flags
from app.risk.rule_based_flags import detect_rule_based_flags
from app.risk.score_calculator import calculate_risk_score


def render() -> None:
    st.header("⚠️ Risk Analysis")
    st.write(
        "Analyses a contract for six risk categories: three detected "
        "deterministically (missing clauses) and three classified by the LLM "
        "against a fixed risk taxonomy. The final score is computed via a "
        "documented deterministic formula — never invented by the LLM."
    )

    if not get_documents():
        st.info("No contracts indexed yet. Go to **Upload** to add contracts.")
        return

    meta = document_selector("Select contract for risk analysis", key="risk_doc")
    if not meta:
        return

    if not st.button("Analyse Risk", type="primary"):
        _render_result(st.session_state.get("last_risk_result"))
        return

    with st.spinner("Running rule-based checks…"):
        try:
            cache = st.session_state.setdefault("extraction_cache", {})
            if meta.document_id not in cache:
                cache[meta.document_id] = extract_fields_from_document(meta.document_id)
            extraction = cache[meta.document_id]
            rule_flags = detect_rule_based_flags(extraction)
        except Exception as exc:
            show_operation_error("Risk field extraction", exc)
            return

    # Keep rule-based findings available when provider classification fails.
    llm_flags = []
    llm_available = True
    with st.spinner("Running LLM risk classification…"):
        try:
            llm_flags = classify_llm_risk_flags(meta.document_id, meta.contract_name)
        except LLMClassificationError:
            llm_available = False
            st.warning(
                "LLM risk classification is unavailable. Rule-based flags are shown below; "
                "check the configured provider and try again to include LLM-classified flags."
            )
        except Exception as exc:
            llm_available = False
            st.warning(
                f"LLM classification could not complete ({type(exc).__name__}). "
                "Rule-based flags are shown below."
            )

    all_flags = rule_flags + llm_flags
    risk_score = calculate_risk_score(all_flags)

    from app.models.risk import RiskAssessment
    assessment = RiskAssessment(
        document_id=meta.document_id,
        contract_name=meta.contract_name,
        flags=all_flags,
        risk_score=risk_score,
    )
    st.session_state["last_risk_result"] = assessment

    if not llm_available:
        st.info(
            "Risk score below reflects rule-based flags only "
            "(LLM-classified flags not available)."
        )

    _render_result(assessment)


def _render_result(assessment) -> None:
    if assessment is None:
        return

    st.divider()
    st.subheader(f"Risk Assessment: **{assessment.contract_name}**")

    risk_score_gauge(assessment.risk_score)
    st.caption(
        "Score formula: `sum(weight_i for each triggered flag)`, clipped to 100. "
        "Weights: missing signature=25, missing termination=20, "
        "missing confidentiality=15, unlimited indemnity=20, "
        "high liability=15, ambiguous payment=10."
    )

    st.divider()

    triggered = assessment.triggered_flags()
    not_triggered = [f for f in assessment.flags if not f.triggered]

    if triggered:
        st.subheader(f"🚨 Triggered Flags ({len(triggered)})")
        for flag in triggered:
            sev = "high" if flag.weight >= 20 else ("medium" if flag.weight >= 15 else "low")
            label = flag.category.value.replace("_", " ").title()
            source_label = (
                "Rule-based" if flag.source == RiskFlagSource.RULE_BASED
                else "LLM-classified"
            )
            with st.expander(
                f"{severity_badge(sev)} **{label}** — {source_label} (+{flag.weight} pts)",
                expanded=True,
            ):
                st.write(flag.explanation)
                if flag.source_contract and flag.source_page:
                    st.caption(
                        f"Source: *{flag.source_contract}*, p. {flag.source_page}"
                    )
                if flag.supporting_text:
                    st.text(flag.supporting_text)
    else:
        st.success("✓ No risk flags triggered.")

    if not_triggered:
        with st.expander(f"Passed checks ({len(not_triggered)})"):
            for flag in not_triggered:
                label = flag.category.value.replace("_", " ").title()
                st.write(f"✓ {label}")
