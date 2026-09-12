"""
compare_page.py

Contract Comparison page. Extracts structured fields from two contracts,
aligns them, and flags inconsistencies via rule-based (numeric/date) and
NLI-based (semantic contradiction) detection.
"""

from __future__ import annotations

import streamlit as st

from app.comparison.comparison_service import compare_contracts
from app.extraction.extraction_service import extract_fields_from_document
from app.frontend.components import document_selector
from app.frontend.state import get_documents
from app.models.comparison import MismatchDetectionMethod


def render() -> None:
    st.header("🔄 Contract Comparison")
    st.write(
        "Select two contracts to compare. Inconsistencies are detected using "
        "deterministic rule-based checks (numeric/date fields) and NLI semantic "
        "contradiction detection (clause text)."
    )

    if len(get_documents()) < 2:
        st.info("You need at least **two indexed contracts** to run a comparison.")
        return

    # ── Contract selectors ───────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        meta_a = document_selector("Contract A", key="compare_a")
    with col_b:
        exclude = meta_a.document_id if meta_a else None
        meta_b = document_selector("Contract B", key="compare_b", exclude_id=exclude)

    if not (meta_a and meta_b):
        return
    if meta_a.document_id == meta_b.document_id:
        st.warning("Please select two different contracts.")
        return

    if not st.button("Compare Contracts", type="primary"):
        _render_result(st.session_state.get("last_comparison_result"))
        return

    with st.spinner("Extracting fields and detecting inconsistencies…"):
        try:
            cache = st.session_state.setdefault("extraction_cache", {})
            if meta_a.document_id not in cache:
                cache[meta_a.document_id] = extract_fields_from_document(meta_a.document_id)
            if meta_b.document_id not in cache:
                cache[meta_b.document_id] = extract_fields_from_document(meta_b.document_id)

            extraction_a = cache[meta_a.document_id]
            extraction_b = cache[meta_b.document_id]

            result = compare_contracts(extraction_a, extraction_b)
            st.session_state["last_comparison_result"] = result

        except Exception as exc:
            st.error(f"Comparison failed: {exc}")
            if "NLI" in str(exc) or "Groundedness" in str(exc):
                st.info(
                    "**Tip:** NLI-based detection requires the DeBERTa model. "
                    "Rule-based results (numeric/date mismatches) still ran."
                )
            return

    _render_result(st.session_state.get("last_comparison_result"))


def _render_result(result) -> None:
    if result is None:
        return

    st.divider()
    st.subheader(
        f"Comparing: **{result.contract_a_name}** vs **{result.contract_b_name}**"
    )

    col_fields, col_issues = st.columns(2)
    col_fields.metric("Fields Compared", result.fields_compared)
    col_issues.metric(
        "Inconsistencies Found",
        len(result.inconsistencies),
        delta=None if result.inconsistencies else "✓ No issues",
    )

    if not result.inconsistencies:
        st.success(
            "✓ No inconsistencies detected across all compared fields "
            f"({result.fields_compared} field types evaluated)."
        )
        return

    for inc in result.inconsistencies:
        method_label = (
            "🔢 Rule-based (numeric/date)"
            if inc.detection_method == MismatchDetectionMethod.RULE_BASED_NUMERIC_DATE
            else f"🧠 NLI semantic contradiction (confidence: {inc.confidence:.0%})"
        )
        field_label = inc.field_type.value.replace("_", " ").title()

        with st.expander(f"⚠ {field_label} — {method_label}", expanded=True):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**{inc.contract_a_name}** (p. {inc.contract_a_page})")
                st.markdown(f"> {inc.contract_a_text[:400]}")
            with col_b:
                st.markdown(f"**{inc.contract_b_name}** (p. {inc.contract_b_page})")
                st.markdown(f"> {inc.contract_b_text[:400]}")
            st.caption(f"**Explanation:** {inc.explanation}")
