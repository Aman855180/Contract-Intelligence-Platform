"""
report_page.py

Executive Summary / Report Generation page. Runs the JSON-first pipeline
(LLM → Pydantic validation → Markdown rendering) and displays the report
with a download button.
"""

from __future__ import annotations

import streamlit as st

from app.extraction.extraction_service import extract_fields_from_document
from app.frontend.components import document_selector
from app.frontend.state import get_documents
from app.reports.exceptions import ReportValidationError
from app.reports.report_service import generate_contract_report
from app.risk.risk_service import assess_risk


def render() -> None:
    st.header("📄 Executive Summary Report")
    st.write(
        "Generates a structured, citation-bearing executive summary. "
        "The LLM output is validated against a Pydantic schema before any "
        "text is displayed — the system retries automatically if the LLM "
        "returns malformed JSON."
    )

    if not get_documents():
        st.info("No contracts indexed yet. Go to **Upload** to add contracts.")
        return

    meta = document_selector("Select contract", key="report_doc")
    if not meta:
        return

    if not st.button("Generate Report", type="primary"):
        _render_result(st.session_state.get("last_report_result"))
        return

    with st.spinner("Extracting fields, assessing risk, and generating report…"):
        try:
            cache = st.session_state.setdefault("extraction_cache", {})
            if meta.document_id not in cache:
                cache[meta.document_id] = extract_fields_from_document(meta.document_id)
            extraction = cache[meta.document_id]

            # Risk is needed for the report's risk flags section. Gracefully
            # degrade to rule-based-only if LLM/Ollama isn't available.
            try:
                assessment = assess_risk(meta.document_id, extraction=extraction)
            except Exception:
                from app.models.risk import RiskAssessment
                from app.risk.rule_based_flags import detect_rule_based_flags
                from app.risk.score_calculator import calculate_risk_score
                rule_flags = detect_rule_based_flags(extraction)
                assessment = RiskAssessment(
                    document_id=meta.document_id,
                    contract_name=meta.contract_name,
                    flags=rule_flags,
                    risk_score=calculate_risk_score(rule_flags),
                )

            report_result = generate_contract_report(
                meta.document_id,
                extraction=extraction,
                assessment=assessment,
            )
            st.session_state["last_report_result"] = report_result

        except ReportValidationError as exc:
            st.error(
                f"Report generation failed after all retries: {exc}\n\n"
                "**Tip:** Ensure Ollama is running with `llama3.1:8b` loaded."
            )
            return
        except Exception as exc:
            err_str = str(exc)
            if "OLLAMA_TIMEOUT" in err_str:
                st.error(
                    "⏱ Ollama is running but report generation timed out. "
                    "Increase `request_timeout_seconds` in `config/settings.yaml` "
                    "(currently 300 s) and try again."
                )
            elif "OLLAMA_UNREACHABLE" in err_str or "refused" in err_str.lower():
                st.error(
                    "🔌 Cannot reach Ollama. Run `ollama serve` in a terminal, "
                    "then `ollama pull llama3.1:8b` if you haven't already."
                )
            else:
                st.error(f"Report generation failed: {exc}")
            return

    _render_result(st.session_state.get("last_report_result"))


def _render_result(report_result) -> None:
    if report_result is None:
        return

    report = report_result.report
    markdown = report_result.markdown

    st.divider()

    # ── Download ─────────────────────────────────────────────────────────────
    st.download_button(
        label="⬇ Download Report (Markdown)",
        data=markdown.encode("utf-8"),
        file_name=f"report_{report.contract_name.replace(' ', '_')}.md",
        mime="text/markdown",
    )

    # ── Metadata row ─────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Key Parties", len(report.key_parties))
    c2.metric("Key Dates", len(report.important_dates))
    c3.metric("Risk Flags", len(report.risk_flags))
    c4.metric("Key Clauses", len(report.key_clauses))

    st.divider()

    # ── Rendered Markdown ─────────────────────────────────────────────────────
    st.markdown(markdown)
