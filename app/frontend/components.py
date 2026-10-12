"""
components.py

Reusable Streamlit UI components shared across pages.
"""

from __future__ import annotations

import streamlit as st

from app.frontend.state import get_documents
from app.models.document import DocumentMetadata


def apply_app_styles() -> None:
    """Apply small layout and focus refinements over the configured Streamlit theme."""
    st.markdown(
        """
        <style>
        [data-testid="stMainBlockContainer"] {
          padding-top: 1.5rem;
          padding-bottom: 2rem;
        }
        [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
          padding-top: .9rem;
          padding-bottom: .9rem;
        }
        [data-testid="stSidebar"] [role="radiogroup"] {
          gap: .25rem;
        }
        h1 { font-size: 1.9rem !important; font-weight: 650 !important; }
        h2 { font-size: 1.4rem !important; font-weight: 600 !important; }
        h3 { font-size: 1.15rem !important; font-weight: 600 !important; }
        [data-testid="stFileUploaderDropzone"] {
          border-color: rgba(145, 176, 205, .55) !important;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
          border-color: #5b9bd5 !important;
        }
        button:focus-visible, a:focus-visible,
        input:focus-visible, textarea:focus-visible,
        [role="radio"]:focus-visible {
          outline: 2px solid #8fc4f3 !important;
          outline-offset: 2px;
        }
        [data-testid="stMarkdownContainer"] table {
          border-collapse: collapse;
        }
        [data-testid="stMarkdownContainer"] th,
        [data-testid="stMarkdownContainer"] td {
          border-bottom: 1px solid rgba(145, 176, 205, .22);
          padding: .45rem .65rem;
        }
        @media (max-width: 760px) {
          [data-testid="stMainBlockContainer"] {
            padding: 1rem .85rem 1.5rem;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def document_selector(
    label: str = "Select contract",
    key: str = "doc_selector",
    exclude_id: str | None = None,
) -> DocumentMetadata | None:
    """Render a selector for an indexed contract."""
    docs = {
        doc_id: meta
        for doc_id, meta in get_documents().items()
        if doc_id != exclude_id
    }

    if not docs:
        st.info("No contracts indexed in this session. Upload one on the **Upload** page.")
        return None

    options = list(docs.keys())
    format_fn = lambda doc_id: docs[doc_id].contract_name  # noqa: E731

    selected_id = st.selectbox(label, options=options, format_func=format_fn, key=key)
    return docs[selected_id] if selected_id else None


def show_operation_error(operation: str, exc: Exception, guidance: str | None = None) -> None:
    """Display an error category without exposing possibly sensitive exception text."""
    st.error(f"{operation} failed ({type(exc).__name__}).")
    if guidance:
        st.info(guidance)


def groundedness_badge(entailment_score: float, is_grounded: bool) -> None:
    """Render a compact, high-contrast groundedness status badge."""
    color = "#18794e" if is_grounded else "#b42318"
    label = "✓ Grounded" if is_grounded else "✗ Ungrounded"
    st.markdown(
        f'<span role="status" aria-label="{label}, entailment {entailment_score:.0%}" '
        f'style="background:{color};color:#fff;padding:2px 8px;'
        f'border-radius:4px;font-size:.85em">{label} '
        f'({entailment_score:.0%})</span>',
        unsafe_allow_html=True,
    )


def risk_score_gauge(score: int) -> None:
    """Render the risk score with a compact, accessible progress indicator."""
    if score >= 60:
        color = "#ff7b72"
        label = "High Risk"
    elif score >= 30:
        color = "#f2c66d"
        label = "Medium Risk"
    else:
        color = "#66d19e"
        label = "Low Risk"

    normalized_score = max(0, min(score, 100))
    st.markdown(
        f'<div style="margin:8px 0">'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:4px">'
        f'<strong>Risk Score: {score}/100</strong>'
        f'<span style="color:{color};font-weight:600">{label}</span></div>'
        f'<div role="progressbar" aria-label="{label}" aria-valuemin="0" '
        f'aria-valuemax="100" aria-valuenow="{normalized_score}" '
        f'style="background:rgba(255,255,255,.14);border-radius:4px;height:12px">'
        f'<div style="background:{color};width:{normalized_score}%;height:12px;'
        f'border-radius:4px"></div></div></div>',
        unsafe_allow_html=True,
    )


def severity_badge(severity: str) -> str:
    return {"high": "🔴", "medium": "🟠", "low": "🟡"}.get(severity, "⚪")


def citation_tag(contract_name: str, page: int | None) -> str:
    page_str = f" p.{page}" if page else ""
    return f"*({contract_name}{page_str})*"
