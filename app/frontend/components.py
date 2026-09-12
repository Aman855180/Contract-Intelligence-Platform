"""
components.py

Reusable Streamlit UI components shared across pages.
"""

from __future__ import annotations

import streamlit as st

from app.frontend.state import get_documents
from app.models.document import DocumentMetadata


def document_selector(
    label: str = "Select contract",
    key: str = "doc_selector",
    exclude_id: str | None = None,
) -> DocumentMetadata | None:
    """
    Render a selectbox for choosing one indexed contract.

    Args:
        label: Label shown above the selectbox.
        key: Streamlit widget key (must be unique per page if called multiple times).
        exclude_id: document_id to omit from the list (used by comparison page
                    so the same document can't be selected as both A and B).

    Returns:
        The selected DocumentMetadata, or None if no documents are indexed or
        the user hasn't selected one.
    """
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


def groundedness_badge(entailment_score: float, is_grounded: bool) -> None:
    """Render a coloured badge showing the NLI groundedness score."""
    color = "#28a745" if is_grounded else "#dc3545"
    label = "✓ Grounded" if is_grounded else "✗ Ungrounded"
    st.markdown(
        f'<span style="background:{color};color:white;padding:2px 8px;'
        f'border-radius:4px;font-size:0.85em">{label} '
        f'({entailment_score:.0%})</span>',
        unsafe_allow_html=True,
    )


def risk_score_gauge(score: int) -> None:
    """Render the numeric risk score with a colour-coded progress bar."""
    if score >= 60:
        color = "#dc3545"  # red
        label = "High Risk"
    elif score >= 30:
        color = "#fd7e14"  # orange
        label = "Medium Risk"
    else:
        color = "#28a745"  # green
        label = "Low Risk"

    st.markdown(
        f"""
        <div style="margin:8px 0">
          <div style="display:flex;justify-content:space-between;margin-bottom:4px">
            <span><strong>Risk Score: {score}/100</strong></span>
            <span style="color:{color};font-weight:bold">{label}</span>
          </div>
          <div style="background:#e9ecef;border-radius:4px;height:14px">
            <div style="background:{color};width:{score}%;height:14px;border-radius:4px"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def severity_badge(severity: str) -> str:
    """Return an emoji prefix for a severity level."""
    return {"high": "🔴", "medium": "🟠", "low": "🟡"}.get(severity, "⚪")


def citation_tag(contract_name: str, page: int | None) -> str:
    """Return a short citation string."""
    page_str = f" p.{page}" if page else ""
    return f"*({contract_name}{page_str})*"
