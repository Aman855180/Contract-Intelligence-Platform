"""
state.py

Session-state helpers for the Streamlit frontend. All writes to
st.session_state go through these helpers so the key names are defined
in one place and can't drift across pages.
"""

from __future__ import annotations

import streamlit as st

from app.models.document import DocumentMetadata


def init_session_state() -> None:
    """Initialise all required session-state keys on first run."""
    defaults: dict = {
        # Maps document_id -> DocumentMetadata for every contract ingested
        # during this session.
        "indexed_documents": {},
        # Cache of {document_id: ContractExtractionResult} to avoid re-running
        # extraction for comparison / risk / report within the same session.
        "extraction_cache": {},
        # Last Q&A result so it persists when the user navigates away.
        "last_qa_result": None,
        # Last comparison result.
        "last_comparison_result": None,
        # Last risk assessment result.
        "last_risk_result": None,
        # Last report result (ReportResult).
        "last_report_result": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def add_document(metadata: DocumentMetadata) -> None:
    st.session_state.indexed_documents[metadata.document_id] = metadata


def remove_document(document_id: str) -> None:
    st.session_state.indexed_documents.pop(document_id, None)
    st.session_state.extraction_cache.pop(document_id, None)


def get_documents() -> dict[str, DocumentMetadata]:
    return st.session_state.indexed_documents


def document_count() -> int:
    return len(st.session_state.indexed_documents)
