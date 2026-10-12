"""
app.py

Contract Intelligence Platform — Streamlit application entry point.

Run with:
    streamlit run app.py

This single file is the only entry point. It:
  1. Initialises the SQLite schema and session state on first run.
  2. Renders a sidebar with navigation and an indexed-contracts summary.
  3. Routes to the active page module.

All model initialisation is lazy (embedding model, NLI, NER) — the first
upload or Q&A triggers the model load; startup remains fast.
"""

from __future__ import annotations

import streamlit as st

# ── Page config must be the very first Streamlit call ────────────────────────
st.set_page_config(
    page_title="Contract Intelligence Platform",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Runtime initialisation (once per session) ────────────────────────────────
from app.database.sqlite_db import init_db  # noqa: E402
from app.frontend.components import apply_app_styles  # noqa: E402
from app.frontend.state import document_count, get_documents, init_session_state  # noqa: E402

# init_db is idempotent — safe to call on every page refresh.
init_db()
init_session_state()

# ── Page imports ─────────────────────────────────────────────────────────────
from app.frontend.pages import (  # noqa: E402
    compare_page,
    qa_page,
    report_page,
    risk_page,
    upload_page,
)

apply_app_styles()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("#### 📋 Contract Intelligence")
    st.caption("AI-powered contract analysis")

    st.divider()

    page = st.radio(
        "Navigation",
        options=[
            "📂 Upload",
            "💬 Q&A",
            "🔄 Compare",
            "⚠️ Risk",
            "📄 Report",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    # ── Indexed documents summary ─────────────────────────────────────────────
    n_docs = document_count()
    if n_docs == 0:
        st.caption("Indexed contracts: 0")
    else:
        st.success(f"**{n_docs}** contract(s) indexed")
        docs = list(get_documents().values())
        for meta in docs[:3]:
            st.caption(f"• {meta.contract_name}")
        if n_docs > 3:
            st.caption(f"And {n_docs - 3} more")

    st.divider()
    st.caption("Streamlit · Gemini / Ollama · ChromaDB · BM25")

# ── Page routing ──────────────────────────────────────────────────────────────
if page == "📂 Upload":
    upload_page.render()
elif page == "💬 Q&A":
    qa_page.render()
elif page == "🔄 Compare":
    compare_page.render()
elif page == "⚠️ Risk":
    risk_page.render()
elif page == "📄 Report":
    report_page.render()
