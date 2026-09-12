"""
qa_page.py

Grounded Contract Q&A page. Every answer includes a citation and an
NLI-based groundedness score. If no supporting evidence is found above the
similarity threshold, the system says so rather than hallucinating.
"""

from __future__ import annotations

import streamlit as st

from app.frontend.components import citation_tag, groundedness_badge
from app.frontend.state import get_documents
from app.llm.qa_service import answer_question


def render() -> None:
    st.header("💬 Contract Q&A")
    st.write(
        "Ask any question about your indexed contracts. "
        "Every answer is grounded in a retrieved passage and verified with an "
        "independent NLI entailment check — the system will not guess."
    )

    if not get_documents():
        st.info("No contracts indexed yet. Go to **Upload** to add contracts.")
        return

    # ── Question input ───────────────────────────────────────────────────────
    question = st.text_input(
        "Your question",
        placeholder="e.g. What is the payment term? / When does this agreement expire?",
    )

    if not st.button("Ask", type="primary") or not question.strip():
        # Show last result if available.
        _render_result(st.session_state.get("last_qa_result"))
        return

    with st.spinner("Retrieving relevant passages and generating answer…"):
        try:
            result = answer_question(question.strip())
            st.session_state["last_qa_result"] = result
        except Exception as exc:
            st.error(f"Q&A failed: {exc}")
            st.info(
                "**Tip:** Q&A requires Ollama running locally with `llama3.1:8b` loaded. "
                "Start it with: `ollama serve` and `ollama pull llama3.1:8b`"
            )
            return

    _render_result(st.session_state.get("last_qa_result"))


def _render_result(result) -> None:
    if result is None:
        return

    st.divider()

    if not result.found_in_contract:
        st.warning(
            "🔍 **No supporting evidence found.** "
            "The retrieved passages did not contain enough information to answer "
            "this question above the similarity threshold. Try rephrasing or "
            "check that the relevant contract has been uploaded."
        )
        return

    # ── Answer ───────────────────────────────────────────────────────────────
    st.subheader("Answer")
    st.write(result.answer)

    # ── Citation ─────────────────────────────────────────────────────────────
    col_cite, col_ground = st.columns([2, 1])
    with col_cite:
        st.caption(
            f"**Source:** {citation_tag(result.source_contract or '', result.source_page)}"
        )
    with col_ground:
        if result.groundedness:
            groundedness_badge(
                result.groundedness.entailment_score,
                result.groundedness.is_grounded,
            )

    # ── Supporting text ──────────────────────────────────────────────────────
    if result.supporting_text:
        with st.expander("Supporting passage from contract"):
            st.markdown(f"> {result.supporting_text}")

    # ── Groundedness detail ──────────────────────────────────────────────────
    if result.groundedness:
        g = result.groundedness
        with st.expander("Groundedness details (NLI scores)"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Entailment", f"{g.entailment_score:.1%}")
            c2.metric("Neutral", f"{g.neutral_score:.1%}")
            c3.metric("Contradiction", f"{g.contradiction_score:.1%}")
            st.caption(
                "Scores from `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` — "
                "not an LLM self-reported confidence."
            )
