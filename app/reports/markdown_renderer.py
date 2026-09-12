"""
markdown_renderer.py

Converts a validated ReportSchema into a Markdown string for display or
export. This is a pure function with no I/O — it only transforms the
already-validated JSON representation into human-readable Markdown.

Per project spec: the LLM never generates Markdown directly. The flow is
always: LLM → JSON → ReportSchema validation → this renderer → Markdown.
Keeping rendering separate means any future output format (HTML, PDF,
DOCX) can be added by writing a new renderer against the same ReportSchema
without touching LLM or validation logic.
"""

from __future__ import annotations

from app.models.report import ReportSchema

_SEVERITY_EMOJI = {"low": "🟡", "medium": "🟠", "high": "🔴"}


def _citation(source_contract: str, source_page: int) -> str:
    """Return a compact inline citation string."""
    return f"*({source_contract}, p. {source_page})*"


def render_markdown(report: ReportSchema) -> str:
    """
    Render a validated ReportSchema as a Markdown string.

    Args:
        report: A fully validated ReportSchema instance.

    Returns:
        A multi-section Markdown string suitable for display in Streamlit
        (st.markdown) or export to a .md file.
    """
    sections: list[str] = []

    # ── Header ──────────────────────────────────────────────────────────────
    sections.append(f"# Executive Summary: {report.contract_name}\n")

    # ── Overview ────────────────────────────────────────────────────────────
    sections.append("## Overview\n")
    sections.append(report.overview + "\n")

    # ── Key Parties ─────────────────────────────────────────────────────────
    sections.append("## Key Parties\n")
    if report.key_parties:
        for party in report.key_parties:
            cite = _citation(party.source_contract, party.source_page)
            sections.append(f"- **{party.name}** — {party.role} {cite}")
        sections.append("")
    else:
        sections.append("*No parties identified.*\n")

    # ── Important Dates ─────────────────────────────────────────────────────
    sections.append("## Important Dates\n")
    if report.important_dates:
        for date in report.important_dates:
            cite = _citation(date.source_contract, date.source_page)
            sections.append(f"- **{date.label}:** {date.date} {cite}")
        sections.append("")
    else:
        sections.append("*No key dates identified.*\n")

    # ── Payment Terms ────────────────────────────────────────────────────────
    sections.append("## Payment Terms\n")
    pt = report.payment_terms
    cite = _citation(pt.source_contract, pt.source_page)
    sections.append(f"{pt.summary} {cite}\n")

    # ── Risk Flags ───────────────────────────────────────────────────────────
    sections.append("## Risk Flags\n")
    if report.risk_flags:
        for flag in report.risk_flags:
            emoji = _SEVERITY_EMOJI.get(flag.severity, "⚪")
            cite = _citation(flag.source_contract, flag.source_page)
            category_label = flag.category.replace("_", " ").title()
            sections.append(f"### {emoji} {category_label} ({flag.severity.capitalize()} Risk)\n")
            sections.append(f"{flag.explanation} {cite}\n")
    else:
        sections.append("*No risk flags triggered.*\n")

    # ── Key Clauses ──────────────────────────────────────────────────────────
    sections.append("## Key Clauses\n")
    if report.key_clauses:
        for clause in report.key_clauses:
            cite = _citation(clause.source_contract, clause.source_page)
            clause_label = clause.clause_type.replace("_", " ").title()
            sections.append(f"### {clause_label}\n")
            sections.append(f"{clause.summary} {cite}\n")
    else:
        sections.append("*No key clauses identified.*\n")

    # ── Recommendations ──────────────────────────────────────────────────────
    sections.append("## Recommendations\n")
    if report.recommendations:
        for rec in report.recommendations:
            sections.append(f"- {rec}")
        sections.append("")
    else:
        sections.append("*No recommendations generated.*\n")

    return "\n".join(sections)
