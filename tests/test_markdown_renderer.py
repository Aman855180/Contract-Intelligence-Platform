"""
test_markdown_renderer.py

Unit tests for app/reports/markdown_renderer.py.
Pure function — no I/O, no stubs needed.
"""

from __future__ import annotations

import pytest

from app.models.report import (
    ImportantDate,
    KeyClause,
    KeyParty,
    PaymentTermsSummary,
    ReportRiskFlag,
    ReportSchema,
)
from app.reports.markdown_renderer import render_markdown


def _minimal_report(document_id: str = "doc-1", contract_name: str = "Test Contract") -> ReportSchema:
    return ReportSchema(
        document_id=document_id,
        contract_name=contract_name,
        overview="This is a services agreement between two companies.",
        key_parties=[
            KeyParty(name="Acme Corp", role="Client", source_contract=contract_name, source_page=1)
        ],
        important_dates=[
            ImportantDate(label="Effective Date", date="January 1, 2025",
                          source_contract=contract_name, source_page=1)
        ],
        payment_terms=PaymentTermsSummary(
            summary="Net 30 days from invoice date.",
            source_contract=contract_name, source_page=2,
        ),
        risk_flags=[
            ReportRiskFlag(category="missing_signature_field", severity="high",
                           explanation="No signature block detected.",
                           source_contract=contract_name, source_page=1),
        ],
        key_clauses=[
            KeyClause(clause_type="termination", summary="60 days written notice required.",
                      source_contract=contract_name, source_page=3),
        ],
        recommendations=["Obtain signatures from both parties.", "Clarify payment schedule."],
    )


class TestRenderMarkdown:
    def test_output_is_a_non_empty_string(self) -> None:
        md = render_markdown(_minimal_report())
        assert isinstance(md, str)
        assert len(md) > 100

    def test_contract_name_in_header(self) -> None:
        md = render_markdown(_minimal_report(contract_name="Acme Services Agreement"))
        assert "# Executive Summary: Acme Services Agreement" in md

    def test_overview_section_present(self) -> None:
        md = render_markdown(_minimal_report())
        assert "## Overview" in md
        assert "services agreement between two companies" in md

    def test_key_party_name_and_role_present(self) -> None:
        md = render_markdown(_minimal_report())
        assert "Acme Corp" in md
        assert "Client" in md

    def test_citation_format_present(self) -> None:
        md = render_markdown(_minimal_report())
        assert "(Test Contract, p. 1)" in md

    def test_important_dates_section(self) -> None:
        md = render_markdown(_minimal_report())
        assert "## Important Dates" in md
        assert "Effective Date" in md
        assert "January 1, 2025" in md

    def test_payment_terms_section(self) -> None:
        md = render_markdown(_minimal_report())
        assert "## Payment Terms" in md
        assert "Net 30 days from invoice date." in md

    def test_risk_flag_severity_emoji_present(self) -> None:
        md = render_markdown(_minimal_report())
        assert "🔴" in md  # high severity flag

    def test_risk_flag_category_humanized(self) -> None:
        md = render_markdown(_minimal_report())
        assert "Missing Signature Field" in md

    def test_key_clause_present(self) -> None:
        md = render_markdown(_minimal_report())
        assert "## Key Clauses" in md
        assert "Termination" in md
        assert "60 days written notice required." in md

    def test_recommendations_present(self) -> None:
        md = render_markdown(_minimal_report())
        assert "## Recommendations" in md
        assert "Obtain signatures from both parties." in md
        assert "Clarify payment schedule." in md

    def test_empty_parties_shows_placeholder(self) -> None:
        report = _minimal_report()
        report.key_parties = []
        md = render_markdown(report)
        assert "No parties identified." in md

    def test_empty_dates_shows_placeholder(self) -> None:
        report = _minimal_report()
        report.important_dates = []
        md = render_markdown(report)
        assert "No key dates identified." in md

    def test_empty_risk_flags_shows_placeholder(self) -> None:
        report = _minimal_report()
        report.risk_flags = []
        md = render_markdown(report)
        assert "No risk flags triggered." in md

    def test_medium_risk_flag_uses_orange_emoji(self) -> None:
        report = _minimal_report()
        report.risk_flags = [
            ReportRiskFlag(category="high_liability_exposure", severity="medium",
                           explanation="Liability exposure detected.",
                           source_contract="Test Contract", source_page=4),
        ]
        md = render_markdown(report)
        assert "🟠" in md

    def test_all_section_headers_present(self) -> None:
        md = render_markdown(_minimal_report())
        for header in [
            "## Overview", "## Key Parties", "## Important Dates",
            "## Payment Terms", "## Risk Flags", "## Key Clauses",
            "## Recommendations",
        ]:
            assert header in md
