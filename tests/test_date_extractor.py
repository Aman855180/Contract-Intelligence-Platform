"""
test_date_extractor.py

Unit tests for app/extraction/date_extractor.py.
"""

from __future__ import annotations

from app.extraction.date_extractor import extract_dates


class TestExtractDates:
    def test_month_name_format(self) -> None:
        dates = extract_dates("This agreement is effective as of January 1, 2025.")
        assert "January 1, 2025" in dates

    def test_month_name_with_ordinal_suffix(self) -> None:
        dates = extract_dates("Signed on March 15th, 2024 by both parties.")
        assert any("March 15th, 2024" == d for d in dates)

    def test_day_before_month_format(self) -> None:
        dates = extract_dates("The agreement was executed on the 15th of March, 2024.")
        assert any("March" in d and "2024" in d for d in dates)

    def test_slash_format(self) -> None:
        dates = extract_dates("Invoice date: 01/15/2024.")
        assert "01/15/2024" in dates

    def test_iso_format(self) -> None:
        dates = extract_dates("Effective date: 2024-01-15.")
        assert "2024-01-15" in dates

    def test_multiple_dates_in_one_text(self) -> None:
        dates = extract_dates(
            "This agreement begins on January 1, 2025 and ends on December 31, 2026."
        )
        assert "January 1, 2025" in dates
        assert "December 31, 2026" in dates
        assert len(dates) == 2

    def test_no_dates_returns_empty_list(self) -> None:
        dates = extract_dates("This clause discusses payment terms with no date mentioned.")
        assert dates == []

    def test_duplicate_dates_deduplicated(self) -> None:
        dates = extract_dates("Effective January 1, 2025. Also valid from January 1, 2025 onward.")
        assert dates.count("January 1, 2025") == 1

    def test_preserves_original_casing(self) -> None:
        dates = extract_dates("Signed on january 1, 2025.")
        assert dates[0] == "january 1, 2025"

    def test_empty_text_returns_empty_list(self) -> None:
        assert extract_dates("") == []
