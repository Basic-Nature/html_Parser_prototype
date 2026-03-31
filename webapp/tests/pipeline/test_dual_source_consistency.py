"""Dual-source consistency tests.

Exercises the diff_dual_sources helper from pipeline_invariants against
various combinations of parser output vs. warehouse rows, covering:
  - exact canonical match (zero diff)
  - normalised-equivalent match (different casing/whitespace)
  - rows present in warehouse but missing from parser
  - rows present in parser but missing from warehouse
  - rows present in both but with differing vote values
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from webapp.tests.helpers.pipeline_invariants import diff_dual_sources, format_diff_report


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

def _row(contest, county, candidate, party, votes):
    return {
        "contest": contest,
        "county": county,
        "candidate": candidate,
        "party": party,
        "votes": votes,
    }


_BASE_ROWS = [
    _row("US Senate", "Franklin", "Alice Smith", "DEM", "1234"),
    _row("US Senate", "Franklin", "Bob Jones", "REP", "987"),
    _row("Governor", "Cuyahoga", "Carol Doe", "DEM", "5500"),
]


# ---------------------------------------------------------------------------
# Zero-diff cases
# ---------------------------------------------------------------------------

class TestExactMatch:
    def test_identical_rows_produce_empty_diff(self):
        diff = diff_dual_sources(_BASE_ROWS, _BASE_ROWS)
        assert diff["missing_parser"] == [], format_diff_report(diff)
        assert diff["missing_warehouse"] == [], format_diff_report(diff)
        assert diff["mismatched_values"] == [], format_diff_report(diff)

    def test_single_row_exact_match(self):
        row = [_row("Race", "County", "Candidate X", "IND", "42")]
        diff = diff_dual_sources(row, row)
        for bucket in diff.values():
            assert bucket == []

    def test_empty_both_sources_produce_empty_diff(self):
        diff = diff_dual_sources([], [])
        assert all(v == [] for v in diff.values())


# ---------------------------------------------------------------------------
# Normalised-equivalent match (casing / whitespace)
# ---------------------------------------------------------------------------

class TestNormalisedEquivalentMatch:
    def test_mixed_case_keys_treated_as_equal(self):
        parser = [_row("US Senate", "Franklin", "alice smith", "dem", "1234")]
        warehouse = [_row("US Senate", "Franklin", "Alice Smith", "DEM", "1234")]
        diff = diff_dual_sources(parser, warehouse)
        assert diff["missing_parser"] == []
        assert diff["missing_warehouse"] == []
        assert diff["mismatched_values"] == []

    def test_leading_trailing_whitespace_ignored(self):
        parser = [_row("  US Senate  ", " Franklin", "Alice Smith ", "DEM", "1234")]
        warehouse = [_row("US Senate", "Franklin", "Alice Smith", "DEM", "1234")]
        diff = diff_dual_sources(parser, warehouse)
        assert all(v == [] for v in diff.values())

    def test_votes_whitespace_ignored(self):
        parser = [_row("Race", "County", "X", "D", " 100 ")]
        warehouse = [_row("Race", "County", "X", "D", "100")]
        diff = diff_dual_sources(parser, warehouse)
        assert diff["mismatched_values"] == []


# ---------------------------------------------------------------------------
# Missing-in-parser
# ---------------------------------------------------------------------------

class TestMissingInParser:
    def test_single_row_missing_in_parser(self):
        parser = _BASE_ROWS[:2]
        warehouse = _BASE_ROWS
        diff = diff_dual_sources(parser, warehouse)
        assert len(diff["missing_parser"]) == 1
        assert diff["missing_warehouse"] == []
        assert diff["mismatched_values"] == []

    def test_all_rows_missing_from_parser(self):
        diff = diff_dual_sources([], _BASE_ROWS)
        assert len(diff["missing_parser"]) == len(_BASE_ROWS)
        assert diff["missing_warehouse"] == []

    def test_missing_parser_item_contains_warehouse_row(self):
        parser = []
        warehouse = [_row("Race", "County", "Z", "X", "99")]
        diff = diff_dual_sources(parser, warehouse)
        assert diff["missing_parser"][0]["warehouse_row"]["candidate"] == "Z"


# ---------------------------------------------------------------------------
# Missing-in-warehouse
# ---------------------------------------------------------------------------

class TestMissingInWarehouse:
    def test_single_row_missing_in_warehouse(self):
        parser = _BASE_ROWS
        warehouse = _BASE_ROWS[:2]
        diff = diff_dual_sources(parser, warehouse)
        assert diff["missing_parser"] == []
        assert len(diff["missing_warehouse"]) == 1
        assert diff["mismatched_values"] == []

    def test_all_rows_missing_from_warehouse(self):
        diff = diff_dual_sources(_BASE_ROWS, [])
        assert len(diff["missing_warehouse"]) == len(_BASE_ROWS)

    def test_missing_warehouse_item_contains_parser_row(self):
        parser = [_row("Race", "County", "Z", "X", "99")]
        diff = diff_dual_sources(parser, [])
        assert diff["missing_warehouse"][0]["parser_row"]["candidate"] == "Z"  # original case


# ---------------------------------------------------------------------------
# Mismatched values
# ---------------------------------------------------------------------------

class TestMismatchedValues:
    def test_vote_count_mismatch_flagged(self):
        parser = [_row("US Senate", "Franklin", "Alice Smith", "DEM", "1000")]
        warehouse = [_row("US Senate", "Franklin", "Alice Smith", "DEM", "1234")]
        diff = diff_dual_sources(parser, warehouse)
        assert len(diff["mismatched_values"]) == 1
        mm = diff["mismatched_values"][0]
        assert "votes" in mm["diffs"]
        assert mm["diffs"]["votes"]["parser"] == "1000"
        assert mm["diffs"]["votes"]["warehouse"] == "1234"

    def test_multiple_mismatches_all_reported(self):
        parser = [
            _row("US Senate", "Franklin", "Alice Smith", "DEM", "999"),
            _row("US Senate", "Franklin", "Bob Jones", "REP", "1"),
        ]
        warehouse = [
            _row("US Senate", "Franklin", "Alice Smith", "DEM", "1234"),
            _row("US Senate", "Franklin", "Bob Jones", "REP", "987"),
        ]
        diff = diff_dual_sources(parser, warehouse)
        assert len(diff["mismatched_values"]) == 2

    def test_exact_match_is_not_reported_as_mismatch(self):
        diff = diff_dual_sources(_BASE_ROWS, _BASE_ROWS)
        assert diff["mismatched_values"] == []


# ---------------------------------------------------------------------------
# Custom key / value fields
# ---------------------------------------------------------------------------

class TestCustomKeyFields:
    def test_custom_key_fields(self):
        parser = [{"race": "Senate", "name": "Alice", "total": "100"}]
        warehouse = [{"race": "Senate", "name": "Bob", "total": "200"}]
        diff = diff_dual_sources(parser, warehouse, key_fields=("race", "name"), value_fields=("total",))
        assert len(diff["missing_parser"]) == 1
        assert len(diff["missing_warehouse"]) == 1
        assert diff["mismatched_values"] == []

    def test_value_field_mismatch_with_custom_fields(self):
        parser = [{"race": "Mayor", "name": "X", "total": "50"}]
        warehouse = [{"race": "Mayor", "name": "X", "total": "99"}]
        diff = diff_dual_sources(parser, warehouse, key_fields=("race", "name"), value_fields=("total",))
        assert len(diff["mismatched_values"]) == 1


# ---------------------------------------------------------------------------
# format_diff_report smoke test
# ---------------------------------------------------------------------------

class TestFormatDiffReport:
    def test_report_contains_category_labels(self):
        parser = [_row("Race", "Co", "A", "D", "10")]
        warehouse = [_row("Race", "Co", "B", "R", "20")]
        diff = diff_dual_sources(parser, warehouse)
        report = format_diff_report(diff)
        assert "MISSING PARSER" in report
        assert "MISSING WAREHOUSE" in report

    def test_report_for_clean_diff_is_short(self):
        diff = diff_dual_sources(_BASE_ROWS, _BASE_ROWS)
        report = format_diff_report(diff)
        # Should still produce output (just show empty buckets)
        assert isinstance(report, str)
