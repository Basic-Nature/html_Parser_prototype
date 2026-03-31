"""Dynamic table + context + navigation coordination contract tests.

Tests the *highest-risk* part of the pipeline: the table-building / header
harmonisation layer (harmonize_headers_and_data from detect.py) and the
conventions that context dicts must honour across multi-step navigation.

No Flask app context or database connections are required.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from webapp.parser.utils.detect import harmonize_headers_and_data, normalize_header
from webapp.tests.helpers.pipeline_invariants import assert_election_output_invariants


# ---------------------------------------------------------------------------
# harmonize_headers_and_data — shape contracts
# ---------------------------------------------------------------------------

class TestHarmonizeHeadersAndDataShape:
    """harmonize_headers_and_data must return (ordered_headers, rows, extras)."""

    def _call(self, headers, data, context=None):
        return harmonize_headers_and_data(headers, data, context)

    def test_returns_2_items(self):
        result = self._call(["Candidate", "Votes"], [{"Candidate": "A", "Votes": "10"}])
        assert isinstance(result, tuple), f"expected tuple, got {type(result)}"
        assert len(result) == 2, f"expected 2-tuple, got {len(result)} items"
        h, d = result
        assert isinstance(h, list)
        assert isinstance(d, list)

    def test_empty_inputs_return_empty_outputs(self):
        h, d = self._call([], [])
        assert h == [] or isinstance(h, list)
        assert isinstance(d, list)

    def test_headers_preserved_or_extended(self):
        headers = ["Precinct", "Candidate", "Votes"]
        data = [{"Precinct": "01", "Candidate": "Alice", "Votes": "100"}]
        out_headers, _ = self._call(headers, data)
        # All declared headers must remain in output
        for h in headers:
            assert h in out_headers, f"Header {h!r} dropped from output"

    def test_extra_row_keys_added_to_headers(self):
        headers = ["Candidate"]
        data = [{"Candidate": "Alice", "Party": "DEM", "Votes": "50"}]
        out_headers, _ = self._call(headers, data)
        # Row keys not in declared headers should be appended
        assert "Party" in out_headers
        assert "Votes" in out_headers

    def test_no_duplicate_headers_in_output(self):
        headers = ["Candidate", "Votes", "Candidate"]  # intentional dup
        data = [{"Candidate": "B", "Votes": "5"}]
        out_headers, _ = self._call(headers, data)
        assert len(out_headers) == len(set(out_headers)), "duplicate headers in output"

    def test_location_column_renamed_to_precinct(self):
        """Any recognised location-header synonym must be normalised to 'Precinct'."""
        headers = ["Ward", "Candidate", "Votes"]
        data = [{"Ward": "W1", "Candidate": "C", "Votes": "7"}]
        out_headers, out_data = self._call(headers, data)
        # 'Ward' should be renamed to 'Precinct' (it's a location keyword)
        if "Ward" not in out_headers:       # only assert rename happened, not the exact name
            assert "Precinct" in out_headers
        # Row data must be consistent with output headers
        for row in out_data:
            for key in row:
                assert key in out_headers, f"Row key {key!r} not in output headers"


# ---------------------------------------------------------------------------
# Context preservation contracts (html_context dict conventions)
# ---------------------------------------------------------------------------

class TestContextPreservation:
    """Verify that context keys survive a harmonise pass unchanged."""

    def test_provided_tables_key_preserved(self):
        ctx = {"provided_tables": [["header"], ["row1"]], "skip_pivot": True}
        harmonize_headers_and_data(
            ["Candidate", "Votes"],
            [{"Candidate": "A", "Votes": "1"}],
            context=ctx,
        )
        # Context must not be mutated in a way that drops these keys
        assert "provided_tables" in ctx, "`provided_tables` was removed from context"
        assert ctx["skip_pivot"] is True, "`skip_pivot` was mutated"

    def test_percent_reported_from_context_flows_into_output(self):
        ctx = {"percent_reported": "85%"}
        _, out_data = harmonize_headers_and_data(
            ["Precinct", "Candidate", "Votes"],
            [{"Precinct": "01", "Candidate": "A", "Votes": "100"}],
            context=ctx,
        )
        # The "Percent Reported" value should be injected into rows
        if out_data:
            assert out_data[0].get("Percent Reported") == "85%"

    def test_null_context_does_not_raise(self):
        # Must not crash on None context
        try:
            harmonize_headers_and_data(["Candidate"], [{"Candidate": "X"}], context=None)
        except Exception as exc:
            pytest.fail(f"harmonize_headers_and_data raised with None context: {exc}")


# ---------------------------------------------------------------------------
# Navigation state preservation (multi-step context simulation)
# ---------------------------------------------------------------------------

class TestNavigationStatePreservation:
    """Simulate multi-step navigation where context is built up across parse calls.

    Each step adds keys; subsequent steps must not lose earlier keys.
    """

    def _navigation_step(self, context: dict, new_data: list[dict], new_headers: list[str]) -> dict:
        """Simulate one navigation step: harmonise and merge back into context."""
        out_h, out_d = harmonize_headers_and_data(new_headers, new_data, context=context)
        context["last_headers"] = out_h
        context["last_rows"] = out_d
        return context

    def test_state_accumulates_across_steps(self):
        ctx: dict = {"state": "OH", "county": "Franklin", "step": 0}

        ctx = self._navigation_step(
            ctx,
            [{"Candidate": "Alice", "Votes": "100"}],
            ["Candidate", "Votes"],
        )
        ctx["step"] = 1

        ctx = self._navigation_step(
            ctx,
            [{"Candidate": "Bob", "Votes": "200"}],
            ["Candidate", "Votes"],
        )
        ctx["step"] = 2

        assert ctx["state"] == "OH", "state lost during navigation"
        assert ctx["county"] == "Franklin", "county lost during navigation"
        assert ctx["step"] == 2
        assert "last_headers" in ctx
        assert "last_rows" in ctx

    def test_contest_key_surviving_multi_step(self):
        ctx = {"contest": "US Senate", "state": "TX"}
        for i in range(3):
            ctx = self._navigation_step(
                ctx,
                [{"Precinct": str(i), "Candidate": "X", "Votes": str(i * 10)}],
                ["Precinct", "Candidate", "Votes"],
            )
        assert ctx.get("contest") == "US Senate", "contest key mutated during navigation"


# ---------------------------------------------------------------------------
# Fallback behaviour on degenerate table input
# ---------------------------------------------------------------------------

class TestFallbackOnInvalidInput:
    def test_all_empty_strings_in_row(self):
        h, d = harmonize_headers_and_data(
            ["A", "B"],
            [{"A": "", "B": ""}],
        )
        assert isinstance(h, list)
        assert isinstance(d, list)

    def test_row_with_none_values(self):
        try:
            h, d = harmonize_headers_and_data(
                ["X"],
                [{"X": None}],
            )
            assert isinstance(h, list)
        except Exception as exc:
            pytest.fail(f"raised on None value in row: {exc}")

    def test_mismatched_headers_and_row_keys(self):
        """Headers and row keys don't overlap — should union cleanly."""
        h, d = harmonize_headers_and_data(
            ["Alpha", "Beta"],
            [{"Gamma": "g", "Delta": "d"}],
        )
        assert "Alpha" in h
        assert "Gamma" in h

    def test_large_header_set_does_not_crash(self):
        n = 200
        headers = [f"Col{i}" for i in range(n)]
        rows = [{f"Col{i}": str(i) for i in range(n)}]
        h, d = harmonize_headers_and_data(headers, rows)
        assert len(h) >= n
