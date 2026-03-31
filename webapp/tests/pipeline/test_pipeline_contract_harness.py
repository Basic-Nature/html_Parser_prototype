"""Pipeline contract harness.

Tests the *shape contract* that every handler must fulfil:
  - Returns a canonical 4-tuple (headers, rows, contest, metadata)
  - metadata never contains an 'error' key on success
  - rows are dicts; headers are strings

Uses lightweight synthetic handlers so no Flask application context is needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Project root on sys.path (mirrors conftest.py setup)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from webapp.parser.utils.shared_logic import safe_parse, validate_handler_result
from webapp.tests.helpers.pipeline_invariants import (
    assert_election_output_invariants,
    assert_no_duplicate_keys,
)


# ---------------------------------------------------------------------------
# Synthetic handler fixtures
# ---------------------------------------------------------------------------

class _GoodHandler:
    """Minimal well-behaved parse handler."""

    def parse(self, page=None, html_context=None, **_kwargs):
        headers = ["Candidate", "Party", "Votes"]
        rows = [
            {"Candidate": "Alice Smith", "Party": "DEM", "Votes": "1234"},
            {"Candidate": "Bob Jones", "Party": "REP", "Votes": "987"},
        ]
        contest = "US House District 1"
        metadata = {"state": "NY", "county": "Albany", "source": "test"}
        return headers, rows, contest, metadata


class _MultiTableHandler:
    """Handler returning more rows + a distractor column."""

    def parse(self, page=None, html_context=None, **_kwargs):
        headers = ["Precinct", "Candidate", "Party", "Votes", "Percent Reported"]
        rows = [
            {"Precinct": "01", "Candidate": "Alice", "Party": "DEM", "Votes": "500", "Percent Reported": "100%"},
            {"Precinct": "01", "Candidate": "Bob", "Party": "REP", "Votes": "300", "Percent Reported": "100%"},
            {"Precinct": "02", "Candidate": "Alice", "Party": "DEM", "Votes": "700", "Percent Reported": "100%"},
            {"Precinct": "02", "Candidate": "Bob", "Party": "REP", "Votes": "400", "Percent Reported": "100%"},
        ]
        contest = "Governor"
        metadata = {"state": "CA", "county": "Los Angeles", "table_count": 2}
        return headers, rows, contest, metadata


class _MetadataPropagationHandler:
    """Handler that carries html_context metadata through to output."""

    def parse(self, page=None, html_context=None, **_kwargs):
        ctx = html_context or {}
        meta = {
            "state": ctx.get("state", ""),
            "county": ctx.get("county", ""),
            "source_url": ctx.get("source_url", ""),
        }
        return ["Candidate", "Votes"], [{"Candidate": "Eve", "Votes": "42"}], "Prop 99", meta


class _ErrorFallbackHandler:
    """Handler that raises — safe_parse must return an error metadata dict."""

    def parse(self, page=None, **_kwargs):
        raise ValueError("simulated parse failure")


# ---------------------------------------------------------------------------
# validate_handler_result unit tests
# ---------------------------------------------------------------------------

class TestValidateHandlerResult:
    def test_canonical_4tuple_passthrough(self):
        result = (["H1", "H2"], [{"H1": "a", "H2": "b"}], "Contest", {"state": "NY"})
        h, r, c, m = validate_handler_result(result)
        assert h == ["H1", "H2"]
        assert r == [{"H1": "a", "H2": "b"}]
        assert c == "Contest"
        assert m == {"state": "NY"}

    def test_3tuple_gets_empty_metadata(self):
        result = (["H1"], [{"H1": "x"}], "My Contest")
        h, r, c, m = validate_handler_result(result)
        assert c == "My Contest"
        assert m == {}

    def test_list_only_result(self):
        rows = [{"A": "1"}, {"A": "2"}]
        h, r, c, m = validate_handler_result(rows)
        assert h == []
        assert r == rows
        assert c == ""
        assert "error" not in m

    def test_none_result_gives_error_metadata(self):
        h, r, c, m = validate_handler_result(None)
        assert h == []
        assert r == []
        assert "error" in m

    def test_empty_tuple_gives_error_metadata(self):
        h, r, c, m = validate_handler_result(())
        assert "error" in m


# ---------------------------------------------------------------------------
# safe_parse contract tests
# ---------------------------------------------------------------------------

class TestSafeParseContract:
    def test_good_handler_returns_4tuple(self):
        result = safe_parse(_GoodHandler())
        assert_election_output_invariants(result, label="good_handler")

    def test_good_handler_headers_non_empty(self):
        headers, rows, contest, metadata = safe_parse(_GoodHandler())
        assert headers, "headers must not be empty for a well-formed handler"
        assert len(rows) >= 1

    def test_good_handler_no_error_in_metadata(self):
        _, _, _, metadata = safe_parse(_GoodHandler())
        assert "error" not in metadata

    def test_erroring_handler_returns_error_metadata(self):
        headers, rows, contest, metadata = safe_parse(_ErrorFallbackHandler())
        assert headers == []
        assert rows == []
        assert "error" in metadata
        assert metadata["error"] == "exception"

    def test_none_handler_returns_no_handler_error(self):
        _, _, _, meta = safe_parse(None)
        assert meta.get("error") == "no_handler"

    def test_callable_function_handler(self):
        def _fn(page=None, **kw):
            return ["C", "V"], [{"C": "X", "V": "10"}], "Race", {}

        result = safe_parse(_fn)
        assert_election_output_invariants(result, label="fn_handler")

    def test_html_context_is_forwarded(self):
        ctx = {"state": "TX", "county": "Travis", "source_url": "https://example.gov"}
        _, _, contest, meta = safe_parse(
            _MetadataPropagationHandler(),
            html_context=ctx,
        )
        assert meta.get("state") == "TX"
        assert meta.get("county") == "Travis"
        assert contest == "Prop 99"


# ---------------------------------------------------------------------------
# Multi-table + distractor column tests
# ---------------------------------------------------------------------------

class TestMultiTableContract:
    def test_multi_table_output_shape(self):
        result = safe_parse(_MultiTableHandler())
        assert_election_output_invariants(result, label="multi_table")

    def test_no_duplicate_keys_per_precinct_candidate(self):
        _, rows, _, _ = safe_parse(_MultiTableHandler())
        assert_no_duplicate_keys(
            rows,
            key_fields=("Precinct", "Candidate"),
            label="multi_table",
        )

    def test_distractor_column_present_in_headers(self):
        headers, _, _, _ = safe_parse(_MultiTableHandler())
        assert "Percent Reported" in headers


# ---------------------------------------------------------------------------
# Metadata propagation contract
# ---------------------------------------------------------------------------

class TestMetadataPropagation:
    def test_state_and_county_flow_through(self):
        ctx = {"state": "FL", "county": "Miami-Dade", "source_url": ""}
        _, _, _, meta = safe_parse(_MetadataPropagationHandler(), html_context=ctx)
        assert meta["state"] == "FL"
        assert meta["county"] == "Miami-Dade"

    def test_missing_context_yields_empty_strings(self):
        _, _, _, meta = safe_parse(_MetadataPropagationHandler())
        assert meta["state"] == ""
        assert meta["county"] == ""
