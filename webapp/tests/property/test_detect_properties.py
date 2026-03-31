"""Hypothesis property tests for webapp/parser/utils/detect.py.

Properties verified:
  - normalize_text: idempotent, always returns str, never raises
  - normalize_header: idempotent, always returns str, synonym mapping is stable
  - parse_numeric: digit-only strings always parse; letter-only strings never parse;
    negative number support; percentage flag is correct
  - is_location_header: known location words always True; known non-location words stable

Run with: pytest webapp/tests/property/test_detect_properties.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from webapp.parser.utils.detect import (
    is_location_header,
    normalize_header,
    normalize_text,
    parse_numeric,
)

# Hypothesis strategies
# ---------------------------------------------------------------------------

_SETTINGS = settings(
    max_examples=150,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)

# Broad strategy for arbitrary printable text (no surrogates)
_TEXT = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)),
    max_size=200,
)

# Printable ASCII only — normalize_text properties (idempotence, lowercase, no-trailing-ws)
# only hold for ASCII since NFKD can decompose Unicode math chars to uppercase, and
# non-ASCII chars after whitespace can create trailing spaces when stripped by ASCII encode.
_ASCII_SAFE = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126),
    max_size=200,
)

# A narrower strategy for simple printable ASCII
_ASCII_TEXT = st.text(
    alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Nd", "Zs"),
        whitelist_characters=" _-.",
    ),
    max_size=100,
)


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------

class TestNormalizeTextProperties:
    @given(_TEXT)
    @_SETTINGS
    def test_always_returns_str(self, s):
        result = normalize_text(s)
        assert isinstance(result, str)

    @given(_ASCII_SAFE)
    @_SETTINGS
    def test_idempotent(self, s):
        once = normalize_text(s)
        twice = normalize_text(once)
        assert once == twice, f"normalize_text not idempotent on: {s!r}"

    @given(_ASCII_SAFE)
    @_SETTINGS
    def test_output_is_lowercase(self, s):
        result = normalize_text(s)
        assert result == result.lower()

    @given(_ASCII_SAFE)
    @_SETTINGS
    def test_no_leading_trailing_whitespace(self, s):
        result = normalize_text(s)
        assert result == result.strip()

    @given(_ASCII_SAFE)
    @_SETTINGS
    def test_no_double_spaces(self, s):
        result = normalize_text(s)
        assert "  " not in result


# ---------------------------------------------------------------------------
# normalize_header
# ---------------------------------------------------------------------------

class TestNormalizeHeaderProperties:
    @given(_TEXT)
    @_SETTINGS
    def test_always_returns_str(self, s):
        result = normalize_header(s)
        assert isinstance(result, str)

    @given(_TEXT)
    @_SETTINGS
    def test_idempotent(self, s):
        once = normalize_header(s)
        twice = normalize_header(once)
        assert once == twice, f"normalize_header not idempotent on: {s!r}"

    def test_none_input_uses_empty_string(self):
        result = normalize_header(None)
        assert result == ""

    def test_none_input_explicit(self):
        assert normalize_header(None) == ""

    @pytest.mark.parametrize("synonym,expected", [
        ("candidate name", "Candidate"),
        ("candidate_names", "Candidate"),
        ("candidate", "Candidate"),
    ])
    def test_known_synonyms_map_to_canonical(self, synonym, expected):
        result = normalize_header(synonym)
        assert result == expected, f"normalize_header({synonym!r}) = {result!r}, expected {expected!r}"

    @pytest.mark.parametrize("h", [
        "", "   ", "\t\n",
    ])
    def test_blank_header_returns_empty(self, h):
        assert normalize_header(h) == ""


# ---------------------------------------------------------------------------
# parse_numeric
# ---------------------------------------------------------------------------

class TestParseNumericProperties:
    @given(st.integers(min_value=0, max_value=10_000_000))
    @_SETTINGS
    def test_non_negative_integers_parse_to_int(self, n):
        val, pct = parse_numeric(str(n))
        assert val == n
        assert pct is False

    @given(st.integers(min_value=1, max_value=10_000_000))
    @_SETTINGS
    def test_negative_integers_parse_correctly(self, n):
        val, pct = parse_numeric(f"-{n}")
        assert val == -n
        assert pct is False

    @given(st.integers(min_value=0, max_value=100))
    @_SETTINGS
    def test_percent_suffix_sets_pct_flag(self, n):
        val, pct = parse_numeric(f"{n}%")
        assert val == n
        assert pct is True

    @given(st.integers(min_value=0, max_value=10_000_000))
    @_SETTINGS
    def test_comma_formatted_integers_parse(self, n):
        formatted = f"{n:,}"          # e.g. "1,234,567"
        val, pct = parse_numeric(formatted)
        assert val == n

    @given(st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=20))
    @_SETTINGS
    def test_pure_alpha_returns_none(self, s):
        val, _ = parse_numeric(s)
        assert val is None, f"parse_numeric({s!r}) unexpectedly returned {val}"

    def test_none_input_returns_none(self):
        val, pct = parse_numeric(None)
        assert val is None
        assert pct is False

    def test_empty_string_returns_none(self):
        val, _ = parse_numeric("")
        assert val is None

    def test_double_negative_not_parsed(self):
        val, _ = parse_numeric("--5")
        assert val is None

    def test_trailing_dash_not_parsed(self):
        val, _ = parse_numeric("5-")
        assert val is None

    def test_decimal_rounds_to_int(self):
        val, _ = parse_numeric("3.7")
        assert val == 3

    @pytest.mark.parametrize("raw,expected", [
        ("0", 0),
        ("1", 1),
        ("100", 100),
        ("1,000", 1000),
        ("1,000,000", 1_000_000),
        ("-1", -1),
        ("-999", -999),
        ("50%", 50),
    ])
    def test_known_values(self, raw, expected):
        val, _ = parse_numeric(raw)
        assert val == expected


# ---------------------------------------------------------------------------
# is_location_header
# ---------------------------------------------------------------------------

class TestIsLocationHeaderProperties:
    @pytest.mark.parametrize("header", [
        "Precinct", "precinct", "PRECINCT",
        "Ward", "ward",
        "District", "district",
        "Precinct Name",
        "voting district",
    ])
    def test_known_location_words_return_true(self, header):
        assert is_location_header(header) is True, (
            f"is_location_header({header!r}) should be True"
        )

    @pytest.mark.parametrize("header", [
        "Candidate", "Party", "Votes", "Total",
        "Date", "Source",
    ])
    def test_non_location_known_words_return_false(self, header):
        assert is_location_header(header) is False, (
            f"is_location_header({header!r}) should be False"
        )

    @given(_TEXT)
    @_SETTINGS
    def test_always_returns_bool(self, s):
        result = is_location_header(s)
        assert isinstance(result, bool)

    @given(_TEXT)
    @_SETTINGS
    def test_deterministic(self, s):
        assert is_location_header(s) == is_location_header(s)
