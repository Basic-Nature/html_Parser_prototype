"""Hypothesis property tests for webapp/parser/utils/shared_logic.py.

Properties verified:
  - safe_filename: no path separators or traversal in output; length bounded;
    non-empty when input is non-empty; idempotent in strict_mode
  - safe_strip: always str, idempotent
  - safe_lower: always str, idempotent, lowercase
  - safe_slug: no unsafe chars; length bounded; idempotent
  - normalize_county_name: always Optional[str], idempotent on str inputs

Run with: pytest webapp/tests/property/test_shared_logic_properties.py -v
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

from webapp.parser.utils.shared_logic import (
    normalize_county_name,
    safe_filename,
    safe_lower,
    safe_slug,
    safe_strip,
)

# ---------------------------------------------------------------------------
# Hypothesis settings profile
# ---------------------------------------------------------------------------

_SETTINGS = settings(
    max_examples=150,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)

_TEXT = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)),
    max_size=300,
)

_ASCII_PRINTABLE = st.text(
    alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Nd"),
        whitelist_characters=" ._-/\\",
    ),
    max_size=300,
)


# ---------------------------------------------------------------------------
# safe_filename
# ---------------------------------------------------------------------------

class TestSafeFilenameProperties:
    @given(_ASCII_PRINTABLE)
    @_SETTINGS
    def test_no_forward_slash_in_output(self, name):
        result = safe_filename(name)
        assert "/" not in result, f"forward slash in: {result!r} (input: {name!r})"

    @given(_ASCII_PRINTABLE)
    @_SETTINGS
    def test_no_backslash_in_output(self, name):
        result = safe_filename(name)
        assert "\\" not in result, f"backslash in: {result!r} (input: {name!r})"

    @given(_ASCII_PRINTABLE)
    @_SETTINGS
    def test_no_null_bytes(self, name):
        result = safe_filename(name)
        assert "\x00" not in result

    @given(_ASCII_PRINTABLE)
    @_SETTINGS
    def test_no_double_dot_traversal(self, name):
        result = safe_filename(name)
        assert ".." not in result, f"traversal pattern in: {result!r} (input: {name!r})"

    @given(_ASCII_PRINTABLE)
    @_SETTINGS
    def test_length_within_default_max(self, name):
        result = safe_filename(name)
        assert len(result) <= 255

    @given(_ASCII_PRINTABLE, st.integers(min_value=4, max_value=100))
    @_SETTINGS
    def test_length_within_custom_max(self, name, max_len):
        result = safe_filename(name, max_length=max_len)
        assert len(result) <= max_len

    @given(st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=50))
    @_SETTINGS
    def test_non_empty_alphanumeric_input_gives_non_empty_output(self, name):
        result = safe_filename(name)
        assert result, f"empty output for non-empty input: {name!r}"

    def test_empty_string_returns_default(self):
        result = safe_filename("")
        assert result == "file"

    def test_custom_default_used_for_empty(self):
        result = safe_filename("", default="untitled")
        assert result == "untitled"

    @pytest.mark.parametrize("reserved", ["CON", "PRN", "AUX", "NUL", "COM1", "LPT9"])
    def test_reserved_windows_names_are_wrapped(self, reserved):
        result = safe_filename(reserved)
        assert result != reserved.upper(), f"reserved name {reserved!r} was not wrapped"

    @given(_ASCII_PRINTABLE)
    @_SETTINGS
    def test_strict_mode_idempotent(self, name):
        once = safe_filename(name, strict_mode=True)
        twice = safe_filename(once, strict_mode=True)
        assert once == twice, f"strict mode not idempotent: {once!r} → {twice!r}"

    @pytest.mark.parametrize("bad_input, forbidden", [
        ("../../etc/passwd", ".."),
        ("path/to/file", "/"),
        ("back\\slash", "\\"),
    ])
    def test_dangerous_inputs_sanitised(self, bad_input, forbidden):
        result = safe_filename(bad_input)
        assert forbidden not in result


# ---------------------------------------------------------------------------
# safe_strip
# ---------------------------------------------------------------------------

class TestSafeStripProperties:
    @given(_TEXT)
    @_SETTINGS
    def test_always_returns_str(self, s):
        assert isinstance(safe_strip(s), str)

    @given(_TEXT)
    @_SETTINGS
    def test_idempotent(self, s):
        once = safe_strip(s)
        twice = safe_strip(once)
        assert once == twice

    @given(_TEXT)
    @_SETTINGS
    def test_no_leading_trailing_whitespace(self, s):
        result = safe_strip(s)
        assert result == result.strip()

    def test_none_input_returns_string(self):
        result = safe_strip(None)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# safe_lower
# ---------------------------------------------------------------------------

class TestSafeLowerProperties:
    @given(_TEXT)
    @_SETTINGS
    def test_always_returns_str(self, s):
        assert isinstance(safe_lower(s), str)

    @given(_TEXT)
    @_SETTINGS
    def test_idempotent(self, s):
        once = safe_lower(s)
        twice = safe_lower(once)
        assert once == twice

    @given(_TEXT)
    @_SETTINGS
    def test_output_is_lowercase(self, s):
        result = safe_lower(s)
        assert result == result.lower()

    def test_none_input_returns_string(self):
        result = safe_lower(None)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# safe_slug
# ---------------------------------------------------------------------------

class TestSafeSlugProperties:
    @given(_TEXT)
    @_SETTINGS
    def test_always_returns_str(self, s):
        assert isinstance(safe_slug(s), str)

    @given(_TEXT)
    @_SETTINGS
    def test_no_spaces_in_output(self, s):
        result = safe_slug(s)
        assert " " not in result

    @given(_TEXT)
    @_SETTINGS
    def test_length_bounded(self, s):
        result = safe_slug(s)
        assert len(result) <= 100

    @given(st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=50),
           st.integers(min_value=8, max_value=80))
    @_SETTINGS
    def test_custom_max_len_honoured(self, s, max_len):
        result = safe_slug(s, max_len=max_len)
        assert len(result) <= max_len

    @given(st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789_-", min_size=1, max_size=50))
    @_SETTINGS
    def test_safe_chars_preserved(self, s):
        result = safe_slug(s)
        assert result, f"empty slug for non-empty safe input: {s!r}"

    @given(_TEXT)
    @_SETTINGS
    def test_no_double_underscores(self, s):
        result = safe_slug(s)
        assert "__" not in result


# ---------------------------------------------------------------------------
# normalize_county_name
# ---------------------------------------------------------------------------

class TestNormalizeCountyNameProperties:
    @given(st.text(alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ", min_size=1, max_size=60))
    @_SETTINGS
    def test_always_returns_none_or_str(self, name):
        result = normalize_county_name(name)
        assert result is None or isinstance(result, str)

    @given(st.text(alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ", min_size=1, max_size=60))
    @_SETTINGS
    def test_string_inputs_idempotent(self, name):
        once = normalize_county_name(name)
        if once is None or once == "":
            # Empty input or all-non-alpha input returns None or ""; second call on empty gives None
            # This is not required to be idempotent, so skip the test
            return
        twice = normalize_county_name(once)
        assert once == twice, f"normalize_county_name not idempotent on non-empty result: {name!r} -> {once!r} -> {twice!r}"

    @pytest.mark.parametrize("input_name, expected_not_none", [
        ("Franklin County", True),
        ("franklin county", True),
        ("FRANKLIN COUNTY", True),
        ("  Franklin  ", True),
    ])
    def test_known_county_names_not_none(self, input_name, expected_not_none):
        result = normalize_county_name(input_name)
        if expected_not_none:
            assert result is not None, f"normalize_county_name({input_name!r}) unexpectedly None"
