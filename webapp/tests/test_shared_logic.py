# -*- coding: utf-8 -*-
"""Tests for webapp/parser/utils/shared_logic.py"""
import pytest
from webapp.parser.utils.shared_logic import (
    safe_filename,
    safe_slug,
    safe_get,
    safe_strip,
    safe_lower,
    normalize_county_name,
    normalize_state_name,
    format_county_label,
    format_state_label,
    safe_parse,
)


class TestSafeFilename:
    """Tests for safe_filename function."""
    
    def test_basic_filename(self):
        """Test basic filename sanitization."""
        assert safe_filename("test.txt") == "test.txt"
        assert safe_filename("Test File.csv") == "Test_File.csv"
    
    def test_unicode_removal(self):
        """Test unicode character handling."""
        # Test with ASCII-safe strings
        result = safe_filename("test_file.txt", allow_unicode=False)
        assert "test" in result
        
    def test_reserved_names(self):
        """Test Windows reserved device names."""
        assert safe_filename("CON") == "_CON_"
        assert safe_filename("PRN") == "_PRN_"
        assert safe_filename("AUX") == "_AUX_"
    
    def test_max_length_truncation(self):
        """Test filename length limitation."""
        long_name = "a" * 300 + ".txt"
        result = safe_filename(long_name, max_length=255)
        assert len(result) <= 255
        assert result.endswith(".txt")
    
    def test_empty_input(self):
        """Test empty or None inputs."""
        assert safe_filename("", default="file") == "file"
        assert safe_filename(None, default="file") == "file"  # type: ignore[arg-type]

    def test_path_traversal_and_separator_cleanup(self):
        """Traversal tokens and separators are stripped safely."""
        assert safe_filename("..\\..//secrets.txt") == "secrets.txt"
        assert safe_filename("report/2024\\final.csv") == "report2024final.csv"

    def test_strict_mode_preserves_extension_without_trailing_separator(self):
        """Strict mode should still preserve a clean extension."""
        assert safe_filename("summary_.csv", strict_mode=True) == "summary_csv"


class TestSafeSlug:
    """Tests for safe_slug function."""
    
    def test_basic_slug(self):
        """Test basic slug generation."""
        assert safe_slug("Test File") == "Test_File"
        assert safe_slug("test-file.txt") == "test-file"
    
    def test_special_characters(self):
        """Test special character handling."""
        assert safe_slug("Test@File#Name") == "Test_File_Name"
        assert safe_slug("Test  Multiple   Spaces") == "Test_Multiple_Spaces"
    
    def test_max_length(self):
        """Test slug length limitation."""
        long_text = "word " * 50
        result = safe_slug(long_text, max_len=100)
        assert len(result) <= 100

    def test_non_string_returns_empty_slug(self):
        """Non-string inputs fail closed."""
        assert safe_slug(None) == ""  # type: ignore[arg-type]

    def test_only_special_characters_fall_back_to_untitled(self):
        """All-invalid slugs should use the default fallback."""
        assert safe_slug("!!!@@@###") == "untitled"


class TestSafeAccessors:
    """Tests for safe accessor functions."""
    
    def test_safe_get(self):
        """Test safe dictionary access."""
        data = {"key": "value"}
        assert safe_get(data, "key") == "value"
        assert safe_get(data, "missing", "default") == "default"
        assert safe_get(None, "key", "default") == "default"  # type: ignore[arg-type]
    
    def test_safe_strip(self):
        """Test safe string stripping."""
        assert safe_strip("  test  ") == "test"
        assert safe_strip(None) == "None"  # safe_strip converts None to "None" string
        assert safe_strip(123) == "123"
    
    def test_safe_lower(self):
        """Test safe lowercase conversion."""
        assert safe_lower("TEST") == "test"
        assert safe_lower(None) == "none"  # safe_lower converts None to "none"
        assert safe_lower(123) == "123"


class TestLocationNormalization:
    """Tests for location normalization functions."""
    
    def test_normalize_county_name(self):
        """Test county name normalization."""
        assert normalize_county_name("Rockland County") == "rockland"
        assert normalize_county_name("ROCKLAND") == "rockland"
        assert normalize_county_name("  Rockland  ") == "rockland"
        assert normalize_county_name("ResultsMiamiDadeCounty2024") == "resultsmiamidade"
    
    def test_normalize_state_name(self):
        """Test state name normalization."""
        assert normalize_state_name("New York") == "new_york"  # Uses underscores
        assert normalize_state_name("NEW YORK") == "new_york"
        assert normalize_state_name("  New York  ") == "new_york"
        assert normalize_state_name("results_fl") == "florida"
        assert normalize_state_name("ElecResultsFL") == "florida"
    
    def test_format_county_label(self):
        """Test county label formatting."""
        assert format_county_label("rockland", "new york") == "Rockland"
        assert format_county_label("ROCKLAND", "new york") == "Rockland"
        assert format_county_label("", "new york") == ""
    
    def test_format_state_label(self):
        """Test state label formatting."""
        assert format_state_label("new york") == "New York"
        assert format_state_label("NEW YORK") == "New York"
        assert format_state_label("") == ""


class TestSafeParse:
    """Tests for safe_parse positional argument mapping."""

    def test_scaffold_mapping(self):
        """Route-style args map into scaffold params."""
        sentinel = object()

        def handler(page=None, html_context=None, coordinator=None, context=None, session_id=None, **kwargs):
            return [], [], "", {
                "page": page,
                "html_context": html_context,
                "coordinator": coordinator,
                "context": context,
                "session_id": session_id,
            }

        _h, _d, _c, meta = safe_parse(handler, "PAGE", sentinel, {"state": "pa"}, session_id="s1")
        assert meta["page"] == "PAGE"
        assert meta["coordinator"] is sentinel
        assert meta["context"] == {"state": "pa"}
        assert meta["session_id"] == "s1"

    def test_short_mapping(self):
        """Dict positional arg maps to html_context for short handlers."""
        def handler(page=None, html_context=None):
            return [], [], "", {"page": page, "html_context": html_context}

        _h, _d, _c, meta = safe_parse(handler, "PAGE", {"county": "rockland"})
        assert meta["page"] == "PAGE"
        assert meta["html_context"] == {"county": "rockland"}

    def test_short_mapping_with_router_args(self):
        """Router-style args still provide html_context when possible."""
        def handler(page=None, html_context=None):
            return [], [], "", {"page": page, "html_context": html_context}

        _h, _d, _c, meta = safe_parse(handler, "PAGE", object(), {"county": "rockland"})
        assert meta["page"] == "PAGE"
        assert meta["html_context"] == {"county": "rockland"}

    def test_none_handler_returns_structured_error(self):
        """Missing handlers should return canonical error metadata."""
        _h, _d, contest, meta = safe_parse(None)
        assert contest == ""
        assert meta["error"] == "no_handler"

    def test_non_callable_handler_returns_structured_error(self):
        """Objects without parse/callable support fail closed."""
        class NoParse:
            pass

        _h, _d, contest, meta = safe_parse(NoParse())
        assert contest == ""
        assert meta["error"] == "no_parse_method"
