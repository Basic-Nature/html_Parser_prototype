"""Tests for Context_Integration/librarian.py"""
import pytest
import tempfile
from pathlib import Path
from webapp.parser.Context_Integration.librarian import (
    load_context_library,
    update_context_library,
    parse_filename_for_location,
)


class TestLibrarian:
    """Tests for librarian functions."""
    
    def test_parse_filename_for_location(self):
        """Test filename parsing for location metadata."""
        test_cases: list[tuple[str, dict]] = [
            ("2024_General_NewYork_Rockland.csv", {"state": "NewYork", "county": "Rockland", "year": 2024}),
            ("Election_Results_2022.pdf", {"year": 2022}),
            ("Governor_Race_California.html", {"state": "California"}),
        ]
        
        for filename, expected in test_cases:
            result = parse_filename_for_location(filename)
            for key, value in expected.items():
                assert result.get(key) == value


class TestContextLibrary:
    """Tests for context library management."""
    
    def test_load_context_library(self):
        """Test loading context library."""
        library = load_context_library()
        
        assert isinstance(library, dict)
        # Should have expected keys
        assert "contests" in library or "panels" in library
    
    def test_update_context_library(self, temp_output_dir):
        """Test updating context library."""
        # This test would require a mock context library file
        # Implementation depends on test environment setup
        pass
