"""Tests for webapp/parser/utils/detect.py"""
import pytest
from webapp.parser.utils.detect import (
    normalize_text,
    normalize_header,
    is_location_header,
    dynamic_detect_location_header,
    detect_candidate_column,
    harmonize_headers_and_data,
    parse_numeric,
    dedupe_headers_with_suffix,
)


class TestNormalization:
    """Tests for normalization functions."""
    
    def test_normalize_text(self):
        """Test text normalization."""
        assert normalize_text("Test String") == "test string"
        assert normalize_text("UPPERCASE") == "uppercase"
        assert normalize_text("  Extra   Spaces  ") == "extra spaces"
    
    def test_normalize_header(self):
        """Test header normalization."""
        assert normalize_header("Candidate Name") == "Candidate"
        assert normalize_header("Total Vote") == "Total Vote"
        assert normalize_header("% reported") == "Percent Reported"


class TestLocationDetection:
    """Tests for location detection functions."""
    
    def test_is_location_header(self):
        """Test location header detection."""
        assert is_location_header("Precinct") is True
        assert is_location_header("County") is True
        assert is_location_header("District") is True
        assert is_location_header("Candidate") is False
    
    def test_dynamic_detect_location_header(self):
        """Test dynamic location header detection."""
        headers = ["Precinct", "Candidate", "Votes", "Percent"]
        location, percent, entity = dynamic_detect_location_header(headers)
        assert location == "Precinct"
        assert percent is None

    def test_dynamic_detect_location_header_skips_percent_like_headers(self):
        """Location detection should not confuse percent headers as locations."""
        headers = ["% Reported", "County", "Votes"]
        location, percent, entity = dynamic_detect_location_header(headers)
        assert location == "County"
        assert percent == "% Reported"


class TestCandidateDetection:
    """Tests for candidate column detection."""
    
    def test_detect_candidate_column_by_keyword(self):
        """Test candidate column detection using keywords."""
        headers = ["Candidate", "Party", "Votes"]
        data = [{"Candidate": "John Doe", "Party": "Democratic", "Votes": "1000"}]
        result = detect_candidate_column(headers, data)
        assert result == "Candidate"
    
    def test_detect_candidate_column_no_match(self):
        """Test when no candidate column is found."""
        headers = ["Column1", "Column2", "Column3"]
        data = [{"Column1": "Data", "Column2": "More", "Column3": "Info"}]
        result = detect_candidate_column(headers, data)
        assert result is None

    def test_detect_candidate_column_ignores_generic_headers_even_with_name_like_values(self):
        """Generic column names should not be promoted without a stronger signal."""
        headers = ["Column1", "Column2", "Votes"]
        data = [
            {"Column1": "John Doe", "Column2": "District 1", "Votes": "1000"},
            {"Column1": "Jane Smith", "Column2": "District 2", "Votes": "900"},
        ]
        result = detect_candidate_column(headers, data)
        assert result is None


class TestHarmonization:
    """Tests for data harmonization."""
    
    def test_harmonize_headers_and_data(self):
        """Test header and data harmonization."""
        headers = ["Candidate", "Votes", "Precinct"]
        data = [
            {"Candidate": "John Doe", "Votes": "1000", "Precinct": "1A"},
            {"Candidate": "Jane Smith", "Votes": "900", "Precinct": "1A"}
        ]
        result_headers, result_data = harmonize_headers_and_data(headers, data)
        
        # Precinct should be first
        assert result_headers[0] == "Precinct"
        # Candidate should be present
        assert "Candidate" in result_headers
        # Data integrity maintained
        assert len(result_data) == 2

    def test_harmonize_headers_and_data_deduplicates_same_precinct_candidate(self):
        """Duplicate location/candidate rows should collapse."""
        headers = ["Precinct", "Candidate", "Votes"]
        data = [
            {"Precinct": "1A", "Candidate": "John Doe", "Votes": "1000"},
            {"Precinct": "1A", "Candidate": "John Doe", "Votes": "1000"},
        ]
        _headers, result_data = harmonize_headers_and_data(headers, data)
        assert len(result_data) == 1


class TestNumericParsing:
    """Tests for numeric parsing."""
    
    def test_parse_numeric_with_commas(self):
        """Test parsing numbers with commas."""
        value, is_percent = parse_numeric("1,234")
        assert value == 1234
        assert is_percent is False
    
    def test_parse_numeric_with_percent(self):
        """Test parsing percentages."""
        value, is_percent = parse_numeric("55.2%")
        assert value == 55
        assert is_percent is True
    
    def test_parse_numeric_invalid(self):
        """Test parsing invalid numeric values."""
        value, is_percent = parse_numeric("abc")
        assert value is None
        assert is_percent is False

    def test_parse_numeric_negative_number(self):
        """Negative integers are parsed correctly."""
        value, is_percent = parse_numeric("-42")
        assert value == -42
        assert is_percent is False

    def test_parse_numeric_negative_with_commas(self):
        """Negative values with comma-separators are parsed correctly."""
        value, is_percent = parse_numeric("-3,500")
        assert value == -3500
        assert is_percent is False

    def test_parse_numeric_lone_minus_is_rejected(self):
        """A bare minus sign with no digits returns None."""
        value, is_percent = parse_numeric("-")
        assert value is None
        assert is_percent is False


class TestHeaderDeduplication:
    """Tests for header deduplication."""
    
    def test_dedupe_headers_with_suffix(self):
        """Test deduplication with suffixes."""
        headers = ["Name", "Name", "Votes", "Name"]
        result = dedupe_headers_with_suffix(headers)
        assert result == ["Name", "Name_2", "Votes", "Name_3"]
    
    def test_dedupe_empty_headers(self):
        """Test handling of empty headers."""
        headers = ["Name", "", "", "Votes"]
        result = dedupe_headers_with_suffix(headers)
        # Empty headers should get column numbers
        assert "Column" in result[1]
        assert "Votes" in result
