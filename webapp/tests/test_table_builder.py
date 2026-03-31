"""Tests for webapp/parser/utils/table_builder.py"""
import pytest
from webapp.parser.utils.table_builder import build_table_noninteractive


class TestTableBuilder:
    """Tests for table building functionality."""
    
    def test_build_simple_table(self, mock_coordinator):
        """Test building a simple table."""
        headers = ["Candidate", "Votes", "Percent"]
        data = [
            {"Candidate": "John Doe", "Votes": "1000", "Percent": "55%"},
            {"Candidate": "Jane Smith", "Votes": "900", "Percent": "45%"}
        ]
        context = {"contest": "Test Contest", "state": "New York", "county": "Rockland"}
        
        result_headers, result_data, entity_info = build_table_noninteractive(
            domain="test",
            headers=headers,
            data=data,
            coordinator=mock_coordinator,
            context=context,
            pivot_to_wide=False,
            debug=False
        )
        
        assert result_headers is not None
        assert result_data is not None
        assert len(result_data) > 0
    
    def test_build_table_with_pivot(self, mock_coordinator):
        """Test table building with wide pivot."""
        headers = ["Precinct", "Candidate", "Votes"]
        data = [
            {"Precinct": "1A", "Candidate": "John Doe", "Votes": "500"},
            {"Precinct": "1A", "Candidate": "Jane Smith", "Votes": "400"},
            {"Precinct": "1B", "Candidate": "John Doe", "Votes": "500"},
            {"Precinct": "1B", "Candidate": "Jane Smith", "Votes": "500"}
        ]
        context = {"contest": "Test Contest"}
        
        result_headers, result_data, entity_info = build_table_noninteractive(
            domain="test",
            headers=headers,
            data=data,
            coordinator=mock_coordinator,
            context=context,
            pivot_to_wide=True,
            debug=False
        )
        
        # After pivot, should have candidate columns
        assert any("John Doe" in str(h) or "Jane Smith" in str(h) for h in result_headers)

    def test_build_table_adds_default_division_type(self, mock_coordinator):
        """Division Type defaults to State when requested without explicit division_type."""
        headers = ["Candidate", "Votes"]
        data = [{"Candidate": "John Doe", "Votes": "1000"}]
        context = {"contest": "Governor", "include_division_type_column": True}

        result_headers, result_data, _entity_info = build_table_noninteractive(
            domain="test",
            headers=headers,
            data=data,
            coordinator=mock_coordinator,
            context=context,
            pivot_to_wide=False,
            debug=False,
        )

        assert result_headers[0] == "Division Type"
        assert result_data[0]["Division Type"] == "State"

    def test_build_table_respects_explicit_division_type(self, mock_coordinator):
        """Explicit division_type should be propagated into all rows."""
        headers = ["Candidate", "Votes"]
        data = [{"Candidate": "John Doe", "Votes": "1000"}]
        context = {
            "contest": "Mayor",
            "include_division_type_column": True,
            "division_type": "County",
        }

        result_headers, result_data, _entity_info = build_table_noninteractive(
            domain="test",
            headers=headers,
            data=data,
            coordinator=mock_coordinator,
            context=context,
            pivot_to_wide=False,
            debug=False,
        )

        assert "Division Type" in result_headers
        assert all(row["Division Type"] == "County" for row in result_data)


class TestTableCore:
    """Tests for table_core.py functionality."""
    
    def test_robust_table_extraction(self, mock_page, mock_coordinator):
        """Test robust table extraction."""
        from webapp.parser.utils.table_core import robust_table_extraction
        
        context = {
            "session_id": "test_session",
            "contest": "Test Contest",
            "coordinator": mock_coordinator
        }
        
        headers, data = robust_table_extraction(mock_page, context)
        
        # Should return lists even if empty
        assert isinstance(headers, list)
        assert isinstance(data, list)
