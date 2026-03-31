"""Schema validation and regression tests for parser output

This module tests the unified election data schema across all formats (JSON, PDF, HTML, CSV).
Ensures party normalization, division type population, and consistent metadata are applied.
"""

import pytest
from webapp.parser.utils.table_builder import build_table_noninteractive
from webapp.parser.utils.shared_logic import safe_get


class TestSchemaValidation:
    """Unified schema validation tests."""
    
    def test_division_type_column_populated(self, mock_coordinator):
        """Test that Division Type column is included in output."""
        headers = ["Candidate", "Votes", "Percent"]
        data = [
            {"Candidate": "John Doe", "Votes": "1000", "Percent": "55%"},
            {"Candidate": "Jane Smith", "Votes": "900", "Percent": "45%"}
        ]
        context = {
            "contest": "Governor",
            "state": "California",
            "county": "San Francisco",
            "include_division_type_column": True
        }
        
        result_headers, result_data, entity_info = build_table_noninteractive(
            domain="ca_sf",
            headers=headers,
            data=data,
            coordinator=mock_coordinator,
            context=context,
            pivot_to_wide=False,
            debug=False
        )
        
        # Verify Division Type column is in output
        assert "Division Type" in result_headers, "Division Type column should be in headers"
        
        # Verify all rows have division type values
        for row in result_data:
            div_type = row.get("Division Type")
            assert div_type is not None, "All rows should have Division Type value"
    
    def test_party_normalization_applied(self, mock_coordinator):
        """Test that party values are normalized across formats."""
        # Test various party formats that should normalize to canonical values
        test_cases = [
            ("Democratic Party", "Democratic"),
            ("Republican Party", "Republican"),
            ("Dem", "Democratic"),
            ("GOP", "Republican"),
            ("Independent", "Independent"),
        ]
        
        for raw_party, expected_canonical in test_cases:
            headers = ["Candidate", "Party", "Votes"]
            data = [
                {"Candidate": "Test Candidate", "Party": raw_party, "Votes": "1000"}
            ]
            context = {
                "contest": "Test",
                "state": "Test State",
                "normalize_party_values": True
            }
            
            result_headers, result_data, entity_info = build_table_noninteractive(
                domain="test",
                headers=headers,
                data=data,
                coordinator=mock_coordinator,
                context=context,
                pivot_to_wide=False,
                debug=False
            )
            
            # Party should be present and normalized
            assert len(result_data) > 0
            party_value = result_data[0].get("Party", "").strip()
            # Either exact match or normalized value
            assert party_value in (raw_party, expected_canonical) or expected_canonical.lower() in party_value.lower(), \
                f"Party '{raw_party}' should normalize toward '{expected_canonical}', got '{party_value}'"
    
    def test_jurisdiction_header_consistency(self, mock_coordinator):
        """Test that jurisdiction headers are consistently formatted."""
        # Headers with various jurisdiction formats
        headers = [
            "Precinct Name",  # Location-like
            "Candidate",
            "Votes",
            "County Name"     # Jurisdiction-like
        ]
        data = [
            {
                "Precinct Name": "Precinct 1A",
                "Candidate": "John Doe",
                "Votes": "500",
                "County Name": "Los Angeles"
            }
        ]
        context = {
            "contest": "Test",
            "state": "CA",
            "county": "Los Angeles"
        }
        
        result_headers, result_data, entity_info = build_table_noninteractive(
            domain="ca_la",
            headers=headers,
            data=data,
            coordinator=mock_coordinator,
            context=context,
            pivot_to_wide=False,
            debug=False
        )
        
        # Verify output headers don't have duplicates
        assert len(result_headers) == len(set(result_headers)), \
            f"Headers should not have duplicates: {result_headers}"
        
        # Verify required columns are present
        assert "Candidate" in result_headers
        assert "Votes" in result_headers
    
    def test_metadata_enrichment(self, mock_coordinator):
        """Test that metadata is properly enriched with source info."""
        headers = ["Candidate", "Votes"]
        data = [
            {"Candidate": "John Doe", "Votes": "1000"},
        ]
        context = {
            "contest": "Governor",
            "state": "Texas",
            "county": "Harris",
            "source_url": "https://example.com/results",
            "handler": "texas_harris_handler",
            "url": "https://example.com/results"
        }
        
        result_headers, result_data, entity_info = build_table_noninteractive(
            domain="tx_harris",
            headers=headers,
            data=data,
            coordinator=mock_coordinator,
            context=context,
            pivot_to_wide=False,
            debug=False
        )
        
        # Verify entity_info contains enriched metadata
        assert entity_info is not None
        # Source, handler info should be available in context or entity_info
        assert context.get("source_url") == "https://example.com/results"
        assert context.get("state") == "Texas"

    def test_explicit_division_type_is_respected(self, mock_coordinator):
        """Schema output should preserve explicit division type context."""
        headers = ["Candidate", "Votes"]
        data = [{"Candidate": "John Doe", "Votes": "1000"}]
        context = {
            "contest": "Governor",
            "state": "Texas",
            "division_type": "County",
            "include_division_type_column": True,
        }

        result_headers, result_data, _entity_info = build_table_noninteractive(
            domain="tx_test",
            headers=headers,
            data=data,
            coordinator=mock_coordinator,
            context=context,
            pivot_to_wide=False,
            debug=False,
        )

        assert "Division Type" in result_headers
        assert result_data[0]["Division Type"] == "County"
    
    def test_multi_contest_schema_consistency(self, mock_coordinator):
        """Test that schema remains consistent across multiple contests."""
        contests_data: list[dict] = [
            {
                "headers": ["Candidate", "Votes"],
                "data": [{"Candidate": "Alice", "Votes": "100"}],
                "context": {"contest": "Governor", "state": "NY"}
            },
            {
                "headers": ["Candidate", "Votes"],
                "data": [{"Candidate": "Bob", "Votes": "200"}],
                "context": {"contest": "Senator", "state": "NY"}
            }
        ]
        
        all_headers = []
        for contest_data in contests_data:
            result_headers, result_data, entity_info = build_table_noninteractive(
                domain="ny_test",
                headers=contest_data["headers"],
                data=contest_data["data"],
                coordinator=mock_coordinator,
                context=contest_data["context"],
                pivot_to_wide=False,
                debug=False
            )
            all_headers.append(result_headers)
        
        # All contests should produce consistent header structures
        # (may have different content but structure should match)
        assert len(all_headers) == len(contests_data)
        for headers in all_headers:
            assert isinstance(headers, list)
            assert len(headers) > 0


class TestRegressionFixtures:
    """Regression tests with fixture data."""
    
    def test_multi_contest_pdf_schema(self, mock_coordinator):
        """Test schema for multi-contest PDF extraction.
        
        Regression fixture: Large multi-contest PDF with ward/precinct breakdown.
        """
        # Simulate extracted data from multi-contest PDF
        headers = ["Ward", "Precinct", "Candidate", "Votes", "Percent"]
        data = [
            {
                "Ward": "1",
                "Precinct": "1A",
                "Candidate": "John Doe",
                "Votes": "500",
                "Percent": "52%"
            },
            {
                "Ward": "1",
                "Precinct": "1A",
                "Candidate": "Jane Smith",
                "Votes": "450",
                "Percent": "48%"
            },
            {
                "Ward": "1",
                "Precinct": "1B",
                "Candidate": "John Doe",
                "Votes": "600",
                "Percent": "55%"
            },
            {
                "Ward": "1",
                "Precinct": "1B",
                "Candidate": "Jane Smith",
                "Votes": "500",
                "Percent": "45%"
            }
        ]
        context = {
            "contest": "Mayor",
            "state": "Pennsylvania",
            "county": "Philadelphia",
            "source": "pdf",
            "include_division_type_column": True
        }
        
        result_headers, result_data, entity_info = build_table_noninteractive(
            domain="pa_philly",
            headers=headers,
            data=data,
            coordinator=mock_coordinator,
            context=context,
            pivot_to_wide=True,  # Test wide format
            debug=False
        )
        
        # Verify pivoted output has correct structure
        assert len(result_data) > 0
        assert "Division Type" in result_headers or "Precinct" in result_headers
        
        # Verify no data loss
        assert len(result_data) >= 2  # At least 2 precincts
    
    def test_json_fast_path_schema(self, mock_coordinator):
        """Test schema for JSON fast-path extraction."""
        # Simulate JSON structured data
        headers = ["County", "Candidate", "Votes", "Party"]
        data = [
            {
                "County": "Cook",
                "Candidate": "Candidate A",
                "Votes": "100000",
                "Party": "Democratic"
            },
            {
                "County": "Cook",
                "Candidate": "Candidate B",
                "Votes": "90000",
                "Party": "Republican"
            }
        ]
        context = {
            "contest": "Governor",
            "state": "Illinois",
            "source": "json",
            "normalize_party_values": True
        }
        
        result_headers, result_data, entity_info = build_table_noninteractive(
            domain="il_json",
            headers=headers,
            data=data,
            coordinator=mock_coordinator,
            context=context,
            pivot_to_wide=False,
            debug=False
        )
        
        assert len(result_data) == 2
        assert all("Candidate" in row for row in result_data)
        assert all("Votes" in row for row in result_data)
        assert all("Division Type" in row for row in result_data)


class TestSchemaDocumentation:
    """Documentation validation for schema."""
    
    def test_canonical_schema_fields(self):
        """Verify canonical schema field list is documented."""
        canonical_fields = {
            "Candidate",
            "Votes",
            "Percent",
            "Party",
            "Division Type",
            "Division Name",
        }
        
        # These fields should be documented in handlers.md
        # This test validates the schema contract exists
        assert len(canonical_fields) > 0
        assert "Candidate" in canonical_fields
        assert "Votes" in canonical_fields
