"""Tests for Context_Integration/context_coordinator.py"""
import pytest
from webapp.parser.Context_Integration.context_coordinator import (
    ContextCoordinator,
    get_semantic_score,
    dynamic_state_county_detection,
)


class TestContextCoordinator:
    """Tests for ContextCoordinator class."""
    
    def test_coordinator_initialization(self):
        """Test coordinator initialization."""
        coordinator = ContextCoordinator()
        assert coordinator is not None
        assert hasattr(coordinator, 'library')
    
    def test_extract_entities(self):
        """Test entity extraction."""
        coordinator = ContextCoordinator()
        text = "Election in New York County on November 5, 2024"
        entities = coordinator.extract_entities(text)
        
        assert isinstance(entities, list)
        # Should detect location entities
        assert any("New York" in str(entity) for entity in entities)


class TestSemanticScore:
    """Tests for semantic scoring."""
    
    def test_semantic_score_exact_match(self):
        """Test exact match scoring."""
        score = get_semantic_score("New York", "New York")
        assert score > 0.9  # Should be very high
    
    def test_semantic_score_partial_match(self):
        """Test partial match scoring."""
        score = get_semantic_score("New York County", "New York")
        assert 0.3 < score < 0.9  # Moderate score
    
    def test_semantic_score_no_match(self):
        """Test no match scenario."""
        score = get_semantic_score("California", "New York")
        assert score < 0.3  # Low score

    def test_semantic_score_invalid_input(self):
        """Invalid inputs should fail closed to 0."""
        assert get_semantic_score(None, "New York") == 0.0  # type: ignore[arg-type]
        assert get_semantic_score("", "New York") == 0.0


class TestStateCountyDetection:
    """Tests for state/county detection."""
    
    def test_dynamic_state_county_detection(self, mock_coordinator):
        """Test dynamic state/county detection."""
        text = "Election Results for Rockland County, New York"
        state, county = dynamic_state_county_detection(text, coordinator=mock_coordinator)
        
        # Should detect state and county
        assert state is not None or county is not None

    def test_dynamic_state_county_detection_from_context_fields(self):
        """Direct context should resolve county/state in lightweight mode."""
        county, state = dynamic_state_county_detection({"county": "Rockland County", "state": "NY"})[:2]
        assert county == "rockland"
        assert state == "new_york"

    def test_dynamic_state_county_detection_with_simple_text_returns_two_tuple(self, mock_coordinator):
        """String input path used by tests should return a lightweight pair."""
        result = dynamic_state_county_detection("Election Results for Albany County, New York", coordinator=mock_coordinator)
        assert isinstance(result, tuple)
        assert len(result) == 2
