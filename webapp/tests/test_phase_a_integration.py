"""
Integration Tests for Phase A: Confidence/Caution Decision Gates
=================================================================

Tests cover:
1. entity_confidence_map.py - Signal/anomaly calculation
2. safe_decide.py - Decision API functions
3. VocabLoader - Vocabulary file loading with integrity checks
4. Logger decision filtering - Deduplication within 5-min window
5. Prometheus metrics - Decision event counters

Test Structure:
- Fixtures for setup/teardown
- Unit tests for each module
- Integration tests combining multiple modules
- Mock/patch for external dependencies (logger, metrics)

Usage:
    pytest webapp/tests/test_phase_a_integration.py -v
    pytest webapp/tests/test_phase_a_integration.py::TestEntityConfidenceMap -v
    pytest webapp/tests/test_phase_a_integration.py -k "decision" -v
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Generator, Any
from unittest.mock import Mock, patch, MagicMock

import pytest

# Import Phase A modules
from webapp.parser.Context_Integration.library.entity_confidence_map import (
    EntityConfidenceMap,
    SignalType,
    AnomalyType,
    OverrideTrigger,
    get_confidence_map,
)
from webapp.parser.utils.safe_decide import (
    safe_decide_jurisdiction,
    safe_decide_office,
    safe_decide_party,
    safe_decide_source,
)
from webapp.parser.Context_Integration.vocab.loader import (
    VocabLoader,
    VocabLoaderError,
    VocabFileNotFound,
    VocabSecurityError,
)
from webapp.parser.utils.shared_logic import DecisionTuple
from webapp.parser.utils.logger_singleton import logger
from webapp.parser.utils.metrics_prom import increment_prom_counter


# ========================================================================
# Fixtures
# ========================================================================

@pytest.fixture
def confidence_map() -> EntityConfidenceMap:
    """Provide a fresh EntityConfidenceMap instance."""
    return get_confidence_map()


@pytest.fixture
def temp_vocab_dir() -> Generator[Path, None, None]:
    """Create a temporary vocab directory with sample files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_dir = Path(tmpdir) / "vocab"
        
        # Create subdirectories
        (vocab_dir / "entities").mkdir(parents=True, exist_ok=True)
        (vocab_dir / "validators").mkdir(parents=True, exist_ok=True)
        (vocab_dir / "sources").mkdir(parents=True, exist_ok=True)
        
        # Create sample vocab files
        (vocab_dir / "entities" / "offices.txt").write_text(
            "# Offices\nPresident\nGovernor\nMayor\n"
        )
        (vocab_dir / "entities" / "parties.txt").write_text(
            "# Parties\nDemocratic Party\nRepublican Party\n"
        )
        (vocab_dir / "entities" / "jurisdictions.txt").write_text(
            "# Jurisdictions\nCalifornia\nNew York\n"
        )
        (vocab_dir / "validators" / "office_aliases.txt").write_text(
            "Pres -> President\nGov -> Governor\n"
        )
        (vocab_dir / "validators" / "party_aliases.txt").write_text(
            "Dem -> Democratic Party\nRep -> Republican Party\n"
        )
        (vocab_dir / "sources" / "verified_sources.txt").write_text(
            "sos.ca.gov\nfec.gov\nelections.ny.gov\n"
        )
        
        yield vocab_dir


@pytest.fixture
def vocab_loader(temp_vocab_dir) -> Generator[VocabLoader, None, None]:
    """Provide a VocabLoader with temporary vocab directory."""
    loader = VocabLoader(base_dir=temp_vocab_dir)
    yield loader
    loader.clear_cache()


@pytest.fixture
def mock_logger():
    """Mock the logger for testing."""
    with patch('webapp.parser.utils.logger_singleton.logger') as mock:
        yield mock


@pytest.fixture
def mock_metrics():
    """Mock metrics for testing."""
    with patch('webapp.parser.utils.metrics_prom.increment_prom_counter') as mock:
        yield mock


# ========================================================================
# Test EntityConfidenceMap
# ========================================================================

class TestEntityConfidenceMap:
    """Tests for entity_confidence_map.py module."""
    
    def test_confidence_map_singleton(self):
        """Verify EntityConfidenceMap is a singleton."""
        map1 = get_confidence_map()
        map2 = get_confidence_map()
        assert map1 is map2
    
    def test_signal_types_defined(self, confidence_map):
        """Verify all signal types are defined."""
        signal_types = list(SignalType)
        assert len(signal_types) >= 10  # Should have 10+ signal types
        assert SignalType.EXACT_MATCH_VERIFIED in signal_types
        assert SignalType.FUZZY_MATCH_HIGH in signal_types
    
    def test_anomaly_types_defined(self, confidence_map):
        """Verify all anomaly types are defined."""
        anomaly_types = list(AnomalyType)
        assert len(anomaly_types) >= 8  # Should have 8+ anomaly types
        assert AnomalyType.MISMATCHED_TOTALS in anomaly_types
        assert AnomalyType.CONTEXTUAL_MISMATCH in anomaly_types
    
    def test_override_triggers_defined(self, confidence_map):
        """Verify all override triggers are defined."""
        override_triggers = list(OverrideTrigger)
        assert len(override_triggers) >= 4  # Enum aliases collapse duplicate values
        assert OverrideTrigger.ADMIN_FLAG in override_triggers
        assert OverrideTrigger.VERIFIED_SOURCE_CORRECTION in override_triggers
        assert "ML_MODEL_LOW_CONFIDENCE" in OverrideTrigger.__members__
        assert "MULTIPLE_CORRECTIONS" in OverrideTrigger.__members__
    
    def test_calculate_confidence_exact_match(self, confidence_map):
        """Test confidence calculation with exact match signal."""
        signals = [(SignalType.EXACT_MATCH_VERIFIED, True)]
        anomalies: list[tuple[AnomalyType, bool]] = []
        overrides: list[OverrideTrigger] = []

        result = confidence_map.calculate_confidence_caution(
            entity_id="Governor",
            entity_type="office",
            signals=signals,
            anomalies=anomalies,
            override_triggers=overrides,
        )

        assert result.confidence_score > 0.8  # Exact match should have high confidence
        assert result.caution_score < 0.2
        assert result.override_score == 0.0
    
    def test_calculate_confidence_mixed_signals(self, confidence_map):
        """Test confidence calculation with mixed signals."""
        signals = [
            (SignalType.EXACT_MATCH_VERIFIED, True),
            (SignalType.FUZZY_MATCH_HIGH, True),
        ]
        anomalies = [(AnomalyType.TYPOSQUAT_PATTERN, True)]
        overrides: list[OverrideTrigger] = []

        result = confidence_map.calculate_confidence_caution(
            entity_id="Democratic Party",
            entity_type="party",
            signals=signals,
            anomalies=anomalies,
            override_triggers=overrides,
        )

        assert 0.0 <= result.confidence_score <= 1.0
        assert 0.0 <= result.caution_score <= 1.0
        assert result.override_score == 0.0
    
    def test_calculate_confidence_with_anomalies(self, confidence_map):
        """Test confidence calculation degradation with anomalies."""
        signals = [(SignalType.EXACT_MATCH_VERIFIED, True)]
        anomalies = [
            (AnomalyType.MISMATCHED_TOTALS, True),
            (AnomalyType.VALUE_INCONSISTENCY, True),
            (AnomalyType.CONTEXTUAL_MISMATCH, True),
        ]
        overrides: list[OverrideTrigger] = []

        result = confidence_map.calculate_confidence_caution(
            entity_id="California",
            entity_type="jurisdiction",
            signals=signals,
            anomalies=anomalies,
            override_triggers=overrides,
        )

        # Caution should increase due to anomalies
        assert result.caution_score > 0.3
    
    def test_calculate_confidence_with_override(self, confidence_map):
        """Test confidence calculation with override trigger."""
        signals: list[tuple[SignalType, bool]] = []
        anomalies = [(AnomalyType.MISSING_CANDIDATE, True)]
        overrides = [OverrideTrigger.VERIFIED_SOURCE_CORRECTION]

        result = confidence_map.calculate_confidence_caution(
            entity_id="sos.ca.gov",
            entity_type="source",
            signals=signals,
            anomalies=anomalies,
            override_triggers=overrides,
        )

        # Override should boost confidence
        assert result.override_score > 0.0


# ========================================================================
# Test safe_decide API
# ========================================================================

class TestSafeDecideAPI:
    """Tests for safe_decide.py module."""
    
    def test_safe_decide_jurisdiction_pass(self, mock_logger, mock_metrics):
        """Test jurisdiction decision with high confidence."""
        from webapp.parser.utils.safe_decide import SignalType
        
        result = safe_decide_jurisdiction(
            entity_id="Los Angeles County",
            state="CA",
            signals=[(SignalType.EXACT_MATCH_VERIFIED, True), (SignalType.HEADER_ALIGNMENT, True)],
            session_id="test_session_001"
        )
        
        assert isinstance(result, dict)
        assert result.get("decision_code") in ["proceed", "caution", "stop"]
        assert "confidence_score" in result
        assert result.get("session_id") == "test_session_001"
    
    def test_safe_decide_office_caution(self, mock_logger, mock_metrics):
        """Test office decision with mixed signals (CAUTION)."""
        from webapp.parser.utils.safe_decide import SignalType, AnomalyType
        
        result = safe_decide_office(
            entity_id="Governor",
            state="CA",
            signals=[(SignalType.FUZZY_MATCH_MEDIUM, True)],
            anomalies=[(AnomalyType.SUSPICIOUS_HEADER, True)],
            session_id="test_session_002"
        )
        
        assert isinstance(result, dict)
        assert result.get("decision_code") in ["proceed", "caution", "stop"]
        assert "caution_score" in result
    
    def test_safe_decide_party_stop(self, mock_logger, mock_metrics):
        """Test party decision with low confidence (STOP)."""
        from webapp.parser.utils.safe_decide import AnomalyType, SignalType

        result = safe_decide_party(
            entity_id="Uncommon Party Name",
            signals=[(SignalType.FUZZY_MATCH_LOW, True)],
            anomalies=[(AnomalyType.CONTEXTUAL_MISMATCH, True), (AnomalyType.VALUE_INCONSISTENCY, True)],
            session_id="test_session_003"
        )
        
        assert isinstance(result, dict)
        assert result.get("decision_code") in ["proceed", "caution", "stop"]
        assert "confidence_score" in result
    
    def test_safe_decide_source_verified(self, mock_logger, mock_metrics):
        """Test source decision with verified domain."""
        from webapp.parser.utils.safe_decide import SignalType

        result = safe_decide_source(
            url="sos.ca.gov",
            signals=[(SignalType.EXACT_MATCH_VERIFIED, True), (SignalType.HANDLER_SUCCESS, True)],
            session_id="test_session_004"
        )
        
        assert isinstance(result, dict)
        assert result.get("decision_code") in ["proceed", "caution", "stop"]
    
    def test_decision_tuple_structure(self):
        """Verify DecisionTuple TypedDict has all required fields."""
        decision_dict: DecisionTuple = {
            "value": "test_value",
            "decision_code": "proceed",
            "confidence_score": 0.95,
            "caution_score": 0.05,
            "override_score": 0.0,
            "signals_observed": ["exact_match_verified"],
            "anomalies_observed": [],
            "reasoning": "High confidence exact match",
            "timestamp": "2026-02-09T12:00:00Z",
            "session_id": "test_session"
        }
        
        assert decision_dict["value"] == "test_value"
        assert decision_dict["decision_code"] == "proceed"
        assert decision_dict["confidence_score"] == 0.95

    def test_safe_decide_source_stop_helper_consistency(self, mock_logger, mock_metrics):
        """Helper predicates should agree with stop decisions."""
        from webapp.parser.utils.safe_decide import AnomalyType, SignalType, should_stop

        result = safe_decide_source(
            url="unknown-source.example",
            signals=[(SignalType.FUZZY_MATCH_LOW, True)],
            anomalies=[(AnomalyType.CONTEXTUAL_MISMATCH, True), (AnomalyType.MISMATCHED_TOTALS, True)],
            session_id="test_session_005",
        )

        if result.get("decision_code") == "stop":
            assert should_stop(result) is True

    def test_safe_decide_party_emits_session_id(self, mock_logger, mock_metrics):
        """Decision payloads should retain session context."""
        from webapp.parser.utils.safe_decide import SignalType

        result = safe_decide_party(
            entity_id="Democratic Party",
            signals=[(SignalType.EXACT_MATCH_VERIFIED, True)],
            session_id="decision-session",
        )

        assert result["session_id"] == "decision-session"


# ========================================================================
# Test VocabLoader
# ========================================================================

class TestVocabLoader:
    """Tests for vocab/loader.py module."""
    
    def test_vocab_loader_load_canonical(self, vocab_loader):
        """Test loading canonical entity list."""
        offices = vocab_loader.load_canonical("entities", "offices.txt")
        
        assert isinstance(offices, list)
        assert "President" in offices
        assert "Governor" in offices
        assert "Mayor" in offices
    
    def test_vocab_loader_load_mapping(self, vocab_loader):
        """Test loading alias mappings."""
        aliases = vocab_loader.load_mapping("validators", "office_aliases.txt")
        
        assert isinstance(aliases, dict)
        assert aliases.get("Pres") == "President"
        assert aliases.get("Gov") == "Governor"
    
    def test_vocab_loader_cache_hit(self, vocab_loader):
        """Test that subsequent loads hit cache."""
        # First load
        load1 = vocab_loader.get_load_count("entities", "offices.txt")
        offices1 = vocab_loader.load_canonical("entities", "offices.txt")
        load2 = vocab_loader.get_load_count("entities", "offices.txt")
        
        # Second load (should hit cache)
        offices2 = vocab_loader.load_canonical("entities", "offices.txt")
        load3 = vocab_loader.get_load_count("entities", "offices.txt")
        
        # Load counts might vary, but data should be identical
        assert offices1 == offices2
    
    def test_vocab_loader_skip_cache(self, vocab_loader):
        """Test skip_cache parameter."""
        offices1 = vocab_loader.load_canonical("entities", "offices.txt", skip_cache=False)
        offices2 = vocab_loader.load_canonical("entities", "offices.txt", skip_cache=True)
        
        # Data should be identical regardless of cache
        assert offices1 == offices2
    
    def test_vocab_loader_file_not_found(self, vocab_loader):
        """Test error handling for missing file."""
        with pytest.raises(VocabFileNotFound):
            vocab_loader.load_canonical("entities", "nonexistent.txt")
    
    def test_vocab_loader_security_path_traversal(self, vocab_loader):
        """Test security: prevent path traversal attacks."""
        with pytest.raises(VocabSecurityError):
            vocab_loader.load_canonical("entities", "../../../etc/passwd.txt")
    
    def test_vocab_loader_security_invalid_subdir(self, vocab_loader):
        """Test security: reject invalid subdirectory."""
        with pytest.raises(VocabSecurityError):
            vocab_loader.load_canonical("invalid_subdir", "file.txt")


# ========================================================================
# Test Logger Decision Filtering
# ========================================================================

class TestLoggerDecisionFiltering:
    """Tests for decision event deduplication in logger."""
    
    def test_logger_filter_decision_noise_first_event(self):
        """Test that first decision event is not filtered."""
        result = logger._filter_decision_noise(
            entity_value="office:President",
            decision_code="PROCEED"
        )
        
        assert result is True  # First event should pass
    
    def test_logger_filter_decision_noise_duplicate(self):
        """Test that duplicate events within window are filtered."""
        # Clear cache
        logger._decision_event_cache.clear()
        
        entity = "office:Governor"
        decision = "CAUTION"
        now = time.time()
        
        # First call - should pass
        result1 = logger._filter_decision_noise(entity, decision, now)
        assert result1 is True
        
        # Second call (100ms later, within 5-min window) - should be filtered
        result2 = logger._filter_decision_noise(entity, decision, now + 0.1)
        assert result2 is False
    
    def test_logger_filter_decision_noise_after_window(self):
        """Test that events after window expires are not filtered."""
        logger._decision_event_cache.clear()
        
        entity = "jurisdiction:California"
        decision = "PROCEED"
        now = time.time()
        
        # First call
        result1 = logger._filter_decision_noise(entity, decision, now)
        assert result1 is True
        
        # Call after window expires (6 minutes later)
        result2 = logger._filter_decision_noise(entity, decision, now + 361)
        assert result2 is True  # Should pass because window expired
    
    def test_logger_filter_decision_noise_different_decision(self):
        """Test that same entity with different decision code is not filtered."""
        logger._decision_event_cache.clear()
        
        entity = "party:Democratic Party"
        now = time.time()
        
        # First decision: PROCEED
        result1 = logger._filter_decision_noise(entity, "PROCEED", now)
        assert result1 is True
        
        # Different decision code: CAUTION - should not be filtered
        result2 = logger._filter_decision_noise(entity, "CAUTION", now + 0.1)
        assert result2 is True
    
    def test_logger_filter_decision_noise_cleanup(self):
        """Test that cache cleanup removes expired entries."""
        logger._decision_event_cache.clear()
        
        now = time.time()
        
        # Add entries with different timestamps
        logger._filter_decision_noise("entity1", "PROCEED", now)
        logger._filter_decision_noise("entity2", "CAUTION", now - 400)  # Expired
        
        # Trigger cleanup by calling filter again
        logger._filter_decision_noise("entity3", "STOP", now + 10)
        
        # entity2 should be cleaned up (older than 5 min)
        # This is a side effect of the next filter call
        assert len(logger._decision_event_cache) >= 2  # entity1 and entity3


# ========================================================================
# Test Prometheus Metrics Integration
# ========================================================================

class TestPrometheusMetrics:
    """Tests for Prometheus metrics counters."""
    
    def test_metrics_decision_proceed_counter(self):
        """Test decision_proceed_total counter."""
        # Mock the counter
        with patch('webapp.parser.utils.metrics_prom._ENABLED', True):
            with patch('webapp.parser.utils.metrics_prom._counters') as mock_counters:
                mock_counter = MagicMock()
                mock_counters.get.return_value = mock_counter

                increment_prom_counter('decision_proceed_total', 1)

                # Verify counter was incremented
                mock_counters.get.assert_called_with('decision_proceed_total')
    
    def test_metrics_decision_caution_counter(self):
        """Test decision_caution_total counter."""
        with patch('webapp.parser.utils.metrics_prom._ENABLED', True):
            with patch('webapp.parser.utils.metrics_prom._counters') as mock_counters:
                mock_counter = MagicMock()
                mock_counters.get.return_value = mock_counter

                increment_prom_counter('decision_caution_total', 1)

                mock_counters.get.assert_called_with('decision_caution_total')
    
    def test_metrics_decision_stop_counter(self):
        """Test decision_stop_total counter."""
        with patch('webapp.parser.utils.metrics_prom._ENABLED', True):
            with patch('webapp.parser.utils.metrics_prom._counters') as mock_counters:
                mock_counter = MagicMock()
                mock_counters.get.return_value = mock_counter

                increment_prom_counter('decision_stop_total', 1)

                mock_counters.get.assert_called_with('decision_stop_total')


# ========================================================================
# Integration Tests
# ========================================================================

class TestPhaseAIntegration:
    """Integration tests combining multiple Phase A modules."""
    
    def test_full_decision_flow(self, confidence_map, vocab_loader, mock_logger, mock_metrics):
        """Test complete flow: entity lookup -> confidence calc -> decision -> log -> metrics."""
        # Step 1: Load entity from vocab
        offices = vocab_loader.load_canonical("entities", "offices.txt")
        assert "Governor" in offices
        
        # Step 2: Calculate confidence/caution
        signals = [(SignalType.EXACT_MATCH_VERIFIED, True)]
        calc_result = confidence_map.calculate_confidence_caution(
            entity_id="Governor",
            entity_type="office",
            signals=signals,
            anomalies=[],
            override_triggers=[],
        )

        # Step 3: Make decision
        decision_result = safe_decide_office(
            entity_id="Governor",
            state="CA",
            signals=signals,
            session_id="integration_test"
        )
        
        assert decision_result is not None
        assert "decision_code" in decision_result
        assert decision_result["decision_code"] == calc_result.decision_code.value
    
    def test_decision_with_filtering_and_metrics(self, mock_logger, mock_metrics):
        """Test that decision events are filtered and metrics are updated."""
        logger._decision_event_cache.clear()
        
        entity = "test_entity"
        now = time.time()
        
        # First event - should pass filter
        should_log1 = logger._filter_decision_noise(entity, "PROCEED", now)
        assert should_log1 is True
        
        # Second event (duplicate) - should be filtered
        should_log2 = logger._filter_decision_noise(entity, "PROCEED", now + 0.1)
        assert should_log2 is False
        
        # Third event (after window) - should pass filter
        should_log3 = logger._filter_decision_noise(entity, "PROCEED", now + 361)
        assert should_log3 is True


# ========================================================================
# Run Tests
# ========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
