"""Tests for health/session_manager.py"""
import pytest
from webapp.parser.health.session_manager import SessionManager
from webapp.parser.utils.session_state import SessionState, PipelinePhase


class TestSessionManager:
    """Tests for SessionManager class."""
    
    def test_create_session(self):
        """Test session creation."""
        manager = SessionManager()
        session_id = "test_session_123"
        metadata = manager.ensure_session(session_id, username="testuser")
        
        assert metadata["session_id"] == session_id
        assert metadata["username"] == "testuser"
        assert "created" in metadata
    
    def test_session_state_management(self):
        """Test session state transitions."""
        manager = SessionManager()
        session_id = "test_session_456"
        manager.ensure_session(session_id)
        
        # Test state transition
        result = manager.set_state(
            session_id,
            SessionState.RUNNING,
            phase=PipelinePhase.RUN
        )
        
        assert result is not None
        assert result["state"] == SessionState.RUNNING.value
        assert result["phase"] == PipelinePhase.RUN.value
    
    def test_session_manual_source(self):
        """Test manual source management."""
        manager = SessionManager()
        session_id = "test_session_789"
        manager.ensure_session(session_id)
        
        manager.set_manual_source(session_id, "uploads", origin="user")
        
        assert manager.get_manual_source(session_id) == "uploads"
        assert manager.get_manual_source_origin(session_id) == "user"
    
    def test_session_cleanup(self):
        """Test session cleanup."""
        manager = SessionManager()
        session_id = "test_session_cleanup"
        manager.ensure_session(session_id)
        
        assert manager.has_session(session_id)
        
        manager.delete_session(session_id)
        
        assert not manager.has_session(session_id)

    def test_create_session_defaults_manual_source_fields(self):
        manager = SessionManager()
        metadata = manager.ensure_session("default_source_session")

        assert metadata["manual_source"] == "input"
        assert metadata["manual_source_origin"] == "default"

    def test_set_state_updates_manual_source_extras(self):
        manager = SessionManager()
        session_id = "manual_source_update"
        manager.ensure_session(session_id)

        result = manager.set_state(
            session_id,
            SessionState.RUNNING,
            extras={"manual_source": "uploads", "manual_source_origin": "user"},
        )

        assert result is not None
        assert result["manual_source"] == "uploads"
        assert manager.get_manual_source(session_id) == "uploads"
        assert manager.get_manual_source_origin(session_id) == "user"

    def test_delete_session_clears_manual_source_metadata(self):
        manager = SessionManager()
        session_id = "session_delete_cleanup"
        manager.ensure_session(session_id)
        manager.set_manual_source(session_id, "uploads", origin="user")

        manager.delete_session(session_id)

        assert manager.get_manual_source(session_id) == "input"
        assert manager.get_manual_source_origin(session_id) == "default"
