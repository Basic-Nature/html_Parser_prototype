"""Tests for handlers/batch_handler.py"""
import pytest
from unittest.mock import Mock
from webapp.parser.handlers.batch_handler import BatchProcessor


class TestBatchProcessor:
    """Tests for BatchProcessor class."""
    
    def test_batch_processor_initialization(self, mock_coordinator, mock_page):
        """Test batch processor initialization."""
        processor = BatchProcessor(
            coordinator=mock_coordinator,
            handler=Mock(),
            page=mock_page,
            base_context={},
            selected_races=[{"title": "Contest 1"}, {"title": "Contest 2"}],
            initial_result=None,
            session_id="test_session",
            target_url="https://example.com",
            output_dir="/tmp",
            processed_info=Mock(),
            ai_analyze_results=Mock(),
            stream_results=Mock(),
            mark_url_processed=Mock(),
        )
        
        assert processor.total_expected == 2
        assert len(processor.original_races) == 2

    def test_batch_processor_counts_initial_result_without_matching_race(self, mock_coordinator, mock_page):
        processor = BatchProcessor(
            coordinator=mock_coordinator,
            handler=Mock(),
            page=mock_page,
            base_context={},
            selected_races=[{"title": "Contest 1"}],
            initial_result=(['Candidate'], [{'Candidate': 'Alice'}], 'Contest 2', {}),
            session_id="test_session",
            target_url="https://example.com",
            output_dir="/tmp",
            processed_info=Mock(),
            ai_analyze_results=Mock(),
            stream_results=Mock(),
            mark_url_processed=Mock(),
        )

        assert processor.total_expected == 2

    def test_batch_processor_emit_result_rejects_incomplete_payload(self, mock_coordinator, mock_page):
        processor = BatchProcessor(
            coordinator=mock_coordinator,
            handler=Mock(),
            page=mock_page,
            base_context={},
            selected_races=[],
            initial_result=None,
            session_id="test_session",
            target_url="https://example.com",
            output_dir="/tmp",
            processed_info=Mock(),
            ai_analyze_results=Mock(),
            stream_results=Mock(),
            mark_url_processed=Mock(),
        )

        accepted = processor._emit_result(([], [], '', {}), None, index=1)

        assert accepted is False
        assert processor.results == []
