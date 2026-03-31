"""Tests for webapp/parser/handlers/formats/csv_handler.py"""
import csv
from unittest.mock import patch
import pytest
import tempfile
from pathlib import Path


class TestCSVHandler:
    """Tests for CSV handler."""
    
    def test_parse_csv_basic(self, temp_output_dir, mock_coordinator):
        """Test basic CSV parsing."""
        from webapp.parser.handlers.formats.csv_handler import parse_csv_election_results
        
        # Create test CSV
        csv_path = temp_output_dir / "test.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Candidate", "Party", "Votes"])
            writer.writeheader()
            writer.writerow({"Candidate": "John Doe", "Party": "Democratic", "Votes": "1000"})
            writer.writerow({"Candidate": "Jane Smith", "Party": "Republican", "Votes": "900"})
        
        headers, data, contest, metadata = parse_csv_election_results(
            str(csv_path),
            session_id="test_session",
            coordinator=mock_coordinator
        )
        
        assert headers is not None
        assert data is not None
        assert contest is not None
        assert metadata is not None
        assert len(data) > 0
    
    def test_parse_csv_with_contest_column(self, temp_output_dir, mock_coordinator):
        """Test CSV parsing with explicit contest column."""
        from webapp.parser.handlers.formats.csv_handler import parse_csv_election_results
        
        csv_path = temp_output_dir / "test_contest.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Contest", "Candidate", "Votes"])
            writer.writeheader()
            writer.writerow({"Contest": "Governor", "Candidate": "John Doe", "Votes": "1000"})
            writer.writerow({"Contest": "Governor", "Candidate": "Jane Smith", "Votes": "900"})
        
        headers, data, contest, metadata = parse_csv_election_results(
            str(csv_path),
            session_id="test_session",
            coordinator=mock_coordinator
        )
        
        assert contest == "Governor"
        assert len(data) == 2

    def test_parse_csv_filters_blank_rows(self, temp_output_dir, mock_coordinator):
        from webapp.parser.handlers.formats.csv_handler import parse_csv_election_results

        csv_path = temp_output_dir / "blank_rows.csv"
        csv_path.write_text(
            "Candidate,Party,Votes\nJohn Doe,Democratic,1000\n,,\nJane Smith,Republican,900\n",
            encoding="utf-8",
        )

        headers, data, contest, metadata = parse_csv_election_results(
            str(csv_path),
            session_id="test_session",
            coordinator=mock_coordinator,
        )

        assert len(data) >= 2
        assert metadata["row_count"] >= 2

    def test_parse_csv_uses_filename_when_contest_not_detected(self, temp_output_dir, mock_coordinator):
        from webapp.parser.handlers.formats.csv_handler import parse_csv_election_results

        csv_path = temp_output_dir / "mystery_results.csv"
        csv_path.write_text("Candidate,Party,Votes\nJohn Doe,Democratic,1000\n", encoding="utf-8")

        with patch("webapp.parser.handlers.formats.csv_handler.detect_contest_titles_from_text", return_value=[]):
            with patch("webapp.parser.handlers.formats.csv_handler.parse_filename_for_location", return_value={}):
                _headers, _data, contest, metadata = parse_csv_election_results(
                    str(csv_path),
                    session_id="test_session",
                    coordinator=mock_coordinator,
                )

        assert contest == "mystery_results"
        assert metadata["contest_selection_mode"] == "single_detected"

    def test_parse_csv_returns_error_when_no_contest_selected(self, temp_output_dir, mock_coordinator):
        from webapp.parser.handlers.formats.csv_handler import parse_csv_election_results

        csv_path = temp_output_dir / "no_selection.csv"
        csv_path.write_text("Candidate,Party,Votes\nJohn Doe,Democratic,1000\n", encoding="utf-8")

        with patch("webapp.parser.handlers.formats.csv_handler.select_contest_auto_first", return_value=None):
            with patch.dict("os.environ", {"SMART_ELECTIONS_FORCE_CONTEST_PROMPT": "1"}, clear=False):
                headers, data, contest, metadata = parse_csv_election_results(
                    str(csv_path),
                    session_id="test_session",
                    coordinator=mock_coordinator,
                )

        assert headers == []
        assert data == []
        assert contest == ""
        assert metadata["error"] == "No contest selected"
