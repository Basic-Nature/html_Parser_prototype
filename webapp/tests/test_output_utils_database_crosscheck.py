"""Unit tests for database cross-check logic in output utilities."""

from __future__ import annotations

import os

from webapp.parser.utils.output_utils import (
    _build_database_cross_check,
    _cross_check_profile_for_source,
    _should_fail_database_cross_check,
)


class TestBuildDatabaseCrossCheck:
    def test_no_reference_returns_unavailable(self):
        result = _build_database_cross_check(
            source_url="https://example.com",
            headers=["Candidate", "Votes"],
            rows=[{"Candidate": "Alice", "Votes": "10"}],
            contest="Mayor",
            state="CA",
            county="Los Angeles",
            reference_source=None,
            reference_metadata=None,
        )
        assert result["status"] == "unavailable"
        assert result["mismatches"] == []

    def test_match_when_counts_and_labels_align(self):
        result = _build_database_cross_check(
            source_url="https://example.com",
            headers=["Candidate", "Votes"],
            rows=[{"Candidate": "Alice", "Votes": "10"}, {"Candidate": "Bob", "Votes": "20"}],
            contest="Mayor",
            state="CA",
            county="Los Angeles",
            reference_source="warehouse",
            reference_metadata={
                "contest": "Mayor",
                "state": "CA",
                "county": "Los Angeles",
                "row_count": 2,
                "candidate_count": 2,
            },
        )
        assert result["status"] == "match"
        assert result["mismatches"] == []

    def test_mismatch_detected_for_large_row_count_delta(self):
        result = _build_database_cross_check(
            source_url="https://example.com",
            headers=["Candidate", "Votes"],
            rows=[{"Candidate": "Alice", "Votes": "10"}],
            contest="Mayor",
            state="CA",
            county="Los Angeles",
            reference_source="warehouse",
            reference_metadata={
                "contest": "Mayor",
                "state": "CA",
                "county": "Los Angeles",
                "row_count": 9,
                "candidate_count": 1,
            },
        )
        assert result["status"] == "mismatch"
        assert any(m.get("field") == "row_count" for m in result["mismatches"])

    def test_candidate_count_falls_back_to_candidate_columns(self):
        result = _build_database_cross_check(
            source_url="https://example.com",
            headers=["Alice - Total Votes", "Bob - Total Votes"],
            rows=[{"Alice - Total Votes": "10", "Bob - Total Votes": "20"}],
            contest="Mayor",
            state="CA",
            county="Los Angeles",
            reference_source="warehouse",
            reference_metadata={
                "contest": "Mayor",
                "state": "CA",
                "county": "Los Angeles",
                "row_count": 1,
                "candidate_count": 2,
            },
        )
        assert result["status"] == "match"
        assert result["extracted"]["candidate_count"] == 2

    def test_label_mismatch_detected(self):
        result = _build_database_cross_check(
            source_url="https://example.com",
            headers=["Candidate", "Votes"],
            rows=[{"Candidate": "Alice", "Votes": "10"}],
            contest="Governor",
            state="CA",
            county="Los Angeles",
            reference_source="warehouse",
            reference_metadata={
                "contest": "Mayor",
                "state": "CA",
                "county": "Orange",
                "row_count": 1,
                "candidate_count": 1,
            },
        )
        assert result["status"] == "mismatch"
        fields = {m.get("field") for m in result["mismatches"]}
        assert "contest" in fields
        assert "county" in fields

    def test_verified_datasets_stricter_row_delta_than_warehouse(self):
        rows = [{"Candidate": "Alice", "Votes": "10"}, {"Candidate": "Bob", "Votes": "20"}]
        reference = {
            "contest": "Mayor",
            "state": "CA",
            "county": "Los Angeles",
            "row_count": 4,
            "candidate_count": 2,
        }

        strict_result = _build_database_cross_check(
            source_url="https://example.com",
            headers=["Candidate", "Votes"],
            rows=rows,
            contest="Mayor",
            state="CA",
            county="Los Angeles",
            reference_source="verified_datasets",
            reference_metadata=reference,
        )
        relaxed_result = _build_database_cross_check(
            source_url="https://example.com",
            headers=["Candidate", "Votes"],
            rows=rows,
            contest="Mayor",
            state="CA",
            county="Los Angeles",
            reference_source="warehouse",
            reference_metadata=reference,
        )

        assert strict_result["status"] == "mismatch"
        assert relaxed_result["status"] == "match"

    def test_profile_env_override_applied(self):
        os.environ["DB_CROSSCHECK_WAREHOUSE_ROW_DELTA_ABS"] = "0"
        os.environ["DB_CROSSCHECK_WAREHOUSE_ROW_DELTA_RATIO"] = "0.0"
        try:
            result = _build_database_cross_check(
                source_url="https://example.com",
                headers=["Candidate", "Votes"],
                rows=[{"Candidate": "Alice", "Votes": "10"}],
                contest="Mayor",
                state="CA",
                county="Los Angeles",
                reference_source="warehouse",
                reference_metadata={
                    "contest": "Mayor",
                    "state": "CA",
                    "county": "Los Angeles",
                    "row_count": 2,
                    "candidate_count": 1,
                },
            )
            assert result["status"] == "mismatch"
            assert result["profile"]["row_delta_abs"] == 0
        finally:
            os.environ.pop("DB_CROSSCHECK_WAREHOUSE_ROW_DELTA_ABS", None)
            os.environ.pop("DB_CROSSCHECK_WAREHOUSE_ROW_DELTA_RATIO", None)


class TestCrossCheckGate:
    def test_context_flag_overrides_environment(self):
        os.environ["DB_CROSSCHECK_FAIL_ON_MISMATCH"] = "false"
        try:
            assert _should_fail_database_cross_check({"database_cross_check_fail_on_mismatch": True}) is True
            assert _should_fail_database_cross_check({"database_cross_check_fail_on_mismatch": False}) is False
        finally:
            os.environ.pop("DB_CROSSCHECK_FAIL_ON_MISMATCH", None)

    def test_environment_flag_used_when_context_missing(self):
        os.environ["DB_CROSSCHECK_FAIL_ON_MISMATCH"] = "true"
        try:
            assert _should_fail_database_cross_check({}) is True
        finally:
            os.environ.pop("DB_CROSSCHECK_FAIL_ON_MISMATCH", None)


class TestCrossCheckProfile:
    def test_default_profile_fields_present(self):
        profile = _cross_check_profile_for_source(None)
        assert "row_delta_abs" in profile
        assert "row_delta_ratio" in profile
        assert "candidate_delta_abs" in profile
