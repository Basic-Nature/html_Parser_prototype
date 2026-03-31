"""Tests for centralized database comparison decision and cross-check helpers."""

from __future__ import annotations

import os

from webapp.parser.utils import database_comparison as dc


class TestEvaluateUrlProcessingPolicy:
    def test_skip_database_check_processes_url(self):
        decision = dc.evaluate_url_processing_policy(
            "https://example.com/results",
            skip_database_check=True,
        )

        assert decision["should_skip"] is False
        assert decision["decision"] == "database_check_disabled"
        assert decision["checked"] is False

    def test_force_reparse_overrides_existing_data(self, monkeypatch):
        monkeypatch.setattr(
            dc,
            "check_existing_finalized_data",
            lambda *args, **kwargs: (True, "warehouse", {"state": "NY"}),
        )

        decision = dc.evaluate_url_processing_policy(
            "https://example.com/results",
            force_reparse=True,
        )

        assert decision["should_skip"] is False
        assert decision["decision"] == "force_reparse"
        assert decision["checked"] is False

    def test_existing_data_skips_url(self, monkeypatch):
        monkeypatch.setattr(
            dc,
            "check_existing_finalized_data",
            lambda *args, **kwargs: (True, "verified_datasets", {"contest": "Mayor"}),
        )

        decision = dc.evaluate_url_processing_policy("https://example.com/results")

        assert decision["checked"] is True
        assert decision["should_skip"] is True
        assert decision["decision"] == "skipped_data_exists"
        assert decision["data_source"] == "verified_datasets"
        assert decision["metadata"]["contest"] == "Mayor"

    def test_no_existing_data_processes_url(self, monkeypatch):
        monkeypatch.setattr(
            dc,
            "check_existing_finalized_data",
            lambda *args, **kwargs: (False, None, None),
        )

        decision = dc.evaluate_url_processing_policy("https://example.com/results")

        assert decision["checked"] is True
        assert decision["should_skip"] is False
        assert decision["decision"] == "process"


class TestCentralizedCrossCheckHelpers:
    def test_cross_check_profile_env_override(self):
        os.environ["DB_CROSSCHECK_WAREHOUSE_ROW_DELTA_ABS"] = "0"
        os.environ["DB_CROSSCHECK_WAREHOUSE_ROW_DELTA_RATIO"] = "0.0"
        try:
            profile = dc.cross_check_profile_for_source("warehouse")
            assert profile["row_delta_abs"] == 0
            assert profile["row_delta_ratio"] == 0.0
        finally:
            os.environ.pop("DB_CROSSCHECK_WAREHOUSE_ROW_DELTA_ABS", None)
            os.environ.pop("DB_CROSSCHECK_WAREHOUSE_ROW_DELTA_RATIO", None)

    def test_should_fail_database_cross_check_context_override(self):
        os.environ["DB_CROSSCHECK_FAIL_ON_MISMATCH"] = "false"
        try:
            assert dc.should_fail_database_cross_check({"database_cross_check_fail_on_mismatch": True}) is True
            assert dc.should_fail_database_cross_check({"database_cross_check_fail_on_mismatch": False}) is False
        finally:
            os.environ.pop("DB_CROSSCHECK_FAIL_ON_MISMATCH", None)

    def test_build_database_cross_check_mismatch(self):
        result = dc.build_database_cross_check(
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
