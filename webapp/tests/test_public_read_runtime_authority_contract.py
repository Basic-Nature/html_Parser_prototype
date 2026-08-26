from __future__ import annotations

from pathlib import Path

from webapp.parser.services import public_read_runtime as runtime


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_scope_directory_preserves_jurisdiction_type_and_county_compatibility():
    payload = runtime.build_scope_directory_payload(
        [
            ("Arizona", "Pima", "county"),
            ("Arizona", "Congressional District 7", "district"),
            ("Arizona", "Pima", "county"),
            ("New York", "Rockland", "county"),
        ],
        [2024, 2022, 2024],
        ["President", "U.S. House", "President"],
    )

    assert payload["success"] is True
    assert payload["authority"] == "canonical_production"
    assert payload["counties"]["Arizona"] == ["Pima"]
    assert "Congressional District 7" not in payload["counties"]["Arizona"]
    assert {
        "name": "Congressional District 7",
        "type": "district",
    } in payload["jurisdictions"]["Arizona"]
    assert payload["total_counties"] == 2
    assert payload["total_jurisdictions"] == 3
    assert payload["years"] == [2024, 2022]


def test_unavailable_operational_values_are_null_not_zero():
    worklist = runtime._legacy_worklist_unavailable()
    stats = runtime._unavailable_operational_stats(
        {"result_rows": 226042, "races": 10}
    )

    assert worklist["available"] is False
    assert worklist["total"] is None
    assert worklist["records"] == []
    assert worklist["visibility"] == "public_projection"

    assert stats["available"] is False
    assert stats["stats"]["total_races"] is None
    assert stats["stats"]["dl1_ready"] is None
    assert stats["stats"]["production_records"] == 226042
    assert stats["semantic_contract"]["zero"] == "numeric_zero_only"


def test_public_runtime_has_no_google_or_default_sqlite_fallback():
    source = (
        REPO_ROOT
        / "webapp/parser/services/public_read_runtime.py"
    ).read_text(encoding="utf-8")

    assert "google_sheets_client" not in source
    assert "sqlite:///election_data.db" not in source
    assert "DATABASE_URL" not in source
    assert '"e7b2c4d91f60"' in source


def test_public_routes_use_runtime_readers_and_protected_routes_stay_delegated():
    election_routes = (
        REPO_ROOT
        / "webapp/parser/routes/election_data_blueprint.py"
    ).read_text(encoding="utf-8")
    workflow_routes = (
        REPO_ROOT
        / "webapp/parser/routes/workflow_blueprint.py"
    ).read_text(encoding="utf-8")

    assert "read_public_scope_directory" in election_routes
    assert "read_public_election_stats" in election_routes
    assert "read_public_worklist" in election_routes
    assert '"election_data_worklist_public_projection"' in election_routes

    assert "read_public_workflow_facets" in workflow_routes
    assert "read_public_workflow_stats" in workflow_routes
    assert 'return _call_handler("api_workflow_v1_items")' in workflow_routes
    assert '"api_election_data_worklist_overview"' in election_routes
    assert 'return _call_handler("api_election_data_worklist_overview")' in election_routes
