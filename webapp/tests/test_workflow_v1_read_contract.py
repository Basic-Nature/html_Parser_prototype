"""Contracts for the read-only workflow_v1 operational API."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from flask import Flask
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from webapp.parser.routes.workflow_blueprint import (
    create_workflow_v1_blueprint,
)
from webapp.parser.services.workflow_reader import (
    WORKFLOW_READ_SCHEMA_VERSION,
    WorkflowReadValidationError,
    read_workflow_facets,
    read_workflow_item_detail,
    read_workflow_items,
    read_workflow_stats,
)
from webapp.parser.utils.models import (
    Base,
    WorkflowDiscrepancy,
    WorkflowItem,
    WorkflowPass,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVICE_PATH = (
    REPO_ROOT
    / "webapp"
    / "parser"
    / "services"
    / "workflow_reader.py"
)
BLUEPRINT_PATH = (
    REPO_ROOT
    / "webapp"
    / "parser"
    / "routes"
    / "workflow_blueprint.py"
)
APP_PATH = REPO_ROOT / "webapp" / "Smart_Elections_Parser_Webapp.py"


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _seed_items(session):
    first = WorkflowItem(
        lifecycle_state="active",
        current_stage="independent_acquisition",
        stage_condition="in_progress",
        priority=5,
        election_year=2024,
        state="Arizona",
        jurisdiction_name="Pima",
        jurisdiction_type="county",
        contest="President",
        source_race_id="AZ-2024-PRES",
        source_url="https://example.invalid/az",
        workflow_metadata={
            "unknown_component": None,
            "signed_example": -4,
        },
    )
    second = WorkflowItem(
        lifecycle_state="blocked",
        current_stage="source_intake",
        stage_condition="pending",
        priority=9,
        election_year=2024,
        state="Arizona",
        jurisdiction_name=None,
        jurisdiction_type=None,
        contest="US Senate",
        source_race_id="AZ-2024-SEN",
        blocked_reason_code="source_missing",
        workflow_metadata={},
    )
    session.add_all([first, second])
    session.flush()

    pass_one = WorkflowPass(
        workflow_item_id=first.id,
        pass_number=1,
        pass_label="DL1",
        revision_number=1,
        is_current=True,
        status="submitted",
        assigned_principal="analyst-a",
        semantic_validation_result={
            "reported_value": None,
            "signed_value": -4,
        },
    )
    session.add(pass_one)
    session.commit()
    return first, second, pass_one


def test_list_contract_preserves_null_signed_and_precise_jurisdiction(
    db_session,
) -> None:
    first, _, _ = _seed_items(db_session)

    payload = read_workflow_items(
        db_session,
        {"state": "Arizona", "contest": "President"},
    )

    assert payload["success"] is True
    assert payload["schema_version"] == WORKFLOW_READ_SCHEMA_VERSION
    assert payload["authority"] == {
        "kind": "operational_workflow",
        "canonical": False,
        "source": "postgresql",
        "read_only": True,
        "lineage_inferred": False,
    }
    assert payload["pagination"]["total"] == 1

    row = payload["items"][0]
    assert row["id"] == str(first.id)
    assert row["scope"]["jurisdiction_name"] == "Pima"
    assert row["scope"]["jurisdiction_type"] == "county"
    assert row["workflow_metadata"]["unknown_component"] is None
    assert row["workflow_metadata"]["signed_example"] == -4
    assert row["canonical_reference"]["race_id"] is None
    assert row["canonical_reference"]["lineage_inferred"] is False


def test_list_does_not_accept_county_as_jurisdiction_alias(
    db_session,
) -> None:
    _seed_items(db_session)

    payload = read_workflow_items(
        db_session,
        {"county": "Pima"},
    )

    assert "county" not in payload["filters"]
    assert payload["pagination"]["total"] == 2


def test_facets_are_self_excluding_and_preserve_null_values(
    db_session,
) -> None:
    _seed_items(db_session)

    payload = read_workflow_facets(
        db_session,
        {"contest": "President"},
    )

    assert payload["facet_mode"] == "self_excluding"
    assert "jurisdiction" in payload["axes"]

    contest_values = {
        row["value"]: row["count"]
        for row in payload["facets"]["contest"]
    }
    assert contest_values == {
        "President": 1,
        "US Senate": 1,
    }

    jurisdiction_values = payload["facets"]["jurisdiction"]
    assert {
        "value": {"name": "Pima", "type": "county"},
        "count": 1,
    } in jurisdiction_values


def test_stats_return_operational_counts_only(db_session) -> None:
    _seed_items(db_session)

    payload = read_workflow_stats(
        db_session,
        {"state": "Arizona"},
    )

    assert payload["total"] == 2
    assert payload["action_counts"]["blocked"] == 1
    assert payload["action_counts"]["ready_for_publication"] == 0
    assert payload["action_counts"]["published"] == 0


def test_detail_returns_passes_without_reinterpreting_values(
    db_session,
) -> None:
    first, _, pass_one = _seed_items(db_session)

    payload = read_workflow_item_detail(db_session, first.id)

    assert payload is not None
    assert payload["item"]["id"] == str(first.id)
    assert len(payload["passes"]) == 1
    assert payload["passes"][0]["id"] == str(pass_one.id)
    assert (
        payload["passes"][0]["semantic_validation_result"][
            "reported_value"
        ]
        is None
    )
    assert (
        payload["passes"][0]["semantic_validation_result"]["signed_value"]
        == -4
    )


def test_detail_preserves_explicit_discrepancy_presence_state(
    db_session,
) -> None:
    first, _, pass_one = _seed_items(db_session)
    other_pass = WorkflowPass(
        workflow_item_id=first.id,
        pass_number=2,
        pass_label="DL2",
        revision_number=1,
        is_current=True,
        status="submitted",
    )
    db_session.add(other_pass)
    db_session.flush()

    from webapp.parser.utils.models import WorkflowComparison

    comparison = WorkflowComparison(
        workflow_item_id=first.id,
        left_pass_id=pass_one.id,
        right_pass_id=other_pass.id,
        comparison_version=1,
        status="differences",
        strict_equality_passed=False,
        difference_count=1,
    )
    db_session.add(comparison)
    db_session.flush()

    discrepancy = WorkflowDiscrepancy(
        comparison_id=comparison.id,
        workflow_item_id=first.id,
        category="null_vs_zero",
        semantic_key={"candidate": "Example"},
        left_value=None,
        right_value=0,
        left_value_state="missing",
        right_value_state="reported",
        resolution_status="open",
    )
    db_session.add(discrepancy)
    db_session.commit()

    payload = read_workflow_item_detail(db_session, first.id)
    assert payload is not None
    row = payload["discrepancies"][0]

    assert row["left_value"] is None
    assert row["right_value"] == 0
    assert row["left_value_state"] == "missing"
    assert row["right_value_state"] == "reported"


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"limit": "0"}, "limit"),
        ({"offset": "-1"}, "offset"),
        ({"year": "not-a-year"}, "year"),
        ({"canonical_linked": "maybe"}, "canonical_linked"),
    ],
)
def test_invalid_read_parameters_fail_closed(
    db_session,
    params,
    message,
) -> None:
    with pytest.raises(WorkflowReadValidationError) as excinfo:
        read_workflow_items(db_session, params)
    assert message in str(excinfo.value)


def test_blueprint_dispatch_contract() -> None:
    app = Flask(__name__)
    app.config["_WORKFLOW_V1_ROUTE_HANDLERS"] = {
        "api_workflow_v1_items": lambda: (
            {"success": True, "kind": "items"},
            200,
        ),
        "api_workflow_v1_item_detail": (
            lambda item_id: (
                {
                    "success": True,
                    "kind": "detail",
                    "item_id": str(item_id),
                },
                200,
            )
        ),
        "api_workflow_v1_facets": lambda: (
            {"success": True, "kind": "facets"},
            200,
        ),
        "api_workflow_v1_stats": lambda: (
            {"success": True, "kind": "stats"},
            200,
        ),
    }
    app.register_blueprint(create_workflow_v1_blueprint())
    client = app.test_client()

    assert client.get("/api/workflow/v1/items").status_code == 200
    assert client.get("/api/workflow/v1/facets").status_code == 200
    assert client.get("/api/workflow/v1/stats").status_code == 200

    item_id = uuid4()
    response = client.get(f"/api/workflow/v1/items/{item_id}")
    assert response.status_code == 200
    assert response.get_json()["item_id"] == str(item_id)


def test_blueprint_returns_structured_500_when_unconfigured() -> None:
    app = Flask(__name__)
    app.register_blueprint(create_workflow_v1_blueprint())
    client = app.test_client()

    response = client.get("/api/workflow/v1/items")
    assert response.status_code == 500
    assert response.get_json() == {
        "error": "Workflow v1 routes are not configured."
    }


def test_reader_has_no_runtime_legacy_or_write_authority() -> None:
    source = SERVICE_PATH.read_text(encoding="utf-8")
    lowered = source.lower()

    forbidden_authorities = (
        "google_sheets",
        "db_lite",
        "warehouse_election_results",
    )
    for token in forbidden_authorities:
        assert token not in lowered

    forbidden_writes = (
        ".commit(",
        ".add(",
        ".add_all(",
        ".delete(",
        ".update(",
        "insert(",
    )
    for token in forbidden_writes:
        assert token not in source

    assert "SET TRANSACTION READ ONLY" in source


def test_app_registers_workflow_v1_as_separate_read_plane() -> None:
    source = APP_PATH.read_text(encoding="utf-8")

    assert (
        "from webapp.parser.routes.workflow_blueprint "
        "import create_workflow_v1_blueprint"
    ) in source
    assert (
        "from webapp.parser.services.workflow_reader import ("
    ) in source
    assert (
        "app.register_blueprint(create_workflow_v1_blueprint())"
    ) in source
    assert 'app.config["_WORKFLOW_V1_ROUTE_HANDLERS"]' in source

    for handler in (
        "api_workflow_v1_items",
        "api_workflow_v1_item_detail",
        "api_workflow_v1_facets",
        "api_workflow_v1_stats",
    ):
        assert f'"{handler}": {handler}' in source


def test_workflow_v1_source_does_not_modify_frontend_authorities() -> None:
    blueprint_source = BLUEPRINT_PATH.read_text(encoding="utf-8")
    assert "/api/workflow/v1/" in blueprint_source

    for legacy_route in (
        "/api/election_data/worklist",
        "/api/election_data/db_lite/finalized",
        "/api/election_data/db_lite/down_ballot",
    ):
        assert legacy_route not in blueprint_source

def test_workflow_v1_is_excluded_from_legacy_endpoint_alias_backfill() -> None:
    source = APP_PATH.read_text(encoding="utf-8")

    assert (
        'legacy_alias_excluded_namespaces = {"workflow_v1_routes"}'
        in source
    )
    assert (
        "if endpoint_namespace in legacy_alias_excluded_namespaces:"
        in source
    )
    assert (
        "if namespaced_endpoint_namespace in legacy_alias_excluded_namespaces:"
        in source
    )
