from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from flask import Flask
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from webapp.parser.auth.capability_policy import (
    Capability,
    CapabilityPolicyError,
    assert_fresh_proof_review,
    assert_trusted_action,
)
from webapp.parser.routes.workflow_contributor_blueprint import (
    create_workflow_contributor_blueprint,
)
from webapp.parser.services.workflow_actions import (
    WorkflowActionConflict,
    WorkflowSourceAccessDenied,
    WorkflowSourceNotApproved,
    assert_independent_second_pass,
    claim_first_workflow_pass,
    read_approved_workflow_source,
)
from webapp.parser.utils.models import (
    Base,
    WorkflowEvent,
    WorkflowItem,
    WorkflowPass,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_PATH = REPO_ROOT / "webapp" / "Smart_Elections_Parser_Webapp.py"


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )
    session = Session()
    try:
        yield session
    finally:
        session.rollback()
        session.close()
        engine.dispose()


def _authority(state: str, *, fresh: bool = False):
    return {
        "state": state,
        "authenticated": state != "anonymous",
        "fresh_proof": fresh,
    }


def _seed_item(
    session,
    *,
    source_url: str = "https://sos.example.gov/results.pdf",
):
    row = WorkflowItem(
        id=uuid4(),
        lifecycle_state="queued",
        current_stage="source_intake",
        stage_condition="pending",
        priority=0,
        election_year=2024,
        election_date=None,
        state="Iowa",
        jurisdiction_name=None,
        jurisdiction_type=None,
        contest="President",
        office_basic="President",
        election_type=None,
        source_race_id="2024PRESIA",
        source_url=source_url,
        canonical_race_id=None,
        blocked_reason_code=None,
        blocker_detail=None,
        created_by_principal=None,
        workflow_metadata={},
        row_version=1,
    )
    session.add(row)
    session.commit()
    return row


def test_trusted_action_requires_authenticated_authority_not_tier_alone():
    with pytest.raises(CapabilityPolicyError):
        assert_trusted_action(_authority("anonymous"), 0)

    assert (
        assert_trusted_action(
            _authority("fresh_certificate", fresh=True),
            0,
        )
        is Capability.TRUSTED_ACTION
    )
    assert (
        assert_trusted_action(
            _authority("certificate_session"),
            0,
        )
        is Capability.TRUSTED_ACTION
    )

    with pytest.raises(CapabilityPolicyError):
        assert_trusted_action(_authority("development_bypass"), 3)


def test_fresh_review_requires_fresh_certificate_and_reviewer_tier():
    with pytest.raises(CapabilityPolicyError):
        assert_fresh_proof_review(_authority("certificate_session"), 3)

    with pytest.raises(CapabilityPolicyError):
        assert_fresh_proof_review(
            _authority("fresh_certificate", fresh=True),
            0,
        )

    assert (
        assert_fresh_proof_review(
            _authority("fresh_certificate", fresh=True),
            1,
        )
        is Capability.FRESH_PROOF_REVIEW
    )


def test_approved_source_requires_exact_curated_registry_entry(
    db_session,
    tmp_path,
):
    registry = tmp_path / "urls.txt"
    registry.write_text(
        "\n".join(
            [
                "# === Curated | test ===",
                (
                    "2024\tPresident\tIowa\tstatewide\tPDF\tCertified\t"
                    "https://sos.example.gov/results.pdf"
                ),
                "# === Backlog ===",
                (
                    "2020\tPresident\tIowa\tstatewide\tPDF\tArchive\t"
                    "https://sos.example.gov/archive.pdf"
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    item = _seed_item(db_session)
    payload = read_approved_workflow_source(
        db_session,
        item.id,
        principal="cert:reader",
        registry_path=registry,
    )
    assert payload["registry_category"] == "curated"
    assert payload["source_url"] == "https://sos.example.gov/results.pdf"
    assert payload["source_url_editable"] is False
    assert payload["arbitrary_url_execution"] is False

    other = _seed_item(
        db_session,
        source_url="https://sos.example.gov/archive.pdf",
    )
    with pytest.raises(WorkflowSourceNotApproved):
        read_approved_workflow_source(
            db_session,
            other.id,
            principal="cert:reader",
            registry_path=registry,
        )


def test_approved_source_requires_claimable_or_assigned_task(
    db_session,
    tmp_path,
):
    registry = tmp_path / "urls.txt"
    registry.write_text(
        "\n".join(
            [
                "# === Curated | test ===",
                (
                    "2024\tPresident\tIowa\tstatewide\tPDF\tCertified\t"
                    "https://sos.example.gov/results.pdf"
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    item = _seed_item(db_session)
    item.lifecycle_state = "active"
    item.current_stage = "independent_acquisition"
    item.stage_condition = "in_progress"
    item.row_version = 2

    assigned_pass = WorkflowPass(
        workflow_item_id=item.id,
        pass_number=1,
        pass_label="DL1",
        revision_number=1,
        is_current=True,
        status="in_progress",
        assigned_principal="cert:owner",
    )
    db_session.add(assigned_pass)
    db_session.commit()

    allowed = read_approved_workflow_source(
        db_session,
        item.id,
        principal="cert:owner",
        registry_path=registry,
    )
    assert allowed["source_url"] == "https://sos.example.gov/results.pdf"

    with pytest.raises(WorkflowSourceAccessDenied):
        read_approved_workflow_source(
            db_session,
            item.id,
            principal="cert:other",
            registry_path=registry,
        )


def test_first_claim_is_transaction_local_and_audited(db_session):
    item = _seed_item(db_session)
    timestamp = datetime(2026, 8, 28, 1, 23, 45, tzinfo=timezone.utc)

    payload = claim_first_workflow_pass(
        db_session,
        item.id,
        principal="cert:example-fingerprint",
        expected_row_version=1,
        now=timestamp,
    )

    assert payload["committed"] is False
    assert payload["row_version"] == 2

    refreshed = db_session.get(WorkflowItem, item.id)
    assert refreshed.lifecycle_state == "active"
    assert refreshed.current_stage == "independent_acquisition"
    assert refreshed.stage_condition == "in_progress"
    assert refreshed.row_version == 2

    passes = (
        db_session.query(WorkflowPass)
        .filter(WorkflowPass.workflow_item_id == item.id)
        .all()
    )
    assert len(passes) == 1
    assert passes[0].pass_label == "DL1"
    assert passes[0].assigned_principal == "cert:example-fingerprint"

    events = (
        db_session.query(WorkflowEvent)
        .filter(WorkflowEvent.workflow_item_id == item.id)
        .all()
    )
    assert len(events) == 1
    assert events[0].event_type == "pass_claimed"
    assert events[0].related_pass_id == passes[0].id
    assert events[0].prior_state["row_version"] == 1
    assert events[0].new_state["row_version"] == 2

    db_session.rollback()


def test_claim_fails_closed_on_version_and_existing_pass(db_session):
    item = _seed_item(db_session)

    with pytest.raises(WorkflowActionConflict):
        claim_first_workflow_pass(
            db_session,
            item.id,
            principal="cert:one",
            expected_row_version=9,
        )
    db_session.rollback()

    claim_first_workflow_pass(
        db_session,
        item.id,
        principal="cert:one",
        expected_row_version=1,
    )

    with pytest.raises(WorkflowActionConflict):
        claim_first_workflow_pass(
            db_session,
            item.id,
            principal="cert:two",
            expected_row_version=2,
        )

    db_session.rollback()


def test_second_pass_independence_rejects_same_principal():
    with pytest.raises(WorkflowActionConflict):
        assert_independent_second_pass("cert:a", "cert:a")
    assert_independent_second_pass("cert:a", "cert:b")


def test_contributor_blueprint_dispatch_contract():
    app = Flask(__name__)
    item_id = uuid4()

    app.config["_WORKFLOW_CONTRIBUTOR_ROUTE_HANDLERS"] = {
        "api_workflow_v1_contributor_source": (
            lambda item_id: (
                {"success": True, "kind": "source", "item_id": str(item_id)},
                200,
            )
        ),
        "api_workflow_v1_claim_first_pass": (
            lambda item_id: (
                {"success": True, "kind": "claim", "item_id": str(item_id)},
                200,
            )
        ),
    }
    app.register_blueprint(create_workflow_contributor_blueprint())

    client = app.test_client()
    assert client.get(
        f"/api/workflow/v1/contributor/items/{item_id}/source"
    ).status_code == 200
    assert client.post(
        f"/api/workflow/v1/contributor/items/{item_id}/passes/1/claim"
    ).status_code == 200


def test_composition_root_claim_is_default_off_and_authority_guarded():
    source = APP_PATH.read_text(encoding="utf-8")

    assert (
        'os.environ.get("WORKFLOW_CONTRIBUTOR_MUTATIONS_ENABLED", "false")'
        in source
    )
    assert "create_workflow_contributor_blueprint" in source
    assert "api_workflow_v1_claim_first_pass" in source
    assert "api_workflow_v1_contributor_source" in source
    assert "assert_trusted_action" in source
    assert '"expected_row_version" not in body' in source
