from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from webapp.parser.services.workflow_actions import (
    WORKFLOW_DL2_CLAIM_CONTRACT,
    WorkflowActionConflict,
    claim_second_workflow_pass,
)
from webapp.parser.utils.models import (
    Base,
    WorkflowComparison,
    WorkflowEvent,
    WorkflowItem,
    WorkflowPass,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ACTIONS_PATH = REPO_ROOT / "webapp" / "parser" / "services" / "workflow_actions.py"
APP_PATH = REPO_ROOT / "webapp" / "Smart_Elections_Parser_Webapp.py"
BLUEPRINT_PATH = (
    REPO_ROOT / "webapp" / "parser" / "routes" / "workflow_contributor_blueprint.py"
)


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


def _seed_eligible(session, *, dl1_principal="cert:dl1"):
    item = WorkflowItem(
        id=uuid4(),
        lifecycle_state="active",
        current_stage="independent_acquisition",
        stage_condition="ready",
        priority=0,
        election_year=2024,
        state="Iowa",
        jurisdiction_name=None,
        jurisdiction_type=None,
        contest="President",
        office_basic="President",
        source_race_id="2024PRESIA",
        source_url="https://sos.example.gov/results.pdf",
        workflow_metadata={},
        row_version=3,
    )
    session.add(item)
    session.flush()

    dl1 = WorkflowPass(
        workflow_item_id=item.id,
        pass_number=1,
        pass_label="DL1",
        revision_number=1,
        is_current=True,
        status="submitted",
        assigned_principal=dl1_principal,
        source_evidence_ref="workflow://source/dl1",
        staging_batch_id=None,
        candidate_check_status="complete",
        candidate_check_result={"server_owned": True},
        semantic_validation_status="complete",
        semantic_validation_result={"server_owned": True},
        started_at=datetime(2026, 9, 8, 23, 0, tzinfo=timezone.utc),
        submitted_at=datetime(2026, 9, 8, 23, 30, tzinfo=timezone.utc),
    )
    session.add(dl1)
    session.commit()
    return item, dl1


def test_dl2_claim_contract_transaction_local_and_audited(db_session):
    item, dl1 = _seed_eligible(db_session)
    timestamp = datetime(2026, 9, 9, 2, 30, tzinfo=timezone.utc)

    payload = claim_second_workflow_pass(
        db_session,
        item.id,
        principal="cert:dl2",
        expected_row_version=3,
        now=timestamp,
    )

    assert payload["contract"] == WORKFLOW_DL2_CLAIM_CONTRACT
    assert payload["pass_number"] == 2
    assert payload["pass_label"] == "DL2"
    assert payload["revision_number"] == 1
    assert payload["status"] == "in_progress"
    assert payload["row_version"] == 4
    assert payload["committed"] is False
    assert payload["comparison_created"] is False
    assert payload["strict_comparison_stage_advanced"] is False
    assert payload["dl1_pass_id"] == str(dl1.id)

    refreshed = db_session.get(WorkflowItem, item.id)
    assert refreshed.lifecycle_state == "active"
    assert refreshed.current_stage == "independent_acquisition"
    assert refreshed.stage_condition == "in_progress"
    assert refreshed.row_version == 4

    dl2 = (
        db_session.query(WorkflowPass)
        .filter(
            WorkflowPass.workflow_item_id == item.id,
            WorkflowPass.pass_number == 2,
            WorkflowPass.is_current.is_(True),
        )
        .one()
    )
    assert dl2.assigned_principal == "cert:dl2"
    assert dl2.source_evidence_ref is None
    assert dl2.staging_batch_id is None
    assert dl2.candidate_check_status is None
    assert dl2.candidate_check_result is None
    assert dl2.semantic_validation_status is None
    assert dl2.semantic_validation_result is None
    assert dl2.submitted_at is None

    event = (
        db_session.query(WorkflowEvent)
        .filter(
            WorkflowEvent.workflow_item_id == item.id,
            WorkflowEvent.related_pass_id == dl2.id,
        )
        .one()
    )
    assert event.event_type == "pass_claimed"
    assert event.stage == "independent_acquisition"
    assert event.event_metadata["contract"] == WORKFLOW_DL2_CLAIM_CONTRACT
    assert event.event_metadata["pass_number"] == 2
    assert event.event_metadata["pass_label"] == "DL2"
    assert event.event_metadata["dl1_pass_id"] == str(dl1.id)
    assert event.event_metadata["principal_independence_enforced"] is True


def test_dl2_claim_rejects_same_principal(db_session):
    item, _ = _seed_eligible(db_session, dl1_principal="cert:same")
    with pytest.raises(WorkflowActionConflict):
        claim_second_workflow_pass(
            db_session,
            item.id,
            principal="cert:same",
            expected_row_version=3,
        )


def test_dl2_claim_requires_completed_dl1_validations(db_session):
    item, dl1 = _seed_eligible(db_session)
    dl1.semantic_validation_status = "pending"
    db_session.commit()

    with pytest.raises(WorkflowActionConflict):
        claim_second_workflow_pass(
            db_session,
            item.id,
            principal="cert:dl2",
            expected_row_version=3,
        )


def test_dl2_claim_fails_closed_on_row_version_and_ready_state(db_session):
    item, _ = _seed_eligible(db_session)

    with pytest.raises(WorkflowActionConflict):
        claim_second_workflow_pass(
            db_session,
            item.id,
            principal="cert:dl2",
            expected_row_version=99,
        )
    db_session.rollback()

    item = db_session.get(WorkflowItem, item.id)
    item.stage_condition = "in_progress"
    db_session.commit()

    with pytest.raises(WorkflowActionConflict):
        claim_second_workflow_pass(
            db_session,
            item.id,
            principal="cert:dl2",
            expected_row_version=3,
        )


def test_dl2_claim_rejects_existing_current_dl2(db_session):
    item, _ = _seed_eligible(db_session)
    db_session.add(
        WorkflowPass(
            workflow_item_id=item.id,
            pass_number=2,
            pass_label="DL2",
            revision_number=1,
            is_current=True,
            status="in_progress",
            assigned_principal="cert:other",
        )
    )
    db_session.commit()

    with pytest.raises(WorkflowActionConflict):
        claim_second_workflow_pass(
            db_session,
            item.id,
            principal="cert:new",
            expected_row_version=3,
        )


def test_dl2_claim_preserves_dl1_and_creates_no_comparison(db_session):
    item, dl1 = _seed_eligible(db_session)
    before = {
        "status": dl1.status,
        "assigned_principal": dl1.assigned_principal,
        "source_evidence_ref": dl1.source_evidence_ref,
        "candidate_check_status": dl1.candidate_check_status,
        "candidate_check_result": deepcopy(dl1.candidate_check_result),
        "semantic_validation_status": dl1.semantic_validation_status,
        "semantic_validation_result": deepcopy(dl1.semantic_validation_result),
        "submitted_at": dl1.submitted_at,
    }

    claim_second_workflow_pass(
        db_session,
        item.id,
        principal="cert:dl2",
        expected_row_version=3,
    )

    refreshed_dl1 = db_session.get(WorkflowPass, dl1.id)
    after = {
        "status": refreshed_dl1.status,
        "assigned_principal": refreshed_dl1.assigned_principal,
        "source_evidence_ref": refreshed_dl1.source_evidence_ref,
        "candidate_check_status": refreshed_dl1.candidate_check_status,
        "candidate_check_result": refreshed_dl1.candidate_check_result,
        "semantic_validation_status": refreshed_dl1.semantic_validation_status,
        "semantic_validation_result": refreshed_dl1.semantic_validation_result,
        "submitted_at": refreshed_dl1.submitted_at,
    }
    assert after == before
    assert db_session.query(WorkflowComparison).count() == 0


def test_dl2_claim_source_route_and_composition_contract():
    actions = ACTIONS_PATH.read_text(encoding="utf-8")
    app = APP_PATH.read_text(encoding="utf-8")
    blueprint = BLUEPRINT_PATH.read_text(encoding="utf-8")

    assert 'WORKFLOW_DL2_CLAIM_CONTRACT = "workflow_dl2_claim_operation_v1"' in actions
    assert "def claim_second_workflow_pass(" in actions
    assert "assert_dl2_claimable(" in actions
    assert "session.commit(" not in actions
    assert "/passes/2/claim" in blueprint
    assert 'methods=["POST"]' in blueprint
    assert "api_workflow_v1_claim_second_pass" in blueprint
    assert "CAP_DL2_CLAIM" in app
    assert "_workflow_contributor_authority(CAP_DL2_CLAIM)" in app
    assert "claim_second_workflow_pass(" in app
    assert 'set(body) != {"expected_row_version"}' in app
    assert "WORKFLOW_CONTRIBUTOR_MUTATIONS_ENABLED" in app
