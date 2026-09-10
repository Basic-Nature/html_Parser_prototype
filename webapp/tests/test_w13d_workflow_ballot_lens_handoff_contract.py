from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from webapp.parser.contracts.workflow_authorization import (
    ROLE_CONTRIBUTOR,
    ROLE_REVIEWER,
)
from webapp.parser.services.workflow_ballot_lens_handoff import (
    WORKFLOW_BALLOT_LENS_BROWSER_KEYS,
    WORKFLOW_BALLOT_LENS_HANDOFF_CONTRACT,
    WorkflowBallotLensHandoffDenied,
    project_workflow_ballot_lens_handoff,
)
from webapp.parser.utils.models import (
    Base,
    CanonicalElectionRace,
    CanonicalSourceArtifact,
    CanonicalVerificationEvent,
    WorkflowItem,
    WorkflowPass,
)


URL = "https://results.example.gov/2024/general"
PRINCIPAL = "cert:contributor"


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:", future=True)

    @event.listens_for(engine, "connect")
    def _fk(dbapi_connection, _):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(
        bind=engine,
        future=True,
        expire_on_commit=False,
        class_=Session,
    )
    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()
        engine.dispose()


def registry(tmp_path: Path) -> Path:
    path = tmp_path / "urls.txt"
    path.write_text(
        "# === CURATED ===\n"
        "2024\tPresident\tExample\tstatewide\tHTML\taccepted\t"
        + URL
        + "\n",
        encoding="utf-8",
    )
    return path


def seed(db_session: Session):
    now = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
    payload_artifact = CanonicalSourceArtifact(
        artifact_role="canonical_finalized_payload",
        filename="payload.json",
        sha256="1" * 64,
        row_count=10,
        race_count=1,
        provenance={"authority": "accepted_payload"},
    )
    approval_artifact = CanonicalSourceArtifact(
        artifact_role="production_approval_workflow",
        filename="approval.json",
        sha256="2" * 64,
        row_count=1,
        race_count=1,
        provenance={"authority": "accepted_approval"},
    )
    db_session.add_all([payload_artifact, approval_artifact])
    db_session.flush()

    race = CanonicalElectionRace(
        source_race_id="seed-race",
        election_year=2024,
        state="Example",
        contest="President",
        production_status="prod_loaded",
        selected_dl_source="DL1",
        source_url=URL,
        verification_status="verified",
        verified_at=now,
        payload_artifact_id=payload_artifact.id,
        approval_artifact_id=approval_artifact.id,
        qa_metadata={"accepted_seed": True},
    )
    db_session.add(race)
    db_session.flush()

    db_session.add(CanonicalVerificationEvent(
        race_id=race.id,
        stage="production_approval",
        status="approved",
        selected_dl_source="DL1",
        actor="governed-seed",
        occurred_at=now,
        event_metadata={"authority": "accepted_seed_verification"},
    ))

    item = WorkflowItem(
        lifecycle_state="active",
        current_stage="independent_acquisition",
        stage_condition="in_progress",
        election_year=2024,
        state="Example",
        contest="President",
        source_url=URL,
        canonical_race_id=race.id,
        row_version=7,
        workflow_metadata={"source_authority": "maintained_url_registry"},
    )
    db_session.add(item)
    db_session.flush()

    workflow_pass = WorkflowPass(
        workflow_item_id=item.id,
        pass_number=1,
        pass_label="DL1",
        revision_number=1,
        is_current=True,
        status="in_progress",
        assigned_principal=PRINCIPAL,
        started_at=now,
    )
    db_session.add(workflow_pass)
    db_session.flush()
    return item, workflow_pass


def nested_keys(value):
    found = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            found.add(str(key))
            found.update(nested_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            found.update(nested_keys(nested))
    return found


def test_handoff_returns_exact_browser_authority_without_source_or_identity(
    db_session,
    tmp_path,
):
    item, workflow_pass = seed(db_session)

    payload = project_workflow_ballot_lens_handoff(
        db_session,
        item.id,
        principal=PRINCIPAL,
        internal_roles=[ROLE_CONTRIBUTOR],
        registry_path=registry(tmp_path),
    )

    assert payload["success"] is True
    assert payload["contract"] == WORKFLOW_BALLOT_LENS_HANDOFF_CONTRACT
    assert payload["workflow_item_id"] == str(item.id)
    assert payload["workflow_pass_id"] == str(workflow_pass.id)
    assert payload["expected_row_version"] == 7
    assert payload["can_execute_ballot_lens"] is True
    assert payload["principal_disclosed"] is False
    assert payload["source_url_disclosed"] is False
    assert payload["browser_payload_keys"] == list(
        WORKFLOW_BALLOT_LENS_BROWSER_KEYS
    )

    forbidden = {
        "source_url",
        "assigned_principal",
        "reviewer_principal",
        "created_by_principal",
        "roles",
        "capabilities",
        "direct_urls",
        "registry_source_id",
    }
    assert nested_keys(payload).isdisjoint(forbidden)
    assert URL not in repr(payload)


def test_handoff_requires_current_pass_assigned_to_requesting_principal(
    db_session,
    tmp_path,
):
    item, _ = seed(db_session)

    with pytest.raises(
        WorkflowBallotLensHandoffDenied,
        match=r"^Workflow Ballot Lens handoff denied\.$",
    ):
        project_workflow_ballot_lens_handoff(
            db_session,
            item.id,
            principal="cert:other",
            internal_roles=[ROLE_CONTRIBUTOR],
            registry_path=registry(tmp_path),
        )


def test_reviewer_role_does_not_inherit_parser_execution(
    db_session,
    tmp_path,
):
    item, _ = seed(db_session)

    with pytest.raises(WorkflowBallotLensHandoffDenied):
        project_workflow_ballot_lens_handoff(
            db_session,
            item.id,
            principal=PRINCIPAL,
            internal_roles=[ROLE_REVIEWER],
            registry_path=registry(tmp_path),
        )
