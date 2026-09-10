from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from webapp.parser.contracts.workflow_authorization import ROLE_CONTRIBUTOR
from webapp.parser.services.trusted_parser_source_policy import (
    PROVENANCE_LEGACY_PRODUCTION,
)
from webapp.parser.services.workflow_ballot_lens_execution import (
    WorkflowBallotLensExecutionDenied,
    authorize_workflow_ballot_lens_execution,
)
from webapp.parser.services.workflow_ballot_lens_runtime_context import (
    WorkflowBallotLensRuntimeContextDenied,
    build_workflow_ballot_lens_server_context,
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


def registry(tmp_path: Path, section: str = "CURATED") -> Path:
    path = tmp_path / "urls.txt"
    path.write_text(
        "# === " + section + " ===\n"
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

    verification = CanonicalVerificationEvent(
        race_id=race.id,
        stage="production_approval",
        status="approved",
        selected_dl_source="DL1",
        actor="governed-seed",
        occurred_at=now,
        event_metadata={"authority": "accepted_seed_verification"},
    )
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
    db_session.add_all([verification, item])
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
    return item, workflow_pass, race, verification


def request(item, workflow_pass):
    return {
        "workflow_item_id": str(item.id),
        "workflow_pass_id": str(workflow_pass.id),
        "expected_row_version": int(item.row_version),
    }


def test_canonical_bootstrap_is_structured_qc_authority(
    db_session,
    tmp_path,
):
    item, workflow_pass, _, _ = seed(db_session)
    payload = request(item, workflow_pass)
    context = build_workflow_ballot_lens_server_context(
        db_session,
        payload,
        registry_path=registry(tmp_path),
    )

    assert context.source_url == URL
    assert context.qc_evidence["qc_backed"] is True
    assert (
        context.qc_evidence["provenance_class"]
        == PROVENANCE_LEGACY_PRODUCTION
    )
    assert context.registry_state["exact_registry_identity"] is True
    assert context.registry_state["registry_category"] == "curated"
    assert context.registry_state["deprecated"] is False

    authority = authorize_workflow_ballot_lens_execution(
        payload,
        principal=PRINCIPAL,
        internal_roles=[ROLE_CONTRIBUTOR],
        server_context=context,
    )
    assert authority.resolved_source_url == URL
    assert URL not in repr(authority.safe_projection())


def test_bare_verified_status_without_structured_event_is_denied(
    db_session,
    tmp_path,
):
    item, workflow_pass, _, verification = seed(db_session)
    db_session.delete(verification)
    db_session.flush()

    with pytest.raises(
        WorkflowBallotLensRuntimeContextDenied,
        match=r"^Workflow Ballot Lens runtime context denied\.$",
    ):
        build_workflow_ballot_lens_server_context(
            db_session,
            request(item, workflow_pass),
            registry_path=registry(tmp_path),
        )


def test_canonical_approval_artifact_is_required(
    db_session,
    tmp_path,
):
    item, workflow_pass, race, _ = seed(db_session)
    race.qa_metadata = {}
    db_session.flush()

    with pytest.raises(WorkflowBallotLensRuntimeContextDenied):
        build_workflow_ballot_lens_server_context(
            db_session,
            request(item, workflow_pass),
            registry_path=registry(tmp_path),
        )


def test_deprecated_registry_state_cannot_become_execution_authority(
    db_session,
    tmp_path,
):
    item, workflow_pass, _, _ = seed(db_session)
    payload = request(item, workflow_pass)
    context = build_workflow_ballot_lens_server_context(
        db_session,
        payload,
        registry_path=registry(tmp_path, section="DEPRECATED"),
    )
    assert context.registry_state["deprecated"] is True

    with pytest.raises(WorkflowBallotLensExecutionDenied):
        authorize_workflow_ballot_lens_execution(
            payload,
            principal=PRINCIPAL,
            internal_roles=[ROLE_CONTRIBUTOR],
            server_context=context,
        )


def test_item_and_pass_must_be_exact_current_acquisition_context(
    db_session,
    tmp_path,
):
    item, workflow_pass, _, _ = seed(db_session)
    workflow_pass.status = "submitted"
    db_session.flush()

    context = build_workflow_ballot_lens_server_context(
        db_session,
        request(item, workflow_pass),
        registry_path=registry(tmp_path),
    )
    with pytest.raises(WorkflowBallotLensExecutionDenied):
        authorize_workflow_ballot_lens_execution(
            request(item, workflow_pass),
            principal=PRINCIPAL,
            internal_roles=[ROLE_CONTRIBUTOR],
            server_context=context,
        )
