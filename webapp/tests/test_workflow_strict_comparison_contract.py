from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from webapp.parser.contracts.workflow_comparison import (
    WORKFLOW_COMPARISON_PAYLOAD_CONTRACT,
    WORKFLOW_COMPARISON_PAYLOAD_SCHEMA_VERSION,
    WORKFLOW_COMPARISON_VERSION,
    semantic_sha256,
)
from webapp.parser.services.workflow_pre_qc_validation import (
    validate_first_workflow_pass_pre_qc,
    validate_second_workflow_pass_pre_qc,
)
from webapp.parser.services.workflow_staging_binding import (
    begin_workflow_staging_binding,
    finalize_workflow_staging_binding,
)
from webapp.parser.services.workflow_strict_comparison import (
    WORKFLOW_STRICT_COMPARISON_CAPABILITY,
    WORKFLOW_STRICT_COMPARISON_ROLE,
    WorkflowStrictComparisonConflict,
    execute_strict_workflow_comparison,
)
from webapp.parser.utils.models import (
    Base,
    StagingElectionResult,
    WorkflowComparison,
    WorkflowDiscrepancy,
    WorkflowEvent,
    WorkflowItem,
    WorkflowPass,
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


@pytest.fixture()
def registry(tmp_path: Path) -> Path:
    path = tmp_path / "urls.txt"
    path.write_text(
        "\n".join([
            "# === Curated | test ===",
            (
                "2024\tPresident\tIowa\tstatewide\tPDF\tCertified\t"
                "https://sos.example.gov/results.pdf"
            ),
        ])
        + "\n",
        encoding="utf-8",
    )
    return path


def _semantic(votes: int = 10):
    return {
        "scope": {
            "election_year": 2024,
            "election_date": "2024-11-05",
            "state": "Iowa",
            "jurisdiction_name": None,
            "jurisdiction_type": None,
            "contest": "President",
        },
        "records": [{
            "reporting_unit": {
                "name": "Precinct 1",
                "type": "precinct",
            },
            "percent_reporting": {
                "state": "value",
                "value": "100",
            },
            "vote_methods": [
                "Election Day",
                "Provisional",
            ],
            "method_totals": [
                {
                    "method": "Election Day",
                    "state": "value",
                    "votes": votes,
                },
                {
                    "method": "Provisional",
                    "state": "value",
                    "votes": 0,
                },
            ],
            "candidates": [{
                "name": "Jane Doe",
                "party": "DEM",
                "method_votes": [
                    {
                        "method": "Election Day",
                        "state": "value",
                        "votes": votes,
                    },
                    {
                        "method": "Provisional",
                        "state": "value",
                        "votes": 0,
                    },
                ],
                "total_votes": {
                    "state": "value",
                    "votes": votes,
                },
            }],
            "grand_total": {
                "state": "value",
                "votes": votes,
            },
        }],
    }


def _raw(semantic) -> bytes:
    return json.dumps(
        semantic,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _payload(item, workflow_pass, semantic, artifact_ref, artifact_hash):
    binding = {
        "workflow_item_id": str(item.id),
        "workflow_pass_id": str(workflow_pass.id),
        "pass_number": workflow_pass.pass_number,
        "revision_number": workflow_pass.revision_number,
        "source_evidence_ref": workflow_pass.source_evidence_ref,
        "staging_batch_id": str(workflow_pass.staging_batch_id),
        "normalized_artifact_ref": artifact_ref,
        "normalized_artifact_sha256": artifact_hash,
    }
    return {
        "schema": WORKFLOW_COMPARISON_PAYLOAD_CONTRACT,
        "schema_version": WORKFLOW_COMPARISON_PAYLOAD_SCHEMA_VERSION,
        "comparison_version": WORKFLOW_COMPARISON_VERSION,
        "binding": binding,
        "semantic": semantic,
        "semantic_sha256": semantic_sha256(semantic),
    }


def _seed_ready_pair(
    session,
    registry,
    *,
    left_votes=10,
    right_votes=10,
):
    item = WorkflowItem(
        id=uuid4(),
        lifecycle_state="active",
        current_stage="independent_acquisition",
        stage_condition="in_progress",
        priority=0,
        election_year=2024,
        election_date=date(2024, 11, 5),
        state="Iowa",
        jurisdiction_name=None,
        jurisdiction_type=None,
        contest="President",
        office_basic="President",
        election_type=None,
        source_race_id="2024PRESIA",
        source_url="https://sos.example.gov/results.pdf",
        canonical_race_id=None,
        blocked_reason_code=None,
        blocker_detail=None,
        created_by_principal=None,
        workflow_metadata={},
        row_version=5,
    )
    dl1 = WorkflowPass(
        id=uuid4(),
        workflow_item_id=item.id,
        pass_number=1,
        pass_label="DL1",
        revision_number=1,
        is_current=True,
        status="in_progress",
        assigned_principal="principal:dl1",
        source_evidence_ref=None,
        staging_batch_id=None,
        candidate_check_status=None,
        candidate_check_result=None,
        semantic_validation_status=None,
        semantic_validation_result=None,
        submitted_at=None,
    )
    dl2 = WorkflowPass(
        id=uuid4(),
        workflow_item_id=item.id,
        pass_number=2,
        pass_label="DL2",
        revision_number=1,
        is_current=True,
        status="in_progress",
        assigned_principal="principal:dl2",
        source_evidence_ref=None,
        staging_batch_id=None,
        candidate_check_status=None,
        candidate_check_result=None,
        semantic_validation_status=None,
        semantic_validation_result=None,
        submitted_at=None,
    )
    session.add_all([item, dl1, dl2])
    session.flush()

    artifacts = {}
    for workflow_pass, semantic, label in (
        (dl1, _semantic(left_votes), "dl1"),
        (dl2, _semantic(right_votes), "dl2"),
    ):
        begin_workflow_staging_binding(
            session,
            item.id,
            workflow_pass.id,
            principal=workflow_pass.assigned_principal,
            registry_path=registry,
        )
        session.add(
            StagingElectionResult(
                batch_id=workflow_pass.staging_batch_id,
                state="Iowa",
                county=None,
                source_url=item.source_url,
                raw_html="<table>fixture</table>",
            )
        )
        session.flush()
        raw = _raw(semantic)
        artifact_ref = f"staging://normalized/{label}.json"
        artifact_hash = hashlib.sha256(raw).hexdigest()
        finalize_workflow_staging_binding(
            session,
            item.id,
            workflow_pass.id,
            workflow_pass.staging_batch_id,
            principal=workflow_pass.assigned_principal,
            source_evidence_ref=f"evidence://{label}/source.pdf",
            artifact_ref=artifact_ref,
            artifact_sha256=artifact_hash,
        )
        payload = _payload(
            item,
            workflow_pass,
            semantic,
            artifact_ref,
            artifact_hash,
        )
        if workflow_pass.pass_number == 1:
            validate_first_workflow_pass_pre_qc(
                session,
                item.id,
                workflow_pass.id,
                workflow_pass.staging_batch_id,
                principal=workflow_pass.assigned_principal,
                normalized_payload=payload,
            )
        else:
            validate_second_workflow_pass_pre_qc(
                session,
                item.id,
                workflow_pass.id,
                workflow_pass.staging_batch_id,
                principal=workflow_pass.assigned_principal,
                normalized_payload=payload,
            )
        workflow_pass.status = "submitted"
        workflow_pass.submitted_at = datetime(
            2026, 9, 9, 4, 0, tzinfo=timezone.utc
        )
        artifacts[artifact_ref] = raw

    item.stage_condition = "ready"
    session.flush()
    return item, dl1, dl2, artifacts


def _loader(artifacts):
    def load(ref):
        return artifacts[ref]
    return load


def _adapter(artifact_ref, raw, frozen_binding):
    semantic = json.loads(raw.decode("utf-8"))
    return {
        "schema": WORKFLOW_COMPARISON_PAYLOAD_CONTRACT,
        "schema_version": WORKFLOW_COMPARISON_PAYLOAD_SCHEMA_VERSION,
        "comparison_version": WORKFLOW_COMPARISON_VERSION,
        "binding": dict(frozen_binding),
        "semantic": semantic,
        "semantic_sha256": semantic_sha256(semantic),
    }


def _execute(session, item, artifacts, **overrides):
    kwargs = {
        "expected_row_version": item.row_version,
        "service_version": "strict-comparison-test-v1",
        "normalized_artifact_loader": _loader(artifacts),
        "comparison_payload_adapter": _adapter,
    }
    kwargs.update(overrides)
    return execute_strict_workflow_comparison(
        session,
        item.id,
        **kwargs,
    )


def test_w8j_service_authority_constants_are_service_only():
    assert WORKFLOW_STRICT_COMPARISON_ROLE == "workflow_comparison_service"
    assert WORKFLOW_STRICT_COMPARISON_CAPABILITY == "workflow.comparison.execute"


def test_w8j_equal_comparison_advances_to_qc1_without_discrepancies(
    db_session,
    registry,
):
    item, dl1, dl2, artifacts = _seed_ready_pair(
        db_session,
        registry,
    )
    dl1_snapshot = (dl1.status, dl1.submitted_at, dl1.source_evidence_ref)
    dl2_snapshot = (dl2.status, dl2.submitted_at, dl2.source_evidence_ref)

    result = _execute(db_session, item, artifacts)

    assert result["strict_equality_passed"] is True
    assert result["difference_count"] == 0
    assert result["current_stage"] == "qc1_review"
    assert result["stage_condition"] == "pending"
    assert result["row_version"] == 6
    assert result["committed"] is False
    assert db_session.query(WorkflowComparison).count() == 1
    assert db_session.query(WorkflowDiscrepancy).count() == 0
    assert (
        dl1.status,
        dl1.submitted_at,
        dl1.source_evidence_ref,
    ) == dl1_snapshot
    assert (
        dl2.status,
        dl2.submitted_at,
        dl2.source_evidence_ref,
    ) == dl2_snapshot


def test_w8j_mismatch_creates_open_discrepancies_and_routes_resolution(
    db_session,
    registry,
):
    item, _dl1, _dl2, artifacts = _seed_ready_pair(
        db_session,
        registry,
        left_votes=10,
        right_votes=11,
    )
    result = _execute(db_session, item, artifacts)

    assert result["strict_equality_passed"] is False
    assert result["difference_count"] > 0
    assert result["current_stage"] == "discrepancy_resolution"
    discrepancies = db_session.query(WorkflowDiscrepancy).all()
    assert len(discrepancies) == result["difference_count"]
    assert all(d.resolution_status == "open" for d in discrepancies)
    assert all(d.severity is None for d in discrepancies)


def test_w8j_artifact_hash_mismatch_fails_before_comparison_mutation(
    db_session,
    registry,
):
    item, _dl1, dl2, artifacts = _seed_ready_pair(db_session, registry)
    artifacts[
        dl2.semantic_validation_result["normalized_artifact_ref"]
    ] += b"tamper"

    with pytest.raises(WorkflowStrictComparisonConflict):
        _execute(db_session, item, artifacts)

    assert db_session.query(WorkflowComparison).count() == 0
    assert item.current_stage == "independent_acquisition"
    assert item.stage_condition == "ready"
    assert item.row_version == 5


def test_w8j_adapter_binding_mismatch_fails_closed(
    db_session,
    registry,
):
    item, _dl1, _dl2, artifacts = _seed_ready_pair(db_session, registry)

    def bad_adapter(artifact_ref, raw, frozen_binding):
        payload = _adapter(artifact_ref, raw, frozen_binding)
        payload["binding"]["workflow_pass_id"] = str(uuid4())
        return payload

    with pytest.raises(WorkflowStrictComparisonConflict):
        _execute(
            db_session,
            item,
            artifacts,
            comparison_payload_adapter=bad_adapter,
        )

    assert db_session.query(WorkflowComparison).count() == 0


def test_w8j_stored_pre_qc_hash_mismatch_fails_closed(
    db_session,
    registry,
):
    item, _dl1, dl2, artifacts = _seed_ready_pair(db_session, registry)
    dl2.semantic_validation_result = {
        **dl2.semantic_validation_result,
        "semantic_sha256": "0" * 64,
    }
    db_session.flush()

    with pytest.raises(WorkflowStrictComparisonConflict):
        _execute(db_session, item, artifacts)

    assert db_session.query(WorkflowComparison).count() == 0


def test_w8j_stale_row_version_fails_without_comparison(
    db_session,
    registry,
):
    item, _dl1, _dl2, artifacts = _seed_ready_pair(db_session, registry)

    with pytest.raises(WorkflowStrictComparisonConflict):
        _execute(
            db_session,
            item,
            artifacts,
            expected_row_version=99,
        )

    assert db_session.query(WorkflowComparison).count() == 0


def test_w8j_requires_both_current_submitted_passes(
    db_session,
    registry,
):
    item, _dl1, dl2, artifacts = _seed_ready_pair(db_session, registry)
    dl2.status = "in_progress"
    dl2.submitted_at = None
    db_session.flush()

    with pytest.raises(WorkflowStrictComparisonConflict):
        _execute(db_session, item, artifacts)

    assert db_session.query(WorkflowComparison).count() == 0


def test_w8j_requires_dl1_dl2_principal_independence(
    db_session,
    registry,
):
    item, dl1, dl2, artifacts = _seed_ready_pair(db_session, registry)
    dl2.assigned_principal = dl1.assigned_principal
    db_session.flush()

    with pytest.raises(WorkflowStrictComparisonConflict):
        _execute(db_session, item, artifacts)

    assert db_session.query(WorkflowComparison).count() == 0


def test_w8j_exact_completed_replay_is_idempotent(
    db_session,
    registry,
):
    item, _dl1, _dl2, artifacts = _seed_ready_pair(db_session, registry)
    first = _execute(db_session, item, artifacts)
    event_count = (
        db_session.query(WorkflowEvent)
        .filter(
            WorkflowEvent.event_type.in_((
                "strict_comparison_started",
                "strict_comparison_completed",
            ))
        )
        .count()
    )
    second = _execute(
        db_session,
        item,
        artifacts,
        expected_row_version=first["row_version"],
    )

    assert second["already_compared"] is True
    assert second["comparison_id"] == first["comparison_id"]
    assert db_session.query(WorkflowComparison).count() == 1
    assert (
        db_session.query(WorkflowEvent)
        .filter(
            WorkflowEvent.event_type.in_((
                "strict_comparison_started",
                "strict_comparison_completed",
            ))
        )
        .count()
        == event_count
        == 2
    )


def test_w8j_nonexact_existing_comparison_conflicts(
    db_session,
    registry,
):
    item, dl1, dl2, artifacts = _seed_ready_pair(db_session, registry)
    db_session.add(
        WorkflowComparison(
            workflow_item_id=item.id,
            left_pass_id=dl2.id,
            right_pass_id=dl1.id,
            comparison_version=1,
            status="pending",
        )
    )
    db_session.flush()

    with pytest.raises(WorkflowStrictComparisonConflict):
        _execute(db_session, item, artifacts)


def test_w8j_service_is_commit_free_and_caller_rollback_removes_writes(
    db_session,
    registry,
):
    item, _dl1, _dl2, artifacts = _seed_ready_pair(db_session, registry)
    item_id = item.id
    db_session.commit()
    item = db_session.get(WorkflowItem, item_id)
    result = _execute(db_session, item, artifacts)
    assert result["committed"] is False
    assert db_session.query(WorkflowComparison).count() == 1

    db_session.rollback()

    restored = db_session.get(WorkflowItem, item_id)
    assert restored.current_stage == "independent_acquisition"
    assert restored.stage_condition == "ready"
    assert restored.row_version == 5
    assert db_session.query(WorkflowComparison).count() == 0
