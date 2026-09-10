from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import inspect
import json
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
from webapp.parser.contracts.workflow_canonical_writer import (
    WORKFLOW_CANONICAL_WRITER_CONTRACT_VERSION,
    WORKFLOW_CANONICAL_WRITER_REQUEST_SCHEMA,
    WORKFLOW_CANONICAL_WRITER_RESULT_SCHEMA,
)
from webapp.parser.services import workflow_publication as service
from webapp.parser.services.workflow_publication import (
    WORKFLOW_PUBLICATION_HANDOFF_CONTRACT,
    WORKFLOW_PUBLICATION_LINK_FAILURE_RECOVERY,
    WORKFLOW_PUBLICATION_TRANSACTION_MODEL,
    WorkflowPublicationConflict,
    WorkflowPublicationDependencyUnavailable,
    WorkflowPublicationLinkFailure,
    WorkflowPublicationWriterFailure,
    publish_workflow_item,
)
from webapp.parser.services.workflow_reviews import (
    WorkflowQCReviewConflict,
    load_workflow_publication_approval_authority,
)
from webapp.parser.utils.models import (
    Base,
    WorkflowComparison,
    WorkflowDiscrepancy,
    WorkflowEvent,
    WorkflowItem,
    WorkflowPass,
    WorkflowReview,
)


NOW = datetime(2026, 9, 10, 4, 30, tzinfo=timezone.utc)


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
            "reporting_unit": {"name": "Precinct 1", "type": "precinct"},
            "percent_reporting": {"state": "value", "value": "100"},
            "vote_methods": ["Election Day", "Provisional"],
            "method_totals": [
                {"method": "Election Day", "state": "value", "votes": votes},
                {"method": "Provisional", "state": "value", "votes": 0},
            ],
            "candidates": [{
                "name": "Jane Doe",
                "party": "DEM",
                "method_votes": [
                    {"method": "Election Day", "state": "value", "votes": votes},
                    {"method": "Provisional", "state": "value", "votes": 0},
                ],
                "total_votes": {"state": "value", "votes": votes},
            }],
            "grand_total": {"state": "value", "votes": votes},
        }],
    }


def _payload(item, selected, *, artifact_hash="a" * 64):
    semantic = _semantic()
    return {
        "schema": WORKFLOW_COMPARISON_PAYLOAD_CONTRACT,
        "schema_version": WORKFLOW_COMPARISON_PAYLOAD_SCHEMA_VERSION,
        "comparison_version": WORKFLOW_COMPARISON_VERSION,
        "binding": {
            "workflow_item_id": str(item.id),
            "workflow_pass_id": str(selected.id),
            "pass_number": selected.pass_number,
            "revision_number": selected.revision_number,
            "source_evidence_ref": selected.source_evidence_ref,
            "staging_batch_id": str(selected.staging_batch_id),
            "normalized_artifact_ref": "staging://normalized/dl1.json",
            "normalized_artifact_sha256": artifact_hash,
        },
        "semantic": semantic,
        "semantic_sha256": semantic_sha256(semantic),
    }


def _seed_ready(session, *, row_version=8, published=False):
    item = WorkflowItem(
        lifecycle_state=("published" if published else "ready_for_publication"),
        current_stage="publication_handoff",
        stage_condition=("complete" if published else "ready"),
        election_year=2024,
        state="Iowa",
        contest="President",
        source_url="https://sos.example.gov/results.pdf",
        row_version=row_version,
        canonical_race_id=(uuid4() if published else None),
        created_at=NOW,
        updated_at=NOW,
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
        assigned_principal="principal:dl1",
        source_evidence_ref="evidence://dl1.pdf",
        staging_batch_id=uuid4(),
        candidate_check_status="complete",
        candidate_check_result={"status": "complete"},
        semantic_validation_status="complete",
        semantic_validation_result={"status": "complete"},
        submitted_at=NOW,
        created_at=NOW,
        updated_at=NOW,
    )
    dl2 = WorkflowPass(
        workflow_item_id=item.id,
        pass_number=2,
        pass_label="DL2",
        revision_number=1,
        is_current=True,
        status="submitted",
        assigned_principal="principal:dl2",
        source_evidence_ref="evidence://dl2.pdf",
        staging_batch_id=uuid4(),
        candidate_check_status="complete",
        candidate_check_result={"status": "complete"},
        semantic_validation_status="complete",
        semantic_validation_result={"status": "complete"},
        submitted_at=NOW,
        created_at=NOW,
        updated_at=NOW,
    )
    session.add_all([dl1, dl2])
    session.flush()

    comparison = WorkflowComparison(
        workflow_item_id=item.id,
        left_pass_id=dl1.id,
        right_pass_id=dl2.id,
        comparison_version=WORKFLOW_COMPARISON_VERSION,
        status="complete",
        strict_equality_passed=True,
        difference_count=0,
        difference_summary={},
        checked_at=NOW,
        checked_by_service_version="strict-comparison-test-v1",
        reviewed_by_principal=None,
        reviewed_at=None,
        created_at=NOW,
    )
    session.add(comparison)
    session.flush()

    qc1 = WorkflowReview(
        workflow_item_id=item.id,
        review_stage="qc1",
        reviewer_principal="principal:qc1",
        decision="approved",
        selected_pass_id=dl1.id,
        selected_staging_batch_id=dl1.staging_batch_id,
        checklist_version="qc1-v1",
        checklist_result={"selection": True},
        reason_codes=[],
        notes="",
        reviewed_at=NOW,
    )
    session.add(qc1)
    session.flush()
    qc1_event = WorkflowEvent(
        workflow_item_id=item.id,
        actor_type="principal",
        actor_principal="principal:qc1",
        actor_service=None,
        event_type="qc1_review_approved",
        stage="qc1_review",
        prior_state={},
        new_state={},
        related_pass_id=dl1.id,
        related_comparison_id=comparison.id,
        related_review_id=qc1.id,
        related_staging_batch_id=dl1.staging_batch_id,
        related_canonical_race_id=None,
        reason_code=None,
        summary="qc1",
        event_metadata={
            "contract": "workflow_qc_review_v1",
            "review_stage": "qc1",
            "decision": "approved",
            "selected_pass_id": str(dl1.id),
            "selected_staging_batch_id": str(dl1.staging_batch_id),
            "canonical_writer_invoked": False,
            "client_selected_pass_authority": False,
            "mixed_side_value_merge": False,
            "direct_value_edit": False,
        },
        occurred_at=NOW,
    )
    session.add(qc1_event)
    session.flush()

    qc2 = WorkflowReview(
        workflow_item_id=item.id,
        review_stage="qc2",
        reviewer_principal="principal:qc2",
        decision="approved",
        selected_pass_id=dl1.id,
        selected_staging_batch_id=dl1.staging_batch_id,
        checklist_version="qc2-v1",
        checklist_result={"publication_ready": True},
        reason_codes=[],
        notes="",
        reviewed_at=NOW,
    )
    session.add(qc2)
    session.flush()
    qc2_event = WorkflowEvent(
        workflow_item_id=item.id,
        actor_type="principal",
        actor_principal="principal:qc2",
        actor_service=None,
        event_type="qc2_review_approved",
        stage="qc2_review",
        prior_state={},
        new_state={},
        related_pass_id=dl1.id,
        related_comparison_id=comparison.id,
        related_review_id=qc2.id,
        related_staging_batch_id=dl1.staging_batch_id,
        related_canonical_race_id=None,
        reason_code=None,
        summary="qc2",
        event_metadata={
            "contract": "workflow_qc_review_v1",
            "review_stage": "qc2",
            "decision": "approved",
            "selected_pass_id": str(dl1.id),
            "selected_staging_batch_id": str(dl1.staging_batch_id),
            "qc1_review_id": str(qc1.id),
            "selection_reason": "approved_qc1_review_inherited",
            "canonical_writer_invoked": False,
            "client_selected_pass_authority": False,
            "mixed_side_value_merge": False,
            "direct_value_edit": False,
        },
        occurred_at=NOW,
    )
    session.add(qc2_event)
    session.flush()
    return item, dl1, dl2, comparison, qc1, qc2


def test_publication_service_contract_is_explicit_and_http_free():
    source = inspect.getsource(service)
    assert WORKFLOW_PUBLICATION_HANDOFF_CONTRACT == "workflow_publication_handoff_v1"
    assert "DURABLE_WORKFLOW_START_THEN_CANONICAL_TRANSACTION_THEN_WORKFLOW_LINK" in source
    assert WORKFLOW_PUBLICATION_LINK_FAILURE_RECOVERY == "IDEMPOTENT_WRITER_REPLAY_THEN_WORKFLOW_LINK_RETRY"
    assert "from flask" not in source
    assert "@bp.route" not in source
    assert "client_selected_pass_id" not in source
    assert "client_selected_staging_batch_id" not in source


def test_publication_authority_reconstructs_exact_qc1_qc2_selection(db_session):
    item, dl1, dl2, comparison, qc1, qc2 = _seed_ready(db_session)
    authority = load_workflow_publication_approval_authority(db_session, item.id)
    assert authority["selected_pass"].id == dl1.id
    assert authority["comparison"].id == comparison.id
    assert authority["qc1_review"].id == qc1.id
    assert authority["qc2_review"].id == qc2.id
    assert authority["open_discrepancy_count"] == 0
    assert authority["dl1_principal"] != authority["dl2_principal"]


def test_publication_authority_rejects_qc2_selection_drift(db_session):
    item, _dl1, dl2, _comparison, _qc1, qc2 = _seed_ready(db_session)
    qc2.selected_pass_id = dl2.id
    qc2.selected_staging_batch_id = dl2.staging_batch_id
    db_session.flush()
    with pytest.raises(WorkflowQCReviewConflict):
        load_workflow_publication_approval_authority(db_session, item.id)


def test_publication_authority_rejects_open_discrepancy(db_session):
    item, dl1, _dl2, comparison, *_ = _seed_ready(db_session)
    db_session.add(WorkflowDiscrepancy(
        comparison_id=comparison.id,
        workflow_item_id=item.id,
        category="candidate_vote_value",
        semantic_key={"candidate": "Jane Doe"},
        left_value={"votes": 10},
        right_value={"votes": 9},
        left_value_state="value",
        right_value_state="value",
        resolution_status="open",
        created_at=NOW,
    ))
    db_session.flush()
    with pytest.raises(WorkflowQCReviewConflict):
        load_workflow_publication_approval_authority(db_session, item.id)


def test_publication_authority_can_reconcile_published_state_for_exact_replay(db_session):
    item, dl1, *_ = _seed_ready(db_session, published=True, row_version=9)
    authority = load_workflow_publication_approval_authority(
        db_session,
        item.id,
        require_ready_state=False,
    )
    assert authority["selected_pass"].id == dl1.id
    with pytest.raises(WorkflowQCReviewConflict):
        load_workflow_publication_approval_authority(db_session, item.id)


def test_w5_request_is_server_derived_and_idempotency_excludes_attempt_identity(db_session):
    item, *_ = _seed_ready(db_session)
    authority = load_workflow_publication_approval_authority(db_session, item.id)
    payload = _payload(item, authority["selected_pass"])
    first = service._build_w5_request(
        item=item,
        authority=authority,
        payload=payload,
        handoff_event_id=uuid4(),
        principal="principal:publisher-a",
        expected_row_version=8,
    )
    second = service._build_w5_request(
        item=item,
        authority=authority,
        payload=payload,
        handoff_event_id=uuid4(),
        principal="principal:publisher-b",
        expected_row_version=99,
    )
    assert first["schema"] == WORKFLOW_CANONICAL_WRITER_REQUEST_SCHEMA
    assert first["schema_version"] == WORKFLOW_CANONICAL_WRITER_CONTRACT_VERSION
    assert first["idempotency_key"] == second["idempotency_key"]
    assert first["request_id"] != second["request_id"]
    assert first["workflow"]["publication_handoff_event_id"] != second["workflow"]["publication_handoff_event_id"]
    assert first["approval"]["selected_pass_id"] == str(authority["selected_pass"].id)


def test_dependencies_fail_closed_before_any_publication_attempt():
    with pytest.raises(WorkflowPublicationDependencyUnavailable):
        publish_workflow_item(
            uuid4(),
            principal="principal:publisher",
            expected_row_version=8,
            workflow_session_factory=None,
            canonical_writer=lambda request: request,
            normalized_artifact_loader=lambda ref: b"x",
            comparison_payload_adapter=lambda ref, raw, binding: {},
        )


def test_selected_artifact_sha_is_reverified_before_payload_adapter(monkeypatch):
    item = WorkflowItem(id=uuid4())
    selected = type("Pass", (), {
        "id": uuid4(),
        "staging_batch_id": uuid4(),
        "pass_number": 1,
        "pass_label": "DL1",
    })()
    called = {"adapter": False}
    monkeypatch.setattr(
        service,
        "validate_frozen_workflow_staging_binding",
        lambda *a, **k: {
            "artifact_ref": "staging://x",
            "artifact_sha256": "a" * 64,
            "comparison_binding": {},
        },
    )
    with pytest.raises(WorkflowPublicationConflict):
        service._validate_selected_payload(
            object(),
            {"item": item, "selected_pass": selected},
            normalized_artifact_loader=lambda ref: b"not-the-authoritative-bytes",
            comparison_payload_adapter=lambda *a: called.__setitem__("adapter", True),
        )
    assert called["adapter"] is False


def test_exact_published_replay_bypasses_writer(monkeypatch):
    expected = {"success": True, "already_workflow_published": True}
    monkeypatch.setattr(service, "_published_replay", lambda *a, **k: expected)
    called = {"writer": False}
    result = publish_workflow_item(
        uuid4(),
        principal="principal:publisher",
        expected_row_version=9,
        workflow_session_factory=lambda: object(),
        canonical_writer=lambda request: called.__setitem__("writer", True),
        normalized_artifact_loader=lambda ref: b"x",
        comparison_payload_adapter=lambda ref, raw, binding: {},
    )
    assert result is expected
    assert called["writer"] is False


def test_successful_orchestration_is_prepare_writer_finalize(monkeypatch):
    calls = []
    request = {
        "schema": WORKFLOW_CANONICAL_WRITER_REQUEST_SCHEMA,
        "schema_version": 1,
        "request_id": str(uuid4()),
        "idempotency_key": "a" * 64,
    }
    attempt = {"handoff_event_id": str(uuid4())}
    publication = {
        "canonical_race_id": str(uuid4()),
        "canonical_source_artifact_id": str(uuid4()),
        "canonical_result_count": 1,
        "canonical_vote_component_count": 2,
        "semantic_sha256": "b" * 64,
        "committed_at": "2026-09-10T04:30:00Z",
        "writer_service_version": "writer-v1",
    }
    writer_result = {
        "schema": WORKFLOW_CANONICAL_WRITER_RESULT_SCHEMA,
        "schema_version": 1,
        "request_id": request["request_id"],
        "idempotency_key": request["idempotency_key"],
        "status": "published",
        "success": True,
        "publication": publication,
        "error": None,
    }
    monkeypatch.setattr(service, "_published_replay", lambda *a, **k: None)
    monkeypatch.setattr(
        service,
        "_prepare_attempt",
        lambda *a, **k: (calls.append("prepare") or (request, attempt)),
    )
    monkeypatch.setattr(service, "validate_canonical_writer_result", lambda raw: raw)
    monkeypatch.setattr(service, "assert_canonical_writer_result_matches_request", lambda req, res: None)
    monkeypatch.setattr(
        service,
        "_finalize_success",
        lambda *a, **k: (calls.append("finalize") or {"success": True}),
    )
    result = publish_workflow_item(
        uuid4(),
        principal="principal:publisher",
        expected_row_version=8,
        workflow_session_factory=lambda: object(),
        canonical_writer=lambda req: (calls.append("writer") or writer_result),
        normalized_artifact_loader=lambda ref: b"x",
        comparison_payload_adapter=lambda ref, raw, binding: {},
    )
    assert result["success"] is True
    assert calls == ["prepare", "writer", "finalize"]


def test_valid_writer_failure_is_durably_audited_then_raised(monkeypatch):
    request = {
        "request_id": str(uuid4()),
        "idempotency_key": "a" * 64,
    }
    failed = {
        "schema": WORKFLOW_CANONICAL_WRITER_RESULT_SCHEMA,
        "schema_version": 1,
        "request_id": request["request_id"],
        "idempotency_key": request["idempotency_key"],
        "status": "failed",
        "success": False,
        "publication": None,
        "error": {"code": "write_failed", "message": "rolled back", "retryable": True},
    }
    monkeypatch.setattr(service, "_published_replay", lambda *a, **k: None)
    monkeypatch.setattr(service, "_prepare_attempt", lambda *a, **k: (request, {}))
    monkeypatch.setattr(service, "validate_canonical_writer_result", lambda raw: raw)
    monkeypatch.setattr(service, "assert_canonical_writer_result_matches_request", lambda req, res: None)
    monkeypatch.setattr(
        service,
        "_record_writer_failure",
        lambda *a, **k: {
            "success": False,
            "retryable": True,
            "message": "rolled back",
            "committed": True,
        },
    )
    with pytest.raises(WorkflowPublicationWriterFailure) as excinfo:
        publish_workflow_item(
            uuid4(),
            principal="principal:publisher",
            expected_row_version=8,
            workflow_session_factory=lambda: object(),
            canonical_writer=lambda req: failed,
            normalized_artifact_loader=lambda ref: b"x",
            comparison_payload_adapter=lambda ref, raw, binding: {},
        )
    assert excinfo.value.payload["retryable"] is True
    assert excinfo.value.payload["committed"] is True


def test_malformed_writer_result_fails_closed_without_finalize(monkeypatch):
    monkeypatch.setattr(service, "_published_replay", lambda *a, **k: None)
    monkeypatch.setattr(service, "_prepare_attempt", lambda *a, **k: ({"request_id": str(uuid4()), "idempotency_key": "a" * 64}, {}))
    monkeypatch.setattr(service, "_record_untrusted_writer_failure", lambda *a, **k: str(uuid4()))
    called = {"finalize": False}
    monkeypatch.setattr(
        service,
        "_finalize_success",
        lambda *a, **k: called.__setitem__("finalize", True),
    )
    with pytest.raises(WorkflowPublicationDependencyUnavailable):
        publish_workflow_item(
            uuid4(),
            principal="principal:publisher",
            expected_row_version=8,
            workflow_session_factory=lambda: object(),
            canonical_writer=lambda req: {"bad": "shape"},
            normalized_artifact_loader=lambda ref: b"x",
            comparison_payload_adapter=lambda ref, raw, binding: {},
        )
    assert called["finalize"] is False


def test_writer_exception_is_not_blindly_retried(monkeypatch):
    monkeypatch.setattr(service, "_published_replay", lambda *a, **k: None)
    monkeypatch.setattr(service, "_prepare_attempt", lambda *a, **k: ({"request_id": str(uuid4()), "idempotency_key": "a" * 64}, {}))
    monkeypatch.setattr(service, "_record_untrusted_writer_failure", lambda *a, **k: str(uuid4()))
    calls = {"writer": 0}
    def boom(_request):
        calls["writer"] += 1
        raise RuntimeError("ambiguous")
    with pytest.raises(WorkflowPublicationDependencyUnavailable):
        publish_workflow_item(
            uuid4(),
            principal="principal:publisher",
            expected_row_version=8,
            workflow_session_factory=lambda: object(),
            canonical_writer=boom,
            normalized_artifact_loader=lambda ref: b"x",
            comparison_payload_adapter=lambda ref, raw, binding: {},
        )
    assert calls["writer"] == 1


def test_link_failure_preserves_canonical_result_for_governed_recovery(monkeypatch):
    request = {"request_id": str(uuid4()), "idempotency_key": "a" * 64}
    result = {
        "status": "published",
        "success": True,
        "publication": {},
    }
    monkeypatch.setattr(service, "_published_replay", lambda *a, **k: None)
    monkeypatch.setattr(service, "_prepare_attempt", lambda *a, **k: (request, {}))
    monkeypatch.setattr(service, "validate_canonical_writer_result", lambda raw: raw)
    monkeypatch.setattr(service, "assert_canonical_writer_result_matches_request", lambda req, res: None)
    def fail_link(*a, **k):
        raise WorkflowPublicationLinkFailure(
            "link failed",
            canonical_result=result,
            idempotency_key=request["idempotency_key"],
        )
    monkeypatch.setattr(service, "_finalize_success", fail_link)
    with pytest.raises(WorkflowPublicationLinkFailure) as excinfo:
        publish_workflow_item(
            uuid4(),
            principal="principal:publisher",
            expected_row_version=8,
            workflow_session_factory=lambda: object(),
            canonical_writer=lambda req: result,
            normalized_artifact_loader=lambda ref: b"x",
            comparison_payload_adapter=lambda ref, raw, binding: {},
        )
    assert excinfo.value.idempotency_key == "a" * 64
    assert excinfo.value.canonical_result["status"] == "published"


def test_publication_api_signature_accepts_no_client_selected_authority():
    params = inspect.signature(publish_workflow_item).parameters
    assert set(params) == {
        "item_id",
        "principal",
        "expected_row_version",
        "workflow_session_factory",
        "canonical_writer",
        "normalized_artifact_loader",
        "comparison_payload_adapter",
        "now",
    }
    forbidden = {
        "selected_pass_id",
        "selected_staging_batch_id",
        "qc1_review_id",
        "qc2_review_id",
        "comparison_id",
        "idempotency_key",
        "canonical_race_id",
        "payload",
    }
    assert forbidden.isdisjoint(params)
