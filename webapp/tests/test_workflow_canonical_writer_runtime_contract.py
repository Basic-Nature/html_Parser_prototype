from __future__ import annotations

import copy
from datetime import datetime, timezone
import inspect

from sqlalchemy import create_engine, func, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from webapp.parser.contracts.workflow_canonical_writer import (
    WORKFLOW_CANONICAL_WRITER_REQUEST_SCHEMA,
    WORKFLOW_CANONICAL_WRITER_CONTRACT_VERSION,
    assert_canonical_writer_result_matches_request,
    derive_canonical_writer_idempotency_key,
    validate_canonical_writer_result,
)
from webapp.parser.contracts.workflow_comparison import (
    WORKFLOW_COMPARISON_PAYLOAD_CONTRACT,
    WORKFLOW_COMPARISON_PAYLOAD_SCHEMA_VERSION,
    WORKFLOW_COMPARISON_VERSION,
    semantic_sha256,
)
from webapp.parser.services.workflow_canonical_writer import (
    WORKFLOW_CANONICAL_APPROVAL_ARTIFACT_ROLE,
    WORKFLOW_CANONICAL_HANDOFF_EVENT_REQUIREMENT,
    WORKFLOW_CANONICAL_NULL_MISSING_POLICY,
    WORKFLOW_CANONICAL_PAYLOAD_ARTIFACT_ROLE,
    WORKFLOW_CANONICAL_SOURCE_RACE_ID_RULE,
    WORKFLOW_CANONICAL_WRITE_SCOPE,
    WORKFLOW_CANONICAL_WRITER_RUNTIME_CONTRACT,
    WORKFLOW_CANONICAL_WRITER_SERVICE_VERSION,
    build_workflow_canonical_writer,
    write_workflow_canonical_publication,
)
from webapp.parser.utils.models import (
    Base,
    CanonicalElectionRace,
    CanonicalElectionResult,
    CanonicalSourceArtifact,
    CanonicalVoteComponent,
)


def _db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(
        bind=engine,
        expire_on_commit=False,
        class_=Session,
    )
    return engine, factory


def _semantic():
    return {
        "scope": {
            "election_year": 2024,
            "election_date": "2024-11-05",
            "state": "TX",
            "jurisdiction_name": None,
            "jurisdiction_type": None,
            "contest": "President",
        },
        "records": [
            {
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
                    "Early Voting",
                    "Provisional",
                    "Curbside",
                ],
                "method_totals": [
                    {
                        "method": "Election Day",
                        "state": "value",
                        "votes": 11,
                    },
                    {
                        "method": "Early Voting",
                        "state": "value",
                        "votes": 7,
                    },
                    {
                        "method": "Provisional",
                        "state": "value",
                        "votes": 0,
                    },
                    {
                        "method": "Curbside",
                        "state": "value",
                        "votes": 0,
                    },
                ],
                "candidates": [
                    {
                        "name": "Jane Doe",
                        "party": "DEM",
                        "method_votes": [
                            {
                                "method": "Election Day",
                                "state": "value",
                                "votes": 6,
                            },
                            {
                                "method": "Early Voting",
                                "state": "value",
                                "votes": 4,
                            },
                            {
                                "method": "Provisional",
                                "state": "value",
                                "votes": 0,
                            },
                            {
                                "method": "Curbside",
                                "state": "value",
                                "votes": 0,
                            },
                        ],
                        "total_votes": {
                            "state": "value",
                            "votes": 10,
                        },
                    },
                    {
                        "name": "John Smith",
                        "party": "REP",
                        "method_votes": [
                            {
                                "method": "Election Day",
                                "state": "value",
                                "votes": 5,
                            },
                            {
                                "method": "Early Voting",
                                "state": "value",
                                "votes": 3,
                            },
                            {
                                "method": "Provisional",
                                "state": "value",
                                "votes": 0,
                            },
                            {
                                "method": "Curbside",
                                "state": "value",
                                "votes": 0,
                            },
                        ],
                        "total_votes": {
                            "state": "value",
                            "votes": 8,
                        },
                    },
                ],
                "grand_total": {
                    "state": "value",
                    "votes": 18,
                },
            }
        ],
    }


def _request(
    *,
    item_suffix: str = "0001",
    request_suffix: str = "0501",
    operator: str = "principal:publication",
):
    semantic = _semantic()
    item_id = f"00000000-0000-0000-0000-00000000{item_suffix}"
    pass_id = "00000000-0000-0000-0000-000000000101"
    staging_id = "00000000-0000-0000-0000-000000000010"
    request = {
        "schema": WORKFLOW_CANONICAL_WRITER_REQUEST_SCHEMA,
        "schema_version": WORKFLOW_CANONICAL_WRITER_CONTRACT_VERSION,
        "request_id":
            f"00000000-0000-0000-0000-00000000{request_suffix}",
        "idempotency_key": "0" * 64,
        "workflow": {
            "workflow_item_id": item_id,
            "workflow_row_version": 7,
            "publication_handoff_event_id":
                "00000000-0000-0000-0000-000000000502",
            "publication_operator_principal": operator,
        },
        "approval": {
            "qc1_review_id":
                "00000000-0000-0000-0000-000000000201",
            "qc2_review_id":
                "00000000-0000-0000-0000-000000000202",
            "qc1_decision": "approved",
            "qc2_decision": "approved",
            "selected_pass_id": pass_id,
            "selected_staging_batch_id": staging_id,
        },
        "comparison": {
            "comparison_id":
                "00000000-0000-0000-0000-000000000301",
            "comparison_version": 1,
            "status": "complete",
            "strict_equality_passed": False,
            "open_discrepancy_count": 0,
        },
        "payload": {
            "schema": WORKFLOW_COMPARISON_PAYLOAD_CONTRACT,
            "schema_version":
                WORKFLOW_COMPARISON_PAYLOAD_SCHEMA_VERSION,
            "comparison_version": WORKFLOW_COMPARISON_VERSION,
            "binding": {
                "workflow_item_id": item_id,
                "workflow_pass_id": pass_id,
                "pass_number": 2,
                "revision_number": 1,
                "source_evidence_ref": "official-source:example",
                "staging_batch_id": staging_id,
                "normalized_artifact_ref":
                    "staging://normalized/example.json",
                "normalized_artifact_sha256": "1" * 64,
            },
            "semantic": semantic,
            "semantic_sha256": semantic_sha256(semantic),
        },
    }
    request["idempotency_key"] = (
        derive_canonical_writer_idempotency_key(request)
    )
    return request


def _rehash(request):
    request["payload"]["semantic_sha256"] = semantic_sha256(
        request["payload"]["semantic"]
    )
    request["idempotency_key"] = (
        derive_canonical_writer_idempotency_key(request)
    )
    return request


def _counts(engine):
    with Session(engine) as session:
        return {
            "artifacts": session.scalar(
                select(func.count(CanonicalSourceArtifact.id))
            ),
            "races": session.scalar(
                select(func.count(CanonicalElectionRace.id))
            ),
            "results": session.scalar(
                select(func.count(CanonicalElectionResult.id))
            ),
            "components": session.scalar(
                select(func.count(CanonicalVoteComponent.id))
            ),
        }


def test_runtime_contract_is_internal_canonical_tables_only():
    source = inspect.getsource(
        __import__(
            "webapp.parser.services.workflow_canonical_writer",
            fromlist=["*"],
        )
    )
    assert WORKFLOW_CANONICAL_WRITER_RUNTIME_CONTRACT == (
        "workflow_canonical_writer_runtime_v1"
    )
    assert WORKFLOW_CANONICAL_WRITE_SCOPE == (
        "CANONICAL_SOURCE_ARTIFACT_RACE_RESULT_VOTE_COMPONENT_TABLES_ONLY"
    )
    assert WORKFLOW_CANONICAL_NULL_MISSING_POLICY == (
        "REJECT_UNREPRESENTABLE_CANDIDATE_TOTAL_OR_METHOD_STATE_NO_COERCION"
    )
    assert WORKFLOW_CANONICAL_HANDOFF_EVENT_REQUIREMENT == (
        "CALLER_MUST_SUPPLY_DURABLE_WORKFLOW_HANDOFF_EVENT_ID_BEFORE_CANONICAL_COMMIT"
    )
    assert "WorkflowItem" not in source
    assert "WorkflowEvent" not in source
    assert "flask" not in source.casefold()
    assert "Blueprint" not in source
    assert "with session.begin():" in source
    assert ".commit(" not in source


def test_one_argument_callback_adapter_publishes_representable_request():
    engine, factory = _db()
    writer = build_workflow_canonical_writer(factory)
    assert list(inspect.signature(writer).parameters) == ["request"]

    request = _request()
    result = writer(request)
    assert result["status"] == "published"
    assert result["success"] is True
    validate_canonical_writer_result(result)
    assert_canonical_writer_result_matches_request(request, result)
    assert _counts(engine) == {
        "artifacts": 2,
        "races": 1,
        "results": 2,
        "components": 8,
    }


def test_published_rows_preserve_candidate_method_and_zero_values():
    engine, factory = _db()
    request = _request()
    result = write_workflow_canonical_publication(
        request,
        session_factory=factory,
        now=datetime(2026, 9, 10, 4, 0, tzinfo=timezone.utc),
    )
    assert result["status"] == "published"

    with Session(engine) as session:
        race = session.scalar(select(CanonicalElectionRace))
        assert race is not None
        assert race.source_race_id == request["workflow"]["workflow_item_id"]
        assert race.selected_dl_source == "DL2"
        assert race.state == "TX"
        assert race.contest == "President"
        assert race.election_date.isoformat() == "2024-11-05"
        assert race.source_url is None
        assert race.office_basic is None
        assert race.verification_status == "verified"

        rows = session.scalars(
            select(CanonicalElectionResult).order_by(
                CanonicalElectionResult.source_row_index
            )
        ).all()
        assert [row.candidate for row in rows] == [
            "Jane Doe",
            "John Smith",
        ]
        assert [row.total_votes for row in rows] == [10, 8]
        assert all(row.precinct == "Precinct 1" for row in rows)
        assert all(row.jurisdiction_type == "precinct" for row in rows)
        assert all(row.aggregation_scope == "precinct" for row in rows)

        jane = rows[0]
        components = session.scalars(
            select(CanonicalVoteComponent)
            .where(CanonicalVoteComponent.result_id == jane.id)
            .order_by(CanonicalVoteComponent.vote_method)
        ).all()
        values = {
            component.vote_method: component.votes
            for component in components
        }
        assert values == {
            "Curbside": 0,
            "Early Voting": 4,
            "Election Day": 6,
            "Provisional": 0,
        }


def test_exact_replay_returns_same_linkage_without_duplicate_rows():
    engine, factory = _db()
    first_request = _request()
    first = write_workflow_canonical_publication(
        first_request,
        session_factory=factory,
        now=datetime(2026, 9, 10, 4, 1, tzinfo=timezone.utc),
    )
    assert first["status"] == "published"

    replay = copy.deepcopy(first_request)
    replay["request_id"] = "00000000-0000-0000-0000-000000000599"
    replay["workflow"]["publication_operator_principal"] = (
        "principal:other-publication"
    )
    assert (
        derive_canonical_writer_idempotency_key(replay)
        == first_request["idempotency_key"]
    )
    replay["idempotency_key"] = first_request["idempotency_key"]

    second = write_workflow_canonical_publication(
        replay,
        session_factory=factory,
        now=datetime(2026, 9, 10, 4, 2, tzinfo=timezone.utc),
    )
    assert second["status"] == "already_published"
    assert second["publication"]["canonical_race_id"] == (
        first["publication"]["canonical_race_id"]
    )
    assert second["publication"]["canonical_source_artifact_id"] == (
        first["publication"]["canonical_source_artifact_id"]
    )
    assert second["publication"]["committed_at"] == (
        first["publication"]["committed_at"]
    )
    assert _counts(engine) == {
        "artifacts": 2,
        "races": 1,
        "results": 2,
        "components": 8,
    }


def test_same_workflow_item_different_idempotency_is_rejected():
    engine, factory = _db()
    request = _request()
    first = write_workflow_canonical_publication(
        request,
        session_factory=factory,
    )
    assert first["status"] == "published"
    before = _counts(engine)

    conflict = copy.deepcopy(request)
    conflict["request_id"] = "00000000-0000-0000-0000-000000000598"
    conflict["approval"]["qc2_review_id"] = (
        "00000000-0000-0000-0000-000000000299"
    )
    conflict["idempotency_key"] = (
        derive_canonical_writer_idempotency_key(conflict)
    )
    assert conflict["idempotency_key"] != request["idempotency_key"]

    result = write_workflow_canonical_publication(
        conflict,
        session_factory=factory,
    )
    assert result["status"] == "rejected"
    assert result["error"]["code"] == "idempotency_conflict"
    assert result["error"]["retryable"] is False
    assert _counts(engine) == before


def test_null_or_missing_candidate_method_is_rejected_without_coercion():
    for state in ("null", "missing"):
        engine, factory = _db()
        request = _request(item_suffix="0011")
        component = request["payload"]["semantic"]["records"][0][
            "candidates"
        ][0]["method_votes"][3]
        component["state"] = state
        component["votes"] = None
        _rehash(request)

        result = write_workflow_canonical_publication(
            request,
            session_factory=factory,
        )
        assert result["status"] == "rejected"
        assert result["error"]["code"] == "precondition_failed"
        assert "cannot represent null/missing without coercion" in (
            result["error"]["message"]
        )
        assert _counts(engine) == {
            "artifacts": 0,
            "races": 0,
            "results": 0,
            "components": 0,
        }


def test_null_candidate_total_is_rejected_without_zero_coercion():
    engine, factory = _db()
    request = _request(item_suffix="0012")
    total = request["payload"]["semantic"]["records"][0][
        "candidates"
    ][0]["total_votes"]
    total["state"] = "null"
    total["votes"] = None
    _rehash(request)

    result = write_workflow_canonical_publication(
        request,
        session_factory=factory,
    )
    assert result["status"] == "rejected"
    assert result["error"]["code"] == "precondition_failed"
    assert _counts(engine)["races"] == 0


def test_required_scope_and_reporting_unit_are_fail_closed():
    engine, factory = _db()
    request = _request(item_suffix="0013")
    request["payload"]["semantic"]["scope"]["state"] = None
    _rehash(request)
    result = write_workflow_canonical_publication(
        request,
        session_factory=factory,
    )
    assert result["error"]["code"] == "precondition_failed"
    assert _counts(engine)["races"] == 0

    engine, factory = _db()
    request = _request(item_suffix="0014")
    request["payload"]["semantic"]["records"][0][
        "reporting_unit"
    ]["name"] = None
    _rehash(request)
    result = write_workflow_canonical_publication(
        request,
        session_factory=factory,
    )
    assert result["error"]["code"] == "precondition_failed"
    assert _counts(engine)["races"] == 0


def test_unsupported_dl_number_is_rejected_before_canonical_rows():
    engine, factory = _db()
    request = _request(item_suffix="0015")
    request["payload"]["binding"]["pass_number"] = 3
    _rehash(request)
    result = write_workflow_canonical_publication(
        request,
        session_factory=factory,
    )
    assert result["status"] == "rejected"
    assert result["error"]["code"] == "precondition_failed"
    assert _counts(engine)["races"] == 0


def test_invalid_w5_request_returns_contract_valid_invalid_request():
    engine, factory = _db()
    request = _request(item_suffix="0016")
    request["idempotency_key"] = "f" * 64
    result = write_workflow_canonical_publication(
        request,
        session_factory=factory,
    )
    assert result["status"] == "rejected"
    assert result["success"] is False
    assert result["error"]["code"] == "invalid_request"
    validate_canonical_writer_result(result)
    assert _counts(engine)["races"] == 0


def test_sqlalchemy_write_failure_rolls_back_atomic_canonical_transaction():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    class FailingSession(Session):
        def flush(self, objects=None):
            raise OperationalError(
                "forced canonical write failure",
                {},
                RuntimeError("forced"),
            )

    factory = sessionmaker(
        bind=engine,
        expire_on_commit=False,
        class_=FailingSession,
    )
    request = _request(item_suffix="0017")
    result = write_workflow_canonical_publication(
        request,
        session_factory=factory,
    )
    assert result["status"] == "failed"
    assert result["error"] == {
        "code": "write_failed",
        "message": (
            "Canonical transaction failed and was rolled back; retry may "
            "reconcile an independently completed concurrent publication."
        ),
        "retryable": True,
    }
    assert _counts(engine) == {
        "artifacts": 0,
        "races": 0,
        "results": 0,
        "components": 0,
    }


def test_foreign_existing_canonical_race_is_nonretryable_conflict():
    engine, factory = _db()
    request = _request(item_suffix="0018")
    with factory() as session:
        with session.begin():
            payload = CanonicalSourceArtifact(
                artifact_role="payload",
                filename="foreign-payload.json",
                sha256="8" * 64,
                row_count=1,
                race_count=1,
                provenance={"foreign": True},
            )
            approval = CanonicalSourceArtifact(
                artifact_role="approval",
                filename="foreign-approval.json",
                sha256="9" * 64,
                row_count=1,
                race_count=1,
                provenance={"foreign": True},
            )
            session.add_all([payload, approval])
            session.flush()
            session.add(CanonicalElectionRace(
                source_race_id=request["workflow"]["workflow_item_id"],
                election_year=2024,
                election_date=None,
                date_precision="year",
                state="TX",
                contest="Foreign Race",
                office_basic=None,
                production_status="prod_loaded",
                selected_dl_source="DL1",
                source_url=None,
                verification_status="verified",
                payload_artifact_id=payload.id,
                approval_artifact_id=approval.id,
                qa_metadata={"foreign": True},
            ))

    before = _counts(engine)
    result = write_workflow_canonical_publication(
        request,
        session_factory=factory,
    )
    assert result["status"] == "rejected"
    assert result["error"]["code"] == "canonical_conflict"
    assert result["error"]["retryable"] is False
    assert _counts(engine) == before


def test_payload_and_approval_artifacts_are_distinct_immutable_authorities():
    engine, factory = _db()
    request = _request(item_suffix="0019")
    result = write_workflow_canonical_publication(
        request,
        session_factory=factory,
    )
    assert result["status"] == "published"

    with Session(engine) as session:
        artifacts = session.scalars(
            select(CanonicalSourceArtifact).order_by(
                CanonicalSourceArtifact.artifact_role
            )
        ).all()
        assert len(artifacts) == 2
        by_role = {artifact.artifact_role: artifact for artifact in artifacts}
        payload = by_role[WORKFLOW_CANONICAL_PAYLOAD_ARTIFACT_ROLE]
        approval = by_role[WORKFLOW_CANONICAL_APPROVAL_ARTIFACT_ROLE]
        assert payload.sha256 == "1" * 64
        assert approval.sha256 != payload.sha256
        assert payload.id != approval.id
        assert result["publication"]["canonical_source_artifact_id"] == (
            str(payload.id)
        )


def test_provenance_preserves_reporting_metadata_without_source_inference():
    engine, factory = _db()
    request = _request(item_suffix="0020")
    record = request["payload"]["semantic"]["records"][0]
    record["percent_reporting"] = {"state": "null", "value": None}
    record["method_totals"][3] = {
        "method": "Curbside",
        "state": "missing",
        "votes": None,
    }
    _rehash(request)

    result = write_workflow_canonical_publication(
        request,
        session_factory=factory,
        now=datetime(2026, 9, 10, 4, 30, tzinfo=timezone.utc),
    )
    assert result["status"] == "published"

    with Session(engine) as session:
        race = session.scalar(select(CanonicalElectionRace))
        row = session.scalar(select(CanonicalElectionResult))
        assert race is not None and row is not None
        publication = race.qa_metadata["workflow_publication"]
        reporting = publication["reporting_metadata"][0]
        assert reporting["percent_reporting"] == {
            "state": "null",
            "value": None,
        }
        assert reporting["method_totals"][3] == {
            "method": "Curbside",
            "state": "missing",
            "votes": None,
        }
        assert publication["source_url_authority"] == (
            "W5_REQUEST_HAS_NO_SOURCE_URL_NO_INFERENCE"
        )
        assert publication["office_basic_authority"] == (
            "W5_REQUEST_HAS_NO_OFFICE_BASIC_NO_INFERENCE"
        )
        assert race.source_url is None
        assert race.office_basic is None
        assert row.source_url is None
        assert row.provenance["source_url_authority"] == (
            "W5_REQUEST_HAS_NO_SOURCE_URL_NO_INFERENCE"
        )


def test_source_race_identity_is_deterministic_workflow_item_uuid():
    engine, factory = _db()
    request = _request(item_suffix="0021")
    result = write_workflow_canonical_publication(
        request,
        session_factory=factory,
    )
    assert result["status"] == "published"
    assert WORKFLOW_CANONICAL_SOURCE_RACE_ID_RULE == (
        "CANONICAL_SOURCE_RACE_ID_EQUALS_WORKFLOW_ITEM_UUID"
    )
    with Session(engine) as session:
        race = session.scalar(select(CanonicalElectionRace))
        assert race is not None
        assert race.source_race_id == request["workflow"]["workflow_item_id"]
        publication = race.qa_metadata["workflow_publication"]
        assert publication["source_race_id_rule"] == (
            WORKFLOW_CANONICAL_SOURCE_RACE_ID_RULE
        )
        assert publication["writer_service_version"] == (
            WORKFLOW_CANONICAL_WRITER_SERVICE_VERSION
        )

def test_same_idempotency_key_reconciles_immutable_nonkey_authority():
    engine, factory = _db()
    request = _request(item_suffix="0022")
    first = write_workflow_canonical_publication(
        request,
        session_factory=factory,
    )
    assert first["status"] == "published"

    replay = copy.deepcopy(request)
    replay["request_id"] = "00000000-0000-0000-0000-000000000597"
    replay["comparison"]["strict_equality_passed"] = True
    # W5 intentionally excludes strict_equality_passed from key material.
    assert derive_canonical_writer_idempotency_key(replay) == (
        request["idempotency_key"]
    )
    replay["idempotency_key"] = request["idempotency_key"]

    result = write_workflow_canonical_publication(
        replay,
        session_factory=factory,
    )
    assert result["status"] == "rejected"
    assert result["error"]["code"] == "canonical_conflict"
    assert result["error"]["retryable"] is False
    assert _counts(engine) == {
        "artifacts": 2,
        "races": 1,
        "results": 2,
        "components": 8,
    }

def test_exact_replay_detects_canonical_row_drift_instead_of_succeeding():
    engine, factory = _db()
    request = _request(item_suffix="0023")
    first = write_workflow_canonical_publication(
        request,
        session_factory=factory,
    )
    assert first["status"] == "published"

    with factory() as session:
        with session.begin():
            row = session.scalar(
                select(CanonicalElectionResult).order_by(
                    CanonicalElectionResult.source_row_index
                )
            )
            assert row is not None
            row.total_votes = row.total_votes + 1

    replay = copy.deepcopy(request)
    replay["request_id"] = "00000000-0000-0000-0000-000000000596"
    replay["idempotency_key"] = request["idempotency_key"]
    result = write_workflow_canonical_publication(
        replay,
        session_factory=factory,
    )
    assert result["status"] == "rejected"
    assert result["error"]["code"] == "canonical_conflict"
    assert result["error"]["retryable"] is False

