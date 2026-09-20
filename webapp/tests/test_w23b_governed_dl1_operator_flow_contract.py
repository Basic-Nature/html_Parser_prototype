from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from webapp.parser.services.workflow_dl1_runtime_bridge import (
    WORKFLOW_DL1_RUNTIME_BRIDGE_CONTRACT,
    complete_governed_dl1_from_trusted_run,
)
from webapp.parser.services.workflow_normalized_artifact import (
    WORKFLOW_NORMALIZED_ARTIFACT_CONTRACT,
    build_workflow_semantic,
    normalized_artifact_bytes,
)
from webapp.parser.utils.models import (
    Base,
    StagingElectionResult,
    WorkflowEvent,
    WorkflowItem,
    WorkflowPass,
)

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "webapp" / "Smart_Elections_Parser_Webapp.py"
TEMPLATE = ROOT / "webapp" / "templates" / "worklist.html"
PUBLIC_JS = ROOT / "webapp" / "static" / "js" / "workflow_public.js"
OPERATOR_JS = ROOT / "webapp" / "static" / "js" / "workflow_operator.js"
OUTPUT_UTILS = ROOT / "webapp" / "parser" / "utils" / "output_utils.py"
SOCKET = ROOT / "webapp" / "parser" / "socket_ballot_lens_orchestration.py"
TRUSTED_RUNTIME = (
    ROOT / "webapp" / "parser" / "services" / "trusted_ballot_lens_runtime.py"
)


def _headers():
    return [
        "Precinct",
        "% Precincts Reporting",
        "Election Day Total",
        "Early Voting Total",
        "Absentee Mail Total",
        "Provisional Total",
        "Jane Doe (DEM) - Election Day",
        "Jane Doe (DEM) - Early Voting",
        "Jane Doe (DEM) - Absentee Mail",
        "Jane Doe (DEM) - Provisional",
        "Jane Doe (DEM) - Total Votes",
        "John Smith (REP) - Election Day",
        "John Smith (REP) - Early Voting",
        "John Smith (REP) - Absentee Mail",
        "John Smith (REP) - Provisional",
        "John Smith (REP) - Total Votes",
        "Grand Total",
    ]


def _rows():
    return [{
        "Precinct": "P-001",
        "% Precincts Reporting": "100.00%",
        "Election Day Total": 11,
        "Early Voting Total": 7,
        "Absentee Mail Total": 2,
        "Provisional Total": 0,
        "Jane Doe (DEM) - Election Day": 6,
        "Jane Doe (DEM) - Early Voting": 4,
        "Jane Doe (DEM) - Absentee Mail": 1,
        "Jane Doe (DEM) - Provisional": 0,
        "Jane Doe (DEM) - Total Votes": 11,
        "John Smith (REP) - Election Day": 5,
        "John Smith (REP) - Early Voting": 3,
        "John Smith (REP) - Absentee Mail": 1,
        "John Smith (REP) - Provisional": 0,
        "John Smith (REP) - Total Votes": 9,
        "Grand Total": 20,
    }]


def _scope():
    return {
        "election_year": 2024,
        "election_date": "2024-11-05",
        "state": "Iowa",
        "jurisdiction_name": None,
        "jurisdiction_type": None,
        "contest": "President",
    }


def _observation():
    return {
        "contract": "parser_observation_bundle_v1",
        "authority": {
            "inspection": "noncanonical_parser_evidence",
            "canonical": False,
        },
        "source_stage": "normalized",
        "pipeline_inspection": {
            "contract": "pipeline_inspection_v1",
            "authority": {
                "inspection": "noncanonical_parser_evidence",
                "canonical": False,
            },
        },
        "election_structure": {
            "contract": "election_structure_observation_v1",
            "authority": {
                "inspection": "noncanonical_parser_evidence",
                "canonical": False,
            },
        },
        "raw_rows_included": False,
        "raw_headers_included": False,
        "automatic_timestamp": False,
    }


def test_smart_elections_rows_materialize_existing_w4_semantic_contract():
    semantic = build_workflow_semantic(_headers(), _rows(), scope=_scope())
    assert semantic["scope"] == _scope()
    assert len(semantic["records"]) == 1
    record = semantic["records"][0]
    assert record["reporting_unit"] == {"name": "P-001", "type": "precinct"}
    assert record["percent_reporting"] == {"state": "value", "value": "100"}
    assert record["vote_methods"] == [
        "Election Day",
        "Early Voting",
        "Absentee Mail",
        "Provisional",
    ]
    assert record["candidates"][0]["name"] == "Jane Doe"
    assert record["candidates"][0]["party"] == "DEM"
    assert record["candidates"][1]["name"] == "John Smith"
    assert record["grand_total"] == {"state": "value", "votes": 20}

    first = normalized_artifact_bytes(semantic)
    second = normalized_artifact_bytes(semantic)
    assert first == second
    assert WORKFLOW_NORMALIZED_ARTIFACT_CONTRACT.encode() in first
    assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest()


def test_workflow_bridge_derives_staging_evidence_pre_qc_and_submit_server_side(tmp_path):
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )
    item_id = uuid4()
    pass_id = uuid4()
    with Session() as session:
        item = WorkflowItem(
            id=item_id,
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
            row_version=2,
        )
        workflow_pass = WorkflowPass(
            id=pass_id,
            workflow_item_id=item_id,
            pass_number=1,
            pass_label="DL1",
            revision_number=1,
            is_current=True,
            status="in_progress",
            assigned_principal="principal:dl1",
        )
        session.add_all([item, workflow_pass])
        session.commit()

    registry = tmp_path / "urls.txt"
    registry.write_text(
        "# === Curated | W23B ===\n"
        "2024\tPresident\tIowa\tstatewide\tPDF\tCertified\t"
        "https://sos.example.gov/results.pdf\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "output"
    output_folder = output_root / "w23b"
    output_folder.mkdir(parents=True)
    csv_path = output_folder / "results.csv"
    csv_path.write_text("fixture\n", encoding="utf-8")
    metadata_path = output_folder / "results.metadata.json"
    metadata_path.write_text("{}\n", encoding="utf-8")

    result = complete_governed_dl1_from_trusted_run(
        session_factory=Session,
        workflow_item_id=item_id,
        workflow_pass_id=pass_id,
        principal="principal:dl1",
        expected_row_version=2,
        capture={
            "headers": _headers(),
            "rows": _rows(),
            "csv_path": str(csv_path),
            "metadata_path": str(metadata_path),
            "observations": [_observation()],
        },
        registry_path=registry,
        output_root=output_root,
    )
    assert result["contract"] == WORKFLOW_DL1_RUNTIME_BRIDGE_CONTRACT
    assert result["success"] is True
    assert result["status"] == "submitted"
    assert result["row_version"] == 3
    assert result["pre_qc_complete"] is True
    assert result["staging_row_count"] == 1
    assert result["source_evidence_ref"].startswith("output://w23b/")
    assert "#sha256=" in result["source_evidence_ref"]
    assert result["normalized_artifact_ref"] == (
        "output://w23b/workflow_normalized_semantic.json"
    )

    with Session() as session:
        item = session.get(WorkflowItem, item_id)
        workflow_pass = session.get(WorkflowPass, pass_id)
        assert item.row_version == 3
        assert item.stage_condition == "ready"
        assert workflow_pass.status == "submitted"
        assert workflow_pass.candidate_check_status == "complete"
        assert workflow_pass.semantic_validation_status == "complete"
        assert workflow_pass.source_evidence_ref == result["source_evidence_ref"]
        staged = (
            session.query(StagingElectionResult)
            .filter(StagingElectionResult.batch_id == workflow_pass.staging_batch_id)
            .all()
        )
        assert len(staged) == 1
        assert staged[0].metastats["normalized_row"]["Grand Total"] == 20
        events = [
            row.event_type
            for row in session.query(WorkflowEvent)
            .filter(WorkflowEvent.workflow_item_id == item_id)
            .all()
        ]
        assert "staging_binding_started" in events
        assert "staging_binding_completed" in events
        assert "pre_qc_pass_validated" in events
        assert "pass_submitted" in events

    engine.dispose()


def test_public_workflow_client_remains_get_only_and_operator_client_is_separate():
    public_source = PUBLIC_JS.read_text(encoding="utf-8")
    operator_source = OPERATOR_JS.read_text(encoding="utf-8")
    template = TEMPLATE.read_text(encoding="utf-8")

    assert "workflow:public-items-rendered" in public_source
    for token in (
        "method: 'POST'",
        "method: 'PUT'",
        "method: 'PATCH'",
        "method: 'DELETE'",
    ):
        assert token not in public_source

    assert "workflow_operator.js" in template
    assert "workflow.dl1.claim" in template
    assert "workflow_operator_access" in template

    assert "method: 'POST'" in operator_source
    assert "/passes/1/claim" in operator_source
    assert "/passes/1/submit" not in operator_source
    assert "/passes/2/" not in operator_source
    assert "/reviewer/" not in operator_source
    assert "/publication/" not in operator_source
    assert "source_url_editable !== false" in operator_source
    assert "arbitrary_url_execution !== false" in operator_source
    authority_tokens_removed = (
        operator_source
        .replace("source_url_editable", "")
        .replace("source_url_disclosed", "")
    )
    assert "source_url" not in authority_tokens_removed
    assert "expected_row_version" in operator_source
    assert "workflow_item_id" in operator_source
    assert "workflow_pass_id" in operator_source


def test_server_owns_submit_authority_and_all_mutation_flags_remain_default_off():
    app = APP.read_text(encoding="utf-8")
    output_utils = OUTPUT_UTILS.read_text(encoding="utf-8")
    socket = SOCKET.read_text(encoding="utf-8")
    runtime = TRUSTED_RUNTIME.read_text(encoding="utf-8")

    assert '_WORKFLOW_DL1_CLAIM_REQUEST_KEYS = frozenset({"expected_row_version"})' in app
    assert "body_keys != _WORKFLOW_DL1_CLAIM_REQUEST_KEYS" in app
    assert 'os.environ.get("WORKFLOW_CONTRIBUTOR_MUTATIONS_ENABLED", "false")' in app
    assert 'os.environ.get("WORKFLOW_REVIEWER_MUTATIONS_ENABLED", "false")' in app
    assert 'os.environ.get("WORKFLOW_PUBLICATION_MUTATIONS_ENABLED", "false")' in app

    assert '"capture_finalized_output"' in output_utils
    assert "capture_finalized(" in output_utils
    assert '"capture_persisted_output"' in output_utils
    assert "capture_persisted(" in output_utils
    assert '"capture_parser_observation"' in socket
    assert "capture_observation(payload)" in socket
    assert "complete_governed_dl1_from_trusted_run" in socket
    assert "WORKFLOW_CONTRIBUTOR_MUTATIONS_ENABLED" in socket
    assert "workflow_completion_capture()" in socket
    assert "record_workflow_completion(" in socket

    assert "capture_parser_observation" in runtime
    assert "capture_finalized_output" in runtime
    assert "workflow_completion_capture" in runtime
    assert "record_workflow_completion" in runtime
