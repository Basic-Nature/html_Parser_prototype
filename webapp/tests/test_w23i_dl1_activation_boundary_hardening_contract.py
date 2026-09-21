from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from flask import Flask
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from webapp.parser.auth.workflow_csrf import WorkflowCsrfError, assert_workflow_csrf_token, issue_workflow_csrf_token
from webapp.parser.services.workflow_actions import claim_first_workflow_pass
from webapp.parser.services.workflow_dl1_canary_control import (
    ENV_CLAIM_ENABLED, ENV_COMPLETION_ENABLED, ENV_EXPECTED_ROW_VERSION,
    ENV_ITEM_ID, ENV_PRINCIPAL_SHA256, ENV_RELEASE_ENABLED, ENV_SOURCE_SHA256,
    WORKFLOW_DL1_CANARY_RELEASE_CONTRACT, assert_dl1_canary_claim_binding,
    assert_dl1_canary_completion_context, load_dl1_canary_config,
    release_dl1_canary_claim,
)
from webapp.parser.utils.models import Base, WorkflowEvent, WorkflowItem, WorkflowPass

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_PATH = REPO_ROOT / "webapp" / "Smart_Elections_Parser_Webapp.py"
SOCKET_PATH = REPO_ROOT / "webapp" / "parser" / "socket_ballot_lens_orchestration.py"
OPERATOR_PATH = REPO_ROOT / "webapp" / "static" / "js" / "workflow_operator.js"
WORKLIST_PATH = REPO_ROOT / "webapp" / "templates" / "worklist.html"
BLUEPRINT_PATH = REPO_ROOT / "webapp" / "parser" / "routes" / "workflow_contributor_blueprint.py"
ACTIONS_PATH = REPO_ROOT / "webapp" / "parser" / "services" / "workflow_actions.py"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
    session = Session()
    try:
        yield session
    finally:
        session.rollback(); session.close(); engine.dispose()


@pytest.fixture()
def registry(tmp_path: Path) -> Path:
    path = tmp_path / "urls.txt"
    path.write_text(
        "# === Curated | W23I test ===\n"
        "2024\tPresident\tIowa\tstatewide\tPDF\tCertified\thttps://sos.example.gov/results.pdf\n",
        encoding="utf-8",
    )
    return path


def _seed_item(session):
    item = WorkflowItem(
        id=uuid4(), lifecycle_state="queued", current_stage="source_intake",
        stage_condition="pending", priority=0, election_year=2024,
        election_date=None, state="Iowa", jurisdiction_name=None,
        jurisdiction_type=None, contest="President", office_basic="President",
        election_type=None, source_race_id="2024PRESIA",
        source_url="https://sos.example.gov/results.pdf", canonical_race_id=None,
        blocked_reason_code=None, blocker_detail=None, created_by_principal=None,
        workflow_metadata={}, row_version=1,
    )
    session.add(item); session.commit(); return item


def _env(item_id, row_version=1):
    return {
        ENV_CLAIM_ENABLED: "true", ENV_COMPLETION_ENABLED: "true", ENV_RELEASE_ENABLED: "true",
        ENV_ITEM_ID: str(item_id), ENV_EXPECTED_ROW_VERSION: str(row_version),
        ENV_PRINCIPAL_SHA256: _sha("cert:canary"),
        ENV_SOURCE_SHA256: _sha("https://sos.example.gov/results.pdf"),
    }


def test_granular_flags_split_broad_gate():
    app = APP_PATH.read_text(encoding="utf-8")
    socket = SOCKET_PATH.read_text(encoding="utf-8")
    assert 'os.environ.get("WORKFLOW_CONTRIBUTOR_MUTATIONS_ENABLED", "false")' in app
    assert app.count("if not WORKFLOW_CONTRIBUTOR_MUTATIONS_ENABLED:") == 0
    assert 'os.environ.get("WORKFLOW_DL1_DIRECT_SUBMIT_MUTATIONS_ENABLED", "false")' in app
    assert 'os.environ.get("WORKFLOW_DL2_MUTATIONS_ENABLED", "false")' in app
    assert '"WORKFLOW_CONTRIBUTOR_MUTATIONS_ENABLED"' not in socket
    assert "canary_config.completion_enabled" in socket
    claim_region = app[app.index("def api_workflow_v1_claim_first_pass"):app.index("_WORKFLOW_DL1_SUBMIT_REQUEST_KEYS")]
    assert "canary_config.claim_enabled" in claim_region
    submit_region = app[app.index("def api_workflow_v1_submit_first_pass"):app.index("def api_workflow_v1_claim_second_pass")]
    assert "WORKFLOW_DL1_DIRECT_SUBMIT_MUTATIONS_ENABLED" in submit_region
    dl2_region = app[app.index("def api_workflow_v1_claim_second_pass"):app.index("_WORKFLOW_REVIEWER_RESOLUTION_REQUEST_KEYS")]
    assert dl2_region.count("WORKFLOW_DL2_MUTATIONS_ENABLED") >= 2


def test_workflow_operator_has_csrf_and_remains_claim_only():
    operator = OPERATOR_PATH.read_text(encoding="utf-8")
    template = WORKLIST_PATH.read_text(encoding="utf-8")
    app = APP_PATH.read_text(encoding="utf-8")
    assert operator.count("method: 'POST'") == 1
    assert "/passes/1/claim" in operator
    assert "/passes/1/submit" not in operator and "/passes/2/" not in operator
    assert "X-CSRFToken" in operator and "workflowCsrfToken" in operator
    assert "data-workflow-csrf-token" in template
    assert "workflow_dl1_canary_claim_enabled" in template
    assert "issue_workflow_csrf_token" in app
    assert app.count("assert_workflow_csrf_token(") >= 5


def test_csrf_round_trip():
    app = Flask(__name__); app.secret_key = "w23i-test-secret"; app.config["WTF_CSRF_TIME_LIMIT"] = None
    with app.test_request_context("/worklist"):
        token = issue_workflow_csrf_token(); assert token; assert_workflow_csrf_token(token)
        with pytest.raises(WorkflowCsrfError): assert_workflow_csrf_token("invalid")


def test_canary_release_and_retry_preserve_revisions(db_session, registry):
    item = _seed_item(db_session); config = load_dl1_canary_config(_env(item.id, 1))
    assert_dl1_canary_claim_binding(db_session, item.id, principal="cert:canary", expected_row_version=1, registry_path=registry, config=config)
    claim1 = claim_first_workflow_pass(db_session, item.id, principal="cert:canary", expected_row_version=1, now=datetime(2026,9,20,23,0,tzinfo=timezone.utc))
    assert claim1["revision_number"] == 1 and claim1["row_version"] == 2
    pass1 = db_session.get(WorkflowPass, UUID(claim1["pass_id"]))
    assert_dl1_canary_completion_context(workflow_item_id=item.id, principal="cert:canary", expected_row_version=2, config=config)
    released = release_dl1_canary_claim(db_session, item.id, pass1.id, principal="cert:canary", expected_row_version=2, registry_path=registry, config=config, now=datetime(2026,9,20,23,5,tzinfo=timezone.utc))
    assert released["contract"] == WORKFLOW_DL1_CANARY_RELEASE_CONTRACT and released["row_version"] == 3
    assert pass1.status == "released" and pass1.is_current is False
    assert db_session.query(WorkflowEvent).filter(WorkflowEvent.event_type == "pass_claim_released").count() == 1
    config2 = load_dl1_canary_config(_env(item.id, 3))
    assert_dl1_canary_claim_binding(db_session, item.id, principal="cert:canary", expected_row_version=3, registry_path=registry, config=config2)
    claim2 = claim_first_workflow_pass(db_session, item.id, principal="cert:canary", expected_row_version=3, now=datetime(2026,9,20,23,10,tzinfo=timezone.utc))
    assert claim2["revision_number"] == 2 and claim2["row_version"] == 4
    pass2 = db_session.get(WorkflowPass, UUID(claim2["pass_id"]))
    assert pass2.is_current is True and pass2.revision_number == 2 and pass1.revision_number == 1


def test_release_route_exists_but_operator_does_not_expose_it():
    blueprint = BLUEPRINT_PATH.read_text(encoding="utf-8")
    operator = OPERATOR_PATH.read_text(encoding="utf-8")
    assert "/passes/1/release-canary" in blueprint
    assert "api_workflow_v1_release_first_pass_canary" in blueprint
    assert "release-canary" not in operator


def test_first_claim_revision_monotonic_source_contract():
    actions = ACTIONS_PATH.read_text(encoding="utf-8")
    assert "func.max(WorkflowPass.revision_number)" in actions
    assert "next_revision_number" in actions
    assert '"revision_number": workflow_pass.revision_number' in actions
