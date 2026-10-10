"""J2A static fail-closed assertions tied to non-executing Project preview."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")

def test_socket_project_intent_never_falls_to_legacy():
    text = read("webapp/parser/socket_ballot_lens_orchestration.py")
    dispatch = text.split("def run_ballot_lens_socket_handler(", 1)[1]
    assert "PROJECT_RUN_FORBIDDEN_KEYS" in dispatch
    assert dispatch.index("PROJECT_RUN_FORBIDDEN_KEYS") < dispatch.index("_is_public_registry_intent(payload)")
    assert dispatch.index("PROJECT_RUN_FORBIDDEN_KEYS") < dispatch.index("_initialize_session_and_auth(payload, hooks)")

def test_project_preview_has_no_dispatch_or_token_api():
    routes = read("webapp/parser/routes/election_projects_blueprint.py")
    service = read("webapp/parser/services/project_run_preflight.py")
    assert '@bp.get("/api/projects/v1/<uuid:project_id>/run-preflight")' in routes
    assert "ELECTION_PROJECT_RUN_PREFLIGHT_ENABLED" in routes
    assert '@bp.post("/api/projects/v1/<uuid:project_id>/run-preflight")' not in routes
    assert "def project_run_preflight(" in service
    assert "project_workflow_ballot_lens_handoff(" in service
    assert '"run_dispatched": False' in service
    assert '"execution_authorized": False' in service
    assert '"confirmation_enabled": False' in service
    assert "socketio" not in service.lower()

def test_project_browser_explicit_review_only():
    js = read("webapp/static/js/projects.js")
    page = read("webapp/templates/projects.html")
    assert "projectRunPreflightForm" in page
    assert "Run eligibility review" in page
    assert "projectRunPreflightResult" in page
    assert "Reviewing current Project" in js
    assert "'/run-preflight'" not in js  # Request always includes project-owned path
    assert "run-preflight?" in js
    assert "No run was started" in js
    assert "socket.emit" not in js
    assert "fetch('/api/projects/v1', {method:'POST'" not in js
