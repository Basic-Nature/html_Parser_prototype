"""J2B1 structural security regression; no DB or network."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def read(p):return (ROOT/p).read_text(encoding="utf-8")
def test_admission_default_disabled_and_has_csrf_operation():
    routes=read("webapp/parser/routes/election_projects_blueprint.py")
    assert 'ELECTION_PROJECT_RUN_ADMISSION_ENABLED' in routes
    assert 'return operation(admit,write=True,include_principal=True)' in routes
    assert '@bp.post("/api/projects/v1/<uuid:project_id>/runs")' in routes
    assert 'admission_enabled()' in routes

def test_admission_service_rechecks_authority_and_does_not_execute():
    source=read("webapp/parser/services/project_run_admission.py")
    assert 'project_run_preflight(session,' in source
    assert 'project_workflow_ballot_lens_handoff(' in source
    assert '.with_for_update()' in source
    assert 'session.flush()' in source
    assert 'socketio' not in source.lower()
    assert 'subprocess' not in source.lower()
    assert 'dispatch(' not in source

def test_ui_explicit_secondary_confirmation_and_status():
    js=read("webapp/static/js/projects.js")
    html=read("webapp/templates/projects.html")
    assert 'projectRunAdmit' in html and 'projectRunRecords' in html
    assert 'request.epoch!==preflightEpoch' in js
    assert 'idempotency_key:request.idempotency_key' in js
    assert 'run_dispatched!==false' in js
    assert 'socket.emit' not in js
