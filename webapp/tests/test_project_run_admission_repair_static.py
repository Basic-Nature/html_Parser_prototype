"""J2B1B repair contract guard: the admission ledger cannot dispatch a parser."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
def read(path): return (ROOT/path).read_text(encoding="utf-8")

def test_migration_graph_and_admission_only_state():
    migration=read("alembic/versions/b62e1c490d33_project_run_admission_ledger.py")
    project_migration=read("alembic/versions/ab8a7b16e24f_election_project_foundation.py")
    model=read("webapp/parser/models/election_project.py")
    assert 'down_revision = "ab8a7b16e24f"' in migration
    assert 'revision = "ab8a7b16e24f"' in project_migration
    assert migration.count('"status = \'admitted\'"') == 1
    assert model.count('"status = \'admitted\'"') == 1

def test_idempotent_replay_and_future_state_fail_closed():
    svc=read("webapp/parser/services/project_run_admission.py")
    assert 'with session.begin_nested():' in svc
    assert 'except IntegrityError:' in svc
    assert 'winner.request_fingerprint != digest' in svc
    assert 'if run.status != "admitted":' in svc
    assert 'project_run_state_not_supported' in svc
    assert 'socketio' not in svc.lower() and 'subprocess' not in svc.lower()

def test_browser_auto_discovers_without_granting_execution():
    js=read("webapp/static/js/projects.js")
    assert 'void refreshProjectRuns();' in js
    assert 'admissionReady.epoch===preflightEpoch' in js
    assert "run.status!=='admitted'" in js
    assert 'socket.emit' not in js
