"""Regression: Project request discovery stays independent of a new review epoch."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def test_project_scoped_discovery_is_not_preflight_epoch_scoped():
    text=(ROOT/'webapp/static/js/projects.js').read_text()
    function=text.split('  async function refreshProjectRuns() {',1)[1].split("  byId('projectRefreshRuns')",1)[0]
    assert 'epoch!==preflightEpoch' not in function
    assert 'admissionReady.epoch===preflightEpoch' in function
    assert 'admissionReady.project_id===id' in function
    assert 'admissionReady.expected_project_version===version' in function
    assert 'active.id!==id' in function and 'active.row_version!==version' in function
