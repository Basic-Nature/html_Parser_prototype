"""Projects initial-load classification; never equate a generic 503 with feature disabled."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROJECT_JS = ROOT / "webapp/static/js/projects.js"


def _entry_classifier_source() -> str:
    source = PROJECT_JS.read_text(encoding="utf-8")
    match = re.search(
        r"  function projectEntryState\(status, code\) \{\n"
        r"(?:[^{}]*\n)+?  \}", source
    )
    assert match, "Projects initial-load classifier must remain explicit and locally testable"
    return match.group(0)


def test_entry_failure_ui_wiring_is_restricted_only():
    source = PROJECT_JS.read_text(encoding="utf-8")
    assert "const entryState=projectEntryState(e.status,e.message);" in source
    assert "byId('projectAccessHelp').hidden=(entryState!=='restricted');" in source
    assert "'Projects could not be loaded. Please try again later.'" in source
    assert "msg(e.status===503?" not in source


@pytest.mark.parametrize("status,code,expected", [
    (503, "projects_not_enabled", "disabled"),
    (503, "project_store_unavailable", "unavailable"),
    (503, "project_identity_key_unconfigured", "unavailable"),
    (503, "project_run_admission_disabled", "unavailable"),
    (401, "contributor_access_required", "restricted"),
    (403, "contributor_access_required", "restricted"),
    (500, "internal_error", "unavailable"),
    (404, "request_failed", "unavailable"),
    (0, "network_failure", "unavailable"),
])
def test_entry_failure_state_in_isolated_node(status, code, expected):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is unavailable; browser-state cases need JS execution")
    js = (_entry_classifier_source() + "\nconsole.log(projectEntryState(" +
          str(status) + "," + repr(code) + "));\n")
    result = subprocess.run([node, "-e", js], capture_output=True,
                            text=True, timeout=8, check=True)
    assert result.stdout.strip() == expected
