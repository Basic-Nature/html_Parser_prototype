from __future__ import annotations

import re
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME = REPO_ROOT / "webapp" / "templates" / "worklist.html"
STATIC_FIXTURE = (
    REPO_ROOT
    / "webapp"
    / "static"
    / "html"
    / "smart_elections_worklist.html"
)
JS = (
    REPO_ROOT
    / "webapp"
    / "static"
    / "js"
    / "smart_elections_worklist.js"
)


def test_runtime_worklist_uses_full_tested_workflow_dom():
    runtime = RUNTIME.read_text(encoding="utf-8")
    fixture = STATIC_FIXTURE.read_text(encoding="utf-8")

    assert "<h2>Worklist Source</h2>" in fixture
    assert "<h2>Worklist Source</h2>" in runtime

    for modal_id in (
        "modal-assign-dl",
        "modal-dl-editor",
        "modal-preqc-results",
        "modal-qc1-form",
        "modal-qc2-form",
    ):
        assert f'id="{modal_id}"' in fixture
        assert runtime.count(f'id="{modal_id}"') == 1

    for class_name in (
        "col-race-id",
        "col-state",
        "col-county",
        "col-office",
        "col-step-indicator",
        "col-dl1",
        "col-dl1-status",
        "col-dl2",
        "col-dl2-status",
        "col-preqc",
        "col-qc1",
        "col-qc2",
        "col-workflow",
        "col-actions",
    ):
        assert f'class="{class_name}"' in runtime


def test_runtime_worklist_has_no_duplicate_ids_or_retired_dblite_cards():
    runtime = RUNTIME.read_text(encoding="utf-8")

    counts = Counter(
        re.findall(r'\bid=["\']([^"\']+)["\']', runtime)
    )
    assert {
        key: value
        for key, value in counts.items()
        if value > 1
    } == {}

    for retired in (
        "DB-Lite Finalized",
        "DB-Lite Down-Ballot",
        "dblite-finalized-sheet-name",
        "dblite-finalized-row-count",
        "dblite-finalized-fetch-status",
        "dblite-down-sheet-name",
        "dblite-down-row-count",
        "dblite-down-fetch-status",
    ):
        assert retired not in runtime


def test_runtime_worklist_preserves_flask_navigation_and_csp_assets():
    runtime = RUNTIME.read_text(encoding="utf-8")

    assert "{{ url_for('ballot_lens') }}" in runtime
    assert "{{ url_for('data_framework') }}" in runtime
    assert "filename='css/smart_elections.css'" in runtime
    assert "filename='js/smart_elections_worklist.js'" in runtime
    assert "g.csp_nonce" in runtime
    assert "static_version" in runtime


def test_public_worklist_row_pseudonymizes_operator_identity_only():
    source = JS.read_text(encoding="utf-8")

    assert "publicOperatorId(value, prefix = 'DT')" in source
    assert "Math.imul(hash, 16777619)" in source
    assert "(hash % 9999) + 1" in source

    start = source.index("renderRaceRow(race) {")
    end = source.index("Render status badge", start)
    block = source[start:end]

    assert "this.publicOperatorId(race.dl1_assigned_to, 'DT')" in block
    assert "this.publicOperatorId(race.dl2_assigned_to, 'DT')" in block
    assert "this.escapeHtml(race.dl1_assigned_to" not in block
    assert "this.escapeHtml(race.dl2_assigned_to" not in block

    # Raw fields remain in application logic because this is a presentation
    # boundary, not deletion of operational/audit identity.
    assert "dl1_assigned_to" in source
    assert "dl2_assigned_to" in source
