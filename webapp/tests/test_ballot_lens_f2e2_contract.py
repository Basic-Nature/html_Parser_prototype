from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
F2 = ROOT / "webapp/frontend/ballot-lens"
APP_SHELL = F2 / "app/AppShell.tsx"
REGISTRY_BROWSER = F2 / "components/source/PublicRegistryBrowser.tsx"
SOURCE_PANEL = F2 / "components/source/SourcePanel.tsx"
WORKSPACE = F2 / "components/workspace/WorkspaceShell.tsx"
PUBLIC_SUBMIT = F2 / "services/publicSubmit.ts"
SOCKET_CLIENT = F2 / "services/socketClient.ts"
SOCKET_ADAPTER = F2 / "services/socketAdapter.ts"
SELECTORS = F2 / "state/selectors.ts"
RUN_MACHINE = F2 / "state/runMachine.ts"
TEMPLATE = ROOT / "webapp/templates/ballot_lens_f2.html"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def test_f2e2_lifts_safe_selection_and_execution_authority_to_app_shell():
    app = _read(APP_SHELL)
    browser = _read(REGISTRY_BROWSER)
    source_panel = _read(SOURCE_PANEL)
    selectors = _read(SELECTORS)

    assert "PublicRegistryEnvelope | null" in app
    assert "PublicRegistrySource | null" in app
    assert "selectedSourceId={selectedSource?.registry_source_id ?? ''}" in app
    assert "onRegistryEnvelopeChange" in browser
    assert "onSelectionChange" in browser
    assert "selectedSourceId" in source_panel
    assert "canSubmitApprovedRegistrySource" in selectors
    assert "envelope?.execution_enabled" in selectors
    assert "selectedSource.registry_source_id !== envelope.execution_source_id" in selectors


def test_f2e2_emits_one_registry_id_only_command():
    app = _read(APP_SHELL)
    submit = _read(PUBLIC_SUBMIT)
    client = _read(SOCKET_CLIENT)
    adapter = _read(SOCKET_ADAPTER)

    assert submit.count("socket.emit(") == 1
    assert "PUBLIC_BALLOT_LENS_COMMAND_EVENT = 'ballot_lens'" in submit
    assert "Object.freeze({ registry_source_id: normalizedSourceId })" in submit
    assert "event: 'ballot_lens'" in client
    assert "payload: PublicRegistrySubmitPayload" in client
    assert "socket.emit" not in app
    assert "emit(" not in adapter

    outbound = app + "\n" + submit + "\n" + client
    for forbidden in (
        "direct_urls",
        "target_url",
        "executable_url",
        "file_source",
        "warehouse_override",
        "session_id",
    ):
        assert forbidden not in outbound.lower()


def test_f2e2_reuses_run_machine_through_submission_accepted_only():
    app = _read(APP_SHELL)
    run_machine = _read(RUN_MACHINE)
    workspace = _read(WORKSPACE)

    assert "useReducer(" in app
    assert "reduceRunState" in app
    assert app.index("SOURCE_SELECTED") < app.index("SUBMIT_REQUESTED")
    assert app.index("SUBMIT_REQUESTED") < app.index("SUBMISSION_ACCEPTED")
    assert "SESSION_CORRELATED" not in app
    assert "CHECKPOINT_UPDATED" not in app
    assert "public_registry_result" not in app
    assert "fromTransition" in run_machine
    assert "Submission accepted" in workspace
    assert "Session correlation and parser results remain deferred" in workspace


def test_f2e2_phase_and_ui_are_honest_about_the_boundary():
    template = _read(TEMPLATE)
    browser = _read(REGISTRY_BROWSER)
    workspace = _read(WORKSPACE)

    assert 'data-f2-phase="F2-E2"' in template
    assert "F2 Approved Source Submit" in template
    assert "execution-authorized source" in browser
    assert "No owned session" in workspace
    assert "No parser result yet" in workspace
    assert "does not fabricate preview rows or vote totals" in workspace
