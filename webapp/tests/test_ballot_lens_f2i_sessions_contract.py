from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
F2 = ROOT / "webapp/frontend/ballot-lens"
APP = F2 / "app/AppShell.tsx"
DIAGNOSTICS = F2 / "components/diagnostics/DiagnosticsDrawer.tsx"
SWITCHER = F2 / "components/sessions/SessionSwitcher.tsx"
HISTORY = F2 / "state/sessionHistory.ts"
RUNTIME = F2 / "contracts/runtime.ts"
RUN_MACHINE = F2 / "state/runMachine.ts"
SELECTORS = F2 / "state/selectors.ts"
PACKAGE = F2 / "package.json"
CSS = F2 / "styles/shell.css"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def test_f2i_reuses_server_owned_session_authority_without_protocol_mutation():
    runtime = read(RUNTIME)
    machine = read(RUN_MACHINE)
    history = read(HISTORY)

    assert "readonly sessionId: string | null;" in runtime
    assert "SESSION_CORRELATED" in machine
    assert "function ownsSession(state: RunState, sessionId: string)" in machine
    assert "!ownsSession(state, event.sessionId)" in machine
    assert "incoming.sequence <= current.sequence" in machine

    assert "state.context.sessionId" in history
    assert "if (!sessionId)" in history
    assert "captureOwnedSession" in history
    assert "RunEvent" in history
    assert "RunState" in history


def test_f2i_session_switcher_is_view_only_and_has_no_command_authority():
    switcher = read(SWITCHER)
    history = read(HISTORY)
    diagnostics = read(DIAGNOSTICS)
    combined = "\n".join((switcher, history, diagnostics)).lower()

    assert "view only" in switcher.lower()
    assert "onselect(session.sessionid)" in switcher.lower()
    assert "session history" in switcher.lower()
    assert "historical view" in diagnostics.lower()
    normalized_diagnostics = " ".join(diagnostics.split()).lower()
    assert "never changes parser authority" in normalized_diagnostics

    for forbidden in (
        "socket.emit",
        "socket.on",
        "fetch(",
        "direct_urls",
        "target_url",
        "executable_url",
        "session_id",
        "restoresession",
        "resumesession",
        "rebindsession",
    ):
        assert forbidden not in combined


def test_f2i_app_owns_history_observation_without_changing_execution_gate():
    app = read(APP)
    selectors = read(SELECTORS)

    assert "captureOwnedSession(current, nextState, event)" in app
    assert "event.type === 'SESSION_CORRELATED'" in app
    assert "sessionHistory={sessionHistory}" in app
    assert "selectedSessionId={diagnosticSessionId}" in app
    assert "onSelectSession={setDiagnosticSessionId}" in app

    assert "canSubmitApprovedRegistrySource(registryEnvelope, selectedSource)" in app
    assert "!envelope?.execution_enabled" in selectors
    assert "selectedSource.registry_source_id !== envelope.execution_source_id" in selectors
    assert "return state.status === 'source_selected';" in selectors


def test_f2i_diagnostics_exposes_real_observed_state_without_fake_results():
    diagnostics = read(DIAGNOSTICS)
    history = read(HISTORY)

    assert "Diagnostics &amp; audit trail" in diagnostics
    assert "No correlated runtime events yet." in diagnostics
    assert "Server-created session correlated." in diagnostics
    assert "checkpointCount" in diagnostics
    assert "outputCount" in diagnostics
    assert "observation.eventType" in diagnostics
    assert "checkpointSequence" in diagnostics

    assert "Date.now" not in history
    assert "Math.random" not in history
    assert "fake" not in diagnostics.lower()
    assert "mock" not in diagnostics.lower()


def test_f2i_frontend_contract_and_styles_are_owned_by_isolated_f2_package():
    package = json.loads(read(PACKAGE))
    css = read(CSS)

    assert package["scripts"]["test:contracts"] == (
        "vitest run tests/runMachine.test.ts tests/registry.test.ts "
        "tests/publicRuntime.test.ts tests/socketAdapter.test.ts "
        "tests/publicRuntimeLifecycle.test.ts tests/f2fWorkspace.test.ts "
        "tests/trustedExecution.test.ts tests/sessionHistory.test.ts "
        "--environment node"
    )
    assert ".blf2-session-switcher" in css
    assert ".blf2-session-events" in css
    assert ".blf2-session-facts" in css
    assert "!important" not in css
