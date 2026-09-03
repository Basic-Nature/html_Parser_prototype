from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
F2 = ROOT / "webapp/frontend/ballot-lens"
APP_SHELL = F2 / "app/AppShell.tsx"
LIFECYCLE = F2 / "services/publicRuntimeLifecycle.ts"
PACKAGE = F2 / "package.json"
BACKEND = ROOT / "webapp/parser/socket_ballot_lens_orchestration.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def test_f2e3e4_app_owns_listener_lifecycle_and_server_session():
    app = _read(APP_SHELL)
    lifecycle = _read(LIFECYCLE)

    assert "installPublicRuntimeLifecycle" in app
    assert "getRunState: () => runStateRef.current" in app
    assert "selectedSourceRef.current?.registry_source_id" in app
    assert "detachLifecycle()" in app
    assert "socket.disconnect()" in app
    assert "SESSION_CORRELATED" in lifecycle
    assert "public_registry_runtime_started" not in app


def test_f2e3e4_routes_exact_events_through_existing_authorities():
    lifecycle = _read(LIFECYCLE)

    assert "PUBLIC_SOCKET_OBSERVATION_EVENTS" in lifecycle
    assert "normalizePublicSocketObservation" in lifecycle
    assert "CONNECTION_ESTABLISHED" in lifecycle
    assert "CONNECTION_LOST" in lifecycle
    assert "CONNECTION_RESTORED" in lifecycle
    assert "RUN_TERMINATED" in lifecycle
    assert "observation.result.registry_source_id !== selectedRegistrySourceId" in lifecycle


def test_f2e3e4_handoff_remains_typed_and_memory_only():
    app = _read(APP_SHELL)
    lifecycle = _read(LIFECYCLE)
    package = _read(PACKAGE)

    assert "PublicRuntimeResult | null" in app
    assert "setPublicRuntimeResult" in app
    assert "runtimeResult={publicRuntimeResult}" in app
    assert "persistence: 'memory_only'" in lifecycle
    assert "downloadAvailable: false" in lifecycle
    assert "publicRuntimeLifecycle.test.ts" in package


def test_f2e3e4_does_not_require_backend_contract_mutation():
    backend = _read(BACKEND)

    assert 'reason_code="public_registry_runtime_started"' in backend
    assert '"public_registry_result"' in backend
