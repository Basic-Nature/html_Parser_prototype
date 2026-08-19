# Contracts for Tranche 1H alert-monitor lifecycle and observability.

from __future__ import annotations

from contextlib import contextmanager
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import webapp.parser.Context_Integration.Integrity_check as integrity
import webapp.parser.Context_Integration.context_coordinator as coordinator_mod


ROOT = Path(__file__).resolve().parents[2]
INTEGRITY_PATH = (
    ROOT
    / "webapp"
    / "parser"
    / "Context_Integration"
    / "Integrity_check.py"
)
COORDINATOR_PATH = (
    ROOT
    / "webapp"
    / "parser"
    / "Context_Integration"
    / "context_coordinator.py"
)
SHARED_LOGIC_PATH = (
    ROOT / "webapp" / "parser" / "utils" / "shared_logic.py"
)


class _Rows:
    def __init__(self, values):
        self._values = list(values)

    def all(self):
        return list(self._values)


class _Result:
    def __init__(self, values):
        self._values = values

    def scalars(self):
        return _Rows(self._values)


class _Session:
    def __init__(self, values=None, error=None):
        self.values = list(values or [])
        self.error = error

    def execute(self, _stmt):
        if self.error is not None:
            raise self.error
        return _Result(self.values)


def _session_factory(session):
    @contextmanager
    def _factory():
        yield session

    return _factory


def test_monitor_function_no_longer_spawns_a_thread():
    import ast

    source = INTEGRITY_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "monitor_db_for_alerts"
    )
    segment = ast.get_source_segment(source, node) or ""

    assert "threading.Thread(" not in segment


def test_bounded_monitor_poll_updates_success_health(monkeypatch):
    events = []
    state = {}
    row = SimpleNamespace(
        id=7,
        msg="Example alert",
        context="test",
        level="INFO",
    )

    monkeypatch.setattr(
        integrity,
        "get_session",
        _session_factory(_Session([row])),
    )
    monkeypatch.setattr(
        integrity,
        "log_integrity_monitor",
        lambda event, **_kwargs: events.append(dict(event)),
    )

    integrity.monitor_db_for_alerts(
        poll_interval=0.01,
        state=state,
        max_polls=1,
    )

    assert state["running"] is False
    assert state["db_available"] is True
    assert state["poll_count"] == 1
    assert state["alerts_seen"] == 1
    assert state["last_alert_id"] == 7
    assert state["consecutive_failures"] == 0
    assert state["last_success_at"] is not None

    names = [event.get("event") for event in events]
    assert "alert_monitor_started" in names
    assert "alert_monitor_alert" in names
    assert "alert_monitor_stopped" in names


def test_monitor_preserves_failure_stage_and_exception_type(monkeypatch):
    events = []
    state = {}

    monkeypatch.setattr(
        integrity,
        "get_session",
        _session_factory(
            _Session(
                error=RuntimeError("database unavailable"),
            )
        ),
    )
    monkeypatch.setattr(
        integrity,
        "log_integrity_monitor",
        lambda event, **_kwargs: events.append(dict(event)),
    )

    integrity.monitor_db_for_alerts(
        poll_interval=0.01,
        state=state,
        max_polls=1,
    )

    assert state["running"] is False
    assert state["db_available"] is False
    assert state["consecutive_failures"] == 1
    assert state["last_failure_stage"] == "session_execute"
    assert state["last_error_type"] == "RuntimeError"
    assert "database unavailable" in state["last_error_message"]

    failure = next(
        event
        for event in events
        if event.get("event") == "alert_monitor_failure"
    )
    assert failure["failure_stage"] == "session_execute"
    assert failure["error_type"] == "RuntimeError"


def test_monitor_honors_stop_event_without_waiting_full_interval(monkeypatch):
    state = {}
    stop_event = threading.Event()

    monkeypatch.setattr(
        integrity,
        "get_session",
        _session_factory(_Session([])),
    )
    monkeypatch.setattr(
        integrity,
        "log_integrity_monitor",
        lambda *_args, **_kwargs: None,
    )

    thread = threading.Thread(
        target=integrity.monitor_db_for_alerts,
        kwargs={
            "poll_interval": 30,
            "stop_event": stop_event,
            "state": state,
        },
        daemon=True,
    )
    thread.start()

    deadline = time.time() + 2
    while state.get("poll_count", 0) < 1 and time.time() < deadline:
        time.sleep(0.01)

    stop_event.set()
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert state["running"] is False


def test_coordinator_owns_the_actual_poller_thread(monkeypatch):
    started = threading.Event()

    def fake_monitor(
        poll_interval=10,
        *,
        stop_event=None,
        state=None,
        max_polls=None,
    ):
        del poll_interval, max_polls
        state["running"] = True
        started.set()
        stop_event.wait(2)
        state["running"] = False

    monkeypatch.setattr(
        coordinator_mod,
        "monitor_db_for_alerts",
        fake_monitor,
    )

    coordinator = coordinator_mod.ContextCoordinator.__new__(
        coordinator_mod.ContextCoordinator
    )
    coordinator.alert_monitor = True
    coordinator.alert_monitor_thread = None
    coordinator.alert_monitor_stop_event = threading.Event()
    coordinator.alert_monitor_state = {}

    thread = coordinator.start_alert_monitoring(background=True)

    assert thread is coordinator.alert_monitor_thread
    assert started.wait(1)
    assert thread.is_alive()

    same = coordinator.start_alert_monitoring(background=True)
    assert same is thread

    assert coordinator.stop_alert_monitoring(timeout=1)
    assert not thread.is_alive()
    assert coordinator.alert_monitor_thread is None


def test_coordinator_health_snapshot_exposes_lifecycle(monkeypatch):
    coordinator = coordinator_mod.ContextCoordinator.__new__(
        coordinator_mod.ContextCoordinator
    )
    coordinator.alert_monitor = False
    coordinator.alert_monitor_thread = None
    coordinator.alert_monitor_stop_event = threading.Event()
    coordinator.alert_monitor_state = {}

    coordinator.alert_monitor_state.update(
        {
            "running": False,
            "db_available": False,
            "consecutive_failures": 3,
            "last_failure_stage": "session_execute",
        }
    )

    snapshot = coordinator.get_alert_monitor_health()

    assert snapshot["configured"] is False
    assert snapshot["thread_alive"] is False
    assert snapshot["stop_requested"] is False
    assert snapshot["consecutive_failures"] == 3
    assert snapshot["last_failure_stage"] == "session_execute"


def test_destructor_delegates_to_explicit_stop_method():
    source = COORDINATOR_PATH.read_text(encoding="utf-8")
    assert "self.stop_alert_monitoring(timeout=1.0)" in source


def test_monitor_no_longer_uses_lossy_safe_execute_or_safe_all():
    source = INTEGRITY_PATH.read_text(encoding="utf-8")
    function_source = source[source.index("def monitor_db_for_alerts("):]
    function_source = function_source.split("\ndef ", 1)[0]

    assert "safe_execute(" not in function_source
    assert "safe_all(" not in function_source


def test_shared_safe_helpers_are_not_modified_by_1h():
    source = SHARED_LOGIC_PATH.read_text(encoding="utf-8")
    assert "def safe_execute(" in source
    assert "def safe_all(" in source
