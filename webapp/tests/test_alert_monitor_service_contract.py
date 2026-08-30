from __future__ import annotations

import ast
import threading
from pathlib import Path

from webapp.parser.health.alert_monitor_service import AlertMonitorService


def test_alert_monitor_service_start_is_idempotent_and_stop_is_owned():
    started = threading.Event()
    calls = []

    def fake_monitor(
        poll_interval=10,
        *,
        stop_event=None,
        state=None,
        max_polls=None,
    ):
        del poll_interval, max_polls
        calls.append("started")
        state["running"] = True
        started.set()
        stop_event.wait(2)
        state["running"] = False

    service = AlertMonitorService(
        poll_interval=0.01,
        monitor_callable=fake_monitor,
    )

    first = service.start()
    assert started.wait(1)
    second = service.start()

    assert first is second
    assert calls == ["started"]
    assert service.health()["thread_alive"] is True

    assert service.stop(timeout=1)
    assert not first.is_alive()
    assert service.thread is None


def test_alert_monitor_service_integrity_import_is_lazy():
    source = (
        Path(__file__).resolve().parents[1]
        / "parser"
        / "health"
        / "alert_monitor_service.py"
    ).read_text(encoding="utf-8")

    before_resolver = source.split("def _resolve_monitor", 1)[0]
    assert "Integrity_check" not in before_resolver


def test_context_coordinator_default_is_non_monitoring_without_import():
    source = (
        Path(__file__).resolve().parents[1]
        / "parser"
        / "Context_Integration"
        / "context_coordinator.py"
    ).read_text(encoding="utf-8")

    tree = ast.parse(source)
    init_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ContextCoordinator":
            init_fn = next(
                item
                for item in node.body
                if isinstance(item, ast.FunctionDef) and item.name == "__init__"
            )
            break

    assert init_fn is not None
    args = init_fn.args.args
    defaults = init_fn.args.defaults
    offset = len(args) - len(defaults)
    observed = None
    for idx, arg in enumerate(args):
        if arg.arg == "alert_monitor":
            observed = ast.literal_eval(defaults[idx - offset])
            break

    assert observed is False


def test_webapp_does_not_start_alert_monitor_at_module_import():
    source = (
        Path(__file__).resolve().parents[1]
        / "Smart_Elections_Parser_Webapp.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)

    function_ranges = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function_ranges.append((node.lineno, node.end_lineno or node.lineno))

    start_call_lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "start"
            and isinstance(func.value, ast.Name)
            and func.value.id == "_ALERT_MONITOR_SERVICE"
        ):
            continue
        start_call_lines.append(node.lineno)

    assert len(start_call_lines) == 1
    line = start_call_lines[0]
    assert any(start <= line <= end for start, end in function_ranges)
    assert "def start_alert_monitor_service" in source


def test_gunicorn_worker_hook_and_local_main_start_service():
    root = Path(__file__).resolve().parents[2]
    gunicorn_source = (root / "gunicorn.conf.py").read_text(encoding="utf-8")
    webapp_source = (
        root / "webapp" / "Smart_Elections_Parser_Webapp.py"
    ).read_text(encoding="utf-8")

    assert "def post_worker_init(worker):" in gunicorn_source
    assert "start_alert_monitor_service()" in gunicorn_source
    assert 'if __name__ == "__main__":' in webapp_source
    assert "start_alert_monitor_service()" in webapp_source
    assert "atexit.register(stop_alert_monitor_service)" in webapp_source
