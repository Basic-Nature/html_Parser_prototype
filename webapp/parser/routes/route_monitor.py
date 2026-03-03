from __future__ import annotations

import threading
from datetime import datetime, timezone

from flask import current_app


_ROUTE_MONITOR_LOCK = threading.Lock()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_route_monitor_event(cluster: str, handler_name: str, outcome: str) -> None:
    try:
        app = current_app._get_current_object()
    except Exception:
        return
    if not cluster or not handler_name:
        return
    route_key = f"{cluster}:{handler_name}"
    with _ROUTE_MONITOR_LOCK:
        monitor = app.config.setdefault("_ROUTE_WRAPPER_MONITOR", {
            "created_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
            "routes": {},
        })
        routes = monitor.setdefault("routes", {})
        stats = routes.setdefault(route_key, {
            "cluster": cluster,
            "handler": handler_name,
            "dispatch": 0,
            "success": 0,
            "failure": 0,
            "last_outcome": None,
            "last_seen": None,
        })
        stats["dispatch"] = int(stats.get("dispatch", 0)) + 1
        if outcome == "success":
            stats["success"] = int(stats.get("success", 0)) + 1
        else:
            stats["failure"] = int(stats.get("failure", 0)) + 1
        stats["last_outcome"] = outcome
        stats["last_seen"] = _utc_now_iso()
        monitor["updated_at"] = stats["last_seen"]
