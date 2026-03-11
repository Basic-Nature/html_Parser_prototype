from __future__ import annotations

import os
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Dict

import orjson

from .logger_singleton import logger

_STARTED_AT = time.time()
_LOCK = RLock()
_TOTAL_EVENTS = 0
_COMPONENT_COUNTS: dict[str, int] = {}
_ACTION_COUNTS: dict[str, int] = {}
_RECENT_EVENTS: deque[dict[str, Any]] = deque(maxlen=max(50, int(os.environ.get("ML_TELEMETRY_RECENT_LIMIT", "300"))))

_PERSIST_ENABLED = os.environ.get("ML_TELEMETRY_PERSIST", "true").strip().lower() in {"1", "true", "yes", "on"}
_DEFAULT_LOG_PATH = Path(__file__).resolve().parents[1] / "log" / "ml_usage_telemetry.jsonl"
_LOG_PATH = Path(os.environ.get("ML_TELEMETRY_LOG_PATH", str(_DEFAULT_LOG_PATH)))
if _PERSIST_ENABLED:
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        _PERSIST_ENABLED = False


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_ml_event(
    component: str,
    action: str,
    *,
    session_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    component = (component or "unknown").strip() or "unknown"
    action = (action or "unknown").strip() or "unknown"
    event = {
        "timestamp": _iso_now(),
        "component": component,
        "action": action,
        "session_id": session_id,
        "metadata": metadata or {},
    }

    with _LOCK:
        global _TOTAL_EVENTS
        _TOTAL_EVENTS += 1
        _COMPONENT_COUNTS[component] = _COMPONENT_COUNTS.get(component, 0) + 1
        action_key = f"{component}:{action}"
        _ACTION_COUNTS[action_key] = _ACTION_COUNTS.get(action_key, 0) + 1
        _RECENT_EVENTS.append(event)

    if _PERSIST_ENABLED:
        try:
            with _LOG_PATH.open("ab") as f:
                f.write(orjson.dumps(event) + b"\n")
        except Exception as exc:
            logger.debug({
                "level": "DEBUG",
                "type": "ml_telemetry",
                "message": f"Failed to persist ml telemetry event: {exc}",
                "session_id": session_id,
            })


def get_ml_telemetry_snapshot(*, include_recent: bool = True, limit: int = 100) -> dict[str, Any]:
    with _LOCK:
        recent = list(_RECENT_EVENTS)
        if include_recent:
            recent = recent[-max(1, min(limit, 500)) :]
        else:
            recent = []

        return {
            "started_at": datetime.fromtimestamp(_STARTED_AT, timezone.utc).isoformat(),
            "uptime_sec": int(time.time() - _STARTED_AT),
            "totals": {
                "events": _TOTAL_EVENTS,
                "components": len(_COMPONENT_COUNTS),
                "actions": len(_ACTION_COUNTS),
            },
            "component_counts": dict(sorted(_COMPONENT_COUNTS.items(), key=lambda item: item[1], reverse=True)),
            "action_counts": dict(sorted(_ACTION_COUNTS.items(), key=lambda item: item[1], reverse=True)),
            "recent_events": recent,
            "persist": {
                "enabled": _PERSIST_ENABLED,
                "path": str(_LOG_PATH) if _PERSIST_ENABLED else None,
            },
        }


def reset_ml_telemetry() -> None:
    with _LOCK:
        global _TOTAL_EVENTS
        _TOTAL_EVENTS = 0
        _COMPONENT_COUNTS.clear()
        _ACTION_COUNTS.clear()
        _RECENT_EVENTS.clear()
