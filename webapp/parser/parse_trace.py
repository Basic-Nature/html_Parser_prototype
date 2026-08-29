from __future__ import annotations

"""Behavior-neutral in-memory parser trace contracts."""

from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import re
from threading import RLock
from typing import Any, Iterator
from uuid import uuid4

TRACE_CONTRACT = "parse_run_trace_v1"
ATTEMPT_CONTRACT = "parse_attempt_v1"
OBSERVATION_CONTRACT = "parse_observation_v1"
OUTCOME_CONTRACT = "parse_outcome_v1"
CONFIDENCE_CONTRACT = "parser_confidence_ledger_v1"

MAX_RECENT_TRACES = 200
MAX_ITEMS_PER_TRACE = 512
MAX_VALUE_CHARS = 2000

_OUTCOME_MAP = {
    "success": "SUCCESS",
    "partial": "PARTIAL",
    "error": "FAILED",
    "fail": "FAILED",
    "rejected": "REJECTED",
    "quarantined": "QUARANTINED",
    "cancelled": "CANCELLED",
    "cancel": "CANCELLED",
    "skipped_data_exists": "SKIPPED",
}
_SAFE_TERMINAL_META_KEYS = {
    "snapshot_mode", "fallback", "retrieved_from_database", "handler",
    "state", "county", "contest", "tables_seen", "batch_total",
    "batch_success", "batch_failures",
}
_PROVENANCE = {"OBSERVED", "DETERMINISTIC", "ML_NLP_PROPOSED", "HUMAN_REVIEWED"}

def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

_PUBLIC_URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)

def _clean(value: Any, *, depth: int = 0, redact_urls: bool = False) -> Any:
    """
    Return a bounded JSON-safe value.

    For public traces, URL redaction is enforced here so every observation,
    attempt, outcome, and nested metadata value inherits the same boundary.
    """
    try:
        if depth > 4:
            return "<max-depth>"
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            text = value[:MAX_VALUE_CHARS]
            return _PUBLIC_URL_RE.sub("<redacted-url>", text) if redact_urls else text
        if isinstance(value, dict):
            out = {}
            for key, item in list(value.items())[:100]:
                key_text = str(key)[:120]
                if redact_urls:
                    key_text = _PUBLIC_URL_RE.sub("<redacted-url>", key_text)
                out[key_text] = _clean(
                    item,
                    depth=depth + 1,
                    redact_urls=redact_urls,
                )
            return out
        if isinstance(value, (list, tuple, set)):
            return [
                _clean(
                    item,
                    depth=depth + 1,
                    redact_urls=redact_urls,
                )
                for item in list(value)[:100]
            ]
        text = str(value)[:MAX_VALUE_CHARS]
        return _PUBLIC_URL_RE.sub("<redacted-url>", text) if redact_urls else text
    except Exception:
        return "<trace-value-unavailable>"

def _public_trace(trace: dict[str, Any] | None) -> bool:
    try:
        return bool(trace and trace.get("public_runtime"))
    except Exception:
        return False

@dataclass
class _TraceState:
    trace: dict[str, Any]

_ACTIVE: ContextVar[_TraceState | None] = ContextVar("electionpulse_parse_trace", default=None)
_RECENT: deque[dict[str, Any]] = deque(maxlen=MAX_RECENT_TRACES)
_RECENT_LOCK = RLock()

def _append_bounded(collection: list[dict[str, Any]], item: dict[str, Any]) -> bool:
    try:
        if len(collection) < MAX_ITEMS_PER_TRACE:
            collection.append(item)
            return True
        return False
    except Exception:
        return False

def _active_trace() -> dict[str, Any] | None:
    try:
        state = _ACTIVE.get()
        return state.trace if state is not None else None
    except Exception:
        return None

@contextmanager
def parse_run_scope(*, session_id: str | None, source_ref: str | None, source_scope: str, public_runtime: bool) -> Iterator[dict[str, Any] | None]:
    """
    Own one trace for one source orchestration.

    Trace setup/cleanup failures degrade to no-op behavior. Exceptions raised by
    the parser body are never intercepted or rewritten by this context manager.
    """
    state = None
    token = None
    owner = False

    # Fail open only while establishing trace state.
    try:
        state = _ACTIVE.get()
        if state is None:
            owner = True
            trace = {
                "contract": TRACE_CONTRACT,
                "trace_id": f"ptrace_{uuid4().hex}",
                "session_id": _clean(session_id, redact_urls=bool(public_runtime)),
                "source_ref": None if public_runtime else _clean(source_ref),
                "source_ref_redacted": bool(public_runtime),
                "source_scope": _clean(source_scope, redact_urls=bool(public_runtime)),
                "public_runtime": bool(public_runtime),
                "started_at": _now(),
                "completed_at": None,
                "attempts": [],
                "observations": [],
                "limits": {
                    "max_items_per_collection": MAX_ITEMS_PER_TRACE,
                    "dropped_attempts": 0,
                    "dropped_observations": 0,
                },
                "confidence": {
                    "contract": CONFIDENCE_CONTRACT,
                    "acquisition": None,
                    "structure": None,
                    "semantics": None,
                    "context": None,
                    "normalization": None,
                    "reconciliation": None,
                    "coverage": None,
                },
                "outcome": None,
            }
            state = _TraceState(trace=trace)
            token = _ACTIVE.set(state)
    except Exception:
        state = None
        token = None
        owner = False

    # Deliberately do not catch exceptions thrown by the parser body here.
    try:
        yield state.trace if state is not None else None
    finally:
        if owner and state is not None:
            try:
                state.trace["completed_at"] = _now()
                with _RECENT_LOCK:
                    _RECENT.append(deepcopy(state.trace))
            except Exception:
                pass
            try:
                if token is not None:
                    _ACTIVE.reset(token)
            except Exception:
                pass

def record_parse_observation(*, kind: str, value_summary: Any, provenance: str = "OBSERVED", confidence: float | None = None, source_location: str | None = None) -> bool:
    try:
        trace = _active_trace()
        if trace is None:
            return False
        prov = provenance if provenance in _PROVENANCE else "OBSERVED"
        redact_urls = _public_trace(trace)
        conf = None
        if confidence is not None:
            try:
                conf = max(0.0, min(1.0, float(confidence)))
            except Exception:
                conf = None
        appended = _append_bounded(trace["observations"], {
            "contract": OBSERVATION_CONTRACT,
            "kind": _clean(kind, redact_urls=redact_urls),
            "value_summary": _clean(value_summary, redact_urls=redact_urls),
            "provenance": prov,
            "confidence": conf,
            "source_location": _clean(source_location, redact_urls=redact_urls),
            "observed_at": _now(),
        })
        if not appended:
            limits = trace.get("limits")
            if isinstance(limits, dict):
                limits["dropped_observations"] = int(limits.get("dropped_observations") or 0) + 1
        return True
    except Exception:
        return False

def record_parse_attempt(*, stage: str, strategy: str, selection_reason: str, status: str = "SELECTED", details: Any = None) -> bool:
    try:
        trace = _active_trace()
        if trace is None:
            return False
        now = _now()
        redact_urls = _public_trace(trace)
        appended = _append_bounded(trace["attempts"], {
            "contract": ATTEMPT_CONTRACT,
            "attempt_id": f"pattempt_{uuid4().hex}",
            "stage": _clean(stage, redact_urls=redact_urls),
            "strategy": _clean(strategy, redact_urls=redact_urls),
            "selection_reason": _clean(selection_reason, redact_urls=redact_urls),
            "started_at": now,
            "completed_at": now,
            "status": _clean(status, redact_urls=redact_urls),
            "failure_class": None,
            "failure_reason": None,
            "fallback_to": None,
            "details": _clean(details, redact_urls=redact_urls),
        })
        if not appended:
            limits = trace.get("limits")
            if isinstance(limits, dict):
                limits["dropped_attempts"] = int(limits.get("dropped_attempts") or 0) + 1
        return True
    except Exception:
        return False

def record_parse_transition(*, stage: str, from_strategy: str | None, to_strategy: str | None, reason: str) -> bool:
    return record_parse_observation(
        kind="strategy_transition",
        value_summary={"stage": stage, "from": from_strategy, "to": to_strategy, "reason": reason},
        provenance="DETERMINISTIC",
    )

def observe_terminal_outcome(*, status: Any, reason_code: str | None = None, metadata: dict[str, Any] | None = None) -> bool:
    try:
        trace = _active_trace()
        if trace is None:
            return False
        raw = str(status).strip().lower() if status is not None else ""
        normalized = _OUTCOME_MAP.get(raw, "UNSPECIFIED")
        redact_urls = _public_trace(trace)
        safe_meta = {}
        if isinstance(metadata, dict):
            for key in _SAFE_TERMINAL_META_KEYS:
                if key in metadata:
                    safe_meta[key] = _clean(metadata.get(key), redact_urls=redact_urls)
        outcome = {
            "contract": OUTCOME_CONTRACT,
            "status": normalized,
            "raw_status": _clean(status, redact_urls=redact_urls),
            "reason_code": _clean(reason_code, redact_urls=redact_urls),
            "metadata": safe_meta,
            "observed_at": _now(),
        }
        trace["outcome"] = outcome
        appended = _append_bounded(trace["observations"], {
            "contract": OBSERVATION_CONTRACT,
            "kind": "terminal_outcome_observed",
            "value_summary": deepcopy(outcome),
            "provenance": "OBSERVED",
            "confidence": 1.0,
            "source_location": "mark_url_processed",
            "observed_at": _now(),
        })
        if not appended:
            limits = trace.get("limits")
            if isinstance(limits, dict):
                limits["dropped_observations"] = int(limits.get("dropped_observations") or 0) + 1
        return True
    except Exception:
        return False

def update_confidence_dimension(name: str, value: Any) -> bool:
    try:
        trace = _active_trace()
        if trace is None:
            return False
        ledger = trace.get("confidence")
        if not isinstance(ledger, dict) or name not in ledger or name == "contract":
            return False
        ledger[name] = None if value is None else max(0.0, min(1.0, float(value)))
        return True
    except Exception:
        return False

def get_recent_parse_traces(*, limit: int = 20) -> list[dict[str, Any]]:
    try:
        count = max(0, min(int(limit), MAX_RECENT_TRACES))
        if count == 0:
            return []
        with _RECENT_LOCK:
            return deepcopy(list(_RECENT)[-count:])
    except Exception:
        return []

def clear_recent_parse_traces() -> None:
    try:
        with _RECENT_LOCK:
            _RECENT.clear()
    except Exception:
        return
