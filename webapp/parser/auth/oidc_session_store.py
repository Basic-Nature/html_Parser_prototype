"""W24AUTHOIDC1E — bounded session interface and localhost-only memory store.

Only an opaque random SID goes in the signed Flask cookie; no authorization
record is stored in the cookie. MemorySessionStore has NO cross-process
sharing and MUST NOT be used for production, Azure, or multi-worker login.
It is opt-in localhost testing only. Expiry is bounded to five minutes.
"""
from __future__ import annotations

import json
import re
import secrets
import time
import threading

PREFIX = "electionpulse:oidc:v2:"
MAX_DEV_SESSIONS = 128
MAX_SESSION_SECONDS = 300
_SID_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_PRINCIPAL_PATTERN = re.compile(r"oidc:keycloak:[0-9a-f]{64}\Z")


def issue_session(store, *, principal: str, project_access: bool,
                  token_expires_at: int, now: int | None = None) -> str:
    current = int(time.time()) if now is None else now
    if (not isinstance(principal, str) or not _PRINCIPAL_PATTERN.fullmatch(principal)
        or type(project_access) is not bool or type(token_expires_at) is not int
        or token_expires_at <= current):
        raise ValueError("invalid_session_data")
    ttl = min(MAX_SESSION_SECONDS, token_expires_at - current)
    record = {"principal": principal, "project_access": project_access,
              "issued_at": current, "expires_at": current + ttl}
    for _ in range(3):
        sid = secrets.token_hex(32)
        # Backend must supply atomic NX + bounded TTL. Local memory only
        # supports one process; a future production backend needs a new gate.
        if store.set(PREFIX + sid, json.dumps(record, sort_keys=True), ex=ttl, nx=True):
            return sid
    raise RuntimeError("session_creation_unavailable")


def read_session(store, sid: object, *, now: int | None = None) -> dict | None:
    if not isinstance(sid, str) or not _SID_PATTERN.fullmatch(sid):
        return None
    raw = store.get(PREFIX + sid)
    if raw is None:
        return None
    try:
        record = json.loads(raw)
        current = int(time.time()) if now is None else now
        issued = record["issued_at"]
        expiry = record["expires_at"]
        if (not isinstance(record, dict) or
            not isinstance(record.get("principal"), str) or
            not _PRINCIPAL_PATTERN.fullmatch(record["principal"]) or
            type(record.get("project_access")) is not bool or
            type(issued) is not int or type(expiry) is not int or
            issued > current + 30 or expiry <= current or
            expiry - issued > MAX_SESSION_SECONDS or issued > expiry):
            return None
        return record
    except (ValueError, KeyError, TypeError):
        return None


def revoke_session(store, sid: object) -> bool:
    if not isinstance(sid, str) or not _SID_PATTERN.fullmatch(sid):
        return False
    return bool(store.delete(PREFIX + sid))


class MemorySessionStore:
    """Thread-safe bounded TTL store for explicitly gated localhost development.

    Private process memory only. Uses monotonic expiry to avoid wall-clock
    extension; no storage, redis client, network I/O, or cross-worker access.
    A full store denies new sessions rather than evicting an active user.
    """

    def __init__(self, *, max_sessions: int = MAX_DEV_SESSIONS, clock=None):
        if type(max_sessions) is not int or not (1 <= max_sessions <= MAX_DEV_SESSIONS):
            raise ValueError("invalid_session_capacity")
        self._lock = threading.Lock()
        self._items: dict[str, tuple[str, float]] = {}
        self._capacity = max_sessions
        self._clock = clock or time.monotonic

    def _prune(self, current: float) -> None:
        expired = [k for k, (_, deadline) in self._items.items() if deadline <= current]
        for k in expired:
            self._items.pop(k, None)

    def set(self, key: str, value: str, *, ex: int, nx: bool) -> bool:
        if (not isinstance(key, str) or not key.startswith(PREFIX) or
            not isinstance(value, str) or type(ex) is not int or
            not (1 <= ex <= MAX_SESSION_SECONDS) or nx is not True):
            raise ValueError("invalid_dev_session_write")
        with self._lock:
            now = float(self._clock())
            self._prune(now)
            if key in self._items or len(self._items) >= self._capacity:
                return False
            self._items[key] = (value, now + ex)
            return True

    def get(self, key: str) -> str | None:
        with self._lock:
            now = float(self._clock())
            self._prune(now)
            record = self._items.get(key)
            return record[0] if record is not None else None

    def delete(self, key: str) -> int:
        with self._lock:
            self._prune(float(self._clock()))
            return int(self._items.pop(key, None) is not None)
