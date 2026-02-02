from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from queue import Queue
from threading import RLock, Thread
from typing import Any, Callable, Dict, Optional

from webapp.parser.utils.session_state import DEFAULT_PHASE_BY_STATE, PipelinePhase, SessionState

EmitterFn = Callable[[Any], None]


class SessionManager:
    """Thread-safe container for web parser session state."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._logs: Dict[str, list] = {}
        self._prompt_queues: Dict[str, Queue] = {}
        self._threads: Dict[str, Thread] = {}
        self._active_sessions: set[str] = set()
        self._last_active: Dict[str, float] = {}
        self._manual_source: Dict[str, str] = {}
        self._manual_source_origin: Dict[str, str] = {}
        self._output_bypass: set[str] = set()
        self._recent_cache: Dict[str, Dict[str, Any]] = {}
        self._sid_to_session: Dict[str, str] = {}
        self._ip_ua_to_session: Dict[str, str] = {}
        self._principal_to_session: Dict[str, str] = {}
        self._session_emitters: Dict[str, EmitterFn] = {}
        self._thread_session_map: Dict[int, str] = {}
        self._last_contest_options: Dict[str, Dict[str, Any]] = {}
        self._profile_enabled = os.environ.get("SESSION_PROFILE", "false").lower() in {"1", "true", "yes"}
        self._profile_counts: Dict[str, int] = {}
        self._profile_durations: Dict[str, float] = {}
        self._profile_max: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Session metadata lifecycle
    # ------------------------------------------------------------------
    def ensure_session(self, session_id: str, username: Optional[str] = None) -> Dict[str, Any]:
        start = time.perf_counter() if self._profile_enabled else None
        try:
            with self._lock:
                meta = self._metadata.get(session_id)
                if not meta:
                    meta = self._build_metadata(session_id, username)
                    self._metadata[session_id] = meta
                elif username and not meta.get("username"):
                    meta["username"] = username
                ts = time.time()
                meta["last_active"] = ts
                self._last_active[session_id] = ts
                if "manual_source" not in meta:
                    meta["manual_source"] = self._manual_source.get(session_id, "input")
                if "manual_source_origin" not in meta:
                    meta["manual_source_origin"] = self._manual_source_origin.get(session_id, "default")
                else:
                    origin = self._manual_source_origin.get(session_id)
                    if origin:
                        meta["manual_source_origin"] = origin
                result = dict(meta)
            return result
        finally:
            if start is not None:
                self._record_profile("ensure_session", time.perf_counter() - start)

    def _build_metadata(self, session_id: str, username: Optional[str]) -> Dict[str, Any]:
        return {
            "session_id": session_id,
            "username": username or "anonymous",
            "created": datetime.now(timezone.utc).isoformat(),
            "last_active": time.time(),
            "state": SessionState.IDLE.value,
            "phase": PipelinePhase.PREPARE.value,
            "locked": False,
            "manual_source": self._manual_source.get(session_id, "input"),
            "manual_source_origin": self._manual_source_origin.get(session_id, "default"),
            "principal": None,
            "principal_source": None,
        }

    def has_session(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._metadata

    def get_metadata(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            meta = self._metadata.get(session_id)
            return dict(meta) if meta else None

    def update_metadata(self, session_id: str, **updates: Any) -> Optional[Dict[str, Any]]:
        with self._lock:
            meta = self._metadata.get(session_id)
            if not meta:
                return None
            meta.update(updates)
            if "last_active" in updates:
                try:
                    self._last_active[session_id] = float(updates["last_active"])
                except (TypeError, ValueError):
                    pass
            return dict(meta)

    def set_lock_state(self, session_id: str, locked: bool, status: Optional[str] = None) -> None:
        with self._lock:
            meta = self._metadata.get(session_id)
            if not meta:
                return
            meta["locked"] = locked
            if status:
                meta["state"] = status

    def set_state(
        self,
        session_id: str,
        state: SessionState,
        *,
        phase: Optional[PipelinePhase] = None,
        extras: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            meta = self._metadata.get(session_id)
            if not meta:
                return None
            meta["state"] = state.value
            meta["phase"] = (phase or self._infer_phase_from_state(state)).value
            meta["updated_at"] = datetime.now(timezone.utc).isoformat()
            if extras:
                meta.update(extras)
                manual_source = extras.get("manual_source") if isinstance(extras, dict) else None
                manual_origin = extras.get("manual_source_origin") if isinstance(extras, dict) else None
                if manual_source is not None:
                    self.set_manual_source(session_id, manual_source, origin=manual_origin)
                elif manual_origin is not None:
                    current_source = meta.get("manual_source", self._manual_source.get(session_id, "input"))
                    self.set_manual_source(session_id, current_source, origin=manual_origin)
            self._last_active[session_id] = time.time()
            return dict(meta)

    def _infer_phase_from_state(self, state: SessionState) -> PipelinePhase:
        mapped = DEFAULT_PHASE_BY_STATE.get(state.value)
        if mapped:
            try:
                return PipelinePhase(mapped)
            except ValueError:
                pass
        return PipelinePhase.PREPARE

    def mark_active(self, session_id: str) -> None:
        with self._lock:
            self._active_sessions.add(session_id)

    def mark_inactive(self, session_id: str) -> None:
        with self._lock:
            self._active_sessions.discard(session_id)

    def touch_session(self, session_id: str) -> None:
        ts = time.time()
        with self._lock:
            meta = self._metadata.get(session_id)
            if meta:
                meta["last_active"] = ts
            self._last_active[session_id] = ts

    def get_last_active(self, session_id: str) -> Optional[float]:
        with self._lock:
            return self._last_active.get(session_id)

    def list_active_session_ids(self) -> list[str]:
        with self._lock:
            return list(self._active_sessions)

    def list_active_metadata(self) -> list[Dict[str, Any]]:
        with self._lock:
            return [dict(self._metadata[sid]) for sid in self._active_sessions if sid in self._metadata]

    def list_all_metadata(self) -> list[Dict[str, Any]]:
        with self._lock:
            return [dict(meta) for meta in self._metadata.values()]

    def clone_session(self, old_session: str, new_session: str) -> Dict[str, Any]:
        start = time.perf_counter() if self._profile_enabled else None
        try:
            with self._lock:
                if old_session not in self._metadata:
                    raise KeyError(f"Session '{old_session}' does not exist")
                src = self._metadata[old_session]
                meta = dict(src)
                meta["session_id"] = new_session
                meta["created"] = datetime.now(timezone.utc).isoformat()
                meta["last_active"] = time.time()
                meta["state"] = SessionState.IDLE.value
                meta["phase"] = PipelinePhase.PREPARE.value
                meta["locked"] = False
                manual_source = self._manual_source.get(old_session, "input")
                manual_origin = self._manual_source_origin.get(old_session, "default")
                meta["manual_source"] = manual_source
                meta["manual_source_origin"] = manual_origin
                self._metadata[new_session] = meta
                self._logs[new_session] = list(self._logs.get(old_session, []))
                self._prompt_queues.pop(new_session, None)
                self._threads.pop(new_session, None)
                self._manual_source[new_session] = manual_source
                if manual_origin != "default":
                    self._manual_source_origin[new_session] = manual_origin
                else:
                    self._manual_source_origin.pop(new_session, None)
                if old_session in self._output_bypass:
                    self._output_bypass.add(new_session)
                self._last_active[new_session] = meta["last_active"]
                self._active_sessions.add(new_session)
                self._recent_cache.pop(new_session, None)
                result = dict(meta)
            return result
        finally:
            if start is not None:
                self._record_profile("clone_session", time.perf_counter() - start)

    # ------------------------------------------------------------------
    # Logs
    # ------------------------------------------------------------------
    def append_log(self, session_id: str, log_obj: Dict[str, Any], *, max_count: int, trim_to: int) -> list:
        with self._lock:
            logs = self._logs.setdefault(session_id, [])
            logs.append(log_obj)
            if len(logs) > max_count:
                del logs[0: len(logs) - trim_to]
            return list(logs)

    def get_logs(self, session_id: str) -> list:
        with self._lock:
            return list(self._logs.get(session_id, []))

    def set_logs(self, session_id: str, logs: list) -> None:
        with self._lock:
            self._logs[session_id] = list(logs)

    # ------------------------------------------------------------------
    # Prompt queues & threads
    # ------------------------------------------------------------------
    def get_prompt_queue(self, session_id: str) -> Queue:
        with self._lock:
            queue = self._prompt_queues.get(session_id)
            if queue is None:
                queue = Queue()
                self._prompt_queues[session_id] = queue
            return queue

    def drop_prompt_queue(self, session_id: str) -> None:
        with self._lock:
            self._prompt_queues.pop(session_id, None)

    def set_thread(self, session_id: str, thread: Thread) -> None:
        with self._lock:
            self._threads[session_id] = thread

    def get_thread(self, session_id: str) -> Optional[Thread]:
        with self._lock:
            return self._threads.get(session_id)

    def pop_thread(self, session_id: str) -> Optional[Thread]:
        with self._lock:
            return self._threads.pop(session_id, None)

    # ------------------------------------------------------------------
    # Mapping helpers
    # ------------------------------------------------------------------
    def bind_socket(self, socket_sid: str, session_id: str) -> None:
        with self._lock:
            self._sid_to_session[socket_sid] = session_id

    def resolve_socket(self, socket_sid: str) -> Optional[str]:
        with self._lock:
            return self._sid_to_session.get(socket_sid)

    def unbind_socket(self, socket_sid: str) -> Optional[str]:
        with self._lock:
            return self._sid_to_session.pop(socket_sid, None)

    def bind_fingerprint(self, fingerprint: str, session_id: str) -> None:
        with self._lock:
            self._ip_ua_to_session[fingerprint] = session_id

    def resolve_fingerprint(self, fingerprint: str) -> Optional[str]:
        with self._lock:
            return self._ip_ua_to_session.get(fingerprint)

    def unbind_fingerprints_for_session(self, session_id: str) -> None:
        """Remove any cached fingerprint bindings that still point at session_id."""
        with self._lock:
            stale = [fp for fp, sid in self._ip_ua_to_session.items() if sid == session_id]
            for fp in stale:
                self._ip_ua_to_session.pop(fp, None)

    # Principal binding (client cert or SSO)
    def set_principal(self, session_id: str, principal: Optional[str], source: Optional[str] = None) -> bool:
        """Bind a principal to a session; returns False if the principal is bound elsewhere."""
        if not principal:
            return False
        with self._lock:
            existing = self._principal_to_session.get(principal)
            if existing and existing != session_id:
                return False
            self._principal_to_session[principal] = session_id
            meta = self._metadata.get(session_id)
            if meta:
                meta["principal"] = principal
                meta["principal_source"] = source
            return True

    def resolve_principal(self, principal: str) -> Optional[str]:
        with self._lock:
            return self._principal_to_session.get(principal)

    def unbind_principal_for_session(self, session_id: str) -> None:
        with self._lock:
            stale = [p for p, sid in self._principal_to_session.items() if sid == session_id]
            for p in stale:
                self._principal_to_session.pop(p, None)

    def bind_thread_id(self, thread_id: int, session_id: str) -> None:
        with self._lock:
            self._thread_session_map[thread_id] = session_id

    def resolve_thread_id(self, thread_id: int) -> Optional[str]:
        with self._lock:
            return self._thread_session_map.get(thread_id)

    def unbind_thread_id(self, thread_id: int) -> Optional[str]:
        with self._lock:
            return self._thread_session_map.pop(thread_id, None)

    def register_emitter(self, session_id: str, emitter: EmitterFn) -> None:
        with self._lock:
            self._session_emitters[session_id] = emitter

    def pop_emitter(self, session_id: str) -> Optional[EmitterFn]:
        with self._lock:
            return self._session_emitters.pop(session_id, None)

    # ------------------------------------------------------------------
    # Session flags
    # ------------------------------------------------------------------
    def set_manual_source(self, session_id: str, source: str, *, origin: Optional[str] = None) -> None:
        with self._lock:
            normalized_origin = origin or ("default" if source == "input" else "user")
            self._manual_source[session_id] = source
            if normalized_origin == "default" and source == "input":
                self._manual_source_origin.pop(session_id, None)
            else:
                self._manual_source_origin[session_id] = normalized_origin
            meta = self._metadata.get(session_id)
            if meta:
                meta["manual_source"] = source
                meta["manual_source_origin"] = normalized_origin

    def get_manual_source(self, session_id: str, default: str = "input") -> str:
        with self._lock:
            return self._manual_source.get(session_id, default)

    def get_manual_source_origin(self, session_id: str, default: str = "default") -> str:
        with self._lock:
            return self._manual_source_origin.get(session_id, default)

    def set_output_bypass(self, session_id: str, enabled: bool) -> bool:
        with self._lock:
            if enabled:
                self._output_bypass.add(session_id)
            else:
                self._output_bypass.discard(session_id)
            return enabled

    def is_output_bypassed(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._output_bypass

    # ------------------------------------------------------------------
    # Last contest options for re-emission on reconnect
    # ------------------------------------------------------------------
    def set_last_contest_options(self, session_id: str, payload: Dict[str, Any]) -> None:
        with self._lock:
            # Ensure session exists to handle inferred sessions
            if session_id not in self._metadata:
                self.ensure_session(session_id)
            self._last_contest_options[session_id] = payload

    def get_last_contest_options(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._last_contest_options.get(session_id)

    # ------------------------------------------------------------------
    # Deduplication cache
    # ------------------------------------------------------------------
    def should_emit_message(self, session_id: str, cache_key: str, *, now: float, window: float, max_entries: int) -> bool:
        with self._lock:
            cache = self._recent_cache.setdefault(session_id, {"seen": {}, "order": []})
            last_ts = cache["seen"].get(cache_key)
            if last_ts and (now - last_ts) < window:
                return False
            cache["seen"][cache_key] = now
            cache["order"].append(cache_key)
            if len(cache["order"]) > max_entries:
                overflow = len(cache["order"]) - max_entries
                for _ in range(overflow):
                    old = cache["order"].pop(0)
                    cache["seen"].pop(old, None)
            return True

    def mark_once(self, session_id: str, token: str) -> bool:
        with self._lock:
            cache = self._recent_cache.setdefault(session_id, {"seen": {}, "order": []})
            if cache["seen"].get(token):
                return False
            cache["seen"][token] = time.time()
            return True

    # ------------------------------------------------------------------
    # Profiling helpers
    # ------------------------------------------------------------------
    def _record_profile(self, bucket: str, duration: float) -> None:
        if not self._profile_enabled:
            return
        with self._lock:
            self._profile_counts[bucket] = self._profile_counts.get(bucket, 0) + 1
            self._profile_durations[bucket] = self._profile_durations.get(bucket, 0.0) + duration
            prev_max = self._profile_max.get(bucket, 0.0)
            if duration > prev_max:
                self._profile_max[bucket] = duration

    def get_profile_snapshot(self) -> Dict[str, Dict[str, float | int]]:
        if not self._profile_enabled:
            return {}
        with self._lock:
            snapshot: Dict[str, Dict[str, float | int]] = {}
            for key, count in self._profile_counts.items():
                total = self._profile_durations.get(key, 0.0)
                snapshot[key] = {
                    "count": count,
                    "total_seconds": total,
                    "avg_seconds": total / count if count else 0.0,
                    "max_seconds": self._profile_max.get(key, 0.0),
                }
            return snapshot

    def reset_profile(self) -> None:
        if not self._profile_enabled:
            return
        with self._lock:
            self._profile_counts.clear()
            self._profile_durations.clear()
            self._profile_max.clear()

    # ------------------------------------------------------------------
    # Cleanup helpers
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear all tracked state. Intended for tests only."""
        start = time.perf_counter() if self._profile_enabled else None
        try:
            with self._lock:
                self._metadata.clear()
                self._logs.clear()
                self._prompt_queues.clear()
                self._threads.clear()
                self._manual_source.clear()
                self._manual_source_origin.clear()
                self._output_bypass.clear()
                self._active_sessions.clear()
                self._last_active.clear()
                self._recent_cache.clear()
                self._sid_to_session.clear()
                self._ip_ua_to_session.clear()
                self._principal_to_session.clear()
                self._session_emitters.clear()
                self._thread_session_map.clear()
                self._last_contest_options.clear()
                if self._profile_enabled:
                    self._profile_counts.clear()
                    self._profile_durations.clear()
                    self._profile_max.clear()
        finally:
            if start is not None:
                self._record_profile("reset", time.perf_counter() - start)

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            self._delete_session_locked(session_id)

    def expire_sessions(
        self,
        timeout: float,
        *,
        require_unlocked: bool = True,
        require_no_thread: bool = True,
        grace_period: float = 0.0,
    ) -> list[str]:
        now = time.time()
        expired: list[str] = []
        with self._lock:
            for sid, last in list(self._last_active.items()):
                age = now - last
                if age <= timeout + max(0.0, grace_period):
                    continue
                if require_unlocked:
                    meta = self._metadata.get(sid)
                    if meta and meta.get("locked"):
                        continue
                if require_no_thread:
                    thread = self._threads.get(sid)
                    if thread is not None and thread.is_alive():
                        continue
                expired.append(sid)
                self._delete_session_locked(sid)
        return expired

    def _delete_session_locked(self, session_id: str) -> None:
        self._metadata.pop(session_id, None)
        self._logs.pop(session_id, None)
        self._prompt_queues.pop(session_id, None)
        self._threads.pop(session_id, None)
        self._manual_source.pop(session_id, None)
        self._manual_source_origin.pop(session_id, None)
        self._output_bypass.discard(session_id)
        self._active_sessions.discard(session_id)
        self._last_active.pop(session_id, None)
        self._session_emitters.pop(session_id, None)
        self._recent_cache.pop(session_id, None)
        self._last_contest_options.pop(session_id, None)
        self._sid_to_session = {k: v for k, v in self._sid_to_session.items() if v != session_id}
        self._ip_ua_to_session = {k: v for k, v in self._ip_ua_to_session.items() if v != session_id}
        self._thread_session_map = {k: v for k, v in self._thread_session_map.items() if v != session_id}
        self._principal_to_session = {k: v for k, v in self._principal_to_session.items() if v != session_id}