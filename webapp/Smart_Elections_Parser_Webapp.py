from __future__ import annotations

import os


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y"}


def _should_skip_eventlet_patch() -> tuple[bool, str | None]:
    if _env_flag("SMART_ELECTIONS_SKIP_EVENTLET_PATCH", False):
        return True, "env_skip"
    # During pytest runs we prefer real threading primitives unless explicitly forced.
    if "PYTEST_CURRENT_TEST" in os.environ and not _env_flag("SMART_ELECTIONS_FORCE_EVENTLET_PATCH", False):
        return True, "pytest"
    return False, None


_SKIP_EVENTLET_PATCH, _EVENTLET_SKIP_REASON = _should_skip_eventlet_patch()
_EVENTLET_BOOT_NOTES: list[str] = []

try:
    import eventlet  # type: ignore[import-not-found]
except Exception as exc:  # pragma: no cover - environment specific
    eventlet = None  # type: ignore[assignment]
    _EVENTLET_BOOT_NOTES.append(f"eventlet_import_failed:{exc}")
    _SKIP_EVENTLET_PATCH = True
    if _EVENTLET_SKIP_REASON is None:
        _EVENTLET_SKIP_REASON = "eventlet_import_failed"


_EVENTLET_PATCH_THREAD_ENABLED = _env_flag("SMART_ELECTIONS_EVENTLET_PATCH_THREAD", False)
_EVENTLET_PATCH_OS_ENABLED = _env_flag("SMART_ELECTIONS_EVENTLET_PATCH_OS", False)
_EVENTLET_PATCH_SOCKET_ENABLED = _env_flag("SMART_ELECTIONS_EVENTLET_PATCH_SOCKET", True)
_EVENTLET_PATCH_SELECT_ENABLED = _env_flag("SMART_ELECTIONS_EVENTLET_PATCH_SELECT", True)
_EVENTLET_PATCH_TIME_ENABLED = _env_flag("SMART_ELECTIONS_EVENTLET_PATCH_TIME", True)
_EVENTLET_PATCH_PSYCO_ENABLED = _env_flag("SMART_ELECTIONS_EVENTLET_PATCH_PSYCOPG", False)
_EVENTLET_PATCH_AGGR_ENABLED = _env_flag("SMART_ELECTIONS_EVENTLET_PATCH_AGGRESSIVE", False)
_EVENTLET_PATCH_DNS_ENABLED = _env_flag("SMART_ELECTIONS_EVENTLET_PATCH_DNS", False)

_EVENTLET_PATCH_CONFIG = {
    "os": _EVENTLET_PATCH_OS_ENABLED,
    "select": _EVENTLET_PATCH_SELECT_ENABLED,
    "socket": _EVENTLET_PATCH_SOCKET_ENABLED,
    "thread": _EVENTLET_PATCH_THREAD_ENABLED,
    "time": _EVENTLET_PATCH_TIME_ENABLED,
    "psycopg": _EVENTLET_PATCH_PSYCO_ENABLED,
    "aggressive": _EVENTLET_PATCH_AGGR_ENABLED,
    "dns": _EVENTLET_PATCH_DNS_ENABLED,
}

_EVENTLET_PATCHED_MODULES: list[str] = []
_EVENTLET_PATCH_APPLIED = False

if eventlet and not _SKIP_EVENTLET_PATCH:
    try:
        eventlet.monkey_patch(**_EVENTLET_PATCH_CONFIG)
        _EVENTLET_PATCH_APPLIED = True
        _EVENTLET_PATCHED_MODULES = [name for name, enabled in _EVENTLET_PATCH_CONFIG.items() if enabled]
    except Exception as exc:  # pragma: no cover - runtime safeguard
        _EVENTLET_BOOT_NOTES.append(f"eventlet_patch_failed:{exc}")
        _EVENTLET_PATCH_APPLIED = False
        _SKIP_EVENTLET_PATCH = True
        if _EVENTLET_SKIP_REASON is None:
            _EVENTLET_SKIP_REASON = "eventlet_patch_failed"


_FORCE_THREADING_ASYNC = _env_flag("SMART_ELECTIONS_FORCE_THREADING", False)
if eventlet and not _SKIP_EVENTLET_PATCH and not _FORCE_THREADING_ASYNC:
    _SOCKETIO_ASYNC_MODE = "eventlet"
else:
    _SOCKETIO_ASYNC_MODE = "threading"

_EVENTLET_AVAILABLE = bool(eventlet)
EVENTLET_STATUS = {
    "available": _EVENTLET_AVAILABLE,
    "patched": _EVENTLET_PATCH_APPLIED,
    "patched_modules": list(_EVENTLET_PATCHED_MODULES),
    "patch_config": dict(_EVENTLET_PATCH_CONFIG),
    "skip": bool(_SKIP_EVENTLET_PATCH),
    "skip_reason": _EVENTLET_SKIP_REASON,
    "async_mode": _SOCKETIO_ASYNC_MODE,
    "notes": list(_EVENTLET_BOOT_NOTES),
}

import gzip
import re
import secrets
import shutil
import threading
import time
from datetime import datetime, timezone
from threading import Event, Thread
from typing import Callable, Tuple
from urllib.parse import urlparse

import orjson
import psycopg2
from flask import (
    Flask,
    Response,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    send_from_directory,
    session,
    url_for,
)
from flask_socketio import SocketIO, emit, join_room
from psycopg2 import errors as pg_errors
from werkzeug.exceptions import NotFound

from webapp.parser.config import (
    DATA_API_URL,
    INPUT_DIR,
    LOG_DIR,
    OUTPUT_DIR,
    POSTGRES_DB,
    POSTGRES_HOST,
    POSTGRES_PASSWORD_RAW,
    POSTGRES_PORT,
    POSTGRES_USER_RAW,
    RUN_HISTORY_FILE,
    SUPPORTED_EXTENSION_SET,
    UPLOADS_DIR,
    URL_LIST_FILE,
)
from webapp.parser.health.session_manager import SessionManager
from webapp.parser.utils.logger_singleton import logger, prompt

logger.info({
    "level": "INFO",
    "type": "infra",
    "message": f"SocketIO async mode: {_SOCKETIO_ASYNC_MODE}",
    "details": {
        "eventlet_available": EVENTLET_STATUS["available"],
        "eventlet_patched": EVENTLET_STATUS["patched"],
        "patched_modules": EVENTLET_STATUS["patched_modules"],
        "skip_reason": EVENTLET_STATUS["skip_reason"],
    }
})
from webapp.parser.utils.session_state import (
    DEFAULT_PHASE_BY_STATE,
    PipelinePhase,
    SessionState,
    export_session_enums,
)
from webapp.parser.utils.shared_logic import (
    safe_get,
    safe_is_set,
    safe_lower,
    safe_rsplit,
    safe_sid,
    safe_split,
    safe_strip,
)
from webapp.parser.web_pipeline import (
    cancel_processing,
    cancellation_manager,
    process_urls_for_web,
)
# Lazy DB table init flag
_tables_initialized = False

# Global storage for last contest options for re-emission on reconnect
last_contest_options = {}

try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    # python-dotenv not installed (e.g., on Azure), skip loading .env
    pass

# 2. Flask App & SocketIO Initialization
app = Flask(__name__)
socketio = SocketIO(
    app,
    async_mode=_SOCKETIO_ASYNC_MODE,
    cors_allowed_origins="*",
    ping_interval=10,   # 10s -> matches client pingInterval (10000 ms)
    ping_timeout=60,    # 60s -> matches client pingTimeout (60000 ms)
)

# Central textual MIME set (used by header utilities)
TEXTUAL_MIME_TYPES = {
    "text/html","text/css","application/javascript","text/javascript",
    "application/json","text/plain","application/xhtml+xml",
}

def ensure_utf8(resp: Response) -> Response:
    """
    Final safeguard: ensure textual/plain responses have an explicit UTF-8 charset.
    (Keeps vendor libs / linters like webhint satisfied even after intermediate mutations.)
    """
    ct = resp.headers.get("Content-Type", "")
    # If mimetype was set without charset (Flask may give 'text/plain' etc.)
    if any(ct.startswith(mt) for mt in TEXTUAL_MIME_TYPES) and "charset=" not in ct.lower():
        # Preserve original media type portion only
        media_type = ct.split(";")[0].strip()
        resp.headers["Content-Type"] = f"{media_type}; charset=utf-8"
    return resp
WEBAPP_CONSOLE_LEVELS = set(os.environ.get("WEBAPP_CONSOLE_LEVELS", "ERROR,WARNING").upper().split(","))

class EnsureWsSecurityHeaders:
    """
    WSGI middleware to guarantee Cache-Control and X-Content-Type-Options
    even if Socket.IO / Engine.IO shortcut bypasses Flask after_request.
    """
    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response: Callable):
        def _sr(status: str, headers: list[Tuple[str,str]], exc_info=None):
            has_cache = any(h[0].lower() == 'cache-control' for h in headers)
            has_nosniff = any(h[0].lower() == 'x-content-type-options' for h in headers)
            # Only add if missing
            if not has_cache:
                headers.append(('Cache-Control', 'no-store'))
            if not has_nosniff:
                headers.append(('X-Content-Type-Options', 'nosniff'))
            return start_response(status, headers, exc_info)
        return self.app(environ, _sr)

# Wrap early (immediately after app creation)
app.wsgi_app = EnsureWsSecurityHeaders(app.wsgi_app)

# 3. Session & State Management
session_manager = SessionManager()

ENABLE_FINGERPRINT_SESSION_RECOVERY = os.environ.get(
    "ENABLE_FINGERPRINT_SESSION_RECOVERY",
    "true",
).lower() in {"1", "true", "yes"}

LOG_DEDUPE_WINDOW = float(os.environ.get("LOG_DEDUPE_WINDOW_SEC", "2.0"))
MAX_CACHE_PER_SESSION = 120

SESSION_TIMEOUT = 3600
MAX_LOGS_PER_SESSION = 2000
TRIM_TO = 1500
HEARTBEAT_INTERVAL = 25  # seconds
_shutdown_event = Event()
HEARTBEAT_ENABLED = os.environ.get("HEARTBEAT_ENABLED", "true").lower() == "true"
HB_INTERVAL_OVERRIDE = os.environ.get("HEARTBEAT_INTERVAL")
if HB_INTERVAL_OVERRIDE:
    try:
        HEARTBEAT_INTERVAL = max(3, int(HB_INTERVAL_OVERRIDE))
    except ValueError:
        pass

DIRECT_URL_LIMIT = 20

# 4. Utility Functions

def is_owner(sid, username):
    meta = session_manager.get_metadata(sid) or {}
    return safe_get(meta, 'username') == username

def create_session_metadata(sid, username=None):
    return session_manager.ensure_session(sid, username)

def cleanup_sessions():
    expired = session_manager.expire_sessions(SESSION_TIMEOUT)
    for sid in expired:
        try:
            log_path = os.path.join(LOG_DIR, f"sess_{sid}.ndjson")
            if os.path.exists(log_path):
                os.remove(log_path)
        except Exception:
            pass
        last_contest_options.pop(sid, None)
        session_manager.unbind_fingerprints_for_session(sid)
    if expired:
        emit('session_expired', {'expired_sessions': expired}, broadcast=True)
        broadcast_sessions()

def transition_session(
    session_id: str,
    state: SessionState,
    *,
    locked: bool | None = None,
    phase: PipelinePhase | None = None,
    emit: bool = True,
    broadcast: bool = True,
    extras: dict | None = None,
):
    if not session_id:
        return None
    if not session_manager.has_session(session_id):
        session_manager.ensure_session(session_id)
    extras = dict(extras or {})
    if locked is not None:
        extras["locked"] = locked
    if "manual_source" not in extras:
        extras["manual_source"] = session_manager.get_manual_source(session_id, 'input')
    if "manual_source_origin" not in extras:
        extras["manual_source_origin"] = session_manager.get_manual_source_origin(session_id, 'default')
    updated = session_manager.set_state(session_id, state, phase=phase, extras=extras)
    if not updated:
        return None
    payload = {
        "session_id": session_id,
        "state": updated.get("state"),
        "phase": updated.get("phase"),
        "metadata": updated,
    }
    if emit:
        emit_kwargs = {}
        if not broadcast:
            emit_kwargs["room"] = session_id
        socketio.emit('session_state', payload, **emit_kwargs)
    if broadcast:
        broadcast_sessions()
    return payload

def cleanup_old_log_files(log_dir, active_sessions, keep_days=7):
    """
    Remove session log files not in active_sessions and older than keep_days.
    Never deletes RUN_HISTORY_FILE.
    """
    now = time.time()
    cutoff = now - keep_days * 86400
    for fname in os.listdir(log_dir):
        if not fname.startswith("sess_") or not fname.endswith(".ndjson"):
            continue
        sid = fname[5:-7]
        if sid in active_sessions:
            continue
        fpath = os.path.join(log_dir, fname)
        try:
            stat = os.stat(fpath)
            if stat.st_mtime < cutoff:
                os.remove(fpath)
        except Exception:
            pass

def client_fingerprint():
    try:
        xff = request.headers.get('X-Forwarded-For', '')
        ip = (xff.split(',')[0].strip() if xff else request.remote_addr) or '0.0.0.0'
    except Exception:
        ip = '0.0.0.0'
    ua = request.headers.get('User-Agent', '') or ''
    return f"{ip}|{ua[:64]}"

def resolve_session_id(data=None, create_if_missing=True):
    try:
        socket_sid = safe_sid()
    except Exception:
        socket_sid = getattr(request, 'sid', None)
    if not isinstance(socket_sid, str) or not socket_sid:
        return None
    sid = None
    if isinstance(data, dict):
        sid = safe_get(data, 'session_id')
    if isinstance(sid, str) and sid:
        session_manager.bind_socket(socket_sid, sid)
        return sid

    mapped = session_manager.resolve_socket(socket_sid)
    if isinstance(mapped, str) and mapped:
        return mapped

    cookie_sid = session.get('logical_session_id')
    if isinstance(cookie_sid, str) and cookie_sid:
        session_manager.bind_socket(socket_sid, cookie_sid)
        return cookie_sid

    fingerprint = client_fingerprint() if ENABLE_FINGERPRINT_SESSION_RECOVERY else None
    if ENABLE_FINGERPRINT_SESSION_RECOVERY and fingerprint:
        fp_sid = session_manager.resolve_fingerprint(fingerprint)
        if isinstance(fp_sid, str) and fp_sid:
            session_manager.bind_socket(socket_sid, fp_sid)
            session['logical_session_id'] = fp_sid
            return fp_sid

    if not create_if_missing:
        return None

    new_sid = 'sess_' + os.urandom(6).hex()
    if ENABLE_FINGERPRINT_SESSION_RECOVERY and fingerprint:
        session_manager.bind_fingerprint(fingerprint, new_sid)
    session_manager.bind_socket(socket_sid, new_sid)
    session['logical_session_id'] = new_sid
    session_manager.ensure_session(new_sid)
    return new_sid

def emit_contest_options(session_id: str, contests: list[dict], context: dict | None = None):
    """
    Emit full contest option list (no truncation) to the session room.
    contests: [{ "index": int, "label": str, "meta": Optional[str] }, ...]
    context:  { "state": str, "county": str, "source": str, "handler": str, "url": str, "input_file": str }
    """
    try:
        payload = {
            "session_id": session_id,
            "context": {
                "state": safe_get(context, "state"),
                "county": safe_get(context, "county"),
                "source": safe_get(context, "source"),
                "handler": safe_get(context, "handler"),
                "url": safe_get(context, "url"),
                "input_file": safe_get(context, "input_file"),
            } if isinstance(context, dict) else {},
            "total_count": len(contests or []),
            "options": contests or []
        }
        socketio.emit("contest_options", payload, room=session_id)
        # Store for re-emission on reconnect
        session_manager.set_last_contest_options(session_id, payload)
        last_contest_options[session_id] = payload
        logger.info({
            "level": "INFO",
            "type": "prompt",
            "message": f"Emitted {len(contests or [])} contest options",
            "session_id": session_id
        })
    except Exception as e:
        logger.error({
            "level": "ERROR",
            "type": "prompt",
            "message": f"Failed to emit contest options: {e}",
            "session_id": session_id
        })

def _promote_inner(obj: dict) -> dict:
    inner = obj.get("message")
    if isinstance(inner, str) and inner.strip().startswith("{"):
        try:
            parsed = orjson.loads(inner)
            if isinstance(parsed, dict) and any(k in parsed for k in ("level","type","message","status","timestamp")):
                if not parsed.get("session_id") and obj.get("session_id"):
                    parsed["session_id"] = obj.get("session_id")
                return parsed
        except Exception:
            pass
    if isinstance(inner, dict) and any(k in inner for k in ("level","type","message","status","timestamp")):
        promoted = inner.copy()
        if not promoted.get("session_id") and obj.get("session_id"):
            promoted["session_id"] = obj.get("session_id")
        if not promoted.get("level") and obj.get("level"):
            promoted["level"] = obj.get("level")
        if not promoted.get("type") and obj.get("type"):
            promoted["type"] = obj.get("type")
        return promoted
    return obj

def ensure_db_tables(force: bool = False):
    """
    Ensure SQLAlchemy models are created. Safe to call multiple times.
    Controlled by AUTO_INIT_DB (default true).
    """
    global _tables_initialized
    if _tables_initialized and not force:
        return
    if os.environ.get("AUTO_INIT_DB", "true").lower() not in ("1","true","yes"):
        return
    try:
        from webapp.parser.utils.db_utils import engine
        from webapp.parser.utils.models import Base  # imports metadata
        Base.metadata.create_all(engine)
        _tables_initialized = True
        logger.info({
            "level": "INFO",
            "type": "db",
            "message": "Database tables ensured (create_all executed)",
            "session_id": None
        })
    except Exception as e:
        logger.error({
            "level": "ERROR",
            "type": "db",
            "message": f"Failed to ensure DB tables: {e}",
            "session_id": None
        })

def normalize_log_obj(raw) -> dict:
    """
    Normalize a log object to ensure canonical level/type, timestamp, and message structure.
    Levels: INFO, DEBUG, WARNING, ERROR, CRITICAL, TRACE
    Types: status, input, output, manual_override, ai_analysis, stream, router, handler, batch,
           download, browser, validation, exception, cancel, summary, cache, prompt, heartbeat,
           database, delete, other
    """
    # Canonical type mapping
    TYPE_CANON = {
        "status": "status",
        "input": "input",
        "output": "output",
        "manual": "manual_override",
        "manualoverride": "manual_override",
        "manual_override": "manual_override",
        "ai": "ai_analysis",
        "analysis": "ai_analysis",
        "ai_analysis": "ai_analysis",
        "anomalies": "ai_analysis",
        "streamresults": "stream",
        "stream": "stream",
        "router": "router",
        "handler": "handler",
        "batch": "batch",
        "download": "download",
        "dl": "download",
        "browser": "browser",
        "validation": "validation",
        "exception": "exception",
        "cancel": "cancel",
        "cancellation": "cancel",
        "summary": "summary",
        "cache": "cache",
        "prompt": "prompt",
        "heartbeat": "heartbeat",
        "database": "database",
        "db": "database",
        "delete": "delete",
        "other": "other",
        "fatal": "exception"
    }
    LEVELS = {"INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL", "TRACE"}

    obj = raw
    # Parse string input
    if isinstance(obj, str):
        s = obj.strip()
        if s.startswith("{"):
            try:
                parsed = orjson.loads(s)
                if isinstance(parsed, dict):
                    obj = parsed
            except Exception:
                obj = {"level": "INFO", "type": "raw", "message": obj}
        else:
            obj = {"level": "INFO", "type": "raw", "message": obj}
    if not isinstance(obj, dict):
        obj = {"level": "INFO", "type": "raw", "message": str(obj)}

    # Promote nested JSON if wrapped repeatedly
    while True:
        new = _promote_inner(obj)
        if new is obj:
            break
        obj = new

    # Heuristic type inference before defaults
    msg = obj.get("message")
    if not obj.get("type"):
        if isinstance(msg, str):
            mlow = msg.lower()
            if "heartbeat" in mlow:
                obj["type"] = "heartbeat"
            elif "cancellation" in mlow:
                obj["type"] = "cancel"
            elif "error" in mlow:
                obj["type"] = "exception"
            elif "warning" in mlow:
                obj["type"] = "status"
            elif "session started" in mlow or "launching parser" in mlow:
                obj["type"] = "status"

    # Nested stringified JSON inside message
    if isinstance(msg, str) and '"level"' in msg and msg.strip().startswith("{"):
        try:
            inner = orjson.loads(msg)
            if isinstance(inner, dict):
                obj.setdefault("level", inner.get("level"))
                obj.setdefault("type", inner.get("type"))
                if "message" in inner:
                    obj["message"] = inner.get("message")
        except Exception:
            pass

    # Heartbeat friendly message fill
    if (obj.get("type") == "heartbeat" or obj.get("status") == "alive") and not obj.get("message"):
        obj["message"] = f"[heartbeat] {obj.get('session_id','')}".strip()

    # Normalize level
    level = str(obj.get("level", "INFO")).upper()
    if level not in LEVELS:
        level = "INFO"
    obj["level"] = level

    # Normalize type
    raw_type = str(obj.get("type", "other")).lower().replace("-", "_")
    obj["type"] = TYPE_CANON.get(raw_type, "other")

    # Timestamp normalization (force int ms)
    ts = obj.get("timestamp")
    if isinstance(ts, (int, float)):
        # Detect seconds vs ms
        if ts < 10_000_000_000:  # very likely seconds
            obj["timestamp"] = int(ts * 1000)
        else:
            obj["timestamp"] = int(ts)
    else:
        obj["timestamp"] = int(time.time() * 1000)

    # Large message compression
    m = obj.get("message")
    if isinstance(m, (str, bytes)) and len(m) > 10_000:
        try:
            mb = m.encode("utf-8", "replace") if isinstance(m, str) else m
            obj["message_gzip_base64"] = gzip.compress(mb).hex()
            obj["message_truncated"] = True
            obj["message"] = (m[:2000] if isinstance(m, str) else mb[:2000].decode("utf-8", "replace")) + "...(truncated)"
        except Exception:
            pass
    return obj

def store_log(session_id: str, log_obj: dict):
    if not session_id:
        return
    session_manager.append_log(session_id, log_obj, max_count=MAX_LOGS_PER_SESSION, trim_to=TRIM_TO)
    # Persist to disk using orjson
    try:
        log_path = os.path.join(LOG_DIR, f"sess_{session_id}.ndjson")
        with open(log_path, "ab") as f:
            f.write(orjson.dumps(log_obj) + b"\n")
    except Exception:
        pass

def _heartbeat_loop():
    if not HEARTBEAT_ENABLED:
        return
    while not _shutdown_event.is_set():
        time.sleep(HEARTBEAT_INTERVAL)
        now_ms = int(time.time() * 1000)
        for sid in session_manager.list_active_session_ids():
            if not session_manager.has_session(sid):
                continue
            # Emit ONLY lightweight heartbeat (no parser_output log line)
            try:
                socketio.emit('session_heartbeat', {"session_id": sid, "timestamp": now_ms}, room=sid)
            except Exception:
                pass
            
def socketio_emit_func(line):
    """
    Normalize, deduplicate, store, and emit log lines to Socket.IO.
    Used as the SocketIO emit function for SharedLogger.
    """
    try:
        # Parse or wrap the log line as a dict
        if isinstance(line, str) and not line.strip().startswith("{"):
            obj = {"level": "INFO", "type": "raw", "message": line}
        else:
            try:
                obj = orjson.loads(line) if isinstance(line, str) else line
            except Exception:
                obj = {"level": "INFO", "type": "raw", "message": str(line)}

        # Normalize log object (adds ms timestamp, infers type, etc.)
        obj = normalize_log_obj(obj)
        sid = obj.get("session_id")

        # --- Deduplication (server-side, for noisy categories) ---
        msg = str(obj.get("message", ""))
        t_now = time.time()
        if sid and obj.get("type") in {"input", "status", "raw"} and len(msg) < 600:
            key = f"{obj.get('type')}|{msg}"
            should_emit = session_manager.should_emit_message(
                sid,
                key,
                now=t_now,
                window=LOG_DEDUPE_WINDOW,
                max_entries=MAX_CACHE_PER_SESSION,
            )
            if not should_emit:
                return  # skip duplicate

        # --- Suppress repeated global URL list enumeration inside per-URL runs ---
        if sid and obj.get("type") == "input" and "Loaded" in msg and "raw URLs" in msg:
            if not session_manager.mark_once(sid, "__loaded_urls_once__"):
                return

        if sid and obj.get("type") == "prompt":
            message_lower = str(obj.get("message", "")).lower()
            if not any(term in message_lower for term in ("received", "no contest", "failed", "completed")):
                transition_session(
                    sid,
                    SessionState.WAITING_PROMPT,
                    locked=True,
                    phase=PipelinePhase.RESOLVE,
                    broadcast=False,
                    extras={
                        "manual_source": get_manual_source(sid),
                        "manual_source_origin": get_manual_source_origin(sid),
                    },
                )

        # --- Session ID fallback logic ---
        if not sid:
            # Try thread map
            mapped = session_manager.resolve_thread_id(threading.get_ident())
            if mapped:
                sid = mapped
                obj["session_id"] = sid
        if not sid:
            # Try current socket -> logical session
            try:
                curr_sid = safe_sid()
            except Exception:
                curr_sid = getattr(request, 'sid', None)
            if isinstance(curr_sid, str):
                logical = session_manager.resolve_socket(curr_sid)
                if logical:
                    sid = logical
                    obj["session_id"] = sid
        if not sid:
            # Regex extract from message text
            orig_message = obj.get("message")
            if isinstance(orig_message, str):
                m = re.search(r'\bsession_id=([a-zA-Z0-9_\-]{6,40})\b', orig_message)
                if m:
                    sid = m.group(1)
                    obj["session_id"] = sid

        # --- Store and emit ---
        if sid:
            store_log(sid, obj)
            socketio.emit('parser_output', obj, room=sid)
        else:
            socketio.emit('parser_output', obj)
    except Exception:
        # Optionally, log this error somewhere else if needed
        pass

def get_prompt_queue(session_id):
    return session_manager.get_prompt_queue(session_id)

def broadcast_sessions():
    """
    Safe global session list broadcast.
    Uses socketio.emit so it can be called from worker / background threads
    without a Flask request context.
    """
    try:
        sessions = session_manager.list_active_metadata()
        socketio.emit('session_list', {'sessions': sessions})
    except Exception as e:
        logger.error({
            "level": "ERROR",
            "type": "broadcast",
            "message": f"Failed to broadcast sessions: {e}",
            "session_id": None
        })

def lock_session(sid):
    source = get_manual_source(sid)
    origin = get_manual_source_origin(sid)
    transition_session(
        sid,
        SessionState.RUNNING,
        locked=True,
        phase=PipelinePhase.RUN,
        extras={"manual_source": source, "manual_source_origin": origin},
    )

def unlock_session(sid):
    source = get_manual_source(sid)
    origin = get_manual_source_origin(sid)
    transition_session(
        sid,
        SessionState.IDLE,
        locked=False,
        phase=PipelinePhase.PREPARE,
        extras={"manual_source": source, "manual_source_origin": origin},
    )

def safe_is_alive(session_id: str) -> bool:
    if not session_id:
        return False
    meta = session_manager.get_metadata(session_id)
    if not meta:
        return False
    last_active = session_manager.get_last_active(session_id)
    if last_active and (time.time() - last_active) > SESSION_TIMEOUT:
        return False
    thread: Thread = session_manager.get_thread(session_id)
    if not thread or not thread.is_alive():
        return False
    try:
        flag = cancellation_manager.get_flag(session_id)
        if safe_is_set(flag):
            return False
    except Exception:
        pass
    return True

def is_output_bypassed(session_id: str) -> bool:
    return session_manager.is_output_bypassed(session_id)

def get_manual_source(session_id: str) -> str:
    return session_manager.get_manual_source(session_id, 'input')

def get_manual_source_origin(session_id: str) -> str:
    return session_manager.get_manual_source_origin(session_id, 'default')

def get_all_file_lists() -> dict:
    return {
        "input_files": os.listdir(INPUT_DIR),
        "output_files": os.listdir(OUTPUT_DIR),
        "uploaded_files": os.listdir(UPLOADS_DIR),
    }

@app.get("/api/session/enums")
def get_session_enums() -> Response:
    """Expose session state/phase enumerations for the front-end."""
    return jsonify(export_session_enums())

# Flask Application Security / Config
app.secret_key = os.environ.get("FLASK_SECRET_KEY")
if not app.secret_key:
    raise RuntimeError("FLASK_SECRET_KEY not set in environment variables!")

app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SECURE"] = os.environ.get("FLASK_COOKIE_SECURE", "False").lower() == "true"
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 31536000

@app.before_request
def _csp_nonce():  # noqa: F401  (used via Flask decorator)
    # Generate nonce only if we are likely to serve HTML (skip static + pure API JSON)
    endpoint = (request.endpoint or "").lower()
    wants_html = "text/html" in (request.headers.get("Accept","") or "").lower()
    if endpoint.startswith("static") or (request.path.startswith("/api/") and not wants_html):
        g.csp_nonce = ""  # no nonce needed
    else:
        g.csp_nonce = secrets.token_urlsafe(16)

def build_csp(relaxed: bool, nonce: str) -> str:
    """
    Build Content-Security-Policy.
    Env toggles:
      ALLOW_STYLE_ATTR=true  -> allow Bootstrap (style attributes) via style-src-attr 'unsafe-inline'
      CSP_EXTRA_CONNECT      -> space‑separated extra connect-src origins
      CSP_EXTRA_SCRIPT       -> space‑separated extra script-src origins
    """
    allow_jsdelivr = True
    allow_socketio_cdn = True
    allow_style_attr = os.environ.get("ALLOW_STYLE_ATTR", "0").lower() in ("1","true","yes")
    extra_connect = [s for s in safe_split(os.environ.get("CSP_EXTRA_CONNECT", ""), " ") if s]
    extra_script  = [s for s in safe_split(os.environ.get("CSP_EXTRA_SCRIPT", ""), " ") if s]

    scripts_extra = []
    styles_elem_extra = []
    connect_extra = []

    if relaxed:
        if allow_jsdelivr:
            scripts_extra.append("https://cdn.jsdelivr.net")
            styles_elem_extra.append("https://cdn.jsdelivr.net")
        if allow_socketio_cdn:
            scripts_extra.append("https://cdn.socket.io")
            connect_extra.append("https://cdn.socket.io")

    scripts_extra.extend(extra_script)
    connect_extra.extend(extra_connect)

    # Dedupe while preserving order
    def dedupe(seq):
        seen = set()
        out = []
        for item in seq:
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
        return out

    scripts_extra = dedupe(scripts_extra)
    styles_elem_extra = dedupe(styles_elem_extra)
    connect_extra = dedupe(connect_extra)

    script_src = ["'self'"]
    if nonce:
        script_src.append(f"'nonce-{nonce}'")
    script_src.extend(scripts_extra)

    style_src = ["'self'"]
    style_src_elem = ["'self'", *styles_elem_extra]

    style_src_attr = ["'unsafe-inline'"] if allow_style_attr else ["'none'"]

    connect_src = ["'self'", "ws:", "wss:", *connect_extra]

    directives = [
        ("default-src", "'self'"),
        ("base-uri", "'self'"),
        ("frame-ancestors", "'none'"),
        ("form-action", "'self'"),
        ("object-src", "'none'"),
        ("script-src", " ".join(script_src)),
        ("style-src", " ".join(style_src)),
        ("style-src-elem", " ".join(dedupe(style_src_elem))),
        ("style-src-attr", " ".join(style_src_attr)),
        ("img-src", "'self' data:"),
        ("font-src", "'self' data:"),
        ("connect-src", " ".join(connect_src)),
    ]
    return "; ".join(f"{k} {v}" for k, v in directives)

@app.after_request
def add_headers(response: Response) -> Response:
    """
    Harden outbound responses and ensure UTF-8 charset (also applies to socket.io handshake).
    Order of operations:
      1. Security / caching headers
      2. CSP (only for HTML / XHTML)
      3. Special-case Socket.IO polling (forces text/plain; charset added later)
      4. Final UTF-8 charset normalization (ensure_utf8)
    """
    # websocket handshake detection (pre-upgrade GET)
    upgrade_hdr = (request.headers.get("Upgrade") or "").lower()
    is_ws = 'websocket' in upgrade_hdr or bool(request.environ.get("wsgi.websocket"))

    # Base security headers (also for websocket handshake GET)
    base_headers = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "no-referrer",
        "Permissions-Policy": "geolocation=()",
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Resource-Policy": "same-origin",
    }
    for k, v in base_headers.items():
        response.headers.setdefault(k, v)

    # In add_headers(), strengthen /socket.io/ handling:
    if request.path.startswith('/socket.io/'):
        # Force required headers for handshake + any polling fallback
        response.headers['Cache-Control'] = 'no-store'
        response.headers.setdefault('X-Content-Type-Options', 'nosniff')

    if request.path.startswith('/socket.io/') and 'websocket' in (request.headers.get("Upgrade","").lower()):
        response.headers['Content-Security-Policy'] = "default-src 'none'; connect-src 'self' ws: wss:;"

    if not is_ws:
        # Caching & CSP only for non-upgraded HTTP
        if request.endpoint == "static":
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
            response.headers.pop("Content-Security-Policy", None)
            response.headers.pop("X-XSS-Protection", None)
        else:
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
            response.headers.setdefault("Pragma", "no-cache")
            response.headers.setdefault("Expires", "0")
        relaxed = os.environ.get("CSP_MODE", "RELAXED").upper() == "RELAXED"
        nonce = getattr(g, "csp_nonce", "")
        if response.mimetype and response.mimetype.startswith(("text/html", "application/xhtml")):
            response.headers["Content-Security-Policy"] = build_csp(relaxed, nonce)
        else:
            response.headers.pop("Content-Security-Policy", None)
        pass
    else:
        # For websocket handshake only: add Cache-Control so webhint stops warning
        response.headers.setdefault("Cache-Control", "no-store")

    if request.path.startswith("/socket.io/") and "transport=polling" in request.query_string.decode(errors="ignore"):
        response.headers["Content-Type"] = "text/plain"
        response.headers.setdefault("Cache-Control", "no-store")

    # Vary header normalization
    vary_tokens = {t.strip() for t in (response.headers.get("Vary", "")).split(",") if t.strip()}
    if "Cookie" not in vary_tokens:
        vary_tokens.add("Cookie")
        response.headers["Vary"] = ", ".join(sorted(vary_tokens))

    if request.is_secure:
        response.headers.setdefault(
            "Strict-Transport-Security",
            "max-age=63072000; includeSubDomains; preload"
        )

    # Final charset / textual normalization (single point of truth)
    response = ensure_utf8(response)
    return response

# Data Management Utilities
def add_url() -> None:
    url = input("Enter new URL to add: ").strip()
    if url:
        with open(URL_LIST_FILE, "a", encoding="utf-8") as f:
            f.write(url + "\n")
        logger.info({
            "level": "INFO",
            "type": "status",
            "message": f"[ADDED] {url}",
            "session_id": None
        })

def allowed_file(filename) -> bool:
    if not filename or len(filename) >= 128:
        return False
    parts = safe_rsplit(filename, '.', 1)
    if len(parts) < 2:
        return False
    ext = "." + safe_lower(parts[1])
    return ext in SUPPORTED_EXTENSION_SET

def get_url_list() -> list[str]:
    if not os.path.exists(URL_LIST_FILE):
        return []
    with open(URL_LIST_FILE, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    return urls

def list_urls() -> list[str]:
    if not os.path.exists(URL_LIST_FILE):
        logger.info({
            "level": "INFO",
            "type": "status",
            "message": "No urls.txt found.",
            "session_id": None
        })
        return []
    with open(URL_LIST_FILE, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    logger.info({
        "level": "INFO",
        "type": "status",
        "message": "[URLS.TXT ENTRIES]",
        "session_id": None
    })
    for i, url in enumerate(urls, 1):
        logger.info({
            "level": "INFO",
            "type": "status",
            "message": f"{i}. {url}",
            "session_id": None
        })
    return urls

def log_run_event(event: dict):
    """
    Append a run event (NDJSON). Safe best-effort.
    event: {
      "type": "start" | "end",
      "run_id": str,
      "session_id": str,
      "ts": iso timestamp,
      "source": "input"|"uploads",
      "output_bypass": bool,
      "status": "ok"|"error"|"cancelled",
      "error": optional str,
      "duration_ms": optional int
    }
    """
    try:
        line = orjson.dumps(event) + b"\n"
        with open(RUN_HISTORY_FILE, "ab") as f:
            f.write(line)
    except Exception:
        pass

# Routes
@app.route("/")
def index() -> str:
    return render_template("index.html")

@app.route("/api/urls", methods=["GET", "POST"])
def api_urls():
    urls_file = str(URL_LIST_FILE)
    if request.method == "GET":
        if not os.path.exists(urls_file):
            return jsonify({"urls": []})
        with open(urls_file, "r", encoding="utf-8") as f:
            urls = [
                safe_strip(line) for line in f
                if safe_strip(line) and not safe_strip(line).startswith("#")
            ]
        return jsonify({"urls": urls})
    elif request.method == "POST":
        data = request.get_json() or {}
        url = safe_strip(safe_get(data, "url", ""))
        if not url:
            return jsonify({"success": False, "error": "URL required."}), 400
        with open(urls_file, "a", encoding="utf-8") as f:
            f.write(url + "\n")
        return jsonify({"success": True})

@app.route("/data_framework", methods=["GET"])
def data_framework():
    return render_template("data_framework.html", data_api_url=DATA_API_URL)

@app.route("/api/fs/list", methods=["GET"])
def api_fs_list():
    root = (request.args.get("root") or "").lower().strip()
    subpath = (request.args.get("path") or "").strip().replace("\\", "/")
    roots = {"input": INPUT_DIR, "output": OUTPUT_DIR, "uploads": UPLOADS_DIR}
    base = roots.get(root)
    if not base:
        return jsonify({"root": root, "path": subpath, "entries": []}), 400

    abs_base = os.path.abspath(base)
    want = os.path.normpath(os.path.join(abs_base, subpath))
    if not want.startswith(abs_base):
        return jsonify({"root": root, "path": subpath, "entries": []}), 400
    if not os.path.isdir(want):
        return jsonify({"root": root, "path": subpath, "entries": []})

    entries = []
    try:
        with os.scandir(want) as it:
            for de in it:
                try:
                    st = de.stat(follow_symlinks=False)
                    entries.append({
                        "name": de.name,
                        "type": "dir" if de.is_dir(follow_symlinks=False) else "file",
                        "size": None if de.is_dir(follow_symlinks=False) else int(st.st_size),
                        "modified": int(st.st_mtime * 1000)
                    })
                except Exception:
                    entries.append({
                        "name": de.name,
                        "type": "dir" if de.is_dir(follow_symlinks=False) else "file",
                        "size": None,
                        "modified": None
                    })
        entries.sort(key=lambda e: (e["type"] != "dir", e["name"].lower()))
    except Exception as e:
        logger.error({
            "level": "ERROR",
            "type": "browser",
            "message": f"Failed to list dir {root}:{subpath} -> {e}",
            "session_id": None
        })
        entries = []
    return jsonify({"root": root, "path": subpath, "entries": entries})

@app.route("/api/list_dir", methods=["GET"])
def api_list_dir_compat():
    return api_fs_list()

@app.route("/api/fs/mkdir", methods=["POST"])
def api_fs_mkdir():
    import os
    data = request.get_json(force=True) or {}
    root = (data.get("root") or "").lower().strip()
    subpath = (data.get("path") or "").strip().replace("\\", "/")
    name = (data.get("name") or "").strip()
    if not name or "/" in name or "\\" in name:
        return jsonify({"success": False, "error": "Invalid folder name."}), 400
    roots = {"input": INPUT_DIR, "output": OUTPUT_DIR, "uploads": UPLOADS_DIR}
    base = roots.get(root)
    if not base:
        return jsonify({"success": False, "error": "Invalid root."}), 400
    abs_base = os.path.abspath(base)
    parent = os.path.normpath(os.path.join(abs_base, subpath))
    if not parent.startswith(abs_base):
        return jsonify({"success": False, "error": "Path escape blocked."}), 400
    try:
        target = os.path.normpath(os.path.join(parent, name))
        if not target.startswith(abs_base):
            return jsonify({"success": False, "error": "Path escape blocked."}), 400
        os.makedirs(target, exist_ok=False)
        return jsonify({"success": True})
    except FileExistsError:
        return jsonify({"success": False, "error": "Folder already exists."}), 409
    except Exception as e:
        logger.error({"level":"ERROR","type":"browser","message":f"mkdir failed: {e}","session_id":None})
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/fs/delete", methods=["POST"])
def api_fs_delete():
    data = request.get_json(force=True) or {}
    root = (data.get("root") or "").lower().strip()
    subpath = (data.get("path") or "").strip().replace("\\", "/")
    name = (data.get("name") or "").strip()
    recursive = bool(data.get("recursive"))
    roots = {"input": INPUT_DIR, "output": OUTPUT_DIR, "uploads": UPLOADS_DIR}
    base = roots.get(root)
    if not base or not name:
        return jsonify({"success": False, "error": "Invalid parameters."}), 400
    abs_base = os.path.abspath(base)
    parent = os.path.normpath(os.path.join(abs_base, subpath))
    target = os.path.normpath(os.path.join(parent, name))
    if not parent.startswith(abs_base) or not target.startswith(abs_base):
        return jsonify({"success": False, "error": "Path escape blocked."}), 400
    if not os.path.exists(target):
        return jsonify({"success": False, "error": "Not found."}), 404
    try:
        if os.path.isfile(target):
            os.remove(target)
        elif os.path.isdir(target):
            if recursive:
                shutil.rmtree(target)
            else:
                os.rmdir(target)  # only if empty
        else:
            return jsonify({"success": False, "error": "Unsupported type."}), 400
        return jsonify({"success": True})
    except OSError as e:
        # Common “directory not empty”
        return jsonify({"success": False, "error": str(e)}), 409
    except Exception as e:
        logger.error({"level":"ERROR","type":"browser","message":f"delete failed: {e}","session_id":None})
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/download_fs")
def download_fs():
    import os
    root = (request.args.get("root") or "").lower().strip()
    subpath = (request.args.get("path") or "").strip().replace("\\", "/")
    name = request.args.get("name") or ""
    roots = {"input": INPUT_DIR, "output": OUTPUT_DIR, "uploads": UPLOADS_DIR}
    base = roots.get(root)
    if not base or not name:
        raise NotFound()
    abs_base = os.path.abspath(base)
    want_dir = os.path.normpath(os.path.join(abs_base, subpath))
    if not want_dir.startswith(abs_base):
        raise NotFound()
    fpath = os.path.normpath(os.path.join(want_dir, name))
    if not fpath.startswith(abs_base) or not os.path.isfile(fpath):
        raise NotFound()
    return send_file(fpath, as_attachment=True)

@app.route("/favicon.ico")
def favicon():
    static_root = app.static_folder or "static"
    static_root_abs = os.path.abspath(static_root)
    filename = "favicon.ico"

    accept_header = (request.headers.get("Accept") or "")[:256]
    wants_svg = "image/svg+xml" in accept_header

    ico_path = os.path.abspath(os.path.join(static_root_abs, filename))
    if not ico_path.startswith(static_root_abs + os.sep):
        logger.warning({"type": "sec", "message": "Favicon path escape blocked", "requested": ico_path})
        raise NotFound()

    if os.path.exists(ico_path) and not wants_svg:
        try:
            st = os.stat(ico_path)
            etag = f'W/"ico-{st.st_size}-{int(st.st_mtime)}"'
            inm = request.if_none_match
            if inm and etag in inm:
                resp = Response(status=304)
                resp.headers["ETag"] = etag
                return resp
            resp = send_from_directory(
                static_root_abs,
                filename,
                mimetype="image/x-icon",
                conditional=True,
                max_age=31536000
            )
            resp.headers["ETag"] = etag
            resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
            resp.headers.pop("Content-Security-Policy", None)
            resp.headers.pop("X-XSS-Protection", None)
            return resp
        except OSError:
            pass  # fall through to SVG

    raw_accent = (os.environ.get("FAVICON_ACCENT", "2563eb") or "").strip().lstrip("#")
    if not re.fullmatch(r"[0-9a-fA-F]{3,8}", raw_accent):
        raw_accent = "2563eb"
    accent = raw_accent.lower()

    etag_svg = f'W/"svg-{accent}"'
    if request.if_none_match and etag_svg in request.if_none_match:
        resp = Response(status=304)
        resp.headers["ETag"] = etag_svg
        return resp

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" role="img" aria-label="App icon">
  <rect width="64" height="64" rx="14" fill="#{accent}"/>
  <path d="M18 44V20h6.8l7.2 10.9L39.2 20H46v24h-6V31.8l-7 10.5-7-10.5V44z" fill="#fff"/>
</svg>"""
    resp = Response(svg, mimetype="image/svg+xml")
    resp.headers["ETag"] = etag_svg
    resp.headers["Cache-Control"] = "public, max-age=86400"
    resp.headers.pop("Content-Security-Policy", None)
    resp.headers.pop("X-XSS-Protection", None)
    return resp

@app.route("/robots.txt")
def robots_txt():
    return "User-agent: *\nDisallow: /", 200, {"Content-Type": "text/plain"}

@app.route("/api/warehouse_election_results", methods=["GET"])
def api_warehouse_election_results():
    state = request.args.get("state")
    county = request.args.get("county")
    contest = request.args.get("contest")
    where = []
    params = []
    if state:
        where.append("state = %s")
        params.append(state)
    if county:
        where.append("county = %s")
        params.append(county)
    if contest:
        where.append("contest ILIKE %s")
        params.append(f"%{contest}%")
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    ensure_db_tables()  # attempt upfront (idempotent)
    try:
        conn = psycopg2.connect(
            dbname=POSTGRES_DB,
            user=POSTGRES_USER_RAW,
            password=POSTGRES_PASSWORD_RAW,
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            sslmode=(
                "require"
                if (POSTGRES_HOST not in ("localhost", "127.0.0.1")
                    and os.environ.get("PG_REQUIRE_SSL", "true").lower() == "true")
                else "prefer"
            )
        )
        with conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT *
                FROM warehouse_election_results
                {where_sql}
                ORDER BY 1 DESC
                """,
                params
            )
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        return jsonify({"items": rows, "count": len(rows)})
    except Exception as e:
        msg = str(e)
        missing = ("does not exist" in msg.lower()) or isinstance(e, getattr(pg_errors, "UndefinedTable", tuple()))
        if missing:
            logger.warning({
                "level": "WARNING",
                "type": "db",
                "message": f"Detected missing tables, attempting auto-create then retry: {e}",
                "session_id": None
            })
            ensure_db_tables(force=True)
            try:
                conn = psycopg2.connect(
                    dbname=POSTGRES_DB,
                    user=POSTGRES_USER_RAW,
                    password=POSTGRES_PASSWORD_RAW,
                    host=POSTGRES_HOST,
                    port=POSTGRES_PORT,
                    sslmode="require" if (POSTGRES_HOST not in ("localhost","127.0.0.1")
                        and os.environ.get("PG_REQUIRE_SSL","true").lower() == "true") else "prefer"
                )
                with conn, conn.cursor() as cur:
                    cur.execute(
                        f"""
                        SELECT *
                        FROM warehouse_election_results
                        {where_sql}
                        ORDER BY 1 DESC
                        """,
                        params
                    )
                    cols = [d[0] for d in cur.description]
                    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
                return jsonify({"items": rows, "count": len(rows), "auto_created": True})
            except Exception as e2:
                logger.error({
                    "level": "ERROR",
                    "type": "db",
                    "message": f"DB error after retry: {e2}",
                    "session_id": None
                })
                return jsonify({"error": f"Data API error after init attempt: {e2}"}), 500
        logger.error({
            "level": "ERROR",
            "type": "db",
            "message": f"DB error: {e}",
            "session_id": None
        })
        return jsonify({"error": f"Data API error: {e}"}), 500

@app.route("/delete/input/<filename>", methods=["POST"])
def delete_input_file(filename) -> str:
    file_path = os.path.join(INPUT_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        flash(f"Deleted '{filename}' from input folder.", "success")
    else:
        flash(f"File '{filename}' not found in input folder.", "danger")
    return redirect(request.referrer or url_for("run_parser"))

@app.route("/delete/output/<filename>", methods=["POST"])
def delete_output_file(filename) -> str:
    file_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        flash(f"Deleted '{filename}' from output folder.", "success")
    else:
        flash(f"File '{filename}' not found in output folder.", "danger")
    return redirect(request.referrer or url_for("run_parser"))

@app.route("/delete/uploads/<filename>", methods=["POST"])
def delete_upload_file(filename) -> str:
    file_path = os.path.join(UPLOADS_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        flash(f"Deleted '{filename}' from uploads folder.", "success")
    else:
        flash(f"File '{filename}' not found in uploads folder.", "danger")
    return redirect(request.referrer or url_for("run_parser"))

@app.route("/download/input/<filename>")
def download_input_file(filename) -> str:
    return send_from_directory(INPUT_DIR, filename, as_attachment=True)

@app.route("/download/output/<filename>")
def download_output_file(filename) -> str:
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)

@app.route("/download/uploads/<filename>")
def download_upload_file(filename) -> str:
    return send_from_directory(UPLOADS_DIR, filename, as_attachment=True)

@app.route("/run_parser", methods=["GET", "POST"])
def run_parser():
    try:
        qp_source = safe_lower(request.args.get("source", "")) if request.method == "GET" else ""
        if qp_source in {"input", "uploads"}:
            session['manual_source_pref'] = qp_source
        if request.method == "POST" and "data_file" in request.files:
            file = request.files.get("data_file")
            if file and allowed_file(file.filename):
                filename = file.filename
                file.save(os.path.join(UPLOADS_DIR, filename))
                flash(f"File '{filename}' uploaded successfully.", "success")
            else:
                flash("Invalid file type or no file selected.", "danger")
        file_lists = get_all_file_lists()
        return render_template(
            "run_parser.html",
            input_files=file_lists["input_files"],
            output_files=file_lists["output_files"],
            uploaded_files=file_lists["uploaded_files"],
            manual_source=session.get('manual_source_pref', 'input'),
            allow_style_attr=os.environ.get("ALLOW_STYLE_ATTR", "0").lower() in ("1","true","yes"),
            static_version=os.environ.get("STATIC_VERSION", "v1")
        )
    except Exception:
        import traceback
        print(traceback.format_exc())
        return "Internal Server Error", 500

@app.route("/site.webmanifest")
def site_webmanifest():
    manifest = {
        "id": "/?src=pwa",
        "name": "Smart Elections Parser",
        "short_name": "Parser",
        "description": "Election results parsing interface",
        "lang": "en",
        "dir": "ltr",
        "start_url": "/?utm_source=pwa",
        "scope": "/",
        "display": "standalone",
        "orientation": "any",
        "background_color": "#1a232a",
        "theme_color": "#2563eb",
        "categories": ["productivity", "utilities", "data"],
        "icons": [
            {"src": "/static/icons/icon-192.png", "sizes": "192x192", "type": "image/png"},
            {"src": "/static/icons/icon-512.png", "sizes": "512x512", "type": "image/png"},
            {"src": "/static/icons/icon-maskable-192.png", "sizes": "192x192", "type": "image/png", "purpose": "maskable any"},
            {"src": "/static/icons/icon-maskable-512.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable any"}
        ],
        "shortcuts": [
            {"name": "Run Parser", "short_name": "Run", "url": "/run_parser"},
            {"name": "History", "short_name": "History", "url": "/history"}
        ],
        "prefer_related_applications": False
    }
    payload = orjson.dumps(manifest)
    etag = f'W/"m-{len(payload)}"'
    if etag in (request.if_none_match or []):
        return Response(status=304)
    # Add explicit utf-8 charset for linting tools
    resp = Response(payload, mimetype="application/manifest+json; charset=utf-8")
    resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    resp.headers["ETag"] = etag
    return resp

@app.route("/upload/input", methods=["POST"])
def upload_to_input() -> str:
    file = request.files.get("file")
    logger.info({
        "level": "INFO",
        "type": "status",
        "message": f"Upload to input: {file.filename if file else 'No file'}",
        "session_id": None
    })
    if file and allowed_file(file.filename):
        filename = file.filename
        file.save(os.path.join(INPUT_DIR, filename))
        flash(f"File '{filename}' uploaded to input folder.", "success")
    else:
        flash("Invalid file type or no file selected.", "danger")
    return redirect(request.referrer or url_for("run_parser"))

@app.route("/upload/output", methods=["POST"])
def upload_to_output() -> str:
    file = request.files.get("file")
    logger.info({
        "level": "INFO",
        "type": "status",
        "message": f"Upload to output: {file.filename if file else 'No file'}",
        "session_id": None
    })
    if file and allowed_file(file.filename):
        filename = file.filename
        file.save(os.path.join(OUTPUT_DIR, filename))
        flash(f"File '{filename}' uploaded to output folder.", "success")
    else:
        flash("Invalid file type or no file selected.", "danger")
    return redirect(request.referrer or url_for("run_parser"))

@app.route("/upload/uploads", methods=["POST"])
def upload_to_uploads() -> str:
    file = request.files.get("data_file") or request.files.get("file")
    if file and allowed_file(file.filename):
        filename = file.filename
        file.save(os.path.join(UPLOADS_DIR, filename))
        session['FORCE_PARSE_INPUT_FILE'] = filename
        session['FORCE_PARSE_FORMAT'] = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
        session['manual_source_pref'] = 'uploads'  # default UI to uploads after upload
        flash(f"File '{filename}' uploaded to uploads folder.", "success")
    else:
        flash("Invalid file type or no file selected.", "danger")
    return redirect(request.referrer or url_for("run_parser"))

@app.route("/health")
def health() -> str:
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

@app.route("/heartbeat")
def heartbeat() -> str:
    return {"status": "ok"}

@app.route("/clear_history", methods=["POST"])
def clear_history():
    try:
        if RUN_HISTORY_FILE.exists():
            RUN_HISTORY_FILE.unlink()
        flash("Run history cleared.", "success")
    except Exception as e:
        flash(f"Failed to clear history: {e}", "danger")
    return redirect(url_for("history"))

@app.route("/history")
def history() -> str:
    """
    Show recent parser runs (NOT override snapshots).
    """
    runs = []
    if RUN_HISTORY_FILE.exists():
        try:
            with open(RUN_HISTORY_FILE, "rb") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        evt = orjson.loads(line)
                        runs.append(evt)
                    except Exception:
                        continue
        except Exception:
            pass
    # --- prefer "end" event for each run_id ---
    aggregated = {}
    for evt in runs:
        rid = evt.get("run_id")
        if not rid:
            continue
        # Always update with new event, but prefer "end" event for status
        slot = aggregated.setdefault(rid, {"run_id": rid})
        if evt.get("type") == "end":
            # End event: overwrite all relevant fields
            slot.update(evt)
            slot["completed"] = True
        else:
            # Start or other event: only update if no end event seen yet
            if not slot.get("completed"):
                slot.update(evt)

    # No need to force status to "ok" -- just use the status from the end event if present
    def _ts(v):
        return v.get("ts","")
    ordered = sorted(aggregated.values(), key=_ts, reverse=True)
    return render_template("history.html", runs=ordered)

@app.route("/rerun/<run_id>", methods=["POST"])
def rerun_prior(run_id):
    """
    Trigger a rerun using the recorded source/output_bypass flags.
    This just emits a SocketIO event style workflow (reuse run_parser logic).
    """
    # Minimal metadata lookup
    source = "input"
    output_bypass = False
    if RUN_HISTORY_FILE.exists():
        try:
            with open(RUN_HISTORY_FILE, "rb") as f:
                for line in f:
                    try:
                        evt = orjson.loads(line)
                    except Exception:
                        continue
                    if evt.get("run_id") == run_id and evt.get("type") == "start":
                        source = evt.get("source", source)
                        output_bypass = bool(evt.get("output_bypass", output_bypass))
                        break
        except Exception:
            pass
    # Use a new session (user can change later)
    new_session = 'sess_' + os.urandom(6).hex()
    session['logical_session_id'] = new_session
    flash(f"Re-running prior config (run_id={run_id}) in new session {new_session}", "success")
    # Front-end JS should now request a run (or we can directly invoke)
    return redirect(url_for("run_parser", source=source))

# 6. SocketIO Event Handlers

@socketio.on('contest_selected')
def handle_contest_selected(data) -> None:
    """
    Accepts selection from modal and passes it into the active prompt (if any).
    Expects: { session_id: str, indices: [int] }
    """
    sid = resolve_session_id(data or {}, create_if_missing=False)
    indices = []
    try:
        indices = [int(x) for x in (data.get("indices") or [])]
    except Exception:
        pass
    if not sid or not indices:
        emit('parser_output', normalize_log_obj({
            "level": "WARNING",
            "type": "prompt",
            "message": "No contest selected."
        }), room=getattr(request, 'sid', None))
        return
    # Hand off to the active prompt (same effect as user typing "index" into prompt)
    try:
        prompt_session = prompt.prompt_sessions.get(sid)
        if prompt_session:
            prompt_session.set_response(",".join(str(i) for i in indices))
            emit('parser_output', normalize_log_obj({
                "level": "INFO",
                "type": "prompt",
                "message": f"Contest selection received: {indices}",
                "session_id": sid
            }), room=request.sid)
        else:
            # Fallback: behave like parser_prompt
            handle_parser_prompt({"session_id": sid, "value": ",".join(str(i) for i in indices)})
    except Exception as e:
        emit('parser_output', normalize_log_obj({
            "level": "ERROR",
            "type": "prompt",
            "message": f"Failed to accept selection: {e}",
            "session_id": sid
        }), room=request.sid)

@socketio.on('get_session_history')
def handle_get_session_history(data) -> None:
    sid = resolve_session_id(data, create_if_missing=False)
    try:
        socket_room = safe_sid()
    except Exception:
        socket_room = getattr(request, 'sid', None)
    if not sid:
        emit('session_history', {'session_id': None, 'logs': []}, room=socket_room)
        return
    logs = session_manager.get_logs(sid)
    # If missing in memory, try to load from disk (orjson)
    if (not logs or not isinstance(logs, list)) and os.path.exists(os.path.join(LOG_DIR, f"sess_{sid}.ndjson")):
        try:
            log_path = os.path.join(LOG_DIR, f"sess_{sid}.ndjson")
            with open(log_path, "rb") as f:
                logs = [orjson.loads(line) for line in f if line.strip()]
            session_manager.set_logs(sid, logs)  # restore to memory for future
        except Exception:
            logs = []
    # --- Ensure latest prompt log is present if prompt is still active ---
    prompt_session = getattr(prompt, "prompt_sessions", {}).get(sid)
    if prompt_session and not prompt_session.is_resolved():
        # Check if the last log is already a prompt log with the same message
        last_prompt = None
        for log in reversed(logs):
            if log.get("type") == "prompt":
                last_prompt = log
                break
        prompt_msg = getattr(prompt_session, "prompt_message", None)
        # Only append if not already present or message differs
        if not last_prompt or (prompt_msg and last_prompt.get("message") != prompt_msg):
            logs = list(logs)  # copy to avoid mutating shared state
            logs.append({
                "level": "INFO",
                "type": "prompt",
                "message": prompt_msg or "",
                "session_id": sid,
                "timestamp": int(time.time() * 1000)
            })
    # Deduplicate consecutive prompt logs with the same message
    deduped_logs = []
    last_prompt_msg = None
    for log in logs:
        if log.get("type") == "prompt":
            msg = log.get("message")
            if msg == last_prompt_msg:
                continue
            last_prompt_msg = msg
        deduped_logs.append(log)
    emit('session_history', {'session_id': sid, 'logs': deduped_logs}, room=socket_room)

@socketio.on('clone_session')
def handle_clone_session(data) -> None:
    old_sid = str(data['session_id'])
    if not isinstance(old_sid, str):
        logger.warning(
            {
                "level": "WARNING",
                "type": "status",
                "message": f"Invalid session_id type: {type(old_sid)} value: {old_sid}"
            }
        )
        return
    new_sid = 'sess_' + os.urandom(6).hex()
    try:
        session_manager.clone_session(old_sid, new_sid)
    except KeyError:
        logger.warning(
            {
                "level": "WARNING",
                "type": "status",
                "message": f"Unable to clone missing session: {old_sid}"
            }
        )
        return
    transition_session(
        new_sid,
        SessionState.IDLE,
        locked=False,
        phase=PipelinePhase.PREPARE,
        broadcast=False,
        extras={
            "manual_source": get_manual_source(new_sid),
            "manual_source_origin": get_manual_source_origin(new_sid),
        },
    )
    broadcast_sessions()
    try:
        socket_room = safe_sid()
    except Exception:
        socket_room = getattr(request, 'sid', None)
        if not isinstance(socket_room, str):
            socket_room = None
    emit('session_cloned', {'old_session': old_sid, 'new_session': new_sid}, room=socket_room)

@socketio.on('join')
def on_join(data):
    sid = resolve_session_id(data)
    if not isinstance(sid, str):
        logger.warning(
            {
                "level": "WARNING",
                "type": "status",
                "message": f"Invalid session_id resolved: {sid}"
            }
        )
        return
    join_room(sid)
    username = safe_get(data, 'username')
    meta = session_manager.ensure_session(sid, username)
    session_manager.mark_active(sid)
    session_manager.touch_session(sid)
    try:
        socket_sid = safe_sid()
    except Exception:
        socket_sid = getattr(request, 'sid', None)
    if isinstance(socket_sid, str):
        session_manager.bind_socket(socket_sid, sid)

    phase_value = meta.get("phase") or DEFAULT_PHASE_BY_STATE.get(meta.get("state"), PipelinePhase.PREPARE.value)
    state_value = meta.get("state", SessionState.IDLE.value)
    meta["state"] = state_value
    meta["phase"] = phase_value
    payload = {
        "session_id": sid,
        "state": state_value,
        "phase": phase_value,
        "metadata": meta,
    }
    socketio.emit('session_state', payload, room=sid)
    broadcast_sessions()
    # Notify client that join is complete (for real-time log delivery sync)
    emit('joined', {'session_id': sid}, room=request.sid)

@socketio.on('get_sessions')
def handle_get_sessions():
    cleanup_sessions()
    sessions = session_manager.list_active_metadata()
    emit('session_list', {'sessions': sessions}, broadcast=True)


@socketio.on('connect')
def handle_connect(auth=None):
    try:
        cleanup_sessions()
        session['log_format'] = "json"
        requested = None
        if isinstance(auth, dict):
            requested = safe_get(auth, 'requested_session_id')
        if not requested:
            requested = request.args.get('prev_session_id')
        revived = None
        if requested and session_manager.has_session(requested):
            revived = requested
            session_manager.touch_session(revived)
            cancellation_manager.remove(revived)
            # Do NOT clear prompt session here; let it persist for timeout
            logger.info({
                "level": "INFO",
                "type": "status",
                "message": "Re-associated socket with existing session",
                "session_id": revived
            })
            emit('session_id', {'session_id': revived})
        else:
            cookie_sid = session.get('logical_session_id')
            if cookie_sid and session_manager.has_session(cookie_sid):
                session_manager.mark_active(cookie_sid)
                session_manager.touch_session(cookie_sid)
                revived = cookie_sid
                emit('session_id', {'session_id': revived})
                logger.info({
                    "level": "INFO",
                    "type": "status",
                    "message": "Resurrected prior session after reload",
                    "session_id": revived
                })

        resolved = resolve_session_id({'session_id': revived} if revived else {}, create_if_missing=False)
        if resolved:
            session_manager.touch_session(resolved)

        if revived:
            session_manager.mark_active(revived)
            if revived in last_contest_options:
                socketio.emit("contest_options", last_contest_options[revived], room=revived)
        active = session_manager.list_active_metadata()
        emit('session_list', {'sessions': active})
        logger.info({
            "level": "INFO",
            "type": "status",
            "message": "Socket connected (no auto session creation)",
            "session_id": resolved
        })
    except Exception as e:
        emit('parser_output', {
            "level": "ERROR",
            "type": "error",
            "message": f"Connect error: {e}",
            "session_id": None
        }, room=getattr(request, 'sid', None))

@socketio.on('disconnect')
def handle_disconnect(arg=None) -> None:
    try:
        req_sid = safe_sid()
    except Exception:
        req_sid = getattr(request, 'sid', None)
        if not isinstance(req_sid, str):
            req_sid = None
    logical = session_manager.unbind_socket(req_sid) if req_sid else None
    logger.info({
        "level": "INFO",
        "type": "status",
        "message": f"Client disconnected (socket_sid={req_sid}, session_id={logical})",
        "session_id": logical
    })
    # Do NOT clear prompt session or cancel immediately; let prompt timeout handle it
    # prompt.clear_prompt_session(logical or req_sid)
    # cancellation_manager.remove(logical or req_sid)
    if logical:
        session_manager.pop_emitter(logical)

@socketio.on('set_output_mode')
def handle_set_output_mode(data) -> None:
    mode = safe_lower(safe_get(data, "mode", "live"))
    valid_modes = {"live", "batch"}
    session_id = resolve_session_id(data, create_if_missing=False)
    if not session_id:
        logger.error({
            "level": "ERROR",
            "type": "status",
            "message": "No session_id provided.",
            "session_id": None
        })
        return
    if mode in valid_modes:
        session['output_mode'] = mode
        logger.info({
            "level": "INFO",
            "type": "status",
            "message": f"Output mode set to {mode}.",
            "session_id": session_id
        })
    else:
        logger.error({
            "level": "ERROR",
            "type": "status",
            "message": "Invalid output mode.",
            "session_id": session_id
        })

@socketio.on('parser_prompt')
def handle_parser_prompt(data) -> None:
    print("Received parser_prompt:", data)
    print("Current prompt sessions:", list(prompt.prompt_sessions.keys()))
    session_id = resolve_session_id(data, create_if_missing=False)
    value = data.get("value", "") if isinstance(data, dict) else data
    if not session_id or not session_manager.has_session(session_id):
        logger.error({
            "level": "ERROR",
            "type": "prompt",
            "message": "Invalid or unknown session_id for prompt.",
            "session_id": None,
        })
        return
    prompt_session = prompt.prompt_sessions.get(session_id)
    if prompt_session:
        prompt_session.set_response(value)
        transition_session(
            session_id,
            SessionState.RUNNING,
            locked=True,
            phase=PipelinePhase.RUN,
            broadcast=False,
            extras={
                "manual_source": get_manual_source(session_id),
                "manual_source_origin": get_manual_source_origin(session_id),
            },
        )

@socketio.on('cancel_parser')
def handle_cancel_parser(data=None) -> None:
    session_id = resolve_session_id(data or {}, create_if_missing=False)
    if not session_id:
        logger.error({
            "level": "ERROR",
            "type": "cancel",
            "message": "No session_id provided for cancel.",
            "session_id": None
        })
        return
    cancel_processing(session_id)
    logger.info({
        "level": "INFO",
        "type": "cancel",
        "message": "Cancellation requested",
        "session_id": session_id
    })
    session_manager.pop_thread(session_id)
    session_manager.drop_prompt_queue(session_id)
    try:
        prompt.clear_queued_responses(session_id)
    except Exception:
        logger.debug({
            "level": "DEBUG",
            "type": "cancel",
            "message": "Failed to clear queued prompt responses during cancel.",
            "session_id": session_id
        })
    session_manager.pop_emitter(session_id)
    transition_session(
        session_id,
        SessionState.CANCELLING,
        locked=False,
        phase=PipelinePhase.RUN,
        extras={
            "manual_source": get_manual_source(session_id),
            "manual_source_origin": get_manual_source_origin(session_id),
        },
    )
    cleanup_sessions()

@socketio.on('toggle_output_bypass')
def handle_toggle_output_bypass(data=None):
    sid = resolve_session_id(data or {}, create_if_missing=False)
    if not sid:
        logger.error({
            "level": "ERROR",
            "type": "status",
            "message": "No session_id for output bypass toggle.",
            "session_id": None
        })
        return
    current = session_manager.is_output_bypassed(sid)
    state = session_manager.set_output_bypass(sid, not current)
    emit('output_bypass_state', {"session_id": sid, "output_bypass": state}, room=sid)
    logger.info({
        "level": "INFO",
        "type": "status",
        "message": f"Output bypass {'ENABLED' if state else 'DISABLED'}",
        "session_id": sid
    })

@socketio.on('set_manual_source')
def handle_set_manual_source(data=None):
    sid = resolve_session_id(data or {}, create_if_missing=False)
    source = safe_lower(safe_get(data or {}, 'file_source', ''))
    if not sid or source not in {'input', 'uploads'}:
        logger.error({
            "level": "ERROR",
            "type": "input",
            "message": "Invalid manual source update.",
            "session_id": sid
        })
        return
    origin = safe_lower(safe_get(data or {}, 'origin', 'user' if source == 'uploads' else 'default'))
    if origin not in {'user', 'default', 'server'}:
        origin = 'user' if source == 'uploads' else 'default'
    session_manager.set_manual_source(sid, source, origin=origin)
    transition_session(
        sid,
        SessionState.IDLE,
        locked=False,
        phase=PipelinePhase.SOURCE,
        broadcast=False,
        extras={"manual_source": source, "manual_source_origin": origin},
    )
    broadcast_sessions()
    logger.info({
        "level": "INFO",
        "type": "status",
        "message": f"Manual file source set to '{source}'.",
        "session_id": sid
    })
    # Notify client so UI stays in sync
    emit(
        'manual_source_state',
        {"session_id": sid, "file_source": source, "manual_source_origin": origin},
        room=sid,
    )

@socketio.on('delete_session')
def handle_delete_session(data) -> None:
    sid = resolve_session_id(data or {}, create_if_missing=False)
    if not isinstance(sid, str):
        logger.warning({
            "level": "WARNING",
            "type": "delete",
            "message": f"Invalid session_id type in delete: {type(sid)} value: {sid}",
            "session_id": sid
        })
        return
    session_manager.delete_session(sid)
    # Remove log file from disk
    try:
        log_path = os.path.join(LOG_DIR, f"sess_{sid}.ndjson")
        if os.path.exists(log_path):
            os.remove(log_path)
    except Exception:
        pass
    emit('session_deleted', {'session_id': sid}, broadcast=True)
    broadcast_sessions()

@socketio.on('run_parser')
def handle_run_parser(data=None) -> None:
    """
    Ensures session join is fully propagated before emitting any logs,
    synchronizes session/thread state, and launches the parser pipeline.
    """
    cleanup_sessions()
    payload = data if isinstance(data, dict) else {}
    session_id = resolve_session_id(payload, create_if_missing=True)
    if not isinstance(session_id, str):
        logger.error({
            "level": "ERROR",
            "type": "status",
            "message": "Unable to resolve session_id.",
            "session_id": None
        })
        return

    # --- Ensure join_room is fully propagated before any log emission ---
    join_room(session_id)
    socketio.sleep(0.25)  # More robust than time.sleep for Flask-SocketIO (yields event loop)

    # --- Sync socket/session mapping ---
    try:
        socket_sid = safe_sid()
    except Exception:
        socket_sid = getattr(request, 'sid', None)
    if isinstance(socket_sid, str):
        session_manager.bind_socket(socket_sid, session_id)

    # --- Ensure session metadata exists ---
    if not session_manager.has_session(session_id):
        create_session_metadata(session_id)
    meta = session_manager.get_metadata(session_id) or {}
    session_manager.mark_active(session_id)
    session_manager.touch_session(session_id)

    # --- Prevent concurrent runs ---
    if safe_get(meta, 'locked') and safe_is_alive(session_id):
        logger.error({
            "level": "ERROR",
            "type": "status",
            "message": "Session is locked. Wait for current job to finish.",
            "session_id": session_id
        })
        return
    if safe_is_alive(session_id):
        logger.warning({
            "level": "WARNING",
            "type": "status",
            "message": "Parser already running for this session.",
            "session_id": session_id
        })
        return

    # --- Session config ---
    requested_source = safe_lower(safe_get(payload, 'file_source', get_manual_source(session_id)))
    if requested_source not in {'input', 'uploads'}:
        requested_source = 'input'
    requested_origin = safe_lower(safe_get(payload, 'manual_source_origin', None))
    if requested_origin not in {'user', 'default', 'server'}:
        requested_origin = 'user' if safe_get(payload, 'file_source') == 'uploads' else session_manager.get_manual_source_origin(session_id)

    force_parse_input_file = None
    force_parse_format = None
    manual_upload_rel = None
    manual_upload_name = safe_strip(safe_get(payload, 'manual_upload_name', ''))
    raw_manual_upload_path = safe_strip(safe_get(payload, 'manual_upload_path', ''))
    abs_uploads_dir = os.path.abspath(UPLOADS_DIR)

    if raw_manual_upload_path:
        normalized_rel = raw_manual_upload_path.replace('\\', '/').strip('/')
        candidate_path = os.path.normpath(os.path.join(abs_uploads_dir, normalized_rel))
        if candidate_path.startswith(abs_uploads_dir) and os.path.isfile(candidate_path):
            manual_upload_rel = normalized_rel
            if not manual_upload_name:
                manual_upload_name = os.path.basename(candidate_path)
            requested_source = 'uploads'
            requested_origin = 'user'
            force_parse_input_file = manual_upload_rel
            guessed_ext = ''
            try:
                _, ext = os.path.splitext(manual_upload_name or manual_upload_rel)
                guessed_ext = safe_lower(ext.lstrip('.'))
            except Exception:
                guessed_ext = ''
            if guessed_ext:
                force_parse_format = guessed_ext
            session['FORCE_PARSE_INPUT_FILE'] = manual_upload_rel
            session['FORCE_PARSE_FORMAT'] = force_parse_format or guessed_ext or ''
            session['manual_source_pref'] = 'uploads'
            logger.info({
                "level": "INFO",
                "type": "manual_override",
                "message": f"[ManualOverride] Using uploaded file: {manual_upload_rel}",
                "session_id": session_id
            })
        else:
            logger.warning({
                "level": "WARNING",
                "type": "manual_override",
                "message": f"[ManualOverride] Invalid manual upload selection: {raw_manual_upload_path}",
                "session_id": session_id
            })

    if requested_source == 'uploads' and force_parse_input_file is None:
        force_parse_input_file = session.get('FORCE_PARSE_INPUT_FILE')
        force_parse_format = session.get('FORCE_PARSE_FORMAT')

    raw_direct_urls = safe_get(payload, 'direct_urls', [])
    direct_urls = []
    if isinstance(raw_direct_urls, list):
        for entry in raw_direct_urls:
            url_text = safe_strip(entry)
            if not url_text:
                continue
            try:
                parsed = urlparse(url_text)
            except Exception:
                parsed = None
            if not parsed or parsed.scheme not in {'http', 'https'} or parsed.username or parsed.password:
                logger.warning({
                    "level": "WARNING",
                    "type": "input",
                    "message": f"Ignoring invalid direct URL: {url_text}",
                    "session_id": session_id
                })
                continue
            direct_urls.append(url_text)
    if len(direct_urls) > DIRECT_URL_LIMIT:
        logger.warning({
            "level": "WARNING",
            "type": "input",
            "message": f"Direct URL list trimmed to {DIRECT_URL_LIMIT} entries.",
            "session_id": session_id
        })
        direct_urls = direct_urls[:DIRECT_URL_LIMIT]
    if direct_urls and requested_source == 'uploads':
        logger.warning({
            "level": "WARNING",
            "type": "input",
            "message": "Direct URLs ignored because manual uploads source is active.",
            "session_id": session_id
        })
        direct_urls = []
    if direct_urls:
        logger.info({
            "level": "INFO",
            "type": "input",
            "message": f"Direct URL override engaged with {len(direct_urls)} link(s).",
            "session_id": session_id,
            "urls": direct_urls
        })

    session_manager.set_manual_source(session_id, requested_source, origin=requested_origin)
    output_bypass_flag = is_output_bypassed(session_id)
    lock_session(session_id)


    # --- Register per-session emitter (used by prompt/manual emits) ---
    session_manager.register_emitter(session_id, socketio_emit_func)

    # --- Install dispatcher only ONCE globally (idempotent) ---
    logger.set_mode("webapp")
    logger.set_format("json")
    def filtered_emit(line):
        try:
            obj = orjson.loads(line) if isinstance(line, str) and line.strip().startswith("{") else None
        except Exception:
            obj = None
        lvl = (obj or {}).get("level") or ""
        if lvl.upper() in WEBAPP_CONSOLE_LEVELS:
            logger.enable_console_echo_webapp(True)
        else:
            logger.enable_console_echo_webapp(False)
        socketio_emit_func(line)
    logger.set_socketio_emit_func(filtered_emit)
    prompt.set_mode("webapp")
    prompt.set_socketio_emit_func(lambda msg: socketio.emit(
        'parser_output',
        normalize_log_obj(msg if isinstance(msg, dict) else {
            "level": "info",
            "type": "prompt",
            "message": str(msg),
            "session_id": session_id
        }),
        room=session_id
    ))

    # --- Now emit initial logs (client is guaranteed in room) ---
    logger.info({
        "level": "INFO",
        "type": "status",
        "message": "Parser connected. Starting parser run...",
        "session_id": session_id
    })
    logger.info({
        "level": "INFO",
        "type": "status",
        "message": f"Launching parser (source={requested_source}, output_bypass={'on' if output_bypass_flag else 'off'})",
        "session_id": session_id
    })

    # --- Run event logging ---
    run_id = f"run_{int(time.time()*1000)}"
    start_ts = datetime.now(timezone.utc).isoformat()
    log_run_event({
        "type": "start",
        "run_id": run_id,
        "session_id": session_id,
        "ts": start_ts,
        "source": requested_source,
        "output_bypass": output_bypass_flag,
        "status": "running",
        "manual_upload": manual_upload_rel,
        "direct_url_count": len(direct_urls)
    })

    # --- Prepare cancellation and prompt queue ---
    cancel_flag = cancellation_manager.get_flag(session_id)
    prompt_queue = get_prompt_queue(session_id)

    # --- Launch parser in a dedicated thread ---
    def worker_wrapper():
        start_time = time.time()
        session_manager.bind_thread_id(threading.get_ident(), session_id)
        status = "error"  # Default to error
        err = None
        try:
            process_urls_for_web(
                prompt_queue,
                session_id,
                cancel_flag,
                emit_func=socketio_emit_func,
                output_bypass=output_bypass_flag,
                manual_source=requested_source,
                disable_internal_heartbeat=True,
                force_parse_input_file=force_parse_input_file,
                force_parse_format=force_parse_format,
                urls=direct_urls if direct_urls else None
            )
            logger.info({
                "level": "INFO",
                "type": "status",
                "message": "Parser run completed.",
                "session_id": session_id
            })
            status = "ok"
            err = None
        except Exception as e:
            logger.error({
                "level": "ERROR",
                "type": "exception",
                "message": f"Parser run failed: {e}",
                "session_id": session_id
            })
            status = "error"
            err = str(e)
        finally:
            duration_ms = int((time.time() - start_time)*1000)
            log_run_event({
                "type": "end",
                "run_id": run_id,
                "session_id": session_id,
                "ts": datetime.now(timezone.utc).isoformat(),
                "source": requested_source,
                "output_bypass": output_bypass_flag,
                "status": status,
                "error": err,
                "duration_ms": duration_ms,
                "manual_upload": manual_upload_rel,
                "direct_url_count": len(direct_urls)
            })
            session_manager.pop_thread(session_id)
            final_state = SessionState.COMPLETED
            if safe_is_set(cancel_flag):
                final_state = SessionState.CANCELLED
            elif status != "ok":
                final_state = SessionState.ERROR
            extras = {
                "manual_source": requested_source,
                "manual_source_origin": requested_origin,
                "run_id": run_id,
                "output_bypass": output_bypass_flag,
                "manual_upload_file": manual_upload_rel,
                "direct_url_count": len(direct_urls),
                "direct_urls": direct_urls,
            }
            if err:
                extras["last_error"] = err
            transition_session(
                session_id,
                final_state,
                locked=False,
                phase=None,
                extras=extras,
            )
        session_manager.unbind_thread_id(threading.get_ident())

    thread = socketio.start_background_task(worker_wrapper)
    session_manager.set_thread(session_id, thread)

# Heartbeat thread startup (idempotent)
if 'heartbeat_thread' not in globals() or not isinstance(globals().get('heartbeat_thread'), Thread) or not globals()['heartbeat_thread'].is_alive():
    if HEARTBEAT_ENABLED:
        heartbeat_thread = Thread(target=_heartbeat_loop, name="heartbeat-loop", daemon=True)
        heartbeat_thread.start()

# Proactively ensure tables at startup (non-fatal if fails)
ensure_db_tables()

# Clean up old session log files on startup (keep only active or recent)
cleanup_old_log_files(LOG_DIR, session_manager.list_active_session_ids(), keep_days=7)
        
# 7. Main Entrypoint
if __name__ == "__main__":
    try:
        port = int(os.environ.get("PORT", 5000))
        allow_unsafe = _env_flag("SMART_ELECTIONS_ALLOW_UNSAFE_WERKZEUG", False)
        socketio.run(
            app,
            host="0.0.0.0",
            port=port,
            debug=False,
            use_reloader=False,
            allow_unsafe_werkzeug=allow_unsafe,
        )
    finally:
        _shutdown_event.set()