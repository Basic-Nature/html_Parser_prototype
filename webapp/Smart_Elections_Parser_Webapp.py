from __future__ import annotations

import hmac
import os

# ============================================
# SocketIO Configuration: Threading Framework
# ============================================
# Using Python's native threading framework for reliable, maintainable async support.
# This avoids eventlet (deprecated) and provides stable, predictable behavior.
# ============================================

_SOCKETIO_ASYNC_MODE = "threading"

_SOCKETIO_ENGINE_OPTIONS = {
    "ping_interval": 10,
    "ping_timeout": 60,
    "allow_upgrades": False,
    "transports": ["polling"],
}

_SOCKETIO_CLIENT_TRANSPORTS = ["polling"]

# Env-driven allowlist; avoid wildcard in production. Defaults cover local dev.
_RAW_SOCKETIO_ORIGINS = os.environ.get(
    "SOCKETIO_ALLOWED_ORIGINS",
    "http://localhost:5000,http://127.0.0.1:5000,http://localhost:3000,http://127.0.0.1:3000",
)
SOCKETIO_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in _RAW_SOCKETIO_ORIGINS.split(",")
    if origin.strip()
]
if not SOCKETIO_ALLOWED_ORIGINS:
    SOCKETIO_ALLOWED_ORIGINS = ["http://localhost:5000", "http://127.0.0.1:5000"]

SOCKETIO_CLIENT_CONFIG = {
    "transports": _SOCKETIO_CLIENT_TRANSPORTS,
    "pollingOnly": True,
    "pingInterval": int(_SOCKETIO_ENGINE_OPTIONS["ping_interval"] * 1000),
    "pingTimeout": int(_SOCKETIO_ENGINE_OPTIONS["ping_timeout"] * 1000),
}

SOCKETIO_MESSAGE_QUEUE = os.environ.get("SOCKETIO_MESSAGE_QUEUE")
SOCKETIO_MESSAGE_CHANNEL = os.environ.get("SOCKETIO_MESSAGE_CHANNEL", "socketio")

import csv
import io
import gzip
import json
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
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
from psycopg2 import errors as pg_errors
from sqlalchemy import inspect, text
from sqlalchemy.exc import OperationalError
from werkzeug.exceptions import HTTPException, NotFound

# Socket.IO imports
try:
    from flask_socketio import SocketIO, emit, join_room
except Exception:
    # If flask_socketio is missing at runtime, let the import fail later when starting the server
    # but avoid NameError during static analysis.
    SocketIO = None
    def emit(*a, **k):
        raise RuntimeError('SocketIO not available')
    def join_room(*a, **k):
        raise RuntimeError('SocketIO not available')

# Optional rate limiting (best-effort)
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
except Exception:
    Limiter = None
    def get_remote_address():
        return "unknown"

# Global storage for last contest options for re-emission on reconnect
last_contest_options = {}

# DB tables init flag
_tables_initialized = False

# Local health/session utilities
from webapp.parser.health.integrity_monitor import get_integrity_monitor
from webapp.parser.health.session_manager import SessionManager
from webapp.parser.utils.logger_singleton import logger, prompt
from webapp.parser.utils.session_state import (
    DEFAULT_PHASE_BY_STATE,
    PipelinePhase,
    SessionState,
    export_session_enums,
)

try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    # python-dotenv not installed (e.g., on Azure), skip loading .env
    pass

# Import shared config constants and helper utilities used by many routes
from webapp.parser.config import (
    ALLOW_GOOGLE_DOCS,
    ALLOW_LEGACY_OUTPUT_DOWNLOAD,
    DATA_API_URL,
    DEPLOY_ENV,
    INPUT_DIR,
    LOG_DIR,
    MAX_CSV_ROWS,
    MAX_PDF_PAGES,
    MAX_SOCKET_EVENT_BYTES,
    MAX_SOCKET_LOG_BYTES,
    MAX_UPLOAD_BYTES,
    MAX_UPLOAD_SIZE_MB,
    MAX_XLSX_BYTES,
    OUTPUT_DIR,
    POSTGRES_DB,
    POSTGRES_HOST,
    POSTGRES_PASSWORD_RAW,
    POSTGRES_PORT,
    POSTGRES_USER_RAW,
    PROJECT_ROOT,
    QUICK_COPY_DIR,
    RUN_HISTORY_FILE,
    SUPPORTED_EXTENSION_SET,
    UPLOADS_DIR,
    URL_ALLOWLIST_HOSTS,
    URL_ALLOWLIST_SUFFIXES,
    URL_BLOCK_PRIVATE_IPS,
    URL_ENFORCE_ALLOWLIST,
    URL_LIST_FILE,
)
from webapp.parser.utils.cert_utils import extract_client_principal
from webapp.parser.utils.db_utils import SessionLocal, get_engine
from webapp.parser.utils.misc_utils import extract_url_and_label, load_processed_urls
from webapp.parser.utils.shared_logic import (
    safe_filename,
    safe_get,
    safe_is_set,
    safe_lower,
    safe_rsplit,
    safe_sid,
    safe_split,
    safe_strip,
    safe_validate_external_url,
)
from webapp.parser.utils.models import DataFrameworkPreviewCache
from webapp.parser.web_pipeline import (
    cancel_processing,
    cancellation_manager,
    process_urls_for_web,
)

# Health task security controls
ENABLE_HEALTH_TASKS = os.environ.get("ENABLE_HEALTH_TASKS", "false").lower() in {"1", "true", "yes"}
HEALTH_TASK_TOKEN = os.environ.get("HEALTH_TASK_TOKEN")

# Local, non-DB monitoring log for DB usage/events
DB_MONITOR_FILE = LOG_DIR / "db_monitor.jsonl"
try:
    DB_MONITOR_FILE.touch(exist_ok=True)
except Exception:
    pass

# Flagged URL audit log (rotated daily, small caps)
FLAGGED_URL_SIZE_CAP = 5 * 1024 * 1024  # ~5MB per daily file
FLAGGED_URL_RETENTION_DAYS = 30


def _flagged_url_log_dir() -> Path:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return LOG_DIR


def _rotate_flagged_url_path(now: datetime | None = None) -> Path:
    """Return a safe path for today's flagged URL log, adding part suffix if size cap exceeded."""
    now = now or datetime.now(timezone.utc)
    base = _flagged_url_log_dir()
    prefix = base / f"flagged_urls-{now.strftime('%Y%m%d')}"
    candidate = prefix.with_suffix('.jsonl')
    if candidate.exists() and candidate.stat().st_size >= FLAGGED_URL_SIZE_CAP:
        # find next part
        part = 1
        while True:
            cand = prefix.with_name(f"{prefix.name}-part{part}").with_suffix('.jsonl')
            if not cand.exists() or cand.stat().st_size < FLAGGED_URL_SIZE_CAP:
                candidate = cand
                break
            part += 1
    return candidate


def _prune_flagged_url_logs(now: datetime | None = None) -> None:
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=FLAGGED_URL_RETENTION_DAYS)
    base = _flagged_url_log_dir()
    try:
        for entry in base.glob('flagged_urls-*.jsonl'):
            try:
                mtime = datetime.fromtimestamp(entry.stat().st_mtime, tz=timezone.utc)
                if mtime < cutoff:
                    entry.unlink(missing_ok=True)
            except Exception:
                continue
    except Exception:
        pass


def _is_local_request() -> bool:
    try:
        host = (request.host or "").split(":", 1)[0].lower()
        if host in {"localhost", "127.0.0.1", "::1"}:
            return True
        forwarded_for = (request.headers.get("X-Forwarded-For") or "").split(",", 1)[0].strip()
        remote_addr = forwarded_for or (request.remote_addr or "")
        if remote_addr in {"127.0.0.1", "::1"}:
            return True
        if remote_addr.startswith("127."):
            return True
    except Exception:
        return False
    return False


def _guarded_ingestion_allowed(action: str) -> tuple[bool, str]:
    if DEPLOY_ENV == "local" and _is_local_request():
        return True, "local_bypass"
    guarded_key = (os.environ.get("GUARDED_INGESTION_KEY") or "").strip()
    if not guarded_key:
        return False, "guard_key_missing"
    token_hdr = (request.headers.get("X-Guarded-Ingestion-Key") or request.headers.get("X-Guarded-Key") or "").strip()
    if token_hdr and hmac.compare_digest(token_hdr, guarded_key):
        return True, "guard_header"
    auth_hdr = (request.headers.get("Authorization") or "").strip()
    if auth_hdr.lower().startswith("bearer "):
        try:
            bearer = auth_hdr.split(None, 1)[1].strip()
            if hmac.compare_digest(bearer, guarded_key):
                return True, "guard_bearer"
        except Exception:
            pass
    return False, "guard_key_invalid"


def log_flagged_url(event: dict) -> None:
    payload = dict(event or {})
    payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    path = _rotate_flagged_url_path()
    try:
        with open(path, 'ab') as f:
            f.write(orjson.dumps(payload) + b"\n")
    except Exception:
        # best effort; no raise to avoid blocking request
        return
    _prune_flagged_url_logs()

# 2. Flask App & SocketIO Initialization
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

limiter = None
if Limiter is not None:
    try:
        limiter = Limiter(
            get_remote_address,
            app=app,
            default_limits=[],
            storage_uri=os.environ.get("RATE_LIMIT_STORAGE_URI", "memory://"),
        )
    except Exception:
        limiter = None

socketio_kwargs = {}
if SOCKETIO_MESSAGE_QUEUE:
    socketio_kwargs["message_queue"] = SOCKETIO_MESSAGE_QUEUE
    socketio_kwargs["channel"] = SOCKETIO_MESSAGE_CHANNEL

socketio = SocketIO(
    app,
    async_mode=_SOCKETIO_ASYNC_MODE,
    cors_allowed_origins=SOCKETIO_ALLOWED_ORIGINS,
    **_SOCKETIO_ENGINE_OPTIONS,
    **socketio_kwargs,
)


# Optional Prometheus scrape endpoint
ENABLE_PROMETHEUS = os.environ.get('ENABLE_PROMETHEUS', 'false').lower() in ('1', 'true', 'yes')
TEST_METRICS_ROUTE_ENABLED = os.environ.get('ENABLE_TEST_METRICS_ROUTE', 'false').lower() in ('1', 'true', 'yes') or os.environ.get('FLASK_ENV', '').lower() == 'development' or os.environ.get('PYTEST_CURRENT_TEST', '')
TEST_UI_ROUTES_ENABLED = os.environ.get('ENABLE_TEST_UI_ROUTES', 'false').lower() in ('1', 'true', 'yes') or os.environ.get('FLASK_ENV', '').lower() == 'development'
if ENABLE_PROMETHEUS:
    try:
        from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest

        @app.route('/metrics')
        def metrics():
            try:
                data = generate_latest(REGISTRY)
                return Response(data, mimetype=CONTENT_TYPE_LATEST)
            except Exception as e:
                try:
                    logger.error({"level": "ERROR", "type": "metrics", "message": f"/metrics error: {e}"})
                except Exception:
                    pass
                return Response("Internal metrics error", status=500, mimetype='text/plain')

        # --- Test-only route to increment metrics for deterministic tests ---
        if TEST_METRICS_ROUTE_ENABLED:
            @app.route('/test/metrics/increment', methods=['POST'])
            def test_metrics_increment():
                try:
                    from webapp.parser.utils.metrics_prom import increment_test_counter
                except Exception as e:
                    return jsonify({"success": False, "error": f"metrics_prom import failed: {e}"}), 500
                try:
                    increment_test_counter()
                    return jsonify({"success": True, "message": "Test counter incremented."})
                except Exception as e:
                    return jsonify({"success": False, "error": str(e)}), 500
    except Exception:
        try:
            logger.info({"level": "INFO", "type": "metrics", "message": "Prometheus client not available; /metrics disabled."})
        except Exception:
            pass

# If Prometheus is enabled, try to import the internal metrics module so counters are registered
if os.environ.get('ENABLE_PROMETHEUS', 'false').lower() in ('1', 'true', 'yes'):
    try:
        pass  # ensure counters are created on import
    except Exception:
        try:
            logger.debug({"level": "DEBUG", "type": "metrics", "message": "Failed to import metrics_prom on startup."})
        except Exception:
            pass

# --- Health task orchestration (Azure control center) ---
HEALTH_TASK_DEFINITIONS: dict[str, dict] = {
    "health_router_full": {
        "label": "Full Health Router",
        "description": "Run the entire BotPipeline: clean logs, migrate context, manual correction, and retraining.",
        "command": ["-m", "webapp.parser.health.health_router"],
        "danger": True,
    },
    "manual_correction_auto": {
        "label": "Manual Correction (Auto)",
        "description": "Auto-accept new context entries without prompts using manual_correction_bot --auto.",
        "command": ["-m", "webapp.parser.health.manual_correction_bot", "--auto"],
        "danger": True,
    },
    "manual_correction_enhanced": {
        "label": "Manual Correction (Enhanced)",
        "description": "Launch manual_correction_bot with enhanced review (interactive, slower but precise).",
        "command": ["-m", "webapp.parser.health.manual_correction_bot", "--enhanced"],
        "danger": True,
    },
    "retrain_table_models": {
        "label": "Retrain Table Models",
        "description": "Trigger retrain_table_structure_models to refresh structure detection weights.",
        "command": ["-m", "webapp.parser.health.retrain_table_structure_models"],
    },
    "scan_misaligned": {
        "label": "Scan Misaligned NER",
        "description": "Run scan_misaligned_ner to flag mismatched training samples before retraining.",
        "command": ["-m", "webapp.parser.health.scan_misaligned_ner"],
    },
    "log_cache_cleaner": {
        "label": "Log & Cache Cleaner",
        "description": "Execute log_cache_cleaner_bot to dedupe/cap JSONL files and watch sizes.",
        "command": ["-m", "webapp.parser.health.log_cache_cleaner_bot"],
    },
    "context_migration": {
        "label": "Context Migration",
        "description": "Run context_migration to sync historical context formats with the latest schema.",
        "command": ["-m", "webapp.parser.health.context_migration"],
    },
    "integrity_check_summary": {
        "label": "Integrity Check Summary",
        "description": "Stream Integrity_check findings for the current context library.",
        "command": ["-m", "webapp.parser.health.integrity_check_runner"],
    },
    "dataset_promotion_latest": {
        "label": "Dataset Promotion (Latest)",
        "description": "Promote the newest output folder into warehouse_election_results with guarded batching.",
        "command": ["-m", "webapp.parser.health.dataset_promotion"],
        "danger": True,
    },
}

_HEALTH_TASK_LOCK = threading.Lock()
_HEALTH_TASK_RUNS: dict[str, dict] = {}
_HEALTH_TASK_HISTORY_LIMIT = 20
_HEALTH_TASK_LOG_LIMIT = 20000


def _require_health_auth():
    """Guard health endpoints with enable flag and optional bearer token."""
    if not ENABLE_HEALTH_TASKS:
        return False, (jsonify({"error": "Health tasks disabled"}), 403)

    if HEALTH_TASK_TOKEN:
        auth_header = request.headers.get("Authorization", "") or ""
        token = None
        if auth_header.lower().startswith("bearer "):
            token = auth_header.split(" ", 1)[1].strip()
        if not token:
            token = request.args.get("token", "")
        if token and hmac.compare_digest(token, HEALTH_TASK_TOKEN):
            return True, None
        return False, (jsonify({"error": "Unauthorized"}), 401)

    return True, None


def _health_auth_response():
    allowed, resp = _require_health_auth()
    return None if allowed else resp


def _public_health_task_definitions() -> list[dict]:
    entries = []
    for key, meta in HEALTH_TASK_DEFINITIONS.items():
        entries.append({
            "key": key,
            "label": meta["label"],
            "description": meta["description"],
            "danger": bool(meta.get("danger")),
        })
    return entries


def _get_health_tasks() -> list[dict]:
    with _HEALTH_TASK_LOCK:
        records = [dict(task) for task in _HEALTH_TASK_RUNS.values()]
    records.sort(key=lambda item: item.get("started_at"), reverse=True)
    return records


def _get_health_task(task_id: str) -> dict | None:
    with _HEALTH_TASK_LOCK:
        task = _HEALTH_TASK_RUNS.get(task_id)
        return dict(task) if task else None


def _append_health_task_log(task_id: str, chunk: str) -> None:
    if not chunk:
        return
    if not chunk.endswith("\n"):
        chunk += "\n"
    with _HEALTH_TASK_LOCK:
        record = _HEALTH_TASK_RUNS.get(task_id)
        if not record:
            return
        log = record.get("log", "") + chunk
        if len(log) > _HEALTH_TASK_LOG_LIMIT:
            log = log[-_HEALTH_TASK_LOG_LIMIT:]
        record["log"] = log
        record["last_update"] = datetime.now(timezone.utc).isoformat()


def _trim_health_task_history() -> None:
    if len(_HEALTH_TASK_RUNS) <= _HEALTH_TASK_HISTORY_LIMIT:
        return
    removable = sorted(
        (task for task in _HEALTH_TASK_RUNS.values() if task.get("status") in {"completed", "failed"}),
        key=lambda item: item.get("started_at") or "",
    )
    while len(_HEALTH_TASK_RUNS) > _HEALTH_TASK_HISTORY_LIMIT and removable:
        oldest = removable.pop(0)
        _HEALTH_TASK_RUNS.pop(oldest["id"], None)


def _finalize_health_task(task_id: str, status: str) -> None:
    with _HEALTH_TASK_LOCK:
        record = _HEALTH_TASK_RUNS.get(task_id)
        if not record:
            return
        record["status"] = status
        finished = datetime.now(timezone.utc).isoformat()
        record["ended_at"] = finished
        record["last_update"] = finished


def _launch_health_task(task_key: str) -> dict:
    definition = HEALTH_TASK_DEFINITIONS[task_key]
    task_id = secrets.token_hex(8)
    now_iso = datetime.now(timezone.utc).isoformat()
    record = {
        "id": task_id,
        "task": task_key,
        "label": definition["label"],
        "description": definition["description"],
        "status": "running",
        "log": "",
        "started_at": now_iso,
        "ended_at": None,
        "last_update": now_iso,
        "danger": bool(definition.get("danger")),
    }
    with _HEALTH_TASK_LOCK:
        _HEALTH_TASK_RUNS[task_id] = record
        _trim_health_task_history()
    worker = Thread(target=_run_health_task, args=(task_id,), daemon=True)
    worker.start()
    return dict(record)


def _run_health_task(task_id: str) -> None:
    with _HEALTH_TASK_LOCK:
        record = _HEALTH_TASK_RUNS.get(task_id)
    if not record:
        return
    definition = HEALTH_TASK_DEFINITIONS.get(record["task"])
    if not definition:
        _append_health_task_log(task_id, f"[ERROR] Unknown task '{record['task']}'.")
        _finalize_health_task(task_id, "failed")
        return

    success = False
    try:
        callable_runner = definition.get("callable")
        if callable_runner:
            callable_runner(lambda chunk: _append_health_task_log(task_id, chunk))
            success = True
        else:
            command = [sys.executable, *definition["command"]]
            _append_health_task_log(task_id, f"[CMD] {' '.join(command)}")
            proc = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            try:
                if proc.stdout:
                    for line in proc.stdout:
                        _append_health_task_log(task_id, line.rstrip("\n"))
            finally:
                if proc.stdout:
                    proc.stdout.close()
            returncode = proc.wait()
            success = returncode == 0
            if not success:
                _append_health_task_log(task_id, f"[ERROR] Command exited with {returncode}.")
    except Exception as exc:
        _append_health_task_log(task_id, f"[ERROR] {exc}")
    finally:
        _finalize_health_task(task_id, "completed" if success else "failed")

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


def _is_request_secure() -> bool:
    """Detect HTTPS even when behind a reverse proxy/front door."""
    if request.is_secure:
        return True
    forwarded = (request.headers.get("X-Forwarded-Proto") or "").split(",")[0].strip().lower()
    return forwarded == "https"
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

# Register Verification Framework Blueprint
try:
    from webapp.parser.verification_endpoints import verification_bp
    app.register_blueprint(verification_bp)
    logger.info({
        "level": "INFO",
        "type": "status",
        "message": "Verification Framework blueprint registered",
        "session_id": None
    })
except Exception as e:
    logger.warning({
        "level": "WARNING",
        "type": "status",
        "message": f"Failed to register Verification Framework blueprint: {e}",
        "session_id": None
    })

# Register Quarantine Review Blueprint
try:
    from webapp.parser.quarantine_endpoints import quarantine_bp
    app.register_blueprint(quarantine_bp)
    logger.info({
        "level": "INFO",
        "type": "status",
        "message": "Quarantine Review blueprint registered",
        "session_id": None
    })
except Exception as e:
    logger.warning({
        "level": "WARNING",
        "type": "status",
        "message": f"Failed to register Quarantine Review blueprint: {e}",
        "session_id": None
    })

# Register Data Assurance Blueprint
try:
    from webapp.parser.quality_assurance import qa_bp
    app.register_blueprint(qa_bp)
    logger.info({
        "level": "INFO",
        "type": "status",
        "message": "Data Assurance (DL1/DL2 Classification) blueprint registered",
        "session_id": None
    })
except Exception as e:
    logger.warning({
        "level": "WARNING",
        "type": "status",
        "message": f"Failed to register Data Assurance blueprint: {e}",
        "session_id": None
    })

# 3. Session & State Management
session_manager = SessionManager()

ENABLE_FINGERPRINT_SESSION_RECOVERY = os.environ.get(
    "ENABLE_FINGERPRINT_SESSION_RECOVERY",
    "true",
).lower() in {"1", "true", "yes"}

ALLOW_DEV_NO_PRINCIPAL = os.environ.get("ALLOW_DEV_NO_PRINCIPAL", "false").lower() in {"1", "true", "yes"}
ALLOW_AUTO_SESSION_REUSE = os.environ.get("ALLOW_AUTO_SESSION_REUSE", "true").lower() in {"1", "true", "yes"}
CERT_SESSION_BINDING = os.environ.get("CERT_SESSION_BINDING", "false").lower() in {"1", "true", "yes"}
ALLOW_ANON_NO_PRINCIPAL = os.environ.get("ALLOW_ANON_NO_PRINCIPAL", "true").lower() in {"1", "true", "yes"}
DEV_ISOLATION_BYPASS_ENABLED = os.environ.get("DEV_ISOLATION_BYPASS_ENABLED", "false").lower() in {"1", "true", "yes"}
DEV_ISOLATION_BYPASS_IPS_RAW = os.environ.get("DEV_ISOLATION_BYPASS_IPS", "").strip()

LOG_DEDUPE_WINDOW = float(os.environ.get("LOG_DEDUPE_WINDOW_SEC", "2.0"))
SECURITY_LOG_DEDUPE_WINDOW = float(os.environ.get("SECURITY_LOG_DEDUPE_WINDOW_SEC", "12.0"))
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

# Conservative cleanup controls (avoid tearing down active or stale-but-recoverable sessions)
try:
    SESSION_EXPIRE_GRACE_SEC = max(0, int(os.environ.get("SESSION_EXPIRE_GRACE_SEC", "600")))
except ValueError:
    SESSION_EXPIRE_GRACE_SEC = 600
try:
    SESSION_STALE_RECOVERY_SEC = max(0, int(os.environ.get("SESSION_STALE_RECOVERY_SEC", "120")))
except ValueError:
    SESSION_STALE_RECOVERY_SEC = 120

DIRECT_URL_LIMIT = 20

# 4. Utility Functions

_SOCKET_RATE_BUCKETS: dict[str, dict[str, list[float]]] = {}
_SOCKET_RATE_LIMITS: dict[str, tuple[int, int]] = {
    "ballot_lens": (3, 60),
    "parser_prompt": (60, 60),
    "cancel_parser": (10, 60),
    "prompt_cancel": (10, 60),
    "set_manual_source": (15, 60),
    "toggle_output_bypass": (15, 60),
}

def _socket_payload_too_large(payload) -> bool:
    try:
        if isinstance(payload, (bytes, bytearray)):
            return len(payload) > MAX_SOCKET_EVENT_BYTES
        if isinstance(payload, str):
            return len(payload.encode("utf-8", "ignore")) > MAX_SOCKET_EVENT_BYTES
        blob = orjson.dumps(payload)
        return len(blob) > MAX_SOCKET_EVENT_BYTES
    except Exception:
        return False

def _rate_limit_socket_action(session_id: str | None, action: str) -> bool:
    limit, window = _SOCKET_RATE_LIMITS.get(action, (0, 0))
    if not limit or not window:
        return True
    if not session_id:
        return False
    now = time.time()
    bucket = _SOCKET_RATE_BUCKETS.setdefault(session_id, {}).setdefault(action, [])
    bucket[:] = [ts for ts in bucket if (now - ts) <= window]
    if len(bucket) >= limit:
        return False
    bucket.append(now)
    return True

def _rate_limit(limit: str):
    if limiter is None:
        return lambda fn: fn
    return limiter.limit(limit)

def _generate_upload_filename(original_name: str) -> str:
    ext = os.path.splitext(original_name or "")[1].lower()
    token = secrets.token_urlsafe(16).replace("-", "").replace("_", "")
    base = f"upload_{token}"
    return safe_filename(f"{base}{ext}" if ext else base, strict_mode=True)

def _enforce_request_size() -> tuple[bool, str | None]:
    try:
        content_len = request.content_length
    except Exception:
        content_len = None
    if content_len is not None and content_len > MAX_UPLOAD_BYTES:
        return False, f"Upload exceeds {MAX_UPLOAD_SIZE_MB}MB limit."
    return True, None

def _validate_uploaded_file(path: str, ext: str, session_id: str | None = None) -> tuple[bool, str | None]:
    try:
        size = os.path.getsize(path)
    except Exception:
        return False, "Upload file unreadable."
    if size > MAX_UPLOAD_BYTES:
        return False, f"Upload exceeds {MAX_UPLOAD_SIZE_MB}MB limit."
    ext = (ext or "").lower()
    if ext == ".pdf":
        try:
            import fitz  # PyMuPDF
            with fitz.open(path) as doc:
                if doc.page_count > MAX_PDF_PAGES:
                    return False, f"PDF exceeds {MAX_PDF_PAGES} pages."
        except Exception as exc:
            logger.warning({
                "level": "WARNING",
                "type": "upload",
                "message": f"Failed to inspect PDF: {exc}",
                "session_id": session_id,
            })
    elif ext == ".csv":
        try:
            import csv
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                reader = csv.reader(fh)
                count = 0
                for _ in reader:
                    count += 1
                    if count > MAX_CSV_ROWS:
                        return False, f"CSV exceeds {MAX_CSV_ROWS} rows."
        except Exception as exc:
            logger.warning({
                "level": "WARNING",
                "type": "upload",
                "message": f"Failed to inspect CSV: {exc}",
                "session_id": session_id,
            })
    elif ext in {".xlsx", ".xls"}:
        if size > MAX_XLSX_BYTES:
            return False, "Spreadsheet exceeds size limit."
    return True, None

def _save_uploaded_file(file_obj, dest_dir: str, session_id: str | None = None) -> tuple[bool, str | None, str | None]:
    if not file_obj or not allowed_file(file_obj.filename):
        return False, "Invalid file type or no file selected.", None
    ok, err = _enforce_request_size()
    if not ok:
        return False, err, None
    original_name = file_obj.filename or "upload"
    filename = _generate_upload_filename(original_name)
    save_path = os.path.join(dest_dir, filename)
    try:
        file_obj.save(save_path)
    except Exception as exc:
        return False, f"Failed to save upload: {exc}", None
    ext = os.path.splitext(filename)[1].lower()
    valid, reason = _validate_uploaded_file(save_path, ext, session_id=session_id)
    if not valid:
        try:
            os.remove(save_path)
        except Exception:
            pass
        return False, reason, None
    return True, filename, save_path

def _log_download_access(event: dict) -> None:
    payload = dict(event or {})
    payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    try:
        log_path = LOG_DIR / "download_access.jsonl"
        with open(log_path, "ab") as f:
            f.write(orjson.dumps(payload) + b"\n")
    except Exception:
        pass

def _resolve_output_metadata_path(file_path: str) -> str | None:
    if not file_path:
        return None
    parent = os.path.dirname(file_path)
    candidate = os.path.join(parent, "results.metadata.json")
    if os.path.exists(candidate):
        return candidate
    legacy = os.path.join(parent, "metadata.json")
    if os.path.exists(legacy):
        return legacy
    return None

def _quick_copy_session_dir(session_id: str | None) -> Path | None:
    if not session_id:
        return None
    safe_sid = safe_filename(session_id)
    if not safe_sid:
        return None
    return QUICK_COPY_DIR / safe_sid

def _ensure_quick_copy_dir(session_id: str | None) -> Path | None:
    target = _quick_copy_session_dir(session_id)
    if not target:
        return None
    try:
        target.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.error({
            "level": "ERROR",
            "type": "cache",
            "message": f"Failed to ensure quick copy dir: {exc}",
            "session_id": session_id,
        })
        return None
    return target

def _cleanup_quick_copy_dir(session_id: str | None) -> None:
    target = _quick_copy_session_dir(session_id)
    if not target or not target.exists():
        return
    try:
        shutil.rmtree(target)
    except Exception as exc:
        logger.warning({
            "level": "WARNING",
            "type": "cache",
            "message": f"Failed to cleanup quick copy dir: {exc}",
            "session_id": session_id,
        })

def _unique_quick_copy_name(dest_dir: Path, base_name: str) -> str:
    safe_name = safe_filename(base_name) or "file"
    stem, suffix = os.path.splitext(safe_name)
    if not stem:
        stem = "file"
    candidate = f"{stem}{suffix}"
    if not (dest_dir / candidate).exists():
        return candidate
    for idx in range(1, 1000):
        candidate = f"{stem}-{idx}{suffix}"
        if not (dest_dir / candidate).exists():
            return candidate
    return f"{stem}-{secrets.token_hex(4)}{suffix}"

def _is_output_download_allowed(file_path: str, principal: str | None, session_id: str | None) -> tuple[bool, str]:
    meta_path = _resolve_output_metadata_path(file_path)
    if not meta_path:
        return (ALLOW_LEGACY_OUTPUT_DOWNLOAD, "legacy_missing_metadata") if ALLOW_LEGACY_OUTPUT_DOWNLOAD else (False, "missing_metadata")
    try:
        with open(meta_path, "rb") as fh:
            meta = orjson.loads(fh.read())
    except Exception:
        return (ALLOW_LEGACY_OUTPUT_DOWNLOAD, "metadata_read_failed") if ALLOW_LEGACY_OUTPUT_DOWNLOAD else (False, "metadata_read_failed")
    owner = None
    if isinstance(meta, dict):
        owner = meta.get("principal") or (meta.get("context") or {}).get("principal")
    if owner and principal and owner == principal:
        return True, "principal_match"
    meta_session = meta.get("session_id") if isinstance(meta, dict) else None
    if meta_session and session_id and meta_session == session_id:
        return True, "session_match"
    return False, "ownership_mismatch"

def is_owner(sid, username):
    meta = session_manager.get_metadata(sid) or {}
    return safe_get(meta, 'username') == username

def create_session_metadata(sid, username=None):
    return session_manager.ensure_session(sid, username)

def _recover_stale_session(session_id: str, reason: str) -> bool:
    if not session_id:
        return False
    if safe_is_alive(session_id):
        return False
    last_active = session_manager.get_last_active(session_id) or 0
    age = time.time() - last_active
    if age < SESSION_STALE_RECOVERY_SEC:
        return False
    # Clear prompt state + queues to avoid stale prompts blocking UI.
    try:
        prompt.clear_prompt_session(session_id, delay=0)
    except Exception:
        pass
    session_manager.drop_prompt_queue(session_id)
    session_manager.pop_emitter(session_id)
    # Unlock + reset to idle for safe user recovery
    transition_session(
        session_id,
        SessionState.IDLE,
        locked=False,
        phase=PipelinePhase.PREPARE,
        emit=False,
        broadcast=False,
        extras={"stale_recovered": True, "stale_reason": reason},
    )
    return True

def cleanup_sessions():
    recovered = []
    # Attempt safe recovery for stale sessions (avoid blocking UX on reconnect)
    for sid in session_manager.list_active_session_ids():
        if _recover_stale_session(sid, reason="cleanup"):
            recovered.append(sid)

    expired = session_manager.expire_sessions(
        SESSION_TIMEOUT,
        require_unlocked=True,
        require_no_thread=True,
        grace_period=SESSION_EXPIRE_GRACE_SEC,
    )
    for sid in expired:
        try:
            log_path = os.path.join(LOG_DIR, f"sess_{sid}.ndjson")
            if os.path.exists(log_path):
                os.remove(log_path)
        except Exception:
            pass
        _cleanup_quick_copy_dir(sid)
        last_contest_options.pop(sid, None)
        session_manager.unbind_fingerprints_for_session(sid)
    if expired:
        emit('session_expired', {'expired_sessions': expired}, broadcast=True)
        broadcast_sessions()
    if recovered:
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


def get_request_principal():
    """Return (principal, source, cert_metadata) preferring client cert, then SSO OID."""
    principal, source, cert_meta = extract_client_principal(request.headers)
    if principal:
        return principal, source, cert_meta

    # Dev-only bypass for localhost when explicitly enabled
    host = (request.host or "").lower()
    is_local = _is_local_host(host)
    if ALLOW_DEV_NO_PRINCIPAL and is_local:
        remote = request.remote_addr or "local"
        return f"dev:{remote}", "dev_bypass", None
    return None, None, None


def _is_local_host(host: str) -> bool:
    if not host:
        return False
    return (
        host in {"localhost", "127.0.0.1", "::1", "[::1]"}
        or host.startswith("localhost:")
        or host.startswith("127.0.0.1:")
        or host.startswith("[::1]:")
    )


def _is_azure_environment() -> bool:
    return any(os.environ.get(key) for key in (
        "WEBSITE_INSTANCE_ID",
        "WEBSITE_SITE_NAME",
        "APPSETTING_WEBSITE_SITE_NAME",
        "WEBSITE_HOSTNAME",
        "AZURE_HTTP_USER_AGENT",
    ))


def _get_dev_isolation_bypass_ips() -> set[str]:
    if not DEV_ISOLATION_BYPASS_IPS_RAW:
        return {"127.0.0.1", "::1"}
    return {ip.strip() for ip in DEV_ISOLATION_BYPASS_IPS_RAW.split(",") if ip.strip()}


def _is_dev_isolation_bypass_request() -> bool:
    if not (ALLOW_DEV_NO_PRINCIPAL and DEV_ISOLATION_BYPASS_ENABLED):
        return False
    if _is_azure_environment():
        return False
    host = (request.host or "").lower()
    if not _is_local_host(host):
        return False
    remote = request.remote_addr or ""
    if not remote:
        return False
    allowed_ips = _get_dev_isolation_bypass_ips()
    return remote in allowed_ips


def _resolve_cert_session_id(principal: str | None) -> str | None:
    if not principal or not principal.startswith("cert:"):
        return None
    fingerprint = principal.split(":", 1)[1].strip()
    if not fingerprint:
        return None
    return f"cert_{fingerprint[:32]}"


def _derive_auth_context(principal: str | None, principal_source: str | None) -> dict:
    if principal and principal.startswith("cert:"):
        return {"auth_tier": "cert", "auth_trusted": True, "principal_source": principal_source}
    if principal and principal.startswith("sso:"):
        return {"auth_tier": "sso", "auth_trusted": True, "principal_source": principal_source}
    if principal and principal.startswith("dev:"):
        return {"auth_tier": "dev", "auth_trusted": True, "principal_source": principal_source}
    return {"auth_tier": "anon", "auth_trusted": False, "principal_source": principal_source}


def _apply_auth_context(session_id: str, principal: str | None, principal_source: str | None) -> None:
    if not session_id:
        return
    updates = _derive_auth_context(principal, principal_source)
    session_manager.update_metadata(session_id, **updates)


def _session_has_principal(session_id: str) -> bool:
    meta = session_manager.get_metadata(session_id)
    return bool(meta and meta.get("principal"))

def resolve_session_id(data=None, create_if_missing=True):
    def _log_resolution(decision: str, sid_val: str | None, reason: str | None = None):
        try:
            logger.info({
                "level": "INFO",
                "type": "auth",
                "message": f"Session resolution: {decision}",
                "session_id": sid_val,
                "reason": reason,
            })
        except Exception:
            pass

    def _truthy(val):
        if val is None:
            return False
        if isinstance(val, bool):
            return val
        try:
            s = str(val).strip().lower()
        except Exception:
            return False
        return s in {"1", "true", "yes", "y", "on", "reuse", "reuse_session"}

    try:
        socket_sid = safe_sid()
    except Exception:
        socket_sid = getattr(request, 'sid', None)
    if not isinstance(socket_sid, str) or not socket_sid:
        return None
    principal, principal_source, _ = get_request_principal()

    # Determine whether automatic session reuse is permitted
    reuse_hint = False
    if isinstance(data, dict):
        reuse_hint = _truthy(data.get("reuse_session") or data.get("reuse"))
    try:
        arg_reuse = request.args.get("reuse_session")
        hdr_reuse = request.headers.get("X-Reuse-Session")
        reuse_hint = reuse_hint or _truthy(arg_reuse) or _truthy(hdr_reuse)
    except Exception:
        pass

    allow_reuse = ALLOW_AUTO_SESSION_REUSE or reuse_hint
    # For dev_bypass principals, default to no automatic reuse unless explicitly hinted
    if principal_source == "dev_bypass" and not reuse_hint:
        allow_reuse = False

    sid = None
    if isinstance(data, dict):
        sid = safe_get(data, 'session_id')
    if isinstance(sid, str) and sid:
        session_manager.bind_socket(socket_sid, sid)
        if principal:
            session_manager.set_principal(sid, principal, principal_source)
        _apply_auth_context(sid, principal, principal_source)
        _log_resolution("explicit_session_id", sid, "payload session_id")
        return sid

    # Prefer principal binding to ensure identity fidelity across reconnects
    if allow_reuse and principal:
        mapped_principal = session_manager.resolve_principal(principal)
        if isinstance(mapped_principal, str) and mapped_principal:
            session_manager.bind_socket(socket_sid, mapped_principal)
            session['logical_session_id'] = mapped_principal
            _apply_auth_context(mapped_principal, principal, principal_source)
            _log_resolution("reuse_principal", mapped_principal, principal_source)
            return mapped_principal

    if allow_reuse:
        mapped = session_manager.resolve_socket(socket_sid)
        if isinstance(mapped, str) and mapped:
            if not principal and _session_has_principal(mapped):
                _log_resolution("reuse_socket_blocked", mapped, "principal_required")
            else:
                if principal:
                    session_manager.set_principal(mapped, principal, principal_source)
                _apply_auth_context(mapped, principal, principal_source)
                _log_resolution("reuse_socket", mapped, "socket bind")
                return mapped

    if allow_reuse:
        cookie_sid = session.get('logical_session_id')
        if isinstance(cookie_sid, str) and cookie_sid:
            if not principal and _session_has_principal(cookie_sid):
                _log_resolution("reuse_cookie_blocked", cookie_sid, "principal_required")
            else:
                session_manager.bind_socket(socket_sid, cookie_sid)
                if principal:
                    session_manager.set_principal(cookie_sid, principal, principal_source)
                _apply_auth_context(cookie_sid, principal, principal_source)
                _log_resolution("reuse_cookie", cookie_sid, "logical_session_id cookie")
                return cookie_sid

    fingerprint = client_fingerprint() if ENABLE_FINGERPRINT_SESSION_RECOVERY else None
    if allow_reuse and ENABLE_FINGERPRINT_SESSION_RECOVERY and fingerprint:
        fp_sid = session_manager.resolve_fingerprint(fingerprint)
        if isinstance(fp_sid, str) and fp_sid:
            if not principal and _session_has_principal(fp_sid):
                _log_resolution("reuse_fingerprint_blocked", fp_sid, "principal_required")
            else:
                session_manager.bind_socket(socket_sid, fp_sid)
                session['logical_session_id'] = fp_sid
                if principal:
                    session_manager.set_principal(fp_sid, principal, principal_source)
                _apply_auth_context(fp_sid, principal, principal_source)
                _log_resolution("reuse_fingerprint", fp_sid, "fingerprint")
                return fp_sid

    if not create_if_missing:
        return None

    cert_session_id = _resolve_cert_session_id(principal) if CERT_SESSION_BINDING else None
    if cert_session_id:
        mapped_principal = session_manager.resolve_principal(principal) if principal else None
        if mapped_principal and mapped_principal != cert_session_id:
            cert_session_id = mapped_principal
        if ENABLE_FINGERPRINT_SESSION_RECOVERY and fingerprint:
            session_manager.bind_fingerprint(fingerprint, cert_session_id)
        session_manager.bind_socket(socket_sid, cert_session_id)
        session['logical_session_id'] = cert_session_id
        session_manager.ensure_session(cert_session_id)
        _ensure_quick_copy_dir(cert_session_id)
        session_manager.set_principal(cert_session_id, principal, principal_source)
        _apply_auth_context(cert_session_id, principal, principal_source)
        _log_resolution("new_cert_session", cert_session_id, "cert_binding")
        return cert_session_id

    new_sid = 'sess_' + secrets.token_urlsafe(16)
    if ENABLE_FINGERPRINT_SESSION_RECOVERY and fingerprint:
        session_manager.bind_fingerprint(fingerprint, new_sid)
    session_manager.bind_socket(socket_sid, new_sid)
    session['logical_session_id'] = new_sid
    session_manager.ensure_session(new_sid)
    _ensure_quick_copy_dir(new_sid)
    if principal:
        session_manager.set_principal(new_sid, principal, principal_source)
    _apply_auth_context(new_sid, principal, principal_source)
    _log_resolution("new_session", new_sid, "created")
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
        "auth": "auth",
        "isolation": "isolation",
        "security": "security",
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
        if isinstance(line, (bytes, bytearray)) and len(line) > MAX_SOCKET_LOG_BYTES:
            line = line[:MAX_SOCKET_LOG_BYTES]
        elif isinstance(line, str) and len(line.encode("utf-8", "ignore")) > MAX_SOCKET_LOG_BYTES:
            line = line[:5000] + "...(truncated)"
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
        msg_type = obj.get("type")
        if sid and msg_type in {"input", "status", "raw"} and len(msg) < 600:
            key = f"{msg_type}|{msg}"
            should_emit = session_manager.should_emit_message(
                sid,
                key,
                now=t_now,
                window=LOG_DEDUPE_WINDOW,
                max_entries=MAX_CACHE_PER_SESSION,
            )
            if not should_emit:
                return  # skip duplicate
        if sid and msg_type in {"auth", "isolation", "security"}:
            principal = obj.get("principal") or ""
            reason = obj.get("block_reason") or obj.get("reason") or obj.get("auth_block_reason") or ""
            key = f"{msg_type}|{msg}|{principal}|{reason}"
            should_emit = session_manager.should_emit_message(
                sid,
                key,
                now=t_now,
                window=SECURITY_LOG_DEDUPE_WINDOW,
                max_entries=MAX_CACHE_PER_SESSION,
            )
            if not should_emit:
                return

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

        # --- Special handling for contest_options: emit as dedicated event instead of parser_output ---
        if obj.get("type") == "contest_options" and sid:
            contest_payload = {
                "session_id": sid,
                "context": obj.get("context", {}),
                "total_count": obj.get("total_count", 0),
                "options": obj.get("options", [])
            }
            store_log(sid, obj)
            socketio.emit('contest_options', contest_payload, room=sid)
            session_manager.set_last_contest_options(sid, contest_payload)
            return

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

# Throttle redirect header diagnostics (per host+path)
_REDIRECT_HEADER_LOG_LAST: dict[str, float] = {}

@app.before_request
def redirect_to_https_www():
    """
    Enforce HTTPS and www subdomain for production domain.
    - Redirects http:// to https://
    - Redirects electionpulse.org to www.electionpulse.org
    """
    # Optional diagnostic logging for forwarded headers (guarded to avoid noise)
    log_forwarded = os.environ.get("LOG_REDIRECT_HEADERS", "").lower() in {"1", "true", "yes"}
    # Prefer forwarded host when behind a proxy/CDN (Azure Front Door/App Service)
    forwarded_host = request.headers.get("X-Forwarded-Host")
    raw_host = (forwarded_host or request.host or "").split(",")[0].strip().lower()
    # Normalize host: strip port, handle IPv6 literals
    if raw_host.startswith("[") and "]" in raw_host:
        host_only = raw_host[1:raw_host.index("]")]
    else:
        host_only = raw_host.split(":", 1)[0]

    if log_forwarded:
        forwarded_proto = request.headers.get("X-Forwarded-Proto")
        forwarded_port = request.headers.get("X-Forwarded-Port")
        log_triggered = request.path == "/robots.txt" or not forwarded_host or not forwarded_proto
        ttl_raw = os.environ.get("LOG_REDIRECT_HEADERS_TTL_SEC", "300")
        try:
            ttl_sec = max(0, int(ttl_raw))
        except ValueError:
            ttl_sec = 300
        host_key = host_only or raw_host or "unknown"
        log_key = f"{host_key}|{request.path}"
        now = time.time()
        last_ts = _REDIRECT_HEADER_LOG_LAST.get(log_key)
        should_log = log_triggered and (ttl_sec == 0 or last_ts is None or (now - last_ts) >= ttl_sec)
        if should_log:
            logger.info({
                "level": "INFO",
                "type": "status",
                "message": "[RedirectHeaders] Incoming request headers snapshot",
                "path": request.path,
                "host": request.host,
                "forwarded_host": forwarded_host,
                "forwarded_proto": forwarded_proto,
                "forwarded_port": forwarded_port,
                "session_id": None,
            })
            _REDIRECT_HEADER_LOG_LAST[log_key] = now

    # Skip redirects for local development (handle localhost with/without port, IPv4, IPv6)
    if (host_only in ('localhost', '127.0.0.1', '::1') or 
        raw_host.startswith('localhost:') or 
        raw_host.startswith('127.0.0.1:') or
        raw_host.startswith('[::1]:')):
        return None
    
    # Get the current scheme (check X-Forwarded-Proto for proxy setups like Azure)
    scheme = request.headers.get('X-Forwarded-Proto', request.scheme)
    
    # Production domain configuration
    PRODUCTION_APEX = 'electionpulse.org'
    PRODUCTION_WWW = 'www.electionpulse.org'
    
    # Check if we need to redirect to www or HTTPS
    if host_only == PRODUCTION_APEX:
        # Redirect apex domain to www with HTTPS
        target_url = f"https://{PRODUCTION_WWW}{request.full_path.rstrip('?')}"
        return redirect(target_url, code=301)
    elif host_only == PRODUCTION_WWW and scheme != 'https':
        # Force HTTPS for www subdomain
        target_url = f"https://{PRODUCTION_WWW}{request.full_path.rstrip('?')}"
        return redirect(target_url, code=301)
    
    return None

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
            # Allow jsDelivr for connect-src as well so browser may fetch source-maps
            # (source-map requests are benign but can be blocked by strict CSP).
            connect_extra.append("https://cdn.jsdelivr.net")
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

    if _is_request_secure():
        response.headers.setdefault(
            "Strict-Transport-Security",
            "max-age=63072000; includeSubDomains; preload"
        )

    # Final charset / textual normalization (single point of truth)
    response = ensure_utf8(response)
    return response


@app.errorhandler(Exception)
def _handle_global_exception(e):
    """Global exception handler: log and return JSON for API requests.

    For requests to `/api/...` or requests that accept JSON, return a JSON
    error payload. For other requests, return a plain text 500 response.
    """
    try:
        # Determine HTTP status and message
        if isinstance(e, HTTPException):
            code = getattr(e, 'code', 500) or 500
            description = getattr(e, 'description', str(e))
        else:
            code = 500
            description = str(e)

        # Log structured error
        try:
            logger.error({
                "level": "ERROR",
                "type": "exception",
                "message": f"Unhandled exception: {description}",
                "path": getattr(request, 'path', None),
                "method": getattr(request, 'method', None),
                "session_id": None,
            })
        except Exception:
            pass

        accept = (request.headers.get('Accept') or '').lower()
        wants_json = (request.path or '').startswith('/api/') or 'application/json' in accept or request.is_json

        if wants_json:
            payload = {"error": description}
            return jsonify(payload), code

        # Non-API requests: return safe plain-text message
        safe_msg = "Internal Server Error" if code == 500 else description
        return Response(safe_msg, status=code, mimetype='text/plain')
    except Exception:
        # If the error handler itself fails, avoid raising; return minimal JSON
        try:
            logger.error({"level": "ERROR", "type": "exception", "message": "Exception in error handler"})
        except Exception:
            pass
        return jsonify({"error": "internal"}), 500

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
    urls_out = []
    with open(URL_LIST_FILE, "r", encoding="utf-8") as f:
        for raw in f:
            s = safe_strip(raw)
            if not s or s.startswith('#'):
                continue
            u, lbl = extract_url_and_label(s)
            if u:
                urls_out.append(u)
            else:
                urls_out.append(s)
    return urls_out

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

_SAFE_FILTER_PATTERN = re.compile(r"^[\w\s\-\.,&()/']+$", re.UNICODE)

def _validate_filter_value(name: str, value: str | None, max_len: int = 120) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    cleaned = safe_strip(value)
    if not cleaned:
        return None
    if len(cleaned) > max_len:
        raise ValueError(f"{name} too long")
    lowered = cleaned.lower()
    if any(token in lowered for token in (";", "--", "/*", "*/")):
        raise ValueError(f"{name} contains invalid characters")
    if not _SAFE_FILTER_PATTERN.fullmatch(cleaned):
        raise ValueError(f"{name} contains invalid characters")
    return cleaned

def log_db_monitor_event(event: dict) -> None:
    payload = dict(event or {})
    payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    try:
        with open(DB_MONITOR_FILE, "ab") as f:
            f.write(orjson.dumps(payload) + b"\n")
    except Exception:
        pass

# Routes
@app.route("/")
def index() -> str:
    return render_template("index.html")

@app.route("/api/urls", methods=["GET", "POST"])
@_rate_limit("30/minute")
def api_urls():
    urls_file = str(URL_LIST_FILE)
    try:
        if request.method == "GET":
            if not os.path.exists(urls_file):
                return jsonify({"urls": []})
            urls = []
            with open(urls_file, "r", encoding="utf-8") as f:
                for raw in f:
                    s = safe_strip(raw)
                    if not s or s.startswith('#'):
                        continue
                    u, lbl = extract_url_and_label(s)
                    urls.append(u or s)
            return jsonify({"urls": urls})
        elif request.method == "POST":
            guard_ok, guard_reason = _guarded_ingestion_allowed("api_urls")
            if not guard_ok:
                logger.warning({
                    "level": "WARNING",
                    "type": "security",
                    "message": f"URL ingestion blocked by guarded gate: {guard_reason}",
                    "session_id": None,
                })
                return jsonify({"success": False, "error": "Guarded ingestion key required."}), 403
            data = request.get_json() or {}
            raw_url = safe_strip(safe_get(data, "url", ""))
            if not raw_url:
                return jsonify({"success": False, "error": "URL required."}), 400
            url, lbl = extract_url_and_label(raw_url)
            if not url:
                return jsonify({"success": False, "error": "No valid http(s) URL found."}), 400

            parsed = urlparse(url)
            host = (parsed.hostname or "").lower()
            session_id = safe_strip(safe_get(data, "session_id"))
            allowed, reason = safe_validate_external_url(
                url,
                allowlist_suffixes=URL_ALLOWLIST_SUFFIXES,
                allowlist_hosts=URL_ALLOWLIST_HOSTS,
                enforce_allowlist=URL_ENFORCE_ALLOWLIST,
                block_private_ips=URL_BLOCK_PRIVATE_IPS,
            )
            if not allowed:
                log_flagged_url({
                    "url": url,
                    "reason": reason,
                    "session_id": session_id,
                })
                return jsonify({"success": False, "error": f"URL blocked: {reason}"}), 400
            suspicious_tokens = (
                "dropbox.com",
                "drive.google",
                "docs.google",
                "googleusercontent.com",
                "storage.googleapis",
                "amazonaws.com",
                "s3.amazonaws.com",
                "digitaloceanspaces.com",
                "box.com",
                "onedrive",
                "sharepoint",
                "github.com",
                "raw.githubusercontent",
                "gitlab",
                "pastebin",
                "notion.so",
                "cloudfront.net",
            )
            if ALLOW_GOOGLE_DOCS:
                suspicious_tokens = tuple(
                    tok for tok in suspicious_tokens
                    if tok not in {"drive.google", "docs.google", "googleusercontent.com"}
                )
            if parsed.scheme not in {"http", "https"} or not host:
                log_flagged_url({
                    "url": url,
                    "reason": "invalid_url",
                    "session_id": session_id,
                })
                return jsonify({"success": False, "error": "Only http/https URLs with a host are accepted."}), 400

            if any(tok in host for tok in suspicious_tokens):
                log_flagged_url({
                    "url": url,
                    "reason": "suspicious_host",
                    "host": host,
                    "session_id": session_id,
                })
                return jsonify({"success": False, "error": "Host requires manual review; URL logged for safety."}), 400

            with open(urls_file, "a", encoding="utf-8") as f:
                f.write(url + "\n")
            return jsonify({"success": True})
    except Exception as exc:
        logger.error({"level": "ERROR", "type": "api", "message": f"api_urls GET/POST failed: {exc}", "session_id": None})
        return jsonify({"urls": [], "error": "internal"}), 500


def _load_output_metadata(meta_path: str) -> dict:
    try:
        if meta_path and os.path.exists(meta_path):
            with open(meta_path, "rb") as fh:
                data = orjson.loads(fh.read())
                return data if isinstance(data, dict) else {}
    except Exception:
        return {}
    return {}


def _build_output_lookup_match(url: str, entry: dict) -> dict | None:
    if not isinstance(entry, dict):
        return None
    output_file = entry.get("output_file")
    metadata_path = entry.get("metadata_path")
    output_dir = entry.get("output_dir")

    if not output_dir and isinstance(output_file, str):
        output_dir = os.path.dirname(output_file)
    if not metadata_path and output_dir:
        candidate = os.path.join(output_dir, "results.metadata.json")
        if os.path.exists(candidate):
            metadata_path = candidate

    meta = _load_output_metadata(metadata_path) if metadata_path else {}
    if not output_dir and isinstance(meta.get("output_dir"), str):
        output_dir = meta.get("output_dir")

    output_folder = os.path.basename(output_dir) if output_dir else None
    if not output_folder and isinstance(meta.get("output_base_name"), str):
        output_folder = meta.get("output_base_name")

    if not output_folder:
        return None

    return {
        "url": url,
        "output_folder": output_folder,
        "output_file": output_file or meta.get("csv_path"),
        "metadata_path": metadata_path,
        "created_at": meta.get("created_at"),
        "contest": meta.get("contest") or entry.get("contest"),
        "state": meta.get("state") or entry.get("state"),
        "county": meta.get("county") or entry.get("county"),
        "handler": meta.get("handler") or entry.get("handler"),
        "source_url": meta.get("source_url") or entry.get("source_url"),
    }


@app.route("/api/outputs/lookup", methods=["GET"])
def api_outputs_lookup():
    raw_url = safe_strip(request.args.get("url", ""))
    if not raw_url:
        return jsonify({"matches": [], "error": "URL required."}), 400
    url, _ = extract_url_and_label(raw_url)
    url = url or raw_url

    matches: list[dict] = []
    processed = load_processed_urls()
    entry = processed.get(url)
    if isinstance(entry, dict):
        match = _build_output_lookup_match(url, entry)
        if match:
            matches.append(match)

    if not matches:
        try:
            output_root = os.path.abspath(str(OUTPUT_DIR))
            candidates = []
            with os.scandir(output_root) as it:
                for de in it:
                    if not de.is_dir(follow_symlinks=False):
                        continue
                    try:
                        candidates.append((de.path, de.stat(follow_symlinks=False).st_mtime))
                    except Exception:
                        candidates.append((de.path, 0))
            candidates.sort(key=lambda item: item[1], reverse=True)
            for path, _ in candidates[:200]:
                meta_path = os.path.join(path, "results.metadata.json")
                if not os.path.exists(meta_path):
                    continue
                meta = _load_output_metadata(meta_path)
                source_url = meta.get("source_url")
                if source_url and source_url == url:
                    match = _build_output_lookup_match(url, {
                        "metadata_path": meta_path,
                        "output_dir": path,
                        "output_file": meta.get("csv_path"),
                        "contest": meta.get("contest"),
                        "state": meta.get("state"),
                        "county": meta.get("county"),
                        "handler": meta.get("handler"),
                        "source_url": source_url,
                    })
                    if match:
                        matches.append(match)
        except Exception as exc:
            logger.warning({
                "level": "WARNING",
                "type": "output",
                "message": f"Output lookup fallback scan failed: {exc}",
                "session_id": None,
            })

    return jsonify({"matches": matches, "url": url})


def _get_warehouse_columns(engine) -> set[str]:
    try:
        inspector = inspect(engine)
        cols = inspector.get_columns("warehouse_election_results")
    except Exception:
        return set()
    return {col.get("name") for col in cols if col.get("name")}


@app.route("/api/warehouse/match", methods=["GET"])
def api_warehouse_match():
    raw_url = safe_strip(request.args.get("url", ""))
    if not raw_url:
        return jsonify({"matches": [], "error": "URL required."}), 400
    url, _ = extract_url_and_label(raw_url)
    url = url or raw_url

    ensure_db_tables()
    engine = get_engine()
    columns = _get_warehouse_columns(engine)
    if "source_url" not in columns:
        return jsonify({"matches": [], "url": url, "error": "source_url column missing"})

    select_cols = []
    group_cols = []
    for col in ("state", "county", "contest"):
        if col in columns:
            select_cols.append(col)
            group_cols.append(col)

    handler_col = None
    if "handler_name" in columns:
        handler_col = "handler_name"
    elif "handler" in columns:
        handler_col = "handler"
    if handler_col:
        select_cols.append(f"{handler_col} AS handler")
        group_cols.append(handler_col)

    select_cols.append("source_url")
    group_cols.append("source_url")

    aggregates = ["COUNT(*) AS row_count"]
    if "candidate" in columns:
        aggregates.append("COUNT(DISTINCT candidate) AS candidate_count")
    if "precinct" in columns:
        aggregates.append("COUNT(DISTINCT precinct) AS precinct_count")
    if "election_date" in columns:
        aggregates.append("MAX(election_date) AS latest_election_date")

    select_sql = ", ".join(select_cols + aggregates)
    group_sql = ", ".join(group_cols)
    query = f"""
        SELECT {select_sql}
        FROM warehouse_election_results
        WHERE source_url = :url
        GROUP BY {group_sql}
        ORDER BY row_count DESC
        LIMIT 25
    """

    try:
        with engine.connect() as conn:
            rows = conn.execute(text(query), {"url": url}).mappings().all()
        matches = [dict(row) for row in rows]
        qa_status = None
        try:
            inspector = inspect(engine)
            if inspector.has_table("verified_datasets", schema="verified_data"):
                with engine.connect() as conn:
                    qa_row = conn.execute(
                        text(
                            """
                            SELECT dataset_id, dl_status, extracted_at, trust_score, extraction_confidence
                            FROM verified_data.verified_datasets
                            WHERE source_url = :url
                            ORDER BY extracted_at DESC
                            LIMIT 1
                            """
                        ),
                        {"url": url},
                    ).mappings().first()
                if qa_row:
                    qa_status = dict(qa_row)
        except Exception:
            qa_status = None
        return jsonify({"matches": matches, "url": url, "qa_status": qa_status})
    except Exception as exc:
        logger.warning({
            "level": "WARNING",
            "type": "db",
            "message": f"Warehouse match query failed: {exc}",
            "session_id": None,
        })
        return jsonify({"matches": [], "url": url, "error": "Warehouse match query failed"}), 500


@app.route("/api/warehouse/export", methods=["GET"])
def api_warehouse_export():
    raw_url = safe_strip(request.args.get("url", ""))
    if not raw_url:
        return jsonify({"error": "URL required."}), 400
    url, _ = extract_url_and_label(raw_url)
    url = url or raw_url

    ensure_db_tables()
    engine = get_engine()
    columns = _get_warehouse_columns(engine)
    if "source_url" not in columns:
        return jsonify({"error": "source_url column missing"}), 400

    limit = request.args.get("limit", type=int) or 5000
    limit = max(1, min(MAX_CSV_ROWS, limit))

    try:
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    """
                    SELECT *
                    FROM warehouse_election_results
                    WHERE source_url = :url
                    LIMIT :limit
                    """
                ),
                {"url": url, "limit": limit},
            )
            rows = result.fetchall()
            cols = list(result.keys())
    except Exception as exc:
        logger.warning({
            "level": "WARNING",
            "type": "db",
            "message": f"Warehouse export query failed: {exc}",
            "session_id": None,
        })
        return jsonify({"error": "Warehouse export failed"}), 500

    if not rows:
        return jsonify({"error": "No warehouse rows found for URL."}), 404

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(cols)
    for row in rows:
        writer.writerow(list(row))

    response = Response(output.getvalue(), mimetype="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=warehouse_export.csv"
    return response

@app.route("/api/warehouse/coverage", methods=["GET"])
def api_warehouse_coverage():
    """Return coverage summary: all state/county combinations in warehouse and those missing DL1/DL2"""
    ensure_db_tables()
    engine = get_engine()
    columns = _get_warehouse_columns(engine)
    
    # Ensure state and county columns exist
    if "state" not in columns or "county" not in columns:
        return jsonify({
            "covered": [],
            "missing": [],
            "all_states": [],
            "all_counties": {},
            "total_rows": 0,
            "error": "state or county column missing"
        }), 400

    try:
        with engine.connect() as conn:
            # Get all state/county combinations in warehouse
            query = """
                SELECT DISTINCT state, county, COUNT(*) as row_count
                FROM warehouse_election_results
                WHERE state IS NOT NULL AND county IS NOT NULL
                GROUP BY state, county
            """
            result = conn.execute(text(query))
            covered_rows = result.mappings().all()
            covered = [{"state": row["state"], "county": row["county"], "row_count": row["row_count"]} for row in covered_rows]
            
            # Get all unique states and counties
            state_query = """
                SELECT DISTINCT state FROM warehouse_election_results 
                WHERE state IS NOT NULL
                ORDER BY state
            """
            all_states_result = conn.execute(text(state_query))
            all_states = [row[0] for row in all_states_result]
            
            # Get total row count for health check
            count_query = "SELECT COUNT(*) as cnt FROM warehouse_election_results"
            count_result = conn.execute(text(count_query))
            total_rows = count_result.mappings().first()["cnt"]
        
        # Build all_counties map: state -> list of counties
        all_counties = {}
        for state in all_states:
            with engine.connect() as conn:
                county_query = """
                    SELECT DISTINCT county FROM warehouse_election_results
                    WHERE state = :state AND county IS NOT NULL
                    ORDER BY county
                """
                result = conn.execute(text(county_query), {"state": state})
                all_counties[state] = [row[0] for row in result]
        
        return jsonify({
            "covered": covered,
            "all_states": all_states,
            "all_counties": all_counties,
            "total_rows": total_rows,
            "coverage_summary": {
                "total_states": len(all_states),
                "total_state_county_pairs": len(covered),
                "warehouse_healthy": total_rows > 0
            }
        })
    except Exception as exc:
        logger.warning({
            "level": "WARNING",
            "type": "db",
            "message": f"Warehouse coverage query failed: {exc}",
            "session_id": None,
        })
        return jsonify({"error": "Warehouse coverage query failed", "covered": [], "all_states": [], "all_counties": {}}), 500

@app.route("/data_framework", methods=["GET"])
def data_framework():
    return render_template("data_framework.html", data_api_url=DATA_API_URL)


def _collect_data_framework_scaffold(limit: int = 100) -> dict:
    records = []
    fields = [
        "state",
        "county",
        "contest",
        "handler",
        "row_count",
        "column_count",
        "extraction_confidence",
        "timestamp",
        "source_url",
    ]
    output_dir = Path(OUTPUT_DIR)
    if not output_dir.exists():
        return {"fields": fields, "records": [], "generated_at": datetime.now(timezone.utc).isoformat()}

    for folder in sorted(output_dir.iterdir(), reverse=True):
        if not folder.is_dir():
            continue
        metadata_path = folder / "results.metadata.json"
        if not metadata_path.exists():
            metadata_path = folder / "metadata.json"
        if not metadata_path.exists():
            continue
        try:
            with open(metadata_path, "rb") as fh:
                meta = orjson.loads(fh.read())
        except Exception:
            continue
        if not isinstance(meta, dict):
            continue
        quality = meta.get("quality_metrics") if isinstance(meta.get("quality_metrics"), dict) else {}
        record = {
            "state": meta.get("state"),
            "county": meta.get("county"),
            "contest": meta.get("contest"),
            "handler": meta.get("handler"),
            "row_count": meta.get("row_count"),
            "column_count": meta.get("column_count"),
            "extraction_confidence": quality.get("extraction_confidence"),
            "timestamp": meta.get("timestamp") or folder.name.split("__")[-1],
            "source_url": meta.get("source_url") or (meta.get("context") or {}).get("url"),
        }
        records.append(record)
        if len(records) >= limit:
            break
    return {
        "fields": fields,
        "records": records,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _extract_year_from_text(value: str | None) -> str | None:
    if not value:
        return None
    match = re.search(r"(20\d{2})", str(value))
    if not match:
        return None
    return match.group(1)


def _collect_data_framework_curated(limit: int = 80) -> dict:
    scaffold = _collect_data_framework_scaffold(limit=limit)
    items = []
    for record in scaffold.get("records", []):
        state = record.get("state") or ""
        county = record.get("county") or ""
        contest = record.get("contest") or ""
        updated_at = record.get("timestamp") or ""
        year = _extract_year_from_text(updated_at) or _extract_year_from_text(contest)
        title_parts = [part for part in [contest, state, county] if part]
        title = " • ".join(title_parts) if title_parts else "Curated dataset"
        item_id = "::".join([state or "NA", county or "NA", contest or "NA", updated_at or "NA"])
        items.append({
            "id": item_id,
            "title": title,
            "state": state,
            "county": county,
            "contest": contest,
            "year": year,
            "row_count": record.get("row_count"),
            "column_count": record.get("column_count"),
            "extraction_confidence": record.get("extraction_confidence"),
            "updated_at": updated_at,
            "source_url": record.get("source_url"),
        })
    return {
        "items": items,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _resolve_preview_filters() -> tuple[str | None, str | None, str | None, int | None]:
    state = request.args.get("state")
    county = request.args.get("county")
    contest = request.args.get("contest")
    year_str = request.args.get("year")
    try:
        state = _validate_filter_value("state", state, max_len=64)
        county = _validate_filter_value("county", county, max_len=64)
        contest = _validate_filter_value("contest", contest, max_len=140)
    except ValueError as exc:
        raise ValueError(str(exc))
    year_val = None
    if year_str:
        try:
            year_val = int(year_str)
        except ValueError:
            raise ValueError("year must be an integer")
    return state, county, contest, year_val


def _select_preview_context(conn, state: str | None, county: str | None, contest: str | None, year_val: int | None) -> dict:
    where = []
    params: dict[str, object] = {}
    if state:
        where.append("state = :state")
        params["state"] = state
    if county:
        where.append("county = :county")
        params["county"] = county
    if contest:
        where.append("contest ILIKE :contest")
        params["contest"] = f"%{contest}%"
    if year_val:
        where.append("EXTRACT(YEAR FROM election_date) = :year")
        params["year"] = year_val
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    row = conn.execute(text(
        f"""
        SELECT state,
               county,
               contest,
               EXTRACT(YEAR FROM election_date) AS year
        FROM warehouse_election_results
        {where_sql}
        ORDER BY random()
        LIMIT 1
        """
    ), params).mappings().first()
    return dict(row) if row else {}


def _fetch_preview_rows(conn, state: str | None, county: str | None, contest: str | None, year_val: int | None, limit: int) -> list[dict]:
    columns = [
        "state",
        "county",
        "contest",
        "candidate",
        "party",
        "votes",
        "precinct",
        "election_date",
    ]
    where = []
    params: dict[str, object] = {"limit": limit}
    if state:
        where.append("state = :state")
        params["state"] = state
    if county:
        where.append("county = :county")
        params["county"] = county
    if contest:
        where.append("contest ILIKE :contest")
        params["contest"] = f"%{contest}%"
    if year_val:
        where.append("EXTRACT(YEAR FROM election_date) = :year")
        params["year"] = year_val
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    cols_sql = ", ".join(columns)
    rows = conn.execute(text(
        f"""
        SELECT {cols_sql}
        FROM warehouse_election_results
        {where_sql}
        ORDER BY election_date DESC NULLS LAST, contest ASC
        LIMIT :limit
        """
    ), params).mappings().all()
    return [dict(row) for row in rows]


@app.route("/api/data_framework/preview", methods=["GET"])
def api_data_framework_preview():
    principal, _, _ = get_request_principal()
    if not principal and not ALLOW_DEV_NO_PRINCIPAL:
        return jsonify({"error": "Unauthorized"}), 403
    mode = (request.args.get("mode") or "idle").lower()
    if mode not in ("idle", "active"):
        return jsonify({"error": "mode must be 'idle' or 'active'"}), 400
    try:
        state, county, contest, year_val = _resolve_preview_filters()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    try:
        limit = int(request.args.get("limit") or 120)
    except Exception:
        limit = 120
    limit = max(10, min(500, limit))

    ensure_db_tables()
    now = datetime.now(timezone.utc)
    idle_ttl = timedelta(minutes=10)
    active_ttl = timedelta(hours=2)
    use_active = mode == "active" or any([state, county, contest, year_val])
    expires_at = now + (active_ttl if use_active else idle_ttl)
    session_id = resolve_session_id({}, create_if_missing=True) or "no_session"

    session = SessionLocal()
    try:
        session.query(DataFrameworkPreviewCache).filter(DataFrameworkPreviewCache.expires_at < now).delete(synchronize_session=False)
        session.commit()

        query = session.query(DataFrameworkPreviewCache).filter(
            DataFrameworkPreviewCache.session_id == session_id,
            DataFrameworkPreviewCache.mode == mode,
            DataFrameworkPreviewCache.expires_at > now,
        )
        if state:
            query = query.filter(DataFrameworkPreviewCache.state == state)
        if county:
            query = query.filter(DataFrameworkPreviewCache.county == county)
        if contest:
            query = query.filter(DataFrameworkPreviewCache.contest == contest)
        if year_val:
            query = query.filter(DataFrameworkPreviewCache.year == year_val)
        cached = query.order_by(DataFrameworkPreviewCache.created_at.desc()).first()
        if cached and isinstance(cached.payload, dict):
            cached.last_accessed = now
            session.commit()
            payload = cached.payload.copy()
            payload["cache_id"] = str(cached.id)
            payload["expires_at"] = cached.expires_at.isoformat() if cached.expires_at else None
            payload.setdefault("mode", mode)
            return jsonify(payload)
    finally:
        session.close()

    engine = get_engine()
    preview_context = {}
    rows: list[dict] = []
    try:
        with engine.connect() as conn:
            preview_context = _select_preview_context(conn, state, county, contest, year_val)
            context_state = preview_context.get("state") or state
            context_county = preview_context.get("county") or county
            context_contest = preview_context.get("contest") or contest
            context_year = preview_context.get("year") or year_val
            rows = _fetch_preview_rows(conn, context_state, context_county, context_contest, context_year, limit)
    except Exception as exc:
        logger.warning({
            "level": "WARNING",
            "type": "db",
            "message": f"Preview query failed: {exc}",
            "session_id": session_id,
        })
        rows = []

    schema = list(rows[0].keys()) if rows else []
    payload = {
        "rows": rows,
        "schema": schema,
        "meta": {
            "state": preview_context.get("state") or state,
            "county": preview_context.get("county") or county,
            "contest": preview_context.get("contest") or contest,
            "year": preview_context.get("year") or year_val,
        },
        "generated_at": now.isoformat(),
        "mode": mode,
    }

    session = SessionLocal()
    try:
        cache_row = DataFrameworkPreviewCache(
            session_id=session_id,
            mode=mode,
            state=payload["meta"].get("state"),
            county=payload["meta"].get("county"),
            contest=payload["meta"].get("contest"),
            year=payload["meta"].get("year") if isinstance(payload["meta"].get("year"), int) else None,
            payload=payload,
            expires_at=expires_at,
        )
        session.add(cache_row)
        session.commit()
        payload["cache_id"] = str(cache_row.id)
        payload["expires_at"] = expires_at.isoformat()
    except Exception as exc:
        session.rollback()
        logger.warning({
            "level": "WARNING",
            "type": "db",
            "message": f"Preview cache save failed: {exc}",
            "session_id": session_id,
        })
    finally:
        session.close()

    return jsonify(payload)


@app.route("/api/data_framework/scaffold", methods=["GET"])
def api_data_framework_scaffold():
    principal, _, _ = get_request_principal()
    if not principal and not ALLOW_DEV_NO_PRINCIPAL:
        return jsonify({"error": "Unauthorized"}), 403
    try:
        limit = int(request.args.get("limit") or 100)
        limit = max(1, min(500, limit))
    except Exception:
        limit = 100
    payload = _collect_data_framework_scaffold(limit=limit)
    return jsonify(payload)


@app.route("/api/data_framework/scaffold.csv", methods=["GET"])
def api_data_framework_scaffold_csv():
    principal, _, _ = get_request_principal()
    if not principal and not ALLOW_DEV_NO_PRINCIPAL:
        return Response("Unauthorized", status=403, mimetype="text/plain")
    try:
        limit = int(request.args.get("limit") or 100)
        limit = max(1, min(500, limit))
    except Exception:
        limit = 100
    scaffold = _collect_data_framework_scaffold(limit=limit)
    fields = scaffold.get("fields", [])
    records = scaffold.get("records", [])
    output = []
    output.append(fields)
    for row in records:
        output.append([row.get(field, "") for field in fields])
    resp = Response(mimetype="text/csv; charset=utf-8")
    resp.headers["Content-Disposition"] = "attachment; filename=data_framework_scaffold.csv"
    writer = csv.writer(resp.stream)
    for line in output:
        writer.writerow(line)
    return resp


@app.route("/api/data_framework/curated", methods=["GET"])
def api_data_framework_curated():
    principal, _, _ = get_request_principal()
    if not principal and not ALLOW_DEV_NO_PRINCIPAL:
        return jsonify({"error": "Unauthorized"}), 403
    try:
        limit = int(request.args.get("limit") or 80)
        limit = max(1, min(200, limit))
    except Exception:
        limit = 80
    payload = _collect_data_framework_curated(limit=limit)
    return jsonify(payload)


@app.route("/api/data_framework/warehouse_status", methods=["GET"])
def api_data_framework_warehouse_status():
    principal, _, _ = get_request_principal()
    if not principal and not ALLOW_DEV_NO_PRINCIPAL:
        return jsonify({"error": "Unauthorized"}), 403

    ensure_db_tables()
    engine = get_engine()
    try:
        with engine.connect() as conn:
            exists = conn.execute(text("SELECT to_regclass('workflow.contests')")).scalar()
            if not exists:
                return jsonify({"error": "workflow.contests table not found"}), 503

            summary = conn.execute(text(
                """
                WITH expected AS (
                    SELECT
                        LOWER(TRIM(state)) AS state,
                        LOWER(COALESCE(TRIM(county), '')) AS county,
                        LOWER(TRIM(race)) AS contest,
                        year,
                        priority,
                        status
                    FROM workflow.contests
                    WHERE state IS NOT NULL AND race IS NOT NULL AND year IS NOT NULL
                ),
                warehouse AS (
                    SELECT
                        LOWER(TRIM(state)) AS state,
                        LOWER(COALESCE(TRIM(county), '')) AS county,
                        LOWER(TRIM(contest)) AS contest,
                        EXTRACT(YEAR FROM election_date)::int AS year,
                        COUNT(*) AS rows
                    FROM warehouse_election_results
                    GROUP BY 1,2,3,4
                ),
                missing AS (
                    SELECT e.*
                    FROM expected e
                    LEFT JOIN warehouse w
                      ON e.state = w.state
                     AND e.county = w.county
                     AND e.contest = w.contest
                     AND e.year = w.year
                    WHERE COALESCE(w.rows, 0) = 0
                )
                SELECT
                    (SELECT COUNT(*) FROM expected) AS expected_total,
                    (SELECT COUNT(*) FROM missing) AS missing_total
                """
            )).mappings().first()

            by_priority = conn.execute(text(
                """
                WITH expected AS (
                    SELECT
                        LOWER(TRIM(state)) AS state,
                        LOWER(COALESCE(TRIM(county), '')) AS county,
                        LOWER(TRIM(race)) AS contest,
                        year,
                        priority,
                        status
                    FROM workflow.contests
                    WHERE state IS NOT NULL AND race IS NOT NULL AND year IS NOT NULL
                ),
                warehouse AS (
                    SELECT
                        LOWER(TRIM(state)) AS state,
                        LOWER(COALESCE(TRIM(county), '')) AS county,
                        LOWER(TRIM(contest)) AS contest,
                        EXTRACT(YEAR FROM election_date)::int AS year,
                        COUNT(*) AS rows
                    FROM warehouse_election_results
                    GROUP BY 1,2,3,4
                ),
                missing AS (
                    SELECT e.*
                    FROM expected e
                    LEFT JOIN warehouse w
                      ON e.state = w.state
                     AND e.county = w.county
                     AND e.contest = w.contest
                     AND e.year = w.year
                    WHERE COALESCE(w.rows, 0) = 0
                )
                SELECT priority,
                       COUNT(*) AS missing,
                       (SELECT COUNT(*) FROM expected e2 WHERE e2.priority = missing.priority) AS expected
                FROM missing
                GROUP BY priority
                ORDER BY missing DESC
                """
            )).mappings().all()

            by_status = conn.execute(text(
                """
                WITH expected AS (
                    SELECT
                        LOWER(TRIM(state)) AS state,
                        LOWER(COALESCE(TRIM(county), '')) AS county,
                        LOWER(TRIM(race)) AS contest,
                        year,
                        priority,
                        status
                    FROM workflow.contests
                    WHERE state IS NOT NULL AND race IS NOT NULL AND year IS NOT NULL
                ),
                warehouse AS (
                    SELECT
                        LOWER(TRIM(state)) AS state,
                        LOWER(COALESCE(TRIM(county), '')) AS county,
                        LOWER(TRIM(contest)) AS contest,
                        EXTRACT(YEAR FROM election_date)::int AS year,
                        COUNT(*) AS rows
                    FROM warehouse_election_results
                    GROUP BY 1,2,3,4
                ),
                missing AS (
                    SELECT e.*
                    FROM expected e
                    LEFT JOIN warehouse w
                      ON e.state = w.state
                     AND e.county = w.county
                     AND e.contest = w.contest
                     AND e.year = w.year
                    WHERE COALESCE(w.rows, 0) = 0
                )
                SELECT status,
                       COUNT(*) AS missing,
                       (SELECT COUNT(*) FROM expected e2 WHERE e2.status = missing.status) AS expected
                FROM missing
                GROUP BY status
                ORDER BY missing DESC
                """
            )).mappings().all()

            sample = conn.execute(text(
                """
                WITH expected AS (
                    SELECT
                        state,
                        county,
                        race,
                        year,
                        priority,
                        status
                    FROM workflow.contests
                    WHERE state IS NOT NULL AND race IS NOT NULL AND year IS NOT NULL
                ),
                warehouse AS (
                    SELECT
                        LOWER(TRIM(state)) AS state,
                        LOWER(COALESCE(TRIM(county), '')) AS county,
                        LOWER(TRIM(contest)) AS contest,
                        EXTRACT(YEAR FROM election_date)::int AS year,
                        COUNT(*) AS rows
                    FROM warehouse_election_results
                    GROUP BY 1,2,3,4
                )
                SELECT e.state,
                       e.county,
                       e.race AS contest,
                       e.year,
                       e.priority,
                       e.status
                FROM expected e
                LEFT JOIN warehouse w
                  ON LOWER(TRIM(e.state)) = w.state
                 AND LOWER(COALESCE(TRIM(e.county), '')) = w.county
                 AND LOWER(TRIM(e.race)) = w.contest
                 AND e.year = w.year
                WHERE COALESCE(w.rows, 0) = 0
                ORDER BY e.priority NULLS LAST, e.year DESC
                LIMIT 8
                """
            )).mappings().all()

        payload = {
            "expected_total": int(summary.get("expected_total") or 0) if summary else 0,
            "missing_total": int(summary.get("missing_total") or 0) if summary else 0,
            "by_priority": [dict(row) for row in by_priority],
            "by_status": [dict(row) for row in by_status],
            "sample_missing": [dict(row) for row in sample],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        return jsonify(payload)
    except Exception as exc:
        logger.warning({
            "level": "WARNING",
            "type": "db",
            "message": f"Warehouse status query failed: {exc}",
            "session_id": None,
        })
        return jsonify({"error": "Warehouse status query failed"}), 500


@app.route("/api/data_framework/exports", methods=["GET"])
def api_data_framework_exports():
    """Return daily manifest or fallback to NDJSON exports; read-only endpoint for UI backfill."""
    principal, _, _ = get_request_principal()
    if not principal and not ALLOW_DEV_NO_PRINCIPAL:
        return jsonify({"error": "Unauthorized"}), 403
    try:
        date_str = (request.args.get("date") or datetime.now().strftime("%Y%m%d")).strip()
    except Exception:
        date_str = datetime.now().strftime("%Y%m%d")
    try:
        limit = int(request.args.get("limit") or 100)
    except Exception:
        limit = 100

    exports_dir = LOG_DIR / "data_framework_exports"
    manifest_path = exports_dir / f"exports-{date_str}-manifest.json"
    items = []
    generated_at = None
    if manifest_path.exists():
        try:
            with open(manifest_path, 'rb') as mf:
                payload = orjson.loads(mf.read())
                items = payload.get('items', []) if isinstance(payload, dict) else []
                generated_at = payload.get('generated_at') if isinstance(payload, dict) else None
        except Exception:
            items = []

    # Fallback: read last `limit` entries from exports.jsonl
    if not items:
        exports_file = exports_dir / 'exports.jsonl'
        if exports_file.exists():
            try:
                # read all lines and take last `limit`
                with open(exports_file, 'rb') as ef:
                    lines = [line for line in ef if line.strip()]
                last_lines = lines[-limit:]
                for line in last_lines:
                    try:
                        items.append(orjson.loads(line))
                    except Exception:
                        continue
                generated_at = datetime.now().isoformat()
            except Exception:
                items = []

    return jsonify({"date": date_str, "count": len(items), "generated_at": generated_at, "items": items})


@app.route("/azure_health", methods=["GET"])
def azure_health_page():
    auth_error = _health_auth_response()
    if auth_error:
        return auth_error
    runtime_hints = {
        "async_mode": _SOCKETIO_ASYNC_MODE,
        "async_framework": "threading (native Python)",
        "eventlet_deprecated": "disabled",
        "transports": _SOCKETIO_CLIENT_TRANSPORTS,
        "deploy_env": DEPLOY_ENV or "local",
    }
    return render_template(
        "azure_health.html",
        task_definitions=_public_health_task_definitions(),
        runtime_hints=runtime_hints,
        socketio_client_config=SOCKETIO_CLIENT_CONFIG,
        initial_tasks=_get_health_tasks(),
    )


@app.route("/api/health_tasks", methods=["GET"])
def api_list_health_tasks():
    auth_error = _health_auth_response()
    if auth_error:
        return auth_error
    return jsonify({"tasks": _get_health_tasks()})


@app.route("/api/health_tasks", methods=["POST"])
def api_start_health_task():
    auth_error = _health_auth_response()
    if auth_error:
        return auth_error
    data = request.get_json(silent=True) or {}
    task_key = str(data.get("task") or "").strip()
    if not task_key:
        return jsonify({"error": "Task key required."}), 400
    if task_key not in HEALTH_TASK_DEFINITIONS:
        return jsonify({"error": "Unknown task."}), 404
    record = _launch_health_task(task_key)
    return jsonify({"task": record})


@app.route("/api/health_tasks/<task_id>", methods=["GET"])
def api_health_task_detail(task_id: str):
    auth_error = _health_auth_response()
    if auth_error:
        return auth_error
    record = _get_health_task(task_id)
    if not record:
        return jsonify({"error": "Task not found."}), 404
    return jsonify({"task": record})


@app.route('/test/ui/prompt', methods=['POST'])
def test_ui_prompt():
    if not TEST_UI_ROUTES_ENABLED:
        return jsonify({"error": "Test UI routes disabled"}), 404

    data = request.get_json(silent=True) or {}
    session_id = safe_strip(safe_get(data, "session_id"))
    if not session_id:
        return jsonify({"error": "session_id required"}), 400
    if not session_manager.has_session(session_id):
        return jsonify({"error": "unknown session"}), 404

    title = safe_strip(safe_get(data, "title")) or "Test Prompt"
    message = safe_strip(safe_get(data, "message")) or "Select an option"
    options_raw = safe_get(data, "options")

    options = []
    if isinstance(options_raw, list):
        for idx, opt in enumerate(options_raw):
            try:
                if isinstance(opt, dict):
                    label = safe_strip(opt.get("label") or opt.get("title") or opt.get("name")) or f"Option {idx+1}"
                    meta = safe_strip(opt.get("meta") or opt.get("summary")) or ""
                    options.append({"index": opt.get("index") or idx + 1, "label": label, "meta": meta, "metadata": opt})
                else:
                    options.append({"index": idx + 1, "label": str(opt), "meta": ""})
            except Exception:
                options.append({"index": idx + 1, "label": str(opt), "meta": ""})

    payload = {
        "type": "prompt",
        "message": message,
        "session_id": session_id,
        "context": {
            "title": title,
            "options": options,
        }
    }

    try:
        store_log(session_id, normalize_log_obj({"type": "prompt", "level": "INFO", "message": message, "session_id": session_id}))
    except Exception:
        pass

    try:
        socketio.emit('parser_output', payload, room=session_id)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"success": True, "emitted": True, "session_id": session_id, "options": len(options)})

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
    principal, _, _ = get_request_principal()
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
    principal, _, _ = get_request_principal()
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

@app.route("/api/quick_copy", methods=["POST"])
def api_quick_copy():
    data = request.get_json(silent=True) or {}
    root = (data.get("root") or "output").lower().strip()
    subpath = (data.get("path") or "").strip().replace("\\", "/")
    name = (data.get("name") or "").strip()
    session_id = resolve_session_id(data or {}, create_if_missing=False)
    if not session_id:
        return jsonify({"success": False, "error": "session_id required"}), 400

    roots = {"input": INPUT_DIR, "output": OUTPUT_DIR, "uploads": UPLOADS_DIR}
    base = roots.get(root)
    if not base or not name:
        return jsonify({"success": False, "error": "Invalid parameters."}), 400

    abs_base = os.path.abspath(base)
    want_dir = os.path.normpath(os.path.join(abs_base, subpath))
    if not want_dir.startswith(abs_base):
        return jsonify({"success": False, "error": "Path escape blocked."}), 400
    fpath = os.path.normpath(os.path.join(want_dir, name))
    if not fpath.startswith(abs_base) or not os.path.isfile(fpath):
        return jsonify({"success": False, "error": "Not found."}), 404

    principal, _, _ = get_request_principal()
    if root == "output":
        allowed, reason = _is_output_download_allowed(fpath, principal, session_id)
        if not allowed:
            _log_download_access({
                "principal": principal or "anonymous",
                "session_id": session_id,
                "file": fpath,
                "root": root,
                "allowed": False,
                "reason": reason,
            })
            return jsonify({"success": False, "error": "Unauthorized output copy"}), 403

    dest_dir = _ensure_quick_copy_dir(session_id)
    if not dest_dir:
        return jsonify({"success": False, "error": "Quick copy directory unavailable."}), 500
    try:
        dest_name = _unique_quick_copy_name(dest_dir, name)
        dest_path = dest_dir / dest_name
        shutil.copy2(fpath, dest_path)
    except Exception as exc:
        logger.error({
            "level": "ERROR",
            "type": "cache",
            "message": f"Quick copy failed: {exc}",
            "session_id": session_id,
        })
        return jsonify({"success": False, "error": "Quick copy failed."}), 500

    quick_rel = f"quick_copy/{dest_dir.name}/{dest_name}"
    return jsonify({
        "success": True,
        "url": url_for("static", filename=quick_rel),
        "filename": dest_name,
    })

@app.route("/api/quick_copy/clear", methods=["POST"])
def api_quick_copy_clear():
    data = request.get_json(silent=True) or {}
    session_id = resolve_session_id(data or {}, create_if_missing=False)
    if not session_id:
        return jsonify({"success": False, "error": "session_id required"}), 400
    _cleanup_quick_copy_dir(session_id)
    return jsonify({"success": True})

@app.route("/download_fs")
def download_fs():
    """Enhanced filesystem download with integrity verification."""
    import asyncio
    import os
    from pathlib import Path
    
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
        
    # Get principal and session for tracking
    principal, _, _ = get_request_principal()
    if not principal:
        principal = "anonymous"
    try:
        session_id = resolve_session_id({}, create_if_missing=False) or "no_session"
    except Exception:
        session_id = "no_session"

    if root == "output":
        allowed, reason = _is_output_download_allowed(fpath, principal, session_id)
        _log_download_access({
            "principal": principal,
            "session_id": session_id,
            "file": fpath,
            "root": root,
            "allowed": allowed,
            "reason": reason,
        })
        if not allowed:
            return jsonify({"error": "Unauthorized output download"}), 403
        
    # Only verify integrity for output files (cache deduplication)
    if root == "output":
        monitor = get_integrity_monitor()
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                cache_result = loop.run_until_complete(
                    monitor.get_or_cache_download(
                        file_name=name,
                        principal=principal,
                        session_id=session_id,
                        file_path=Path(fpath)
                    )
                )
                
                if session_id != "no_session":
                    try:
                        socketio.emit('download_ready', {
                            "session_id": session_id,
                            "filename": name,
                            "size": cache_result.get("size"),
                            "hash": cache_result.get("hash"),
                            "cache_hit": cache_result.get("cache_hit", False)
                        }, room=session_id)
                    except Exception:
                        pass
            finally:
                loop.close()
        except Exception as e:
            logger.error({"level": "ERROR", "type": "download", "message": f"Integrity check failed: {e}", "session_id": session_id})
            
    return send_file(fpath, as_attachment=True)


@app.route("/view_csv")
def view_csv():
    import csv
    import html as _html

    # pagination and search parameters
    page = max(1, int(request.args.get('page') or 1))
    page_size = max(10, min(2000, int(request.args.get('page_size') or 200)))
    q = (request.args.get('q') or '').strip().lower()
    highlight = request.args.get('highlight')

    root = (request.args.get("root") or "").lower().strip()
    subpath = (request.args.get("path") or "").strip().replace("\\", "/")
    name = (request.args.get("name") or "")
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

    def esc(s):
        return _html.escape(str(s))

    try:
        header = None
        rows_out = []
        total_matches = 0
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size

        with open(fpath, 'r', encoding='utf-8', errors='replace') as fh:
            reader = csv.reader(fh)
            header = next(reader, [])
            idx = 0  # 0-based index for data rows
            if q:
                # search mode: collect matching rows, compute total_matches
                for row in reader:
                    idx += 1
                    row_text = ' '.join([str(c).lower() for c in row])
                    if q in row_text:
                        if total_matches >= start_idx and len(rows_out) < page_size:
                            rows_out.append(row)
                        total_matches += 1
                        # guard: if we've collected enough and beyond some cap, continue counting but avoid heavy work
                        if total_matches > 20000 and len(rows_out) >= page_size:
                            # keep counting but skip storing
                            continue
            else:
                # pagination without search: skip until start_idx, then capture page_size rows
                for row in reader:
                    if idx >= start_idx and idx < end_idx:
                        rows_out.append(row)
                    idx += 1
                    # early stop when collected enough
                    if idx >= end_idx:
                        # continue to count total rows roughly by fast file iteration
                        break
                # count remaining to compute total_rows
                total_rows = idx
                for _ in reader:
                    total_rows += 1
                total_matches = total_rows

        # parse highlight as int if provided
        try:
            h = int(highlight) if highlight is not None else None
        except Exception:
            h = None

        # build HTML with search and pagination controls
        parts = ["<html><head><meta charset='utf-8'><title>CSV Viewer</title>",
                 "<style>body{font-family:Inter,Segoe UI,Arial,monospace;margin:12px}table{border-collapse:collapse;width:100%;}th,td{border:1px solid #ddd;padding:6px;font-size:12px;text-align:left}tr.highlight{background:#fff3cd}thead th{position:sticky;top:0;background:#f9fafb;z-index:2}.pv{margin-bottom:8px}</style>",
                 "</head><body>"]
        parts.append(f"<h2>{esc(name)}</h2>")
        parts.append(f"<div class=\"pv\"><a href=\"/download_fs?root={esc(root)}&path={esc(subpath)}&name={esc(name)}\" target=\"_blank\">Download CSV</a>")
        parts.append(f" &nbsp; <form style=\"display:inline;margin-left:12px;\" method=\"get\" action=\"/view_csv\">\n<input type=\"hidden\" name=\"root\" value=\"{esc(root)}\">\n<input type=\"hidden\" name=\"path\" value=\"{esc(subpath)}\">\n<input type=\"hidden\" name=\"name\" value=\"{esc(name)}\">\nSearch: <input name=\"q\" value=\"{esc(q)}\"> <input type=\"submit\" value=\"Find\">\n</form>")
        parts.append(" &nbsp; <span style='margin-left:12px'>Filter page: <input id=\"inview-search\" placeholder=\"Filter visible rows...\" style=\"padding:4px 6px;border:1px solid #ccc;border-radius:4px\"></span></div>")

        # pagination summary & controls
        total = total_matches
        total_pages = max(1, (total + page_size - 1) // page_size) if page_size > 0 else 1
        parts.append(f"<div style=\"margin-bottom:8px\">Page {page} of {total_pages} ({total} rows)")
        if page > 1:
            prev_q = f"/view_csv?root={esc(root)}&path={esc(subpath)}&name={esc(name)}&page={page-1}&page_size={page_size}"
            if q:
                prev_q += f"&q={esc(q)}"
            parts.append(f" <a href=\"{prev_q}\">Prev</a> ")
        if page < total_pages:
            next_q = f"/view_csv?root={esc(root)}&path={esc(subpath)}&name={esc(name)}&page={page+1}&page_size={page_size}"
            if q:
                next_q += f"&q={esc(q)}"
            parts.append(f" <a href=\"{next_q}\">Next</a>")
        parts.append("</div>")

        parts.append("<table>")
        if header:
            parts.append("<thead><tr>")
            for c in header:
                parts.append(f"<th>{esc(c)}</th>")
            parts.append("</tr></thead><tbody>")
            # render rows_out; compute displayed index
            base_index = (page - 1) * page_size
            for offset, row in enumerate(rows_out):
                idx_abs = base_index + offset + 1
                is_high = False
                if h is not None:
                    if h == idx_abs or h == idx_abs - 1:
                        is_high = True
                trclass = ' class="highlight"' if is_high else ''
                parts.append(f"<tr{trclass}>")
                for cell in row:
                    cell_text = str(cell)
                    if q:
                        # case-insensitive highlight, escape segments
                        try:
                            pattern = re.compile(re.escape(q), re.I)
                            def _hl(m):
                                return f"<mark>{esc(m.group(0))}</mark>"
                            # escape full cell and then re-apply highlight on original pieces
                            # safer approach: iterate matches and build escaped pieces
                            out = ''
                            last = 0
                            for m in pattern.finditer(cell_text):
                                out += esc(cell_text[last:m.start()])
                                out += f"<mark>{esc(cell_text[m.start():m.end()])}</mark>"
                                last = m.end()
                            out += esc(cell_text[last:])
                            parts.append(f"<td>{out}</td>")
                            continue
                        except Exception:
                            pass
                    parts.append(f"<td>{esc(cell_text)}</td>")
                parts.append("</tr>")
            parts.append("</tbody>")
        else:
            parts.append("<tr><td>(empty)</td></tr>")
        parts.append("</table>")
        # Client-side enhancements: smooth scroll to highlighted row and in-view search/filter
        parts.append(r"""
<script>
document.addEventListener('DOMContentLoaded', function () {
    try {
        // Smooth-scroll to the highlighted row if present
        var highlighted = document.querySelector('tr.highlight');
        if (highlighted) {
            try { highlighted.scrollIntoView({behavior:'smooth', block:'center'}); } catch(e) { highlighted.scrollIntoView(); }
        }

        // In-view filter (client-side) for visible rows on the current page
        var filterInput = document.getElementById('inview-search');
        if (filterInput) {
            var tbody = document.querySelector('table tbody');
            var rows = tbody ? Array.from(tbody.querySelectorAll('tr')) : [];
            var timer = null;
            var applyFilter = function() {
                var q = filterInput.value.trim().toLowerCase();
                if (!q) {
                        rows.forEach(function(r){ r.classList.remove('hidden'); });
                        return;
                    }
                    rows.forEach(function(r){
                        var text = r.textContent.toLowerCase();
                        if (text.indexOf(q) !== -1) r.classList.remove('hidden'); else r.classList.add('hidden');
                    });
            };
            filterInput.addEventListener('input', function(){
                if (timer) clearTimeout(timer);
                timer = setTimeout(applyFilter, 150);
            });
            // quick focus for convenience
            filterInput.addEventListener('keydown', function(e){ if (e.key === 'Escape') { filterInput.value=''; applyFilter(); } });
        }
    } catch (err) {
        console && console.debug && console.debug('viewer enhancements failed', err);
    }
});
</script>
</body></html>
""")
        return Response(''.join(parts), mimetype='text/html; charset=utf-8')
    except Exception:
        raise NotFound()


def _build_or_load_csv_index(csv_path: str, max_rows: int = 200000) -> tuple[int, str] | None:
    """
    Build or load a simple CSV row-to-byte-offset index (data rows, excluding header).
    Writes an index file next to the CSV with suffix `.idx` containing newline-separated offsets.
    Returns (count, index_path) on success, or None on failure.
    """
    try:
        idx_path = csv_path + '.idx'
        if os.path.exists(idx_path):
            # quick validation: try reading first line
            try:
                with open(idx_path, 'rb') as f:
                    _ = f.readline()
                # assume present
                # count entries
                with open(idx_path, 'rb') as f:
                    cnt = sum(1 for _ in f)
                return cnt, idx_path
            except Exception:
                # fallthrough to rebuild
                pass

        offsets = []
        with open(csv_path, 'rb') as fh:
            # read header line
            _ = fh.readline()
            pos = fh.tell()
            idx = 0
            while True:
                line = fh.readline()
                if not line:
                    break
                offsets.append(pos)
                idx += 1
                if idx >= max_rows:
                    break
                pos = fh.tell()

        # write index
        with open(idx_path, 'wb') as f:
            for off in offsets:
                f.write(f"{off}\n".encode('ascii'))
        return len(offsets), idx_path
    except Exception:
        return None


@app.route('/csv_locate')
def csv_locate():
    """Return a viewer URL for a requested CSV row using an index (build on demand).
    Params: root, path, name, row (1-based data row index), page_size (optional)
    """
    root = (request.args.get('root') or '').lower().strip()
    subpath = (request.args.get('path') or '').strip().replace('\\', '/')
    name = (request.args.get('name') or '')
    row = request.args.get('row')
    try:
        row_i = int(row) if row is not None else None
    except Exception:
        row_i = None
    page_size = max(10, min(2000, int(request.args.get('page_size') or 200)))
    roots = {'input': INPUT_DIR, 'output': OUTPUT_DIR, 'uploads': UPLOADS_DIR}
    base = roots.get(root)
    if not base or not name or not row_i or row_i < 1:
        return jsonify({'error': 'invalid parameters'}), 400

    abs_base = os.path.abspath(base)
    want_dir = os.path.normpath(os.path.join(abs_base, subpath))
    if not want_dir.startswith(abs_base):
        return jsonify({'error': 'path escape blocked'}), 400
    fpath = os.path.normpath(os.path.join(want_dir, name))
    if not fpath.startswith(abs_base) or not os.path.isfile(fpath):
        return jsonify({'error': 'not found'}), 404

    idx_info = _build_or_load_csv_index(fpath)
    if idx_info:
        count, idx_path = idx_info
        # ensure requested row exists
        if row_i > count:
            # fallback: compute page using count
            page = (row_i - 1) // page_size + 1
        else:
            page = (row_i - 1) // page_size + 1
    else:
        page = (row_i - 1) // page_size + 1

    viewer = f"/view_csv?root={root}&path={subpath}&name={name}&page={page}&page_size={page_size}&highlight={row_i}"
    return jsonify({'viewer': viewer, 'page': page})

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


# Serve a small set of well-known app-specific files that some browsers/devtools request
@app.route('/.well-known/appspecific/<path:filename>')
def serve_well_known_appspecific(filename):
    try:
        # Serve from the static folder under .well-known/appspecific if present
        well_known_dir = os.path.join(app.static_folder or 'static', '.well-known', 'appspecific')
        return send_from_directory(well_known_dir, filename, as_attachment=False)
    except Exception:
        raise NotFound()


def _normalize_party_bucket(party: str | None) -> str:
    if not party:
        return "other"
    raw = party.strip().lower()
    if raw.startswith("dem"):
        return "dem"
    if raw.startswith("rep"):
        return "rep"
    return "other"


def _compute_dropoff_items(rows: list[tuple], down_contest: str) -> list[dict]:
    grouped: dict[tuple, dict] = {}
    for county, year_val, party, pres_votes, down_votes in rows:
        key = (county or "", int(year_val) if year_val is not None else None)
        bucket = grouped.setdefault(key, {
            "county": county or "",
            "year": int(year_val) if year_val is not None else None,
            "pres_dem": 0,
            "pres_rep": 0,
            "pres_other": 0,
            "down_dem": 0,
            "down_rep": 0,
            "down_other": 0,
            "pres_contest": "President",
            "down_contest": down_contest,
        })
        party_bucket = _normalize_party_bucket(party)
        pres_val = int(pres_votes or 0)
        down_val = int(down_votes or 0)
        bucket[f"pres_{party_bucket}"] += pres_val
        bucket[f"down_{party_bucket}"] += down_val

    items: list[dict] = []
    for entry in grouped.values():
        pres_total = entry["pres_dem"] + entry["pres_rep"] + entry["pres_other"]
        down_total = entry["down_dem"] + entry["down_rep"] + entry["down_other"]
        entry["pres_total"] = pres_total
        entry["down_total"] = down_total
        entry["dem_dropoff"] = entry["pres_dem"] - entry["down_dem"]
        entry["rep_dropoff"] = entry["pres_rep"] - entry["down_rep"]
        entry["other_dropoff"] = entry["pres_other"] - entry["down_other"]
        entry["total_dropoff"] = pres_total - down_total
        entry["dem_pct_dropoff"] = round((entry["dem_dropoff"] / entry["pres_dem"] * 100), 2) if entry["pres_dem"] else 0
        entry["rep_pct_dropoff"] = round((entry["rep_dropoff"] / entry["pres_rep"] * 100), 2) if entry["pres_rep"] else 0
        entry["total_pct_dropoff"] = round((entry["total_dropoff"] / pres_total * 100), 2) if pres_total else 0
        items.append(entry)
    items.sort(key=lambda item: (item.get("year") or 0, item.get("county") or ""), reverse=True)
    return items

@app.route("/api/warehouse_election_results", methods=["GET"])
def api_warehouse_election_results():
    """
    Query election results from warehouse and/or fixtures.
    
        Parameters:
      - state: Two-letter state code (required if searching)
      - county: County name filter (optional)
      - contest: Contest name filter (optional)
      - year: Election year filter (optional)
            - party: Party filter (optional, elector_totals)
            - metric: "rows" | "dropoff" | "elector_totals" (default "rows")
      - limit: Max results (default 500, max 1000)
      - data_source: "fixture" | "live" | "both" (default "both")
        - "fixture": Return only fixture data
        - "live": Return only database data
        - "both": Return both with provenance labels
    """
    principal, principal_source, _ = get_request_principal()
    state = request.args.get("state")
    county = request.args.get("county")
    contest = request.args.get("contest")
    year_str = request.args.get("year")
    data_source = (request.args.get("data_source") or "both").lower()
    metric = (request.args.get("metric") or "rows").lower()
    party = request.args.get("party")
    
    # Validate data_source parameter
    if data_source not in ("fixture", "live", "both"):
        return jsonify({"error": "data_source must be 'fixture', 'live', or 'both'"}), 400
    if metric not in ("rows", "dropoff", "elector_totals"):
        return jsonify({"error": "metric must be 'rows', 'dropoff', or 'elector_totals'"}), 400
    
    limit = request.args.get("limit", type=int)
    limit = max(1, min(1000, limit or 500))
    
    # Determine if DB is available
    db_enabled = os.environ.get("AUTO_INIT_DB", "true").lower() in ("1", "true", "yes")
    
    # Collect all results
    all_results = []

    if metric in ("dropoff", "elector_totals"):
        if not db_enabled:
            return jsonify({"error": "Database disabled"}), 503
        try:
            state_clean = _validate_filter_value("state", state, max_len=64)
            county_clean = _validate_filter_value("county", county, max_len=64)
            contest_clean = _validate_filter_value("contest", contest, max_len=140)
            party_clean = _validate_filter_value("party", party, max_len=32)
        except ValueError as exc:
            log_db_monitor_event({
                "type": "warehouse_query",
                "status": "invalid_filter",
                "error": str(exc),
            })
            return jsonify({"error": str(exc)}), 400

        if not state_clean:
            return jsonify({"error": "state is required for metric queries"}), 400

        year_val = None
        if year_str:
            try:
                year_val = int(year_str)
            except ValueError:
                return jsonify({"error": "year must be an integer"}), 400

        ensure_db_tables()
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
                if metric == "elector_totals":
                    if not contest_clean:
                        return jsonify({"error": "contest is required for elector_totals"}), 400
                    where = ["state = %s", "contest ILIKE %s"]
                    params = [state_clean, f"%{contest_clean}%"]
                    if county_clean:
                        where.append("county = %s")
                        params.append(county_clean)
                    if year_val:
                        where.append("EXTRACT(YEAR FROM election_date) = %s")
                        params.append(year_val)
                    if party_clean:
                        where.append("party ILIKE %s")
                        params.append(f"%{party_clean}%")
                    where_sql = f"WHERE {' AND '.join(where)}"
                    cur.execute(
                        f"""
                        SELECT county,
                               EXTRACT(YEAR FROM election_date) AS year,
                               party,
                               SUM(votes) AS votes
                        FROM warehouse_election_results
                        {where_sql}
                        GROUP BY county, year, party
                        ORDER BY year DESC NULLS LAST, county ASC
                        LIMIT %s
                        """,
                        params + [limit]
                    )
                    items = []
                    for county_val, year_out, party_val, votes in cur.fetchall():
                        items.append({
                            "county": county_val,
                            "year": int(year_out) if year_out is not None else None,
                            "party": party_val,
                            "votes": int(votes or 0),
                            "contest": contest_clean,
                            "state": state_clean,
                        })
                    return jsonify({"items": items, "count": len(items), "metric": metric})

                down_contest = contest_clean or "Senate"
                where = ["state = %s", "(contest ILIKE %s OR contest ILIKE %s)"]
                params = [state_clean, "%President%", f"%{down_contest}%"]
                if county_clean:
                    where.append("county = %s")
                    params.append(county_clean)
                if year_val:
                    where.append("EXTRACT(YEAR FROM election_date) = %s")
                    params.append(year_val)
                where_sql = f"WHERE {' AND '.join(where)}"
                cur.execute(
                    f"""
                    SELECT county,
                           EXTRACT(YEAR FROM election_date) AS year,
                           party,
                           SUM(CASE WHEN contest ILIKE %s THEN votes ELSE 0 END) AS pres_votes,
                           SUM(CASE WHEN contest ILIKE %s THEN votes ELSE 0 END) AS down_votes
                    FROM warehouse_election_results
                    {where_sql}
                    GROUP BY county, year, party
                    ORDER BY year DESC NULLS LAST, county ASC
                    """,
                    ["%President%", f"%{down_contest}%"] + params
                )
                items = _compute_dropoff_items(cur.fetchall(), down_contest)
                if limit:
                    items = items[:limit]
                return jsonify({"items": items, "count": len(items), "metric": metric})
        except Exception as e:
            logger.error({
                "level": "ERROR",
                "type": "db",
                "message": f"DB error (metric={metric}): {e}",
                "session_id": None,
            })
            return jsonify({"error": "Database query failed"}), 500
    
    # Query fixtures if requested
    if data_source in ("fixture", "both"):
        try:
            from webapp.parser import election_fixtures
            if state:
                state_clean = state.upper()
                year_val = None
                if year_str:
                    try:
                        year_val = int(year_str)
                    except ValueError:
                        pass
                
                fixture_results = election_fixtures.get_results_by_state(
                    state_clean,
                    year=year_val,
                    include_data_source=True
                )
                
                # Filter by contest if provided
                if contest:
                    contest_lower = contest.lower()
                    fixture_results = [
                        r for r in fixture_results
                        if contest_lower in r.get('contest', '').lower()
                    ]
                
                all_results.extend(fixture_results[:limit])
        except Exception as e:
            logger.warning({
                "level": "WARNING",
                "type": "api",
                "message": f"Failed to query fixtures: {e}",
                "session_id": None
            })
    
    # Query database if requested and enabled
    if data_source in ("live", "both") and db_enabled:
        try:
            state_clean = _validate_filter_value("state", state, max_len=64)
            county_clean = _validate_filter_value("county", county, max_len=64)
            contest_clean = _validate_filter_value("contest", contest, max_len=140)
        except ValueError as exc:
            log_db_monitor_event({
                "type": "warehouse_query",
                "status": "invalid_filter",
                "error": str(exc),
            })
            return jsonify({"error": str(exc)}), 400
        # Build database query
        where = []
        params = []
        if state_clean:
            where.append("state = %s")
            params.append(state_clean)
        if county_clean:
            where.append("county = %s")
            params.append(county_clean)
        if contest_clean:
            where.append("contest ILIKE %s")
            params.append(f"%{contest_clean}%")
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        limit_sql = "LIMIT %s"
        params.append(limit)
        
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
                    {limit_sql}
                    """,
                    params
                )
                cols = [d[0] for d in cur.description]
                db_rows = [dict(zip(cols, r)) for r in cur.fetchall()]
                
                # Add data_source field
                for row in db_rows:
                    row['data_source'] = 'live'
                
                all_results.extend(db_rows[:limit])
            
            log_db_monitor_event({
                "type": "warehouse_query",
                "status": "ok",
                "state": state,
                "county": county,
                "contest": contest,
                "limit": limit,
                "count": len(db_rows),
                "data_source": data_source,
            })
        except Exception as e:
            msg = str(e)
            if isinstance(e, (psycopg2.OperationalError, OperationalError)):
                logger.warning({
                    "level": "WARNING",
                    "type": "db",
                    "message": f"DB unavailable: {e}",
                    "session_id": None
                })
                log_db_monitor_event({
                    "type": "warehouse_query",
                    "status": "db_unavailable",
                    "error": str(e),
                    "state": state,
                    "county": county,
                    "contest": contest,
                    "limit": limit,
                })
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
                            {limit_sql}
                            """,
                            params
                        )
                        cols = [d[0] for d in cur.description]
                        db_rows = [dict(zip(cols, r)) for r in cur.fetchall()]
                        for row in db_rows:
                            row['data_source'] = 'live'
                        all_results.extend(db_rows[:limit])
                    log_db_monitor_event({
                        "type": "warehouse_query",
                        "status": "ok_after_create",
                        "state": state,
                        "county": county,
                        "contest": contest,
                        "limit": limit,
                        "count": len(db_rows),
                        "data_source": data_source,
                    })
                except Exception as e2:
                    logger.error({
                        "level": "ERROR",
                        "type": "db",
                        "message": f"DB error after retry: {e2}",
                        "session_id": None
                    })
                    log_db_monitor_event({
                        "type": "warehouse_query",
                        "status": "error_after_create",
                        "error": str(e2),
                        "state": state,
                        "county": county,
                        "contest": contest,
                        "limit": limit,
                    })
            else:
                logger.error({
                    "level": "ERROR",
                    "type": "db",
                    "message": f"DB error: {e}",
                    "session_id": None
                })
                log_db_monitor_event({
                    "type": "warehouse_query",
                    "status": "error",
                    "error": str(e),
                    "state": state,
                    "county": county,
                    "contest": contest,
                    "limit": limit,
                })
    
    # Return merged results
    return jsonify({"items": all_results, "count": len(all_results), "data_source": data_source})

@app.route("/delete/input/<filename>", methods=["POST"])
def delete_input_file(filename) -> str:
    file_path = os.path.join(INPUT_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        flash(f"Deleted '{filename}' from input folder.", "success")
    else:
        flash(f"File '{filename}' not found in input folder.", "danger")
    return redirect(request.referrer or url_for("ballot_lens"))

@app.route("/delete/output/<filename>", methods=["POST"])
def delete_output_file(filename) -> str:
    file_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        flash(f"Deleted '{filename}' from output folder.", "success")
    else:
        flash(f"File '{filename}' not found in output folder.", "danger")
    return redirect(request.referrer or url_for("ballot_lens"))

@app.route("/delete/uploads/<filename>", methods=["POST"])
def delete_upload_file(filename) -> str:
    file_path = os.path.join(UPLOADS_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        flash(f"Deleted '{filename}' from uploads folder.", "success")
    else:
        flash(f"File '{filename}' not found in uploads folder.", "danger")
    return redirect(request.referrer or url_for("ballot_lens"))

@app.route("/download/input/<filename>")
def download_input_file(filename) -> str:
    return send_from_directory(INPUT_DIR, filename, as_attachment=True)

@app.route("/download/output/<filename>")
def download_output_file(filename) -> str:
    """Enhanced download with integrity verification and cache deduplication."""
    import asyncio
    from pathlib import Path
    
    # Get principal for deduplication
    principal, principal_source, _ = get_request_principal()
    if not principal:
        principal = "anonymous"
        
    # Get session ID if available
    try:
        session_id = resolve_session_id({}, create_if_missing=False) or "no_session"
    except Exception:
        session_id = "no_session"
        
    file_path = Path(OUTPUT_DIR) / filename
    if not file_path.exists():
        raise NotFound()

    allowed, reason = _is_output_download_allowed(str(file_path), principal, session_id)
    _log_download_access({
        "principal": principal,
        "session_id": session_id,
        "file": str(file_path),
        "root": "output",
        "allowed": allowed,
        "reason": reason,
    })
    if not allowed:
        return Response("Unauthorized output download", status=403, mimetype="text/plain")
        
    # Async integrity verification with cache
    monitor = get_integrity_monitor()
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            cache_result = loop.run_until_complete(
                monitor.get_or_cache_download(
                    file_name=filename,
                    principal=principal,
                    session_id=session_id,
                    file_path=file_path
                )
            )
            
            # Emit download_ready event with integrity info
            if session_id != "no_session":
                try:
                    socketio.emit('download_ready', {
                        "session_id": session_id,
                        "filename": filename,
                        "size": cache_result.get("size"),
                        "hash": cache_result.get("hash"),
                        "cache_hit": cache_result.get("cache_hit", False),
                        "ttl_expires_at": cache_result.get("ttl_expires_at")
                    }, room=session_id)
                except Exception:
                    pass
                    
            logger.info({
                "level": "INFO",
                "type": "download",
                "message": f"File download: {filename} (cache_hit={cache_result.get('cache_hit', False)})",
                "session_id": session_id,
                "principal": principal,
                "file_hash": cache_result.get("hash"),
                "file_size": cache_result.get("size")
            })
        finally:
            loop.close()
    except Exception as e:
        logger.error({
            "level": "ERROR",
            "type": "download",
            "message": f"Download integrity check failed: {e}",
            "session_id": session_id,
            "filename": filename
        })
        
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)

@app.route("/download/uploads/<filename>")
def download_upload_file(filename) -> str:
    return send_from_directory(UPLOADS_DIR, filename, as_attachment=True)

@app.route("/ballot_lens", methods=["GET", "POST"])
def ballot_lens():
    try:
        qp_source = safe_lower(request.args.get("source", "")) if request.method == "GET" else ""
        if qp_source in {"input", "uploads"}:
            session['manual_source_pref'] = qp_source
        if request.method == "POST" and "data_file" in request.files:
            file = request.files.get("data_file")
            guard_ok, guard_reason = _guarded_ingestion_allowed("ballot_lens_upload")
            if not guard_ok:
                logger.warning({
                    "level": "WARNING",
                    "type": "security",
                    "message": f"Upload blocked by guarded ingestion gate: {guard_reason}",
                    "session_id": None,
                })
                flash("Upload blocked: guarded ingestion key required.", "danger")
                return redirect(request.referrer or url_for("ballot_lens"))
            ok, saved_name, err_path = _save_uploaded_file(file, str(UPLOADS_DIR), session_id=None)
            if ok and saved_name:
                session['FORCE_PARSE_INPUT_FILE'] = saved_name
                session['FORCE_PARSE_FORMAT'] = saved_name.rsplit('.', 1)[-1].lower() if '.' in saved_name else ''
                session['manual_source_pref'] = 'uploads'
                flash(f"File '{saved_name}' uploaded successfully.", "success")
            else:
                flash(saved_name or "Invalid file type or no file selected.", "danger")
        file_lists = get_all_file_lists()
        return render_template(
            "ballot_lens.html",
            input_files=file_lists["input_files"],
            output_files=file_lists["output_files"],
            uploaded_files=file_lists["uploaded_files"],
            manual_source=session.get('manual_source_pref', 'input'),
            allow_style_attr=os.environ.get("ALLOW_STYLE_ATTR", "0").lower() in ("1","true","yes"),
            static_version=os.environ.get("STATIC_VERSION", "v1"),
            socketio_client_config=SOCKETIO_CLIENT_CONFIG,
        )
    except Exception:
        import traceback
        print(traceback.format_exc())
        return "Internal Server Error", 500

@app.route("/ballot_lens_modern", methods=["GET"])
def ballot_lens_modern():
    """Redirect to consolidated modern interface at /ballot_lens."""
    return redirect(url_for("ballot_lens"))
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
            {"name": "Ballot Lens", "short_name": "Ballot Lens", "url": "/ballot_lens"},
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

@app.route("/quality_dashboard")
def quality_dashboard():
    """Quality metrics visualization dashboard."""
    return render_template("quality_dashboard.html")

@app.route("/quick-reference")
@app.route("/quick_reference")
def quick_reference_page():
    """Serve the Quick Reference guide with CSP-friendly headers and static CSS."""
    return render_template("quick_reference.html")

@app.route("/api/quality_metrics", methods=["GET"])
def api_quality_metrics():
    """API endpoint for quality metrics data."""
    import json
    from pathlib import Path
    
    # Query parameters for filtering
    handler_filter = request.args.get("handler")
    state_filter = request.args.get("state")
    min_confidence = request.args.get("min_confidence", type=float)
    limit = request.args.get("limit", default=100, type=int)
    
    results = []
    
    # Scan output directory for metadata files
    output_dir = Path(OUTPUT_DIR)
    if not output_dir.exists():
        return jsonify({"metrics": [], "count": 0})
    
    for folder in output_dir.iterdir():
        if not folder.is_dir():
            continue
        
        metadata_file = folder / "metadata.json"
        if not metadata_file.exists():
            continue
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Must have quality metrics
            if "quality_metrics" not in metadata:
                continue
            
            # Apply filters
            if handler_filter and metadata.get("handler") != handler_filter:
                continue
            if state_filter and metadata.get("state") != state_filter:
                continue
            
            quality = metadata.get("quality_metrics", {})
            conf = quality.get("extraction_confidence")
            if min_confidence is not None and (conf is None or conf < min_confidence):
                continue
            
            # Extract relevant fields
            result = {
                "folder": folder.name,
                "handler": metadata.get("handler"),
                "state": metadata.get("state"),
                "county": metadata.get("county"),
                "contest": metadata.get("contest"),
                "row_count": metadata.get("row_count"),
                "column_count": metadata.get("column_count"),
                "quality_metrics": quality,
                "timestamp": metadata.get("timestamp") or folder.name.split("__")[-1],
            }
            results.append(result)
            
            if len(results) >= limit:
                break
        except Exception:
            continue
    
    # Sort by timestamp (newest first)
    results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    return jsonify({"metrics": results, "count": len(results)})

@app.route("/api/auth/certificate_info", methods=["GET"])
def api_auth_certificate_info():
    """
    Return certificate metadata for the current session.
    This endpoint is used by the auth welcome page to display cert info to users.
    """
    principal, principal_source, cert_metadata = get_request_principal()
    if not principal:
        return jsonify({
            "error": "No certificate found",
            "principal": None,
            "cert_metadata": None
        }), 401
    
    # Build response with certificate metadata
    response = {
        "principal": principal,
        "principal_source": principal_source,
        "cert_metadata": cert_metadata or {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_context": {
            "host": request.host or "unknown",
            "remote_addr": request.remote_addr or "unknown"
        }
    }
    
    # Add privilege tier information if available
    if cert_metadata and cert_metadata.get("cn"):
        try:
            from webapp.parser.utils.privilege_tiers import get_principal_tier
            tier = get_principal_tier(principal, principal_source)
            if tier:
                response["privilege_tier"] = tier.value
        except Exception:
            response["privilege_tier"] = "STANDARD_USER"
    
    return jsonify(response)

@app.route("/auth/welcome")
def auth_welcome():
    """
    Render the certificate authentication welcome screen.
    This page displays certificate information to the user and allows them to proceed
    to the main application or view additional details.
    """
    # Get principal to verify cert is present
    principal, principal_source, cert_metadata = get_request_principal()

    if not principal:
        # Redirect to error page or main app
        return redirect(url_for("index"))
    
    # Get session ID if available
    try:
        session_id = resolve_session_id({}, create_if_missing=False)
    except Exception:
        session_id = None
    
    # Pass metadata to template
    return render_template("auth_welcome.html", 
                         principal=principal,
                         principal_source=principal_source,
                         cert_metadata=cert_metadata,
                         session_id=session_id)


@app.route("/upload/input", methods=["POST"])
@_rate_limit("5/minute")
def upload_to_input() -> str:
    file = request.files.get("file")
    logger.info({
        "level": "INFO",
        "type": "status",
        "message": f"Upload to input: {file.filename if file else 'No file'}",
        "session_id": None
    })
    guard_ok, guard_reason = _guarded_ingestion_allowed("upload_input")
    if not guard_ok:
        logger.warning({
            "level": "WARNING",
            "type": "security",
            "message": f"Upload blocked by guarded ingestion gate: {guard_reason}",
            "session_id": None,
        })
        flash("Upload blocked: guarded ingestion key required.", "danger")
        return redirect(request.referrer or url_for("ballot_lens"))
    # Gate uploads: require client principal or ADMIN_JWT_TOKEN fallback
    principal, _, _ = get_request_principal()
    admin_token = os.environ.get("ADMIN_JWT_TOKEN")
    auth_hdr = (request.headers.get("Authorization") or "").strip()
    token_ok = False
    if admin_token and auth_hdr.lower().startswith("bearer "):
        try:
            token_ok = hmac.compare_digest(auth_hdr.split(None, 1)[1].strip(), admin_token)
        except Exception:
            token_ok = False

    if not principal and not token_ok:
        # Quarantine the upload for admin review
        try:
            if file:
                qdir = os.path.join(str(UPLOADS_DIR), "quarantine")
                os.makedirs(qdir, exist_ok=True)
                orig = getattr(file, 'filename', 'upload') or 'upload'
                fname = _generate_upload_filename(orig)
                qname = f"quarantine_{fname}"
                save_path = os.path.join(qdir, qname)
                file.save(save_path)
                log_flagged_url({
                    "event": "upload_quarantine",
                    "original_name": orig,
                    "saved_name": qname,
                    "reason": "missing_principal",
                })
                flash(f"Upload quarantined for review: {qname}", "warning")
            else:
                flash("No file uploaded.", "danger")
        except Exception as e:
            logger.error({"level": "ERROR", "type": "upload", "message": f"Quarantine save failed: {e}"})
            flash("Failed to save upload.", "danger")
        return redirect(request.referrer or url_for("ballot_lens"))

    ok, saved_name, err_path = _save_uploaded_file(file, str(INPUT_DIR), session_id=None)
    if ok and saved_name:
        flash(f"File '{saved_name}' uploaded to input folder.", "success")
    else:
        flash(saved_name or "Invalid file type or no file selected.", "danger")
    return redirect(request.referrer or url_for("ballot_lens"))

@app.route("/upload/output", methods=["POST"])
@_rate_limit("5/minute")
def upload_to_output() -> str:
    file = request.files.get("file")
    logger.info({
        "level": "INFO",
        "type": "status",
        "message": f"Upload to output: {file.filename if file else 'No file'}",
        "session_id": None
    })
    guard_ok, guard_reason = _guarded_ingestion_allowed("upload_output")
    if not guard_ok:
        logger.warning({
            "level": "WARNING",
            "type": "security",
            "message": f"Upload blocked by guarded ingestion gate: {guard_reason}",
            "session_id": None,
        })
        flash("Upload blocked: guarded ingestion key required.", "danger")
        return redirect(request.referrer or url_for("ballot_lens"))
    principal, _, _ = get_request_principal()
    admin_token = os.environ.get("ADMIN_JWT_TOKEN")
    auth_hdr = (request.headers.get("Authorization") or "").strip()
    token_ok = False
    if admin_token and auth_hdr.lower().startswith("bearer "):
        try:
            token_ok = hmac.compare_digest(auth_hdr.split(None, 1)[1].strip(), admin_token)
        except Exception:
            token_ok = False

    if not principal and not token_ok:
        try:
            if file:
                qdir = os.path.join(str(UPLOADS_DIR), "quarantine")
                os.makedirs(qdir, exist_ok=True)
                orig = getattr(file, 'filename', 'upload') or 'upload'
                fname = _generate_upload_filename(orig)
                qname = f"quarantine_{fname}"
                save_path = os.path.join(qdir, qname)
                file.save(save_path)
                log_flagged_url({
                    "event": "upload_quarantine",
                    "original_name": orig,
                    "saved_name": qname,
                    "reason": "missing_principal",
                })
                flash(f"Upload quarantined for review: {qname}", "warning")
            else:
                flash("No file uploaded.", "danger")
        except Exception as e:
            logger.error({"level": "ERROR", "type": "upload", "message": f"Quarantine save failed: {e}"})
            flash("Failed to save upload.", "danger")
        return redirect(request.referrer or url_for("ballot_lens"))

    ok, saved_name, err_path = _save_uploaded_file(file, str(OUTPUT_DIR), session_id=None)
    if ok and saved_name:
        flash(f"File '{saved_name}' uploaded to output folder.", "success")
    else:
        flash(saved_name or "Invalid file type or no file selected.", "danger")
    return redirect(request.referrer or url_for("ballot_lens"))

@app.route("/upload/uploads", methods=["POST"])
@_rate_limit("5/minute")
def upload_to_uploads() -> str:
    file = request.files.get("data_file") or request.files.get("file")
    guard_ok, guard_reason = _guarded_ingestion_allowed("upload_uploads")
    if not guard_ok:
        logger.warning({
            "level": "WARNING",
            "type": "security",
            "message": f"Upload blocked by guarded ingestion gate: {guard_reason}",
            "session_id": None,
        })
        flash("Upload blocked: guarded ingestion key required.", "danger")
        return redirect(request.referrer or url_for("ballot_lens"))
    principal, _, _ = get_request_principal()
    admin_token = os.environ.get("ADMIN_JWT_TOKEN")
    auth_hdr = (request.headers.get("Authorization") or "").strip()
    token_ok = False
    if admin_token and auth_hdr.lower().startswith("bearer "):
        try:
            token_ok = hmac.compare_digest(auth_hdr.split(None, 1)[1].strip(), admin_token)
        except Exception:
            token_ok = False

    if not principal and not token_ok:
        try:
            if file:
                qdir = os.path.join(str(UPLOADS_DIR), "quarantine")
                os.makedirs(qdir, exist_ok=True)
                orig = getattr(file, 'filename', 'upload') or 'upload'
                fname = _generate_upload_filename(orig)
                qname = f"quarantine_{fname}"
                save_path = os.path.join(qdir, qname)
                file.save(save_path)
                log_flagged_url({
                    "event": "upload_quarantine",
                    "original_name": orig,
                    "saved_name": qname,
                    "reason": "missing_principal",
                })
                session['FORCE_PARSE_INPUT_FILE'] = qname
                session['FORCE_PARSE_FORMAT'] = qname.rsplit('.', 1)[-1].lower() if '.' in qname else ''
                session['manual_source_pref'] = 'uploads'
                flash(f"Upload quarantined for review: {qname}", "warning")
            else:
                flash("No file uploaded.", "danger")
        except Exception as e:
            logger.error({"level": "ERROR", "type": "upload", "message": f"Quarantine save failed: {e}"})
            flash("Failed to save upload.", "danger")
        return redirect(request.referrer or url_for("ballot_lens"))

    ok, saved_name, err_path = _save_uploaded_file(file, str(UPLOADS_DIR), session_id=None)
    if ok and saved_name:
        session['FORCE_PARSE_INPUT_FILE'] = saved_name
        session['FORCE_PARSE_FORMAT'] = saved_name.rsplit('.', 1)[-1].lower() if '.' in saved_name else ''
        session['manual_source_pref'] = 'uploads'  # default UI to uploads after upload
        flash(f"File '{saved_name}' uploaded to uploads folder.", "success")
    else:
        flash(saved_name or "Invalid file type or no file selected.", "danger")
    return redirect(request.referrer or url_for("ballot_lens"))

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
    This just emits a SocketIO event style workflow (reuse ballot_lens logic).
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
    new_session = 'sess_' + secrets.token_urlsafe(16)
    session['logical_session_id'] = new_session
    flash(f"Re-running prior config (run_id={run_id}) in new session {new_session}", "success")
    # Front-end JS should now request a run (or we can directly invoke)
    return redirect(url_for("ballot_lens", source=source))

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
    new_sid = 'sess_' + secrets.token_urlsafe(16)
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
    _ensure_quick_copy_dir(new_sid)
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
    _ensure_quick_copy_dir(sid)
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
        principal, principal_source, cert_metadata = get_request_principal()
        if cert_metadata and isinstance(cert_metadata, dict) and cert_metadata.get("error"):
            logger.warning({
                "level": "WARNING",
                "type": "auth",
                "message": "Client certificate metadata parse failed.",
                "session_id": None,
                "principal": principal,
                "error": cert_metadata.get("error"),
            })
        if not principal and not ALLOW_ANON_NO_PRINCIPAL:
            emit('parser_output', {
                "level": "ERROR",
                "type": "auth",
                "message": "Missing client certificate or SSO principal; connection rejected.",
                "session_id": None
            }, room=getattr(request, 'sid', None))
            return False
        
        # --- Certificate validation (Step 4: Check for expiry/changes) ---
        cert_fingerprint = None
        cert_expired = False
        cert_changed = False
        if cert_metadata and isinstance(cert_metadata, dict):
            # Extract fingerprint from cert principal or headers (SHA256)
            try:
                if isinstance(principal, str) and principal.startswith("cert:"):
                    cert_fingerprint = principal.split(":", 1)[1]
                else:
                    cert_header = request.headers.get('X-ARR-ClientCert', '')
                    if cert_header:
                        import hashlib
                        cert_fingerprint = hashlib.sha256(cert_header.encode('utf-8')).hexdigest()[:16]
                    cert_expired = cert_metadata.get('is_expired', False)
                    if cert_expired:
                        logger.warning({
                            "level": "WARNING",
                            "type": "auth",
                            "message": "Client certificate is expired.",
                            "session_id": None,
                            "principal": principal,
                            "cert_cn": cert_metadata.get('cn'),
                            "expiry_date": cert_metadata.get('expiry_date'),
                        })
            except Exception:
                pass
        
        if principal_source == "dev_bypass":
            logger.warning({
                "level": "WARNING",
                "type": "auth",
                "message": "Dev principal bypass active (ALLOW_DEV_NO_PRINCIPAL).",
                "session_id": None,
                "principal": principal,
                "remote_addr": request.remote_addr,
                "host": request.host,
            })
        def _truthy(val):
            if val is None:
                return False
            if isinstance(val, bool):
                return val
            try:
                s = str(val).strip().lower()
            except Exception:
                return False
            return s in {"1", "true", "yes", "y", "on", "reuse", "reuse_session"}

        reuse_hint = False
        if isinstance(auth, dict):
            reuse_hint = _truthy(auth.get('reuse_session') or auth.get('reuse'))
        try:
            arg_reuse = request.args.get('reuse_session')
            hdr_reuse = request.headers.get('X-Reuse-Session')
            reuse_hint = reuse_hint or _truthy(arg_reuse) or _truthy(hdr_reuse)
        except Exception:
            pass

        allow_reuse = ALLOW_AUTO_SESSION_REUSE or reuse_hint
        if principal_source == "dev_bypass" and not reuse_hint:
            allow_reuse = False
        requested = None
        if isinstance(auth, dict):
            requested = safe_get(auth, 'requested_session_id')
        if not requested:
            requested = request.args.get('prev_session_id')
        revived = None
        if allow_reuse and requested and session_manager.has_session(requested):
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
        elif allow_reuse:
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
            _recover_stale_session(resolved, reason="connect")
            
            # Cache certificate and check for changes (Step 4)
            if cert_fingerprint and cert_metadata:
                # Check if cert has changed since last seen
                if session_manager.cert_changed(resolved, cert_fingerprint, cert_metadata):
                    logger.info({
                        "level": "INFO",
                        "type": "auth",
                        "message": "Certificate changed or new for session.",
                        "session_id": resolved,
                        "principal": principal,
                        "cert_cn": cert_metadata.get('cn'),
                        "fingerprint": cert_fingerprint,
                    })
                    transition_session(
                        resolved,
                        SessionState.IDLE,
                        locked=True,
                        phase=PipelinePhase.PREPARE,
                        broadcast=False,
                        extras={
                            "auth_blocked": True,
                            "auth_block_reason": "cert_changed",
                        },
                    )
                    # Emit cert_changed event for frontend to trigger UI update
                    try:
                        socketio.emit('cert_changed', {
                            "session_id": resolved,
                            "fingerprint": cert_fingerprint,
                            "cert_metadata": cert_metadata,
                            "principal": principal
                        }, room=resolved)
                    except Exception:
                        pass
                    try:
                        socketio.emit('auth_blocked', {
                            "session_id": resolved,
                            "reason": "cert_changed",
                        }, room=resolved)
                    except Exception:
                        pass
                
                # Cache the cert for future comparisons
                session_manager.cache_cert(resolved, cert_fingerprint, cert_metadata, principal)
                
                # Check if cert is expired
                if session_manager.cert_expired(resolved):
                    logger.warning({
                        "level": "WARNING",
                        "type": "auth",
                        "message": "Cached certificate is expired.",
                        "session_id": resolved,
                        "principal": principal,
                        "expiry_date": cert_metadata.get('expiry_date'),
                    })
                    # Emit cert_expired event for frontend
                    try:
                        socketio.emit('cert_expired', {
                            "session_id": resolved,
                            "principal": principal,
                            "expiry_date": cert_metadata.get('expiry_date')
                        }, room=resolved)
                    except Exception:
                        pass

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
            "session_id": resolved,
            "principal": principal,
            "principal_source": principal_source,
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
    
    # Get session ID before unbinding
    logical = None
    if req_sid:
        logical = session_manager.resolve_socket(req_sid)
    
    # Unbind socket
    unbound_session = session_manager.unbind_socket(req_sid) if req_sid else None
    logical = logical or unbound_session
    
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


@socketio.on('ack_cert_reauth')
def handle_ack_cert_reauth(data=None) -> None:
    payload = data or {}
    session_id = resolve_session_id(payload, create_if_missing=False)
    if not session_id:
        logger.warning({
            "level": "WARNING",
            "type": "auth",
            "message": "No session_id provided for cert reauth ack.",
            "session_id": None,
        })
        return
    transition_session(
        session_id,
        SessionState.IDLE,
        locked=False,
        phase=PipelinePhase.PREPARE,
        broadcast=False,
        extras={
            "auth_blocked": False,
            "auth_block_reason": None,
        },
    )
    try:
        socketio.emit('auth_unblocked', {
            "session_id": session_id,
        }, room=session_id)
    except Exception:
        pass
    logger.info({
        "level": "INFO",
        "type": "auth",
        "message": "Certificate reauth acknowledged; session unblocked.",
        "session_id": session_id,
    })

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

    if _socket_payload_too_large(data) or _socket_payload_too_large(value):
        logger.error({
            "level": "ERROR",
            "type": "prompt",
            "message": "Prompt payload too large.",
            "session_id": session_id,
        })
        return
    
    # Fallback: if session_id not resolved, try socket mapping
    if not session_id:
        try:
            socket_sid = safe_sid()
        except Exception:
            socket_sid = getattr(request, 'sid', None)
        if isinstance(socket_sid, str):
            session_id = session_manager.resolve_socket(socket_sid)
    
    if not session_id or not session_manager.has_session(session_id):
        logger.error({
            "level": "ERROR",
            "type": "prompt",
            "message": "Invalid or unknown session_id for prompt.",
            "session_id": None,
        })
        return

    if not _rate_limit_socket_action(session_id, "parser_prompt"):
        logger.warning({
            "level": "WARNING",
            "type": "prompt",
            "message": "Rate limit exceeded for prompt responses.",
            "session_id": session_id,
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

@socketio.on('prompt_cancel')
def handle_prompt_cancel(data=None) -> None:
    payload = data or {}
    session_id = resolve_session_id(payload, create_if_missing=False)
    reason = safe_lower(safe_get(payload, 'reason', 'cancel'))
    if not session_id or not session_manager.has_session(session_id):
        logger.error({
            "level": "ERROR",
            "type": "prompt",
            "message": "Invalid or unknown session_id for prompt_cancel.",
            "session_id": None,
        })
        return

    if not _rate_limit_socket_action(session_id, "prompt_cancel"):
        logger.warning({
            "level": "WARNING",
            "type": "prompt",
            "message": "Rate limit exceeded for prompt_cancel.",
            "session_id": session_id,
        })
        return

    prompt_session = prompt.prompt_sessions.get(session_id)
    if prompt_session and not prompt_session.is_resolved():
        try:
            prompt_session.set_response("cancel")
        except Exception:
            try:
                prompt_session.cancel()
            except Exception:
                pass

    try:
        cancel_processing(session_id)
    except Exception:
        pass

    transition_session(
        session_id,
        SessionState.IDLE,
        locked=False,
        phase=PipelinePhase.PREPARE,
        broadcast=False,
        extras={
            "manual_source": get_manual_source(session_id),
            "manual_source_origin": get_manual_source_origin(session_id),
            "prompt_cancelled": True,
            "prompt_cancel_reason": reason,
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

    if not _rate_limit_socket_action(session_id, "cancel_parser"):
        logger.warning({
            "level": "WARNING",
            "type": "cancel",
            "message": "Rate limit exceeded for cancel requests.",
            "session_id": session_id,
        })
        return

    # If a prompt is active, resolve it immediately so the worker unblocks
    prompt_session = prompt.prompt_sessions.get(session_id)
    if prompt_session and not prompt_session.is_resolved():
        try:
            prompt_session.set_response("cancel")
        except Exception:
            try:
                prompt_session.cancel()
            except Exception:
                pass

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
        SessionState.CANCELLED,
        locked=False,
        phase=PipelinePhase.PREPARE,
        extras={
            "manual_source": get_manual_source(session_id),
            "manual_source_origin": get_manual_source_origin(session_id),
            "prompt_cancelled": True,
            "cancel_reason": "user_cancel",
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
    if not _rate_limit_socket_action(sid, "toggle_output_bypass"):
        logger.warning({
            "level": "WARNING",
            "type": "status",
            "message": "Rate limit exceeded for output bypass toggle.",
            "session_id": sid,
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
    if not _rate_limit_socket_action(sid, "set_manual_source"):
        logger.warning({
            "level": "WARNING",
            "type": "input",
            "message": "Rate limit exceeded for manual source updates.",
            "session_id": sid,
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

@socketio.on('ballot_lens')
def handle_ballot_lens(data=None) -> None:
    """
    Ensures session join is fully propagated before emitting any logs,
    synchronizes session/thread state, and launches the parser pipeline.
    """
    cleanup_sessions()
    dev_isolation_bypass = _is_dev_isolation_bypass_request()
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

    if not _rate_limit_socket_action(session_id, "ballot_lens"):
        emit('parser_output', normalize_log_obj({
            "level": "WARNING",
            "type": "status",
            "message": "Rate limit exceeded for starting a job.",
            "session_id": session_id,
        }), room=session_id)
        return

    principal, principal_source, cert_metadata = get_request_principal()

    # --- Ensure join_room is fully propagated before any log emission ---
    join_room(session_id)
    socketio.sleep(0.25)  # More robust than time.sleep for Flask-SocketIO (yields event loop)

    # --- Send session ID back to frontend immediately ---
    emit('session_id', {'session_id': session_id})

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
    session_manager.update_metadata(session_id, dev_isolation_bypass=dev_isolation_bypass)

    if meta.get("auth_blocked"):
        emit('parser_output', normalize_log_obj({
            "level": "INFO",
            "type": "auth",
            "message": "Session blocked due to certificate change. Re-authenticate to continue.",
            "session_id": session_id,
        }), room=session_id)
        return

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
            allowed, reason = safe_validate_external_url(
                url_text,
                allowlist_suffixes=URL_ALLOWLIST_SUFFIXES,
                allowlist_hosts=URL_ALLOWLIST_HOSTS,
                enforce_allowlist=URL_ENFORCE_ALLOWLIST,
                block_private_ips=URL_BLOCK_PRIVATE_IPS,
                allowlist_bypass=dev_isolation_bypass,
            )
            if not allowed:
                logger.warning({
                    "level": "WARNING",
                    "type": "input",
                    "message": f"Blocked direct URL: {reason}",
                    "session_id": session_id,
                    "url": url_text,
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
        guard_ok, guard_reason = _guarded_ingestion_allowed("direct_urls")
        if not guard_ok:
            logger.warning({
                "level": "WARNING",
                "type": "security",
                "message": f"Direct URL ingestion blocked by guarded gate: {guard_reason}",
                "session_id": session_id,
            })
            direct_urls = []
        logger.info({
            "level": "INFO",
            "type": "input",
            "message": f"Direct URL override engaged with {len(direct_urls)} link(s).",
            "session_id": session_id,
            "urls": direct_urls
        })

    warehouse_override_url = safe_strip(safe_get(payload, 'warehouse_override_url', ''))
    if warehouse_override_url:
        logger.info({
            "level": "INFO",
            "type": "status",
            "message": "Warehouse override acknowledged for parse run.",
            "session_id": session_id,
            "warehouse_override_url": warehouse_override_url,
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
        "direct_url_count": len(direct_urls),
        "warehouse_override_url": warehouse_override_url or None,
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
                urls=direct_urls if direct_urls else None,
                principal=principal,
                principal_source=principal_source,
                dev_isolation_bypass=dev_isolation_bypass,
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
                "warehouse_override_url": warehouse_override_url or None,
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

# --- FEC mappings review endpoints ---
MAPPINGS_PATH = os.path.join(str(PROJECT_ROOT), 'webapp', 'parser', 'fixtures', 'mappings.json')
REPORT_JSONL = os.path.join(str(PROJECT_ROOT), 'webapp', 'parser', 'fixtures', 'fuzzy_match_report_full.jsonl')


def _read_jsonl(path):
    out = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(orjson.loads(line) if isinstance(line, (bytes, bytearray)) else json.loads(line))
                except Exception:
                    try:
                        out.append(json.loads(line))
                    except Exception:
                        continue
    except Exception:
        return []
    return out


@app.route('/fec_mappings_review')
def fec_mappings_review():
    # Simple single-file HTML reviewer for fuzzy-match problem rows
    html = '''<!doctype html><html><head><meta charset="utf-8"><title>FEC Mappings Review</title></head><body>
    <h2>FEC Mappings Review</h2>
    <div id="status">Loading...</div>
    <table id="rows" border="1" style="border-collapse:collapse;margin-top:12px;width:100%"><thead><tr><th>File</th><th>Row</th><th>Candidate</th><th>State</th><th>Match</th><th>Score</th><th>Nearest</th><th>Actions</th></tr></thead><tbody></tbody></table>
    <script>
    async function load(){
      const resp = await fetch('/api/fec/problem_rows?limit=200');
      const data = await resp.json();
      document.getElementById('status').innerText = `Loaded ${data.length} rows (showing up to 200)`;
      const tb = document.querySelector('#rows tbody');
      tb.innerHTML = '';
      for(const r of data){
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${r.file}</td><td>${r.row}</td><td>${(r.cand_name||'')}</td><td>${(r.state||'')}</td><td>${r.match_type}</td><td>${r.score||0}</td><td>${(r.candidates||[]).map(c=>`${c.cand_id}(${c.score})`).join('<br>')}</td><td></td>`;
        const actions = tr.querySelector('td:last-child');
        const accept = document.createElement('button'); accept.innerText='Accept/Map';
        accept.onclick = async ()=>{
          const mapped = prompt('Enter mapped candidate id (leave blank to cancel):','');
          if(mapped===null||mapped==='') return;
          const payload = {file:r.file,row:r.row,mapped_id:mapped,note:''};
          await fetch('/api/fec/save_mapping',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
          actions.innerText='Mapped';
        };
        const reject = document.createElement('button'); reject.innerText='Reject';
        reject.onclick = async ()=>{
          if(!confirm('Mark as rejected?')) return;
          const payload = {file:r.file,row:r.row,mapped_id:null,note:'rejected'};
          await fetch('/api/fec/save_mapping',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
          actions.innerText='Rejected';
        };
        actions.appendChild(accept); actions.appendChild(document.createTextNode(' ')); actions.appendChild(reject);
        tb.appendChild(tr);
      }
    }
    load();
    </script>
    </body></html>'''
    return html


@app.route('/api/fec/problem_rows')
def api_fec_problem_rows():
    try:
        min_score = int(request.args.get('min_score') or 70)
    except Exception:
        min_score = 70
    try:
        limit = int(request.args.get('limit') or 200)
    except Exception:
        limit = 200
    rows = _read_jsonl(REPORT_JSONL)
    out = []
    for r in rows:
        try:
            score = int(r.get('score') or 0)
        except Exception:
            score = 0
        if r.get('match_type') != 'exact' or score < min_score:
            out.append(r)
        if len(out) >= limit:
            break
    return jsonify(out)


@app.route('/api/fec/save_mapping', methods=['POST'])
def api_fec_save_mapping():
    data = request.get_json(force=True) or {}

    def _validate(name: str, val, *, allow_null: bool = False, max_len: int = 256):
        if val is None:
            if allow_null:
                return None
            raise ValueError(f"{name} is required")
        if isinstance(val, (int, float)):
            return val
        if not isinstance(val, str):
            raise ValueError(f"{name} must be a string")
        cleaned = val.strip()
        if not cleaned and not allow_null:
            raise ValueError(f"{name} is empty")
        if len(cleaned) > max_len:
            raise ValueError(f"{name} too long")
        return cleaned

    try:
        file_val = _validate('file', data.get('file'), allow_null=False, max_len=160)
        # simple safety check for filename characters
        if not _SAFE_FILTER_PATTERN.fullmatch(file_val):
            raise ValueError('file contains invalid characters')
        row_val = data.get('row')
        if isinstance(row_val, (int, float)):
            row_val = int(row_val)
        else:
            row_val = _validate('row', row_val, allow_null=False, max_len=64)
        mapped_id = data.get('mapped_id')
        if mapped_id is not None:
            mapped_id = _validate('mapped_id', mapped_id, allow_null=True, max_len=128)
        note = data.get('note')
        if note is not None:
            note = _validate('note', note, allow_null=True, max_len=1024)

        entry = {
            'file': file_val,
            'row': row_val,
            'mapped_id': mapped_id,
            'note': note,
            'ts': datetime.now(timezone.utc).isoformat()
        }

        # Ensure directory exists
        d = os.path.dirname(MAPPINGS_PATH)
        os.makedirs(d, exist_ok=True)

        # Atomic read/modify/write with rotation cap
        MAX_ENTRIES = int(os.environ.get('MAPPINGS_MAX_ENTRIES', '5000'))
        tmp_path = MAPPINGS_PATH + '.tmp'
        try:
            if os.path.exists(MAPPINGS_PATH):
                with open(MAPPINGS_PATH, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
                    if not isinstance(existing, list):
                        existing = []
            else:
                existing = []
        except Exception:
            existing = []

        existing.append(entry)
        # Trim oldest entries to MAX_ENTRIES
        if len(existing) > MAX_ENTRIES:
            existing = existing[-MAX_ENTRIES:]

        # Write to temp file then atomically replace
        with open(tmp_path, 'w', encoding='utf-8') as tf:
            json.dump(existing, tf, indent=2)
            tf.flush()
            os.fsync(tf.fileno())
        os.replace(tmp_path, MAPPINGS_PATH)

        return jsonify({'success': True, 'entry': entry})
    except ValueError as ve:
        return jsonify({'success': False, 'error': str(ve)}), 400
    except Exception as exc:
        logger.error({'level': 'ERROR', 'type': 'mappings', 'message': f'Failed to save mapping: {exc}', 'session_id': None})
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return jsonify({'success': False, 'error': str(exc)}), 500

# Heartbeat thread startup (idempotent)
if 'heartbeat_thread' not in globals() or not isinstance(globals().get('heartbeat_thread'), Thread) or not globals()['heartbeat_thread'].is_alive():
    if HEARTBEAT_ENABLED:
        heartbeat_thread = Thread(target=_heartbeat_loop, name="heartbeat-loop", daemon=True)
        heartbeat_thread.start()

# Proactively ensure tables at startup (non-fatal if fails)
try:
    ensure_db_tables()
except Exception:
    # Best-effort only during import; avoid failing import when DB unavailable
    pass

# Clean up old session log files on startup (keep only active or recent)
try:
    cleanup_old_log_files(LOG_DIR, session_manager.list_active_session_ids(), keep_days=7)
except Exception:
    # LOG_DIR or session_manager may not be initialized in some import contexts
    pass
        
# 7. Main Entrypoint
if __name__ == "__main__":
    try:
        port = int(os.environ.get("PORT", 5000))
        allow_unsafe = os.environ.get("SMART_ELECTIONS_ALLOW_UNSAFE_WERKZEUG", "").lower() in {"1", "true", "yes"}
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