from __future__ import annotations

import hmac
import logging
import os
import socket
from typing import Any, Callable, Tuple

# ============================================
# SocketIO Configuration: Threading Framework
# ============================================
# Using Python's native threading framework for reliable, maintainable async support.
# This avoids eventlet (deprecated) and provides stable, predictable behavior.
#
# WebSocket support: Enabled by default to reduce cert prompts in Optional mTLS mode.
# Socket.IO will try WebSocket first, then fall back to polling if needed.
# This minimizes TLS handshakes and browser cert prompts (one persistent connection vs multiple polling requests).
# ============================================

_SOCKETIO_ASYNC_MODE = "threading"

_SOCKETIO_ENGINE_OPTIONS = {
    "ping_interval": 10,
    "ping_timeout": 60,
    "allow_upgrades": True,  # Allow polling→WebSocket upgrade to minimize TLS handshakes
    "transports": ["polling", "websocket"],  # Try WebSocket first, fall back to polling
}

_SOCKETIO_CLIENT_TRANSPORTS = ["websocket", "polling"]  # Prefer WebSocket for persistent connection

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
    "upgrade": True,
    "pingInterval": int(_SOCKETIO_ENGINE_OPTIONS["ping_interval"] * 1000),
    "pingTimeout": int(_SOCKETIO_ENGINE_OPTIONS["ping_timeout"] * 1000),
}

# ElectionPulse application certificate-enforcement policy.
#
# This is intentionally separate from Azure App Service's TLS/client-cert
# mode. Azure is currently configured as Optional Interactive User.
AUTH_MODE = os.environ.get(
    "AUTH_MODE",
    "required",
).strip().lower()  # legacy compatibility only

_CERT_MODE_RAW = os.environ.get(
    "CERT_ENFORCEMENT_MODE",
    "",
).strip().lower()

if _CERT_MODE_RAW:
    CERT_ENFORCEMENT_MODE = _CERT_MODE_RAW
else:
    # Preserve historical ElectionPulse behavior while retiring ambiguous
    # use of the word "optional" inside application policy.
    CERT_ENFORCEMENT_MODE = (
        "disabled"
        if AUTH_MODE in {"disabled", "optional"}
        else "mutations"
    )

if CERT_ENFORCEMENT_MODE not in {
    "disabled",
    "mutations",
}:
    logging.warning(
        "Invalid CERT_ENFORCEMENT_MODE=%r; "
        "defaulting to 'mutations'.",
        CERT_ENFORCEMENT_MODE,
    )

    CERT_ENFORCEMENT_MODE = (
        "mutations"
    )

AZURE_CLIENT_CERT_MODE = (
    os.environ.get(
        "AZURE_CLIENT_CERT_MODE",
        "optional_interactive_user",
    )
    .strip()
    .lower()
    .replace("-", "_")
    .replace(" ", "_")
)


def _auth_mode_requires_certificate() -> bool:
    # Compatibility wrapper retained during Tranche 1.
    _configure_authority_policy_runtime()
    return _authority_policy._auth_mode_requires_certificate()

SOCKETIO_MESSAGE_QUEUE = os.environ.get("SOCKETIO_MESSAGE_QUEUE")
SOCKETIO_MESSAGE_CHANNEL = os.environ.get("SOCKETIO_MESSAGE_CHANNEL", "socketio")

# Cost/simplicity guard: Redis queueing is disabled by default unless explicitly allowed.
# This prevents stale env configuration from re-enabling external Redis usage unexpectedly.
_ALLOW_REDIS_QUEUE = os.environ.get("SOCKETIO_ALLOW_REDIS", "false").lower() in {"1", "true", "yes", "on"}
if SOCKETIO_MESSAGE_QUEUE and str(SOCKETIO_MESSAGE_QUEUE).strip().lower().startswith(("redis://", "rediss://")):
    if not _ALLOW_REDIS_QUEUE:
        logging.warning("SOCKETIO_MESSAGE_QUEUE points to Redis but SOCKETIO_ALLOW_REDIS is false; disabling Redis queue.")
        SOCKETIO_MESSAGE_QUEUE = None

# ---------------------------------------------------------------------------
# Multi-worker message queue: kombu + SQLAlchemy backend
# ---------------------------------------------------------------------------
# When SOCKETIO_USE_DB_QUEUE=true, the existing PostgreSQL database is used as
# the SocketIO message broker via kombu.  This is safe for single-instance
# deployments with GUNICORN_WORKERS > 1 and for Azure App Service when
# sticky-session routing is not available.
#
# Requires:  kombu>=5.6.2  (already in requirements.txt)
# ---------------------------------------------------------------------------
_USE_DB_QUEUE = os.environ.get("SOCKETIO_USE_DB_QUEUE", "false").lower() in {"1", "true"}
if _USE_DB_QUEUE and not SOCKETIO_MESSAGE_QUEUE:
    _db_url = os.environ.get("DATABASE_URL", "")
    # Only accepted relational-DB schemes are safe for kombu's SQLAlchemy transport.
    _KOMBU_SQLA_SCHEMES = (
        "postgresql://", "postgresql+",
        "mysql://", "mysql+",
        "sqlite://",
        "sqla+",
    )
    if _db_url and any(_db_url.startswith(s) for s in _KOMBU_SQLA_SCHEMES):
        if _db_url.startswith("sqla+"):
            # Already prefixed — pass through unchanged (kombu accepts sqla+dialect://...)
            SOCKETIO_MESSAGE_QUEUE = _db_url
        else:
            # Prepend the kombu SQLAlchemy transport prefix
            SOCKETIO_MESSAGE_QUEUE = "sqla+" + _db_url

import asyncio
import csv
import gzip
import io
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
from urllib.parse import urlparse, urlunparse

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
from webapp.parser.filename_parser import parse_filename_simple
from webapp.parser.routes import (
    create_data_framework_blueprint,
    create_election_data_blueprint,
    create_fec_data_assurance_blueprint,
    create_file_io_blueprint,
    create_health_blueprint,
    create_observability_blueprint,
    create_prometheus_metrics_blueprint,
    create_public_pages_blueprint,
    create_session_orchestration_blueprint,
    create_ui_navigation_blueprint,
    create_url_library_blueprint,
    create_utility_admin_blueprint,
)
from webapp.parser.socket_ballot_lens_orchestration import run_ballot_lens_socket_handler
from webapp.parser.url_parser import (
    parse_url_simple,
)
from webapp.parser.utils.cert_utils import extract_client_principal
from webapp.parser.utils.db_utils import SessionLocal, get_engine
from webapp.parser.utils.misc_utils import extract_url_and_label, load_processed_urls
from webapp.parser.utils.models import DataFrameworkPreviewCache
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
from webapp.parser.utils.url_ingestion import url_already_listed
from webapp.parser.web_pipeline import (
    cancel_processing,
    cancellation_manager,
    process_urls_for_web,
)

_DOWNLOAD_READY_SESSIONS: set[str] = set()
_DOWNLOAD_READY_LOCK = threading.Lock()
_DATA_FRAMEWORK_PREVIEW_CACHE_LOCK = threading.Lock()

_API_LATENCY_CACHE_LOCK = threading.Lock()
_API_LATENCY_CACHE: dict[str, dict[str, Any]] = {
    "warehouse_coverage": {"expires_at": 0.0, "payload": None},
    "states_counties": {"expires_at": 0.0, "payload": None},
}


def _int_env(name: str, default: int, *, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.environ.get(name, str(default))))
    except Exception:
        return default


WAREHOUSE_COVERAGE_CACHE_TTL_SEC = _int_env("WAREHOUSE_COVERAGE_CACHE_TTL_SEC", 60, minimum=1)
STATES_COUNTIES_CACHE_TTL_SEC = _int_env("STATES_COUNTIES_CACHE_TTL_SEC", 180, minimum=1)
API_LATENCY_WARN_MS = _int_env("API_LATENCY_WARN_MS", 1000, minimum=0)


def _clone_payload(payload: Any) -> Any:
    if payload is None:
        return None
    try:
        return orjson.loads(orjson.dumps(payload))
    except Exception:
        return payload


def _get_ttl_cache_payload(cache_key: str) -> Any:
    now = time.time()
    with _API_LATENCY_CACHE_LOCK:
        slot = _API_LATENCY_CACHE.get(cache_key)
        if not isinstance(slot, dict):
            return None
        if float(slot.get("expires_at") or 0.0) <= now:
            return None
        return _clone_payload(slot.get("payload"))


def _set_ttl_cache_payload(cache_key: str, payload: Any, ttl_sec: int) -> None:
    now = time.time()
    with _API_LATENCY_CACHE_LOCK:
        _API_LATENCY_CACHE[cache_key] = {
            "expires_at": now + max(1, int(ttl_sec)),
            "payload": _clone_payload(payload),
        }


def _log_endpoint_latency(endpoint_name: str, started_at: float, *, cache_hit: bool = False, context: dict[str, Any] | None = None) -> None:
    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    payload = {
        "level": "WARNING" if elapsed_ms >= API_LATENCY_WARN_MS else "DEBUG",
        "type": "performance",
        "message": f"Endpoint latency: {endpoint_name} {elapsed_ms}ms",
        "session_id": None,
        "endpoint": endpoint_name,
        "latency_ms": elapsed_ms,
        "cache_hit": bool(cache_hit),
    }
    if isinstance(context, dict):
        payload.update(context)

    if elapsed_ms >= API_LATENCY_WARN_MS:
        logger.warning(payload)
    else:
        logger.debug(payload)

def _emit_download_ready(session_id: str, payload: dict, *, force: bool = False) -> bool:
    if not session_id:
        return False
    with _DOWNLOAD_READY_LOCK:
        if session_id in _DOWNLOAD_READY_SESSIONS and not force:
            return False
        _DOWNLOAD_READY_SESSIONS.add(session_id)
    try:
        socketio.emit('download_ready', payload, room=session_id)
        return True
    except Exception:
        return False

# Health task security controls
ENABLE_HEALTH_TASKS = os.environ.get("ENABLE_HEALTH_TASKS", "false").lower() in {"1", "true", "yes"}
HEALTH_TASK_TOKEN = (os.environ.get("HEALTH_TASK_TOKEN") or "").strip()
GUARDED_INGESTION_KEY = (os.environ.get("GUARDED_INGESTION_KEY") or "").strip()

if not GUARDED_INGESTION_KEY:
    raise RuntimeError("GUARDED_INGESTION_KEY must be configured for ingestion security.")

if ENABLE_HEALTH_TASKS and not HEALTH_TASK_TOKEN:
    raise RuntimeError("HEALTH_TASK_TOKEN must be configured when ENABLE_HEALTH_TASKS is enabled.")

# Certificate gating for mutation endpoints (always enforced)
REQUIRE_CERT_FOR_MUTATIONS = True

# Local, non-DB monitoring log for DB usage/events
DB_MONITOR_FILE = LOG_DIR / "db_monitor.jsonl"
try:
    DB_MONITOR_FILE.touch(exist_ok=True)
except Exception:
    pass

# Flagged URL audit log (rotated daily, small caps)
FLAGGED_URL_SIZE_CAP = 5 * 1024 * 1024  # ~5MB per daily file
FLAGGED_URL_RETENTION_DAYS = 30
ENABLE_URL_INGESTION_AUDIT = os.environ.get("ENABLE_URL_INGESTION_AUDIT", "false").lower() in {"1", "true", "yes"}


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
    if not GUARDED_INGESTION_KEY:
        return False, "guard_key_missing"
    token_hdr = (request.headers.get("X-Guarded-Ingestion-Key") or request.headers.get("X-Guarded-Key") or "").strip()
    if token_hdr and hmac.compare_digest(token_hdr, GUARDED_INGESTION_KEY):
        return True, "guard_header"
    auth_hdr = (request.headers.get("Authorization") or "").strip()
    if auth_hdr.lower().startswith("bearer "):
        try:
            bearer = auth_hdr.split(None, 1)[1].strip()
            if hmac.compare_digest(bearer, GUARDED_INGESTION_KEY):
                return True, "guard_bearer"
        except Exception:
            pass
    return False, "guard_key_invalid"


def _request_wants_json() -> bool:
    accept = request.headers.get("Accept", "") or ""
    xhr = (request.headers.get("X-Requested-With", "") or "").lower() == "xmlhttprequest"
    sec_fetch_dest = (request.headers.get("Sec-Fetch-Dest", "") or "").lower()
    if (request.path or "").startswith("/api/"):
        return True
    if request.is_json or "application/json" in accept.lower():
        return True
    if xhr:
        return True
    if sec_fetch_dest == "empty":
        return True
    return False


DISALLOWED_AUTH_NEXT_PREFIXES = (
    "/auth/welcome",
    "/auth/challenge",
    "/api/auth/",
)

def sanitize_internal_next(raw_next: str | None, fallback: str = "/") -> str:
    if not raw_next or not isinstance(raw_next, str):
        return fallback
    raw = raw_next.strip()
    if len(raw) > 2048:
        return fallback
    if "\x00" in raw or any(ord(ch) < 32 for ch in raw):
        return fallback
    if raw.startswith("//") or raw.startswith("\\") or "\\" in raw:
        return fallback
    parsed = urlparse(raw)
    if parsed.scheme or parsed.netloc:
        return fallback
    if not raw.startswith("/"):
        return fallback
    if ".." in parsed.path.split("/"):
        return fallback
    if parsed.path and not parsed.path.startswith("/"):
        return fallback
    normalized_path = parsed.path or "/"
    if any(normalized_path.startswith(prefix) for prefix in DISALLOWED_AUTH_NEXT_PREFIXES):
        return fallback
    normalized_query = f"?{parsed.query}" if parsed.query else ""
    normalized_frag = f"#{parsed.fragment}" if parsed.fragment else ""
    return f"{normalized_path}{normalized_query}{normalized_frag}"


def _configure_authority_status_runtime():
    _authority_status.configure_runtime(
        AZURE_CLIENT_CERT_MODE=AZURE_CLIENT_CERT_MODE,
        CERT_ENFORCEMENT_MODE=CERT_ENFORCEMENT_MODE,
        DEPLOY_ENV=DEPLOY_ENV,
        REQUIRE_CERT_FOR_MUTATIONS=REQUIRE_CERT_FOR_MUTATIONS,
        _auth_mode_requires_certificate=_auth_mode_requires_certificate,
        _is_local_request=_is_local_request,
        _sanitize_cert_metadata_for_status=_sanitize_cert_metadata_for_status,
        api_auth_status=api_auth_status,
        get_request_principal=get_request_principal,
        jsonify=jsonify,
        request=request,
        resolve_session_id=resolve_session_id,
        sanitize_internal_next=sanitize_internal_next,
        url_for=url_for,
    )


def _sanitize_cert_metadata_for_status(cert_metadata: dict | None) -> dict:
    _configure_authority_status_runtime()
    return _authority_status._sanitize_cert_metadata_for_status(cert_metadata)


def _configure_authority_policy_runtime():
    _authority_policy.configure_runtime(
        AZURE_CLIENT_CERT_MODE=AZURE_CLIENT_CERT_MODE,
        CERT_ENFORCEMENT_MODE=CERT_ENFORCEMENT_MODE,
        DEPLOY_ENV=DEPLOY_ENV,
        REQUIRE_CERT_FOR_MUTATIONS=REQUIRE_CERT_FOR_MUTATIONS,
        _auth_mode_requires_certificate=_auth_mode_requires_certificate,
        _cert_required_response=_cert_required_response,
        _is_local_request=_is_local_request,
        _request_wants_json=_request_wants_json,
        emit=emit,
        get_request_principal=get_request_principal,
        hmac=hmac,
        jsonify=jsonify,
        normalize_log_obj=normalize_log_obj,
        os=os,
        redirect=redirect,
        request=request,
        sanitize_internal_next=sanitize_internal_next,
        session_manager=session_manager,
        url_for=url_for,
    )


def _cert_required_response(reason: str):
    _configure_authority_policy_runtime()
    return _authority_policy._cert_required_response(reason)

def _require_client_cert(reason: str):
    _configure_authority_policy_runtime()
    return _authority_policy._require_client_cert(reason)


def _require_cert_for_socket_action(action: str, session_id: str | None = None) -> bool:
    _configure_authority_policy_runtime()
    return _authority_policy._require_cert_for_socket_action(
        action,
        session_id=session_id,
    )


def _ingestion_audit_context(session_id: str | None = None) -> dict:
    try:
        forwarded_for = (request.headers.get("X-Forwarded-For") or "").split(",", 1)[0].strip()
        remote_addr = forwarded_for or (request.remote_addr or "")
        return {
            "path": request.path,
            "method": request.method,
            "host": request.host,
            "remote_addr": remote_addr,
            "user_agent": request.headers.get("User-Agent"),
            "session_id": session_id,
        }
    except Exception:
        return {"session_id": session_id}


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

# Warn at startup when running multiple workers without a message queue.
# In this configuration SocketIO events emitted by one worker will not reach
# clients connected to a different worker, causing missed real-time updates.
_gunicorn_workers = int(os.environ.get("GUNICORN_WORKERS", "1"))
if _gunicorn_workers > 1 and not SOCKETIO_MESSAGE_QUEUE:
    import warnings
    warnings.warn(
        f"GUNICORN_WORKERS={_gunicorn_workers} but SOCKETIO_MESSAGE_QUEUE is not set. "
        "Real-time events may not reach all connected clients. "
        "Set SOCKETIO_USE_DB_QUEUE=true to route events through the existing "
        "PostgreSQL database as the message broker.",
        RuntimeWarning,
        stacklevel=1,
    )

socketio = SocketIO(
    app,
    async_mode=_SOCKETIO_ASYNC_MODE,
    cors_allowed_origins=SOCKETIO_ALLOWED_ORIGINS,
    manage_session=False,
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

        metrics_handlers = {
            "metrics": metrics,
        }
        if TEST_METRICS_ROUTE_ENABLED:
            metrics_handlers["test_metrics_increment"] = test_metrics_increment

        app.config["_PROMETHEUS_METRICS_ROUTE_HANDLERS"] = metrics_handlers

        try:
            app.register_blueprint(
                create_prometheus_metrics_blueprint(
                    include_test_increment=bool(TEST_METRICS_ROUTE_ENABLED),
                )
            )
            logger.info({
                "level": "INFO",
                "type": "metrics",
                "message": "Prometheus metrics routes blueprint registered",
                "session_id": None,
            })
        except Exception as e:
            logger.warning({
                "level": "WARNING",
                "type": "metrics",
                "message": f"Failed to register Prometheus metrics blueprint: {e}",
                "session_id": None,
            })
    except Exception:
        try:
            logger.info({"level": "INFO", "type": "metrics", "message": "Prometheus client not available; /metrics disabled."})
        except Exception:
            pass

# If Prometheus is enabled, try to import the internal metrics module so counters are registered
if os.environ.get('ENABLE_PROMETHEUS', 'false').lower() in ('1', 'true', 'yes'):
    try:
        from webapp.parser.utils import metrics_prom as _metrics_prom
        _ = _metrics_prom  # keep linter happy; import triggers counter registration
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
        "minimum_tier": 2,
        "effect": "mixed_privileged_maintenance",
    },
    "manual_correction_auto": {
        "label": "Manual Correction (Auto)",
        "description": "Auto-accept new context entries without prompts using manual_correction_bot --auto.",
        "command": ["-m", "webapp.parser.health.manual_correction_bot", "--auto"],
        "danger": True,
        "minimum_tier": 2,
        "effect": "learned_context_update",
    },
    "manual_correction_enhanced": {
        "label": "Manual Correction (Enhanced)",
        "description": "Launch manual_correction_bot with enhanced review (interactive, slower but precise).",
        "command": ["-m", "webapp.parser.health.manual_correction_bot", "--enhanced"],
        "danger": True,
        "minimum_tier": 1,
        "effect": "review_assisted_context_update",
    },
    "retrain_table_models": {
        "label": "Retrain Table Models",
        "description": "Trigger retrain_table_structure_models to refresh structure detection weights.",
        "command": ["-m", "webapp.parser.health.retrain_table_structure_models"],
        "danger": True,
        "minimum_tier": 2,
        "effect": "model_training",
    },
    "scan_misaligned": {
        "label": "Scan Misaligned NER",
        "description": "Run scan_misaligned_ner to flag mismatched training samples before retraining.",
        "command": ["-m", "webapp.parser.health.scan_misaligned_ner"],
        "minimum_tier": 1,
        "effect": "training_data_diagnostics",
    },
    "log_cache_cleaner": {
        "label": "Log & Cache Cleaner",
        "description": "Execute log_cache_cleaner_bot to dedupe/cap JSONL files and watch sizes.",
        "command": ["-m", "webapp.parser.health.log_cache_cleaner_bot"],
        "minimum_tier": 2,
        "effect": "runtime_cache_maintenance",
    },
    "context_migration": {
        "label": "Context Migration",
        "description": "Run context_migration to sync historical context formats with the latest schema.",
        "command": ["-m", "webapp.parser.health.context_migration"],
        "minimum_tier": 2,
        "effect": "context_schema_migration",
    },
    "integrity_check_summary": {
        "label": "Integrity Check Summary",
        "description": "Stream Integrity_check findings for the current context library.",
        "command": ["-m", "webapp.parser.health.integrity_check_runner"],
        "minimum_tier": 1,
        "effect": "integrity_diagnostics",
    },
    "dataset_promotion_latest": {
        "label": "Dataset Promotion (Latest)",
        "description": "Promote the newest output folder into warehouse_election_results with guarded batching.",
        "command": ["-m", "webapp.parser.health.dataset_promotion"],
        "danger": True,
        "minimum_tier": 3,
        "effect": "canonical_promotion",
    },
}

_HEALTH_TASK_LOCK = threading.Lock()
_HEALTH_TASK_RUNS: dict[str, dict] = {}
_HEALTH_TASK_HISTORY_LIMIT = 20
_HEALTH_TASK_LOG_LIMIT = 20000


def _require_health_auth():
    """Guard health endpoints with enable flag and optional bearer token."""
    if not ENABLE_HEALTH_TASKS:
        return False, (jsonify({"error": "Health tasks disabled", "reason": "health_tasks_disabled"}), 403)

    if not HEALTH_TASK_TOKEN:
        return False, (jsonify({"error": "Health task token not configured", "reason": "health_token_missing"}), 503)

    auth_header = request.headers.get("Authorization", "") or ""
    token = None
    if auth_header.lower().startswith("bearer "):
        token = auth_header.split(" ", 1)[1].strip()
    if not token:
        token = request.args.get("token", "")
    if token and hmac.compare_digest(token, HEALTH_TASK_TOKEN):
        return True, None
    return False, (jsonify({
        "error": "Unauthorized",
        "reason": "health_token_mismatch",
        "auth_url": url_for("auth_welcome", next=request.url),
    }), 401)


def _health_auth_response(for_html: bool = False):
    allowed, resp = _require_health_auth()
    if not allowed and for_html:
        if isinstance(resp, tuple) and len(resp) > 1 and resp[1] in {401, 403, 503}:
            return redirect(url_for("auth_welcome", next=request.url))
    return None if allowed else resp


def _health_task_access_context(task_key: str) -> dict:
    definition = HEALTH_TASK_DEFINITIONS.get(
        task_key
    )

    if not definition:
        raise KeyError(
            task_key
        )

    principal, principal_source, _ = (
        get_request_principal()
    )

    from webapp.parser.utils.privilege_tiers import (
        PrivilegeTier,
        get_principal_tier,
    )

    actual_tier = get_principal_tier(
        principal,
        principal_source,
    )

    required_tier = PrivilegeTier(
        int(
            definition.get(
                "minimum_tier",
                int(
                    PrivilegeTier.ADMIN_REVIEWER
                ),
            )
        )
    )

    return {
        "allowed": bool(
            principal
            and tier_satisfies(actual_tier, required_tier)
        ),

        "principal_source": (
            principal_source
        ),

        "actual_tier": (
            actual_tier.name
        ),

        "actual_level": (
            int(actual_tier)
        ),

        "required_tier": (
            required_tier.name
        ),

        "required_level": (
            int(required_tier)
        ),

        "effect": str(
            definition.get(
                "effect"
            )
            or "unspecified"
        ),
    }


def _require_health_task_tier(task_key: str):
    context = _health_task_access_context(
        task_key
    )

    if context[
        "allowed"
    ]:
        return None

    return jsonify({
        "error": "Forbidden",

        "reason": (
            "insufficient_health_task_privilege"
        ),

        "task": task_key,

        "effect": context[
            "effect"
        ],

        "required_tier": context[
            "required_tier"
        ],

        "required_level": context[
            "required_level"
        ],

        "actual_tier": context[
            "actual_tier"
        ],

        "actual_level": context[
            "actual_level"
        ],

        "principal_source": context[
            "principal_source"
        ],
    }), 403


def _public_health_task_definitions() -> list[dict]:
    from webapp.parser.utils.privilege_tiers import (
        PrivilegeTier,
        get_principal_tier,
    )

    # This is UI/read-only metadata. Backend launch authorization is still
    # performed independently by _require_health_task_tier().
    principal, principal_source, _ = (
        get_request_principal()
    )

    actual_tier = get_principal_tier(
        principal,
        principal_source,
    )

    entries = []

    for key, meta in (
        HEALTH_TASK_DEFINITIONS.items()
    ):
        minimum_tier = PrivilegeTier(
            int(
                meta.get(
                    "minimum_tier",
                    int(
                        PrivilegeTier.ADMIN_REVIEWER
                    ),
                )
            )
        )

        tier_authorized = bool(
            principal
            and tier_satisfies(actual_tier, minimum_tier)
        )

        entries.append({
            "key": key,

            "label": meta[
                "label"
            ],

            "description": meta[
                "description"
            ],

            # Presentation metadata only.
            # NEVER use this value as an authorization decision.
            "danger": bool(
                meta.get(
                    "danger"
                )
            ),

            "minimum_tier": (
                minimum_tier.name
            ),

            "minimum_level": (
                int(minimum_tier)
            ),

            "effect": str(
                meta.get(
                    "effect"
                )
                or "unspecified"
            ),

            "current_tier": (
                actual_tier.name
            ),

            "current_level": (
                int(actual_tier)
            ),

            "tier_authorized": (
                tier_authorized
            ),
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

# Register Data Framework routes Blueprint
try:
    app.register_blueprint(create_data_framework_blueprint())
    logger.info({
        "level": "INFO",
        "type": "status",
        "message": "Data Framework routes blueprint registered",
        "session_id": None
    })
except Exception as e:
    logger.warning({
        "level": "WARNING",
        "type": "status",
        "message": f"Failed to register Data Framework routes blueprint: {e}",
        "session_id": None
    })

# Register Health routes Blueprint
try:
    app.register_blueprint(create_health_blueprint())
    logger.info({
        "level": "INFO",
        "type": "status",
        "message": "Health routes blueprint registered",
        "session_id": None
    })
except Exception as e:
    logger.warning({
        "level": "WARNING",
        "type": "status",
        "message": f"Failed to register Health routes blueprint: {e}",
        "session_id": None
    })

# Register URL library routes Blueprint
try:
    app.register_blueprint(create_url_library_blueprint())
    logger.info({
        "level": "INFO",
        "type": "status",
        "message": "URL library routes blueprint registered",
        "session_id": None
    })
except Exception as e:
    logger.warning({
        "level": "WARNING",
        "type": "status",
        "message": f"Failed to register URL library routes blueprint: {e}",
        "session_id": None
    })

# Register Election Data workflow routes Blueprint
try:
    app.register_blueprint(create_election_data_blueprint())
    logger.info({
        "level": "INFO",
        "type": "status",
        "message": "Election Data workflow routes blueprint registered",
        "session_id": None
    })
except Exception as e:
    logger.warning({
        "level": "WARNING",
        "type": "status",
        "message": f"Failed to register Election Data workflow routes blueprint: {e}",
        "session_id": None
    })

# Register Utility/Admin routes Blueprint
try:
    app.register_blueprint(create_utility_admin_blueprint())
    logger.info({
        "level": "INFO",
        "type": "status",
        "message": "Utility/Admin routes blueprint registered",
        "session_id": None
    })
except Exception as e:
    logger.warning({
        "level": "WARNING",
        "type": "status",
        "message": f"Failed to register Utility/Admin routes blueprint: {e}",
        "session_id": None
    })

# Register Observability routes Blueprint
try:
    app.register_blueprint(create_observability_blueprint())
    logger.info({
        "level": "INFO",
        "type": "status",
        "message": "Observability routes blueprint registered",
        "session_id": None
    })
except Exception as e:
    logger.warning({
        "level": "WARNING",
        "type": "status",
        "message": f"Failed to register Observability routes blueprint: {e}",
        "session_id": None
    })

# Register File I/O routes Blueprint
try:
    app.register_blueprint(create_file_io_blueprint())
    logger.info({
        "level": "INFO",
        "type": "status",
        "message": "File I/O routes blueprint registered",
        "session_id": None
    })
except Exception as e:
    logger.warning({
        "level": "WARNING",
        "type": "status",
        "message": f"Failed to register File I/O routes blueprint: {e}",
        "session_id": None
    })

# Register UI/Navigation routes Blueprint
try:
    app.register_blueprint(create_ui_navigation_blueprint())
    logger.info({
        "level": "INFO",
        "type": "status",
        "message": "UI/Navigation routes blueprint registered",
        "session_id": None
    })
except Exception as e:
    logger.warning({
        "level": "WARNING",
        "type": "status",
        "message": f"Failed to register UI/Navigation routes blueprint: {e}",
        "session_id": None
    })

# Register Public Pages routes Blueprint
try:
    app.register_blueprint(create_public_pages_blueprint())
    logger.info({
        "level": "INFO",
        "type": "status",
        "message": "Public Pages routes blueprint registered",
        "session_id": None
    })
except Exception as e:
    logger.warning({
        "level": "WARNING",
        "type": "status",
        "message": f"Failed to register Public Pages routes blueprint: {e}",
        "session_id": None
    })

# Register Session Orchestration routes Blueprint
try:
    app.register_blueprint(create_session_orchestration_blueprint())
    logger.info({
        "level": "INFO",
        "type": "status",
        "message": "Session Orchestration routes blueprint registered",
        "session_id": None
    })
except Exception as e:
    logger.warning({
        "level": "WARNING",
        "type": "status",
        "message": f"Failed to register Session Orchestration routes blueprint: {e}",
        "session_id": None
    })

# Register FEC/Data Assurance routes Blueprint
try:
    app.register_blueprint(create_fec_data_assurance_blueprint())
    logger.info({
        "level": "INFO",
        "type": "status",
        "message": "FEC/Data Assurance routes blueprint registered",
        "session_id": None
    })
except Exception as e:
    logger.warning({
        "level": "WARNING",
        "type": "status",
        "message": f"Failed to register FEC/Data Assurance routes blueprint: {e}",
        "session_id": None
    })


def _register_legacy_endpoint_aliases(app: Flask) -> None:
    """Backfill un-namespaced endpoint aliases for legacy url_for calls."""

    alias_map = {
        "index": "public_pages_routes.index",
        "auth_welcome": "public_pages_routes.auth_welcome",
        "ballot_lens": "public_pages_routes.ballot_lens",
        "data_framework": "data_framework_routes.data_framework",
        "health_dashboard": "health_routes.health_dashboard",
        "history": "file_io_routes.history",
        "quality_dashboard": "ui_navigation_routes.quality_dashboard",
        "api_warehouse_election_results": "election_data_routes.api_warehouse_election_results",
        "worklist": "public_pages_routes.worklist",
        "upload_to_uploads": "file_io_routes.upload_to_uploads",
        "upload_to_input": "file_io_routes.upload_to_input",
        "upload_to_output": "file_io_routes.upload_to_output",
        "site_webmanifest": "ui_navigation_routes.site_webmanifest",
    }

    endpoint_index: dict[str, list[str]] = {}
    for endpoint_name in app.view_functions.keys():
        if "." not in endpoint_name:
            continue
        suffix = endpoint_name.rsplit(".", 1)[-1]
        endpoint_index.setdefault(suffix, []).append(endpoint_name)

    for suffix, candidates in endpoint_index.items():
        if suffix in alias_map:
            continue
        if len(candidates) == 1:
            alias_map[suffix] = candidates[0]

    for legacy_endpoint, namespaced_endpoint in alias_map.items():
        if legacy_endpoint in app.view_functions:
            continue

        target_view = app.view_functions.get(namespaced_endpoint)
        if target_view is None:
            logger.debug({
                "level": "DEBUG",
                "type": "status",
                "message": f"Legacy endpoint alias skipped (target missing): {legacy_endpoint} -> {namespaced_endpoint}",
                "session_id": None,
            })
            continue

        target_rules = [rule for rule in app.url_map.iter_rules() if rule.endpoint == namespaced_endpoint]
        if not target_rules:
            logger.debug({
                "level": "DEBUG",
                "type": "status",
                "message": f"Legacy endpoint alias skipped (no rules): {legacy_endpoint} -> {namespaced_endpoint}",
                "session_id": None,
            })
            continue

        for target_rule in target_rules:
            methods = sorted(m for m in target_rule.methods if m not in {"HEAD", "OPTIONS"})
            app.add_url_rule(
                target_rule.rule,
                endpoint=legacy_endpoint,
                view_func=target_view,
                defaults=target_rule.defaults,
                methods=methods or None,
            )


_register_legacy_endpoint_aliases(app)

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
try:
    CERT_SESSION_CAP = max(0, int(os.environ.get("CERT_SESSION_CAP", "0")))
except ValueError:
    CERT_SESSION_CAP = 0

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


from webapp.parser.auth import context as _authority_context
from webapp.parser.auth import policy as _authority_policy
from webapp.parser.auth import status as _authority_status
from webapp.parser.auth.authorization import tier_satisfies
from webapp.parser.auth import socket_lifecycle as _socket_lifecycle


def _configure_authority_context_runtime():
    _authority_context.configure_runtime(
        ALLOW_AUTO_SESSION_REUSE=ALLOW_AUTO_SESSION_REUSE,
        ALLOW_DEV_NO_PRINCIPAL=ALLOW_DEV_NO_PRINCIPAL,
        CERT_SESSION_BINDING=CERT_SESSION_BINDING,
        CERT_SESSION_CAP=CERT_SESSION_CAP,
        ENABLE_FINGERPRINT_SESSION_RECOVERY=ENABLE_FINGERPRINT_SESSION_RECOVERY,
        _apply_auth_context=_apply_auth_context,
        _derive_auth_context=_derive_auth_context,
        _ensure_quick_copy_dir=_ensure_quick_copy_dir,
        _is_local_host=_is_local_host,
        _resolve_cert_session_id=_resolve_cert_session_id,
        _session_has_principal=_session_has_principal,
        client_fingerprint=client_fingerprint,
        extract_client_principal=extract_client_principal,
        get_request_principal=get_request_principal,
        request=request,
        safe_get=safe_get,
        safe_sid=safe_sid,
        secrets=secrets,
        session=session,
        session_manager=session_manager,
    )

def get_request_principal():
    """Return (principal, source, cert_metadata) preferring client cert, then SSO OID."""
    _configure_authority_context_runtime()
    return _authority_context.get_request_principal()



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
    _configure_authority_context_runtime()
    return _authority_context._resolve_cert_session_id(principal)



def _derive_auth_context(principal: str | None, principal_source: str | None) -> dict:
    _configure_authority_context_runtime()
    return _authority_context._derive_auth_context(principal, principal_source)



def _apply_auth_context(session_id: str, principal: str | None, principal_source: str | None) -> None:
    _configure_authority_context_runtime()
    return _authority_context._apply_auth_context(session_id, principal, principal_source)



def _session_has_principal(session_id: str) -> bool:
    _configure_authority_context_runtime()
    return _authority_context._session_has_principal(session_id)


def resolve_session_id(data=None, create_if_missing=True):
    _configure_authority_context_runtime()
    return _authority_context.resolve_session_id(data, create_if_missing)


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
        from webapp.parser.persistence.schema_bootstrap import ensure_application_schema_compat
        from webapp.parser.utils.db_utils import engine
        ensure_application_schema_compat(engine)
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
            if obj.get("type") == "run_summary":
                artifacts = obj.get("artifacts") if isinstance(obj.get("artifacts"), dict) else {}
                artifact_candidates: list[tuple[str, str]] = []
                if artifacts:
                    for kind in ("csv", "xlsx", "metadata", "other"):
                        paths = artifacts.get(kind)
                        if not isinstance(paths, list):
                            continue
                        for rel in paths:
                            if isinstance(rel, str) and rel.strip():
                                artifact_candidates.append((kind, rel.replace("\\", "/")))
                report_path = obj.get("report_path")
                if isinstance(report_path, str) and report_path:
                    try:
                        rel_path = os.path.relpath(report_path, str(OUTPUT_DIR)).replace("\\", "/")
                        artifact_candidates.append(("report", rel_path))
                    except Exception:
                        pass
                if artifact_candidates:
                    preferred_order = {"csv": 0, "xlsx": 1, "report": 2, "metadata": 3, "other": 4}
                    selected_kind, selected_rel = sorted(
                        artifact_candidates,
                        key=lambda item: (preferred_order.get(item[0], 99), item[1]),
                    )[0]
                    try:
                        _emit_download_ready(sid, {
                            "session_id": sid,
                            "filename": os.path.basename(selected_rel),
                            "output_path": selected_rel,
                            "root": "output",
                            "size": None,
                            "source": f"pipeline_{selected_kind}",
                            "artifacts": artifacts,
                        }, force=True)
                    except Exception:
                        pass
            msg_lower = str(obj.get("message", "")).lower()
            if "selector 'table" in msg_lower and "found" in msg_lower:
                _emit_download_ready(sid, {
                    "session_id": sid,
                    "filename": None,
                    "output_path": None,
                    "root": "output",
                    "size": None,
                    "source": "table_detected",
                })
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
    raw_url = input("Enter new URL to add: ").strip()
    if not raw_url:
        return
    if len(raw_url) > 2048 or any(ord(ch) < 32 for ch in raw_url):
        log_flagged_url({
            "event": "url_invalid",
            "url": raw_url,
            "reason": "invalid_chars_or_length",
            "source": "cli",
        })
        logger.warning({"level": "WARNING", "type": "status", "message": "URL too long or invalid.", "session_id": None})
        return
    url, lbl = extract_url_and_label(raw_url)
    if not url:
        logger.warning({"level": "WARNING", "type": "status", "message": "No valid http(s) URL found.", "session_id": None})
        return
    if len(url) > 2048:
        log_flagged_url({
            "event": "url_invalid",
            "url": url,
            "reason": "url_too_long",
            "source": "cli",
        })
        logger.warning({"level": "WARNING", "type": "status", "message": "URL too long.", "session_id": None})
        return
    parsed = urlparse(url)
    if parsed.username or parsed.password:
        log_flagged_url({
            "event": "url_invalid",
            "url": url,
            "reason": "credentials_in_url",
            "source": "cli",
        })
        logger.warning({"level": "WARNING", "type": "status", "message": "URLs with credentials are not allowed.", "session_id": None})
        return
    if parsed.fragment:
        url = urlunparse(parsed._replace(fragment=""))
        parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    allowed, reason = safe_validate_external_url(
        url,
        allowlist_suffixes=URL_ALLOWLIST_SUFFIXES,
        allowlist_hosts=URL_ALLOWLIST_HOSTS,
        enforce_allowlist=URL_ENFORCE_ALLOWLIST,
        block_private_ips=URL_BLOCK_PRIVATE_IPS,
    )
    if not allowed:
        log_flagged_url({
            "event": "url_blocked",
            "url": url,
            "reason": reason,
            "source": "cli",
        })
        logger.warning({"level": "WARNING", "type": "status", "message": f"URL blocked: {reason}", "session_id": None})
        return
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
            "event": "url_invalid",
            "url": url,
            "reason": "invalid_url",
            "source": "cli",
        })
        logger.warning({"level": "WARNING", "type": "status", "message": "Only http/https URLs with a host are accepted.", "session_id": None})
        return
    if any(tok in host for tok in suspicious_tokens):
        log_flagged_url({
            "event": "url_blocked",
            "url": url,
            "reason": "suspicious_host",
            "host": host,
            "source": "cli",
        })
        logger.warning({"level": "WARNING", "type": "status", "message": "Host requires manual review; URL logged for safety.", "session_id": None})
        return
    if url_already_listed(str(URL_LIST_FILE), url):
        logger.info({"level": "INFO", "type": "status", "message": f"[ALREADY PRESENT] {url}", "session_id": None})
        return
    with open(URL_LIST_FILE, "a", encoding="utf-8") as f:
        f.write(url + "\n")
    logger.info({
        "level": "INFO",
        "type": "status",
        "message": f"[ADDED] {url}",
        "session_id": None,
    })
    if ENABLE_URL_INGESTION_AUDIT:
        log_flagged_url({
            "event": "url_ingested",
            "url": url,
            "label": lbl,
            "source": "cli",
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
def index() -> str:
    return render_template("index.html")

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
            cert_resp = _require_client_cert("api_urls")
            if cert_resp:
                return cert_resp
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
            if len(raw_url) > 2048 or any(ord(ch) < 32 for ch in raw_url):
                log_flagged_url({
                    "event": "url_invalid",
                    "url": raw_url,
                    "reason": "invalid_chars_or_length",
                    **_ingestion_audit_context(safe_strip(safe_get(data, "session_id"))),
                })
                return jsonify({"success": False, "error": "URL too long or contains invalid characters."}), 400
            url, lbl = extract_url_and_label(raw_url)
            if not url:
                return jsonify({"success": False, "error": "No valid http(s) URL found."}), 400

            if len(url) > 2048:
                log_flagged_url({
                    "event": "url_invalid",
                    "url": url,
                    "reason": "url_too_long",
                    **_ingestion_audit_context(safe_strip(safe_get(data, "session_id"))),
                })
                return jsonify({"success": False, "error": "URL too long."}), 400

            parsed = urlparse(url)
            if parsed.username or parsed.password:
                log_flagged_url({
                    "event": "url_invalid",
                    "url": url,
                    "reason": "credentials_in_url",
                    **_ingestion_audit_context(safe_strip(safe_get(data, "session_id"))),
                })
                return jsonify({"success": False, "error": "URLs with credentials are not allowed."}), 400
            if parsed.fragment:
                url = urlunparse(parsed._replace(fragment=""))
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
                    "event": "url_blocked",
                    "url": url,
                    "reason": reason,
                    **_ingestion_audit_context(session_id),
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
                    "event": "url_invalid",
                    "url": url,
                    "reason": "invalid_url",
                    **_ingestion_audit_context(session_id),
                })
                return jsonify({"success": False, "error": "Only http/https URLs with a host are accepted."}), 400

            if any(tok in host for tok in suspicious_tokens):
                log_flagged_url({
                    "event": "url_blocked",
                    "url": url,
                    "reason": "suspicious_host",
                    "host": host,
                    **_ingestion_audit_context(session_id),
                })
                return jsonify({"success": False, "error": "Host requires manual review; URL logged for safety."}), 400

            if url_already_listed(urls_file, url):
                return jsonify({"success": True, "already_present": True})

            with open(urls_file, "a", encoding="utf-8") as f:
                f.write(url + "\n")
            if ENABLE_URL_INGESTION_AUDIT:
                log_flagged_url({
                    "event": "url_ingested",
                    "url": url,
                    "label": lbl,
                    "source": "api",
                    **_ingestion_audit_context(session_id),
                })
            return jsonify({"success": True})
    except Exception as exc:
        logger.error({"level": "ERROR", "type": "api", "message": f"api_urls GET/POST failed: {exc}", "session_id": None})
        return jsonify({"urls": [], "error": "internal"}), 500


@_rate_limit("60/minute")
def api_urls_parse():
    """
    Parse URL(s) into structured components for training.
    
    Request body:
        {"url": "https://..."} - Parse single URL
        {"urls": ["https://...", ...]} - Parse multiple URLs
        {"store": true} - Optionally store parsed results to training file
    
    Response:
        Single URL: {"success": true, "parsed": {...}}
        Multiple URLs: {"success": true, "parsed": [{...}, ...]}
    """
    try:
        data = request.get_json() or {}
        single_url = safe_strip(safe_get(data, "url", ""))
        urls_list = data.get("urls", [])
        store_results = data.get("store", False)

        # Determine if single or batch
        if single_url:
            urls_to_parse = [single_url]
            is_batch = False
        elif urls_list and isinstance(urls_list, list):
            urls_to_parse = [safe_strip(u) for u in urls_list if isinstance(u, str) and safe_strip(u)]
            is_batch = True
        else:
            return jsonify({"success": False, "error": "Provide 'url' or 'urls' parameter"}), 400

        if not urls_to_parse:
            return jsonify({"success": False, "error": "No valid URLs provided"}), 400

        # Parse URLs
        parsed_results = []
        for url in urls_to_parse:
            try:
                parsed = parse_url_simple(url)
                parsed_results.append(parsed)
            except Exception as parse_exc:
                logger.warning(f"Failed to parse URL {url}: {parse_exc}")
                parsed_results.append({
                    "url": url,
                    "error": str(parse_exc)
                })

        # Store to training file if requested
        if store_results:
            try:
                training_file = LOG_DIR / "parsed_urls_training.jsonl"
                with open(training_file, "a", encoding="utf-8") as f:
                    for result in parsed_results:
                        if "error" not in result:
                            f.write(orjson.dumps(result).decode("utf-8") + "\n")
            except Exception as store_exc:
                logger.error(f"Failed to store parsed URLs: {store_exc}")

        # Return results
        if is_batch:
            return jsonify({
                "success": True,
                "parsed": parsed_results,
                "count": len(parsed_results)
            }), 200
        else:
            return jsonify({
                "success": True,
                "parsed": parsed_results[0] if parsed_results else {}
            }), 200

    except Exception as exc:
        logger.error({"level": "ERROR", "type": "api", "message": f"api_urls_parse failed: {exc}", "session_id": None})
        return jsonify({"success": False, "error": "internal"}), 500


@_rate_limit("30/minute")
def api_urls_training_data():
    """
    Get parsed URL training data.
    
    Query parameters:
        limit: Max number of records (default: 100, max: 1000)
        offset: Skip first N records (default: 0)
        state: Filter by state code/name
        vendor: Filter by vendor hint
        has_county: Filter to URLs with county data (true/false)
    
    Response:
        {
            "success": true,
            "data": [...],
            "count": N,
            "total": M
        }
    """
    try:
        training_file = LOG_DIR / "parsed_urls_training.jsonl"

        # Parse query parameters
        limit = min(int(request.args.get("limit", 100)), 1000)
        offset = int(request.args.get("offset", 0))
        state_filter = safe_strip(request.args.get("state", "")).upper()
        vendor_filter = safe_strip(request.args.get("vendor", "")).lower()
        has_county_filter = request.args.get("has_county", "").lower() in {"true", "1", "yes"}

        if not training_file.exists():
            return jsonify({
                "success": True,
                "data": [],
                "count": 0,
                "total": 0,
                "message": "No training data available yet"
            }), 200

        # Read and filter data
        all_records = []
        with open(training_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = orjson.loads(line)

                    # Apply filters
                    if state_filter and record.get("state", "").upper() != state_filter:
                        continue
                    if vendor_filter and record.get("vendor_hint", "").lower() != vendor_filter:
                        continue
                    if has_county_filter and not record.get("county"):
                        continue

                    all_records.append(record)
                except Exception:
                    continue

        total = len(all_records)

        # Apply pagination
        paginated_records = all_records[offset:offset + limit]

        return jsonify({
            "success": True,
            "data": paginated_records,
            "count": len(paginated_records),
            "total": total,
            "offset": offset,
            "limit": limit
        }), 200

    except Exception as exc:
        logger.error({"level": "ERROR", "type": "api", "message": f"api_urls_training_data failed: {exc}", "session_id": None})
        return jsonify({"success": False, "error": "internal"}), 500


@_rate_limit("10/hour")
def api_urls_parse_all():
    """
    Parse all URLs from url_library (urls.txt) and store to training file.
    
    This is a batch operation that may take time for large URL lists.
    
    Response:
        {
            "success": true,
            "parsed_count": N,
            "failed_count": M,
            "training_file": "path/to/file.jsonl"
        }
    """
    try:
        urls_file = str(URL_LIST_FILE)
        if not os.path.exists(urls_file):
            return jsonify({
                "success": False,
                "error": "URL library file not found"
            }), 404

        # Read all URLs
        urls_to_parse = []
        with open(urls_file, "r", encoding="utf-8") as f:
            for raw in f:
                s = safe_strip(raw)
                if not s or s.startswith('#'):
                    continue
                u, _ = extract_url_and_label(s)
                if u:
                    urls_to_parse.append(u)

        if not urls_to_parse:
            return jsonify({
                "success": False,
                "error": "No URLs found in library"
            }), 404

        # Parse all URLs
        parsed_count = 0
        failed_count = 0
        training_file = LOG_DIR / "parsed_urls_training.jsonl"

        with open(training_file, "a", encoding="utf-8") as f:
            for url in urls_to_parse:
                try:
                    parsed = parse_url_simple(url)
                    f.write(orjson.dumps(parsed).decode("utf-8") + "\n")
                    parsed_count += 1
                except Exception as parse_exc:
                    logger.warning(f"Failed to parse URL {url}: {parse_exc}")
                    failed_count += 1

        return jsonify({
            "success": True,
            "parsed_count": parsed_count,
            "failed_count": failed_count,
            "training_file": str(training_file),
            "total_urls": len(urls_to_parse)
        }), 200

    except Exception as exc:
        logger.error({"level": "ERROR", "type": "api", "message": f"api_urls_parse_all failed: {exc}", "session_id": None})
        return jsonify({"success": False, "error": str(exc)}), 500


@_rate_limit("60/minute")
def api_filename_parse():
    """
    Parse filename(s) into structured components for metadata extraction.
    
    Similar to /api/urls/parse but for filenames.
    
    Request body:
        {"filename": "Alabama_Jefferson_2024.pdf"} - Parse single filename
        {"filenames": ["file1.pdf", ...]} - Parse multiple filenames
        {"store": true} - Optionally store parsed results to training file
    
    Response:
        Single: {"success": true, "parsed": {...}}
        Multiple: {"success": true, "parsed": [{...}, ...]}
    """
    try:
        data = request.get_json() or {}
        single_filename = safe_strip(safe_get(data, "filename", ""))
        filenames_list = data.get("filenames", [])
        store_results = data.get("store", False)

        # Determine if single or batch
        if single_filename:
            filenames_to_parse = [single_filename]
            is_batch = False
        elif filenames_list and isinstance(filenames_list, list):
            filenames_to_parse = [safe_strip(f) for f in filenames_list if isinstance(f, str) and safe_strip(f)]
            is_batch = True
        else:
            return jsonify({"success": False, "error": "Provide 'filename' or 'filenames' parameter"}), 400

        if not filenames_to_parse:
            return jsonify({"success": False, "error": "No valid filenames provided"}), 400

        # Parse filenames
        parsed_results = []
        for filename in filenames_to_parse:
            try:
                parsed = parse_filename_simple(filename)
                parsed_results.append(parsed)
            except Exception as parse_exc:
                logger.warning(f"Failed to parse filename {filename}: {parse_exc}")
                parsed_results.append({
                    "filename": filename,
                    "error": str(parse_exc)
                })

        # Store to training file if requested
        if store_results:
            try:
                training_file = LOG_DIR / "parsed_filenames_training.jsonl"
                with open(training_file, "a", encoding="utf-8") as f:
                    for result in parsed_results:
                        if "error" not in result:
                            f.write(orjson.dumps(result).decode("utf-8") + "\n")
            except Exception as store_exc:
                logger.error(f"Failed to store parsed filenames: {store_exc}")

        # Return results
        if is_batch:
            return jsonify({
                "success": True,
                "parsed": parsed_results,
                "count": len(parsed_results)
            }), 200
        else:
            return jsonify({
                "success": True,
                "parsed": parsed_results[0] if parsed_results else {}
            }), 200

    except Exception as exc:
        logger.error({"level": "ERROR", "type": "api", "message": f"api_filename_parse failed: {exc}", "session_id": None})
        return jsonify({"success": False, "error": "internal"}), 500


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


def _collect_url_reference_hint(url: str) -> dict:
    hint: dict[str, Any] = {
        "url": url,
        "parsed": {},
        "output_match": None,
        "warehouse": {"row_count": 0, "latest_election_date": None},
        "production": {"exists": False, "source": None},
    }

    try:
        parsed = parse_url_simple(url)
        if isinstance(parsed, dict):
            keep_keys = (
                "state",
                "county",
                "contest_type",
                "year",
                "office",
                "is_federal",
            )
            hint["parsed"] = {k: parsed.get(k) for k in keep_keys if parsed.get(k) not in (None, "")}
    except Exception:
        pass

    try:
        processed = load_processed_urls()
        entry = processed.get(url) if isinstance(processed, dict) else None
        if isinstance(entry, dict):
            match = _build_output_lookup_match(url, entry)
            if isinstance(match, dict):
                hint["output_match"] = {
                    "output_folder": match.get("output_folder"),
                    "contest": match.get("contest"),
                    "state": match.get("state"),
                    "county": match.get("county"),
                    "handler": match.get("handler"),
                }
    except Exception:
        pass

    try:
        from webapp.parser.utils.database_comparison import check_existing_finalized_data

        exists, source, metadata = check_existing_finalized_data(url, session_id=None)
        hint["production"] = {
            "exists": bool(exists),
            "source": source,
            "state": safe_get(metadata or {}, "state", None),
            "county": safe_get(metadata or {}, "county", None),
            "contest": safe_get(metadata or {}, "contest", None),
        }
    except Exception:
        pass

    try:
        ensure_db_tables()
        engine = get_engine()
        columns = _get_warehouse_columns(engine)
        if "source_url" in columns:
            with engine.connect() as conn:
                row = conn.execute(
                    text(
                        """
                        SELECT COUNT(*) AS row_count,
                               MAX(election_date) AS latest_election_date
                        FROM warehouse_election_results
                        WHERE source_url = :url
                        """
                    ),
                    {"url": url},
                ).mappings().first()
            if row:
                hint["warehouse"] = {
                    "row_count": int(row.get("row_count") or 0),
                    "latest_election_date": str(row.get("latest_election_date")) if row.get("latest_election_date") else None,
                }
    except Exception:
        pass

    return hint


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

def api_warehouse_coverage():
    """Return coverage summary: all state/county combinations in warehouse and those missing DL1/DL2"""
    started_at = time.perf_counter()
    cached_payload = _get_ttl_cache_payload("warehouse_coverage")
    if isinstance(cached_payload, dict):
        _log_endpoint_latency("/api/warehouse/coverage", started_at, cache_hit=True)
        return jsonify(cached_payload)

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
            pair_query = """
                SELECT state, county, COUNT(*) AS row_count
                FROM warehouse_election_results
                WHERE state IS NOT NULL AND county IS NOT NULL
                GROUP BY state, county
            """
            covered_rows = conn.execute(text(pair_query)).mappings().all()
            covered = [
                {"state": row["state"], "county": row["county"], "row_count": row["row_count"]}
                for row in covered_rows
            ]

            state_query = """
                SELECT DISTINCT state FROM warehouse_election_results 
                WHERE state IS NOT NULL
                ORDER BY state
            """
            all_states = [row[0] for row in conn.execute(text(state_query))]

            count_query = "SELECT COUNT(*) as cnt FROM warehouse_election_results"
            total_rows = conn.execute(text(count_query)).mappings().first()["cnt"]

        all_counties: dict[str, list[str]] = {state: [] for state in all_states}
        for row in covered_rows:
            state = row.get("state")
            county = row.get("county")
            if not isinstance(state, str) or not isinstance(county, str):
                continue
            all_counties.setdefault(state, []).append(county)

        payload = {
            "covered": covered,
            "all_states": all_states,
            "all_counties": all_counties,
            "total_rows": total_rows,
            "coverage_summary": {
                "total_states": len(all_states),
                "total_state_county_pairs": len(covered),
                "warehouse_healthy": total_rows > 0
            }
        }
        _set_ttl_cache_payload("warehouse_coverage", payload, WAREHOUSE_COVERAGE_CACHE_TTL_SEC)
        _log_endpoint_latency(
            "/api/warehouse/coverage",
            started_at,
            cache_hit=False,
            context={
                "total_rows": int(total_rows or 0),
                "states": len(all_states),
                "pairs": len(covered),
            },
        )
        return jsonify(payload)
    except Exception as exc:
        logger.warning({
            "level": "WARNING",
            "type": "db",
            "message": f"Warehouse coverage query failed: {exc}",
            "session_id": None,
        })
        return jsonify({"error": "Warehouse coverage query failed", "covered": [], "all_states": [], "all_counties": {}}), 500


app.config["_URL_LIBRARY_ROUTE_HANDLERS"] = {
    "api_urls": api_urls,
    "api_urls_parse": api_urls_parse,
    "api_urls_training_data": api_urls_training_data,
    "api_urls_parse_all": api_urls_parse_all,
    "api_filename_parse": api_filename_parse,
    "api_outputs_lookup": api_outputs_lookup,
    "api_warehouse_match": api_warehouse_match,
    "api_warehouse_export": api_warehouse_export,
    "api_warehouse_coverage": api_warehouse_coverage,
}

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
    scaffold = _collect_data_framework_scaffold(limit=limit * 2)  # Fetch extra for dedup
    items = []
    seen_keys = set()

    for record in scaffold.get("records", []):
        state = record.get("state") or ""
        county = record.get("county") or ""
        contest = record.get("contest") or ""
        updated_at = record.get("timestamp") or ""

        # Quality gates: skip low-quality entries
        if not state or state.lower() in ("unknown", "test"):
            continue
        if not contest or contest.lower() in ("unknown", "test"):
            continue
        # Skip entries with trivial row counts (likely test data)
        row_count = record.get("row_count")
        if row_count is not None and row_count < 5:
            continue

        # Deduplicate by (state, county, contest) - keep most recent
        dedup_key = (state.lower(), (county or "").lower(), contest.lower())
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)

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
            "row_count": row_count,
            "column_count": record.get("column_count"),
            "extraction_confidence": record.get("extraction_confidence"),
            "updated_at": updated_at,
            "source_url": record.get("source_url"),
        })

        if len(items) >= limit:
            break

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
        with _DATA_FRAMEWORK_PREVIEW_CACHE_LOCK:
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
        with _DATA_FRAMEWORK_PREVIEW_CACHE_LOCK:
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


def api_data_framework_warehouse_status():
    principal, _, _ = get_request_principal()
    if not principal and not ALLOW_DEV_NO_PRINCIPAL:
        return jsonify({"error": "Unauthorized"}), 403

    ensure_db_tables()
    engine = get_engine()
    state_filter = safe_strip(request.args.get("state", ""))
    state_filter = state_filter.lower() if state_filter else ""
    year_filter = safe_strip(request.args.get("year", ""))
    year_value = None
    if year_filter:
        try:
            year_value = int(year_filter)
        except Exception:
            year_value = None
    try:
        with engine.connect() as conn:
            exists = conn.execute(text("SELECT to_regclass('workflow.contests')")).scalar()
            if not exists:
                return jsonify({
                    "available": False,
                    "error": "workflow.contests table not found",
                    "expected_total": 0,
                    "missing_total": 0,
                    "by_priority": [],
                })

            states_result = conn.execute(text(
                """
                SELECT DISTINCT state
                FROM workflow.contests
                WHERE state IS NOT NULL
                ORDER BY state
                """
            ))
            available_states = [row[0] for row in states_result if row[0]]

            years_params = {}
            years_where = "WHERE state IS NOT NULL AND year IS NOT NULL"
            if state_filter:
                years_where += " AND LOWER(TRIM(state)) = :state"
                years_params["state"] = state_filter
            years_result = conn.execute(text(
                f"""
                SELECT DISTINCT year
                FROM workflow.contests
                {years_where}
                ORDER BY year DESC
                """
            ), years_params)
            available_years = [int(row[0]) for row in years_result if row[0] is not None]

            columns = _get_warehouse_columns(engine)
            has_precinct = "precinct" in columns
            year_filter_sql = "AND year = :year" if year_value is not None else ""
            warehouse_filter_sql = "WHERE state IS NOT NULL"
            if state_filter:
                warehouse_filter_sql += " AND LOWER(TRIM(state)) = :state"
            if year_value is not None:
                warehouse_filter_sql += " AND EXTRACT(YEAR FROM election_date)::int = :year"
            division_case = """
                CASE
                    WHEN county IS NULL OR TRIM(county) = '' THEN 'state'
                    WHEN LOWER(county) LIKE '%district%' OR LOWER(county) LIKE '%dist.%' THEN 'district'
                    ELSE 'county'
                END
            """
            if has_precinct:
                division_case = """
                    CASE
                        WHEN precinct IS NOT NULL AND TRIM(precinct) <> ''
                             AND LOWER(TRIM(precinct)) NOT IN ('all precincts','all') THEN 'precinct'
                        WHEN county IS NULL OR TRIM(county) = '' THEN 'state'
                        WHEN LOWER(county) LIKE '%district%' OR LOWER(county) LIKE '%dist.%' THEN 'district'
                        ELSE 'county'
                    END
                """
            division_params = {}
            division_where = "WHERE state IS NOT NULL"
            if state_filter:
                division_where += " AND LOWER(TRIM(state)) = :state"
                division_params["state"] = state_filter
            if year_value is not None:
                division_where += " AND EXTRACT(YEAR FROM election_date)::int = :year"
                division_params["year"] = year_value
            division_summary = conn.execute(text(
                f"""
                SELECT {division_case} AS division_type,
                       COUNT(*) AS rows
                FROM warehouse_election_results
                {division_where}
                GROUP BY division_type
                ORDER BY rows DESC
                """
            ), division_params).mappings().all()

            division_year_params = {}
            division_year_where = "WHERE state IS NOT NULL AND election_date IS NOT NULL"
            if state_filter:
                division_year_where += " AND LOWER(TRIM(state)) = :state"
                division_year_params["state"] = state_filter
            if year_value is not None:
                division_year_where += " AND EXTRACT(YEAR FROM election_date)::int = :year"
                division_year_params["year"] = year_value
            division_summary_by_year = conn.execute(text(
                f"""
                SELECT EXTRACT(YEAR FROM election_date)::int AS year,
                       {division_case} AS division_type,
                       COUNT(*) AS rows
                FROM warehouse_election_results
                {division_year_where}
                GROUP BY year, division_type
                ORDER BY year DESC, rows DESC
                """
            ), division_year_params).mappings().all()

            state_filter_sql = "AND LOWER(TRIM(state)) = :state" if state_filter else ""
            state_params = {"state": state_filter} if state_filter else {}
            if year_value is not None:
                state_params["year"] = year_value

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
                    {state_filter_sql}
                    {year_filter_sql}
                ),
                warehouse AS (
                    SELECT
                        LOWER(TRIM(state)) AS state,
                        LOWER(COALESCE(TRIM(county), '')) AS county,
                        LOWER(TRIM(contest)) AS contest,
                        EXTRACT(YEAR FROM election_date)::int AS year,
                        COUNT(*) AS rows
                    FROM warehouse_election_results
                    {warehouse_filter_sql}
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
                """.format(
                    state_filter_sql=state_filter_sql,
                    year_filter_sql=year_filter_sql,
                    warehouse_filter_sql=warehouse_filter_sql,
                )
            ), state_params).mappings().first()

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
                    {state_filter_sql}
                    {year_filter_sql}
                ),
                warehouse AS (
                    SELECT
                        LOWER(TRIM(state)) AS state,
                        LOWER(COALESCE(TRIM(county), '')) AS county,
                        LOWER(TRIM(contest)) AS contest,
                        EXTRACT(YEAR FROM election_date)::int AS year,
                        COUNT(*) AS rows
                    FROM warehouse_election_results
                    {warehouse_filter_sql}
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
                """.format(
                    state_filter_sql=state_filter_sql,
                    year_filter_sql=year_filter_sql,
                    warehouse_filter_sql=warehouse_filter_sql,
                )
            ), state_params).mappings().all()

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
                    {state_filter_sql}
                    {year_filter_sql}
                ),
                warehouse AS (
                    SELECT
                        LOWER(TRIM(state)) AS state,
                        LOWER(COALESCE(TRIM(county), '')) AS county,
                        LOWER(TRIM(contest)) AS contest,
                        EXTRACT(YEAR FROM election_date)::int AS year,
                        COUNT(*) AS rows
                    FROM warehouse_election_results
                    {warehouse_filter_sql}
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
                """.format(
                    state_filter_sql=state_filter_sql,
                    year_filter_sql=year_filter_sql,
                    warehouse_filter_sql=warehouse_filter_sql,
                )
            ), state_params).mappings().all()

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
                    {state_filter_sql}
                    {year_filter_sql}
                ),
                warehouse AS (
                    SELECT
                        LOWER(TRIM(state)) AS state,
                        LOWER(COALESCE(TRIM(county), '')) AS county,
                        LOWER(TRIM(contest)) AS contest,
                        EXTRACT(YEAR FROM election_date)::int AS year,
                        COUNT(*) AS rows
                    FROM warehouse_election_results
                    {warehouse_filter_sql}
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
                """.format(
                    state_filter_sql=state_filter_sql,
                    year_filter_sql=year_filter_sql,
                    warehouse_filter_sql=warehouse_filter_sql,
                )
            ), state_params).mappings().all()

        payload = {
            "expected_total": int(summary.get("expected_total") or 0) if summary else 0,
            "missing_total": int(summary.get("missing_total") or 0) if summary else 0,
            "by_priority": [dict(row) for row in by_priority],
            "by_status": [dict(row) for row in by_status],
            "sample_missing": [dict(row) for row in sample],
            "states": available_states,
            "selected_state": state_filter or None,
            "selected_year": year_value,
            "available_years": available_years,
            "division_summary": [
                {"type": row["division_type"], "rows": int(row["rows"])}
                for row in division_summary
            ],
            "division_summary_by_year": [
                {
                    "year": int(row["year"]) if row["year"] is not None else None,
                    "type": row["division_type"],
                    "rows": int(row["rows"]),
                }
                for row in division_summary_by_year
            ],
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


def health_dashboard():
    allowed, resp = _require_health_auth()
    health_controls_enabled = bool(allowed)
    health_state_reason = None
    if not allowed:
        status_code = None
        reason_code = None
        if isinstance(resp, tuple) and len(resp) > 1:
            status_code = resp[1]
            payload = resp[0]
            try:
                if hasattr(payload, "get_json"):
                    payload_json = payload.get_json(silent=True) or {}
                    reason_code = payload_json.get("reason")
            except Exception:
                reason_code = None
        if status_code == 403:
            health_state_reason = reason_code or "health_tasks_disabled"
        elif status_code == 503:
            health_state_reason = reason_code or "health_token_missing"
        elif status_code == 401:
            health_state_reason = reason_code or "health_token_mismatch"
        else:
            return resp

    runtime_hints = {
        "async_mode": _SOCKETIO_ASYNC_MODE,
        "async_framework": "threading (native Python)",
        "eventlet_deprecated": "disabled",
        "transports": _SOCKETIO_CLIENT_TRANSPORTS,
        "deploy_env": DEPLOY_ENV or "local",
    }
    return render_template(
        "health_dashboard.html",
        task_definitions=_public_health_task_definitions(),
        runtime_hints=runtime_hints,
        socketio_client_config=SOCKETIO_CLIENT_CONFIG,
        initial_tasks=_get_health_tasks() if health_controls_enabled else [],
        health_controls_enabled=health_controls_enabled,
        health_state_reason=health_state_reason,
        health_auth_url=url_for("auth_welcome", next=request.url),
    )


def api_list_health_tasks():
    auth_error = _health_auth_response()
    if auth_error:
        return auth_error
    return jsonify({"tasks": _get_health_tasks()})


def api_start_health_task():
    # HEALTH_TASK_TOKEN authenticates access to the control-plane
    # endpoint. It is intentionally NOT a user privilege grant.
    auth_error = (
        _health_auth_response()
    )

    if auth_error:
        return auth_error

    # Preserve the current certificate requirement independently
    # from the privilege-tier decision below.
    cert_resp = _require_client_cert(
        "health_task_start"
    )

    if cert_resp:
        return cert_resp

    data = request.get_json(
        silent=True
    ) or {}

    task_key = str(
        data.get(
            "task"
        )
        or ""
    ).strip()

    if not task_key:
        return jsonify({
            "error": "Task key required."
        }), 400

    if (
        task_key
        not in HEALTH_TASK_DEFINITIONS
    ):
        return jsonify({
            "error": "Unknown task."
        }), 404

    # Authorization is determined by the request principal and
    # ElectionPulse privilege tier, not by possession of the
    # health token alone.
    privilege_error = (
        _require_health_task_tier(
            task_key
        )
    )

    if privilege_error:
        return privilege_error

    record = _launch_health_task(
        task_key
    )

    return jsonify({
        "task": record
    })

def api_health_task_detail(task_id: str):
    auth_error = _health_auth_response()
    if auth_error:
        return auth_error
    record = _get_health_task(task_id)
    if not record:
        return jsonify({"error": "Task not found."}), 404
    return jsonify({"task": record})


def api_health_socket_test():
    """Diagnostic endpoint for testing Socket.IO multi-instance propagation.
    Does not require client cert since it's a test/diagnostic tool.
    """
    auth_error = _health_auth_response()
    if auth_error:
        return auth_error
    # No cert required for diagnostic endpoint
    try:
        test_id = secrets.token_hex(6)
        payload = {
            "test_id": test_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "instance_id": os.environ.get("WEBSITE_INSTANCE_ID") or os.environ.get("WEBSITES_INSTANCE_ID"),
            "hostname": socket.gethostname(),
        }
        # emit() broadcasts to all clients by default when called from a route
        socketio.emit("health_socket_test", payload)
        logger.info({
            "level": "INFO",
            "type": "health",
            "message": "health_socket_test broadcast",
            "session_id": None,
            "test_id": test_id,
            "instance_id": payload.get("instance_id"),
            "hostname": payload.get("hostname"),
        })
        return jsonify({"ok": True, "payload": payload})
    except Exception as e:
        logger.error({
            "level": "ERROR",
            "type": "health",
            "message": "health_socket_test failed",
            "error": str(e),
            "error_type": type(e).__name__,
        })
        return jsonify({"error": f"Socket test failed: {str(e)}"}), 500


app.config["_DATA_FRAMEWORK_ROUTE_HANDLERS"] = {
    "data_framework": data_framework,
    "api_data_framework_preview": api_data_framework_preview,
    "api_data_framework_scaffold": api_data_framework_scaffold,
    "api_data_framework_scaffold_csv": api_data_framework_scaffold_csv,
    "api_data_framework_curated": api_data_framework_curated,
    "api_data_framework_warehouse_status": api_data_framework_warehouse_status,
    "api_data_framework_exports": api_data_framework_exports,
}


app.config["_HEALTH_ROUTE_HANDLERS"] = {
    "health_dashboard": health_dashboard,
    "api_list_health_tasks": api_list_health_tasks,
    "api_start_health_task": api_start_health_task,
    "api_health_task_detail": api_health_task_detail,
    "api_health_socket_test": api_health_socket_test,
}


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


app.config["_SESSION_ORCHESTRATION_ROUTE_HANDLERS"] = {
    "get_session_enums": get_session_enums,
    "test_ui_prompt": test_ui_prompt,
}

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

def api_list_dir_compat():
    return api_fs_list()

def api_fs_mkdir():
    import os
    cert_resp = _require_client_cert("fs_mkdir")
    if cert_resp:
        return cert_resp
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

def api_fs_delete():
    cert_resp = _require_client_cert("fs_delete")
    if cert_resp:
        return cert_resp
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

def api_quick_copy():
    cert_resp = _require_client_cert("quick_copy")
    if cert_resp:
        return cert_resp
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

def api_quick_copy_clear():
    cert_resp = _require_client_cert("quick_copy_clear")
    if cert_resp:
        return cert_resp
    data = request.get_json(silent=True) or {}
    session_id = resolve_session_id(data or {}, create_if_missing=False)
    if not session_id:
        return jsonify({"success": False, "error": "session_id required"}), 400
    _cleanup_quick_copy_dir(session_id)
    return jsonify({"success": True})

def download_fs():
    """Enhanced filesystem download with integrity verification."""

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

def robots_txt():
    return "User-agent: *\nDisallow: /", 200, {"Content-Type": "text/plain"}


# Serve a small set of well-known app-specific files that some browsers/devtools request
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

def delete_input_file(filename) -> str:
    cert_resp = _require_client_cert("delete_input")
    if cert_resp:
        return cert_resp
    file_path = os.path.join(INPUT_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        flash(f"Deleted '{filename}' from input folder.", "success")
    else:
        flash(f"File '{filename}' not found in input folder.", "danger")
    return redirect(request.referrer or url_for("ballot_lens"))

def delete_output_file(filename) -> str:
    cert_resp = _require_client_cert("delete_output")
    if cert_resp:
        return cert_resp
    file_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        flash(f"Deleted '{filename}' from output folder.", "success")
    else:
        flash(f"File '{filename}' not found in output folder.", "danger")
    return redirect(request.referrer or url_for("ballot_lens"))

def delete_upload_file(filename) -> str:
    cert_resp = _require_client_cert("delete_uploads")
    if cert_resp:
        return cert_resp
    file_path = os.path.join(UPLOADS_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        flash(f"Deleted '{filename}' from uploads folder.", "success")
    else:
        flash(f"File '{filename}' not found in uploads folder.", "danger")
    return redirect(request.referrer or url_for("ballot_lens"))

def download_input_file(filename) -> str:
    return send_from_directory(INPUT_DIR, filename, as_attachment=True)

def download_output_file(filename) -> str:
    """Enhanced download with integrity verification and cache deduplication."""

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

def download_upload_file(filename) -> str:
    return send_from_directory(UPLOADS_DIR, filename, as_attachment=True)

def ballot_lens():
    try:
        qp_source = safe_lower(request.args.get("source", "")) if request.method == "GET" else ""
        if qp_source in {"input", "uploads"}:
            session['manual_source_pref'] = qp_source
        if request.method == "POST" and "data_file" in request.files:
            file = request.files.get("data_file")
            cert_resp = _require_client_cert("ballot_lens_upload")
            if cert_resp:
                return cert_resp
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

def ballot_lens_modern():
    """Redirect to consolidated modern interface at /ballot_lens."""
    return redirect(url_for("ballot_lens"))


def worklist():
    """
    Render the SMART Elections Worklist interface.
    
    Displays live Google Sheets worklist with DL1/DL2 standardization,
    Pre-QC, QC1, QC2, and production status tracking.
    """
    try:
        return render_template(
            "worklist.html",
            static_version=os.environ.get("STATIC_VERSION", "v1"),
        )
    except Exception:
        import traceback
        print(traceback.format_exc())
        return "Internal Server Error", 500


def api_validate_urls():
    """
    Validate URLs against existing finalized data.
    
    Checks each URL against:
    - Google Sheets finalized data
    - Warehouse database (warehouse_election_results)
    - verified_datasets table
    
    Request JSON:
        {"urls": ["url1", "url2", ...]}
    
    Response JSON:
        {
            "success": true,
            "results": [
                {
                    "url": "...",
                    "exists": bool,
                    "source": "google_sheets" | "warehouse" | null,
                    "metadata": {...}
                }
            ]
        }
    """
    try:
        from webapp.parser.utils.database_comparison import check_existing_finalized_data

        data = request.get_json()
        if not data or not isinstance(data.get("urls"), list):
            return jsonify({"error": "Invalid request: 'urls' array required"}), 400

        urls = data["urls"]
        if len(urls) > 100:
            return jsonify({"error": "Too many URLs: maximum 100 per request"}), 400

        results = []
        for url in urls:
            if not isinstance(url, str):
                continue

            url = url.strip()
            if not url:
                continue

            # Check if data exists
            data_exists, data_source, metadata = check_existing_finalized_data(
                url,
                session_id=None
            )

            results.append({
                "url": url,
                "exists": data_exists,
                "source": data_source,
                "metadata": metadata or {}
            })

        return jsonify({
            "success": True,
            "results": results
        }), 200

    except Exception as e:
        logger.error({
            "level": "ERROR",
            "type": "api",
            "message": f"URL validation error: {e}",
            "session_id": None,
        })
        return jsonify({"error": f"Validation failed: {str(e)}"}), 500


def api_url_status():
    """
    Query processed URLs with filtering and production status.
    Uses status reconciliation to show TRUE status (parser + worklist).
    
    Query parameters:
        status: Filter by RECONCILED status (success|fail|error|partial|cancelled|pending|skipped_data_exists|production|qc_complete|etc)
        parser_status: Filter by raw parser status only
        state: Filter by state
        county: Filter by county
        from_date: Filter by date (YYYY-MM-DD)
        to_date: Filter by date (YYYY-MM-DD)
        limit: Max results (default: 100, max: 1000)
        offset: Pagination offset
        hide_pii: Remove personal names (default: true)
    
    Returns:
        {
            "success": true,
            "total": 213,
            "filtered": 42,
            "entries": [
                {
                    "url": "...",
                    "label": "AZ President 2024",
                    "parser_status": "pending",
                    "worklist_status": "PROD Loaded",
                    "canonical_status": "production",
                    "status_info": {"icon": "📦", "label": "Production", "badge_class": "success", "authority": "worklist"},
                    "in_production": true,
                    "production_source": "google_sheets",
                    "last_processed": "2026-01-30 12:30:45",
                    "state": "Arizona",
                    "county": null
                }
            ],
            "status_breakdown": {"production": 100, "pending": 50, "fail": 6},
            "canonical_statuses": ["production", "pending", "fail"]
        }
    """
    started_at = time.perf_counter()
    try:
        from datetime import datetime

        from webapp.parser.config import URL_LIST_FILE
        from webapp.parser.utils.database_comparison import check_existing_finalized_data
        from webapp.parser.utils.misc_utils import extract_url_and_label, load_processed_urls
        from webapp.parser.utils.status_reconciliation import StatusReconciliation

        def _truthy(value: str | None) -> bool:
            return str(value or "").strip().lower() in {"1", "true", "yes", "on"}

        # Parse query parameters
        status_filter = request.args.get("status")  # Reconciled status
        parser_status_filter = request.args.get("parser_status")  # Raw parser status
        state_filter = request.args.get("state")
        county_filter = request.args.get("county")
        from_date = request.args.get("from_date")
        to_date = request.args.get("to_date")
        hide_pii = request.args.get("hide_pii", "true").lower() != "false"
        profile_enabled = _truthy(request.args.get("profile"))
        default_include_production = (DEPLOY_ENV or "local").lower() in {"production", "azure", "staging", "ci"}
        include_production_checks = _truthy(
            request.args.get(
                "include_production",
                os.environ.get("URL_STATUS_INCLUDE_PRODUCTION", "true" if default_include_production else "false"),
            )
        )
        try:
            limit = min(max(int(request.args.get("limit", 100)), 1), 1000)
            offset = max(int(request.args.get("offset", 0)), 0)
        except Exception:
            return jsonify({"error": "Invalid limit/offset values"}), 400

        budget_ms = max(
            100,
            int(os.environ.get("URL_STATUS_PRODUCTION_BUDGET_MS", "12000")),
        )
        requested_budget_ms = request.args.get("production_check_budget_ms")
        if requested_budget_ms is not None:
            try:
                budget_ms = max(100, int(requested_budget_ms))
            except Exception:
                return jsonify({"error": "Invalid production_check_budget_ms value"}), 400

        cache_ttl_sec = max(1, int(os.environ.get("URL_STATUS_CACHE_TTL_SEC", "30")))
        cache_key = (
            "url_status::"
            f"status={status_filter or ''}|parser={parser_status_filter or ''}|state={state_filter or ''}|"
            f"county={county_filter or ''}|from={from_date or ''}|to={to_date or ''}|hide_pii={hide_pii}|"
            f"budget={budget_ms}|include_production={include_production_checks}"
        )
        if not profile_enabled:
            cached_payload = _get_ttl_cache_payload(cache_key)
            if isinstance(cached_payload, dict):
                _log_endpoint_latency(
                    "/api/url_status",
                    started_at,
                    cache_hit=True,
                    context={"filtered": cached_payload.get("filtered"), "total": cached_payload.get("total")},
                )
                entries_full = cached_payload.get("entries") if isinstance(cached_payload.get("entries"), list) else []
                entries_page = entries_full[offset:offset + limit]
                response_payload = {
                    "success": True,
                    "total": cached_payload.get("total", 0),
                    "filtered": len(entries_full),
                    "limit": limit,
                    "offset": offset,
                    "entries": entries_page,
                    "status_breakdown": cached_payload.get("status_breakdown", {}),
                    "canonical_statuses": cached_payload.get("canonical_statuses", []),
                }
                return jsonify(response_payload), 200

        # Load URLs from urls.txt
        urls_list = []
        if URL_LIST_FILE.exists():
            with open(URL_LIST_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    url, label = extract_url_and_label(line, allowlist_bypass=True)
                    if url:
                        urls_list.append((url, label or line))
                    else:
                        parts = line.split('\t')
                        if len(parts) >= 7 and parts[6].startswith('http'):
                            url = parts[6].strip()
                            label = f"{parts[0]} {parts[1]} {parts[2]}" if parts[0] != 'TBD' else line
                            urls_list.append((url, label))

        processed_map = load_processed_urls()

        entries = []
        status_breakdown = {}
        canonical_statuses = set()

        profile_data = {
            "include_production_checks": include_production_checks,
            "production_checks_requested": 0,
            "production_checks_executed": 0,
            "production_checks_skipped_budget": 0,
            "production_check_elapsed_ms_total": 0,
            "slowest_production_checks": [],
            "budget_ms": budget_ms,
        }

        production_budget_started = time.perf_counter()
        budget_exhausted = False

        for url, label in urls_list:
            processed_entry = processed_map.get(url)
            parser_status = None
            last_processed = None
            state = None
            county = None

            if processed_entry:
                parser_status = processed_entry.get('status')
                last_processed = processed_entry.get('timestamp')
                state = processed_entry.get('state')
                county = processed_entry.get('county')
            else:
                parser_status = 'pending'

            in_production = False
            prod_source = None
            prod_metadata = None
            worklist_status = None

            if include_production_checks:
                profile_data["production_checks_requested"] += 1
                elapsed_budget_ms = int((time.perf_counter() - production_budget_started) * 1000)
                if elapsed_budget_ms < budget_ms:
                    check_started = time.perf_counter()
                    try:
                        in_production, prod_source, prod_metadata = check_existing_finalized_data(url)
                    except Exception as check_exc:
                        logger.warning({
                            "level": "WARNING",
                            "type": "api",
                            "message": f"Production status check failed for URL: {check_exc}",
                            "session_id": None,
                            "url": url,
                        })
                        in_production, prod_source, prod_metadata = False, None, None
                    check_elapsed_ms = int((time.perf_counter() - check_started) * 1000)
                    profile_data["production_checks_executed"] += 1
                    profile_data["production_check_elapsed_ms_total"] += check_elapsed_ms
                    if check_elapsed_ms >= 500:
                        profile_data["slowest_production_checks"].append({
                            "url": url,
                            "elapsed_ms": check_elapsed_ms,
                        })
                else:
                    budget_exhausted = True
                    profile_data["production_checks_skipped_budget"] += 1

            if prod_metadata and isinstance(prod_metadata, dict):
                state = state or prod_metadata.get('state')
                county = county or prod_metadata.get('county')
                worklist_status = prod_metadata.get('status')

            canonical_status, status_info = StatusReconciliation.reconcile(
                url=url,
                parser_status=parser_status,
                worklist_status=worklist_status,
                production_source=prod_source,
                last_processed=last_processed,
            )

            if status_filter and canonical_status != status_filter:
                continue
            if parser_status_filter and parser_status != parser_status_filter:
                continue
            if state_filter and state != state_filter:
                continue
            if county_filter and county != county_filter:
                continue
            if from_date and last_processed:
                try:
                    proc_date = datetime.strptime(last_processed, '%Y-%m-%d %H:%M:%S')
                    filter_date = datetime.strptime(from_date, '%Y-%m-%d')
                    if proc_date < filter_date:
                        continue
                except Exception:
                    pass
            if to_date and last_processed:
                try:
                    proc_date = datetime.strptime(last_processed, '%Y-%m-%d %H:%M:%S')
                    filter_date = datetime.strptime(to_date, '%Y-%m-%d')
                    if proc_date > filter_date:
                        continue
                except Exception:
                    pass

            entry = {
                'url': url,
                'label': label,
                'parser_status': parser_status,
                'worklist_status': worklist_status,
                'canonical_status': canonical_status,
                'status_info': status_info,
                'in_production': in_production,
                'production_source': prod_source,
                'last_processed': last_processed,
                'state': state,
                'county': county,
            }

            if hide_pii and 'metadata' in entry:
                entry['metadata'] = {
                    k: v
                    for k, v in entry['metadata'].items()
                    if k not in ['assigned_to', 'dl1', 'dl2']
                }

            entries.append(entry)
            status_breakdown[canonical_status] = status_breakdown.get(canonical_status, 0) + 1
            canonical_statuses.add(canonical_status)

        canonical_statuses_sorted = sorted(
            canonical_statuses,
            key=lambda s: StatusReconciliation.get_status_priority(s),
        )

        profile_data["slowest_production_checks"] = sorted(
            profile_data["slowest_production_checks"],
            key=lambda item: item.get("elapsed_ms", 0),
            reverse=True,
        )[:10]
        profile_data["budget_exhausted"] = budget_exhausted
        profile_data["total_elapsed_ms"] = int((time.perf_counter() - started_at) * 1000)

        total_filtered = len(entries)
        entries_page = entries[offset:offset + limit]
        response_payload = {
            "success": True,
            "total": len(urls_list),
            "filtered": total_filtered,
            "limit": limit,
            "offset": offset,
            "entries": entries_page,
            "status_breakdown": status_breakdown,
            "canonical_statuses": canonical_statuses_sorted,
        }

        if profile_enabled:
            response_payload["profile"] = profile_data

        if not profile_enabled:
            cache_payload = {
                "total": len(urls_list),
                "entries": entries,
                "status_breakdown": status_breakdown,
                "canonical_statuses": canonical_statuses_sorted,
            }
            _set_ttl_cache_payload(cache_key, cache_payload, cache_ttl_sec)

        _log_endpoint_latency(
            "/api/url_status",
            started_at,
            context={
                "total": len(urls_list),
                "filtered": total_filtered,
                "checks_executed": profile_data["production_checks_executed"],
                "checks_skipped_budget": profile_data["production_checks_skipped_budget"],
                "budget_ms": budget_ms,
                "profile": profile_enabled,
                "include_production": include_production_checks,
            },
        )
        return jsonify(response_payload), 200

    except Exception as e:
        logger.error({
            "level": "ERROR",
            "type": "api",
            "message": f"URL status query error: {e}",
            "session_id": None,
        })
        _log_endpoint_latency("/api/url_status", started_at, context={"failed": True})
        return jsonify({"error": f"Query failed: {str(e)}"}), 500


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

def quality_dashboard():
    """Quality metrics visualization dashboard."""
    return render_template("quality_dashboard.html")


def _probe_module_version(module_name: str) -> dict[str, Any]:
    try:
        spec = importlib.util.find_spec(module_name)
    except Exception:
        return {"installed": False, "version": None, "notes": ["Failed to inspect module spec."]}

    if spec is None:
        return {"installed": False, "version": None, "notes": []}

    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", None)
        if version is None and hasattr(module, "version"):
            version = module.version
    except Exception as exc:
        return {"installed": False, "version": None, "notes": [f"Import failed: {exc}"]}

    return {"installed": True, "version": str(version) if version is not None else "unknown", "notes": []}


def _probe_binary(binary_name: str, env_var: str | None = None, is_path_dir: bool = False) -> dict[str, Any]:
    env_value = os.environ.get(env_var) if env_var else None
    result = {
        "binary_name": binary_name,
        "env_var": env_var,
        "env_value": env_value,
        "resolved_path": None,
        "available": False,
        "version": None,
        "notes": [],
    }

    candidate = None
    if env_value:
        env_path = os.path.expanduser(env_value)
        if is_path_dir and os.path.isdir(env_path):
            candidate = os.path.join(env_path, binary_name)
            if os.name == "nt":
                candidate = os.path.join(env_path, f"{binary_name}.exe")
        elif os.path.isfile(env_path) and os.access(env_path, os.X_OK):
            candidate = env_path
        elif not is_path_dir and shutil.which(env_path):
            candidate = shutil.which(env_path)
        else:
            result["notes"].append(f"Environment variable {env_var} points to a non-executable path.")

    if not candidate:
        candidate = shutil.which(binary_name)

    if candidate:
        result["resolved_path"] = os.path.abspath(candidate)
        result["available"] = True
        try:
            version_proc = subprocess.run(
                [result["resolved_path"], "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                env=os.environ,
            )
            result["version"] = (version_proc.stdout or version_proc.stderr).strip().splitlines()[0]
        except Exception as exc:
            result["notes"].append(f"Version probe failed: {exc}")
    else:
        result["notes"].append(f"Could not locate binary {binary_name}.")

    return result


def _build_ocr_diagnostics() -> dict[str, Any]:
    diagnostics = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "deploy_env": os.environ.get("DEPLOY_ENV", "local"),
        "enable_ocr": bool(os.environ.get("ENABLE_OCR", "true").lower() in ("1", "true", "yes")),
        "enable_ocr_force": bool(os.environ.get("ENABLE_OCR_FORCE", "false").lower() in ("1", "true", "yes")),
        "tesseract_cmd_env": os.environ.get("TESSERACT_CMD"),
        "poppler_path_env": os.environ.get("POPPLER_PATH"),
        "pytesseract": _probe_module_version("pytesseract"),
        "pdf2image": _probe_module_version("pdf2image"),
        "fitz": _probe_module_version("fitz"),
        "tesseract": _probe_binary("tesseract", env_var="TESSERACT_CMD"),
        "pdftoppm": _probe_binary("pdftoppm", env_var="POPPLER_PATH", is_path_dir=True),
    }
    return diagnostics


def ocr_diagnostics():
    """OCR environment and runtime diagnostics page."""
    diagnostics = _build_ocr_diagnostics()
    return render_template("ocr_diagnostics.html", diagnostics=diagnostics)


def api_ocr_diagnostics():
    """Return OCR environment diagnostics as JSON for observability."""
    diagnostics = _build_ocr_diagnostics()
    return jsonify(diagnostics)


def _load_integrity_trends() -> tuple[list[dict[str, Any]], str, bool]:
    """Load integrity trends from primary file locations with cached fallback.

    Returns: (trends, source_path, from_cache)
    """

    repo_root = Path(__file__).resolve().parent.parent

    candidate_paths = [
        repo_root / "tools" / "debug_headless_output" / "context_digest_trends.json",
        repo_root / "webapp" / "parser" / "Context_Integration" / "Context_Library" / "log" / "context_digest_trends.json",
    ]
    cache_path = repo_root / "tools" / "tmp" / "integrity_trends_last.json"

    def _normalize_trends(raw: Any) -> list[dict[str, Any]]:
        if isinstance(raw, dict):
            raw = raw.get("trends", [])
        if not isinstance(raw, list):
            return []

        normalized: list[dict[str, Any]] = []
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            if "timestamp" not in entry and entry.get("generated_at"):
                entry = {**entry, "timestamp": entry.get("generated_at")}
            normalized.append(entry)
        return normalized

    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            trends = _normalize_trends(loaded)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(trends), encoding="utf-8")
            return trends, str(path), False
        except Exception as exc:
            logger.warning(f"Failed reading integrity trends from {path}: {exc}")

    if cache_path.exists():
        try:
            cached_raw = json.loads(cache_path.read_text(encoding="utf-8"))
            trends = _normalize_trends(cached_raw)
            return trends, str(cache_path), True
        except Exception as exc:
            logger.warning(f"Failed reading integrity trends cache: {exc}")

    return [], "", False

def api_integrity_trends():
    """API endpoint for context digest trends data."""
    try:
        trends, source, from_cache = _load_integrity_trends()
        return jsonify({
            "trends": trends,
            "count": len(trends),
            "source": source,
            "from_cache": from_cache,
        })
    except Exception as e:
        logger.error(f"Failed to load integrity trends: {e}")
        return jsonify({"error": "Failed to load trends", "trends": [], "count": 0}), 500

def api_integrity_signal():
    """API endpoint to compute integrity signal with custom thresholds."""

    # Get custom thresholds from request
    thresholds = request.get_json() or {}
    conf_drop_threshold = thresholds.get("confDropThreshold", 0.08)
    unknown_spike_threshold = thresholds.get("unknownSpikeThreshold", 0.1)
    review_spike_threshold = thresholds.get("reviewSpikeThreshold", 5.0)
    baseline_window = int(thresholds.get("baselineWindow", 30) or 30)
    recent_window = int(thresholds.get("recentWindow", 5) or 5)

    try:
        trends, source, from_cache = _load_integrity_trends()
        if len(trends) < 2:
            return jsonify({
                "signal": {
                    "status": "insufficient_data",
                    "entry_count": len(trends),
                    "alerts": [],
                },
                "source": source,
                "from_cache": from_cache,
            })

        # Lazy import to avoid circular deps
        from tools.analyze_context_digest_trends import compute_integrity_signal

        repo_root = Path(__file__).resolve().parent.parent
        trend_file_path = repo_root / "tools" / "tmp" / "integrity_trends_working.json"
        trend_file_path.parent.mkdir(parents=True, exist_ok=True)
        trend_file_path.write_text(json.dumps(trends), encoding="utf-8")
        signal = compute_integrity_signal(
            trend_file=trend_file_path,
            window=baseline_window,
            recent=recent_window,
            conf_drop_threshold=conf_drop_threshold,
            unknown_spike_threshold=unknown_spike_threshold,
            review_spike_threshold=review_spike_threshold,
        )

        return jsonify({"signal": signal, "source": source, "from_cache": from_cache})
    except Exception as e:
        logger.error(f"Failed to compute integrity signal: {e}")
        return jsonify({"error": "Failed to compute signal", "signal": {"status": "error", "alerts": []}}), 500

def api_integrity_export():
    """API endpoint to export integrity report as JSON."""

    try:
        trends, source, from_cache = _load_integrity_trends()
        if not trends:
            return jsonify({"error": "No trends data available"}), 404

        # Compute signal
        from tools.analyze_context_digest_trends import compute_integrity_signal
        repo_root = Path(__file__).resolve().parent.parent
        trend_file_path = repo_root / "tools" / "tmp" / "integrity_trends_working.json"
        trend_file_path.parent.mkdir(parents=True, exist_ok=True)
        trend_file_path.write_text(json.dumps(trends), encoding="utf-8")
        signal = compute_integrity_signal(trend_file=trend_file_path)

        # Build report
        report = {
            "exported_at": datetime.now().isoformat(),
            "source": source,
            "from_cache": from_cache,
            "thresholds": {
                "confDropThreshold": 0.08,
                "unknownSpikeThreshold": 0.1,
                "reviewSpikeThreshold": 5.0,
                "baselineWindow": 30,
                "recentWindow": 5
            },
            "signal": signal,
            "trends": trends
        }

        return jsonify(report)
    except Exception as e:
        logger.error(f"Failed to export integrity report: {e}")
        return jsonify({"error": "Failed to export report"}), 500

def url_status_dashboard():
    """URL processing status dashboard with production comparison."""
    return render_template("url_status_dashboard.html")

def quick_reference_page():
    """Serve the Quick Reference guide with CSP-friendly headers and static CSS."""
    return render_template("quick_reference.html")


app.config["_UI_NAVIGATION_ROUTE_HANDLERS"] = {
    "favicon": favicon,
    "robots_txt": robots_txt,
    "serve_well_known_appspecific": serve_well_known_appspecific,
    "site_webmanifest": site_webmanifest,
    "quality_dashboard": quality_dashboard,
    "url_status_dashboard": url_status_dashboard,
    "quick_reference_page": quick_reference_page,
}

def api_quality_metrics():
    """API endpoint for quality metrics data."""

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


def api_ml_usage():
    """Return runtime ML/NLP usage telemetry and recent model activity."""
    try:
        from webapp.parser.utils.ml_telemetry import get_ml_telemetry_snapshot

        include_recent = request.args.get("include_recent", "true").strip().lower() != "false"
        try:
            limit = int(request.args.get("limit", 120))
        except Exception:
            limit = 120
        limit = max(1, min(limit, 500))

        snapshot = get_ml_telemetry_snapshot(include_recent=include_recent, limit=limit)
        return jsonify({"success": True, "telemetry": snapshot}), 200
    except Exception as exc:
        logger.error({
            "level": "ERROR",
            "type": "api",
            "message": f"ML usage telemetry query failed: {exc}",
            "session_id": None,
        })
        return jsonify({"success": False, "error": str(exc)}), 500


def api_ml_pipeline_profile():
    """Return pipeline-ingestion profile for ML/NLP tuning visibility."""
    try:
        from webapp.parser.utils.ml_pipeline_profile import get_ml_pipeline_profile

        profile = get_ml_pipeline_profile()
        return jsonify({"success": True, "profile": profile}), 200
    except Exception as exc:
        logger.error({
            "level": "ERROR",
            "type": "api",
            "message": f"ML pipeline profile query failed: {exc}",
            "session_id": None,
        })
        return jsonify({"success": False, "error": str(exc)}), 500


def api_ml_vocab_alignment():
    """Return alias->canonical alignment health for election vocab mappings."""
    try:
        from webapp.parser.utils.ml_vocab_alignment import get_vocab_alignment_report

        try:
            sample_limit = int(request.args.get("sample_limit", 25))
        except Exception:
            sample_limit = 25
        report = get_vocab_alignment_report(sample_limit=sample_limit)
        return jsonify({"success": True, "alignment": report}), 200
    except Exception as exc:
        logger.error({
            "level": "ERROR",
            "type": "api",
            "message": f"ML vocab alignment query failed: {exc}",
            "session_id": None,
        })
        return jsonify({"success": False, "error": str(exc)}), 500


def api_ml_vocab_alignment_suggestions():
    """Return top canonical target suggestions for unresolved entity alias mappings."""
    try:
        from webapp.parser.utils.ml_vocab_alignment import get_vocab_alignment_suggestions

        try:
            limit = int(request.args.get("limit", 50))
        except Exception:
            limit = 50
        try:
            min_score = float(request.args.get("min_score", 0.45))
        except Exception:
            min_score = 0.45

        payload = get_vocab_alignment_suggestions(limit=limit, min_score=min_score)
        return jsonify({"success": True, "suggestions": payload}), 200
    except Exception as exc:
        logger.error({
            "level": "ERROR",
            "type": "api",
            "message": f"ML vocab alignment suggestions query failed: {exc}",
            "session_id": None,
        })
        return jsonify({"success": False, "error": str(exc)}), 500


def api_ml_vocab_alignment_suggestions_export():
    """Export unresolved-entity suggestion rows as JSON or CSV for bulk review."""
    try:
        from webapp.parser.utils.ml_vocab_alignment import get_vocab_alignment_suggestions
        import csv
        import io
        import re

        try:
            limit = int(request.args.get("limit", 50))
        except Exception:
            limit = 50
        try:
            min_score = float(request.args.get("min_score", 0.45))
        except Exception:
            min_score = 0.45
        try:
            high_confidence_min_score = float(request.args.get("high_confidence_min_score", 0.75))
        except Exception:
            high_confidence_min_score = 0.75
        high_confidence_min_score = max(0.0, min(high_confidence_min_score, 1.0))
        try:
            apply_ready_min_score = float(request.args.get("apply_ready_min_score", 0.90))
        except Exception:
            apply_ready_min_score = 0.90
        apply_ready_min_score = max(0.0, min(apply_ready_min_score, 1.0))

        def _normalize_for_apply(value: str | None) -> str:
            text = (value or "").strip().lower()
            text = text.replace("&", " and ")
            text = text.replace("/", " ")
            text = text.replace("-", " ")
            text = re.sub(r"[^a-z0-9\s]", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text

        export_format = (request.args.get("format", "json") or "json").strip().lower()
        if export_format not in {"json", "csv"}:
            return jsonify({"success": False, "error": "Invalid format. Use 'json' or 'csv'."}), 400

        export_mode = (request.args.get("export_mode", "all") or "all").strip().lower()
        if export_mode not in {"all", "high_confidence", "apply_ready"}:
            return jsonify({
                "success": False,
                "error": "Invalid export_mode. Use 'all', 'high_confidence', or 'apply_ready'.",
            }), 400

        payload = get_vocab_alignment_suggestions(limit=limit, min_score=min_score)
        rows_all = payload.get("suggestions") or []

        if export_mode == "high_confidence":
            rows = [
                row for row in rows_all
                if isinstance(row, dict)
                and isinstance(row.get("best_score"), (int, float))
                and float(row.get("best_score", 0.0)) >= high_confidence_min_score
            ]
        elif export_mode == "apply_ready":
            rows = []
            for row in rows_all:
                if not isinstance(row, dict):
                    continue
                best_score = row.get("best_score")
                if not isinstance(best_score, (int, float)) or float(best_score) < apply_ready_min_score:
                    continue

                options = row.get("suggestions") or []
                if not options or not isinstance(options[0], dict):
                    continue
                top_canonical = options[0].get("canonical")
                current_target = row.get("target")

                if _normalize_for_apply(top_canonical) == _normalize_for_apply(current_target):
                    rows.append(row)
        else:
            rows = rows_all

        payload_filtered = dict(payload)
        payload_filtered["suggestions"] = rows
        payload_filtered["suggestion_count"] = len(rows)
        payload_filtered["export_mode"] = export_mode
        payload_filtered["high_confidence_min_score"] = high_confidence_min_score
        payload_filtered["apply_ready_min_score"] = apply_ready_min_score
        payload_filtered["total_suggestion_count_before_filter"] = len(rows_all)

        if export_mode == "high_confidence":
            file_suffix = "_high_confidence"
        elif export_mode == "apply_ready":
            file_suffix = "_apply_ready"
        else:
            file_suffix = ""

        if export_format == "json":
            body = {
                "success": True,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "suggestions": payload_filtered,
            }
            response = jsonify(body)
            response.headers["Content-Disposition"] = f"attachment; filename=ml_vocab_alignment_suggestions{file_suffix}.json"
            return response, 200

        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow([
            "file",
            "alias",
            "current_target",
            "best_score",
            "suggestion_1",
            "score_1",
            "suggestion_2",
            "score_2",
            "suggestion_3",
            "score_3",
        ])
        for row in rows:
            options = row.get("suggestions") or []
            option_1 = options[0] if len(options) > 0 else {}
            option_2 = options[1] if len(options) > 1 else {}
            option_3 = options[2] if len(options) > 2 else {}
            writer.writerow([
                row.get("file"),
                row.get("alias"),
                row.get("target"),
                row.get("best_score"),
                option_1.get("canonical"),
                option_1.get("score"),
                option_2.get("canonical"),
                option_2.get("score"),
                option_3.get("canonical"),
                option_3.get("score"),
            ])

        return Response(
            buffer.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment; filename=ml_vocab_alignment_suggestions{file_suffix}.csv"},
        )
    except Exception as exc:
        logger.error({
            "level": "ERROR",
            "type": "api",
            "message": f"ML vocab alignment suggestions export failed: {exc}",
            "session_id": None,
        })
        return jsonify({"success": False, "error": str(exc)}), 500


def api_preingest_url_glimpse():
    """Capture pre-ingest screenshot/DOM glimpse and return quick risk flags for a URL."""
    started_at = time.perf_counter()
    try:
        from webapp.parser.utils.url_glimpse import build_glimpse_risk_flags, capture_url_glimpse

        url = (request.args.get("url") or "").strip()
        if not url:
            return jsonify({"success": False, "error": "Missing required query param: url"}), 400

        allowlist_bypass_requested = (request.args.get("allowlist_bypass", "false") or "").strip().lower() in {"1", "true", "yes", "on"}
        allowlist_bypass = bool(allowlist_bypass_requested and _is_local_request())

        allowed, reason = safe_validate_external_url(url, allowlist_bypass=allowlist_bypass)
        if not allowed:
            return jsonify({"success": False, "error": "url_not_allowed", "reason": reason}), 400

        try:
            timeout_ms = int(request.args.get("timeout_ms", 45000))
        except Exception:
            timeout_ms = 45000
        try:
            wait_ms = int(request.args.get("wait_ms", 1800))
        except Exception:
            wait_ms = 1800

        timeout_ms = max(5000, min(timeout_ms, 120000))
        wait_ms = max(0, min(wait_ms, 10000))

        out_dir = Path("tools") / "debug_headless_output"
        glimpse = capture_url_glimpse(url, out_dir=out_dir, timeout_ms=timeout_ms, wait_ms=wait_ms)
        risk = build_glimpse_risk_flags(glimpse)

        response_payload = {
            "success": True,
            "url": url,
            "allowlist_bypass": allowlist_bypass,
            "risk": risk,
            "artifacts": {
                "json_report": glimpse.get("json_report"),
                "html_snapshot": glimpse.get("html_snapshot"),
                "screenshot": glimpse.get("screenshot"),
            },
            "glimpse": {
                "status": glimpse.get("status"),
                "content_type": glimpse.get("content_type"),
                "title": glimpse.get("title"),
                "table_count": glimpse.get("table_count"),
                "table_rows_estimate": glimpse.get("table_rows_estimate"),
                "has_election_terms": glimpse.get("has_election_terms"),
                "error": glimpse.get("error"),
            },
        }

        _log_endpoint_latency(
            "/api/preingest_url_glimpse",
            started_at,
            context={
                "url": url,
                "risk_level": risk.get("risk_level"),
                "tables_found": risk.get("tables_found"),
                "has_election_terms": risk.get("has_election_terms"),
            },
        )
        return jsonify(response_payload), 200
    except Exception as exc:
        logger.error({
            "level": "ERROR",
            "type": "api",
            "message": f"Pre-ingest URL glimpse failed: {exc}",
            "session_id": None,
        })
        _log_endpoint_latency("/api/preingest_url_glimpse", started_at, context={"failed": True})
        return jsonify({"success": False, "error": str(exc)}), 500


app.config["_OBSERVABILITY_ROUTE_HANDLERS"] = {
    "api_integrity_trends": api_integrity_trends,
    "api_integrity_signal": api_integrity_signal,
    "api_integrity_export": api_integrity_export,
    "api_quality_metrics": api_quality_metrics,
    "api_ml_usage": api_ml_usage,
    "api_ml_pipeline_profile": api_ml_pipeline_profile,
    "api_ml_vocab_alignment": api_ml_vocab_alignment,
    "api_ml_vocab_alignment_suggestions": api_ml_vocab_alignment_suggestions,
    "api_ml_vocab_alignment_suggestions_export": api_ml_vocab_alignment_suggestions_export,
    "api_preingest_url_glimpse": api_preingest_url_glimpse,
    "api_ocr_diagnostics": api_ocr_diagnostics,
}

def api_auth_status():
    _configure_authority_status_runtime()
    return _authority_status.api_auth_status()

def api_auth_certificate_info():
    _configure_authority_status_runtime()
    return _authority_status.api_auth_certificate_info()


def api_route_wrapper_monitor_snapshot():
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


def api_route_wrapper_monitor_snapshot():
    cert_resp = _require_client_cert("route_wrapper_monitor_snapshot")
    if cert_resp:
        return cert_resp

    principal, principal_source, _ = get_request_principal()
    if not principal:
        return jsonify({"error": "Unauthorized"}), 401

    monitor = app.config.get("_ROUTE_WRAPPER_MONITOR")
    monitor_routes = monitor.get("routes", {}) if isinstance(monitor, dict) else {}

    totals = {
        "dispatch": 0,
        "success": 0,
        "failure": 0,
        "route_count": 0,
    }
    clusters: dict[str, dict] = {}

    if isinstance(monitor_routes, dict):
        totals["route_count"] = len(monitor_routes)
        for _, stats in monitor_routes.items():
            if not isinstance(stats, dict):
                continue
            cluster = str(stats.get("cluster") or "unknown")
            cluster_bucket = clusters.setdefault(cluster, {
                "dispatch": 0,
                "success": 0,
                "failure": 0,
                "routes": 0,
            })
            dispatch = int(stats.get("dispatch") or 0)
            success = int(stats.get("success") or 0)
            failure = int(stats.get("failure") or 0)
            totals["dispatch"] += dispatch
            totals["success"] += success
            totals["failure"] += failure
            cluster_bucket["dispatch"] += dispatch
            cluster_bucket["success"] += success
            cluster_bucket["failure"] += failure
            cluster_bucket["routes"] += 1

    return jsonify({
        "success": True,
        "monitor": {
            "created_at": monitor.get("created_at") if isinstance(monitor, dict) else None,
            "updated_at": monitor.get("updated_at") if isinstance(monitor, dict) else None,
            "totals": totals,
            "clusters": clusters,
            "routes": monitor_routes,
        },
        "request_context": {
            "principal": principal,
            "principal_source": principal_source,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    })


app.config["_UTILITY_ADMIN_ROUTE_HANDLERS"] = {
    "api_fs_list": api_fs_list,
    "api_list_dir_compat": api_list_dir_compat,
    "api_fs_mkdir": api_fs_mkdir,
    "api_fs_delete": api_fs_delete,
    "api_quick_copy": api_quick_copy,
    "api_quick_copy_clear": api_quick_copy_clear,
    "api_validate_urls": api_validate_urls,
    "api_url_status": api_url_status,
    "api_auth_certificate_info": api_auth_certificate_info,
    "api_auth_status": api_auth_status,
    "api_route_wrapper_monitor_snapshot": api_route_wrapper_monitor_snapshot,
}

def auth_welcome():
    principal, principal_source, cert_metadata = (
        get_request_principal()
    )

    certificate_present = bool(
        principal
        and principal.startswith(
            "cert:"
        )
    )

    normalized_target = sanitize_internal_next(
        (
            request.args.get("next")
            or request.referrer
            or url_for("index")
        ),
        fallback=url_for(
            "ballot_lens"
        ),
    )

    try:
        session_id = resolve_session_id(
            {},
            create_if_missing=False,
        )
    except Exception:
        session_id = None

    certificate_required = bool(
        _auth_mode_requires_certificate()
        and REQUIRE_CERT_FOR_MUTATIONS
        and not (
            DEPLOY_ENV == "local"
            and _is_local_request()
        )
        and not certificate_present
    )

    challenge_attempted = (
        str(
            request.args.get(
                "challenged"
            )
            or ""
        )
        .strip()
        .lower()
        in {
            "1",
            "true",
            "yes",
        }
    )

    status_code = (
        401
        if certificate_required
        else 200
    )

    return (
        render_template(
            "auth_welcome.html",

            principal=principal,

            principal_source=(
                principal_source
            ),

            cert_metadata=(
                cert_metadata
                if certificate_present
                else None
            ),

            session_id=session_id,

            require_cert=(
                certificate_required
            ),

            certificate_present=(
                certificate_present
            ),

            challenge_attempted=(
                challenge_attempted
            ),

            auth_reason=(
                "certificate_required"
                if certificate_required
                else None
            ),

            target_url=(
                normalized_target
            ),

            certificate_policy=(
                CERT_ENFORCEMENT_MODE
            ),

            azure_client_cert_mode=(
                AZURE_CLIENT_CERT_MODE
            ),
        ),

        status_code,
    )

def auth_challenge():
    # Explicit navigation checkpoint.
    #
    # Only the current request principal can satisfy certificate presence.
    # Existing session/cache state is intentionally ignored.

    next_url = sanitize_internal_next(
        request.args.get("next"),
        fallback=url_for(
            "ballot_lens"
        ),
    )

    if (
        not _auth_mode_requires_certificate()
        or not REQUIRE_CERT_FOR_MUTATIONS
    ):
        return redirect(
            next_url
        )

    if (
        DEPLOY_ENV == "local"
        and _is_local_request()
    ):
        return redirect(
            next_url
        )

    principal, _, _ = (
        get_request_principal()
    )

    if (
        principal
        and principal.startswith(
            "cert:"
        )
    ):
        return redirect(
            next_url
        )

    return redirect(
        url_for(
            "auth_welcome",
            next=next_url,
            challenged="1",
        )
    )

app.config["_PUBLIC_PAGES_ROUTE_HANDLERS"] = {
    "index": index,
    "ballot_lens": ballot_lens,
    "ballot_lens_modern": ballot_lens_modern,
    "worklist": worklist,
    "auth_welcome": auth_welcome,
    "auth_challenge": auth_challenge,
    "ocr_diagnostics": ocr_diagnostics,
}


@_rate_limit("5/minute")
def upload_to_input() -> str:
    wants_json = _request_wants_json()
    file = request.files.get("file") or request.files.get("csv_file") or request.files.get("data_file")
    cert_resp = _require_client_cert("upload_input")
    if cert_resp:
        return cert_resp
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
        if wants_json:
            return jsonify({"success": False, "error": "Guarded ingestion key required."}), 403
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
        if wants_json:
            return jsonify({"success": False, "error": "Upload quarantined: verified principal required."}), 403
        return redirect(request.referrer or url_for("ballot_lens"))

    ok, saved_name, err_path = _save_uploaded_file(file, str(INPUT_DIR), session_id=None)
    if ok and saved_name:
        if wants_json:
            return jsonify({"success": True, "filename": saved_name, "destination": "input"})
        flash(f"File '{saved_name}' uploaded to input folder.", "success")
    else:
        if wants_json:
            return jsonify({"success": False, "error": saved_name or "Invalid file type or no file selected."}), 400
        flash(saved_name or "Invalid file type or no file selected.", "danger")
    return redirect(request.referrer or url_for("ballot_lens"))

@_rate_limit("5/minute")
def upload_to_output() -> str:
    wants_json = _request_wants_json()
    file = request.files.get("file") or request.files.get("csv_file") or request.files.get("data_file")
    cert_resp = _require_client_cert("upload_output")
    if cert_resp:
        return cert_resp
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
        if wants_json:
            return jsonify({"success": False, "error": "Guarded ingestion key required."}), 403
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
        if wants_json:
            return jsonify({"success": False, "error": "Upload quarantined: verified principal required."}), 403
        return redirect(request.referrer or url_for("ballot_lens"))

    ok, saved_name, err_path = _save_uploaded_file(file, str(OUTPUT_DIR), session_id=None)
    if ok and saved_name:
        if wants_json:
            return jsonify({"success": True, "filename": saved_name, "destination": "output"})
        flash(f"File '{saved_name}' uploaded to output folder.", "success")
    else:
        if wants_json:
            return jsonify({"success": False, "error": saved_name or "Invalid file type or no file selected."}), 400
        flash(saved_name or "Invalid file type or no file selected.", "danger")
    return redirect(request.referrer or url_for("ballot_lens"))

@_rate_limit("5/minute")
def upload_to_uploads() -> str:
    wants_json = _request_wants_json()
    file = request.files.get("data_file") or request.files.get("file")
    cert_resp = _require_client_cert("upload_uploads")
    if cert_resp:
        return cert_resp
    guard_ok, guard_reason = _guarded_ingestion_allowed("upload_uploads")
    if not guard_ok:
        logger.warning({
            "level": "WARNING",
            "type": "security",
            "message": f"Upload blocked by guarded ingestion gate: {guard_reason}",
            "session_id": None,
        })
        if wants_json:
            return jsonify({"success": False, "error": "Guarded ingestion key required."}), 403
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
        if wants_json:
            return jsonify({"success": False, "error": "Upload quarantined: verified principal required."}), 403
        return redirect(request.referrer or url_for("ballot_lens"))

    ok, saved_name, err_path = _save_uploaded_file(file, str(UPLOADS_DIR), session_id=None)
    if ok and saved_name:
        session['FORCE_PARSE_INPUT_FILE'] = saved_name
        session['FORCE_PARSE_FORMAT'] = saved_name.rsplit('.', 1)[-1].lower() if '.' in saved_name else ''
        session['manual_source_pref'] = 'uploads'  # default UI to uploads after upload

        # Parse filename for metadata hints
        try:
            parsed_filename = parse_filename_simple(saved_name)
            # Store parsed metadata in session for later use
            if parsed_filename.get('state'):
                session['PARSED_STATE_HINT'] = parsed_filename['state']
            if parsed_filename.get('county'):
                session['PARSED_COUNTY_HINT'] = parsed_filename['county']
            if parsed_filename.get('year'):
                session['PARSED_YEAR_HINT'] = parsed_filename['year']
            if parsed_filename.get('contest_type'):
                session['PARSED_CONTEST_HINT'] = parsed_filename['contest_type']

            # Log parsed metadata for debugging
            logger.info({
                "level": "INFO",
                "type": "upload",
                "message": "Parsed filename metadata",
                "filename": saved_name,
                "parsed_metadata": {
                    "state": parsed_filename.get('state'),
                    "county": parsed_filename.get('county'),
                    "year": parsed_filename.get('year'),
                    "contest_type": parsed_filename.get('contest_type')
                }
            })
        except Exception as e:
            logger.warning(f"Failed to parse filename metadata: {e}")

        if wants_json:
            return jsonify({"success": True, "filename": saved_name, "destination": "uploads"})
        flash(f"File '{saved_name}' uploaded to uploads folder.", "success")
    else:
        if wants_json:
            return jsonify({"success": False, "error": saved_name or "Invalid file type or no file selected."}), 400
        flash(saved_name or "Invalid file type or no file selected.", "danger")
    return redirect(request.referrer or url_for("ballot_lens"))

def health() -> str:
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


app.config["_HEALTH_ROUTE_HANDLERS"]["health"] = health

def heartbeat() -> str:
    return {"status": "ok"}

def clear_history():
    try:
        if RUN_HISTORY_FILE.exists():
            RUN_HISTORY_FILE.unlink()
        flash("Run history cleared.", "success")
    except Exception as e:
        flash(f"Failed to clear history: {e}", "danger")
    return redirect(url_for("history"))

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


app.config["_FILE_IO_ROUTE_HANDLERS"] = {
    "download_fs": download_fs,
    "view_csv": view_csv,
    "csv_locate": csv_locate,
    "delete_input_file": delete_input_file,
    "delete_output_file": delete_output_file,
    "delete_upload_file": delete_upload_file,
    "download_input_file": download_input_file,
    "download_output_file": download_output_file,
    "download_upload_file": download_upload_file,
    "upload_to_input": upload_to_input,
    "upload_to_output": upload_to_output,
    "upload_to_uploads": upload_to_uploads,
    "heartbeat": heartbeat,
    "clear_history": clear_history,
    "history": history,
    "rerun_prior": rerun_prior,
}


# =====================================================================
# 5. ELECTION DATA WORKFLOW - SMART Elections DL1/DL2 Pipeline
# =====================================================================
# Worklist management, Pre-QC comparison, QC1/QC2 checkpoints
# Role enforcement: DL1 ≠ DL2 ≠ QC1 ≠ QC2
# Complete audit trail with chain of custody

def api_election_data_worklist():
    """
    Get Worklist - all races with step-by-step status tracking.
    
    Query params:
    - state: filter by state
    - year: filter by year
    - status: filter by workflow_status (step_1|step_2|step_3|step_4|completed)
    - limit: max records (default 100)
    """
    principal, _, _ = get_request_principal()
    if not principal and not ALLOW_DEV_NO_PRINCIPAL:
        return jsonify({"error": "Unauthorized"}), 403

    def _overview_status_to_workflow(value: str) -> str:
        text = (value or '').strip().lower()
        if not text:
            return 'step_1'
        if 'prod loaded' in text or 'completed' in text or 'final' in text:
            return 'completed'
        if 'qc2' in text or 'step 4' in text:
            return 'step_4'
        if 'qc1' in text or 'step 3' in text:
            return 'step_3'
        if 'dl2' in text or 'step 2' in text:
            return 'step_2'
        if 'dl1' in text or 'step 1' in text:
            return 'step_1'
        return 'step_2'

    def _is_truthy(value: str) -> bool:
        return str(value or '').strip().lower() in ('true', 'yes', 'y', '1', 'pass', 'passed', 'done')

    def _derive_stage_status(is_complete: bool, owner: str) -> str:
        if is_complete:
            return 'ready_for_qc'
        if str(owner or '').strip():
            return 'in_progress'
        return 'pending'

    def _build_worklist_from_overview(limit: int):
        from webapp.parser.data_standardization.google_sheets_client import fetch_worklist_overview

        overview = fetch_worklist_overview()
        if not overview.success:
            return None, overview.error or 'Failed to fetch worklist overview'

        state_filter = (request.args.get('state') or '').strip().lower()
        year_filter = (request.args.get('year') or '').strip()
        status_filter = (request.args.get('status') or '').strip().lower()

        rows = []
        for idx, row in enumerate(overview.records, start=1):
            source_url = (
                row.get('Source Link')
                or row.get('Step 1')
                or row.get('Source URL')
                or row.get('URL')
                or ''
            )
            dl1_value = row.get('Download 1') or row.get('Step 2') or ''
            dl2_value = row.get('Download 2') or row.get('0.00%') or ''
            preqc_value = row.get('Run Pre-Check') or row.get('Pre-QC Auto-check') or ''
            status_text = row.get('Status') or row.get('Work in Progress - DL2') or row.get('Standardization Process') or ''

            dl1_owner = row.get('Work in Progress - DL1') or row.get('DL1 Owner') or ''
            dl2_owner = row.get('Work in Progress - DL2') or row.get('DL2 Owner') or ''
            qc1_owner = row.get('QC1 Owner') or ''
            qc2_owner = row.get('QC2 Owner') or ''

            dl1_complete = _is_truthy(row.get('DL1 Complete')) or _is_truthy(dl1_value)
            dl2_complete = _is_truthy(row.get('DL2 Complete')) or _is_truthy(dl2_value)

            race_id = (
                row.get('QC ID')
                or row.get('RACE ID')
                or row.get('Race ID')
                or row.get('race_id')
                or f'overview_{idx}'
            )

            preqc_norm = str(preqc_value).strip().lower()
            preqc_details = str(row.get('Pre-QC Results') or '').strip().lower()
            if _is_truthy(preqc_norm) or 'passed' in preqc_details:
                preqc_result = 'passed'
            elif 'fail' in preqc_details or 'discrep' in preqc_details:
                preqc_result = 'review_needed'
            else:
                preqc_result = 'pending'

            year_value = str(row.get('Year') or '').strip()
            state_value = str(row.get('State') or '').strip()
            workflow_status = _overview_status_to_workflow(status_text)
            if workflow_status == 'step_1' and dl2_complete:
                workflow_status = 'step_2'

            if state_filter and state_value.lower() != state_filter:
                continue
            if year_filter and year_filter != year_value:
                continue
            if status_filter and workflow_status.lower() != status_filter:
                continue

            rows.append({
                'id': idx,
                'race_id': str(race_id),
                'year': int(year_value) if year_value.isdigit() else (year_value or None),
                'state': state_value,
                'county': row.get('County') or row.get('County/District') or '',
                'office': row.get('Race') or row.get('Office') or row.get('Contest') or '',
                'source_url': source_url,
                'dl1_assigned_to': dl1_owner,
                'dl1_status': _derive_stage_status(dl1_complete, dl1_owner),
                'dl2_assigned_to': dl2_owner,
                'dl2_status': _derive_stage_status(dl2_complete, dl2_owner),
                'preqc_result': preqc_result,
                'qc1_assigned_to': qc1_owner,
                'qc1_status': row.get('QC1 Status') or ('completed' if workflow_status in ('step_4', 'completed') else 'pending'),
                'qc1_selected_dl': row.get('QC1 Selected DL') or None,
                'qc2_assigned_to': qc2_owner,
                'qc2_status': row.get('QC2 Status') or ('completed' if workflow_status == 'completed' else 'pending'),
                'workflow_status': workflow_status,
                'updated_at': None,
            })

        return rows[:limit], None

    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from webapp.parser.models.election_data import DownloadRecord

        # Store DB URL in env or fallback to default
        db_url = os.getenv('DATABASE_URL', 'sqlite:///election_data.db')
        engine = create_engine(db_url)
        Session = sessionmaker(bind=engine)
        session = Session()

        try:
            query = session.query(DownloadRecord)

            # Apply filters
            for param, field in [('state', 'state'), ('year', 'year')]:
                if request.args.get(param):
                    val = request.args.get(param)
                    if param == 'year':
                        val = int(val) if val.isdigit() else val
                    query = query.filter(getattr(DownloadRecord, field) == val)

            if request.args.get('status'):
                query = query.filter(DownloadRecord.workflow_status == request.args.get('status'))

            limit = min(int(request.args.get('limit', 100)), 500)
            total = query.count()
            records = query.limit(limit).all()

            # Convert to dict
            worklist = [{
                'id': r.id,
                'race_id': r.race_id,
                'year': r.year,
                'state': r.state,
                'county': r.county,
                'office': r.office,
                'source_url': r.source_url,
                'dl1_assigned_to': r.dl1_assigned_to,
                'dl1_status': r.dl1_status,
                'dl2_assigned_to': r.dl2_assigned_to,
                'dl2_status': r.dl2_status,
                'preqc_result': r.preqc_result,
                'qc1_assigned_to': r.qc1_assigned_to,
                'qc1_status': r.qc1_status,
                'qc1_selected_dl': r.qc1_selected_dl,
                'qc2_assigned_to': r.qc2_assigned_to,
                'qc2_status': r.qc2_status,
                'workflow_status': r.workflow_status,
                'updated_at': r.updated_at.isoformat() if r.updated_at else None,
            } for r in records]

            return jsonify({'success': True, 'total': total, 'records': worklist}), 200

        finally:
            session.close()

    except Exception as e:
        err_msg = str(e)
        if 'no such table: download_records' in err_msg.lower():
            try:
                limit = min(int(request.args.get('limit', 100)), 500)
                worklist, fallback_error = _build_worklist_from_overview(limit)
                if worklist is not None:
                    return jsonify({
                        'success': True,
                        'total': len(worklist),
                        'records': worklist,
                        'source': 'google_sheets_overview_fallback',
                        'warning': 'SQL worklist table not initialized; returning overview fallback data',
                    }), 200
                return jsonify({'success': False, 'error': fallback_error}), 500
            except Exception as fallback_exc:
                logger.error(f"Worklist fallback failed: {fallback_exc}")
        logger.error(f"Error fetching worklist: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


def api_election_data_worklist_overview():
    """
    Fetch worklist overview data from Google Sheets.

    Query params:
    - limit: max records (default 200)
    - sheet: override sheet name (defaults to GOOGLE_SHEETS_WORKLIST_OVERVIEW_SHEET)
    """
    principal, _, _ = get_request_principal()
    if not principal and not ALLOW_DEV_NO_PRINCIPAL:
        return jsonify({"error": "Unauthorized"}), 403

    try:
        from webapp.parser.data_standardization.google_sheets_client import fetch_worklist_overview

        limit = min(int(request.args.get('limit', 200)), 2000)
        sheet_name = request.args.get('sheet')
        result = fetch_worklist_overview(sheet_name=sheet_name)

        if not result.success:
            return jsonify({'success': False, 'error': result.error or 'Failed to fetch sheet'}), 500

        records = result.records[:limit]

        return jsonify({
            'success': True,
            'sheet_name': result.sheet_name,
            'row_count': result.row_count,
            'records': records,
        }), 200
    except ValueError as e:
        # Google Sheets not configured or invalid parameters
        error_msg = str(e)
        logger.warning(f"Worklist endpoint not available: {error_msg}")
        hint = (
            "\nFor local development, you can use: "
            "GOOGLE_APPLICATION_CREDENTIALS=/path/to/google_service_account.json"
            "\nFor Azure, configure individual GOOGLE_SHEETS_SA_* environment variables."
        )
        return jsonify({
            'success': False,
            'error': 'Google Sheets access not configured',
            'detail': error_msg + hint
        }), 503
    except Exception as e:
        logger.error(f"Error fetching worklist overview: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


def api_election_data_db_lite_finalized():
    """
    Fetch Finalized Data sheet from SMART Elections Database-Lite.

    Query params:
    - limit: max records (default 200)
    - state: optional state filter
    - year: optional year filter
    - county: optional county/district filter
    - contest: optional office/contest filter
    """
    principal, _, _ = get_request_principal()
    if not principal and not ALLOW_DEV_NO_PRINCIPAL:
        return jsonify({"error": "Unauthorized"}), 403

    try:
        from webapp.parser.data_standardization.google_sheets_client import get_election_data_client

        limit = min(int(request.args.get('limit', 200)), 2000)
        state_filter = (request.args.get('state') or '').strip().lower()
        year_filter = (request.args.get('year') or '').strip()
        county_filter = (request.args.get('county') or '').strip().lower()
        contest_filter = (request.args.get('contest') or '').strip().lower()
        client = get_election_data_client()
        result = client.fetch_finalized_data()

        if not result.success:
            return jsonify({'success': False, 'error': result.error or 'Failed to fetch sheet'}), 500

        records = result.records

        if state_filter:
            records = [r for r in records if (str(r.get('State') or '').strip().lower() == state_filter)]
        if year_filter:
            records = [r for r in records if year_filter in str(r.get('Year') or '')]
        if county_filter:
            records = [
                r for r in records
                if str(r.get('County/District') or r.get('County') or '').strip().lower() == county_filter
            ]
        if contest_filter:
            records = [
                r for r in records
                if str(r.get('Office') or r.get('Contest') or '').strip().lower() == contest_filter
            ]

        filtered_count = len(records)
        records = records[:limit]

        return jsonify({
            'success': True,
            'sheet_name': result.sheet_name,
            'row_count': result.row_count,
            'filtered_count': filtered_count,
            'records': records,
        }), 200
    except ValueError as e:
        # Google Sheets not configured or invalid parameters
        error_msg = str(e)
        logger.warning(f"DB-Lite finalized endpoint not available: {error_msg}")
        hint = (
            "\nFor local development, you can use: "
            "GOOGLE_APPLICATION_CREDENTIALS=/path/to/google_service_account.json"
            "\nFor Azure, configure individual GOOGLE_SHEETS_SA_* environment variables."
        )
        return jsonify({
            'success': False,
            'error': 'Google Sheets access not configured',
            'detail': error_msg + hint
        }), 503
    except Exception as e:
        logger.error(f"Error fetching DB-Lite finalized data: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


def api_election_data_db_lite_down_ballot():
    """
    Fetch Down-Ballot Calculations sheet from SMART Elections Database-Lite.

    Query params:
    - limit: max records (default 200)
    """
    principal, _, _ = get_request_principal()
    if not principal and not ALLOW_DEV_NO_PRINCIPAL:
        return jsonify({"error": "Unauthorized"}), 403

    try:
        from webapp.parser.data_standardization.google_sheets_client import get_election_data_client

        limit = min(int(request.args.get('limit', 200)), 2000)
        client = get_election_data_client()
        result = client.fetch_down_ballot_calculations()

        if not result.success:
            return jsonify({'success': False, 'error': result.error or 'Failed to fetch sheet'}), 500

        records = result.records[:limit]

        return jsonify({
            'success': True,
            'sheet_name': result.sheet_name,
            'row_count': result.row_count,
            'records': records,
        }), 200
    except ValueError as e:
        # Google Sheets not configured or invalid parameters
        error_msg = str(e)
        logger.warning(f"DB-Lite down-ballot endpoint not available: {error_msg}")
        hint = (
            "\nFor local development, you can use: "
            "GOOGLE_APPLICATION_CREDENTIALS=/path/to/google_service_account.json"
            "\nFor Azure, configure individual GOOGLE_SHEETS_SA_* environment variables."
        )
        return jsonify({
            'success': False,
            'error': 'Google Sheets access not configured',
            'detail': error_msg + hint
        }), 503
    except Exception as e:
        logger.error(f"Error fetching DB-Lite down-ballot data: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


def api_election_data_google_sheets_health():
    """
    Verify Google Sheets access for worklist overview + DB-Lite sheets.
    """
    principal, _, _ = get_request_principal()
    if not principal and not ALLOW_DEV_NO_PRINCIPAL:
        return jsonify({"error": "Unauthorized"}), 403

    try:
        from webapp.parser.data_standardization.google_sheets_client import (
            fetch_worklist_overview,
            get_election_data_client,
        )

        results = {}

        try:
            worklist_result = fetch_worklist_overview()
            results['worklist_overview'] = {
                'success': worklist_result.success,
                'sheet_name': worklist_result.sheet_name,
                'row_count': worklist_result.row_count,
                'error': worklist_result.error,
            }
        except Exception as e:
            results['worklist_overview'] = {
                'success': False,
                'sheet_name': None,
                'row_count': 0,
                'error': str(e),
            }

        try:
            client = get_election_data_client()
            finalized_result = client.fetch_finalized_data()
            results['db_lite_finalized'] = {
                'success': finalized_result.success,
                'sheet_name': finalized_result.sheet_name,
                'row_count': finalized_result.row_count,
                'error': finalized_result.error,
            }
        except Exception as e:
            results['db_lite_finalized'] = {
                'success': False,
                'sheet_name': None,
                'row_count': 0,
                'error': str(e),
            }

        try:
            client = get_election_data_client()
            down_ballot_result = client.fetch_down_ballot_calculations()
            results['db_lite_down_ballot'] = {
                'success': down_ballot_result.success,
                'sheet_name': down_ballot_result.sheet_name,
                'row_count': down_ballot_result.row_count,
                'error': down_ballot_result.error,
            }
        except Exception as e:
            results['db_lite_down_ballot'] = {
                'success': False,
                'sheet_name': None,
                'row_count': 0,
                'error': str(e),
            }

        overall_ok = all(result.get('success') for result in results.values())

        return jsonify({
            'success': overall_ok,
            'results': results,
        }), 200 if overall_ok else 503

    except Exception as e:
        logger.error(f"Error checking Google Sheets health: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


def api_election_data_states_counties():
    """
    Return authoritative state-to-county mappings from Google Sheets.
    
    This replaces unreliable URL-based heuristics with clean, normalized data
    from the SMART Elections Database-Lite Finalized Data sheet.
    
    Returns:
        {
            "success": true,
            "states": ["Alabama", "Alaska", ...],
            "counties": {
                "Alabama": ["Autauga", "Baldwin", ...],
                "Alaska": ["District 01", "District 02", ...],
                ...
            },
            "total_states": 50,
            "total_counties": 3143
        }
    """
    started_at = time.perf_counter()
    principal, _, _ = get_request_principal()
    if not principal and not ALLOW_DEV_NO_PRINCIPAL:
        # Graceful degradation: return empty mappings for unauthenticated users
        # This prevents 403 errors in console when users browse without cert
        _log_endpoint_latency("/api/election_data/states_counties", started_at, cache_hit=False, context={"auth": "required"})
        return jsonify({
            "success": True,
            "states": [],
            "counties": {},
            "total_states": 0,
            "total_counties": 0,
            "note": "Authentication required for state/county mappings"
        }), 200

    cached_payload = _get_ttl_cache_payload("states_counties")
    if isinstance(cached_payload, dict):
        _log_endpoint_latency(
            "/api/election_data/states_counties",
            started_at,
            cache_hit=True,
            context={
                "total_states": len(cached_payload.get("states") or []),
                "total_counties": int(cached_payload.get("total_counties") or 0),
            },
        )
        return jsonify(cached_payload), 200

    try:
        from collections import defaultdict

        from webapp.parser.data_standardization.google_sheets_client import get_election_data_client

        client = get_election_data_client()
        result = client.fetch_finalized_data()

        if not result.success:
            return jsonify({
                'success': False,
                'error': result.error or 'Failed to fetch Google Sheets data'
            }), 500

        # Build normalized state-to-county mappings
        state_counties = defaultdict(set)
        years_set = set()
        contests_set = set()

        for record in result.records:
            state = record.get('State', '').strip()
            county = record.get('County/District', '').strip()

            if state and county:
                # Normalize: title case, deduplicate
                state_normalized = state.title()
                county_normalized = county.title()
                state_counties[state_normalized].add(county_normalized)

            year_value = str(record.get('Year', '')).strip()
            if year_value:
                years_set.add(year_value)

            contest_value = str(record.get('Office', '') or record.get('Contest', '')).strip()
            if contest_value:
                contests_set.add(contest_value)

        # Convert to sorted lists for consistent ordering
        states_list = sorted(state_counties.keys())
        counties_dict = {
            state: sorted(list(counties))
            for state, counties in state_counties.items()
        }

        total_counties = sum(len(counties) for counties in counties_dict.values())

        payload = {
            'success': True,
            'states': states_list,
            'counties': counties_dict,
            'years': sorted(list(years_set), reverse=True),
            'contests': sorted(list(contests_set)),
            'total_states': len(states_list),
            'total_counties': total_counties,
        }
        _set_ttl_cache_payload("states_counties", payload, STATES_COUNTIES_CACHE_TTL_SEC)
        _log_endpoint_latency(
            "/api/election_data/states_counties",
            started_at,
            cache_hit=False,
            context={
                "total_states": len(states_list),
                "total_counties": total_counties,
            },
        )
        return jsonify(payload), 200

    except Exception as e:
        logger.error(f"Error fetching states/counties mapping: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


def api_assign_dl_owner(race_id):
    """
    Assign DL1 or DL2 owner to a race.
    
    Body: {'dl': 'DL1'|'DL2', 'assigned_to': 'username}
    """
    principal, _, _ = get_request_principal()
    if not principal and not ALLOW_DEV_NO_PRINCIPAL:
        return jsonify({"error": "Unauthorized"}), 403

    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from webapp.parser.models.election_data import DownloadRecord

        data = request.get_json() or {}
        dl = data.get('dl', '').upper()  # DL1 or DL2
        assigned_to = data.get('assigned_to', principal)

        if dl not in ('DL1', 'DL2'):
            return jsonify({'success': False, 'error': 'dl must be DL1 or DL2'}), 400

        db_url = os.getenv('DATABASE_URL', 'sqlite:///election_data.db')
        engine = create_engine(db_url)
        Session = sessionmaker(bind=engine)
        session = Session()

        try:
            record = session.query(DownloadRecord).filter(DownloadRecord.race_id == race_id).first()

            if not record:
                return jsonify({'success': False, 'error': f'Race {race_id} not found'}), 404

            # Enforce role separation: DL1 ≠ DL2
            if dl == 'DL1':
                if record.dl2_assigned_to and record.dl2_assigned_to == assigned_to:
                    return jsonify({
                        'success': False,
                        'error': f'{assigned_to} is already assigned to DL2 - cannot also assign to DL1'
                    }), 400
                record.dl1_assigned_to = assigned_to
                record.dl1_status = 'pending'
            else:  # DL2
                if record.dl1_assigned_to and record.dl1_assigned_to == assigned_to:
                    return jsonify({
                        'success': False,
                        'error': f'{assigned_to} is already assigned to DL1 - cannot also assign to DL2'
                    }), 400
                record.dl2_assigned_to = assigned_to
                record.dl2_status = 'pending'

            record.updated_at = datetime.utcnow()
            session.commit()

            return jsonify({
                'success': True,
                'message': f'{assigned_to} assigned to {dl} for race {race_id}'
            }), 200

        finally:
            session.close()

    except Exception as e:
        logger.error(f"Error assigning DL owner: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


def api_preqc_check(race_id):
    """
    Run Pre-QC Auto-check: strict equality + fuzzy matching between DL1 and DL2.
    
    Returns discrepancy report for QC1 review.
    """
    principal, _, _ = get_request_principal()
    if not principal and not ALLOW_DEV_NO_PRINCIPAL:
        return jsonify({"error": "Unauthorized"}), 403

    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from webapp.parser.data_standardization.election_data_standardizer import (
            PreQCComparisonEngine,
        )
        from webapp.parser.models.election_data import (
            DownloadRecord,
            PreQCComparison,
            ValidationRecord_DL1,
            ValidationRecord_DL2,
        )

        db_url = os.getenv('DATABASE_URL', 'sqlite:///election_data.db')
        engine = create_engine(db_url)
        Session = sessionmaker(bind=engine)
        session = Session()

        try:
            # Get DL1 and DL2 records
            dl1 = session.query(ValidationRecord_DL1).filter(
                ValidationRecord_DL1.race_id == race_id
            ).first()
            dl2 = session.query(ValidationRecord_DL2).filter(
                ValidationRecord_DL2.race_id == race_id
            ).first()

            if not dl1 or not dl2:
                return jsonify({
                    'success': False,
                    'error': 'Both DL1 and DL2 records required for Pre-QC comparison'
                }), 400

            # Convert to dict for comparison
            dl1_dict = {
                'race_id': dl1.race_id,
                'standardized_candidate_name': dl1.standardized_candidate_name,
                'ballot_party': dl1.ballot_party,
                'fec_party': dl1.fec_party,
                'fec_id': dl1.fec_id,
                'total_votes': dl1.total_votes,
                'is_write_in': dl1.is_write_in,
            }
            dl2_dict = {
                'race_id': dl2.race_id,
                'standardized_candidate_name': dl2.standardized_candidate_name,
                'ballot_party': dl2.ballot_party,
                'fec_party': dl2.fec_party,
                'fec_id': dl2.fec_id,
                'total_votes': dl2.total_votes,
                'is_write_in': dl2.is_write_in,
            }

            # Run Pre-QC comparison
            preqc_result = PreQCComparisonEngine.compare_records(dl1_dict, dl2_dict)

            # Store result
            preqc = PreQCComparison(
                race_id=race_id,
                dl1_record_id=dl1.id,
                dl2_record_id=dl2.id,
                strict_equality_passed=preqc_result.strict_passed,
                fuzzy_match_confidence=preqc_result.fuzzy_confidence,
                fuzzy_candidate_confidence=preqc_result.candidate_confidence,
                fuzzy_party_confidence=preqc_result.party_confidence,
                fuzzy_fec_id_confidence=preqc_result.fec_id_confidence,
                discrepancy_count=preqc_result.discrepancy_count,
                discrepancy_fields=json.dumps(preqc_result.discrepancies),
                comparison_status=preqc_result.status,
                comparison_summary=preqc_result.summary,
                checked_by=principal,
            )
            session.add(preqc)

            # Update DownloadRecord
            download = session.query(DownloadRecord).filter(
                DownloadRecord.race_id == race_id
            ).first()
            if download:
                download.preqc_auto_check_completed = True
                download.preqc_result = preqc_result.status
                download.preqc_strict_passed = preqc_result.strict_passed
                download.preqc_fuzzy_score = preqc_result.fuzzy_confidence
                download.preqc_discrepancy_count = preqc_result.discrepancy_count
                download.preqc_checked_at = datetime.utcnow()

            session.commit()

            return jsonify({
                'success': True,
                'preqc_result': {
                    'race_id': preqc_result.race_id,
                    'strict_passed': preqc_result.strict_passed,
                    'fuzzy_confidence': round(preqc_result.fuzzy_confidence, 3),
                    'status': preqc_result.status,
                    'summary': preqc_result.summary,
                    'discrepancy_count': preqc_result.discrepancy_count,
                    'discrepancies': preqc_result.discrepancies,
                }
            }), 200

        finally:
            session.close()

    except Exception as e:
        logger.error(f"Error running Pre-QC check for {race_id}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


def api_qc1_submit(race_id):
    """
    Submit QC1 form and approve/reject data for QC2.
    
    Body: {
      'selected_dl': 'DL1'|'DL2',
      'inspection_result': 'pass'|'fail',
      'checklist_results': {...},
      'notes': 'optional notes'
    }
    """
    principal, _, _ = get_request_principal()
    if not principal and not ALLOW_DEV_NO_PRINCIPAL:
        return jsonify({"error": "Unauthorized"}), 403

    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from webapp.parser.models.election_data import (
            DownloadRecord,
            PreQCComparison,
            QC1Checkpoint,
        )

        data = request.get_json() or {}
        selected_dl = data.get('selected_dl', '').upper()

        if selected_dl not in ('DL1', 'DL2'):
            return jsonify({'success': False, 'error': 'selected_dl must be DL1 or DL2'}), 400

        db_url = os.getenv('DATABASE_URL', 'sqlite:///election_data.db')
        engine = create_engine(db_url)
        Session = sessionmaker(bind=engine)
        session = Session()

        try:
            download = session.query(DownloadRecord).filter(
                DownloadRecord.race_id == race_id
            ).first()

            if not download:
                return jsonify({'success': False, 'error': f'Race {race_id} not found'}), 404

            # Enforce role separation: QC1 cannot be DL1 or DL2 owner
            if principal in (download.dl1_assigned_to, download.dl2_assigned_to):
                return jsonify({
                    'success': False,
                    'error': 'QC1 designee cannot also be DL1 or DL2 owner'
                }), 400

            # Get Pre-QC results
            preqc = session.query(PreQCComparison).filter(
                PreQCComparison.race_id == race_id
            ).order_by(PreQCComparison.checked_at.desc()).first()

            # Create QC1 checkpoint
            qc1 = QC1Checkpoint(
                download_record_id=download.id,
                preqc_comparison_id=preqc.id if preqc else None,
                reviewed_by=principal,
                reviewed_at=datetime.utcnow(),
                qc1_checklist_results=json.dumps(data.get('checklist_results', {})),
                data_inspection_result=data.get('inspection_result', 'pending'),
                data_inspection_notes=data.get('notes', ''),
                selected_dl_source=selected_dl,
                approval_status='approved' if data.get('inspection_result') == 'pass' else 'rejected',
            )
            session.add(qc1)

            # Update DownloadRecord
            download.qc1_assigned_to = principal
            download.qc1_status = 'completed'
            download.qc1_selected_dl = selected_dl
            download.qc1_completed_at = datetime.utcnow()
            download.qc1_data_inspection_result = data.get('inspection_result')
            download.workflow_status = 'step_3' if data.get('inspection_result') == 'pass' else 'step_2_review'

            session.commit()

            return jsonify({
                'success': True,
                'message': f'QC1 review completed for {race_id}',
                'qc1_id': qc1.id,
                'workflow_status': download.workflow_status,
            }), 200

        finally:
            session.close()

    except Exception as e:
        logger.error(f"Error submitting QC1 for {race_id}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


def api_election_data_stats():
    """Get overall election data pipeline statistics."""
    principal, _, _ = get_request_principal()
    if not principal and not ALLOW_DEV_NO_PRINCIPAL:
        return jsonify({"error": "Unauthorized"}), 403

    def _build_stats_from_overview() -> dict:
        from webapp.parser.data_standardization.google_sheets_client import fetch_worklist_overview

        overview = fetch_worklist_overview()
        if not overview.success:
            return {
                'total_races': 0,
                'dl1_ready': 0,
                'dl2_ready': 0,
                'preqc_passed': 0,
                'qc1_pending': 0,
                'qc2_pending': 0,
                'production_records': 0,
            }

        rows = overview.records
        preqc_passed = 0
        for row in rows:
            value = str(row.get('Run Pre-Check') or row.get('Pre-QC Auto-check') or '').strip().lower()
            if value in ('true', 'yes', 'pass', 'passed'):
                preqc_passed += 1

        return {
            'total_races': len(rows),
            'dl1_ready': sum(1 for r in rows if str(r.get('Download 1') or r.get('Step 2') or '').strip()),
            'dl2_ready': sum(1 for r in rows if str(r.get('Download 2') or r.get('0.00%') or '').strip()),
            'preqc_passed': preqc_passed,
            'qc1_pending': 0,
            'qc2_pending': 0,
            'production_records': 0,
        }

    try:
        from sqlalchemy import create_engine, func
        from sqlalchemy.orm import sessionmaker

        from webapp.parser.models.election_data import (
            DownloadRecord,
            ElectionResult,
        )

        db_url = os.getenv('DATABASE_URL', 'sqlite:///election_data.db')
        engine = create_engine(db_url)
        Session = sessionmaker(bind=engine)
        session = Session()

        try:
            stats = {
                'total_races': session.query(func.count(DownloadRecord.id)).scalar() or 0,
                'dl1_ready': session.query(func.count(DownloadRecord.id)).filter(
                    DownloadRecord.dl1_status == 'ready_for_qc'
                ).scalar() or 0,
                'dl2_ready': session.query(func.count(DownloadRecord.id)).filter(
                    DownloadRecord.dl2_status == 'ready_for_qc'
                ).scalar() or 0,
                'preqc_passed': session.query(func.count(DownloadRecord.id)).filter(
                    DownloadRecord.preqc_result == 'passed'
                ).scalar() or 0,
                'qc1_pending': session.query(func.count(DownloadRecord.id)).filter(
                    DownloadRecord.qc1_status == 'pending'
                ).scalar() or 0,
                'qc2_pending': session.query(func.count(DownloadRecord.id)).filter(
                    DownloadRecord.qc2_status == 'pending'
                ).scalar() or 0,
                'production_records': session.query(func.count(ElectionResult.id)).scalar() or 0,
            }

            return jsonify({'success': True, 'stats': stats}), 200

        finally:
            session.close()

    except Exception as e:
        err_msg = str(e)
        if 'no such table: download_records' in err_msg.lower():
            try:
                stats = _build_stats_from_overview()
                return jsonify({
                    'success': True,
                    'stats': stats,
                    'source': 'google_sheets_overview_fallback',
                    'warning': 'SQL worklist table not initialized; returning overview-derived stats',
                }), 200
            except Exception as fallback_exc:
                logger.error(f"Stats fallback failed: {fallback_exc}")
        logger.error(f"Error fetching stats: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


app.config["_ELECTION_DATA_ROUTE_HANDLERS"] = {
    "api_election_data_worklist": api_election_data_worklist,
    "api_election_data_worklist_overview": api_election_data_worklist_overview,
    "api_election_data_db_lite_finalized": api_election_data_db_lite_finalized,
    "api_election_data_db_lite_down_ballot": api_election_data_db_lite_down_ballot,
    "api_election_data_google_sheets_health": api_election_data_google_sheets_health,
    "api_election_data_states_counties": api_election_data_states_counties,
    "api_assign_dl_owner": api_assign_dl_owner,
    "api_preqc_check": api_preqc_check,
    "api_qc1_submit": api_qc1_submit,
    "api_election_data_stats": api_election_data_stats,
    "api_warehouse_election_results": api_warehouse_election_results,
}


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
        if not _socket_lifecycle.socket_connection_admitted(
            principal,
            allow_anonymous=ALLOW_ANON_NO_PRINCIPAL,
        ):
            emit('parser_output', {
                "level": "ERROR",
                "type": "auth",
                "message": "Missing client certificate or SSO principal; connection rejected.",
                "session_id": None
            }, room=getattr(request, 'sid', None))
            return False

        # --- Certificate validation (Step 4: Check for expiry/changes) ---
        cert_fingerprint = _socket_lifecycle.derive_certificate_fingerprint(
            principal,
            cert_metadata,
            request.headers.get("X-ARR-ClientCert", ""),
            logger=logger,
        )

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
        reuse_hint, allow_reuse = (
            _authority_context.resolve_session_reuse_policy(
                auth,
                principal_source,
                allow_auto_session_reuse=ALLOW_AUTO_SESSION_REUSE,
                request_args=getattr(request, "args", None),
                request_headers=getattr(request, "headers", None),
            )
        )
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

            # Cache certificate and apply socket authority transitions.
            if cert_fingerprint and cert_metadata:
                _socket_lifecycle.process_certificate_session_authority(
                    resolved,
                    principal,
                    cert_metadata,
                    cert_fingerprint,
                    session_manager=session_manager,
                    transition_session=transition_session,
                    idle_state=SessionState.IDLE,
                    prepare_phase=PipelinePhase.PREPARE,
                    emit_event=socketio.emit,
                    logger=logger,
                )

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
    return _socket_lifecycle.disconnect_socket_authority(
        safe_sid=safe_sid,
        request_sid=getattr(request, "sid", None),
        session_manager=session_manager,
        logger=logger,
    )


@socketio.on('ack_cert_reauth')
def handle_ack_cert_reauth(data=None) -> None:
    _socket_lifecycle.acknowledge_certificate_reauth(
        data,
        resolve_session_id=resolve_session_id,
        transition_session=transition_session,
        idle_state=SessionState.IDLE,
        prepare_phase=PipelinePhase.PREPARE,
        emit_event=socketio.emit,
        logger=logger,
    )

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
    run_ballot_lens_socket_handler(
        data=data,
        hooks={
            "cleanup_sessions": cleanup_sessions,
            "is_dev_isolation_bypass_request": _is_dev_isolation_bypass_request,
            "resolve_session_id": resolve_session_id,
            "rate_limit_socket_action": _rate_limit_socket_action,
            "emit": emit,
            "normalize_log_obj": normalize_log_obj,
            "get_request_principal": get_request_principal,
            "request": request,
            "require_cert_for_socket_action": _require_cert_for_socket_action,
            "require_cert_for_mutations": REQUIRE_CERT_FOR_MUTATIONS,
            "join_room": join_room,
            "socketio": socketio,
            "safe_sid": safe_sid,
            "session_manager": session_manager,
            "create_session_metadata": create_session_metadata,
            "safe_get": safe_get,
            "safe_is_alive": safe_is_alive,
            "safe_lower": safe_lower,
            "safe_strip": safe_strip,
            "get_manual_source": get_manual_source,
            "os": os,
            "uploads_dir": UPLOADS_DIR,
            "session": session,
            "urlparse": urlparse,
            "safe_validate_external_url": safe_validate_external_url,
            "url_allowlist_suffixes": URL_ALLOWLIST_SUFFIXES,
            "url_allowlist_hosts": URL_ALLOWLIST_HOSTS,
            "url_enforce_allowlist": URL_ENFORCE_ALLOWLIST,
            "url_block_private_ips": URL_BLOCK_PRIVATE_IPS,
            "direct_url_limit": DIRECT_URL_LIMIT,
            "guarded_ingestion_allowed": _guarded_ingestion_allowed,
            "collect_url_reference_hint": _collect_url_reference_hint,
            "is_output_bypassed": is_output_bypassed,
            "lock_session": lock_session,
            "socketio_emit_func": socketio_emit_func,
            "logger": logger,
            "orjson": orjson,
            "webapp_console_levels": WEBAPP_CONSOLE_LEVELS,
            "prompt": prompt,
            "time": time,
            "datetime": datetime,
            "timezone": timezone,
            "log_run_event": log_run_event,
            "cancellation_manager": cancellation_manager,
            "get_prompt_queue": get_prompt_queue,
            "path_cls": Path,
            "output_dir": OUTPUT_DIR,
            "threading": threading,
            "process_urls_for_web": process_urls_for_web,
            "emit_download_ready": _emit_download_ready,
            "transition_session": transition_session,
            "session_state": SessionState,
            "safe_is_set": safe_is_set,
        },
    )

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


def api_fec_save_mapping():
    cert_resp = _require_client_cert("fec_save_mapping")
    if cert_resp:
        return cert_resp
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

# ============================================
# Data Assurance API Endpoints (QA Panel Integration)
# ============================================

def api_data_assurance_classify():
    """
    Classify parsed election data as DL1 (Data Level 1) with auto QA checks.
    Stores results in PostgreSQL and optionally writes to ner_training_data for ML.
    """
    cert_resp = _require_client_cert("data_assurance_classify")
    if cert_resp:
        return cert_resp
    try:
        from webapp.parser.utils.ml_telemetry import record_ml_event
        from webapp.parser.utils.nlp_entity_extractor import extract_training_entities

        data = request.get_json()
        metadata = data.get("metadata", {})
        parsed_data = data.get("parsed_data", {})

        # Generate unique dataset ID
        import hashlib
        from datetime import datetime
        dataset_id = hashlib.sha256(
            f"{metadata.get('state', '')}{metadata.get('county', '')}{metadata.get('contest', '')}{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]

        # Run auto QA checks
        headers = parsed_data.get("headers", [])
        rows = parsed_data.get("rows", [])

        detected_issues = []
        confidence_score = 100.0

        # Check: Missing headers
        if not headers:
            detected_issues.append({
                "issue_type": "missing_headers",
                "severity": "ERROR",
                "description": "No column headers detected",
            })
            confidence_score -= 30

        # Check: Empty data
        if not rows or len(rows) == 0:
            detected_issues.append({
                "issue_type": "empty_data",
                "severity": "CRITICAL",
                "description": "No data rows found",
            })
            confidence_score -= 40

        # Check: Mismatched column counts
        if headers and rows:
            expected_cols = len(headers)
            mismatched_rows = sum(1 for row in rows if len(row) != expected_cols)
            if mismatched_rows > 0:
                detected_issues.append({
                    "issue_type": "column_mismatch",
                    "severity": "WARNING",
                    "description": f"{mismatched_rows} rows have mismatched column counts",
                    "affected_rows": mismatched_rows,
                })
                confidence_score -= min(20, mismatched_rows * 0.5)

        # Determine DL status
        dl_status = "DL1"  # Always start at DL1; manual review promotes to DL2
        if confidence_score < 50:
            dl_status = "REJECTED"

        record_ml_event(
            "data_assurance",
            "classification_scored",
            metadata={
                "dataset_id": dataset_id,
                "dl_status": dl_status,
                "confidence_score": confidence_score,
                "issues": len(detected_issues),
                "rows": len(rows or []),
                "headers": len(headers or []),
            },
        )

        # Store in PostgreSQL (data_assurance_classifications table)
        from webapp.parser.utils.db_utils import SessionLocal
        with SessionLocal() as db_session:
            from datetime import datetime
            db_session.execute(
                text("""
                    INSERT INTO data_assurance_classifications 
                    (dataset_id, dl_status, confidence_score, detected_issues, metadata, created_at)
                    VALUES (:dataset_id, :dl_status, :confidence_score, :detected_issues, :metadata, :created_at)
                """),
                {
                    "dataset_id": dataset_id,
                    "dl_status": dl_status,
                    "confidence_score": confidence_score,
                    "detected_issues": orjson.dumps(detected_issues).decode(),
                    "metadata": orjson.dumps(metadata).decode(),
                    "created_at": datetime.utcnow(),
                }
            )
            db_session.commit()

        # Optionally write to NER training data if REVIEW_WITH_MANUAL_BOT is enabled
        from webapp.parser.config import REVIEW_WITH_MANUAL_BOT
        if REVIEW_WITH_MANUAL_BOT and dl_status == "DL1" and rows:
            try:
                # Extract text samples for NER training
                text_samples = []
                for row in rows[:10]:  # Sample first 10 rows
                    text = " ".join(str(cell) for cell in row if cell)
                    if text:
                        text_samples.append(text)

                # Entity extraction for NER training payloads
                for text in text_samples:
                    entities = extract_training_entities(text, max_entities=40)
                    with SessionLocal() as db_session:
                        db_session.execute(
                            text("""
                                INSERT INTO ner_training_data (text, entities, source, verified, created_at)
                                VALUES (:text, :entities, :source, :verified, :created_at)
                            """),
                            {
                                "text": text,
                                "entities": orjson.dumps(entities).decode(),
                                "source": f"qa_panel_{dataset_id}",
                                "verified": False,  # Will be True after manual review
                                "created_at": datetime.utcnow(),
                            }
                        )
                        db_session.commit()

                record_ml_event(
                    "data_assurance",
                    "ner_training_samples_written",
                    metadata={
                        "dataset_id": dataset_id,
                        "sample_count": len(text_samples),
                    },
                )
            except Exception as e:
                logger.warning(f"[QA] Failed to write NER training data: {e}")
                record_ml_event(
                    "data_assurance",
                    "ner_training_samples_failed",
                    metadata={"dataset_id": dataset_id, "error": str(e)},
                )

        return jsonify({
            "dataset_id": dataset_id,
            "dl_status": dl_status,
            "confidence_score": confidence_score,
            "detected_issues": detected_issues,
            "created_at": datetime.utcnow().isoformat(),
        })

    except Exception as e:
        logger.error(f"[QA] Classification failed: {e}")
        return jsonify({"error": str(e)}), 500


def api_data_assurance_promote():
    """
    Promote verified dataset from DL1 to DL2 after manual review.
    Updates PostgreSQL and marks associated NER training data as verified.
    """
    cert_resp = _require_client_cert("data_assurance_promote")
    if cert_resp:
        return cert_resp
    try:
        from datetime import datetime

        from webapp.parser.utils.db_utils import SessionLocal
        from webapp.parser.utils.privilege_tiers import PrivilegeTier, get_principal_tier

        principal, principal_source, _ = get_request_principal()
        if not principal:
            return jsonify({"error": "Unauthorized"}), 401

        principal_tier = get_principal_tier(principal, principal_source)
        if not tier_satisfies(principal_tier, PrivilegeTier.ADMIN_REVIEWER):
            logger.warning({
                "level": "WARNING",
                "type": "auth",
                "message": "Data assurance promotion denied: insufficient privilege tier.",
                "session_id": None,
                "principal": principal,
                "principal_source": principal_source,
                "required_tier": PrivilegeTier.ADMIN_REVIEWER.name,
                "actual_tier": principal_tier.name,
            })
            return jsonify({
                "error": "Forbidden",
                "required_tier": PrivilegeTier.ADMIN_REVIEWER.name,
                "actual_tier": principal_tier.name,
            }), 403

        data = request.get_json()
        dataset_id = data.get("dataset_id")

        if not dataset_id:
            return jsonify({"error": "dataset_id required"}), 400

        # Update classification status
        with SessionLocal() as db_session:
            result = db_session.execute(
                text("""
                    UPDATE data_assurance_classifications
                    SET dl_status = 'DL2', promoted_at = :promoted_at, reviewer_principal = :reviewer
                    WHERE dataset_id = :dataset_id
                    RETURNING dl_status, confidence_score, detected_issues, created_at, promoted_at
                """),
                {
                    "dataset_id": dataset_id,
                    "promoted_at": datetime.utcnow(),
                    "reviewer": principal,
                }
            )
            row = result.fetchone()
            if not row:
                return jsonify({"error": "Dataset not found"}), 404

            db_session.commit()

            # Mark associated NER training data as verified
            db_session.execute(
                text("""
                    UPDATE ner_training_data
                    SET verified = TRUE
                    WHERE source = :source
                """),
                {"source": f"qa_panel_{dataset_id}"}
            )
            db_session.commit()

        return jsonify({
            "dataset_id": dataset_id,
            "dl_status": "DL2",
            "confidence_score": row[1],
            "detected_issues": orjson.loads(row[2]) if row[2] else [],
            "created_at": row[3].isoformat() if row[3] else None,
            "promoted_at": row[4].isoformat() if row[4] else None,
            "reviewer_principal": principal,
        })

    except Exception as e:
        logger.error(f"[QA] Promotion failed: {e}")
        return jsonify({"error": str(e)}), 500


def api_data_assurance_pending_reviews():
    """
    Fetch pending DL2 reviews (DL1 datasets awaiting manual verification).
    """
    try:
        limit = int(request.args.get("limit", 50))

        from webapp.parser.utils.db_utils import SessionLocal
        with SessionLocal() as db_session:
            result = db_session.execute(
                text("""
                    SELECT dataset_id, dl_status, confidence_score, detected_issues, metadata, created_at
                    FROM data_assurance_classifications
                    WHERE dl_status = 'DL1'
                    ORDER BY created_at DESC
                    LIMIT :limit
                """),
                {"limit": limit}
            )

            pending_reviews = []
            for row in result:
                pending_reviews.append({
                    "dataset_id": row[0],
                    "dl_status": row[1],
                    "confidence_score": row[2],
                    "detected_issues": orjson.loads(row[3]) if row[3] else [],
                    "metadata": orjson.loads(row[4]) if row[4] else {},
                    "created_at": row[5].isoformat() if row[5] else None,
                })

            return jsonify({"pending_reviews": pending_reviews})

    except Exception as e:
        logger.error(f"[QA] Failed to fetch pending reviews: {e}")
        return jsonify({"error": str(e)}), 500


app.config["_FEC_DATA_ASSURANCE_ROUTE_HANDLERS"] = {
    "fec_mappings_review": fec_mappings_review,
    "api_fec_problem_rows": api_fec_problem_rows,
    "api_fec_save_mapping": api_fec_save_mapping,
    "api_data_assurance_classify": api_data_assurance_classify,
    "api_data_assurance_promote": api_data_assurance_promote,
    "api_data_assurance_pending_reviews": api_data_assurance_pending_reviews,
}


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