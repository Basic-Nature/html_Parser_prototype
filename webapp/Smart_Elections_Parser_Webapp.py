from __future__ import annotations
# Smart_Elections_Parser_Webapp.py
# -----------------------------------------------------------
# Web Application for Smart Elections Parser
# -----------------------------------------------------------
# 1. Imports & Environment Setup
from datetime import datetime, timezone
import secrets
from flask import (
    Flask, render_template, request, redirect, session,
    url_for, flash, send_file, send_from_directory,
    jsonify, Response, g
)
from flask_socketio import emit, SocketIO, join_room
import orjson
import os
import time
import re
from werkzeug.exceptions import NotFound
from threading import Thread, RLock, Event
import gzip
from queue import Queue
import psycopg2
import threading
import re


# Project-specific imports
from webapp.parser import data_manager
from webapp.parser.utils.shared_logic import (
    safe_get, safe_split, safe_lower, safe_is_set, safe_append,
    safe_sid, safe_rsplit, safe_strip
)
from webapp.parser.web_pipeline import (
    process_urls_for_web, cancel_processing, cancellation_manager,
)
from webapp.parser.config import (
    INPUT_DIR, OUTPUT_DIR, UPLOADS_DIR, URL_LIST_FILE,
    SUPPORTED_FORMATS, DATA_API_URL, POSTGRES_DB, POSTGRES_USER_RAW, POSTGRES_PASSWORD_RAW,
    POSTGRES_HOST, POSTGRES_PORT, RUN_HISTORY_FILE
)
from webapp.parser.utils.logger_singleton import logger, console, prompt

# 2. Flask App & SocketIO Initialization
app = Flask(__name__)
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    ping_interval=20,
    ping_timeout=60,
    async_mode=None  # let it auto-select, but ensures consistent options
)

# 3. Session & State Management
session_prompt_queues: dict[str, Queue] = {}
session_threads: dict[str, Thread] = {}
active_sessions_backend: set[str] = set()
session_last_active: dict[str, float] = {}
session_metadata: dict[str, dict] = {}
session_logs: dict[str, list] = {}

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

# Output bypass (per logical session)
output_bypass_sessions: set[str] = set()

# Manual format file source per session ('input' or 'uploads')
manual_source_sessions: dict[str, str] = {}

# Map socket connection -> logical session_id room
sid_to_session: dict[str, str] = {}

# Map IP/UA fingerprint -> default logical session; and per-session emitters
ip_ua_to_session: dict[str, str] = {}
session_emitters: dict[str, callable] = {}
_registry_lock = RLock()

# Thread -> session mapping (for logger emits outside Flask request context)
thread_session_map: dict[int, str] = {}

# 4. Utility Functions
def is_owner(sid, username):
    meta = safe_get(session_metadata, sid, {})
    return safe_get(meta, 'username') == username

def create_session_metadata(sid, username=None):
    session_metadata[sid] = {
        "session_id": sid,
        "username": username or "anonymous",
        "created": datetime.now(timezone.utc).isoformat(),
        "last_active": time.time(),
        "parser_status": "idle",
        "locked": False,
    }

def cleanup_sessions():
    now = time.time()
    expired = []
    for sid, last_active in list(session_last_active.items()):
        if now - last_active > SESSION_TIMEOUT:
            expired.append(sid)
            active_sessions_backend.discard(sid)
            del session_last_active[sid]
            session_metadata.pop(sid, None)
            session_logs.pop(sid, None)
    if expired:
        emit('session_expired', {'expired_sessions': expired}, broadcast=True)

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
    with _registry_lock:
        sid = None
        if isinstance(data, dict):
            sid = safe_get(data, 'session_id')
        if isinstance(sid, str) and sid:
            sid_to_session[socket_sid] = sid
            return sid
        mapped = sid_to_session.get(socket_sid)
        if isinstance(mapped, str) and mapped:
            return mapped
        cookie_sid = session.get('logical_session_id')
        if isinstance(cookie_sid, str) and cookie_sid:
            sid_to_session[socket_sid] = cookie_sid
            return cookie_sid
        fp = client_fingerprint()
        fp_sid = ip_ua_to_session.get(fp)
        if isinstance(fp_sid, str) and fp_sid:
            sid_to_session[socket_sid] = fp_sid
            session['logical_session_id'] = fp_sid
            return fp_sid
        if not create_if_missing:
            return None
        new_sid = 'sess_' + os.urandom(6).hex()
        ip_ua_to_session[fp] = new_sid
        sid_to_session[socket_sid] = new_sid
        session['logical_session_id'] = new_sid
        if new_sid not in session_metadata:
            create_session_metadata(new_sid)
        return new_sid

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

def normalize_log_obj(raw) -> dict:
    obj = raw
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
                obj["type"] = "cancellation"
            elif "error" in mlow:
                obj["type"] = "error"
            elif "warning" in mlow:
                obj["type"] = "warning"
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

    # Normalize level/type
    obj["level"] = str(obj.get("level", "INFO")).upper()
    obj["type"] = str(obj.get("type", "other")).lower()

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
    logs = session_logs.setdefault(session_id, [])
    logs.append(log_obj)
    if len(logs) > MAX_LOGS_PER_SESSION:
        del logs[0: len(logs) - TRIM_TO]

def emit_log(level: str, type_: str, message: str, session_id: str | None, **extra):
    base = {
        "level": level.upper(),
        "type": (type_ or "status").lower(),
        "message": message,
        "session_id": session_id,
        "timestamp": int(time.time() * 1000)
    }
    base.update(extra)
    norm = normalize_log_obj(base)
    if session_id:
        store_log(session_id, norm)
        socketio.emit('parser_output', norm, room=session_id)
    else:
        socketio.emit('parser_output', norm)

def _heartbeat_loop():
    if not HEARTBEAT_ENABLED:
        return
    while not _shutdown_event.is_set():
        time.sleep(HEARTBEAT_INTERVAL)
        now_ms = int(time.time() * 1000)
        for sid in list(active_sessions_backend):
            if sid not in session_metadata:
                continue
            # Emit ONLY lightweight heartbeat (no parser_output log line)
            try:
                socketio.emit('session_heartbeat', {"session_id": sid, "timestamp": now_ms}, room=sid)
            except Exception:
                pass
            
def route_logger_emit(line):
    try:
        if isinstance(line, str) and not line.strip().startswith("{"):
            obj = {"level": "INFO", "type": "raw", "message": line}
        else:
            try:
                obj = orjson.loads(line) if isinstance(line, str) else line
            except Exception:
                obj = {"level": "INFO", "type": "raw", "message": str(line)}
        # Force normalization (adds ms timestamp / inference)
        obj = normalize_log_obj(obj)
        sid = obj.get("session_id")
        if not sid:
            # 1) Thread mapping (background worker threads)
            sid = thread_session_map.get(threading.get_ident())
            if sid:
                obj["session_id"] = sid
        if not sid:
            try:
                curr_sid = safe_sid()
            except Exception:
                curr_sid = getattr(request, 'sid', None)
            if isinstance(curr_sid, str):
                logical = sid_to_session.get(curr_sid)
                if logical:
                    obj["session_id"] = sid = logical
        if not sid:
            # 2) Pattern extraction from message (e.g. "... (session_id=abcd)")
            msg = obj.get("message")
            if isinstance(msg, str):
                m = re.search(r'\bsession_id=([a-zA-Z0-9_\-]{6,40})\b', msg)
                if m:
                    sid = m.group(1)
                    obj["session_id"] = sid
        if sid:
            store_log(sid, obj)
            socketio.emit('parser_output', obj, room=sid)
        else:
            socketio.emit('parser_output', obj)
    except Exception:
        pass

def get_prompt_queue(session_id):
    if session_id not in session_prompt_queues:
        session_prompt_queues[session_id] = Queue()
    return session_prompt_queues[session_id]

def broadcast_sessions():
    """
    Safe global session list broadcast.
    Uses socketio.emit so it can be called from worker / background threads
    without a Flask request context.
    """
    try:
        sessions = [
            session_metadata[sid]
            for sid in active_sessions_backend
            if sid in session_metadata
        ]
        socketio.emit('session_list', {'sessions': sessions})
    except Exception as e:
        logger.error({
            "level": "ERROR",
            "type": "broadcast",
            "message": f"Failed to broadcast sessions: {e}",
            "session_id": None
        })

def lock_session(sid):
    session_metadata[sid]['locked'] = True
    session_metadata[sid]['parser_status'] = 'running'
    broadcast_sessions()

def unlock_session(sid):
    if sid in session_metadata:
        session_metadata[sid]['locked'] = False
        session_metadata[sid]['parser_status'] = 'idle'
    broadcast_sessions()

def safe_is_alive(session_id: str) -> bool:
    if not session_id:
        return False
    meta = session_metadata.get(session_id)
    if not meta:
        return False
    last_active = session_last_active.get(session_id)
    if last_active and (time.time() - last_active) > SESSION_TIMEOUT:
        return False
    thread: Thread = session_threads.get(session_id)
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
    return session_id in output_bypass_sessions

def get_manual_source(session_id: str) -> str:
    return manual_source_sessions.get(session_id, 'input')

def log_parser_status(msg, session_id=None, rich=False) -> None:
    if logger.mode == "webapp":
        status_msg = f"{msg} (session_id={session_id})" if session_id else msg
        logger.info({
            "level": "INFO",
            "message": status_msg,
            "session_id": session_id
        })
    elif rich:
        console.panel(f"{msg}\nSession: {session_id}", title="Parser Status")
    else:
        log_msg = f"{msg} (session_id={session_id})" if session_id else msg
        logger.info(log_msg)

def get_all_file_lists() -> dict:
    return {
        "input_files": os.listdir(INPUT_DIR),
        "output_files": os.listdir(OUTPUT_DIR),
        "uploaded_files": os.listdir(UPLOADS_DIR),
    }

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
        seen = set(); out=[]
        for x in seq:
            if x not in seen:
                seen.add(x); out.append(x)
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
    Harden outbound responses and ensure UTF-8 charset.
    """
    if request.environ.get("wsgi.websocket"):
        return response

    textual_mimes = {
        "text/html",
        "text/css",
        "application/javascript",
        "text/javascript",
        "application/json",
        "text/plain",
        "application/xhtml+xml",
    }
    if response.mimetype in textual_mimes:
        ct = (response.content_type or "")
        if "charset=" not in ct.lower():
            response.headers["Content-Type"] = f"{response.mimetype}; charset=utf-8"

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

    response.headers.pop("X-XSS-Protection", None)

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

    if request.is_secure:
        response.headers.setdefault(
            "Strict-Transport-Security",
            "max-age=63072000; includeSubDomains; preload"
        )

    if response.mimetype and response.mimetype.startswith(("text/html", "application/xhtml")):
        response.headers["Content-Security-Policy"] = build_csp(relaxed, nonce)
    else:
        response.headers.pop("Content-Security-Policy", None)

    vary = response.headers.get("Vary", "")
    vary_tokens = {t.strip() for t in vary.split(",") if t.strip()}
    if "Cookie" not in vary_tokens:
        vary_tokens.add("Cookie")
        response.headers["Vary"] = ", ".join(sorted(vary_tokens))
    # Normalize Engine.IO polling content-type (webhint wants lowercase utf-8)
    if request.path.startswith("/socket.io/"):
        ct = response.headers.get("Content-Type","")
        if ct.lower().startswith("text/plain"):
            # force lowercase utf-8
            if "charset=" in ct.lower():
                response.headers["Content-Type"] = "text/plain; charset=utf-8"
            else:
                response.headers["Content-Type"] = ct.split(";")[0] + "; charset=utf-8"
    return response

# Data Management Utilities
def add_url() -> None:
    url = input("Enter new URL to add: ").strip()
    if url:
        with open(URL_LIST_FILE, "a", encoding="utf-8") as f:
            f.write(url + "\n")
        log_parser_status(f"[ADDED] {url}")

def allowed_file(filename) -> bool:
    parts = safe_rsplit(filename, '.', 1)
    ext = safe_lower(parts[1]) if len(parts) > 1 else ''
    return filename and ext in SUPPORTED_FORMATS and len(filename) < 128

def get_url_list() -> list[str]:
    if not os.path.exists(URL_LIST_FILE):
        return []
    with open(URL_LIST_FILE, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    return urls

def list_urls() -> list[str]:
    if not os.path.exists(URL_LIST_FILE):
        log_parser_status("[INFO] No urls.txt found.")
        return []
    with open(URL_LIST_FILE, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    log_parser_status("\n[URLS.TXT ENTRIES]")
    for i, url in enumerate(urls, 1):
        log_parser_status(f"{i}. {url}")
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

@app.route("/api/warehouse_election_results", methods=["GET"])
def api_warehouse_election_results():
    state = request.args.get("state")
    county = request.args.get("county")
    contest = request.args.get("contest")
    where = []
    params = []
    if state: where.append("state = %s"); params.append(state)
    if county: where.append("county = %s"); params.append(county)
    if contest: where.append("contest ILIKE %s"); params.append(f"%{contest}%")
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
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
        log_parser_status(f"DB error: {e}")
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
    resp = Response(payload, mimetype="application/manifest+json")
    resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    resp.headers["ETag"] = etag
    return resp

@app.route("/upload/input", methods=["POST"])
def upload_to_input() -> str:
    file = request.files.get("file")
    log_parser_status(f"Upload to input: {file.filename if file else 'No file'}")
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
    if file and allowed_file(file.filename):
        filename = file.filename
        file.save(os.path.join(OUTPUT_DIR, filename))
        flash(f"File '{filename}' uploaded to output folder.", "success")
    else:
        flash("Invalid file type or no file selected.", "danger")
    return redirect(request.referrer or url_for("run_parser"))

@app.route("/upload/uploads", methods=["POST"])
def upload_to_uploads() -> str:
    file = request.files.get("file")
    if file and allowed_file(file.filename):
        filename = file.filename
        file.save(os.path.join(UPLOADS_DIR, filename))
        flash(f"File '{filename}' uploaded to uploads folder.", "success")
    else:
        flash("Invalid file type or no file selected.", "danger")
    return redirect(request.referrer or url_for("run_parser"))

@app.route("/health")
def health() -> str:
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

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
    # Derive aggregated runs: pair start/end by run_id
    aggregated = {}
    for evt in runs:
        rid = evt.get("run_id")
        if not rid:
            continue
        slot = aggregated.setdefault(rid, {"run_id": rid})
        slot.update(evt)
    # Sort by latest timestamp descending
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
    logs = session_logs.get(sid, [])
    if not isinstance(logs, list):
        logs = []
    emit('session_history', {'session_id': sid, 'logs': logs}, room=socket_room)

@socketio.on('clone_session')
def handle_clone_session(data) -> None:
    old_sid = str(data['session_id'])
    if not isinstance(old_sid, str):
        logger.warning(f"Invalid session_id type: {type(old_sid)} value: {old_sid}")
        return
    new_sid = 'sess_' + os.urandom(6).hex()
    session_metadata[new_sid] = dict(session_metadata[old_sid])
    session_metadata[new_sid]['session_id'] = new_sid
    session_metadata[new_sid]['created'] = datetime.now(timezone.utc).isoformat()
    session_metadata[new_sid]['last_active'] = time.time()
    session_logs[new_sid] = list(session_logs.get(old_sid, []))
    active_sessions_backend.add(new_sid)
    session_last_active[new_sid] = time.time()
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
        logger.warning(f"Invalid session_id resolved: {sid}")
        return
    join_room(sid)
    active_sessions_backend.add(sid)
    session_last_active[sid] = time.time()
    if sid not in session_metadata:
        create_session_metadata(sid, safe_get(data, 'username'))

@socketio.on('get_sessions')
def handle_get_sessions():
    cleanup_sessions()
    sessions = [session_metadata[sid] for sid in active_sessions_backend if sid in session_metadata]
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
        if requested and requested in session_metadata:
            revived = requested
            session_last_active[revived] = time.time()
            cancellation_manager.remove(revived)
            prompt.clear_prompt_session(revived)
            log_parser_status("Re-associated socket with existing session", revived)
            emit('session_id', {'session_id': revived})
        # DO NOT create a new session automatically here
        resolved = resolve_session_id({'session_id': revived} if revived else {}, create_if_missing=False)
        if resolved:
            session_last_active[resolved] = time.time()
        # Send currently active sessions (only those explicitly joined earlier)
        active = [session_metadata[sid] for sid in active_sessions_backend if sid in session_metadata]
        emit('session_list', {'sessions': active})
        log_parser_status("Socket connected (no auto session creation)", resolved)
    except Exception as e:
        emit('parser_output', {
            "level": "ERROR",
            "message": f"Connect error: {e}",
            "session_id": None
        }, room=getattr(request, 'sid', None))

@socketio.on('disconnect')
def handle_disconnect() -> None:
    try:
        req_sid = safe_sid()
    except Exception:
        req_sid = getattr(request, 'sid', None)
        if not isinstance(req_sid, str):
            req_sid = None
    logical = sid_to_session.pop(req_sid, None)
    log_parser_status(f"Client disconnected (socket_sid={req_sid}, session_id={logical})", logical)
    prompt.clear_prompt_session(logical or req_sid)
    cancellation_manager.remove(logical or req_sid)
    with _registry_lock:
        if logical in session_emitters:
            session_emitters.pop(logical, None)

@socketio.on('set_output_mode')
def handle_set_output_mode(data) -> None:
    mode = safe_lower(safe_get(data, "mode", "live"))
    valid_modes = {"live", "batch"}
    session_id = resolve_session_id(data, create_if_missing=False)
    if not session_id:
        emit_log("ERROR", "status", "No session_id provided.", None)
        return
    if mode in valid_modes:
        session['output_mode'] = mode
        emit_log("INFO", "status", f"Output mode set to {mode}.", session_id)
    else:
        emit_log("ERROR", "status", "Invalid output mode.", session_id)

@socketio.on('parser_prompt')
def handle_parser_prompt(data) -> None:
    session_id = resolve_session_id(data, create_if_missing=False)
    value = data.get("value", "") if isinstance(data, dict) else data
    if not session_id or session_id not in session_metadata:
        emit_log("ERROR", "prompt", "Invalid or unknown session_id for prompt.", None, color="#eb4f43")
        return
    prompt_session = prompt.prompt_sessions.get(session_id)
    if prompt_session:
        prompt_session.set_response(value)

@socketio.on('cancel_parser')
def handle_cancel_parser(data=None) -> None:
    session_id = resolve_session_id(data or {}, create_if_missing=False)
    if not session_id:
        emit_log("ERROR", "cancel", "No session_id provided for cancel.", None)
        return
    cancel_processing(session_id)
    emit_log("CANCELLED", "cancel", "Cancellation requested", session_id)
    if session_id in session_threads:
        session_threads.pop(session_id, None)
    if session_id in session_prompt_queues:
        session_prompt_queues.pop(session_id, None)
    with _registry_lock:
        session_emitters.pop(session_id, None)
    unlock_session(session_id)
    cleanup_sessions()

@socketio.on('toggle_output_bypass')
def handle_toggle_output_bypass(data=None):
    sid = resolve_session_id(data or {}, create_if_missing=False)
    if not sid:
        emit_log("ERROR", "error", "No session_id for output bypass toggle.", None)
        return
    if sid in output_bypass_sessions:
        output_bypass_sessions.remove(sid)
        state = False
    else:
        output_bypass_sessions.add(sid)
        state = True
    emit('output_bypass_state', {"session_id": sid, "output_bypass": state}, room=sid)
    emit_log("INFO", "status", f"Output bypass {'ENABLED' if state else 'DISABLED'}", sid)

@socketio.on('set_manual_source')
def handle_set_manual_source(data=None):
    sid = resolve_session_id(data or {}, create_if_missing=False)
    source = safe_lower(safe_get(data or {}, 'file_source', ''))
    if not sid or source not in {'input', 'uploads'}:
        emit_log("ERROR", "input", "Invalid manual source update.", sid)
        return
    manual_source_sessions[sid] = source
    emit_log("INFO", "status", f"Manual file source set to '{source}'.", sid)
    socketio.emit('manual_source_state', {"session_id": sid, "file_source": source}, room=sid)

@socketio.on('delete_session')
def handle_delete_session(data) -> None:
    sid = resolve_session_id(data or {}, create_if_missing=False)
    if not isinstance(sid, str):
        logger.warning(f"Invalid session_id type in delete: {type(sid)} value: {sid}")
        return
    active_sessions_backend.discard(sid)
    session_last_active.pop(sid, None)
    session_metadata.pop(sid, None)
    session_prompt_queues.pop(sid, None)
    session_threads.pop(sid, None)
    session_logs.pop(sid, None)
    with _registry_lock:
        session_emitters.pop(sid, None)
    emit('session_deleted', {'session_id': sid}, broadcast=True)

@socketio.on('run_parser')
def handle_run_parser(data=None) -> None:
    cleanup_sessions()
    payload = data if isinstance(data, dict) else {}
    session_id = resolve_session_id(payload, create_if_missing=True)
    if not isinstance(session_id, str):
        emit_log("ERROR", "status", "Unable to resolve session_id.", None)
        return
    join_room(session_id)
    try:
        socket_sid = safe_sid()
    except Exception:
        socket_sid = getattr(request, 'sid', None)
    if isinstance(socket_sid, str):
        sid_to_session[socket_sid] = session_id
    if session_id not in session_metadata:
        create_session_metadata(session_id)
    meta = safe_get(session_metadata, session_id, {})
    if safe_get(meta, 'locked') and safe_is_alive(session_id):
        emit_log("ERROR", "status", "Session is locked. Wait for current job to finish.", session_id)
        return
    if safe_is_alive(session_id):
        emit_log("WARNING", "status", "Parser already running for this session.", session_id)
        return
    requested_source = safe_lower(safe_get(payload, 'file_source', get_manual_source(session_id)))
    if requested_source not in {'input', 'uploads'}:
        requested_source = 'input'
    manual_source_sessions[session_id] = requested_source
    output_bypass_flag = is_output_bypassed(session_id)
    lock_session(session_id)
    log_parser_status("Parser connected. Starting parser run...", session_id, rich=True)
    cancel_flag = cancellation_manager.get_flag(session_id)
    prompt_queue = get_prompt_queue(session_id)

    def emit_to_socketio(line):
        sid = session_id
        try:
            if isinstance(line, dict):
                raw_obj = line
            elif isinstance(line, str):
                txt = line.strip()
                if txt.startswith("{"):
                    try:
                        raw_obj = orjson.loads(txt)
                    except Exception:
                        raw_obj = {"level": "INFO", "type": "raw", "message": line}
                else:
                    raw_obj = {"level": "INFO", "type": "raw", "message": line}
            else:
                raw_obj = {"level": "INFO", "type": "raw", "message": str(line)}
            raw_obj.setdefault("session_id", sid)
            norm = normalize_log_obj(raw_obj)
            if norm.get("type") == "heartbeat":
                store_log(sid, norm)
                socketio.emit('session_heartbeat', {"session_id": sid, "timestamp": norm["timestamp"]}, room=sid)
                return
            store_log(sid, norm)
            socketio.emit('parser_output', norm, room=sid)
        except Exception as e:
            # Log the emit failure via dispatcher (so you can see it in CLI & UI)
            logger.error({
                "level": "ERROR",
                "type": "exception",
                "message": f"emit_to_socketio failure: {e}",
                "session_id": sid
            })
    # Register per-session emitter (used by prompt / manual emits)
    with _registry_lock:
        session_emitters[session_id] = emit_to_socketio
    # Install dispatcher only ONCE globally (idempotent)
    logger.set_mode("webapp")
    logger.set_format("json")
    logger.set_socketio_emit_func(route_logger_emit)
    prompt.set_mode("webapp")
    prompt.set_socketio_emit_func(lambda msg: socketio.emit(
        'parser_output',
        normalize_log_obj(msg if isinstance(msg, dict) else {
            "level": "PROMPT",
            "type": "prompt",
            "message": str(msg),
            "session_id": session_id
        }),
        room=session_id
    ))

    emit_log(
        "INFO",
        "status",
        f"Launching parser (source={requested_source}, output_bypass={'on' if output_bypass_flag else 'off'})",
        session_id
    )
    run_id = f"run_{int(time.time()*1000)}"
    start_ts = datetime.now(timezone.utc).isoformat()
    log_run_event({
        "type": "start",
        "run_id": run_id,
        "session_id": session_id,
        "ts": start_ts,
        "source": requested_source,
        "output_bypass": output_bypass_flag,
        "status": "running"
    })
    
    def worker_wrapper():
        start_time = time.time()
        # Map this worker thread to session for logger emits lacking session_id
        thread_session_map[threading.get_ident()] = session_id
        try:
            process_urls_for_web(
                prompt_queue,
                session_id,
                cancel_flag,
                emit_func=emit_to_socketio,
                output_bypass=output_bypass_flag,
                manual_source=requested_source,
                disable_internal_heartbeat=True
            )
            emit_log("INFO", "status", "Parser run completed.", session_id)
            status = "ok"
            err = None
        except Exception as e:
            emit_log("ERROR", "exception", f"Parser run failed: {e}", session_id)
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
                "duration_ms": duration_ms
            })
            unlock_session(session_id)
            thread_session_map.pop(threading.get_ident(), None)

    thread = Thread(target=worker_wrapper, name=f"parser-{session_id}", daemon=True)
    thread.start()
    session_threads[session_id] = thread

# Heartbeat thread startup (idempotent)
if 'heartbeat_thread' not in globals() or not isinstance(globals().get('heartbeat_thread'), Thread) or not globals()['heartbeat_thread'].is_alive():
    if HEARTBEAT_ENABLED:
        heartbeat_thread = Thread(target=_heartbeat_loop, name="heartbeat-loop", daemon=True)
        heartbeat_thread.start()
        
# 7. Main Entrypoint
if __name__ == "__main__":
    try:
        port = int(os.environ.get("PORT", 5000))
        socketio.run(app, host="0.0.0.0", port=port)
    finally:
        _shutdown_event.set()