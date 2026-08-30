import os

# Bind to the App Service assigned port (default 8000 for local runs)
port = os.environ.get("PORT", "8000")
bind = f"0.0.0.0:{port}"

# ---------------------------------------------------------------------------
# Worker class & concurrency
# ---------------------------------------------------------------------------
# Flask-SocketIO uses async_mode="threading", so all concurrency within a
# single process is handled by Python threads — no eventlet/gevent needed.
#
# Worker class strategy:
#   "sync"    – 1 request at a time per worker; safe but low throughput.
#   "gthread" – thread-pool inside each worker; recommended when GUNICORN_THREADS > 1.
#               Handles many concurrent users within a *single* process.
#
# ⚠️  Multi-worker mode (GUNICORN_WORKERS > 1) requires a SocketIO message
#     queue so that events emitted by one worker reach clients connected to
#     another.  Set SOCKETIO_USE_DB_QUEUE=true to use the existing PostgreSQL
#     database via kombu (requires kombu>=5.6.2 in requirements).
#     Without a message queue, keep GUNICORN_WORKERS=1 (the default).
# ---------------------------------------------------------------------------
_worker_class = os.environ.get("GUNICORN_WORKER_CLASS", "sync")
worker_class = _worker_class

# Number of worker processes.
# Default is 1; a single gthread worker handles concurrent sessions via threads.
workers = int(os.environ.get("GUNICORN_WORKERS", "1"))

# Thread count per worker — only applied when worker_class="gthread".
# Gunicorn ignores this setting for "sync" workers.
# Default 4 handles typical multi-user concurrency.
threads = int(os.environ.get("GUNICORN_THREADS", "4"))

# Timeouts tuned for larger PDF/OCR runs
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "240"))
keepalive = int(os.environ.get("GUNICORN_KEEPALIVE", "15"))

# Logging to stdout/stderr keeps App Service log streaming happy
accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("GUNICORN_LOGLEVEL", "info")

# Graceful restart knobs (optional overrides)
max_requests = int(os.environ.get("GUNICORN_MAX_REQUESTS", "0"))
max_requests_jitter = int(os.environ.get("GUNICORN_MAX_REQUESTS_JITTER", "0"))
graceful_timeout = int(os.environ.get("GUNICORN_GRACEFUL_TIMEOUT", "90"))


def post_worker_init(worker):
    """Start process-local alert monitoring after Gunicorn worker init."""
    del worker
    from webapp.Smart_Elections_Parser_Webapp import (
        start_alert_monitor_service,
    )

    start_alert_monitor_service()
