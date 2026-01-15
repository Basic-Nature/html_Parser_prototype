import os

# Bind to the App Service assigned port (default 8000 for local runs)
port = os.environ.get("PORT", "8000")
bind = f"0.0.0.0:{port}"

# Use sync worker (simple, stable) or gevent for better concurrency
# For WebSocket support, Flask-SocketIO now uses threading (native Python async mode)
worker_class = "sync"

# Leave a single worker by default; Azure load-balances instances externally.
workers = int(os.environ.get("GUNICORN_WORKERS", "1"))

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
