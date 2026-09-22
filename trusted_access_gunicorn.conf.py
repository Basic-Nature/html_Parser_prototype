from __future__ import annotations

import os

port = os.environ.get("PORT", "8000")
bind = f"0.0.0.0:{port}"

worker_class = "sync"
workers = 1
threads = 4

timeout = 240
keepalive = 15
graceful_timeout = 90

accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("GUNICORN_LOGLEVEL", "info")

max_requests = 0
max_requests_jitter = 0
preload_app = False
