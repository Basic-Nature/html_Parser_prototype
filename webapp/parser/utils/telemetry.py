import os
import time
import hashlib
import json
from typing import Any, Dict

try:
    # project config (LOG_DIR) lives at webapp/parser/config.py
    from ..config import LOG_DIR
except Exception:
    LOG_DIR = os.path.join(os.getcwd(), 'logs')

try:
    # local logger singleton used across project
    from .logger_singleton import logger
except Exception:
    logger = None

TELEMETRY_PATH = os.path.join(str(LOG_DIR), 'telemetry.jsonl')
os.makedirs(str(LOG_DIR), exist_ok=True)


def _derive_url_fields(u: str) -> Dict[str, str]:
    out = {"url_hash": None, "url_domain": None}
    try:
        from urllib.parse import urlparse
        p = urlparse(u or "")
        out["url_domain"] = (p.hostname or "")
        out["url_hash"] = hashlib.sha1((u or "").encode('utf-8')).hexdigest()[:12]
    except Exception:
        pass
    return out


def emit_telemetry_event(event: str, payload: Dict[str, Any] | None = None) -> None:
    payload = dict(payload or {})
    payload.setdefault("event", event)
    payload.setdefault("ts_ms", int(time.time() * 1000))
    payload.setdefault("ts_iso", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    # Derive safe URL tokens if present
    url = payload.get("url") or payload.get("target_url") or payload.get("source_url")
    if isinstance(url, str) and url:
        derived = _derive_url_fields(url)
        payload.setdefault("url_domain", derived.get("url_domain"))
        payload.setdefault("url_hash", derived.get("url_hash"))
        # By default do not store full URL to telemetry to avoid PII/leakage
        if os.environ.get("TELEMETRY_INCLUDE_URL", "false").lower() in ("1", "true", "yes"):
            payload.setdefault("url_full", url)
    try:
        if logger is not None:
            logger.info({"level": "INFO", "type": "telemetry", "message": event, **payload})
    except Exception:
        pass

    # Best-effort append to telemetry JSONL
    try:
        with open(TELEMETRY_PATH, 'ab') as f:
            line = json.dumps(payload, default=str, ensure_ascii=False).encode('utf-8') + b"\n"
            f.write(line)
    except Exception:
        # swallow telemetry write errors
        pass
