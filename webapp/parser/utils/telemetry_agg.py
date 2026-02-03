import json
import os
import time
from typing import Any, Dict

try:
    from ..config import LOG_DIR
except Exception:
    LOG_DIR = os.path.join(os.getcwd(), 'logs')

AGG_PATH = os.path.join(str(LOG_DIR), 'telemetry_counters.json')
os.makedirs(str(LOG_DIR), exist_ok=True)

def _read() -> Dict[str, Any]:
    try:
        if not os.path.exists(AGG_PATH):
            return {}
        with open(AGG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

def _write(data: Dict[str, Any]) -> None:
    tmp = AGG_PATH + '.tmp'
    try:
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, AGG_PATH)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass

def get_counters() -> Dict[str, Any]:
    return _read()

def increment_counter(name: str, amount: int = 1) -> None:
    if os.environ.get('ENABLE_TELEMETRY_AGG', 'true').lower() not in ('1', 'true', 'yes'):
        return
    data = _read()
    try:
        val = int(data.get(name, 0))
    except Exception:
        val = 0
    data[name] = val + int(amount)
    data.setdefault('last_updated_ms', int(time.time() * 1000))
    _write(data)
    # Also push to Prometheus if available
    try:
        if os.environ.get('ENABLE_PROMETHEUS', 'false').lower() in ('1', 'true', 'yes'):
            try:
                from .metrics_prom import increment_prom_counter
                increment_prom_counter(name, int(amount))
            except Exception:
                pass
    except Exception:
        pass

def set_counter(name: str, value: int) -> None:
    data = _read()
    data[name] = int(value)
    data.setdefault('last_updated_ms', int(time.time() * 1000))
    _write(data)

def reset_counters() -> None:
    _write({})
