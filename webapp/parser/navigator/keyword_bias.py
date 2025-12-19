from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict, Iterable, List

import orjson

BIAS_MAP_PATH = Path(__file__).with_name("navigation_keyword_bias.jsonl")

_cache: List[Dict] = []
_loaded = False
_lock = threading.RLock()


def _iter_lines(path: Path) -> Iterable[Dict]:
    if not path.exists() or not path.is_file():
        return []

    def _gen():
        with path.open("rb") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    obj = orjson.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    yield obj

    return _gen()


def load_keyword_bias() -> List[Dict]:
    global _loaded
    with _lock:
        if _loaded:
            return list(_cache)
        entries = []
        for obj in _iter_lines(BIAS_MAP_PATH):
            selector = obj.get("selector")
            phrases = obj.get("phrases") or []
            if not selector or not isinstance(phrases, list):
                continue
            entries.append(
                {
                    "selector": str(selector),
                    "phrases": [str(p).lower() for p in phrases if isinstance(p, (str, bytes))],
                    "confidence": float(obj.get("confidence", 0.0) or 0.0),
                    "max_wait_ms": obj.get("max_wait_ms"),
                    "autoscroll_ms": obj.get("autoscroll_ms"),
                }
            )
        _cache[:] = entries
        _loaded = True
        return list(_cache)

