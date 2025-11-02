"""
structure_cache.py
Lightweight in-memory cache + signature helper for table structures.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List

from .detect import normalize_header

_STRUCTURE_CACHE: Dict[str, Dict[str, Any]] = {}

def table_signature(headers: List[str]) -> str:
    norm = [normalize_header(h) for h in headers or []]
    sig_str = "|".join(norm)
    return hashlib.sha1(sig_str.encode("utf-8")).hexdigest()

def cache_table_structure(domain: str, signature: str, payload: Any):
    if not domain or not signature:
        return
    _STRUCTURE_CACHE.setdefault(domain, {})
    _STRUCTURE_CACHE[domain][signature] = payload

def get_cached_structure(domain: str, signature: str):
    return _STRUCTURE_CACHE.get(domain, {}).get(signature)