"""Generic server-approved CSV dispatch. Jurisdiction-neutral by construction.

The server owns the selected source ID, URL and projection. This module never
accepts browser URLs or dispatches to jurisdiction/contest-specific parsers.
"""
from __future__ import annotations
import hashlib
from typing import Any, Callable

class ApprovedCSVDispatchError(ValueError):
    pass


def parse_runtime_approved_csv(runtime: Any, payload: bytes, digest: str, parse_generic: Callable[..., Any]):
    sid = getattr(runtime, "registry_source_id", None)
    url = getattr(runtime, "approved_target_url", None)
    projection = getattr(runtime, "source_projection", None)
    if not isinstance(sid, str) or not sid.strip() or not isinstance(url, str) or not url.startswith("https://") or not isinstance(projection, dict):
        raise ApprovedCSVDispatchError("CSV_SERVER_APPROVED_SOURCE_CONTEXT_MISSING")
    if not isinstance(payload, bytes) or not isinstance(digest, str) or hashlib.sha256(payload).hexdigest() != digest:
        raise ApprovedCSVDispatchError("CSV_ACQUIRED_SOURCE_DIGEST_MISMATCH")
    return parse_generic(payload, contest=projection.get("contest", ""),
                         year=projection.get("year", ""), state=projection.get("state", ""),
                         source_sha256=digest, source_url=url)
