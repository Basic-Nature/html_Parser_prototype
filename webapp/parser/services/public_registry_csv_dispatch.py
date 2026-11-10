"""Explicit server-approved CSV selector; exact Virginia 2024 Senate source only.

Never accepts a client-provided URL or discovers a new source. This module is
inactive until reviewed, applied, and protected by the production approval route.
Other approved CSV sources retain the existing strict generic decoder/parser.
"""
from __future__ import annotations
import hashlib
from typing import Any, Callable

VA_SOURCE_ID_SHA = "569cb4cb878d78535d833bda69a24fb484fa0abc2e0f6a17a0a97a3c154a587e"
VA_URL_SHA = "6c7aa1279ae1b19f66d46571e7d9b15da4177b0465770c629d2aa445c34582cc"
VA_SOURCE_SHA = "95e1128a9aa40b72a22ac6be6a5a6b5c1238722b6473070fab5969616c3d43db"

class ApprovedCSVDispatchError(ValueError):
    pass


def parse_runtime_approved_csv(runtime: Any, payload: bytes, digest: str, parse_generic: Callable[..., Any]):
    """Select the pinned source adapter without weakening generic CSV policy.

    Registry selection must already be performed on the server. A match to
    EITHER Virginia identifier triggers a requirement for BOTH, fail-closed.
    Unknown sources never get CP1252 fallback and remain strict UTF-8.
    """
    sid = getattr(runtime, "registry_source_id", None)
    url = getattr(runtime, "approved_target_url", None)
    projection = getattr(runtime, "source_projection", None)
    if not isinstance(sid, str) or not isinstance(url, str) or not isinstance(projection, dict):
        raise ApprovedCSVDispatchError("CSV_SERVER_APPROVED_SOURCE_CONTEXT_MISSING")
    if not isinstance(payload, bytes) or not isinstance(digest, str) or hashlib.sha256(payload).hexdigest() != digest:
        raise ApprovedCSVDispatchError("CSV_ACQUIRED_SOURCE_DIGEST_MISMATCH")
    sid_match = hashlib.sha256(sid.encode("utf-8")).hexdigest() == VA_SOURCE_ID_SHA
    url_match = hashlib.sha256(url.encode("utf-8")).hexdigest() == VA_URL_SHA
    if sid_match or url_match:
        if not (sid_match and url_match):
            raise ApprovedCSVDispatchError("CSV_VIRGINIA_APPROVED_REGISTRY_PAIR_MISMATCH")
        if not (projection.get("state") == "Virginia" and str(projection.get("year")) == "2024"
                and projection.get("contest") == "Senate"):
            raise ApprovedCSVDispatchError("CSV_VIRGINIA_SERVER_CONTEST_SCOPE_MISMATCH")
        if digest != VA_SOURCE_SHA:
            raise ApprovedCSVDispatchError("CSV_VIRGINIA_PINNED_SOURCE_CHANGED_REVIEW_REQUIRED")
        from webapp.parser.services.virginia_2024_senate_csv import parse_virginia_2024_senate
        return parse_virginia_2024_senate(payload, source_url=url, source_sha256=digest)
    return parse_generic(
        payload, contest=projection.get("contest", ""),
        year=projection.get("year", ""), state=projection.get("state", ""),
        source_sha256=digest, source_url=url,
    )
