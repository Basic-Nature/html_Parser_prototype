# Safe browser-console event observation boundary.
#
# This service does not invoke Playwright and does not create another raw
# console artifact. The existing URL-glimpse JSON remains the sole raw bounded
# console persistence. Explicit observation emits deterministic hashes/counts
# only and excludes raw console text and raw URLs.

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable


BROWSER_CONSOLE_EVENT_OBSERVATION_CONTRACT = "browser_console_event_observation_v1"
BROWSER_CONSOLE_EVENT_OBSERVATION_AUTHORITY = "NONCANONICAL_OBSERVATION"

EVENT_LIST_IDENTITY_SEMANTICS = (
    "SHA256_OF_CANONICAL_JSON_BOUNDED_CONSOLE_EVENT_LIST_V1"
)
REQUESTED_URL_IDENTITY_SEMANTICS = (
    "SHA256_OF_UTF8_REQUESTED_NAVIGATION_URL_ARGUMENT_V1"
)
FINAL_URL_IDENTITY_SEMANTICS = (
    "SHA256_OF_UTF8_FINAL_PLAYWRIGHT_PAGE_URL_V1"
)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_event_list_bytes(events: list[dict[str, Any]]) -> bytes:
    return (
        json.dumps(
            events,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def _event_type_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        event_type = str(event.get("type") or "")
        counts[event_type] = counts.get(event_type, 0) + 1
    return dict(sorted(counts.items()))


def observe_browser_console_event_if_requested(
    events: list[dict[str, Any]],
    *,
    requested_url: str,
    final_url: str,
    capture_role: str,
    observation_emit_func: Callable[[dict[str, Any]], Any] | None = None,
) -> list[dict[str, Any]]:
    # Dormant path first: no validation, traversal, serialization, hashing,
    # URL access/normalization, or copying.
    if observation_emit_func is None:
        return events

    if not callable(observation_emit_func):
        raise TypeError("observation_emit_func must be callable or None")
    if not isinstance(events, list):
        raise TypeError("events must be a list")
    if not isinstance(requested_url, str):
        raise TypeError("requested_url must be str")
    if not isinstance(final_url, str):
        raise TypeError("final_url must be str")

    canonical_events = _canonical_event_list_bytes(events)
    payload = {
        "contract": BROWSER_CONSOLE_EVENT_OBSERVATION_CONTRACT,
        "authority": BROWSER_CONSOLE_EVENT_OBSERVATION_AUTHORITY,
        "canonical": False,
        "event_count": len(events),
        "event_type_counts": _event_type_counts(events),
        "event_list_sha256": _sha256(canonical_events),
        "event_list_identity_semantics": EVENT_LIST_IDENTITY_SEMANTICS,
        "requested_url_sha256": _sha256(requested_url.encode("utf-8")),
        "requested_url_identity_semantics": REQUESTED_URL_IDENTITY_SEMANTICS,
        "final_url_sha256": _sha256(final_url.encode("utf-8")),
        "final_url_identity_semantics": FINAL_URL_IDENTITY_SEMANTICS,
        "capture_role": str(capture_role),
        "raw_console_text_included": False,
        "raw_requested_url_included": False,
        "raw_final_url_included": False,
        "filesystem_path_included": False,
        "automatic_timestamp": False,
    }
    observation_emit_func(payload)
    return events
