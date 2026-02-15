from __future__ import annotations

from typing import Any, Dict

_PARITY_CONTEXT_KEY = "_router_parity_note"
_PARITY_METADATA_KEY = "router_parity_note"
_MAX_PARITY_LEN = 160


def safe_parity_note(note: str | None) -> str | None:
    if not isinstance(note, str):
        return None
    cleaned = note.strip()
    if not cleaned or len(cleaned) > _MAX_PARITY_LEN:
        return None
    if any(ord(ch) < 32 for ch in cleaned):
        return None
    return cleaned


def attach_router_parity_note(context: Dict[str, Any], note: str | None) -> None:
    if not isinstance(context, dict):
        return
    cleaned = safe_parity_note(note)
    if not cleaned:
        return
    context[_PARITY_CONTEXT_KEY] = cleaned


def extract_router_parity_note(context: Dict[str, Any] | None) -> str | None:
    if not isinstance(context, dict):
        return None
    return safe_parity_note(context.get(_PARITY_CONTEXT_KEY))


def attach_parity_note_to_metadata(metadata: Dict[str, Any] | None, note: str | None) -> Dict[str, Any]:
    cleaned = safe_parity_note(note)
    if metadata is None:
        metadata = {}
    if not cleaned or not isinstance(metadata, dict):
        return metadata
    metadata.setdefault(_PARITY_METADATA_KEY, cleaned)
    return metadata
