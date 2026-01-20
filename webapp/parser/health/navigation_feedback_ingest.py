from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import orjson

NAV_LOG_FILENAME = "navigation_learning_log.jsonl"
NAV_FEEDBACK_FILENAME = "navigation_feedback_selection_log.jsonl"
NAV_OFFSET_FILENAME = ".navigation_feedback_offset"
NAV_FIELD_TYPE = "navigation_feedback"
NAV_SUCCESS_RESULT = "nav_success"
NAV_FAILURE_RESULT = "nav_failure"

__all__ = [
    "ingest_navigation_feedback",
    "NAV_LOG_FILENAME",
    "NAV_FEEDBACK_FILENAME",
    "NAV_OFFSET_FILENAME",
    "NAV_FIELD_TYPE",
]


def ingest_navigation_feedback(log_dir: str | Path) -> int:
    """Convert new navigation telemetry entries into correction-friendly logs.

    Returns the number of new entries written. Keeps track of the source log
    offset so repeated calls are incremental.
    """

    directory = Path(log_dir)
    log_path = directory / NAV_LOG_FILENAME
    if not log_path.exists() or log_path.stat().st_size == 0:
        return 0

    offset_path = directory / NAV_OFFSET_FILENAME
    output_path = directory / NAV_FEEDBACK_FILENAME

    last_offset = _read_offset(offset_path)
    file_size = log_path.stat().st_size
    if last_offset > file_size:
        last_offset = 0

    processed = 0
    with log_path.open("rb") as source, output_path.open("ab") as sink:
        source.seek(last_offset)
        for raw in source:
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = orjson.loads(raw)
            except Exception:
                continue
            formatted = _format_entry(entry)
            if not formatted:
                continue
            sink.write(orjson.dumps(formatted) + b"\n")
            processed += 1
        current_offset = source.tell()

    _write_offset(offset_path, current_offset)
    return processed


def _read_offset(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return max(0, int(path.read_text().strip() or 0))
    except Exception:
        return 0


def _write_offset(path: Path, value: int) -> None:
    try:
        path.write_text(str(value))
    except Exception:
        pass


def _format_entry(entry: Dict[str, Any]) -> Dict[str, Any] | None:
    if not isinstance(entry, dict):
        return None

    telemetry = entry.get("telemetry") or []
    context_before = entry.get("context_before") or {}
    context_after = entry.get("context_after") or {}
    metadata = entry.get("metadata") or {}

    state = context_after.get("state") or context_before.get("state")
    county = context_after.get("county") or context_before.get("county")
    script_id = entry.get("script_id") or metadata.get("script_id") or "unknown_script"
    base_context_key = "::".join([value for value in (state, county) if value])
    context_key = base_context_key or script_id

    summary: Dict[str, Any] = {
        "script_id": script_id,
        "state": state,
        "county": county,
        "success": bool(entry.get("success")),
        "action_count": len(telemetry),
        "page_url": metadata.get("page_url") or metadata.get("url") or context_after.get("url") or context_before.get("url"),
    }
    if not summary["success"]:
        failure_reason = metadata.get("error") or metadata.get("notes") or entry.get("error")
        if failure_reason:
            summary["failure_reason"] = failure_reason
    if metadata:
        summary["metadata"] = metadata

    result = NAV_SUCCESS_RESULT if summary["success"] else NAV_FAILURE_RESULT
    return {
        "timestamp": entry.get("timestamp"),
        "field_type": NAV_FIELD_TYPE,
        "result": result,
        "context_key": context_key or "default",
        "extracted_value": summary,
        "telemetry": telemetry,
        "pre_context": context_before,
        "post_context": context_after,
        "metadata": metadata,
    }
