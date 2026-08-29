"""Pure in-memory public Ballot Lens execution/output foundation.

This module intentionally does not wire Socket.IO, parser execution, browser
navigation, filesystem output, database access, workflow/canonical writes, or
learning/telemetry. It packages already-structured election rows for the
future anonymous Curated-registry execution path.

The parser remains responsible for election extraction and Smart Elections
structuring. This module does not recompute candidate totals or normalize
election semantics; in particular, semantic NULL remains distinct from zero.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from .public_ballot_lens_policy import (
    DEFAULT_PUBLIC_RUN_POLICY,
    PUBLIC_REGISTRY_SOURCE_ID_PATTERN,
    PublicBallotLensRunPolicy,
)


class PublicBallotLensExecutionError(RuntimeError):
    pass


PUBLIC_SOURCE_PROJECTION_KEYS = frozenset(
    {
        "registry_source_id",
        "year",
        "contest",
        "state",
        "scope",
        "format",
        "registry_category",
    }
)


@dataclass(frozen=True)
class PublicBallotLensExecutionContext:
    contract: str = "ballot_lens_public_execution_context_v1"
    server_resolved_registry_source_only: bool = True
    one_source_per_run: bool = True
    principal: None = None
    fabricated_principal: bool = False
    memory_preview_only: bool = True
    persistent_output_write: bool = False
    processed_urls_global_write: bool = False
    output_cache_write: bool = False
    download_manifest_write: bool = False
    pipeline_report_write: bool = False
    data_framework_audit_export_write: bool = False
    database_cross_check: bool = False
    learning_write: bool = False
    ml_training_telemetry_write: bool = False
    diagnostic_artifact_write: bool = False
    manual_captcha_assist: bool = False
    selenium_fallback: bool = False


DEFAULT_PUBLIC_EXECUTION_CONTEXT = PublicBallotLensExecutionContext()


def assert_public_execution_context(
    context: PublicBallotLensExecutionContext,
) -> PublicBallotLensExecutionContext:
    if context != DEFAULT_PUBLIC_EXECUTION_CONTEXT:
        raise PublicBallotLensExecutionError(
            "Public Ballot Lens execution context drifted from the "
            "frozen no-write anonymous policy."
        )
    return context


def _json_safe_clone(value: Any, *, depth: int = 0) -> Any:
    if depth > 32:
        raise PublicBallotLensExecutionError(
            "Public preview value nesting exceeds the safety limit."
        )

    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise PublicBallotLensExecutionError(
                "Public preview cannot contain NaN or infinite values."
            )
        return value

    if isinstance(value, Mapping):
        cloned: dict[str, Any] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise PublicBallotLensExecutionError(
                    "Public preview object keys must be strings."
                )
            cloned[key] = _json_safe_clone(
                nested,
                depth=depth + 1,
            )
        return cloned

    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [
            _json_safe_clone(item, depth=depth + 1)
            for item in value
        ]

    raise PublicBallotLensExecutionError(
        "Public preview contains a non-JSON election value."
    )


def _validate_source_projection(
    *,
    registry_source_id: str,
    source_projection: Mapping[str, object],
) -> dict[str, str]:
    source_id = str(registry_source_id or "").strip()
    if not PUBLIC_REGISTRY_SOURCE_ID_PATTERN.fullmatch(source_id):
        raise PublicBallotLensExecutionError(
            "Invalid public registry source identifier."
        )

    if not isinstance(source_projection, Mapping):
        raise PublicBallotLensExecutionError(
            "Public source projection must be a mapping."
        )

    if frozenset(source_projection.keys()) != PUBLIC_SOURCE_PROJECTION_KEYS:
        raise PublicBallotLensExecutionError(
            "Public source projection contains missing or forbidden fields."
        )

    projected: dict[str, str] = {}
    for key in sorted(PUBLIC_SOURCE_PROJECTION_KEYS):
        value = source_projection.get(key)
        if not isinstance(value, str):
            raise PublicBallotLensExecutionError(
                f"Public source projection field {key!r} must be a string."
            )
        projected[key] = value.strip()

    if projected["registry_source_id"] != source_id:
        raise PublicBallotLensExecutionError(
            "Public source projection does not match the run source ID."
        )
    if projected["registry_category"] != "curated":
        raise PublicBallotLensExecutionError(
            "Anonymous Ballot Lens execution requires a Curated source."
        )

    return projected


def _validate_finalized_rows(
    *,
    headers: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> tuple[list[str], list[dict[str, Any]]]:
    if isinstance(headers, (str, bytes, bytearray)):
        raise PublicBallotLensExecutionError(
            "Public preview headers must be a sequence of column names."
        )
    if isinstance(rows, (str, bytes, bytearray)):
        raise PublicBallotLensExecutionError(
            "Public preview rows must be a sequence of row mappings."
        )

    clean_headers: list[str] = []
    seen: set[str] = set()
    for header in headers:
        if not isinstance(header, str) or not header.strip():
            raise PublicBallotLensExecutionError(
                "Public preview headers must be non-empty strings."
            )
        if header in seen:
            raise PublicBallotLensExecutionError(
                "Public preview headers must be unique."
            )
        seen.add(header)
        clean_headers.append(header)

    clean_rows: list[dict[str, Any]] = []
    header_set = set(clean_headers)
    for row in rows:
        if not isinstance(row, Mapping):
            raise PublicBallotLensExecutionError(
                "Public preview rows must be mappings."
            )
        unknown = [
            key
            for key in row.keys()
            if not isinstance(key, str) or key not in header_set
        ]
        if unknown:
            raise PublicBallotLensExecutionError(
                "Public preview row contains a field absent from headers."
            )
        clean_rows.append(
            {
                key: _json_safe_clone(value)
                for key, value in row.items()
            }
        )

    return clean_headers, clean_rows


def _serialized_preview_bytes(payload: Mapping[str, object]) -> bytes:
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except Exception as exc:
        raise PublicBallotLensExecutionError(
            "Public memory preview could not be serialized safely."
        ) from exc


def build_public_memory_preview(
    *,
    registry_source_id: str,
    source_projection: Mapping[str, object],
    headers: Sequence[str],
    rows: Sequence[Mapping[str, object]],
    policy: PublicBallotLensRunPolicy = DEFAULT_PUBLIC_RUN_POLICY,
    execution_context: PublicBallotLensExecutionContext = (
        DEFAULT_PUBLIC_EXECUTION_CONTEXT
    ),
) -> dict[str, object]:
    """Package already-finalized election rows without writing anywhere."""
    assert_public_execution_context(execution_context)

    source = _validate_source_projection(
        registry_source_id=registry_source_id,
        source_projection=source_projection,
    )
    clean_headers, clean_rows = _validate_finalized_rows(
        headers=headers,
        rows=rows,
    )

    payload: dict[str, object] = {
        "contract": "ballot_lens_public_memory_preview_v1",
        "registry_source_id": registry_source_id,
        "source": source,
        "headers": clean_headers,
        "rows": clean_rows,
        "row_count": len(clean_rows),
        "output_mode": "MEMORY_PREVIEW_ONLY",
        "download_available": False,
        "persistent_output": False,
        "execution_context_contract": execution_context.contract,
    }

    encoded = _serialized_preview_bytes(payload)
    if len(encoded) > int(policy.public_output_max_bytes):
        raise PublicBallotLensExecutionError(
            "Public memory preview exceeds the result byte limit."
        )

    return payload


def serialize_public_memory_preview(
    preview: Mapping[str, object],
    *,
    policy: PublicBallotLensRunPolicy = DEFAULT_PUBLIC_RUN_POLICY,
) -> bytes:
    encoded = _serialized_preview_bytes(preview)
    if len(encoded) > int(policy.public_output_max_bytes):
        raise PublicBallotLensExecutionError(
            "Public memory preview exceeds the result byte limit."
        )
    return encoded


class PublicRunMemoryState:
    """Run-scoped progress/result state with no persistent backing store."""

    def __init__(
        self,
        *,
        registry_source_id: str,
        policy: PublicBallotLensRunPolicy = DEFAULT_PUBLIC_RUN_POLICY,
        execution_context: PublicBallotLensExecutionContext = (
            DEFAULT_PUBLIC_EXECUTION_CONTEXT
        ),
    ) -> None:
        assert_public_execution_context(execution_context)
        source_id = str(registry_source_id or "").strip()
        if not PUBLIC_REGISTRY_SOURCE_ID_PATTERN.fullmatch(source_id):
            raise PublicBallotLensExecutionError(
                "Invalid public registry source identifier."
            )

        self.registry_source_id = source_id
        self.policy = policy
        self.execution_context = execution_context
        self._progress_events: list[dict[str, object]] = []
        self._progress_bytes = 0

    def record_progress(
        self,
        *,
        processed: int,
        total_entries: int,
        status_counts: Mapping[str, int] | None = None,
    ) -> dict[str, object]:
        if isinstance(processed, bool) or not isinstance(processed, int):
            raise PublicBallotLensExecutionError(
                "Public progress processed count must be an integer."
            )
        if isinstance(total_entries, bool) or not isinstance(
            total_entries,
            int,
        ):
            raise PublicBallotLensExecutionError(
                "Public progress total count must be an integer."
            )
        if processed < 0 or total_entries < 0 or processed > total_entries:
            raise PublicBallotLensExecutionError(
                "Public progress counts are inconsistent."
            )

        counts: dict[str, int] = {}
        for key, value in (status_counts or {}).items():
            if (
                not isinstance(key, str)
                or isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise PublicBallotLensExecutionError(
                    "Public progress status counts are invalid."
                )
            counts[key] = value

        event: dict[str, object] = {
            "type": "run_progress",
            "processed": processed,
            "total_entries": total_entries,
            "status_counts": counts,
        }

        encoded = _serialized_preview_bytes(event)
        if len(encoded) > int(self.policy.socket_event_max_bytes):
            raise PublicBallotLensExecutionError(
                "Public progress event exceeds the socket-event byte limit."
            )
        if (
            self._progress_bytes + len(encoded)
            > int(self.policy.cumulative_public_log_max_bytes)
        ):
            raise PublicBallotLensExecutionError(
                "Public progress stream exceeds the cumulative byte limit."
            )

        self._progress_events.append(event)
        self._progress_bytes += len(encoded)
        return dict(event)

    def progress_events(self) -> list[dict[str, object]]:
        return [
            _json_safe_clone(event)
            for event in self._progress_events
        ]

    def build_preview(
        self,
        *,
        source_projection: Mapping[str, object],
        headers: Sequence[str],
        rows: Sequence[Mapping[str, object]],
    ) -> dict[str, object]:
        preview = build_public_memory_preview(
            registry_source_id=self.registry_source_id,
            source_projection=source_projection,
            headers=headers,
            rows=rows,
            policy=self.policy,
            execution_context=self.execution_context,
        )
        preview["progress"] = self.progress_events()

        encoded = _serialized_preview_bytes(preview)
        if len(encoded) > int(self.policy.public_output_max_bytes):
            raise PublicBallotLensExecutionError(
                "Public memory preview exceeds the result byte limit."
            )

        return preview
