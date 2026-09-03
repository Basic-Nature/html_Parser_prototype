"""Run-scoped public Ballot Lens runtime safety binding.

This module binds the previously accepted public memory-preview and Playwright
egress policies. Importing it does not execute the parser, launch a browser,
open a network connection, touch a database, or write an artifact.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

from .public_ballot_lens_egress import (
    DnsResolver,
    PublicBrowserEgressGuard,
    default_dns_resolver,
)
from .public_ballot_lens_execution import (
    DEFAULT_PUBLIC_EXECUTION_CONTEXT,
    PublicRunMemoryState,
    assert_public_execution_context,
    serialize_public_memory_preview,
)
from .public_ballot_lens_policy import (
    DEFAULT_PUBLIC_RUN_POLICY,
    PUBLIC_REGISTRY_SOURCE_ID_PATTERN,
    PublicBallotLensRunPolicy,
    PublicRunAdmissionController,
    PublicRunAdmissionLease,
)


class PublicBallotLensRuntimeError(RuntimeError):
    pass


SafeEmit = Callable[[dict[str, object]], None]

PUBLIC_TERMINAL_REASON_CODES = frozenset(
    {
        "public_download_fallback_disabled",
        "public_memory_preview_missing",
        "public_challenge_assist_disabled",
    }
)

PUBLIC_CHECKPOINT_DEFINITIONS: dict[str, str] = {
    "source.resolve": "Resolve Source",
    "provider.detect": "Provider Detection",
    "source.acquire": "Acquire",
    "structure.detect": "Detect Structure",
    "contest.select": "Contest Selection",
    "vote_methods.detect": "Vote Method Selection",
    "normalize.rows": "Normalize",
    "validate.results": "Validate",
    "preview.publish": "Preview",
}
PUBLIC_CHECKPOINT_STATES = frozenset({"pending", "active", "complete", "warning", "error"})
PUBLIC_ACTION_TYPES = frozenset({"contest_selection", "vote_method_selection", "challenge", "other"})

_PUBLIC_ADMISSION_CONTROLLER = PublicRunAdmissionController(
    DEFAULT_PUBLIC_RUN_POLICY
)


@dataclass
class PublicBallotLensRuntime:
    registry_source_id: str
    source_projection: dict[str, str]
    approved_target_url: str
    policy: PublicBallotLensRunPolicy = DEFAULT_PUBLIC_RUN_POLICY
    resolver: DnsResolver = default_dns_resolver
    safe_emit: SafeEmit | None = None
    memory_state: PublicRunMemoryState = field(init=False)
    egress_guard: PublicBrowserEgressGuard = field(init=False)
    finalized_previews: list[dict[str, object]] = field(
        default_factory=list,
        init=False,
    )
    status_counts: dict[str, int] = field(
        default_factory=dict,
        init=False,
    )
    _last_status: str | None = field(default=None, init=False)
    _terminal_status: str | None = field(default=None, init=False)
    _terminal_reason_code: str | None = field(default=None, init=False)

    _checkpoint_sequence: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        assert_public_execution_context(
            DEFAULT_PUBLIC_EXECUTION_CONTEXT
        )
        source_id = str(self.registry_source_id or "").strip()
        if not PUBLIC_REGISTRY_SOURCE_ID_PATTERN.fullmatch(source_id):
            raise PublicBallotLensRuntimeError(
                "Public runtime requires a valid registry source ID."
            )
        if not isinstance(self.source_projection, dict):
            raise PublicBallotLensRuntimeError(
                "Public runtime source projection must be a dict."
            )
        if self.source_projection.get("registry_source_id") != source_id:
            raise PublicBallotLensRuntimeError(
                "Public runtime source projection ID mismatch."
            )
        if self.source_projection.get("registry_category") != "curated":
            raise PublicBallotLensRuntimeError(
                "Public runtime requires a Curated source projection."
            )
        target = str(self.approved_target_url or "").strip()
        if not target:
            raise PublicBallotLensRuntimeError(
                "Public runtime requires a server-resolved target."
            )

        self.registry_source_id = source_id
        self.approved_target_url = target
        self.memory_state = PublicRunMemoryState(
            registry_source_id=source_id,
            policy=self.policy,
            execution_context=DEFAULT_PUBLIC_EXECUTION_CONTEXT,
        )
        self.egress_guard = PublicBrowserEgressGuard(
            approved_target_url=target,
            policy=self.policy,
            resolver=self.resolver,
        )

    def install_sync_page_guard(
        self,
        page: Any,
        target_url: str,
    ) -> None:
        if page is None:
            raise PublicBallotLensRuntimeError(
                "Public runtime cannot guard a missing page."
            )
        self.egress_guard.validate_initial_target(target_url)
        route = getattr(page, "route", None)
        if not callable(route):
            raise PublicBallotLensRuntimeError(
                "Public runtime requires Playwright request routing."
            )
        route("**/*", self.egress_guard.handle_sync_route)

    def _safe_checkpoint_text(self, value: object | None, *, field: str, max_length: int, required: bool = False) -> str | None:
        if value is None:
            if required:
                raise PublicBallotLensRuntimeError(f"Public runtime {field} is required.")
            return None
        text = str(value).strip()
        if not text:
            if required:
                raise PublicBallotLensRuntimeError(f"Public runtime {field} is required.")
            return None
        if self.approved_target_url and self.approved_target_url in text:
            raise PublicBallotLensRuntimeError(f"Public runtime {field} cannot disclose the approved target.")
        return text[:max_length]

    def record_checkpoint(self, *, checkpoint_id: str, state: str, reason_code: str | None = None, summary: str | None = None, evidence_count: int = 0, requires_action: bool = False, action_type: str | None = None) -> dict[str, object]:
        checkpoint_key = str(checkpoint_id or "").strip()
        label = PUBLIC_CHECKPOINT_DEFINITIONS.get(checkpoint_key)
        if label is None:
            raise PublicBallotLensRuntimeError("Unknown public runtime checkpoint.")
        state_key = str(state or "").strip().lower()
        if state_key not in PUBLIC_CHECKPOINT_STATES:
            raise PublicBallotLensRuntimeError("Unknown public runtime checkpoint state.")
        if isinstance(evidence_count, bool) or not isinstance(evidence_count, int) or evidence_count < 0:
            raise PublicBallotLensRuntimeError("Public runtime checkpoint evidence count must be non-negative.")
        if not isinstance(requires_action, bool):
            raise PublicBallotLensRuntimeError("Public runtime checkpoint requires_action must be boolean.")
        action_key = None
        if action_type is not None:
            action_key = str(action_type or "").strip().lower()
            if action_key not in PUBLIC_ACTION_TYPES:
                raise PublicBallotLensRuntimeError("Unknown public runtime checkpoint action type.")
        if requires_action != (action_key is not None):
            raise PublicBallotLensRuntimeError("Public runtime checkpoint action fields disagree.")
        self._checkpoint_sequence += 1
        checkpoint: dict[str, object] = {
            "checkpoint_id": checkpoint_key,
            "sequence": self._checkpoint_sequence,
            "state": state_key,
            "label": label,
            "reason_code": self._safe_checkpoint_text(reason_code, field="checkpoint reason code", max_length=128),
            "summary": self._safe_checkpoint_text(summary, field="checkpoint summary", max_length=360),
            "evidence_count": evidence_count,
            "requires_action": requires_action,
            "action_type": action_key,
            "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        if self.safe_emit is not None:
            level = "ERROR" if state_key == "error" else "WARNING" if state_key == "warning" else "INFO"
            self.safe_emit({
                "level": level,
                "type": "public_registry_checkpoint",
                "message": "Structured public parser checkpoint updated.",
                "reason_code": "public_registry_checkpoint_updated",
                "registry_source_id": self.registry_source_id,
                "checkpoint": dict(checkpoint),
            })
        return checkpoint

    def record_action_required(self, *, prompt_id: str, checkpoint_id: str, action_type: str, summary: str) -> dict[str, object]:
        prompt_key = self._safe_checkpoint_text(prompt_id, field="action prompt ID", max_length=128, required=True)
        checkpoint_key = str(checkpoint_id or "").strip()
        if checkpoint_key not in PUBLIC_CHECKPOINT_DEFINITIONS:
            raise PublicBallotLensRuntimeError("Unknown public runtime action checkpoint.")
        action_key = str(action_type or "").strip().lower()
        if action_key not in PUBLIC_ACTION_TYPES:
            raise PublicBallotLensRuntimeError("Unknown public runtime action type.")
        safe_summary = self._safe_checkpoint_text(summary, field="action summary", max_length=360, required=True)
        action: dict[str, object] = {"prompt_id": prompt_key, "checkpoint_id": checkpoint_key, "action_type": action_key, "summary": safe_summary}
        if self.safe_emit is not None:
            self.safe_emit({
                "level": "WARNING",
                "type": "public_registry_action_required",
                "message": "Structured public parser action is required.",
                "reason_code": "public_registry_action_required",
                "registry_source_id": self.registry_source_id,
                "action": dict(action),
            })
        return action

    def record_result_checkpoints(self, *, headers: Sequence[str], contest: object | None) -> None:
        contest_present = bool(str(contest or "").strip())
        self.record_checkpoint(
            checkpoint_id="contest.select",
            state="complete" if contest_present else "warning",
            reason_code="public_contest_context_present" if contest_present else "public_contest_context_missing",
            summary="Parser result returned contest context." if contest_present else "Parser result did not expose contest context.",
            evidence_count=1 if contest_present else 0,
        )
        method_headers = [
            header for header in headers
            if isinstance(header, str) and " - " in header
            and not header.endswith(" - Total Votes") and not header.endswith(" - Total")
        ]
        self.record_checkpoint(
            checkpoint_id="vote_methods.detect",
            state="complete" if method_headers else "warning",
            reason_code="public_vote_method_columns_present" if method_headers else "public_vote_method_columns_not_observed",
            summary="Method-specific result columns were observed." if method_headers else "No method-specific result columns were observed.",
            evidence_count=len(method_headers),
        )

    def capture_finalized_output(
        self,
        *,
        headers: Sequence[str],
        rows: Sequence[Mapping[str, object]],
    ) -> dict[str, object]:
        preview = self.memory_state.build_preview(
            source_projection=self.source_projection,
            headers=headers,
            rows=rows,
        )
        encoded = serialize_public_memory_preview(
            preview,
            policy=self.policy,
        )
        current_size = sum(
            len(
                serialize_public_memory_preview(
                    item,
                    policy=self.policy,
                )
            )
            for item in self.finalized_previews
        )
        if (
            current_size + len(encoded)
            > int(self.policy.public_output_max_bytes)
        ):
            raise PublicBallotLensRuntimeError(
                "Combined public previews exceed the output byte limit."
            )
        self.record_checkpoint(
            checkpoint_id="normalize.rows",
            state="complete",
            reason_code="public_finalized_row_shape_reached",
            summary="Finalized Smart Elections row shape reached the public memory boundary.",
            evidence_count=len(rows),
        )
        self.record_checkpoint(
            checkpoint_id="validate.results",
            state="complete",
            reason_code="public_memory_preview_contract_validated",
            summary="Public memory row/header contract validated; no canonical truth claim is implied.",
            evidence_count=len(rows),
        )
        self.finalized_previews.append(preview)
        self.record_checkpoint(
            checkpoint_id="preview.publish",
            state="complete",
            reason_code="public_memory_preview_retained",
            summary="Validated result retained in app-owned memory preview.",
            evidence_count=len(rows),
        )
        return preview

    def record_processed_status(
        self,
        *,
        status: str,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        normalized = str(status or "unknown").strip().lower() or "unknown"
        if self._last_status:
            old_count = int(
                self.status_counts.get(self._last_status, 0)
            )
            if old_count > 0:
                self.status_counts[self._last_status] = old_count - 1
        self.status_counts[normalized] = (
            int(self.status_counts.get(normalized, 0)) + 1
        )
        self._last_status = normalized

        completed = {
            "success",
            "partial",
            "error",
            "fail",
            "rejected",
            "quarantined",
            "skipped_data_exists",
            "cancelled",
        }
        if normalized in completed:
            self._terminal_status = normalized
            self._terminal_reason_code = None
            if isinstance(metadata, Mapping):
                reason_value = metadata.get("reason_code")
                if isinstance(reason_value, str):
                    candidate = reason_value.strip().lower()
                    if candidate in PUBLIC_TERMINAL_REASON_CODES:
                        self._terminal_reason_code = candidate

        processed = 1 if normalized in completed else 0
        try:
            event = self.memory_state.record_progress(
                processed=processed,
                total_entries=1,
                status_counts={
                    key: int(value)
                    for key, value in self.status_counts.items()
                    if int(value) > 0
                },
            )
        except Exception:
            return
        if self.safe_emit is not None:
            self.safe_emit(dict(event))

    def summary_counts(self) -> dict[str, int]:
        return {
            key: int(value)
            for key, value in self.status_counts.items()
            if int(value) > 0
        }

    def result_payload(self) -> dict[str, object]:
        result: dict[str, object] = {
            "contract": "ballot_lens_public_runtime_result_v1",
            "registry_source_id": self.registry_source_id,
            "source": dict(self.source_projection),
            "outputs": list(self.finalized_previews),
            "status_counts": self.summary_counts(),
            "terminal_status": self._terminal_status,
            "terminal_reason_code": self._terminal_reason_code,
            "download_available": False,
            "persistent_output": False,
        }
        encoded = json.dumps(
            result,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(encoded) > int(self.policy.public_output_max_bytes):
            raise PublicBallotLensRuntimeError(
                "Public runtime result exceeds the byte limit."
            )
        return result


_ACTIVE_PUBLIC_RUNTIME: ContextVar[
    PublicBallotLensRuntime | None
] = ContextVar(
    "electionpulse_active_public_ballot_lens_runtime",
    default=None,
)


def current_public_runtime() -> PublicBallotLensRuntime | None:
    return _ACTIVE_PUBLIC_RUNTIME.get()


def require_public_runtime() -> PublicBallotLensRuntime:
    runtime = current_public_runtime()
    if runtime is None:
        raise PublicBallotLensRuntimeError(
            "Public Ballot Lens runtime is not active."
        )
    return runtime


@contextmanager
def activate_public_runtime(
    runtime: PublicBallotLensRuntime,
) -> Iterator[PublicBallotLensRuntime]:
    if not isinstance(runtime, PublicBallotLensRuntime):
        raise PublicBallotLensRuntimeError(
            "Invalid public runtime activation object."
        )
    if current_public_runtime() is not None:
        raise PublicBallotLensRuntimeError(
            "Nested public runtime activation is not allowed."
        )
    token: Token[PublicBallotLensRuntime | None] = (
        _ACTIVE_PUBLIC_RUNTIME.set(runtime)
    )
    try:
        yield runtime
    finally:
        _ACTIVE_PUBLIC_RUNTIME.reset(token)


@contextmanager
def activate_admitted_public_runtime(
    runtime: PublicBallotLensRuntime,
    *,
    client_key: str,
    server_session_id: str,
) -> Iterator[PublicBallotLensRuntime]:
    lease: PublicRunAdmissionLease = (
        _PUBLIC_ADMISSION_CONTROLLER.acquire(
            client_key=client_key,
            server_session_id=server_session_id,
        )
    )
    try:
        with activate_public_runtime(runtime):
            yield runtime
    finally:
        lease.release()
