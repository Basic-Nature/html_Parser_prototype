"""Run-scoped public Ballot Lens runtime safety binding.

This module binds the previously accepted public memory-preview and Playwright
egress policies. Importing it does not execute the parser, launch a browser,
open a network connection, touch a database, or write an artifact.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
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
        self.finalized_previews.append(preview)
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
