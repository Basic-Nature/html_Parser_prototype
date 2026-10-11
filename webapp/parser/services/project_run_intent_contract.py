"""J2B candidate: pure, *non-authoritative* browser run-intent vocabulary.

This module does not authenticate, authorize, touch a DB, dispatch a parser,
issue a token or accept a URL. Server-side admission must revalidate every
J2A Project/Registry/Workflow/capability condition in one transaction.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Mapping
from uuid import UUID

CONTRACT = "project_run_intent_v1"
REQUIRED_BROWSER_KEYS = frozenset({
    "source_ref_id", "workflow_item_id", "expected_project_version", "idempotency_key",
})


class InvalidRunIntent(ValueError):
    """Fail-closed invalid browser intent (NOT an authorization failure)."""


def _canonical_uuid(value: object) -> UUID:
    if not isinstance(value, str) or len(value) != 36:
        raise InvalidRunIntent("invalid_run_selector")
    try:
        result = UUID(value)
    except (TypeError, ValueError, AttributeError) as exc:
        raise InvalidRunIntent("invalid_run_selector") from exc
    if str(result) != value.lower():
        raise InvalidRunIntent("invalid_run_selector")
    return result


@dataclass(frozen=True)
class RunIntent:
    source_ref_id: UUID
    workflow_item_id: UUID
    expected_project_version: int
    idempotency_key: UUID

    def safe_request_fingerprint_fields(self) -> tuple[str, str, int]:
        """Intended input to a keyed, server-owned request fingerprint.

        *Not* an authorization proof. Include project/owner context in final
        service fingerprint and keep its private signing key on the server.
        """
        return (str(self.source_ref_id), str(self.workflow_item_id), self.expected_project_version)


def parse_browser_run_intent(raw: object) -> RunIntent:
    if not isinstance(raw, Mapping) or frozenset(raw) != REQUIRED_BROWSER_KEYS:
        raise InvalidRunIntent("invalid_run_fields")
    version = raw.get("expected_project_version")
    if type(version) is not int or not 1 <= version <= 2147483647:
        raise InvalidRunIntent("invalid_project_version")
    return RunIntent(
        source_ref_id=_canonical_uuid(raw.get("source_ref_id")),
        workflow_item_id=_canonical_uuid(raw.get("workflow_item_id")),
        expected_project_version=version,
        idempotency_key=_canonical_uuid(raw.get("idempotency_key")),
    )


class RunState(str, Enum):
    ADMITTED = "admitted"            # durable request exists; NOT dispatched
    DISPATCH_PENDING = "dispatch_pending"  # server-owned lease/outbox only
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    STALE = "stale"


ALLOWED_TRANSITIONS = {
    RunState.ADMITTED: frozenset({RunState.DISPATCH_PENDING, RunState.CANCELLED, RunState.STALE}),
    RunState.DISPATCH_PENDING: frozenset({RunState.RUNNING, RunState.FAILED, RunState.CANCELLED, RunState.STALE}),
    RunState.RUNNING: frozenset({RunState.COMPLETED, RunState.FAILED, RunState.CANCELLED}),
    RunState.COMPLETED: frozenset(),
    RunState.FAILED: frozenset(),
    RunState.CANCELLED: frozenset(),
    RunState.STALE: frozenset(),
}


def require_state_transition(current: RunState, next_state: RunState) -> None:
    """Only syntax of transitions; trusted dispatcher must enforce authority.

    Completion additionally requires independently verified output evidence,
    which MUST be checked in the future DB-backed service.
    """
    if not isinstance(current, RunState) or not isinstance(next_state, RunState):
        raise InvalidRunIntent("invalid_run_state")
    if next_state not in ALLOWED_TRANSITIONS[current]:
        raise InvalidRunIntent("invalid_run_transition")
