"""Off-runtime table-review session orchestration.

This service composes accepted boundaries without wiring them to production:

    typed command
      -> accepted state-machine transition/effect plan
      -> mutation effects: accepted mutation-effect executor
      -> replace working table only

Completion semantics:
    COMPLETE_ACCEPTED -> current working snapshot
    COMPLETE_ORIGINAL -> immutable original snapshot

REJECT/RETRY semantics:
    REJECT keeps current working table while entering RETRY_DECISION.
    retry=True returns to PRIMARY_REVIEW with current working table preserved.
    retry=False completes with the immutable original snapshot.

Candidate navigation fails closed because candidate materialization remains a
separate future authority.
"""

from __future__ import annotations

import copy
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ..contracts.table_structure_review import (
    TableStructureReviewCommand,
    TableStructureReviewDecision,
    TableStructureReviewRequest,
    TableStructureReviewResult,
)
from ..contracts.table_structure_review_session import (
    TableStructureReviewSession,
    TableStructureReviewSessionStep,
    TableStructureReviewTableSnapshot,
)
from .table_structure_review_effect_executor import (
    execute_table_structure_review_mutation_effect,
)
from .table_structure_review_state_machine import (
    TableStructureReviewEffectKind,
    initialize_review_state,
    transition_review_state,
)


SERVICE_VERSION = "table_structure_review_session_v1"
RUNTIME_CALLER_WIRED = False
TRANSPORT_WIRED = False
FRONTEND_WIRED = False
CANDIDATE_MATERIALIZATION_IMPLEMENTED = False
LEARNING_SIDE_EFFECT_AUTHORITY = False
CANONICAL_AUTHORITY = False


class TableStructureReviewSessionError(ValueError):
    """Raised when an effect cannot be safely interpreted by this session."""


_MUTATION_EFFECT_KINDS = frozenset(
    {
        TableStructureReviewEffectKind.REQUEST_REMOVE_COLUMNS,
        TableStructureReviewEffectKind.REQUEST_REORDER_COLUMNS,
        TableStructureReviewEffectKind.REQUEST_RENAME_COLUMNS,
        TableStructureReviewEffectKind.REQUEST_ADD_COLUMNS,
    }
)


def _freeze_snapshot(
    headers: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> TableStructureReviewTableSnapshot:
    frozen_headers = tuple(copy.deepcopy(list(headers)))
    frozen_rows = tuple(
        MappingProxyType(copy.deepcopy(dict(row)))
        for row in rows
    )
    return TableStructureReviewTableSnapshot(
        headers=frozen_headers,
        rows=frozen_rows,
    )


def _freeze_context(
    context: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if context is None:
        return None
    return MappingProxyType(copy.deepcopy(dict(context)))


def _mutable_rows(
    snapshot: TableStructureReviewTableSnapshot,
) -> list[dict[str, Any]]:
    return [
        copy.deepcopy(dict(row))
        for row in snapshot.rows
    ]


def initialize_table_structure_review_session(
    request: TableStructureReviewRequest,
    headers: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    *,
    harmonization_context: Mapping[str, Any] | None = None,
    source_location_label: str | None = None,
) -> TableStructureReviewSession:
    """Create a session with isolated original and working table snapshots."""

    if not isinstance(request, TableStructureReviewRequest):
        raise TableStructureReviewSessionError(
            "request must be TableStructureReviewRequest"
        )
    if source_location_label is not None and not isinstance(
        source_location_label,
        str,
    ):
        raise TableStructureReviewSessionError(
            "source_location_label must be str or None"
        )

    # Build independent snapshots so no mutation path can alias baseline rows.
    original = _freeze_snapshot(headers, rows)
    working = _freeze_snapshot(headers, rows)

    return TableStructureReviewSession(
        review_id=request.review_id,
        state=initialize_review_state(request),
        original=original,
        working=working,
        harmonization_context=_freeze_context(harmonization_context),
        source_location_label=source_location_label,
    )


def _completion_from_snapshot(
    snapshot: TableStructureReviewTableSnapshot,
    decision: TableStructureReviewDecision,
) -> TableStructureReviewResult:
    return TableStructureReviewResult(
        headers=snapshot.headers,
        rows=snapshot.rows,
        decision=decision,
    )


def advance_table_structure_review_session(
    session: TableStructureReviewSession,
    command: TableStructureReviewCommand,
) -> TableStructureReviewSessionStep:
    """Apply one command to a session without any runtime/transport authority."""

    if not isinstance(session, TableStructureReviewSession):
        raise TableStructureReviewSessionError(
            "session must be TableStructureReviewSession"
        )
    if not isinstance(command, TableStructureReviewCommand):
        raise TableStructureReviewSessionError(
            "command must be TableStructureReviewCommand"
        )
    if command.review_id != session.review_id:
        raise TableStructureReviewSessionError(
            "command review_id does not match active session"
        )

    transition = transition_review_state(
        session.state,
        command,
    )
    effect = transition.effect

    if effect.kind is TableStructureReviewEffectKind.REQUEST_CANDIDATE_MATERIALIZATION:
        raise TableStructureReviewSessionError(
            "candidate navigation requires candidate materialization authority; "
            "C2G 2.8.27 fails closed"
        )

    effect_execution = None
    completion = None
    next_working = session.working

    if effect.kind in _MUTATION_EFFECT_KINDS:
        effect_execution = execute_table_structure_review_mutation_effect(
            session.working.headers,
            _mutable_rows(session.working),
            effect,
            harmonization_context=session.harmonization_context,
            source_location_label=session.source_location_label,
        )
        next_working = _freeze_snapshot(
            effect_execution.headers,
            effect_execution.rows,
        )

    elif effect.kind is TableStructureReviewEffectKind.COMPLETE_ACCEPTED:
        completion = _completion_from_snapshot(
            session.working,
            TableStructureReviewDecision.ACCEPTED_REVIEW_STRUCTURE,
        )

    elif effect.kind is TableStructureReviewEffectKind.COMPLETE_ORIGINAL:
        completion = _completion_from_snapshot(
            session.original,
            TableStructureReviewDecision.ORIGINAL_STRUCTURE_RETAINED,
        )

    elif effect.kind in {
        TableStructureReviewEffectKind.REQUEST_RETRY_DECISION,
        TableStructureReviewEffectKind.RETURN_TO_PRIMARY_REVIEW,
    }:
        # Control-flow only. The current working table intentionally survives.
        pass

    else:
        raise TableStructureReviewSessionError(
            f"unsupported session effect: {effect.kind.value}"
        )

    session_after = TableStructureReviewSession(
        review_id=session.review_id,
        state=transition.state_after,
        original=session.original,
        working=next_working,
        harmonization_context=session.harmonization_context,
        source_location_label=session.source_location_label,
    )

    return TableStructureReviewSessionStep(
        session_before=session,
        transition=transition,
        session_after=session_after,
        effect_execution=effect_execution,
        completion=completion,
    )
