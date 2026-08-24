"""Server-side table-review state-machine boundary scaffold.

This module owns phase transition validation and typed effect planning only.
It does not invoke parser review code, mutate table rows/headers, materialize
candidate structures, execute learning side effects, perform transport, or
exercise canonical authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from ..contracts.table_structure_review import (
    TableStructureReviewAction,
    TableStructureReviewCommand,
    TableStructureReviewDecision,
    TableStructureReviewRequest,
)


SERVICE_VERSION = "table_structure_review_state_machine_v1"

RUNTIME_CALLER_WIRED = False
TRANSPORT_WIRED = False
PARSER_MUTATION_APPLICATION_IMPLEMENTED = False
LEARNING_SIDE_EFFECT_APPLICATION_IMPLEMENTED = False
CANDIDATE_MATERIALIZATION_IMPLEMENTED = False
CLI_REPLACEMENT_AUTHORIZED = False
CANONICAL_AUTHORITY = False


class TableStructureReviewStateMachineError(ValueError):
    """Raised when a typed command violates the review state-machine contract."""


class TableStructureReviewPhase(str, Enum):
    PRIMARY_REVIEW = "PRIMARY_REVIEW"
    RETRY_DECISION = "RETRY_DECISION"
    COMPLETED = "COMPLETED"


class TableStructureReviewEffectKind(str, Enum):
    COMPLETE_ACCEPTED = "COMPLETE_ACCEPTED"
    REQUEST_RETRY_DECISION = "REQUEST_RETRY_DECISION"
    RETURN_TO_PRIMARY_REVIEW = "RETURN_TO_PRIMARY_REVIEW"
    COMPLETE_ORIGINAL = "COMPLETE_ORIGINAL"
    REQUEST_REMOVE_COLUMNS = "REQUEST_REMOVE_COLUMNS"
    REQUEST_REORDER_COLUMNS = "REQUEST_REORDER_COLUMNS"
    REQUEST_RENAME_COLUMNS = "REQUEST_RENAME_COLUMNS"
    REQUEST_ADD_COLUMNS = "REQUEST_ADD_COLUMNS"
    REQUEST_CANDIDATE_MATERIALIZATION = "REQUEST_CANDIDATE_MATERIALIZATION"


_PRIMARY_MUTATION_ACTIONS = frozenset(
    {
        TableStructureReviewAction.REMOVE_COLUMNS,
        TableStructureReviewAction.REORDER_COLUMNS,
        TableStructureReviewAction.RENAME_COLUMNS,
        TableStructureReviewAction.ADD_COLUMNS,
    }
)


@dataclass(frozen=True)
class TableStructureReviewMachineState:
    """Immutable coordination state; carries no table-data mutation authority."""

    review_id: str
    phase: TableStructureReviewPhase
    candidate_index: int
    candidates_total: int
    allowed_primary_actions: Tuple[TableStructureReviewAction, ...]
    decision: Optional[TableStructureReviewDecision] = None

    def __post_init__(self) -> None:
        if not isinstance(self.review_id, str) or not self.review_id:
            raise TableStructureReviewStateMachineError(
                "review_id must be a non-empty string"
            )
        if not isinstance(self.phase, TableStructureReviewPhase):
            raise TableStructureReviewStateMachineError(
                "phase must be TableStructureReviewPhase"
            )
        if isinstance(self.candidate_index, bool) or not isinstance(
            self.candidate_index, int
        ):
            raise TableStructureReviewStateMachineError(
                "candidate_index must be an integer"
            )
        if isinstance(self.candidates_total, bool) or not isinstance(
            self.candidates_total, int
        ):
            raise TableStructureReviewStateMachineError(
                "candidates_total must be an integer"
            )
        if self.candidates_total < 1:
            raise TableStructureReviewStateMachineError(
                "candidates_total must be at least 1"
            )
        if not 1 <= self.candidate_index <= self.candidates_total:
            raise TableStructureReviewStateMachineError(
                "candidate_index must be within candidates_total"
            )
        if not isinstance(self.allowed_primary_actions, tuple):
            raise TableStructureReviewStateMachineError(
                "allowed_primary_actions must be a tuple"
            )
        if not self.allowed_primary_actions:
            raise TableStructureReviewStateMachineError(
                "allowed_primary_actions must not be empty"
            )
        for action in self.allowed_primary_actions:
            if not isinstance(action, TableStructureReviewAction):
                raise TableStructureReviewStateMachineError(
                    "allowed_primary_actions must contain typed actions"
                )
        if (
            TableStructureReviewAction.RETRY_DECISION
            in self.allowed_primary_actions
        ):
            raise TableStructureReviewStateMachineError(
                "RETRY_DECISION is phase-specific, not a primary action"
            )
        if len(set(self.allowed_primary_actions)) != len(
            self.allowed_primary_actions
        ):
            raise TableStructureReviewStateMachineError(
                "allowed_primary_actions must not contain duplicates"
            )

        if self.phase is TableStructureReviewPhase.COMPLETED:
            if self.decision is None:
                raise TableStructureReviewStateMachineError(
                    "completed state requires a decision"
                )
        elif self.decision is not None:
            raise TableStructureReviewStateMachineError(
                "non-completed state must not carry a decision"
            )


@dataclass(frozen=True)
class TableStructureReviewEffect:
    """Immutable typed effect plan. No parser mutation is applied here."""

    kind: TableStructureReviewEffectKind
    indices: Tuple[int, ...] = ()
    names: Tuple[str, ...] = ()
    renames: Tuple[Tuple[int, str], ...] = ()
    navigation_delta: int = 0
    candidate_materialization_required: bool = False
    parser_mutation_applied: bool = False
    learning_side_effect_applied: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.kind, TableStructureReviewEffectKind):
            raise TableStructureReviewStateMachineError(
                "effect kind must be TableStructureReviewEffectKind"
            )
        if self.parser_mutation_applied:
            raise TableStructureReviewStateMachineError(
                "state-machine scaffold cannot claim parser mutation was applied"
            )
        if self.learning_side_effect_applied:
            raise TableStructureReviewStateMachineError(
                "state-machine scaffold cannot claim learning side effects"
            )


@dataclass(frozen=True)
class TableStructureReviewTransition:
    """A validated phase transition and its non-executing effect plan."""

    state_before: TableStructureReviewMachineState
    command: TableStructureReviewCommand
    state_after: TableStructureReviewMachineState
    effect: TableStructureReviewEffect


def initialize_review_state(
    request: TableStructureReviewRequest,
) -> TableStructureReviewMachineState:
    """Create PRIMARY_REVIEW coordination state from an accepted typed request."""

    if not isinstance(request, TableStructureReviewRequest):
        raise TableStructureReviewStateMachineError(
            "request must be TableStructureReviewRequest"
        )

    return TableStructureReviewMachineState(
        review_id=request.review_id,
        phase=TableStructureReviewPhase.PRIMARY_REVIEW,
        candidate_index=request.candidate_index,
        candidates_total=request.candidates_total,
        allowed_primary_actions=request.allowed_actions,
        decision=None,
    )


def _completed_state(
    state: TableStructureReviewMachineState,
    decision: TableStructureReviewDecision,
) -> TableStructureReviewMachineState:
    return TableStructureReviewMachineState(
        review_id=state.review_id,
        phase=TableStructureReviewPhase.COMPLETED,
        candidate_index=state.candidate_index,
        candidates_total=state.candidates_total,
        allowed_primary_actions=state.allowed_primary_actions,
        decision=decision,
    )


def _primary_state(
    state: TableStructureReviewMachineState,
    *,
    candidate_index: Optional[int] = None,
) -> TableStructureReviewMachineState:
    return TableStructureReviewMachineState(
        review_id=state.review_id,
        phase=TableStructureReviewPhase.PRIMARY_REVIEW,
        candidate_index=(
            state.candidate_index
            if candidate_index is None
            else candidate_index
        ),
        candidates_total=state.candidates_total,
        allowed_primary_actions=state.allowed_primary_actions,
        decision=None,
    )


def _retry_state(
    state: TableStructureReviewMachineState,
) -> TableStructureReviewMachineState:
    return TableStructureReviewMachineState(
        review_id=state.review_id,
        phase=TableStructureReviewPhase.RETRY_DECISION,
        candidate_index=state.candidate_index,
        candidates_total=state.candidates_total,
        allowed_primary_actions=state.allowed_primary_actions,
        decision=None,
    )


def _mutation_effect(
    command: TableStructureReviewCommand,
) -> TableStructureReviewEffect:
    if command.action is TableStructureReviewAction.REMOVE_COLUMNS:
        return TableStructureReviewEffect(
            kind=TableStructureReviewEffectKind.REQUEST_REMOVE_COLUMNS,
            indices=tuple(command.payload["indices"]),
        )

    if command.action is TableStructureReviewAction.REORDER_COLUMNS:
        return TableStructureReviewEffect(
            kind=TableStructureReviewEffectKind.REQUEST_REORDER_COLUMNS,
            indices=tuple(command.payload["order"]),
        )

    if command.action is TableStructureReviewAction.RENAME_COLUMNS:
        return TableStructureReviewEffect(
            kind=TableStructureReviewEffectKind.REQUEST_RENAME_COLUMNS,
            renames=tuple(command.payload["renames"].items()),
        )

    if command.action is TableStructureReviewAction.ADD_COLUMNS:
        return TableStructureReviewEffect(
            kind=TableStructureReviewEffectKind.REQUEST_ADD_COLUMNS,
            names=tuple(command.payload["names"]),
        )

    raise TableStructureReviewStateMachineError(
        f"unsupported mutation effect for {command.action.value}"
    )


def transition_review_state(
    state: TableStructureReviewMachineState,
    command: TableStructureReviewCommand,
) -> TableStructureReviewTransition:
    """Validate one typed command and produce the next state/effect plan."""

    if not isinstance(state, TableStructureReviewMachineState):
        raise TableStructureReviewStateMachineError(
            "state must be TableStructureReviewMachineState"
        )
    if not isinstance(command, TableStructureReviewCommand):
        raise TableStructureReviewStateMachineError(
            "command must be TableStructureReviewCommand"
        )
    if command.review_id != state.review_id:
        raise TableStructureReviewStateMachineError(
            "command review_id does not match active state"
        )
    if state.phase is TableStructureReviewPhase.COMPLETED:
        raise TableStructureReviewStateMachineError(
            "completed review state cannot accept commands"
        )

    if state.phase is TableStructureReviewPhase.RETRY_DECISION:
        if command.action is not TableStructureReviewAction.RETRY_DECISION:
            raise TableStructureReviewStateMachineError(
                "retry phase accepts RETRY_DECISION only"
            )

        retry = command.payload["retry"]

        if retry:
            after = _primary_state(state)
            effect = TableStructureReviewEffect(
                kind=TableStructureReviewEffectKind.RETURN_TO_PRIMARY_REVIEW
            )
        else:
            after = _completed_state(
                state,
                TableStructureReviewDecision.ORIGINAL_STRUCTURE_RETAINED,
            )
            effect = TableStructureReviewEffect(
                kind=TableStructureReviewEffectKind.COMPLETE_ORIGINAL
            )

        return TableStructureReviewTransition(
            state_before=state,
            command=command,
            state_after=after,
            effect=effect,
        )

    if command.action is TableStructureReviewAction.RETRY_DECISION:
        raise TableStructureReviewStateMachineError(
            "RETRY_DECISION is valid only after REJECT"
        )

    if command.action not in state.allowed_primary_actions:
        raise TableStructureReviewStateMachineError(
            f"{command.action.value} is not allowed in this review request"
        )

    if command.action is TableStructureReviewAction.ACCEPT:
        after = _completed_state(
            state,
            TableStructureReviewDecision.ACCEPTED_REVIEW_STRUCTURE,
        )
        effect = TableStructureReviewEffect(
            kind=TableStructureReviewEffectKind.COMPLETE_ACCEPTED
        )

    elif command.action is TableStructureReviewAction.REJECT:
        after = _retry_state(state)
        effect = TableStructureReviewEffect(
            kind=TableStructureReviewEffectKind.REQUEST_RETRY_DECISION
        )

    elif command.action in _PRIMARY_MUTATION_ACTIONS:
        after = state
        effect = _mutation_effect(command)

    elif command.action in {
        TableStructureReviewAction.NEXT_CANDIDATE,
        TableStructureReviewAction.PREVIOUS_CANDIDATE,
    }:
        delta = (
            1
            if command.action is TableStructureReviewAction.NEXT_CANDIDATE
            else -1
        )
        zero_based = state.candidate_index - 1
        next_zero_based = (
            zero_based + delta
        ) % state.candidates_total
        next_index = next_zero_based + 1

        after = _primary_state(
            state,
            candidate_index=next_index,
        )
        effect = TableStructureReviewEffect(
            kind=(
                TableStructureReviewEffectKind
                .REQUEST_CANDIDATE_MATERIALIZATION
            ),
            navigation_delta=delta,
            candidate_materialization_required=True,
        )

    else:
        raise TableStructureReviewStateMachineError(
            f"unhandled primary review action: {command.action.value}"
        )

    return TableStructureReviewTransition(
        state_before=state,
        command=command,
        state_after=after,
        effect=effect,
    )