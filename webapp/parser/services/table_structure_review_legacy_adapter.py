"""Noncontrolling adapter between legacy CLI review grammar and typed review contracts.

This module may translate already-observed legacy review state and user tokens.
It must not invoke the legacy review loop, mutate parser tables, write learning
state, generate IDs/timestamps, perform transport, or exercise canonical authority.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from ..contracts.table_structure_review import (
    TableStructureReviewAction,
    TableStructureReviewCommand,
    TableStructureReviewDecision,
    TableStructureReviewRequest,
    TableStructureReviewResult,
)


ADAPTER_VERSION = "table_structure_review_legacy_adapter_v1"

RUNTIME_CONTROL_AUTHORITY = False
PARSER_MUTATION_AUTHORITY = False
LEARNING_SIDE_EFFECT_AUTHORITY = False
TRANSPORT_AUTHORITY = False
CLI_REPLACEMENT_AUTHORIZED = False
REVIEW_ID_GENERATION_AUTHORITY = False
TIMESTAMP_GENERATION_AUTHORITY = False
CANONICAL_AUTHORITY = False

PRIMARY_ALLOWED_ACTIONS = (
    TableStructureReviewAction.ACCEPT,
    TableStructureReviewAction.REJECT,
    TableStructureReviewAction.REMOVE_COLUMNS,
    TableStructureReviewAction.REORDER_COLUMNS,
    TableStructureReviewAction.RENAME_COLUMNS,
    TableStructureReviewAction.ADD_COLUMNS,
    TableStructureReviewAction.NEXT_CANDIDATE,
    TableStructureReviewAction.PREVIOUS_CANDIDATE,
)


class LegacyTableStructureReviewAdapterError(ValueError):
    """Raised when legacy evidence cannot be represented without semantic loss."""


def _normalize_legacy_token(value: str) -> str:
    if not isinstance(value, str):
        raise LegacyTableStructureReviewAdapterError(
            "legacy response must be a string"
        )
    return value.strip().lower()


def _require_headers(candidate_headers: Sequence[str]) -> tuple[str, ...]:
    if isinstance(candidate_headers, (str, bytes)):
        raise LegacyTableStructureReviewAdapterError(
            "candidate_headers must be a sequence of strings"
        )
    headers = tuple(candidate_headers)
    if not headers or not all(isinstance(header, str) for header in headers):
        raise LegacyTableStructureReviewAdapterError(
            "candidate_headers must contain at least one string"
        )
    return headers


def _copy_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    copied = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise LegacyTableStructureReviewAdapterError(
                "legacy review rows must be mappings"
            )
        copied.append(dict(row))
    return tuple(copied)


def build_review_request_from_legacy_preview(
    *,
    review_id: str,
    session_id: Optional[str],
    domain: str,
    contest: Optional[str],
    candidate_headers: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    candidate_index_zero_based: int,
    candidates_total: int,
    ml_avg_confidence: Optional[float],
) -> TableStructureReviewRequest:
    """Project already-observed legacy preview state into the inert typed request."""

    if isinstance(candidate_index_zero_based, bool) or not isinstance(
        candidate_index_zero_based, int
    ):
        raise LegacyTableStructureReviewAdapterError(
            "candidate_index_zero_based must be an integer"
        )

    headers = _require_headers(candidate_headers)
    preview_rows = _copy_rows(rows[:5])

    projected = []
    for row in preview_rows:
        projected.append(
            {
                header: row[header] if header in row else ""
                for header in headers
            }
        )

    return TableStructureReviewRequest(
        review_id=review_id,
        session_id=session_id,
        domain=domain,
        contest=contest,
        candidate_headers=headers,
        rows_preview=tuple(projected),
        candidate_index=candidate_index_zero_based + 1,
        candidates_total=candidates_total,
        ml_avg_confidence=ml_avg_confidence,
        allowed_actions=PRIMARY_ALLOWED_ACTIONS,
    )


def classify_legacy_primary_response(
    raw_response: str,
) -> Optional[TableStructureReviewAction]:
    """Map the legacy primary prompt token to an action without executing it."""

    token = _normalize_legacy_token(raw_response)

    mapping = {
        "": TableStructureReviewAction.ACCEPT,
        "y": TableStructureReviewAction.ACCEPT,
        "yes": TableStructureReviewAction.ACCEPT,
        "n": TableStructureReviewAction.REJECT,
        "no": TableStructureReviewAction.REJECT,
        "c": TableStructureReviewAction.REMOVE_COLUMNS,
        "o": TableStructureReviewAction.REORDER_COLUMNS,
        "r": TableStructureReviewAction.RENAME_COLUMNS,
        "a": TableStructureReviewAction.ADD_COLUMNS,
        "next": TableStructureReviewAction.NEXT_CANDIDATE,
        "nxt": TableStructureReviewAction.NEXT_CANDIDATE,
        "prev": TableStructureReviewAction.PREVIOUS_CANDIDATE,
        "previous": TableStructureReviewAction.PREVIOUS_CANDIDATE,
    }
    return mapping.get(token)


def adapt_legacy_no_payload_primary_response(
    *,
    review_id: str,
    raw_response: str,
) -> Optional[TableStructureReviewCommand]:
    """Create commands only for legacy primary actions that need no extra payload."""

    action = classify_legacy_primary_response(raw_response)

    if action is None:
        return None

    if action not in {
        TableStructureReviewAction.ACCEPT,
        TableStructureReviewAction.REJECT,
        TableStructureReviewAction.NEXT_CANDIDATE,
        TableStructureReviewAction.PREVIOUS_CANDIDATE,
    }:
        return None

    return TableStructureReviewCommand(
        review_id=review_id,
        action=action,
        payload=None,
    )


def adapt_legacy_retry_response(
    *,
    review_id: str,
    raw_response: str,
) -> TableStructureReviewCommand:
    """Preserve legacy retry semantics: only y/yes retries; every other token does not."""

    token = _normalize_legacy_token(raw_response)

    return TableStructureReviewCommand(
        review_id=review_id,
        action=TableStructureReviewAction.RETRY_DECISION,
        payload={"retry": token in {"y", "yes"}},
    )


def _parse_comma_digit_indices(
    raw_indices: str,
) -> tuple[int, ...]:
    if not isinstance(raw_indices, str):
        raise LegacyTableStructureReviewAdapterError(
            "legacy index text must be a string"
        )

    return tuple(
        int(item) - 1
        for item in raw_indices.split(",")
        if item.strip().isdigit()
    )


def build_remove_columns_command(
    *,
    review_id: str,
    candidate_headers: Sequence[str],
    raw_indices: str,
) -> Optional[TableStructureReviewCommand]:
    """Mirror legacy remove-index filtering without mutating headers or rows."""

    headers = _require_headers(candidate_headers)
    indices = tuple(
        index
        for index in _parse_comma_digit_indices(raw_indices)
        if 0 <= index < len(headers)
    )

    if not indices:
        return None

    return TableStructureReviewCommand(
        review_id=review_id,
        action=TableStructureReviewAction.REMOVE_COLUMNS,
        payload={"indices": indices},
    )


def build_reorder_columns_command(
    *,
    review_id: str,
    candidate_headers: Sequence[str],
    raw_order: str,
) -> Optional[TableStructureReviewCommand]:
    """Mirror legacy reorder token filtering without applying the order."""

    headers = _require_headers(candidate_headers)

    if not isinstance(raw_order, str):
        raise LegacyTableStructureReviewAdapterError(
            "legacy reorder text must be a string"
        )

    zero_based = tuple(
        int(item) - 1
        for item in raw_order.replace(",", " ").split()
        if item.strip().isdigit()
        and 0 < int(item) <= len(headers)
    )

    if not zero_based:
        return None

    return TableStructureReviewCommand(
        review_id=review_id,
        action=TableStructureReviewAction.REORDER_COLUMNS,
        payload={"order": zero_based},
    )


def build_rename_columns_command(
    *,
    review_id: str,
    candidate_headers: Sequence[str],
    raw_indices: str,
    raw_new_names: Sequence[str],
) -> Optional[TableStructureReviewCommand]:
    """Adapt the legacy multi-step rename only when it is losslessly representable."""

    headers = _require_headers(candidate_headers)
    selected = tuple(
        index
        for index in _parse_comma_digit_indices(raw_indices)
        if 0 <= index < len(headers)
    )

    if not selected:
        return None

    if len(set(selected)) != len(selected):
        raise LegacyTableStructureReviewAdapterError(
            "duplicate legacy rename indices cannot be represented losslessly "
            "by the typed rename mapping"
        )

    if isinstance(raw_new_names, (str, bytes)):
        raise LegacyTableStructureReviewAdapterError(
            "raw_new_names must be a sequence aligned to selected indices"
        )

    names = tuple(raw_new_names)

    if len(names) != len(selected):
        raise LegacyTableStructureReviewAdapterError(
            "raw_new_names count must match selected legacy rename indices"
        )

    renames = {}
    for index, raw_name in zip(selected, names):
        if not isinstance(raw_name, str):
            raise LegacyTableStructureReviewAdapterError(
                "legacy rename responses must be strings"
            )
        new_name = raw_name.strip()
        if new_name:
            renames[index] = new_name

    if not renames:
        return None

    return TableStructureReviewCommand(
        review_id=review_id,
        action=TableStructureReviewAction.RENAME_COLUMNS,
        payload={"renames": renames},
    )


def build_add_columns_command(
    *,
    review_id: str,
    candidate_headers: Sequence[str],
    raw_names: str,
) -> Optional[TableStructureReviewCommand]:
    """Mirror legacy add-column trimming and duplicate suppression."""

    headers = _require_headers(candidate_headers)

    if not isinstance(raw_names, str):
        raise LegacyTableStructureReviewAdapterError(
            "legacy add-column text must be a string"
        )

    existing = set(headers)
    names = []

    for raw_name in raw_names.split(","):
        name = raw_name.strip()
        if not name or name in existing:
            continue
        names.append(name)
        existing.add(name)

    if not names:
        return None

    return TableStructureReviewCommand(
        review_id=review_id,
        action=TableStructureReviewAction.ADD_COLUMNS,
        payload={"names": tuple(names)},
    )


def build_review_result_from_legacy_return(
    *,
    headers: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    accepted_review_structure: bool,
) -> TableStructureReviewResult:
    """Project an already-produced legacy return without invoking or mutating the loop."""

    if type(accepted_review_structure) is not bool:
        raise LegacyTableStructureReviewAdapterError(
            "accepted_review_structure must be bool"
        )

    typed_headers = _require_headers(headers)
    typed_rows = _copy_rows(rows)

    decision = (
        TableStructureReviewDecision.ACCEPTED_REVIEW_STRUCTURE
        if accepted_review_structure
        else TableStructureReviewDecision.ORIGINAL_STRUCTURE_RETAINED
    )

    return TableStructureReviewResult(
        headers=typed_headers,
        rows=typed_rows,
        decision=decision,
    )