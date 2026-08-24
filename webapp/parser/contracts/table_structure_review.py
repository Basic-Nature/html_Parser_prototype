"""Typed, noncanonical contract scaffold for table-structure human review.

This module deliberately contains no runtime transport, parser invocation,
persistence, browser, frontend, Socket.IO, or canonical promotion behavior.

It models the state/action/result vocabulary inventoried from the existing CLI
review workflow while leaving that workflow unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, ClassVar, Mapping, Optional, Tuple


CONTRACT_VERSION = "table_structure_review_v1"
CANONICAL_AUTHORITY = False
RUNTIME_TRANSPORT_WIRED = False
CLI_REPLACEMENT_AUTHORIZED = False
PARSER_CONTROL_FLOW_AUTHORITY = False
LEARNING_SIDE_EFFECT_AUTHORITY = False
REVIEW_ID_GENERATION_AUTHORITY = False
TIMESTAMP_GENERATION_AUTHORITY = False

MAX_PREVIEW_ROWS = 5


ReviewScalar = Optional[str | int | float | bool]
ReviewRow = Mapping[str, ReviewScalar]


class TableStructureReviewContractError(ValueError):
    """Raised when typed review evidence violates the scaffold contract."""


class TableStructureReviewAction(str, Enum):
    """Typed actions derived from the existing CLI review state machine."""

    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    RETRY_DECISION = "RETRY_DECISION"
    REMOVE_COLUMNS = "REMOVE_COLUMNS"
    REORDER_COLUMNS = "REORDER_COLUMNS"
    RENAME_COLUMNS = "RENAME_COLUMNS"
    ADD_COLUMNS = "ADD_COLUMNS"
    NEXT_CANDIDATE = "NEXT_CANDIDATE"
    PREVIOUS_CANDIDATE = "PREVIOUS_CANDIDATE"


class TableStructureReviewDecision(str, Enum):
    """Explicit noncanonical result metadata derived from existing returns."""

    ACCEPTED_REVIEW_STRUCTURE = "ACCEPTED_REVIEW_STRUCTURE"
    ORIGINAL_STRUCTURE_RETAINED = "ORIGINAL_STRUCTURE_RETAINED"


_NO_PAYLOAD_ACTIONS = frozenset(
    {
        TableStructureReviewAction.ACCEPT,
        TableStructureReviewAction.REJECT,
        TableStructureReviewAction.NEXT_CANDIDATE,
        TableStructureReviewAction.PREVIOUS_CANDIDATE,
    }
)


def _require_exact_mapping_keys(
    payload: Mapping[str, Any],
    expected: frozenset[str],
    *,
    action: TableStructureReviewAction,
) -> None:
    actual = frozenset(payload.keys())
    if actual != expected:
        raise TableStructureReviewContractError(
            f"{action.value} payload keys must be exactly "
            f"{sorted(expected)!r}; got {sorted(actual)!r}"
        )


def _require_integer_sequence(value: Any, *, field_name: str) -> None:
    if not isinstance(value, (tuple, list)):
        raise TableStructureReviewContractError(
            f"{field_name} must be a tuple/list of integers"
        )
    if not value:
        raise TableStructureReviewContractError(
            f"{field_name} must not be empty"
        )
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise TableStructureReviewContractError(
                f"{field_name} must contain integers only"
            )


def _require_string_sequence(value: Any, *, field_name: str) -> None:
    if not isinstance(value, (tuple, list)):
        raise TableStructureReviewContractError(
            f"{field_name} must be a tuple/list of strings"
        )
    if not value:
        raise TableStructureReviewContractError(
            f"{field_name} must not be empty"
        )
    for item in value:
        if not isinstance(item, str):
            raise TableStructureReviewContractError(
                f"{field_name} must contain strings only"
            )


def _validate_scalar(value: Any, *, location: str) -> None:
    if value is None:
        return
    if type(value) not in (str, int, float, bool):
        raise TableStructureReviewContractError(
            f"{location} contains unsupported value type "
            f"{type(value).__name__}"
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise TableStructureReviewContractError(
            f"{location} must not contain NaN or Infinity"
        )


def _validate_rows(
    rows: Tuple[ReviewRow, ...],
    *,
    field_name: str,
    max_rows: Optional[int] = None,
) -> None:
    if not isinstance(rows, tuple):
        raise TableStructureReviewContractError(
            f"{field_name} must be a tuple"
        )
    if max_rows is not None and len(rows) > max_rows:
        raise TableStructureReviewContractError(
            f"{field_name} may contain at most {max_rows} rows"
        )

    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TableStructureReviewContractError(
                f"{field_name}[{row_index}] must be a mapping"
            )
        for key, value in row.items():
            if not isinstance(key, str):
                raise TableStructureReviewContractError(
                    f"{field_name}[{row_index}] keys must be strings"
                )
            _validate_scalar(
                value,
                location=f"{field_name}[{row_index}][{key!r}]",
            )


def _validate_headers(
    headers: Tuple[str, ...],
    *,
    field_name: str,
) -> None:
    if not isinstance(headers, tuple):
        raise TableStructureReviewContractError(
            f"{field_name} must be a tuple"
        )
    if not headers:
        raise TableStructureReviewContractError(
            f"{field_name} must not be empty"
        )
    if not all(isinstance(header, str) for header in headers):
        raise TableStructureReviewContractError(
            f"{field_name} must contain strings only"
        )


@dataclass(frozen=True)
class TableStructureReviewRequest:
    """Immutable review-state envelope with no execution authority."""

    contract_version: ClassVar[str] = CONTRACT_VERSION
    canonical_authority: ClassVar[bool] = CANONICAL_AUTHORITY
    runtime_transport_wired: ClassVar[bool] = RUNTIME_TRANSPORT_WIRED

    review_id: str
    session_id: Optional[str]
    domain: str
    contest: Optional[str]
    candidate_headers: Tuple[str, ...]
    rows_preview: Tuple[ReviewRow, ...]
    candidate_index: int
    candidates_total: int
    ml_avg_confidence: Optional[float]
    allowed_actions: Tuple[TableStructureReviewAction, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.review_id, str) or not self.review_id:
            raise TableStructureReviewContractError(
                "review_id must be a non-empty caller-provided string"
            )
        if self.session_id is not None and not isinstance(
            self.session_id, str
        ):
            raise TableStructureReviewContractError(
                "session_id must be str or None"
            )
        if not isinstance(self.domain, str) or not self.domain:
            raise TableStructureReviewContractError(
                "domain must be a non-empty string"
            )
        if self.contest is not None and not isinstance(self.contest, str):
            raise TableStructureReviewContractError(
                "contest must be str or None"
            )

        _validate_headers(
            self.candidate_headers,
            field_name="candidate_headers",
        )
        _validate_rows(
            self.rows_preview,
            field_name="rows_preview",
            max_rows=MAX_PREVIEW_ROWS,
        )

        if isinstance(self.candidate_index, bool) or not isinstance(
            self.candidate_index, int
        ):
            raise TableStructureReviewContractError(
                "candidate_index must be an integer"
            )
        if isinstance(self.candidates_total, bool) or not isinstance(
            self.candidates_total, int
        ):
            raise TableStructureReviewContractError(
                "candidates_total must be an integer"
            )
        if self.candidates_total < 1:
            raise TableStructureReviewContractError(
                "candidates_total must be at least 1"
            )
        if not 1 <= self.candidate_index <= self.candidates_total:
            raise TableStructureReviewContractError(
                "candidate_index must be within candidates_total"
            )

        if self.ml_avg_confidence is not None:
            if isinstance(self.ml_avg_confidence, bool) or not isinstance(
                self.ml_avg_confidence, (int, float)
            ):
                raise TableStructureReviewContractError(
                    "ml_avg_confidence must be numeric or None"
                )
            if not math.isfinite(float(self.ml_avg_confidence)):
                raise TableStructureReviewContractError(
                    "ml_avg_confidence must not be NaN or Infinity"
                )

        if not isinstance(self.allowed_actions, tuple):
            raise TableStructureReviewContractError(
                "allowed_actions must be a tuple"
            )
        if not self.allowed_actions:
            raise TableStructureReviewContractError(
                "allowed_actions must not be empty"
            )
        for action in self.allowed_actions:
            if not isinstance(action, TableStructureReviewAction):
                raise TableStructureReviewContractError(
                    "allowed_actions must contain TableStructureReviewAction"
                )
        if len(set(self.allowed_actions)) != len(self.allowed_actions):
            raise TableStructureReviewContractError(
                "allowed_actions must not contain duplicates"
            )


@dataclass(frozen=True)
class TableStructureReviewCommand:
    """Typed human-review command. Validation is fail-closed."""

    contract_version: ClassVar[str] = CONTRACT_VERSION
    canonical_authority: ClassVar[bool] = CANONICAL_AUTHORITY
    runtime_transport_wired: ClassVar[bool] = RUNTIME_TRANSPORT_WIRED

    review_id: str
    action: TableStructureReviewAction
    payload: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.review_id, str) or not self.review_id:
            raise TableStructureReviewContractError(
                "review_id must be a non-empty caller-provided string"
            )
        if not isinstance(self.action, TableStructureReviewAction):
            raise TableStructureReviewContractError(
                "action must be TableStructureReviewAction"
            )

        if self.action in _NO_PAYLOAD_ACTIONS:
            if self.payload is not None:
                raise TableStructureReviewContractError(
                    f"{self.action.value} does not accept a payload"
                )
            return

        if not isinstance(self.payload, Mapping):
            raise TableStructureReviewContractError(
                f"{self.action.value} requires a mapping payload"
            )

        if self.action is TableStructureReviewAction.RETRY_DECISION:
            _require_exact_mapping_keys(
                self.payload,
                frozenset({"retry"}),
                action=self.action,
            )
            if type(self.payload["retry"]) is not bool:
                raise TableStructureReviewContractError(
                    "RETRY_DECISION payload 'retry' must be bool"
                )
            return

        if self.action is TableStructureReviewAction.REMOVE_COLUMNS:
            _require_exact_mapping_keys(
                self.payload,
                frozenset({"indices"}),
                action=self.action,
            )
            _require_integer_sequence(
                self.payload["indices"],
                field_name="indices",
            )
            return

        if self.action is TableStructureReviewAction.REORDER_COLUMNS:
            _require_exact_mapping_keys(
                self.payload,
                frozenset({"order"}),
                action=self.action,
            )
            _require_integer_sequence(
                self.payload["order"],
                field_name="order",
            )
            return

        if self.action is TableStructureReviewAction.RENAME_COLUMNS:
            _require_exact_mapping_keys(
                self.payload,
                frozenset({"renames"}),
                action=self.action,
            )
            renames = self.payload["renames"]
            if not isinstance(renames, Mapping) or not renames:
                raise TableStructureReviewContractError(
                    "renames must be a non-empty mapping"
                )
            for index, name in renames.items():
                if isinstance(index, bool) or not isinstance(index, int):
                    raise TableStructureReviewContractError(
                        "rename keys must be integer column indices"
                    )
                if not isinstance(name, str):
                    raise TableStructureReviewContractError(
                        "rename values must be strings"
                    )
            return

        if self.action is TableStructureReviewAction.ADD_COLUMNS:
            _require_exact_mapping_keys(
                self.payload,
                frozenset({"names"}),
                action=self.action,
            )
            _require_string_sequence(
                self.payload["names"],
                field_name="names",
            )
            return

        raise TableStructureReviewContractError(
            f"unhandled typed action: {self.action.value}"
        )


@dataclass(frozen=True)
class TableStructureReviewResult:
    """Typed noncanonical result; carries no promotion or persistence authority."""

    contract_version: ClassVar[str] = CONTRACT_VERSION
    canonical_authority: ClassVar[bool] = CANONICAL_AUTHORITY
    runtime_transport_wired: ClassVar[bool] = RUNTIME_TRANSPORT_WIRED

    headers: Tuple[str, ...]
    rows: Tuple[ReviewRow, ...]
    decision: TableStructureReviewDecision

    def __post_init__(self) -> None:
        _validate_headers(self.headers, field_name="headers")
        _validate_rows(self.rows, field_name="rows")

        if not isinstance(self.decision, TableStructureReviewDecision):
            raise TableStructureReviewContractError(
                "decision must be TableStructureReviewDecision"
            )