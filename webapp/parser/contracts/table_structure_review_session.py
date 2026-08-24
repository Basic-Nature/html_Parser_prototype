"""Typed off-runtime table-review session contract.

C2G 2.8.27 owns a coherent original table snapshot separately from the
current working table.  It deliberately does not reproduce the raw-Git legacy
hybrid REJECT/no-retry behavior of original headers plus current/mutated data.

This contract has no runtime transport, persistence, learning, or canonical
authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .table_structure_review import TableStructureReviewResult
    from .table_structure_review_execution import (
        TableStructureReviewEffectExecutionResult,
    )
    from ..services.table_structure_review_state_machine import (
        TableStructureReviewMachineState,
        TableStructureReviewTransition,
    )


CONTRACT_VERSION = "table_structure_review_session_v1"
RUNTIME_TRANSPORT_WIRED = False
CANONICAL_AUTHORITY = False


@dataclass(frozen=True)
class TableStructureReviewTableSnapshot:
    """Snapshot envelope.

    The session service constructs rows as read-only mapping proxies after
    deep-copy isolation. Election table cell values are preserved as supplied;
    generalized deep immutability of arbitrary nested cell objects is outside
    this checkpoint.
    """

    headers: Tuple[str, ...]
    rows: Tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class TableStructureReviewSession:
    """Off-runtime review session with separate baseline and working tables."""

    contract_version: ClassVar[str] = CONTRACT_VERSION
    runtime_transport_wired: ClassVar[bool] = RUNTIME_TRANSPORT_WIRED
    canonical_authority: ClassVar[bool] = CANONICAL_AUTHORITY

    review_id: str
    state: "TableStructureReviewMachineState"
    original: TableStructureReviewTableSnapshot
    working: TableStructureReviewTableSnapshot
    harmonization_context: Optional[Mapping[str, Any]] = None
    source_location_label: Optional[str] = None


@dataclass(frozen=True)
class TableStructureReviewSessionStep:
    """One state-machine transition interpreted against session-owned tables."""

    session_before: TableStructureReviewSession
    transition: "TableStructureReviewTransition"
    session_after: TableStructureReviewSession
    effect_execution: Optional["TableStructureReviewEffectExecutionResult"]
    completion: Optional["TableStructureReviewResult"]
