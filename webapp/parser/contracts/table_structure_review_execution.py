"""Typed result contract for off-runtime table-review mutation execution.

C2G 2.8.25 composes already-accepted pure mutation semantics with the
already-accepted copied-input legacy harmonization parity boundary.

This contract is not a runtime transport contract, not a normalized election
schema, and not canonical authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple


CONTRACT_VERSION = "table_structure_review_effect_execution_v1"
RUNTIME_TRANSPORT_WIRED = False
CANONICAL_AUTHORITY = False


@dataclass(frozen=True)
class TableStructureReviewEffectExecutionResult:
    """Immutable provenance-rich result of one mutation-effect execution."""

    effect_kind: str
    pre_harmonization_headers: Tuple[str, ...]
    pre_harmonization_rows: Tuple[Mapping[str, Any], ...]
    headers: Tuple[str, ...]
    rows: Tuple[Mapping[str, Any], ...]
    source_location_label: str | None
    pure_mutation_applied: bool = True
    harmonization_applied: bool = True
    legacy_output_semantics_preserved: bool = True
    caller_input_mutation_preserved: bool = False
    candidate_materialization_applied: bool = False
    learning_side_effect_applied: bool = False
    runtime_transport_wired: bool = RUNTIME_TRANSPORT_WIRED
    canonical_authority: bool = CANONICAL_AUTHORITY
