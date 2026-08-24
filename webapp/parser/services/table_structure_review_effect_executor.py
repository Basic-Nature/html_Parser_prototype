"""Off-runtime executor for table-review mutation effects.

Supported effects only:
- REQUEST_REMOVE_COLUMNS
- REQUEST_REORDER_COLUMNS
- REQUEST_RENAME_COLUMNS
- REQUEST_ADD_COLUMNS

Each supported effect is executed as:
    accepted pure mutation -> accepted copied-input harmonization boundary

Control-flow, retry, completion, navigation, and candidate-materialization
effects fail closed here.  The state machine itself remains planning-only and
does not import or call this service in C2G 2.8.25.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

from ..contracts.table_structure_review_execution import (
    TableStructureReviewEffectExecutionResult,
)
from ..contracts.table_structure_review_harmonization import (
    TableStructureHarmonizationRequest,
)
from .table_structure_review_harmonization import (
    harmonize_table_structure_review,
)
from .table_structure_review_mutations import (
    TableStructureMutationResult,
    apply_add_columns,
    apply_remove_columns,
    apply_rename_columns,
    apply_reorder_columns,
)
from .table_structure_review_state_machine import (
    TableStructureReviewEffect,
    TableStructureReviewEffectKind,
)


SERVICE_VERSION = "table_structure_review_effect_executor_v1"
STATE_MACHINE_CALLER_WIRED = False
RUNTIME_CALLER_WIRED = False
TRANSPORT_WIRED = False
CANDIDATE_MATERIALIZATION_AUTHORITY = False
LEARNING_SIDE_EFFECT_AUTHORITY = False
CANONICAL_AUTHORITY = False


class TableStructureReviewEffectExecutionError(ValueError):
    """Raised when an effect is outside this executor's mutation-only scope."""


_SUPPORTED_MUTATION_EFFECTS = frozenset(
    {
        TableStructureReviewEffectKind.REQUEST_REMOVE_COLUMNS,
        TableStructureReviewEffectKind.REQUEST_REORDER_COLUMNS,
        TableStructureReviewEffectKind.REQUEST_RENAME_COLUMNS,
        TableStructureReviewEffectKind.REQUEST_ADD_COLUMNS,
    }
)


def _apply_mutation_effect(
    headers: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    effect: TableStructureReviewEffect,
) -> TableStructureMutationResult:
    if effect.kind is TableStructureReviewEffectKind.REQUEST_REMOVE_COLUMNS:
        return apply_remove_columns(
            headers,
            rows,
            effect.indices,
        )

    if effect.kind is TableStructureReviewEffectKind.REQUEST_REORDER_COLUMNS:
        # The accepted state machine deliberately stores REORDER command
        # payload["order"] in effect.indices.
        return apply_reorder_columns(
            headers,
            rows,
            effect.indices,
        )

    if effect.kind is TableStructureReviewEffectKind.REQUEST_RENAME_COLUMNS:
        rename_items = tuple(effect.renames)
        rename_map = dict(rename_items)

        if len(rename_map) != len(rename_items):
            raise TableStructureReviewEffectExecutionError(
                "rename effect contains duplicate column indices"
            )

        return apply_rename_columns(
            headers,
            rows,
            rename_map,
        )

    if effect.kind is TableStructureReviewEffectKind.REQUEST_ADD_COLUMNS:
        return apply_add_columns(
            headers,
            rows,
            effect.names,
        )

    raise TableStructureReviewEffectExecutionError(
        f"unsupported mutation effect: {effect.kind.value}"
    )


def execute_table_structure_review_mutation_effect(
    headers: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    effect: TableStructureReviewEffect,
    *,
    harmonization_context: Mapping[str, Any] | None = None,
    source_location_label: str | None = None,
) -> TableStructureReviewEffectExecutionResult:
    """Execute exactly one planned mutation effect off-runtime, then harmonize."""

    if not isinstance(effect, TableStructureReviewEffect):
        raise TableStructureReviewEffectExecutionError(
            "effect must be TableStructureReviewEffect"
        )

    if effect.kind not in _SUPPORTED_MUTATION_EFFECTS:
        raise TableStructureReviewEffectExecutionError(
            "executor accepts mutation effects only; "
            f"received {effect.kind.value}"
        )

    if effect.parser_mutation_applied:
        raise TableStructureReviewEffectExecutionError(
            "planned effect must not already claim parser mutation"
        )

    if effect.learning_side_effect_applied:
        raise TableStructureReviewEffectExecutionError(
            "planned effect must not claim learning side effects"
        )

    original_headers = copy.deepcopy(list(headers))
    original_rows = copy.deepcopy([dict(row) for row in rows])

    mutation = _apply_mutation_effect(
        original_headers,
        original_rows,
        effect,
    )

    harmonized = harmonize_table_structure_review(
        TableStructureHarmonizationRequest(
            headers=tuple(mutation.headers),
            rows=tuple(copy.deepcopy(mutation.rows)),
            context=(
                copy.deepcopy(dict(harmonization_context))
                if harmonization_context is not None
                else None
            ),
            source_location_label=source_location_label,
        )
    )

    return TableStructureReviewEffectExecutionResult(
        effect_kind=effect.kind.value,
        pre_harmonization_headers=tuple(copy.deepcopy(mutation.headers)),
        pre_harmonization_rows=tuple(copy.deepcopy(mutation.rows)),
        headers=tuple(copy.deepcopy(harmonized.headers)),
        rows=tuple(copy.deepcopy(harmonized.rows)),
        source_location_label=harmonized.source_location_label,
    )
