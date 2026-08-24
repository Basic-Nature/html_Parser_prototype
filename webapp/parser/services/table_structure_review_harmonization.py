"""Copied-input parity adapter for the legacy table harmonizer.

The legacy harmonizer is deliberately invoked only on deep-copied inputs.
Its returned headers/rows retain legacy OUTPUT semantics for C2G 2.8.23.
Its incidental mutation of caller-owned row dictionaries is not propagated.

No state machine, parser runtime, transport, frontend, learning, or
canonical-write path imports this adapter in this checkpoint.
"""

from __future__ import annotations

import copy
from typing import Any

from ..contracts.table_structure_review_harmonization import (
    TableStructureHarmonizationRequest,
    TableStructureHarmonizationResult,
)
from ..utils.detect import harmonize_headers_and_data


def harmonize_table_structure_review(
    request: TableStructureHarmonizationRequest,
) -> TableStructureHarmonizationResult:
    """Apply the current legacy harmonizer to isolated working copies only."""

    working_headers = copy.deepcopy(list(request.headers))
    working_rows = copy.deepcopy([dict(row) for row in request.rows])
    working_context: dict[str, Any] | None = (
        copy.deepcopy(dict(request.context))
        if request.context is not None
        else None
    )

    output_headers, output_rows = harmonize_headers_and_data(
        working_headers,
        working_rows,
        working_context,
    )

    return TableStructureHarmonizationResult(
        headers=tuple(copy.deepcopy(output_headers)),
        rows=tuple(copy.deepcopy(output_rows)),
        source_location_label=request.source_location_label,
    )
