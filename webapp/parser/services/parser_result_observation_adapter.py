"""Behavior-neutral adaptation of an already-finalized parser result.

This service wraps final parser headers/rows in the existing typed
``TablePipelineResult`` contract solely for noncanonical observation projection.

It does not:
- invoke a parser, table builder, OCR, browser, finalizer, or persistence layer;
- alter headers, rows, cell values, candidate names, vote methods, or totals;
- hash files, paths, URLs, or source content;
- add timestamps or canonical authority.

An upstream immutable-content SHA-256 may be supplied when already known.
Missing identity remains unknown.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from webapp.parser.contracts.table_pipeline import (
    SourceProvenance,
    TablePipelineResult,
    TableStage,
    TransformationRecord,
)

PARSER_RESULT_OBSERVATION_ADAPTER_CONTRACT = (
    "parser_result_observation_adapter_v1"
)


def adapt_final_parser_result_for_observation(
    headers: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    *,
    source_type: str,
    source_sha256: str | None = None,
) -> TablePipelineResult:
    """Wrap an existing final parser table without changing semantic values."""
    if isinstance(headers, (str, bytes)) or not isinstance(headers, Sequence):
        raise TypeError("headers must be a non-string sequence")
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError("rows must be a non-string sequence")

    copied_headers: list[str] = []
    for header in headers:
        if not isinstance(header, str):
            raise TypeError("every header must be a string")
        copied_headers.append(header)

    copied_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("every row must be a mapping")
        copied_rows.append(dict(row))

    provenance = SourceProvenance(
        source_type=source_type,
        source_sha256=source_sha256,
        metadata={
            "adapter": PARSER_RESULT_OBSERVATION_ADAPTER_CONTRACT,
            "semantic_value_mutation": False,
        },
    )
    boundary_record = TransformationRecord(
        sequence=0,
        from_stage=TableStage.INTERPRETED,
        to_stage=TableStage.INTERPRETED,
        operation="final_parser_result_observation_adaptation",
        rule_source=(
            "parser_result_observation_adapter."
            "adapt_final_parser_result_for_observation"
        ),
        confidence=None,
        details={
            "adapter": PARSER_RESULT_OBSERVATION_ADAPTER_CONTRACT,
            "source_type": source_type,
            "semantic_value_mutation": False,
        },
    )

    return TablePipelineResult.from_sequences(
        stage=TableStage.INTERPRETED,
        headers=copied_headers,
        rows=copied_rows,
        source_provenance=provenance,
        transformations=(boundary_record,),
    )
