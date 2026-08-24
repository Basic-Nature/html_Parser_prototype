"""Pure JSON-safe inspection projection for TablePipelineResult.

C2G 1.7 intentionally adds no route, persistence, database access, timestamps,
rows, headers, source URI, or source metadata.

The projection exists so future Ballot Lens inspection UI/API work can consume a
stable noncanonical evidence contract without treating parser evidence as
publication authority.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Mapping, Sequence

from webapp.parser.contracts.table_pipeline import (
    CompletenessInfo,
    PipelineWarning,
    SourceLocation,
    SourceProvenance,
    TablePipelineResult,
    TransformationRecord,
)

INSPECTION_CONTRACT = "pipeline_inspection_v1"
INSPECTION_AUTHORITY = "noncanonical_parser_evidence"


def _json_safe(value: Any, *, path: str) -> Any:
    """Return JSON-compatible primitives or fail closed.

    Unsupported objects are not stringified because doing so would hide type
    loss. Non-finite floats are rejected because they are not portable JSON
    values and can create implementation-specific behavior.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite float is not inspection-safe at {path}")
        return value

    if isinstance(value, Enum):
        return _json_safe(value.value, path=path)

    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"inspection mapping key must be str at {path}")
            out[key] = _json_safe(item, path=f"{path}.{key}")
        return out

    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]

    raise TypeError(
        f"unsupported inspection value type at {path}: "
        f"{type(value).__name__}"
    )


def _project_location(location: SourceLocation | None) -> dict[str, Any] | None:
    if location is None:
        return None
    return {
        "page_number": location.page_number,
        "table_index": location.table_index,
        "row_index": location.row_index,
        "column_index": location.column_index,
        "selector": location.selector,
    }


def _project_provenance(provenance: SourceProvenance) -> dict[str, Any]:
    """Project deliberately bounded provenance.

    source_uri and metadata are intentionally excluded in v1 because either can
    contain credentials, local paths, session material, or source-specific
    details that require a separate redaction contract before HTTP exposure.
    """

    return {
        "source_type": provenance.source_type,
        "source_sha256": provenance.source_sha256,
        "artifact_id": provenance.artifact_id,
        "evidence_ref": provenance.evidence_ref,
        "location": _project_location(provenance.location),
        "source_uri_included": False,
        "source_metadata_included": False,
    }


def _project_transformation(
    record: TransformationRecord,
) -> dict[str, Any]:
    return {
        "sequence": record.sequence,
        "from_stage": record.from_stage.value,
        "to_stage": record.to_stage.value,
        "operation": record.operation,
        "rule_source": record.rule_source,
        "confidence": record.confidence,
        "evidence_refs": list(record.evidence_refs),
        "details": _json_safe(
            record.details,
            path=f"transformations[{record.sequence}].details",
        ),
    }


def _project_warning(
    warning: PipelineWarning,
    *,
    index: int,
) -> dict[str, Any]:
    return {
        "code": warning.code,
        "message": warning.message,
        "stage": warning.stage.value,
        "severity": warning.severity.value,
        "requires_review": warning.requires_review,
        "evidence_refs": list(warning.evidence_refs),
        "details": _json_safe(
            warning.details,
            path=f"warnings[{index}].details",
        ),
    }


def _project_completeness(
    completeness: CompletenessInfo,
) -> dict[str, Any]:
    return {
        "state": completeness.state.value,
        "expected_count": completeness.expected_count,
        "observed_count": completeness.observed_count,
        "missing_count": completeness.missing_count,
        "null_value_count": completeness.null_value_count,
        "is_complete": completeness.is_complete,
        "notes": list(completeness.notes),
    }


def project_pipeline_inspection(
    result: TablePipelineResult,
) -> dict[str, Any]:
    """Project one parser result to the noncanonical inspection v1 contract."""

    transformations = [
        _project_transformation(record)
        for record in result.transformations
    ]
    warnings = [
        _project_warning(warning, index=index)
        for index, warning in enumerate(result.warnings)
    ]

    return {
        "contract": INSPECTION_CONTRACT,
        "authority": {
            "inspection": INSPECTION_AUTHORITY,
            "canonical": False,
            "write_kind": result.write_kind.value,
        },
        "stage": result.stage.value,
        "source_provenance": _project_provenance(result.source_provenance),
        "summary": {
            "header_count": len(result.headers),
            "row_count": len(result.rows),
            "transformation_count": len(transformations),
            "warning_count": len(warnings),
        },
        "completeness": _project_completeness(result.completeness),
        "transformations": transformations,
        "warnings": warnings,
        "rows_included": False,
        "headers_included": False,
        "automatic_timestamp": False,
    }