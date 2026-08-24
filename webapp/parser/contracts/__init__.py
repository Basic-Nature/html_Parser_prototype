"""Typed parser contracts introduced by C2G.

These modules define semantic boundaries. Importing them must not perform
extraction, persistence, canonical promotion, database access, or I/O.
"""

from .table_pipeline import (
    CompletenessInfo,
    CompletenessState,
    GovernedBoundary,
    PipelineWarning,
    SourceLocation,
    SourceProvenance,
    TablePipelineResult,
    TableStage,
    TransformationRecord,
    WarningSeverity,
    collect_transformations,
    current_transformation_sequence,
    record_transformation,
    is_forward_stage_transition,
)

__all__ = [
    "CompletenessInfo",
    "CompletenessState",
    "GovernedBoundary",
    "PipelineWarning",
    "SourceLocation",
    "SourceProvenance",
    "TablePipelineResult",
    "TableStage",
    "TransformationRecord",
    "WarningSeverity",
    "collect_transformations",
    "current_transformation_sequence",
    "record_transformation",
    "is_forward_stage_transition",
]