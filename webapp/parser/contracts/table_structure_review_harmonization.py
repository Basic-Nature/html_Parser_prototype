"""Typed request/result contract for table-structure-review harmonization.

C2G 2.8.23 intentionally separates:
    pure review mutations -> explicit harmonization boundary

This contract is jurisdiction-neutral.  It preserves the caller-declared
source location label as evidence metadata while allowing the legacy
harmonizer's OUTPUT semantics to be observed separately.

This is not a canonical election schema and is not an export profile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple


LEGACY_HARMONIZATION_KNOWN_HAZARDS: Tuple[str, ...] = (
    "PERCENT_REPORTED_ZERO_TRUTHINESS",
    "PERCENT_ACCUMULATOR_TRUTHINESS",
    "LEGACY_LOCATION_LABEL_COLLAPSE",
    "LEGACY_CALLER_OWNED_ROW_MUTATION",
    "ROW_DEDUP_CAN_CHANGE_CARDINALITY",
    "ROW_ONLY_EXTRA_HEADER_SET_ORDER",
    "CASE_INSENSITIVE_HEADER_DEDUP_CAN_DROP_COLUMNS",
)


@dataclass(frozen=True)
class TableStructureHarmonizationRequest:
    """Typed input to the isolated harmonization parity boundary."""

    headers: Tuple[str, ...]
    rows: Tuple[Mapping[str, Any], ...]
    context: Mapping[str, Any] | None = None
    source_location_label: str | None = None


@dataclass(frozen=True)
class TableStructureHarmonizationResult:
    """Legacy-output-parity result plus explicit non-authority metadata."""

    headers: Tuple[str, ...]
    rows: Tuple[Mapping[str, Any], ...]
    source_location_label: str | None
    legacy_output_semantics_preserved: bool = True
    caller_input_mutation_preserved: bool = False
    normalized_internal_schema_authority: bool = False
    canonical_authority: bool = False
    known_hazards: Tuple[str, ...] = LEGACY_HARMONIZATION_KNOWN_HAZARDS
