"""Behavior-neutral parser confidence contracts.

These are observations, not an authority score. Confidence dimensions remain
independent so later policies can inspect finite differences per dimension.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import Optional

from ._contract_validation import require_finite_number


PARSER_CONFIDENCE_CONTRACT = "parser_confidence_vector_v1"


class ConfidenceDimension(str, Enum):
    ACQUISITION = "acquisition"
    TEXT_RECOGNITION = "text_recognition"
    GEOMETRY = "geometry"
    STRUCTURE = "structure"
    SEMANTICS = "semantics"
    CONTEXT = "context"
    COVERAGE = "coverage"
    NORMALIZATION = "normalization"
    RECONCILIATION = "reconciliation"


CONFIDENCE_DIMENSION_NAMES = tuple(
    item.value for item in ConfidenceDimension
)


def _validate_unit_interval(name: str, value: Optional[float]) -> None:
    require_finite_number(
        name,
        value,
        minimum=0.0,
        maximum=1.0,
        allow_none=True,
    )


def _validate_delta(name: str, value: Optional[float]) -> None:
    require_finite_number(
        name,
        value,
        minimum=-1.0,
        maximum=1.0,
        allow_none=True,
    )


@dataclass(frozen=True)
class ConfidenceVector:
    """Independent parser-confidence observations.

    Deliberately has no ``overall_confidence`` field.
    """

    acquisition: Optional[float] = None
    text_recognition: Optional[float] = None
    geometry: Optional[float] = None
    structure: Optional[float] = None
    semantics: Optional[float] = None
    context: Optional[float] = None
    coverage: Optional[float] = None
    normalization: Optional[float] = None
    reconciliation: Optional[float] = None

    def __post_init__(self) -> None:
        for item in fields(self):
            _validate_unit_interval(item.name, getattr(self, item.name))

    def as_dict(self) -> dict[str, Optional[float]]:
        return {item.name: getattr(self, item.name) for item in fields(self)}

    def value(self, dimension: ConfidenceDimension) -> Optional[float]:
        if not isinstance(dimension, ConfidenceDimension):
            raise ValueError("dimension must be a ConfidenceDimension")
        return getattr(self, dimension.value)

    def known_dimensions(self) -> tuple[ConfidenceDimension, ...]:
        return tuple(
            dimension
            for dimension in ConfidenceDimension
            if self.value(dimension) is not None
        )

    def delta_from(self, before: "ConfidenceVector") -> "ConfidenceDelta":
        if not isinstance(before, ConfidenceVector):
            raise ValueError("before must be a ConfidenceVector")
        values: dict[str, Optional[float]] = {}
        for dimension in ConfidenceDimension:
            prior = before.value(dimension)
            current = self.value(dimension)
            values[dimension.value] = (
                None
                if prior is None or current is None
                else float(current) - float(prior)
            )
        return ConfidenceDelta(**values)


@dataclass(frozen=True)
class ConfidenceDelta:
    """Signed finite difference between two confidence observations."""

    acquisition: Optional[float] = None
    text_recognition: Optional[float] = None
    geometry: Optional[float] = None
    structure: Optional[float] = None
    semantics: Optional[float] = None
    context: Optional[float] = None
    coverage: Optional[float] = None
    normalization: Optional[float] = None
    reconciliation: Optional[float] = None

    def __post_init__(self) -> None:
        for item in fields(self):
            _validate_delta(item.name, getattr(self, item.name))

    def as_dict(self) -> dict[str, Optional[float]]:
        return {item.name: getattr(self, item.name) for item in fields(self)}

    def value(self, dimension: ConfidenceDimension) -> Optional[float]:
        if not isinstance(dimension, ConfidenceDimension):
            raise ValueError("dimension must be a ConfidenceDimension")
        return getattr(self, dimension.value)

    def has_nonnegative_values_for(
        self,
        dimensions: tuple[ConfidenceDimension, ...],
    ) -> bool:
        """Require explicit, non-negative gain evidence for each dimension."""
        for dimension in dimensions:
            value = self.value(dimension)
            if value is None or value < 0.0:
                return False
        return True
