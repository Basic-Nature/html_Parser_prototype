"""Shared validation helpers for parser contract values.

These helpers are intentionally dependency-free and behavior-neutral.
"""

from __future__ import annotations

import math
import re
from typing import Optional


_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def require_nonempty_text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def require_finite_number(
    name: str,
    value: float | int | None,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    allow_none: bool = False,
) -> Optional[float]:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{name} must not be None")
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not bool")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and numeric < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and numeric > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return numeric


def require_nonnegative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def require_positive_int(name: str, value: int) -> int:
    require_nonnegative_int(name, value)
    if value == 0:
        raise ValueError(f"{name} must be > 0")
    return value


def require_sha256(name: str, value: str) -> str:
    require_nonempty_text(name, value)
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256")
    return value.lower()
