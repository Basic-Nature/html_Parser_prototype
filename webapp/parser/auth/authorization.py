# Canonical minimum-tier authorization primitives.
#
# This module answers only whether an already-resolved privilege tier satisfies
# a required minimum tier. Identity resolution, tier vocabulary, trust scoring,
# HTTP denial payloads, and route-specific policy remain in their current
# owners during Tranche 1E-A.

from __future__ import annotations

from enum import IntEnum


TierValue = IntEnum | int


def tier_satisfies(
    actual_tier: TierValue | None,
    required_tier: TierValue,
) -> bool:
    if actual_tier is None:
        return False

    return int(actual_tier) >= int(required_tier)
