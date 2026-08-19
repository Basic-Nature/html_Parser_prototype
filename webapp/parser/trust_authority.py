# Live privilege-tier effects on URL trust decisions.
#
# This module owns only the tier-aware adjustments that are active in the
# runtime URL trust path. Generic trust-score thresholds remain owned by
# url_trust_scorer.py. Legacy, currently-unused tier threshold surfaces remain
# in privilege_tiers.py for compatibility until a later cleanup proves they
# can be retired safely.

from __future__ import annotations

from typing import Mapping, Any

from webapp.parser.utils.privilege_tiers import PrivilegeTier


SUSPICIOUS_ADMIN_BOOST_TLDS = {
    ".xyz",
    ".top",
    ".faith",
    ".zip",
    ".icu",
    ".click",
    ".download",
    ".gq",
    ".ml",
}


def should_apply_admin_boost_policy(
    trust_factors: Mapping[str, Any],
    tier: PrivilegeTier | int,
    domain: str,
    *,
    domain_allowlisted: bool,
) -> bool:
    """Return whether the live admin trust-score boost is permitted."""
    tier_value = int(tier)

    if tier_value not in (
        int(PrivilegeTier.ADMIN_REVIEWER),
        int(PrivilegeTier.ADMIN_FULL_TRUST),
        int(PrivilegeTier.ROOT_ADMIN),
    ):
        return False

    if any(domain.endswith(tld) for tld in SUSPICIOUS_ADMIN_BOOST_TLDS):
        return False

    return bool(
        trust_factors.get("verified_domain", False)
        or trust_factors.get("gov_domain", False)
        or domain_allowlisted
    )


def admin_boost_amount(
    tier: PrivilegeTier | int | None,
) -> int:
    """Return the boost amount used by the live URL trust scorer."""
    if tier is None:
        return 0

    tier_value = int(tier)

    if tier_value in (
        int(PrivilegeTier.ADMIN_FULL_TRUST),
        int(PrivilegeTier.ROOT_ADMIN),
    ):
        return 10

    if tier_value == int(PrivilegeTier.ADMIN_REVIEWER):
        return 5

    return 0


def should_quarantine_for_tier(
    trust_score: int,
    privilege_tier: PrivilegeTier | int | None,
    *,
    low_threshold: int,
    medium_threshold: int,
) -> bool:
    """Apply the live tier-aware quarantine policy."""
    if privilege_tier is not None:
        tier_value = int(privilege_tier)

        if tier_value == int(PrivilegeTier.ROOT_ADMIN):
            return False

        if tier_value == int(PrivilegeTier.ADMIN_FULL_TRUST):
            return 40 <= trust_score < medium_threshold

    return low_threshold <= trust_score < medium_threshold


def should_reject_for_tier(
    trust_score: int,
    privilege_tier: PrivilegeTier | int | None,
    *,
    low_threshold: int,
) -> bool:
    """Apply the live tier-aware rejection policy."""
    if privilege_tier is not None:
        tier_value = int(privilege_tier)

        if tier_value == int(PrivilegeTier.ROOT_ADMIN):
            return False

        if tier_value == int(PrivilegeTier.ADMIN_FULL_TRUST):
            return trust_score < 20

    return trust_score < low_threshold
