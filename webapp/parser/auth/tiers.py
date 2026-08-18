
# Canonical privilege-tier vocabulary helpers.
#
# This transitional module centralizes tier-name normalization and presentation
# without moving the existing PrivilegeTier enum or principal-to-tier resolver.
#
# PrivilegeTier and get_principal_tier remain owned by
# webapp.parser.utils.privilege_tiers during Tranche 1D so enum module identity,
# principal mapping, trust scoring, health authorization, and existing imports
# remain stable.

from __future__ import annotations

from webapp.parser.utils.privilege_tiers import PrivilegeTier


_REQUIRED_TIER_ALIASES: dict[str, PrivilegeTier] = {
    "reviewer": PrivilegeTier.STANDARD_USER,
    "standard_user": PrivilegeTier.STANDARD_USER,
    "admin_reviewer": PrivilegeTier.ADMIN_REVIEWER,
    "admin_full_trust": PrivilegeTier.ADMIN_FULL_TRUST,
    "root_admin": PrivilegeTier.ROOT_ADMIN,
}


def normalize_required_tier(tier: str) -> PrivilegeTier:
    # Unknown labels intentionally fail closed to ROOT_ADMIN.
    normalized = str(tier or "").strip().lower().replace("-", "_")
    return _REQUIRED_TIER_ALIASES.get(
        normalized,
        PrivilegeTier.ROOT_ADMIN,
    )


def tier_level(tier: PrivilegeTier | int) -> int:
    return int(PrivilegeTier(int(tier)))


def tier_name(tier: PrivilegeTier | int) -> str:
    return PrivilegeTier(int(tier)).name


def tier_display(tier: PrivilegeTier | int) -> str:
    return PrivilegeTier(int(tier)).name_display
