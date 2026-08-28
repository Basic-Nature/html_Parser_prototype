from __future__ import annotations

from enum import Enum
from typing import Any, Mapping


class Capability(str, Enum):
    PUBLIC_READ = "public_read"
    TRUSTED_ACTION = "trusted_action"
    FRESH_PROOF_REVIEW = "fresh_proof_review"
    SYSTEM_ONLY = "system_only"


SAFE_READ_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

PUBLIC_READ_SURFACES = frozenset(
    {
        "ballotlens_canonical",
        "data_framework_scaffold",
        "data_framework_scaffold_csv",
        "data_framework_curated",
        "data_framework_warehouse_status",
        "data_framework_canonical_facets",
        "election_data_states_counties",
        "election_data_stats",
        "election_data_worklist_public_projection",
        "workflow_v1_public_items",
        "workflow_v1_facets",
        "workflow_v1_stats",
    }
)

TRUSTED_ACTION_AUTHORITY_STATES = frozenset(
    {
        "fresh_certificate",
        "certificate_session",
    }
)

FRESH_PROOF_REVIEW_AUTHORITY_STATES = frozenset(
    {
        "fresh_certificate",
    }
)


class CapabilityPolicyError(RuntimeError):
    pass


def assert_public_read_surface(surface: str, method: str) -> Capability:
    normalized_surface = str(surface or "").strip()
    normalized_method = str(method or "").strip().upper()

    if normalized_surface not in PUBLIC_READ_SURFACES:
        raise CapabilityPolicyError(
            f"Surface is not approved for public read: {normalized_surface!r}"
        )

    if normalized_method not in SAFE_READ_METHODS:
        raise CapabilityPolicyError(
            "Public-read capability cannot authorize mutation method "
            f"{normalized_method!r}"
        )

    return Capability.PUBLIC_READ


def _authority_state(authority: Mapping[str, Any] | None) -> str:
    if not isinstance(authority, Mapping):
        return ""
    return str(authority.get("state") or "").strip()


def _authority_authenticated(authority: Mapping[str, Any] | None) -> bool:
    return bool(
        isinstance(authority, Mapping)
        and authority.get("authenticated") is True
    )


def _require_minimum_tier(actual_tier: Any, minimum_tier: Any) -> None:
    # Tier is authorization context only. It is never identity.
    if actual_tier is None:
        raise CapabilityPolicyError("Authenticated privilege tier is required.")
    try:
        actual = int(actual_tier)
        required = int(minimum_tier)
    except (TypeError, ValueError) as exc:
        raise CapabilityPolicyError("Privilege tier is invalid.") from exc

    if actual < required:
        raise CapabilityPolicyError(
            f"Privilege tier {actual} does not satisfy required tier {required}."
        )


def assert_trusted_action(
    authority: Mapping[str, Any] | None,
    actual_tier: Any,
    *,
    minimum_tier: Any = 0,
) -> Capability:
    """Authorize ordinary authenticated contributor actions."""

    if not _authority_authenticated(authority):
        raise CapabilityPolicyError(
            "Trusted action requires authenticated authority."
        )

    state = _authority_state(authority)
    if state not in TRUSTED_ACTION_AUTHORITY_STATES:
        raise CapabilityPolicyError(
            f"Authority state {state!r} cannot perform trusted action."
        )

    _require_minimum_tier(actual_tier, minimum_tier)
    return Capability.TRUSTED_ACTION


def assert_fresh_proof_review(
    authority: Mapping[str, Any] | None,
    actual_tier: Any,
    *,
    minimum_tier: Any = 1,
) -> Capability:
    """Authorize high-risk review only with fresh proof plus reviewer tier."""

    if not _authority_authenticated(authority):
        raise CapabilityPolicyError(
            "Fresh-proof review requires authenticated authority."
        )

    state = _authority_state(authority)
    if state not in FRESH_PROOF_REVIEW_AUTHORITY_STATES:
        raise CapabilityPolicyError(
            f"Authority state {state!r} is not fresh proof."
        )

    if not isinstance(authority, Mapping) or authority.get("fresh_proof") is not True:
        raise CapabilityPolicyError(
            "Fresh-proof review requires current fresh proof."
        )

    _require_minimum_tier(actual_tier, minimum_tier)
    return Capability.FRESH_PROOF_REVIEW
