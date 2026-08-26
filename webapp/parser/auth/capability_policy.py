from __future__ import annotations

from enum import Enum


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
        "workflow_v1_facets",
        "workflow_v1_stats",
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
