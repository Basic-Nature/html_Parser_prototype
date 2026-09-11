from __future__ import annotations

from typing import Any, Mapping

from webapp.parser.auth.authority_model import classify_authority
from webapp.parser.auth.capability_policy import (
    CapabilityPolicyError,
    assert_trusted_action,
)
from webapp.parser.utils.privilege_tiers import get_principal_tier

DATA_FRAMEWORK_OPERATOR_ACCESS_CONTRACT = "data_framework_operator_access_v1"


def project_data_framework_operator_access(
    authority: Mapping[str, Any] | None,
    actual_tier: Any,
) -> dict[str, object]:
    """Return the identity-free Data Framework mutation-control projection."""

    authenticated = bool(
        isinstance(authority, Mapping)
        and authority.get("authenticated") is True
    )
    can_upload_input = False
    if authenticated:
        try:
            assert_trusted_action(authority, actual_tier, minimum_tier=0)
            can_upload_input = True
        except (CapabilityPolicyError, TypeError, ValueError):
            can_upload_input = False

    return {
        "contract": DATA_FRAMEWORK_OPERATOR_ACCESS_CONTRACT,
        "authenticated": authenticated,
        "can_upload_input": can_upload_input,
        "mutation_controls_disclosed": can_upload_input,
        "principal_disclosed": False,
    }


def resolve_data_framework_operator_access(
    principal: str | None,
    principal_source: str | None,
) -> dict[str, object]:
    """Resolve request authority into the safe Data Framework UI projection."""

    authority = classify_authority(principal, principal_source)
    actual_tier = None
    if authority.get("authenticated") is True:
        try:
            actual_tier = get_principal_tier(principal, principal_source or "")
        except Exception:
            actual_tier = None

    return project_data_framework_operator_access(authority, actual_tier)
