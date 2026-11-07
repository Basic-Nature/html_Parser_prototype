"""W24AUTHOIDC1D candidate pure authorization adapter.

Input MUST be validated claims from a cryptographically verified, issuer-bound,
audience-validated access token. Never feed decoded or client-supplied JSON here.
No direct parsing, workflow, reviewer, publication or certificate authorization.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Any

CLIENT_ID = "electionpulse-web"
ISSUER = "https://auth.smartelections.cloud/realms/smartelections"
ROLE_MAP = {
    "electionpulse-workflow-contributor": "workflow_contributor",
    "electionpulse-workflow-reviewer": "workflow_reviewer",
    "electionpulse-workflow-publication-operator": "workflow_publication_operator",
    "electionpulse-workflow-auditor": "workflow_auditor",
}


@dataclass(frozen=True)
class RoleDecision:
    authenticated: bool
    project_access: bool
    internal_roles: tuple[str, ...]
    certificate_trusted: bool = False
    reviewer_operations_authorized: bool = False
    publication_operations_authorized: bool = False
    parser_upload_authorized: bool = False
    arbitrary_url_execution_authorized: bool = False


def roles_from_verified_access_token_claims(
    claims: Mapping[str, Any], *, client_id: str = CLIENT_ID,
    issuer: str = ISSUER, audience: str = CLIENT_ID,
) -> RoleDecision:
    """Map separately-validated Keycloak access-token claims into *inert* roles.

    Important: this function is NOT a JWT verifier. Its caller MUST first
    independently verify JWT cryptographic signature/JWKS algorithm allowlist,
    iss, exp/nbf/iat, aud, azp, accepted client and relevant token binding.
    The values checked here are defense-in-depth and must not replace that step.
    """
    denied = RoleDecision(False, False, ())
    if not isinstance(claims, Mapping):
        return denied
    if (not isinstance(client_id, str) or not client_id or
        not isinstance(audience, str) or not audience or
        claims.get("iss") != issuer):
        return denied
    aud = claims.get("aud")
    audiences = aud if isinstance(aud, list) else [aud]
    if not audiences or any(not isinstance(a, str) for a in audiences):
        return denied
    if audience not in audiences:
        return denied
    azp = claims.get("azp")
    if azp != client_id:
        return denied
    sub = claims.get("sub")
    if not isinstance(sub, str) or not (1 <= len(sub) <= 256):
        return denied
    access = claims.get("resource_access")
    if not isinstance(access, Mapping):
        return RoleDecision(True, False, ())
    own = access.get(client_id)
    if not isinstance(own, Mapping):
        return RoleDecision(True, False, ())
    roles = own.get("roles")
    if not isinstance(roles, list) or any(not isinstance(x, str) for x in roles):
        return RoleDecision(True, False, ())
    internal = tuple(sorted({ROLE_MAP[r] for r in roles if r in ROLE_MAP}))
    # Contributor alone initially grants private Project ownership. Reviewer,
    # publication and auditor names are recorded only: NO elevated capabilities.
    return RoleDecision(True, "workflow_contributor" in internal, internal)
