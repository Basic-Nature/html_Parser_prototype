"""Provider-neutral authentication authority description.

This module intentionally does not perform authentication. It converts the
already-resolved ElectionPulse principal/source pair into stable vocabulary
that HTTP, Socket.IO, UI status, and future identity providers can share.

Current production authority:
- anonymous
- fresh ElectionPulse mTLS certificate
- bounded ElectionPulse certificate-backed session

Reserved future seam:
- Keycloak OIDC/federated identity

The current cert:<fingerprint> principal remains a compatibility identifier for
this tranche. A later migration can map certificate credentials to opaque
ElectionPulse device subjects without changing this authority vocabulary.
"""

from __future__ import annotations

AUTHORITY_CONTRACT_VERSION = "auth_authority_v1"

STATE_ANONYMOUS = "anonymous"
STATE_FRESH_CERTIFICATE = "fresh_certificate"
STATE_CERTIFICATE_SESSION = "certificate_session"
STATE_FEDERATED_IDENTITY = "federated_identity"
STATE_DEVELOPMENT_BYPASS = "development_bypass"
STATE_AUTHENTICATED_OTHER = "authenticated_other"

PROVIDER_ELECTIONPULSE_MTLS = "electionpulse_mtls"
PROVIDER_KEYCLOAK = "keycloak"
PROVIDER_LEGACY_SSO = "legacy_sso"
PROVIDER_DEVELOPMENT = "development"


def classify_authority(
    principal: str | None,
    principal_source: str | None,
) -> dict[str, object]:
    """Return a JSON-safe description of current application authority."""

    authenticated = bool(principal)

    certificate_principal = bool(
        isinstance(principal, str)
        and principal.startswith("cert:")
    )

    certificate_session = bool(
        certificate_principal
        and principal_source == "certificate_session"
    )

    fresh_certificate = bool(
        certificate_principal
        and not certificate_session
    )

    keycloak_identity = bool(
        principal_source in {
            "keycloak_oidc",
            "oidc:keycloak",
        }
        or (
            isinstance(principal, str)
            and principal.startswith("oidc:keycloak:")
        )
    )

    legacy_sso = bool(
        isinstance(principal, str)
        and principal.startswith("sso:")
    )

    development_bypass = bool(
        principal_source == "dev_bypass"
        or (
            isinstance(principal, str)
            and principal.startswith("dev:")
        )
    )

    if not authenticated:
        state = STATE_ANONYMOUS
        provider = None
        authentication_method = None
    elif fresh_certificate:
        state = STATE_FRESH_CERTIFICATE
        provider = PROVIDER_ELECTIONPULSE_MTLS
        authentication_method = "mtls"
    elif certificate_session:
        state = STATE_CERTIFICATE_SESSION
        provider = PROVIDER_ELECTIONPULSE_MTLS
        authentication_method = "mtls_session"
    elif keycloak_identity:
        state = STATE_FEDERATED_IDENTITY
        provider = PROVIDER_KEYCLOAK
        authentication_method = "oidc"
    elif legacy_sso:
        state = STATE_FEDERATED_IDENTITY
        provider = PROVIDER_LEGACY_SSO
        authentication_method = "sso"
    elif development_bypass:
        state = STATE_DEVELOPMENT_BYPASS
        provider = PROVIDER_DEVELOPMENT
        authentication_method = "dev_bypass"
    else:
        state = STATE_AUTHENTICATED_OTHER
        provider = None
        authentication_method = principal_source

    return {
        "contract_version": AUTHORITY_CONTRACT_VERSION,
        "state": state,
        "provider": provider,
        "authentication_method": authentication_method,
        "authenticated": authenticated,
        "fresh_proof": fresh_certificate,
        "certificate_present": fresh_certificate,
        "certificate_session_authenticated": certificate_session,
        "certificate_backed_authority": bool(
            fresh_certificate
            or certificate_session
        ),
    }
