"""Authoritative authentication status/read-model projection.

This module owns safe request-scoped authentication status presentation.
Identity resolution remains in auth.context. Certificate enforcement remains
in auth.policy. Privilege hierarchy ownership remains in privilege_tiers.py.

Runtime dependencies are rebound by compatibility wrappers in
Smart_Elections_Parser_Webapp.py so existing request behavior, route-handler
names, and monkeypatch seams remain intact during Tranche 1.
"""

from __future__ import annotations

from contextvars import ContextVar
from datetime import datetime, timezone


_RUNTIME_BINDINGS: ContextVar[dict[str, object]] = ContextVar(
    "electionpulse_authority_status_runtime",
    default={},
)


def configure_runtime(**bindings: object) -> None:
    """Bind current composition-root dependencies for this call context."""
    _RUNTIME_BINDINGS.set(dict(bindings))


def _runtime_binding(name: str):
    bindings = _RUNTIME_BINDINGS.get()
    if name not in bindings:
        raise RuntimeError(f"Authority status runtime binding missing: {name}")
    return bindings[name]


def _sanitize_cert_metadata_for_status(cert_metadata: dict | None) -> dict:
    if not isinstance(cert_metadata, dict):
        return {}

    allowed = {
        "cn",
        "issuer",
        "serial_number",
        "issued_date",
        "expiry_date",
        "expiry_days",
        "key_algorithm",
        "is_expired",
        "trust_required",
        "trust_valid",
        "trust_reason",
    }

    return {
        key: cert_metadata[key]
        for key in allowed
        if key in cert_metadata
    }


def api_auth_status():
    AZURE_CLIENT_CERT_MODE = _runtime_binding("AZURE_CLIENT_CERT_MODE")
    CERT_ENFORCEMENT_MODE = _runtime_binding("CERT_ENFORCEMENT_MODE")
    DEPLOY_ENV = _runtime_binding("DEPLOY_ENV")
    REQUIRE_CERT_FOR_MUTATIONS = _runtime_binding("REQUIRE_CERT_FOR_MUTATIONS")
    _auth_mode_requires_certificate = _runtime_binding("_auth_mode_requires_certificate")
    _is_local_request = _runtime_binding("_is_local_request")
    _sanitize_cert_metadata_for_status = _runtime_binding("_sanitize_cert_metadata_for_status")
    get_request_principal = _runtime_binding("get_request_principal")
    jsonify = _runtime_binding("jsonify")
    request = _runtime_binding("request")
    resolve_session_id = _runtime_binding("resolve_session_id")
    sanitize_internal_next = _runtime_binding("sanitize_internal_next")
    url_for = _runtime_binding("url_for")

    # Authoritative request-scoped authentication/certificate state.
    #
    # Session state is included only for UX continuity. Certificate presence
    # is derived solely from the current request principal.

    from webapp.parser.utils.cert_utils import (
        observe_client_certificate_transport,
    )

    certificate_transport = observe_client_certificate_transport(
        request.headers,
    )

    principal, principal_source, cert_metadata = (
        get_request_principal()
    )

    certificate_present = bool(
        principal
        and principal.startswith(
            "cert:"
        )
    )

    authenticated = bool(
        principal
    )

    next_target = sanitize_internal_next(
        request.args.get("next"),
        fallback=url_for(
            "ballot_lens"
        ),
    )

    try:
        session_id = resolve_session_id(
            {},
            create_if_missing=False,
        )
    except Exception:
        session_id = None

    try:
        from webapp.parser.utils.privilege_tiers import (
            get_principal_tier,
        )

        tier = get_principal_tier(
            principal,
            principal_source,
        )
    except Exception:
        tier = None

    if tier is not None:
        raw_tier_name = (
            getattr(
                tier,
                "name",
                None,
            )
            or getattr(
                tier,
                "value",
                None,
            )
            or "STANDARD_USER"
        )

        if isinstance(
            raw_tier_name,
            str,
        ):
            privilege_tier = (
                raw_tier_name
            )
        else:
            privilege_tier = {
                0: "STANDARD_USER",
                1: "ADMIN_REVIEWER",
                2: "ADMIN_FULL_TRUST",
                3: "ROOT_ADMIN",
            }.get(
                int(raw_tier_name),
                "STANDARD_USER",
            )

        try:
            privilege_level = (
                int(tier)
            )
        except Exception:
            privilege_level = {
                "STANDARD_USER": 0,
                "ADMIN_REVIEWER": 1,
                "ADMIN_FULL_TRUST": 2,
                "ROOT_ADMIN": 3,
            }.get(
                privilege_tier,
                0,
            )

        privilege_display = (
            getattr(
                tier,
                "name_display",
                None,
            )
            or privilege_tier
            .replace("_", " ")
            .title()
        )

    else:
        privilege_tier = (
            "STANDARD_USER"
        )

        privilege_level = 0

        privilege_display = (
            "Standard User"
        )

    local_certificate_bypass = bool(
        DEPLOY_ENV == "local"
        and _is_local_request()
    )

    certificate_required_for_mutations = bool(
        _auth_mode_requires_certificate()
        and REQUIRE_CERT_FOR_MUTATIONS
        and not local_certificate_bypass
    )

    response = {
        "authenticated": authenticated,

        "certificate_present": (
            certificate_present
        ),

        "certificate_transport": (
            certificate_transport
        ),


        "certificate_required_for_mutations": (
            certificate_required_for_mutations
        ),

        "certificate_action_required": bool(
            certificate_required_for_mutations
            and not certificate_present
        ),

        "principal": principal,

        "principal_source": (
            principal_source
        ),

        "cert_metadata": (
            _sanitize_cert_metadata_for_status(
                cert_metadata
            )
        ),

        "privilege_tier": (
            privilege_tier
        ),

        "privilege_level": (
            privilege_level
        ),

        "privilege_display": (
            privilege_display
        ),

        "certificate_policy": (
            CERT_ENFORCEMENT_MODE
        ),

        "azure_client_cert_mode": (
            AZURE_CLIENT_CERT_MODE
        ),

        "challenge_url": url_for(
            "auth_challenge",
            next=next_target,
        ),

        "auth_url": url_for(
            "auth_welcome",
            next=next_target,
        ),

        "status_source": (
            "current_request_principal"
        ),

        "session_context": {
            "session_id": session_id,

            "host": (
                request.host
                or "unknown"
            ),

            "remote_addr": (
                request.remote_addr
                or "unknown"
            ),

            # Explicit contract: session/cache state is never certificate proof.
            "certificate_proof_cached": False,
        },

        "timestamp": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
    }

    return jsonify(
        response
    )


def api_auth_certificate_info():
    """
    Legacy compatibility alias for auth status.
    This endpoint now behaves as a safe inspection endpoint.
    """
    api_auth_status = _runtime_binding("api_auth_status")
    return api_auth_status()
