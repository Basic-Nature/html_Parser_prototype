"""W24AUTHOIDC1E — independently verify signed Keycloak *access* JWTs.

Unlike Authlib's verified ID-token UserInfo, an access token must not be
trusted through payload decoding or the presence of a ``userinfo`` dict.
Only RS256 and the pinned Keycloak issuer/audience/client are accepted.
No token or cryptographic key material is logged or stored.
"""
from __future__ import annotations

import re
from collections.abc import Mapping

from .electionpulse_client_roles import CLIENT_ID, ISSUER

JWKS_URL = ISSUER + "/protocol/openid-connect/certs"


def verify_keycloak_access_token(token: object, *, _test_signing_key=None) -> dict:
    """Verify a JWT signature and its identity/audience/lifetime before role mapping.

    ``_test_signing_key`` is an internal offline-test seam, never supplied from
    client input or runtime configuration. Production fetches pinned HTTPS JWKS.
    """
    import jwt
    from jwt import PyJWKClient

    if not isinstance(token, str) or not (32 <= len(token) <= 16384):
        raise ValueError("invalid_access_token_format")
    try:
        key = (_test_signing_key if _test_signing_key is not None else
               PyJWKClient(JWKS_URL, cache_jwk_set=True, lifespan=300, timeout=5)
               .get_signing_key_from_jwt(token).key)
        claims = jwt.decode(
            token, key=key, algorithms=["RS256"], issuer=ISSUER,
            audience=CLIENT_ID,
            options={"require": ["iss", "aud", "sub", "exp", "iat"]},
            leeway=30,
        )
    except Exception:
        # Keep token fragments, provider errors, and URLs out of response/logs.
        raise ValueError("unverified_access_token") from None
    if not isinstance(claims, dict) or claims.get("azp") != CLIENT_ID:
        raise ValueError("invalid_access_token_client")
    subject = claims.get("sub")
    if not isinstance(subject, str) or not re.fullmatch(r"[\x21-\x7e]{1,256}", subject):
        raise ValueError("invalid_access_token_subject")
    for key in ("exp", "iat"):
        if not isinstance(claims.get(key), int) or isinstance(claims[key], bool):
            raise ValueError("invalid_access_token_times")
    return claims
