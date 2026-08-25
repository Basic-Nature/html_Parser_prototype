from types import SimpleNamespace

import webapp.parser.auth.context as auth_context


CERT_PRINCIPAL = "cert:" + ("a" * 64)


def _configure(session, extractor, *, ttl=1800):
    request = SimpleNamespace(
        headers={},
        host="www.electionpulse.org",
        remote_addr="203.0.113.10",
    )

    auth_context.configure_runtime(
        ALLOW_DEV_NO_PRINCIPAL=False,
        CERT_SESSION_AUTH_TTL_SECONDS=ttl,
        _is_local_host=lambda _host: False,
        extract_client_principal=extractor,
        request=request,
        session=session,
    )


def test_current_certificate_establishes_bounded_session_authority(monkeypatch):
    browser_session = {}

    monkeypatch.setattr(auth_context.time, "time", lambda: 1000)

    _configure(
        browser_session,
        lambda _headers: (
            CERT_PRINCIPAL,
            "X-ARR-ClientCert",
            {"cn": "ElectionPulse Interactive Client"},
        ),
    )

    principal, source, metadata = auth_context.get_request_principal()

    assert principal == CERT_PRINCIPAL
    assert source == "X-ARR-ClientCert"
    assert metadata["cn"] == "ElectionPulse Interactive Client"
    assert browser_session["certificate_session_principal"] == CERT_PRINCIPAL
    assert browser_session["certificate_session_established_at"] == 1000


def test_later_request_uses_session_authority_without_claiming_current_cert(monkeypatch):
    browser_session = {
        "certificate_session_principal": CERT_PRINCIPAL,
        "certificate_session_established_at": 1000,
    }

    monkeypatch.setattr(auth_context.time, "time", lambda: 1100)

    _configure(
        browser_session,
        lambda _headers: (None, None, None),
    )

    principal, source, metadata = auth_context.get_request_principal()

    assert principal == CERT_PRINCIPAL
    assert source == "certificate_session"
    assert metadata is None


def test_expired_certificate_session_returns_to_anonymous(monkeypatch):
    browser_session = {
        "certificate_session_principal": CERT_PRINCIPAL,
        "certificate_session_established_at": 1000,
    }

    monkeypatch.setattr(auth_context.time, "time", lambda: 5000)

    _configure(
        browser_session,
        lambda _headers: (None, None, None),
        ttl=1800,
    )

    principal, source, metadata = auth_context.get_request_principal()

    assert principal is None
    assert source is None
    assert metadata is None
    assert "certificate_session_principal" not in browser_session
    assert "certificate_session_established_at" not in browser_session


def test_status_keeps_current_certificate_proof_distinct_from_session_authority():
    source = open(
        "webapp/parser/auth/status.py",
        "r",
        encoding="utf-8",
    ).read()

    assert '"certificate_proof_cached": False' in source
    assert '"certificate_session_authenticated"' in source
    assert '"certificate_backed_authority"' in source

    # Status must consume the centralized provider-neutral authority model
    # rather than duplicating certificate/session classification locally.
    assert "authority = classify_authority(" in source
    assert 'authority["certificate_present"]' in source
    assert 'authority["certificate_backed_authority"]' in source


def test_http_session_resolution_no_longer_requires_socket_sid():
    source = open(
        "webapp/parser/auth/context.py",
        "r",
        encoding="utf-8",
    ).read()

    assert "socket_sid = None" in source
    assert "def _bind_socket_if_available" in source

    forbidden = (
        "if not isinstance(socket_sid, str) or not socket_sid:\n"
        "        return None"
    )
    assert forbidden not in source

def test_authority_model_explicitly_distinguishes_fresh_cert_and_session():
    from webapp.parser.auth.authority_model import classify_authority

    fresh = classify_authority(
        CERT_PRINCIPAL,
        "X-ARR-ClientCert",
    )

    assert fresh["state"] == "fresh_certificate"
    assert fresh["provider"] == "electionpulse_mtls"
    assert fresh["authentication_method"] == "mtls"
    assert fresh["fresh_proof"] is True
    assert fresh["certificate_present"] is True
    assert fresh["certificate_session_authenticated"] is False

    bounded = classify_authority(
        CERT_PRINCIPAL,
        "certificate_session",
    )

    assert bounded["state"] == "certificate_session"
    assert bounded["provider"] == "electionpulse_mtls"
    assert bounded["authentication_method"] == "mtls_session"
    assert bounded["fresh_proof"] is False
    assert bounded["certificate_present"] is False
    assert bounded["certificate_session_authenticated"] is True
    assert bounded["certificate_backed_authority"] is True


def test_authority_model_reserves_keycloak_oidc_without_enabling_it():
    from webapp.parser.auth.authority_model import classify_authority

    authority = classify_authority(
        "oidc:keycloak:subject-placeholder",
        "keycloak_oidc",
    )

    assert authority["state"] == "federated_identity"
    assert authority["provider"] == "keycloak"
    assert authority["authentication_method"] == "oidc"
    assert authority["certificate_backed_authority"] is False


def test_auth_navigation_uses_explicit_certificate_backed_authority_contract():
    source = open(
        "webapp/Smart_Elections_Parser_Webapp.py",
        "r",
        encoding="utf-8",
    ).read()

    assert "authority = classify_authority(" in source
    assert '"certificate_backed_authority"' in source
    assert "Future high-risk step-up operations" in source


def test_auth_status_exposes_provider_neutral_authority_descriptor():
    source = open(
        "webapp/parser/auth/status.py",
        "r",
        encoding="utf-8",
    ).read()

    assert '"authority": authority' in source
    assert "classify_authority(" in source
