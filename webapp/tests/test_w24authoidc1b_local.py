"""W24AUTHOIDC1B local Flask integration. All OIDC tokens here are TEST DOUBLES.

These Flask tests use explicitly mocked cryptographic verification. Real RSA
negative tests are in test_w24authoidc1e_cryptographic_roles.py. Browser OIDC
state, nonce and PKCE validation remains pending live provider qualification.
"""
from __future__ import annotations

import os
import time
import pytest
from flask import Flask, redirect
from webapp.parser.auth import context as ctx
from webapp.parser.auth import keycloak_oidc as oidc
from webapp.parser.auth.authority_model import classify_authority
from webapp.parser.auth.capability_policy import assert_trusted_action, CapabilityPolicyError
from webapp.parser.routes.public_pages_blueprint import create_public_pages_blueprint
from webapp.parser.routes.election_projects_blueprint import create_election_projects_blueprint
from webapp.parser.services import election_projects as projects_service


class FakeDb:
    def rollback(self): pass
    def close(self): pass


class FakeRemote:
    roles = ["electionpulse-workflow-contributor"]
    signed_claims = True

    def authorize_redirect(self, redirect_uri):
        assert redirect_uri == "http://localhost:5000/auth/oidc/callback"
        return redirect("https://auth.smartelections.cloud/realms/smartelections/protocol/openid-connect/auth")

    def authorize_access_token(self):
        if not self.signed_claims:
            return {"access_token": "UNTRUSTED_TEST_ONLY"}
        now = int(time.time())
        return {
            "access_token": "SYNTHETIC_ACCESS_JWT_TEST_SEAM_ONLY",
            "id_token": "TEST_ONLY__CRYPTOGRAPHICALLY_UNVERIFIED__NEVER_PRODUCTION",
            "userinfo": {"iss": oidc.ISSUER, "aud": "electionpulse-web",
                         "sub": "pseudonymous-test-user", "iat": now, "exp": now + 1800,
                         "nonce": "simulated-nonce"},
        }


@pytest.fixture
def integration(monkeypatch):
    monkeypatch.setenv("ELECTIONPULSE_KEYCLOAK_OIDC_ENABLED", "true")
    monkeypatch.setenv("DEPLOY_ENV", "local")
    monkeypatch.setenv("ELECTIONPULSE_KEYCLOAK_CLIENT_ID", "electionpulse-web")
    monkeypatch.setenv("ELECTIONPULSE_KEYCLOAK_CLIENT_SECRET", "dummy-secret-for-test-only")
    monkeypatch.setenv("ELECTIONPULSE_KEYCLOAK_REDIRECT_URI", "http://localhost:5000/auth/oidc/callback")
    monkeypatch.setenv("ELECTIONPULSE_OIDC_SUBJECT_HMAC_KEY", "t" * 48)
    monkeypatch.setenv("ELECTIONPULSE_OIDC_SESSION_BACKEND", "memory_dev")
    monkeypatch.setenv("ELECTIONPULSE_OIDC_ALLOW_DEV_MEMORY_SESSIONS", "true")
    for key in ("WEBSITE_SITE_NAME", "WEBSITE_INSTANCE_ID", "WEBSITE_HOSTNAME",
                "CONTAINER_APP_NAME", "K_SERVICE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ELECTION_PROJECTS_ENABLED", "true")
    monkeypatch.setenv("ELECTION_PROJECT_OWNER_HMAC_KEY", "w" * 48)
    app = Flask(__name__, template_folder="../templates")
    app.secret_key = "offline-test-flask-secret-key-allow-crypto-only-in-test"
    app.config["TESTING"] = True
    remote = FakeRemote()
    monkeypatch.setattr(oidc, "_remote", lambda: remote)
    def fake_verified_token(token):
        assert token == "SYNTHETIC_ACCESS_JWT_TEST_SEAM_ONLY"
        now = int(time.time())
        return {"iss": oidc.ISSUER, "aud": ["electionpulse-web"], "azp": "electionpulse-web",
                "sub": "pseudonymous-test-user", "iat": now, "exp": now+1800,
                "resource_access": {"electionpulse-web": {"roles": list(remote.roles)}}}
    monkeypatch.setattr(oidc, "verify_keycloak_access_token", fake_verified_token)
    monkeypatch.setattr(projects_service, "projects", lambda db, owner: [])
    def principal_resolver():
        ctx.configure_runtime(
            ALLOW_DEV_NO_PRINCIPAL=False,
            CERT_SESSION_AUTH_TTL_SECONDS=600,
            _is_local_host=lambda host: True,
            extract_client_principal=lambda headers: (None, None, None),
            request=__import__("flask").request,
            session=__import__("flask").session,
        )
        return ctx.get_request_principal()
    app.register_blueprint(create_public_pages_blueprint())
    app.register_blueprint(create_election_projects_blueprint(
        principal_resolver=principal_resolver,
        session_factory=lambda: FakeDb(),
    ))
    return app, remote


def test_anonymous_project_denied_and_status(integration):
    app, remote = integration
    with app.test_client() as client:
        assert client.get("/auth/oidc/status", base_url="http://localhost:5000").json["authenticated"] is False
        assert client.get("/api/projects/v1", base_url="http://localhost:5000").status_code == 401
        assert client.get("/projects", base_url="http://localhost:5000").status_code == 200


def test_local_login_and_analyst_project_access_without_certificate_elevation(integration):
    app, remote = integration
    with app.test_client() as client:
        start = client.get("/auth/oidc/login?next=/projects", base_url="http://localhost:5000")
        assert start.status_code == 302 and start.location.startswith("https://auth.smartelections.cloud/")
        result = client.get("/auth/oidc/callback?state=synthetic", base_url="http://localhost:5000")
        assert result.status_code == 303 and result.location == "/projects"
        status = client.get("/auth/oidc/status", base_url="http://localhost:5000").json
        assert status["authenticated"] is True and status["project_access"] is True
        assert status["parser_upload_authorized"] is False and status["trusted_certificate"] is False
        assert client.get("/api/projects/v1", base_url="http://localhost:5000").status_code == 200
        authority = classify_authority("oidc:keycloak:" + "a" * 64, "keycloak_oidc")
        with pytest.raises(CapabilityPolicyError):
            assert_trusted_action(authority, 100)


def test_no_role_still_authenticates_but_denies_projects(integration):
    app, remote = integration
    remote.roles = []
    with app.test_client() as client:
        assert client.get("/auth/oidc/login", base_url="http://localhost:5000").status_code == 302
        assert client.get("/auth/oidc/callback?state=synthetic", base_url="http://localhost:5000").status_code == 303
        assert client.get("/auth/oidc/status", base_url="http://localhost:5000").json["authenticated"] is True
        assert client.get("/api/projects/v1", base_url="http://localhost:5000").status_code == 403


def test_unverified_callback_fails_closed(integration):
    app, remote = integration
    remote.signed_claims = False
    with app.test_client() as client:
        assert client.get("/auth/oidc/login", base_url="http://localhost:5000").status_code == 302
        assert client.get("/auth/oidc/callback?state=synthetic", base_url="http://localhost:5000").status_code == 401
        assert client.get("/auth/oidc/status", base_url="http://localhost:5000").json["authenticated"] is False


def test_host_and_default_off_fail_closed(integration, monkeypatch):
    app, remote = integration
    with app.test_client() as client:
        assert client.get("/auth/oidc/login", base_url="http://example.com").status_code == 404
        monkeypatch.setenv("ELECTIONPULSE_KEYCLOAK_OIDC_ENABLED", "false")
        assert client.get("/auth/oidc/login", base_url="http://localhost:5000").status_code == 404


def test_mismatch_token_fields_are_rejected(integration):
    now = int(time.time())
    base = {"iss":oidc.ISSUER,"sub":"test","aud":"electionpulse-web",
            "iat":now,"exp":now+1000,"nonce":"test"}
    for mutated in [dict(base, iss="https://evil.invalid"),dict(base,aud="doors"),
                    dict(base,exp=now-1),dict(base,nonce=None),dict(base,sub="")]:
        with pytest.raises(ValueError):
            oidc._validated_id_token(mutated,now)


def test_open_redirect_is_denied(integration):
    assert oidc._safe_next("https://evil.invalid") == "/projects"
    assert oidc._safe_next("//evil.invalid") == "/projects"
    assert oidc._safe_next("/projects") == "/projects"


def test_dev_backend_requires_explicit_opt_in(integration, monkeypatch):
    app, _ = integration
    monkeypatch.delenv("ELECTIONPULSE_OIDC_ALLOW_DEV_MEMORY_SESSIONS")
    with app.test_client() as client:
        assert client.get("/auth/oidc/login", base_url="http://localhost:5000").status_code == 503


def test_dev_backend_rejects_azure_even_with_local_host(integration, monkeypatch):
    app, _ = integration
    monkeypatch.setenv("WEBSITE_SITE_NAME", "not-a-real-azure-site")
    with app.test_client() as client:
        assert client.get("/auth/oidc/login", base_url="http://localhost:5000").status_code == 503


def test_dev_backend_rejects_production_mode(integration, monkeypatch):
    app, _ = integration
    monkeypatch.setenv("DEPLOY_ENV", "production")
    with app.test_client() as client:
        assert client.get("/auth/oidc/login", base_url="http://localhost:5000").status_code == 503


def test_nonlocal_host_refuses_memory_backend(integration, monkeypatch):
    app, _ = integration
    monkeypatch.setenv("ELECTIONPULSE_KEYCLOAK_REDIRECT_URI", "https://www.electionpulse.org/auth/oidc/callback")
    app.config.update(SESSION_COOKIE_SECURE=True, SESSION_COOKIE_HTTPONLY=True,
                      SESSION_COOKIE_SAMESITE="Lax")
    with app.test_client() as client:
        assert client.get("/auth/oidc/login", base_url="https://www.electionpulse.org").status_code == 503


def test_loss_of_process_memory_revokes_session(integration):
    app, _ = integration
    with app.test_client() as client:
        assert client.get("/auth/oidc/login", base_url="http://localhost:5000").status_code == 302
        assert client.get("/auth/oidc/callback?state=synthetic", base_url="http://localhost:5000").status_code == 303
        assert client.get("/auth/oidc/status", base_url="http://localhost:5000").json["project_access"] is True
        app.extensions.pop("electionpulse_oidc_dev_memory")
        assert client.get("/auth/oidc/status", base_url="http://localhost:5000").json["authenticated"] is False
        assert client.get("/api/projects/v1", base_url="http://localhost:5000").status_code == 401
