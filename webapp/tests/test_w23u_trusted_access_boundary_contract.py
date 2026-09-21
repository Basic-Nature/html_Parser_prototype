from __future__ import annotations

import hashlib
from types import SimpleNamespace
from urllib.parse import parse_qs, quote, urlparse

import pytest
from flask import Flask

from webapp.parser.routes.public_pages_blueprint import create_public_pages_blueprint

PUBLIC_BASE = "https://www.electionpulse.org"
BOUNDARY_BASE = "https://trusted-access.electionpulse.org"
SECRET = "w23u-test-secret-material-which-is-at-least-thirty-two-bytes"

@pytest.fixture()
def app():
    app = Flask(__name__)
    app.config.update(TESTING=True, SECRET_KEY="w23u-flask-test-secret")
    app.register_blueprint(create_public_pages_blueprint())
    return app

def _set_public_env(monkeypatch):
    monkeypatch.setenv("CERTIFICATE_AUTH_ENABLED", "true")
    monkeypatch.setenv("TRUSTED_ACCESS_BASE_URL", BOUNDARY_BASE)
    monkeypatch.setenv("TRUSTED_ACCESS_PUBLIC_BASE_URL", PUBLIC_BASE)
    monkeypatch.setenv("TRUSTED_ACCESS_HANDOFF_SECRET", SECRET)
    monkeypatch.delenv("TRUSTED_ACCESS_BOUNDARY_ENABLED", raising=False)

def _set_boundary_env(monkeypatch):
    monkeypatch.setenv("TRUSTED_ACCESS_BOUNDARY_ENABLED", "true")
    monkeypatch.setenv("TRUSTED_ACCESS_PUBLIC_BASE_URL", PUBLIC_BASE)
    monkeypatch.setenv("TRUSTED_ACCESS_HANDOFF_SECRET", SECRET)

def test_public_start_issues_browser_bound_state_without_raw_state_in_session(app, monkeypatch):
    _set_public_env(monkeypatch)
    client = app.test_client()
    response = client.get("/auth/certificate/start?next=/worklist", base_url=PUBLIC_BASE)
    assert response.status_code == 302
    parsed = urlparse(response.headers["Location"])
    assert parsed.netloc == "trusted-access.electionpulse.org"
    assert parsed.path == "/auth/certificate/verify"
    query = parse_qs(parsed.query)
    assert query["return_to"] == ["/worklist"]
    state = query["state"][0]
    with client.session_transaction(base_url=PUBLIC_BASE) as browser_session:
        assert browser_session["trusted_access_state_sha256"] == hashlib.sha256(state.encode()).hexdigest()
        assert state not in set(map(str, browser_session.values()))
    assert "no-store" in response.headers["Cache-Control"].lower()

def test_boundary_verify_emits_opaque_handoff(app, monkeypatch):
    from webapp.parser.auth import trusted_access_boundary as boundary
    _set_boundary_env(monkeypatch)
    client = app.test_client()
    principal = "cert:" + ("a" * 64)
    monkeypatch.setattr(
        boundary,
        "_resolve_fresh_durable_certificate",
        lambda _headers: SimpleNamespace(
            resolved=True,
            protected_operation_eligible=True,
            compatibility_principal=principal,
        ),
    )
    state = "s" * 43
    response = client.get(
        f"/auth/certificate/verify?return_to=/worklist&state={state}",
        base_url=BOUNDARY_BASE,
    )
    assert response.status_code == 302
    parsed = urlparse(response.headers["Location"])
    assert parsed.netloc == "www.electionpulse.org"
    assert parsed.path == "/auth/certificate/complete"
    token = parse_qs(parsed.query)["handoff"][0]
    assert principal not in token
    assert ("a" * 64) not in token

def test_boundary_verify_rejects_unresolved_certificate(app, monkeypatch):
    from webapp.parser.auth import trusted_access_boundary as boundary
    _set_boundary_env(monkeypatch)
    client = app.test_client()
    monkeypatch.setattr(
        boundary,
        "_resolve_fresh_durable_certificate",
        lambda _headers: SimpleNamespace(
            resolved=False,
            protected_operation_eligible=False,
            compatibility_principal=None,
        ),
    )
    response = client.get(
        f"/auth/certificate/verify?return_to=/worklist&state={'s' * 43}",
        base_url=BOUNDARY_BASE,
    )
    assert response.status_code == 403

def test_public_complete_redeems_once_into_existing_certificate_session(app, monkeypatch):
    from webapp.parser.auth import trusted_access_boundary as boundary
    _set_public_env(monkeypatch)
    client = app.test_client()
    start = client.get("/auth/certificate/start?next=/worklist", base_url=PUBLIC_BASE)
    state = parse_qs(urlparse(start.headers["Location"]).query)["state"][0]
    principal = "cert:" + ("b" * 64)
    token = boundary._issue_handoff(principal, state=state, return_target="/worklist")
    complete = (
        "/auth/certificate/complete?handoff="
        + quote(token, safe="")
        + "&state="
        + quote(state, safe="")
    )
    response = client.get(complete, base_url=PUBLIC_BASE)
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/worklist")
    with client.session_transaction(base_url=PUBLIC_BASE) as browser_session:
        assert browser_session["certificate_session_principal"] == principal
        assert isinstance(browser_session["certificate_session_established_at"], int)
        assert "trusted_access_state_sha256" not in browser_session
    replay = client.get(complete, base_url=PUBLIC_BASE)
    assert replay.status_code == 400
    assert replay.get_json()["error"] == "trusted_access_state_invalid"

def test_complete_fails_closed_on_wrong_public_host(app, monkeypatch):
    from webapp.parser.auth import trusted_access_boundary as boundary
    _set_public_env(monkeypatch)
    client = app.test_client()
    state = "x" * 43
    with client.session_transaction(base_url=PUBLIC_BASE) as browser_session:
        browser_session["trusted_access_state_sha256"] = hashlib.sha256(state.encode()).hexdigest()
        browser_session["trusted_access_state_issued_at"] = 1000
    monkeypatch.setattr("webapp.parser.auth.trusted_access.time.time", lambda: 1000)
    token = boundary._issue_handoff("cert:" + ("c" * 64), state=state, return_target="/worklist")
    response = client.get(
        "/auth/certificate/complete?handoff=" + quote(token, safe="") + "&state=" + state,
        base_url="https://evil.example",
    )
    assert response.status_code == 403

def test_projected_boundary_avoids_legacy_pins_and_handoff_table():
    source = open("webapp/parser/auth/trusted_access_boundary.py", encoding="utf-8").read()
    assert "trusted_access_handoffs" not in source
    assert "TRUSTED_CLIENT_CERT_FINGERPRINTS" not in source
    assert "ROOT_ADMIN_CERT_FINGERPRINTS" not in source
    assert "resolve_enrolled_mtls_principal" in source
    assert "SET TRANSACTION READ ONLY" in source

def test_projected_certificate_routes_are_get_only():
    source = open("webapp/parser/routes/public_pages_blueprint.py", encoding="utf-8").read()
    assert '"/auth/certificate/verify"' in source
    assert '"/auth/certificate/complete"' in source
    cert_section = source[source.index('"/auth/certificate/start"'):source.index('"/ocr_diagnostics"')]
    assert 'methods=["POST"]' not in cert_section
