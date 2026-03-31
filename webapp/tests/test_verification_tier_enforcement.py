from __future__ import annotations

from flask import Flask, jsonify

from webapp.parser import verification_endpoints as ve


def _build_app() -> Flask:
    app = Flask(__name__)

    @app.route("/tier/admin-reviewer")
    @ve._require_verifier_tier("admin_reviewer")
    def _admin_reviewer_route():
        return jsonify({"ok": True})

    return app


def test_verifier_tier_requires_authenticated_principal(monkeypatch):
    app = _build_app()

    def _mock_extract_client_principal(_headers):
        return None, ""

    monkeypatch.setattr(
        "webapp.parser.utils.cert_utils.extract_client_principal",
        _mock_extract_client_principal,
    )

    with app.test_client() as client:
        resp = client.get("/tier/admin-reviewer")

    assert resp.status_code == 401
    payload = resp.get_json()
    assert payload["error"] == "Unauthorized"


def test_verifier_tier_blocks_insufficient_tier(monkeypatch):
    app = _build_app()

    def _mock_extract_client_principal(_headers):
        return "sso:reviewer@example.org", "sso_oid"

    monkeypatch.setattr(
        "webapp.parser.utils.cert_utils.extract_client_principal",
        _mock_extract_client_principal,
    )

    with app.test_client() as client:
        resp = client.get("/tier/admin-reviewer")

    assert resp.status_code == 403
    payload = resp.get_json()
    assert payload["error"] == "Forbidden"
    assert payload["required_tier"] == "ADMIN_REVIEWER"
    assert payload["actual_tier"] == "STANDARD_USER"


def test_verifier_tier_allows_sufficient_tier(monkeypatch):
    app = _build_app()

    monkeypatch.setenv("ADMIN_FULL_TRUST_PRINCIPALS", "admin@example.org")

    def _mock_extract_client_principal(_headers):
        return "sso:admin@example.org", "sso_oid"

    monkeypatch.setattr(
        "webapp.parser.utils.cert_utils.extract_client_principal",
        _mock_extract_client_principal,
    )

    with app.test_client() as client:
        resp = client.get("/tier/admin-reviewer")

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["ok"] is True


def test_unknown_required_tier_defaults_to_root_admin(monkeypatch):
    app = Flask(__name__)

    @app.route("/tier/unknown")
    @ve._require_verifier_tier("unknown-tier")
    def _unknown_tier_route():
        return jsonify({"ok": True})

    def _mock_extract_client_principal(_headers):
        return "sso:admin@example.org", "sso_oid"

    monkeypatch.setattr(
        "webapp.parser.utils.cert_utils.extract_client_principal",
        _mock_extract_client_principal,
    )
    monkeypatch.setenv("ADMIN_FULL_TRUST_PRINCIPALS", "admin@example.org")

    with app.test_client() as client:
        resp = client.get("/tier/unknown")

    assert resp.status_code == 403
    payload = resp.get_json()
    assert payload["required_tier"] == "ROOT_ADMIN"
    assert payload["actual_tier"] == "ADMIN_FULL_TRUST"
