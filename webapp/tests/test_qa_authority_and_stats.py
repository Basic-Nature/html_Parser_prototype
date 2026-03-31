from __future__ import annotations

from flask import Flask

from webapp.parser.quality_assurance import qa_endpoints as qae


def _build_app() -> Flask:
    app = Flask(__name__)
    app.register_blueprint(qae.qa_bp)
    return app


def test_verify_and_promote_requires_admin_reviewer_tier(monkeypatch):
    app = _build_app()

    monkeypatch.setattr(qae, "QA_REQUIRE_CERT_AUTH", True)
    monkeypatch.setattr(
        qae,
        "extract_client_principal",
        lambda _headers: ("sso:user@example.org", "sso_oid", None),
    )
    monkeypatch.setattr(qae, "promote_to_dl2", lambda **_kwargs: True)

    with app.test_client() as client:
        resp = client.post(
            "/api/data-assurance/verify-and-promote",
            json={
                "dataset_id": "dataset-1",
                "certification_reason": "reviewed",
            },
        )

    assert resp.status_code == 403
    payload = resp.get_json()
    assert payload["error"] == "Forbidden"
    assert payload["required_tier"] == "ADMIN_REVIEWER"
    assert payload["actual_tier"] == "STANDARD_USER"


def test_verify_and_promote_allows_admin_reviewer_tier(monkeypatch):
    app = _build_app()

    monkeypatch.setattr(qae, "QA_REQUIRE_CERT_AUTH", True)
    monkeypatch.setenv("ADMIN_REVIEWER_PRINCIPALS", "reviewer@example.org")
    monkeypatch.setattr(
        qae,
        "extract_client_principal",
        lambda _headers: ("sso:reviewer@example.org", "sso_oid", None),
    )
    monkeypatch.setattr(qae, "promote_to_dl2", lambda **_kwargs: True)

    with app.test_client() as client:
        resp = client.post(
            "/api/data-assurance/verify-and-promote",
            json={
                "dataset_id": "dataset-2",
                "certification_reason": "reviewed",
            },
        )

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["success"] is True
    assert payload["verified_by"] == "sso:reviewer@example.org"


def test_stats_reports_rejected_count(monkeypatch):
    app = _build_app()

    monkeypatch.setattr(qae, "QA_REQUIRE_CERT_AUTH", False)
    monkeypatch.setattr(qae, "extract_client_principal", lambda _headers: (None, None, None))

    monkeypatch.setattr(qae, "get_pending_dl2_reviews", lambda: [{"id": 1}, {"id": 2}])
    monkeypatch.setattr(
        qae,
        "get_dl2_inventory",
        lambda **_kwargs: [
            {"trust_score": 80.0, "extraction_confidence": 0.9},
            {"trust_score": 70.0, "extraction_confidence": 0.8},
        ],
    )
    monkeypatch.setattr(qae, "get_rejected_count", lambda: 7)

    with app.test_client() as client:
        resp = client.get("/api/data-assurance/stats")

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["dl1_pending_count"] == 2
    assert payload["dl2_verified_count"] == 2
    assert payload["rejected_count"] == 7
    assert payload["avg_trust_score"] == 75.0
    assert payload["avg_extraction_confidence"] == 0.85
