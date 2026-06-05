from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

try:
    import webapp.Smart_Elections_Parser_Webapp as appmod
except ImportError as exc:  # pragma: no cover
    pytest.skip(f"Cannot import webapp app: {exc}", allow_module_level=True)


@pytest.fixture
def client():
    appmod.app.config["TESTING"] = True
    with appmod.app.test_client() as test_client:
        yield test_client


def test_cert_required_response_redirect_contract():
    with appmod.app.test_request_context("/upload/input", headers={"Accept": "text/html"}):
        resp = appmod._cert_required_response("upload_input")

    assert resp.status_code in (301, 302)
    location = resp.headers.get("Location", "")
    assert "/auth/welcome" in location
    assert "next=" in location


def test_cert_required_response_json_contract():
    with appmod.app.test_request_context("/upload/input", headers={"Accept": "application/json"}):
        resp, status = appmod._cert_required_response("upload_input")

    payload = resp.get_json()
    assert status == 401
    assert payload["error"] == "certificate_required"
    assert payload["reason"] == "upload_input"
    assert "/auth/welcome" in payload["auth_url"]


def test_api_auth_certificate_info_requires_cert_contract(client, monkeypatch):
    monkeypatch.setattr(appmod, "REQUIRE_CERT_FOR_MUTATIONS", True)
    monkeypatch.setattr(appmod, "DEPLOY_ENV", "ci")
    monkeypatch.setattr(appmod, "get_request_principal", lambda: (None, None, None))

    resp = client.get("/api/auth/certificate_info", headers={"Accept": "application/json"})
    payload = resp.get_json()

    assert resp.status_code == 401
    assert payload["error"] == "certificate_required"
    assert payload["reason"] == "certificate_info"
    assert isinstance(payload.get("auth_url"), str) and "/auth/welcome" in payload["auth_url"]

def test_auth_welcome_sets_csp_nonce_header(client, monkeypatch):
    monkeypatch.setattr(appmod, "get_request_principal", lambda: (None, None, None))

    resp = client.get("/auth/welcome", headers={"Accept": "text/html"})

    assert resp.status_code == 401
    csp_header = resp.headers.get("Content-Security-Policy")
    assert csp_header is not None
    assert "script-src" in csp_header
    assert "nonce-" in csp_header
    assert b"script nonce=\"" in resp.data



def test_api_auth_certificate_info_success_contract(client, monkeypatch):
    monkeypatch.setattr(appmod, "_require_client_cert", lambda _reason: None)
    monkeypatch.setattr(
        appmod,
        "get_request_principal",
        lambda: (
            "cert:test-user",
            "x_arr_clientcert",
            {"cn": "Test User", "issuer": "Test CA", "serial_number": "abc123"},
        ),
    )

    with patch(
        "webapp.parser.utils.privilege_tiers.get_principal_tier",
        return_value=SimpleNamespace(value="ADMIN_REVIEWER"),
    ):
        resp = client.get("/api/auth/certificate_info", headers={"Accept": "application/json"})

    payload = resp.get_json()
    assert resp.status_code == 200
    assert payload["principal"] == "cert:test-user"
    assert payload["principal_source"] == "x_arr_clientcert"
    assert payload["cert_metadata"]["cn"] == "Test User"
    assert payload["privilege_tier"] == "ADMIN_REVIEWER"
    assert isinstance(payload.get("timestamp"), str)
    assert isinstance(payload.get("session_context"), dict)


def test_api_ml_usage_contract(client, monkeypatch):
    snapshot = {
        "started_at": "2026-03-09T00:00:00Z",
        "uptime_sec": 10,
        "totals": {"events": 5, "components": 2, "actions": 4},
        "component_counts": {"parser": 3},
        "action_counts": {"parser:run": 2},
        "recent_events": [],
        "persist": {"enabled": True, "path": "x"},
    }
    monkeypatch.setattr("webapp.parser.utils.ml_telemetry.get_ml_telemetry_snapshot", lambda **_kwargs: snapshot)

    resp = client.get("/api/ml_usage?limit=7")
    payload = resp.get_json()

    assert resp.status_code == 200
    assert payload["success"] is True
    assert payload["telemetry"]["totals"]["events"] == 5


def test_api_ml_usage_failure_contract(client, monkeypatch):
    monkeypatch.setattr(
        "webapp.parser.utils.ml_telemetry.get_ml_telemetry_snapshot",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("telemetry failed")),
    )

    resp = client.get("/api/ml_usage")
    payload = resp.get_json()

    assert resp.status_code == 500
    assert payload["success"] is False
    assert "telemetry failed" in payload["error"]


def test_api_ml_pipeline_profile_contract(client, monkeypatch):
    profile = {
        "telemetry": {"totals": {"events": 1, "components": 1, "actions": 1}},
        "training_inputs": {"ml_usage_telemetry_rows": 1},
        "mapping_catalog": {"vocab_root": "x"},
    }
    monkeypatch.setattr("webapp.parser.utils.ml_pipeline_profile.get_ml_pipeline_profile", lambda: profile)

    resp = client.get("/api/ml_pipeline_profile")
    payload = resp.get_json()

    assert resp.status_code == 200
    assert payload["success"] is True
    assert payload["profile"]["mapping_catalog"]["vocab_root"] == "x"


def test_api_ml_vocab_alignment_contract(client, monkeypatch):
    report = {
        "entities": {"entry_count": 3},
        "validators": {"mapping_count": 5},
        "entity_only": {"resolution_rate": 90.0},
        "samples": {"unresolved": []},
    }
    monkeypatch.setattr("webapp.parser.utils.ml_vocab_alignment.get_vocab_alignment_report", lambda **_kwargs: report)

    resp = client.get("/api/ml_vocab_alignment?sample_limit=5")
    payload = resp.get_json()

    assert resp.status_code == 200
    assert payload["success"] is True
    assert payload["alignment"]["entity_only"]["resolution_rate"] == 90.0


def test_api_ml_vocab_alignment_suggestions_contract(client, monkeypatch):
    suggestions = {
        "limit": 10,
        "min_score": 0.45,
        "unresolved_entity_alias_total": 2,
        "suggestion_count": 1,
        "suggestions": [{"file": "a.txt", "alias": "gov", "target": "governor", "best_score": 0.88}],
    }
    monkeypatch.setattr("webapp.parser.utils.ml_vocab_alignment.get_vocab_alignment_suggestions", lambda **_kwargs: suggestions)

    resp = client.get("/api/ml_vocab_alignment_suggestions?limit=10&min_score=0.4")
    payload = resp.get_json()

    assert resp.status_code == 200
    assert payload["success"] is True
    assert payload["suggestions"]["suggestion_count"] == 1


def test_api_ml_vocab_alignment_suggestions_failure_contract(client, monkeypatch):
    monkeypatch.setattr(
        "webapp.parser.utils.ml_vocab_alignment.get_vocab_alignment_suggestions",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("suggestions failed")),
    )

    resp = client.get("/api/ml_vocab_alignment_suggestions")
    payload = resp.get_json()

    assert resp.status_code == 500
    assert payload["success"] is False
    assert "suggestions failed" in payload["error"]


def test_api_ml_vocab_alignment_suggestions_export_invalid_format(client):
    resp = client.get("/api/ml_vocab_alignment_suggestions/export?format=xml")
    payload = resp.get_json()

    assert resp.status_code == 400
    assert payload["success"] is False
    assert "Invalid format" in payload["error"]


def test_api_preingest_url_glimpse_missing_url_contract(client):
    resp = client.get("/api/preingest_url_glimpse")
    payload = resp.get_json()

    assert resp.status_code == 400
    assert payload["success"] is False
    assert "Missing required query param" in payload["error"]


def test_api_preingest_url_glimpse_blocked_url_contract(client, monkeypatch):
    monkeypatch.setattr(appmod, "safe_validate_external_url", lambda _url, allowlist_bypass=False: (False, "blocked"))

    resp = client.get("/api/preingest_url_glimpse?url=https://blocked.example")
    payload = resp.get_json()

    assert resp.status_code == 400
    assert payload["success"] is False
    assert payload["error"] == "url_not_allowed"
    assert payload["reason"] == "blocked"


def test_api_preingest_url_glimpse_success_contract(client, monkeypatch):
    monkeypatch.setattr(appmod, "safe_validate_external_url", lambda _url, allowlist_bypass=False: (True, "ok"))
    monkeypatch.setattr(
        "webapp.parser.utils.url_glimpse.capture_url_glimpse",
        lambda *_args, **_kwargs: {
            "json_report": "tools/debug_headless_output/example.json",
            "html_snapshot": "tools/debug_headless_output/example.html",
            "screenshot": "tools/debug_headless_output/example.png",
            "status": 200,
            "content_type": "text/html",
            "title": "Example",
            "table_count": 3,
            "table_rows_estimate": 20,
            "has_election_terms": True,
            "error": None,
        },
    )
    monkeypatch.setattr(
        "webapp.parser.utils.url_glimpse.build_glimpse_risk_flags",
        lambda _glimpse: {
            "status_code": 200,
            "content_type": "text/html",
            "content_type_supported": True,
            "tables_found": True,
            "has_election_terms": True,
            "risk_level": "low",
        },
    )

    resp = client.get("/api/preingest_url_glimpse?url=https://example.org/results")
    payload = resp.get_json()

    assert resp.status_code == 200
    assert payload["success"] is True
    assert payload["risk"]["risk_level"] == "low"
    assert payload["glimpse"]["table_count"] == 3
    assert payload["artifacts"]["json_report"].endswith("example.json")


def test_api_preingest_url_glimpse_runtime_failure_contract(client, monkeypatch):
    monkeypatch.setattr(appmod, "safe_validate_external_url", lambda _url, allowlist_bypass=False: (True, "ok"))
    monkeypatch.setattr(
        "webapp.parser.utils.url_glimpse.capture_url_glimpse",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("glimpse failed")),
    )

    resp = client.get("/api/preingest_url_glimpse?url=https://example.org/results")
    payload = resp.get_json()

    assert resp.status_code == 500
    assert payload["success"] is False
    assert "glimpse failed" in payload["error"]
