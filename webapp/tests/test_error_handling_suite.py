from __future__ import annotations

from flask import jsonify
import pytest

try:
    import webapp.Smart_Elections_Parser_Webapp as appmod
except ImportError as exc:  # pragma: no cover
    pytest.skip(f"Cannot import webapp app: {exc}", allow_module_level=True)


@pytest.fixture
def non_propagating_client():
    previous_testing = appmod.app.config.get("TESTING")
    previous_propagate = appmod.app.config.get("PROPAGATE_EXCEPTIONS")
    appmod.app.config["TESTING"] = False
    appmod.app.config["PROPAGATE_EXCEPTIONS"] = False
    try:
        with appmod.app.test_client() as test_client:
            yield test_client
    finally:
        appmod.app.config["TESTING"] = previous_testing
        appmod.app.config["PROPAGATE_EXCEPTIONS"] = previous_propagate


def test_file_io_not_configured_returns_structured_500(non_propagating_client, monkeypatch):
    monkeypatch.setitem(appmod.app.config, "_FILE_IO_ROUTE_HANDLERS", None)

    response = non_propagating_client.get("/history")
    payload = response.get_json()

    assert response.status_code == 500
    assert payload["error"] == "File I/O routes are not configured."


def test_data_framework_handler_exception_returns_500(non_propagating_client, monkeypatch):
    def _boom_handler():
        raise RuntimeError("preview exploded")

    monkeypatch.setitem(
        appmod.app.config,
        "_DATA_FRAMEWORK_ROUTE_HANDLERS",
        {"api_data_framework_preview": _boom_handler},
    )

    response = non_propagating_client.get("/api/data_framework/preview")

    assert response.status_code == 500


def test_data_framework_not_configured_returns_structured_500(non_propagating_client, monkeypatch):
    monkeypatch.setitem(appmod.app.config, "_DATA_FRAMEWORK_ROUTE_HANDLERS", None)

    response = non_propagating_client.get("/api/data_framework/preview")
    payload = response.get_json()

    assert response.status_code == 500
    assert payload["error"] == "Data framework routes are not configured."


def test_vocab_alignment_export_invalid_mode_returns_400(non_propagating_client):
    response = non_propagating_client.get("/api/ml_vocab_alignment_suggestions/export?export_mode=bad")
    payload = response.get_json()

    assert response.status_code == 400
    assert payload["success"] is False
    assert "Invalid export_mode" in payload["error"]


def test_preingest_glimpse_runtime_error_returns_500(non_propagating_client, monkeypatch):
    monkeypatch.setattr(appmod, "safe_validate_external_url", lambda _url, allowlist_bypass=False: (True, "ok"))
    monkeypatch.setattr(
        "webapp.parser.utils.url_glimpse.capture_url_glimpse",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("capture failed")),
    )

    response = non_propagating_client.get("/api/preingest_url_glimpse?url=https://example.org/results")
    payload = response.get_json()

    assert response.status_code == 500
    assert payload["success"] is False
    assert "capture failed" in payload["error"]


def test_cert_required_response_json_contract(non_propagating_client):
    response = non_propagating_client.get("/api/auth/certificate_info", headers={"Accept": "application/json"})
    payload = response.get_json()

    assert response.status_code in (200, 401)
    if response.status_code == 401:
        assert payload.get("error") in {"certificate_required", "No certificate found"}
    else:
        assert isinstance(payload, dict)
