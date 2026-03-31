from __future__ import annotations

from flask import jsonify
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


def test_file_io_heartbeat_and_legacy_heartbeat_routes(client, monkeypatch):
    def _heartbeat_handler():
        return jsonify({"ok": True, "path": "heartbeat"}), 200

    monkeypatch.setitem(
        appmod.app.config,
        "_FILE_IO_ROUTE_HANDLERS",
        {"heartbeat": _heartbeat_handler},
    )

    modern = client.get("/heartbeat")
    legacy = client.get("/Heartbeat")

    assert modern.status_code == 200
    assert legacy.status_code == 200
    assert modern.get_json()["ok"] is True
    assert legacy.get_json()["path"] == "heartbeat"


def test_file_io_route_returns_500_when_handler_missing(client, monkeypatch):
    monkeypatch.setitem(appmod.app.config, "_FILE_IO_ROUTE_HANDLERS", {})

    response = client.get("/history")
    payload = response.get_json()

    assert response.status_code == 500
    assert "Missing file I/O handler" in payload["error"]


def test_data_framework_preview_dispatch_success(client, monkeypatch):
    def _preview_handler():
        return jsonify({"success": True, "records": []}), 200

    monkeypatch.setitem(
        appmod.app.config,
        "_DATA_FRAMEWORK_ROUTE_HANDLERS",
        {"api_data_framework_preview": _preview_handler},
    )

    response = client.get("/api/data_framework/preview")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["success"] is True
    assert isinstance(payload["records"], list)


def test_data_framework_preview_returns_500_when_missing_handler(client, monkeypatch):
    monkeypatch.setitem(appmod.app.config, "_DATA_FRAMEWORK_ROUTE_HANDLERS", {})

    response = client.get("/api/data_framework/preview")
    payload = response.get_json()

    assert response.status_code == 500
    assert "Missing data framework handler" in payload["error"]


def test_file_io_history_dispatch_success(client, monkeypatch):
    def _history_handler():
        return jsonify({"entries": [{"id": 1}], "success": True}), 200

    monkeypatch.setitem(
        appmod.app.config,
        "_FILE_IO_ROUTE_HANDLERS",
        {"history": _history_handler},
    )

    response = client.get("/history")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["success"] is True
    assert payload["entries"] == [{"id": 1}]


def test_data_framework_preview_allows_query_passthrough(client, monkeypatch):
    def _preview_handler():
        return jsonify({"success": True, "query_seen": True}), 200

    monkeypatch.setitem(
        appmod.app.config,
        "_DATA_FRAMEWORK_ROUTE_HANDLERS",
        {"api_data_framework_preview": _preview_handler},
    )

    response = client.get("/api/data_framework/preview?limit=5&sort=votes")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["success"] is True
    assert payload["query_seen"] is True


def test_auth_welcome_blocks_post_method(client):
    response = client.post("/auth/welcome")
    assert response.status_code == 405


def test_upload_input_blocks_get_method(client):
    response = client.get("/upload/input")
    assert response.status_code == 405
