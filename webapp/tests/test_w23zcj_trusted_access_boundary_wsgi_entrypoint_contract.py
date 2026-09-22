from __future__ import annotations

from pathlib import Path

from webapp.trusted_access_boundary_app import create_trusted_access_boundary_app


def _explicit_methods(rule):
    return sorted(set(rule.methods or ()) - {"HEAD", "OPTIONS"})


def test_dedicated_boundary_route_surface_is_minimal():
    app = create_trusted_access_boundary_app()
    routes = {
        rule.rule: _explicit_methods(rule)
        for rule in app.url_map.iter_rules()
    }
    assert app.static_folder is None
    assert routes == {
        "/": ["GET"],
        "/auth/certificate/verify": ["GET"],
    }


def test_boundary_root_boots_without_public_application_secrets(monkeypatch):
    monkeypatch.delenv("GUARDED_INGESTION_KEY", raising=False)
    monkeypatch.delenv("FLASK_SECRET_KEY", raising=False)
    monkeypatch.delenv("TRUSTED_ACCESS_BOUNDARY_ENABLED", raising=False)

    app = create_trusted_access_boundary_app()
    response = app.test_client().get("/")

    assert response.status_code == 200
    assert response.get_data(as_text=True) == "ok\n"
    assert "no-store" in response.headers["Cache-Control"].lower()


def test_boundary_verify_fails_closed_while_boundary_feature_is_disabled(monkeypatch):
    monkeypatch.delenv("GUARDED_INGESTION_KEY", raising=False)
    monkeypatch.delenv("FLASK_SECRET_KEY", raising=False)
    monkeypatch.delenv("TRUSTED_ACCESS_BOUNDARY_ENABLED", raising=False)

    app = create_trusted_access_boundary_app()
    response = app.test_client().get(
        "/auth/certificate/verify?state=" + ("s" * 43)
    )

    assert response.status_code == 404
    assert response.get_json()["error"] == "trusted_access_boundary_disabled"
    assert "no-store" in response.headers["Cache-Control"].lower()


def test_boundary_wsgi_source_does_not_import_full_public_application():
    source = Path("webapp/trusted_access_boundary_app.py").read_text(
        encoding="utf-8"
    )
    assert "Smart_Elections_Parser_Webapp" not in source
    assert "create_public_pages_blueprint" not in source
    assert "GUARDED_INGESTION_KEY" not in source
    assert "FLASK_SECRET_KEY" not in source


def test_boundary_gunicorn_config_has_no_public_app_worker_hook():
    source = Path("trusted_access_gunicorn.conf.py").read_text(
        encoding="utf-8"
    )
    assert "post_worker_init" not in source
    assert "Smart_Elections_Parser_Webapp" not in source
    assert 'os.environ.get("PORT", "8000")' in source
    assert 'bind = f"0.0.0.0:{port}"' in source
