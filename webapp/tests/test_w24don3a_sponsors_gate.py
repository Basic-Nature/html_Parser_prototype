"""W24DON3A detached offline qualification — synthetic flags only; no network."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
import types
import uuid
import pytest
from flask import Flask, g

ROOT = Path(__file__).resolve().parents[1]
ROUTE_PATH = ROOT / "parser" / "routes" / "public_pages_blueprint.py"
TEMPLATE_DIR = ROOT / "templates"
SPONSOR_URL = "https://github.com/sponsors/Basic-Nature"

def _module():
    prefix = "_don3a_" + uuid.uuid4().hex
    package = types.ModuleType(prefix)
    package.__path__ = [str(ROUTE_PATH.parent)]
    sys.modules[prefix] = package
    snap = types.ModuleType(prefix + ".public_support_snapshot")
    snap.load_approved_public_snapshot = lambda *_args, **_kwargs: None
    sys.modules[snap.__name__] = snap
    monitor = types.ModuleType(prefix + ".route_monitor")
    monitor.record_route_monitor_event = lambda *_args, **_kwargs: None
    sys.modules[monitor.__name__] = monitor
    spec = importlib.util.spec_from_file_location(prefix + ".public_pages_blueprint", ROUTE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

@pytest.fixture
def app(tmp_path):
    m = _module()
    flask = Flask("don3a_qualified", root_path=str(ROOT), template_folder=str(TEMPLATE_DIR), static_folder=str(ROOT / "static"), instance_path=str(tmp_path))
    flask.config.update(TESTING=True,
        EP_SUPPORT_PAGE_PUBLICATION_ENABLED=False,
        EP_SUPPORT_LOCAL_PREVIEW_ENABLED=False,
        EP_SUPPORT_FINANCIAL_PUBLICATION_ENABLED=False,
        EP_SUPPORT_PAYMENTS_ENABLED=False,
        EP_SUPPORT_GITHUB_SPONSORS_ENABLED=False)
    @flask.before_request
    def before():
        g.csp_nonce = "synthetic-only"
    flask.register_blueprint(m.create_public_pages_blueprint())
    return flask

def test_default_is_404(app):
    assert app.test_client().get("/support").status_code == 404
    assert app.test_client().get("/support/financial-snapshot").status_code == 404

def test_preview_never_exposes_sponsor_link(app):
    app.config.update(EP_SUPPORT_LOCAL_PREVIEW_ENABLED=True, EP_SUPPORT_GITHUB_SPONSORS_ENABLED=True)
    r=app.test_client().get("/support")
    assert r.status_code == 200
    html=r.get_data(as_text=True)
    assert "PRIVATE PREVIEW ONLY" in html
    assert SPONSOR_URL not in html

def test_public_page_without_sponsor_authorization(app):
    app.config["EP_SUPPORT_PAGE_PUBLICATION_ENABLED"] = True
    html=app.test_client().get("/support").get_data(as_text=True)
    assert "PUBLIC INFORMATION ONLY" in html
    assert SPONSOR_URL not in html
    assert "Payments currently disabled" in html

def test_public_sponsor_link_external_only_and_independent_of_finance(app):
    app.config.update(EP_SUPPORT_PAGE_PUBLICATION_ENABLED=True, EP_SUPPORT_GITHUB_SPONSORS_ENABLED=True)
    html=app.test_client().get("/support").get_data(as_text=True)
    assert html.count(SPONSOR_URL) == 1
    assert 'rel="noopener noreferrer external"' in html
    assert 'target="_blank"' in html
    assert "Support through GitHub Sponsors" in html
    assert "GitHub sponsorship is off-site" in html
    assert "No financial or card information" in html
    assert "Payments currently disabled" not in html
    assert "/support/financial-snapshot" not in html
    assert "support_snapshot.js" not in html
    assert app.test_client().get("/support/financial-snapshot").status_code == 404

def test_string_flag_fails_closed(app):
    app.config.update(EP_SUPPORT_PAGE_PUBLICATION_ENABLED=True, EP_SUPPORT_GITHUB_SPONSORS_ENABLED="true")
    html=app.test_client().get("/support").get_data(as_text=True)
    assert SPONSOR_URL not in html

def test_reserved_payment_flag_does_not_authorize_sponsors(app):
    app.config.update(EP_SUPPORT_PAGE_PUBLICATION_ENABLED=True, EP_SUPPORT_PAYMENTS_ENABLED=True)
    html=app.test_client().get("/support").get_data(as_text=True)
    assert SPONSOR_URL not in html

def test_sponsors_do_not_enable_financial_snapshot(app):
    app.config.update(EP_SUPPORT_PAGE_PUBLICATION_ENABLED=True,
                      EP_SUPPORT_GITHUB_SPONSORS_ENABLED=True,
                      EP_SUPPORT_FINANCIAL_PUBLICATION_ENABLED=True,
                      EP_SUPPORT_APPROVED_FINANCIAL_RECORD_ID="review-123456")
    assert app.test_client().get("/support/financial-snapshot").status_code == 404
