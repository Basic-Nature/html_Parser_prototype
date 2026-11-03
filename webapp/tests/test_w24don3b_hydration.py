"""Read-only shadow feature-gate qualification; synthetic environment settings only."""
from __future__ import annotations
import importlib
from pathlib import Path
import sys
import types
import uuid
import pytest
from flask import Flask, g

ROOT = Path(__file__).resolve().parents[1]
ROUTES = ROOT / 'parser' / 'routes'
TEMPLATES = ROOT / 'templates'
SPONSOR = 'https://github.com/sponsors/Basic-Nature'


def isolated_blueprint():
    prefix = '_w24don3b_' + uuid.uuid4().hex
    mod = types.ModuleType(prefix)
    mod.__path__ = [str(ROUTES)]
    sys.modules[prefix] = mod
    return importlib.import_module(prefix + '.public_pages_blueprint')


def app_with_environment(tmp_path, *, initial_config=None):
    app = Flask('w24don3b', root_path=str(ROOT), template_folder=str(TEMPLATES),
                static_folder=str(ROOT / 'static'), instance_path=str(tmp_path))
    app.config['TESTING'] = True
    if initial_config: app.config.update(initial_config)
    @app.before_request
    def nonce():
        g.csp_nonce = 'synthetic-nonce'
    app.register_blueprint(isolated_blueprint().create_public_pages_blueprint())
    return app


@pytest.mark.parametrize('public,sponsor,expected', [
    (None, None, 404),
    ('false', 'true', 404),
    ('TRUE', 'true', 200),
    ('true', 'true', 200),
    ('yes', 'true', 404),
    ('1', 'true', 404),
    ('  true  ', ' true ', 200),
])
def test_public_and_sponsor_env_hydration(monkeypatch, tmp_path, public, sponsor, expected):
    for key, value in [('EP_SUPPORT_PAGE_PUBLICATION_ENABLED', public),
                       ('EP_SUPPORT_GITHUB_SPONSORS_ENABLED', sponsor)]:
        if value is None: monkeypatch.delenv(key, raising=False)
        else: monkeypatch.setenv(key, value)
    app = app_with_environment(tmp_path)
    response = app.test_client().get('/support')
    assert response.status_code == expected
    assert app.config['EP_SUPPORT_PAGE_PUBLICATION_ENABLED'] is (expected == 200)
    if expected == 200:
        assert SPONSOR in response.get_data(as_text=True)
    assert app.test_client().get('/support/financial-snapshot').status_code == 404


def test_sponsor_disabled_by_default(monkeypatch, tmp_path):
    monkeypatch.setenv('EP_SUPPORT_PAGE_PUBLICATION_ENABLED', 'true')
    monkeypatch.delenv('EP_SUPPORT_GITHUB_SPONSORS_ENABLED', raising=False)
    app = app_with_environment(tmp_path)
    response = app.test_client().get('/support')
    assert response.status_code == 200
    assert SPONSOR not in response.get_data(as_text=True)
    assert app.config['EP_SUPPORT_GITHUB_SPONSORS_ENABLED'] is False


def test_sponsor_true_does_not_activate_page(monkeypatch, tmp_path):
    monkeypatch.delenv('EP_SUPPORT_PAGE_PUBLICATION_ENABLED', raising=False)
    monkeypatch.setenv('EP_SUPPORT_GITHUB_SPONSORS_ENABLED', 'true')
    app = app_with_environment(tmp_path)
    assert app.test_client().get('/support').status_code == 404


def test_explicit_config_false_overrides_env(monkeypatch, tmp_path):
    monkeypatch.setenv('EP_SUPPORT_PAGE_PUBLICATION_ENABLED', 'true')
    monkeypatch.setenv('EP_SUPPORT_GITHUB_SPONSORS_ENABLED', 'true')
    app = app_with_environment(tmp_path, initial_config={'EP_SUPPORT_PAGE_PUBLICATION_ENABLED': False})
    assert app.test_client().get('/support').status_code == 404


def test_string_config_does_not_override_exact_boolean_guard(monkeypatch, tmp_path):
    monkeypatch.setenv('EP_SUPPORT_PAGE_PUBLICATION_ENABLED', 'true')
    monkeypatch.setenv('EP_SUPPORT_GITHUB_SPONSORS_ENABLED', 'true')
    app = app_with_environment(tmp_path, initial_config={'EP_SUPPORT_PAGE_PUBLICATION_ENABLED': 'true'})
    assert app.test_client().get('/support').status_code == 404


def test_financial_and_preview_flags_not_hydrated(monkeypatch, tmp_path):
    monkeypatch.setenv('EP_SUPPORT_PAGE_PUBLICATION_ENABLED', 'true')
    monkeypatch.setenv('EP_SUPPORT_GITHUB_SPONSORS_ENABLED', 'true')
    monkeypatch.setenv('EP_SUPPORT_LOCAL_PREVIEW_ENABLED', 'true')
    monkeypatch.setenv('EP_SUPPORT_FINANCIAL_PUBLICATION_ENABLED', 'true')
    app = app_with_environment(tmp_path)
    assert 'EP_SUPPORT_LOCAL_PREVIEW_ENABLED' not in app.config
    assert 'EP_SUPPORT_FINANCIAL_PUBLICATION_ENABLED' not in app.config
    assert app.test_client().get('/support').status_code == 200
    assert app.test_client().get('/support/financial-snapshot').status_code == 404


def test_financial_unavailable_with_sponsors_active(monkeypatch, tmp_path):
    monkeypatch.setenv('EP_SUPPORT_PAGE_PUBLICATION_ENABLED', 'true')
    monkeypatch.setenv('EP_SUPPORT_GITHUB_SPONSORS_ENABLED', 'true')
    app = app_with_environment(tmp_path, initial_config={'EP_SUPPORT_FINANCIAL_PUBLICATION_ENABLED': True,
        'EP_SUPPORT_APPROVED_FINANCIAL_RECORD_ID': 'review-123456'})
    response = app.test_client().get('/support')
    assert response.status_code == 200
    assert SPONSOR in response.get_data(as_text=True)
    assert app.test_client().get('/support/financial-snapshot').status_code == 404
