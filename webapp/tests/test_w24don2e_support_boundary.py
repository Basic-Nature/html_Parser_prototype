"""W24DON2E: detached Flask route and file-boundary contracts; synthetic funds only."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import types
import uuid

import pytest
from flask import Flask, g

ROUTES = Path(__file__).resolve().parents[1] / 'parser' / 'routes'
TEMPLATES = Path(__file__).resolve().parents[1] / 'templates'


def load_modules():
    # Isolated module namespace avoids importing the full ElectionPulse WSGI app.
    prefix = '_w24don2e_routes_' + uuid.uuid4().hex
    package = types.ModuleType(prefix)
    package.__path__ = [str(ROUTES)]
    sys.modules[prefix] = package
    return importlib.import_module(prefix + '.public_pages_blueprint'), importlib.import_module(prefix + '.public_support_snapshot')


@pytest.fixture
def harness(tmp_path):
    blueprint, validator = load_modules()
    app = Flask('don2e_detached', root_path=str(TEMPLATES.parent),
                template_folder=str(TEMPLATES), static_folder=str(TEMPLATES.parent/'static'),
                instance_path=str(tmp_path))
    app.config.update(TESTING=True, EP_SUPPORT_PAGE_PUBLICATION_ENABLED=False,
                      EP_SUPPORT_LOCAL_PREVIEW_ENABLED=False,
                      EP_SUPPORT_FINANCIAL_PUBLICATION_ENABLED=False,
                      EP_SUPPORT_PAYMENTS_ENABLED=False)
    @app.before_request
    def nonce():
        g.csp_nonce = 'synthetic-test-nonce'
    app.register_blueprint(blueprint.create_public_pages_blueprint())
    return app, validator, tmp_path


def document():
    return {
        'schema': 'ELECTIONPULSE_SUPPORT_PUBLIC_SNAPSHOT_REV2',
        'source_revision': 'public-v1234', 'snapshot_date': '2026-10-29',
        'publish_approved': True, 'approval_evidence': {'record_id': 'approved-review-123'},
        'founder': {'provider_record_payment_subtotal_usd': 12.00,
                    'invoice_paid_status_azure_usd': 6.00,
                    'applied_payment_github_usd': 4.00,
                    'received_payment_godaddy_usd': 2.00},
        'contributions': {'gross_captured_usd': 10.00, 'net_settled_usd': 9.00,
                          'spent_on_project_usd': 3.00, 'committed_unspent_usd': 2.00,
                          'unallocated_available_usd': 4.00},
        'monthly': [{'period': '2026-10', 'azure_paid_invoice_usd': 6.00,
                     'github_applied_payments_usd': 4.00,
                     'godaddy_paid_receipt_usd': 2.00,
                     'provider_payment_sum_usd': 12.00}],
    }


def write(tmp, obj):
    path = tmp/'support'/'approved_public_snapshot.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding='utf-8')
    return path


def publish_settings(app):
    app.config.update(EP_SUPPORT_PAGE_PUBLICATION_ENABLED=True,
                      EP_SUPPORT_FINANCIAL_PUBLICATION_ENABLED=True,
                      EP_SUPPORT_APPROVED_FINANCIAL_RECORD_ID='approved-review-123')


def test_default_off_and_reserved_payment_flag(harness):
    app, _, _ = harness
    app.config['EP_SUPPORT_PAYMENTS_ENABLED'] = True
    client = app.test_client()
    assert client.get('/support').status_code == 404
    assert client.get('/support/financial-snapshot').status_code == 404


def test_private_preview_never_fetches_server_financials(harness):
    app, _, _ = harness
    app.config['EP_SUPPORT_LOCAL_PREVIEW_ENABLED'] = True
    response = app.test_client().get('/support')
    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert 'PRIVATE PREVIEW ONLY' in page
    assert 'support_snapshot.js' in page
    assert app.test_client().get('/support/financial-snapshot').status_code == 404


def test_public_copy_has_no_unapproved_provider_history(harness):
    app, _, _ = harness
    app.config['EP_SUPPORT_PAGE_PUBLICATION_ENABLED'] = True
    response = app.test_client().get('/support')
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'PUBLIC INFORMATION ONLY' in html
    assert 'support_snapshot.js' not in html
    assert 'Oct 27 receipt' not in html
    assert 'May 2025 – Oct 2026' not in html
    assert 'Jan 16, 2026' not in html
    assert 'February and March 2026' not in html
    assert 'Payments currently disabled' in html
    assert '/support/financial-snapshot' not in html


def test_public_financial_flag_without_record_is_not_release(harness):
    app, _, tmp = harness
    app.config['EP_SUPPORT_PAGE_PUBLICATION_ENABLED'] = True
    app.config['EP_SUPPORT_FINANCIAL_PUBLICATION_ENABLED'] = True
    assert app.test_client().get('/support/financial-snapshot').status_code == 404
    app.config['EP_SUPPORT_APPROVED_FINANCIAL_RECORD_ID'] = 'approved-review-123'
    assert app.test_client().get('/support/financial-snapshot').status_code == 404


def test_approved_snapshot_only_when_every_gate_matches(harness):
    app, _, tmp = harness
    write(tmp, document())
    publish_settings(app)
    res = app.test_client().get('/support/financial-snapshot')
    assert res.status_code == 200
    assert 'no-store' in res.headers['Cache-Control']
    assert res.headers['X-Content-Type-Options'] == 'nosniff'
    data = res.get_json()
    assert data['contributions']['net_settled_usd'] == 9.0
    assert data['founder']['provider_record_payment_subtotal_usd'] == 12.0
    assert 'street_address' not in json.dumps(data)
    app.config['EP_SUPPORT_APPROVED_FINANCIAL_RECORD_ID'] = 'different-approval'
    assert app.test_client().get('/support/financial-snapshot').status_code == 404


@pytest.mark.parametrize('change', [
    lambda d: d.update(contact_email='personal@example.invalid'),
    lambda d: d['approval_evidence'].update(private_notes='never-public'),
    lambda d: d['contributions'].update(net_settled_usd=None),
    lambda d: d['contributions'].update(unallocated_available_usd=6.0),
    lambda d: d['monthly'][0].update(provider_payment_sum_usd=13.0),
    lambda d: d['monthly'].append(d['monthly'][0].copy()),
    lambda d: d.update(snapshot_date='2026-02-31'),
])
def test_unapproved_or_corrupt_payload_rejected(harness, change):
    app, _, tmp = harness
    publish_settings(app)
    d = document()
    change(d)
    write(tmp,d)
    assert app.test_client().get('/support/financial-snapshot').status_code == 404


def test_private_directory_symlink_not_followed(harness):
    app, _, tmp = harness
    publish_settings(app)
    other = tmp/'elsewhere.json'
    other.write_text(json.dumps(document()))
    d = tmp/'support'
    d.mkdir()
    path = d/'approved_public_snapshot.json'
    try:
        path.symlink_to(other)
    except (OSError, NotImplementedError):
        pytest.skip('Symlinks unavailable for this Windows user')
    assert app.test_client().get('/support/financial-snapshot').status_code == 404


def test_wrong_type_flags_fail_closed(harness):
    app, _, tmp = harness
    write(tmp, document())
    publish_settings(app)
    app.config['EP_SUPPORT_FINANCIAL_PUBLICATION_ENABLED'] = 'true'
    assert app.test_client().get('/support/financial-snapshot').status_code == 404
    app.config['EP_SUPPORT_FINANCIAL_PUBLICATION_ENABLED'] = True
    app.config['EP_SUPPORT_PAGE_PUBLICATION_ENABLED'] = 'true'
    assert app.test_client().get('/support/financial-snapshot').status_code == 404


def test_public_snapshot_is_not_deployed_static(harness):
    app, _, _ = harness
    static_file = Path(__file__).resolve().parents[1]/'static'/'js'/'support_snapshot.js'
    content = static_file.read_text(encoding='utf-8')
    assert 'publish_approved: false' in content
    assert 'founder: null' in content
    assert 'contributions: null' in content
    assert 'monthly: []' in content
    assert 'approved_public_snapshot.json' not in content
