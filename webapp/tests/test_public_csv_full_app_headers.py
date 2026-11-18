"""Synthetic full-application regression: CSV route plus global response middleware."""
from collections import OrderedDict

import pytest

from webapp.parser.services import public_csv_delivery as delivery

SOURCE = 'blsrc_v1_' + 'a' * 64
ENDPOINT = '/api/public/csv-download'


@pytest.fixture
def csv_app(monkeypatch):
    from webapp import Smart_Elections_Parser_Webapp as appmod

    # Exercise the actual app and route, not a separately registered blueprint.
    monkeypatch.setitem(appmod.app.config, 'TESTING', True)
    monkeypatch.setenv('BALLOT_LENS_PUBLIC_CSV_DOWNLOAD_ENABLED', 'true')
    broker = delivery.OneUseCsvBroker()
    monkeypatch.setattr(delivery, 'BROKER', broker)
    monkeypatch.setattr(delivery, '_REQUEST_TIMES', OrderedDict())
    assert appmod.app.url_map.bind('localhost').match(ENDPOINT, method='POST')[0] == 'file_io_routes.public_csv_download'
    preview = {
        'contract': 'ballot_lens_public_memory_preview_v1',
        'registry_source_id': SOURCE, 'output_mode': 'MEMORY_PREVIEW_ONLY',
        'persistent_output': False, 'row_count': 1,
        'headers': ['Precinct', 'Synthetic - Election Day', 'Synthetic - Provisional', 'Grand Total'],
        'rows': [{'Precinct': 'Synthetic District', 'Synthetic - Election Day': 0,
                  'Synthetic - Provisional': None, 'Grand Total': 0}],
    }
    content = delivery.encode_finalized_preview(preview, source_id=SOURCE)
    with appmod.app.test_client() as client:
        yield client, broker, content


def test_successful_csv_preserves_private_cache_contract(csv_app):
    client, broker, content = csv_app
    token = broker.publish(source_id=SOURCE, content=content)['token']
    result = client.post(ENDPOINT, json={'token': token})
    assert result.status_code == 200
    assert result.data == content
    assert result.headers['Content-Type'].lower() == 'text/csv; charset=utf-8'
    assert result.headers['Content-Disposition'] == 'attachment; filename="electionpulse-approved-source.csv"'
    assert result.headers['Pragma'] == 'no-cache'
    assert result.headers['Referrer-Policy'] == 'no-referrer'
    assert result.headers['X-Content-Type-Options'] == 'nosniff'
    assert 'origin' in {x.strip().lower() for x in result.headers['Vary'].split(',')}
    replay = client.post(ENDPOINT, json={'token': token})
    assert replay.status_code == 404
    assert replay.headers['Cache-Control'] == 'no-store, no-cache, must-revalidate'
    cache = {x.strip().lower() for x in result.headers['Cache-Control'].split(',')}
    assert {'no-store', 'private', 'max-age=0'} <= cache and 'public' not in cache, 'CSV_FULL_APP_CACHE_CONTRACT'


def test_csv_invalid_and_cross_origin_requests_fail_closed(csv_app):
    client, broker, content = csv_app
    assert client.post(ENDPOINT, json={'token': 'invalid'}).status_code == 404
    token = broker.publish(source_id=SOURCE, content=content)['token']
    assert client.post(ENDPOINT, json={'token': token}, headers={'Origin': 'https://example.invalid'}).status_code == 404
    assert client.post(ENDPOINT, json={'token': token, 'extra': True}).status_code == 404
    good = client.post(ENDPOINT, json={'token': token})
    assert good.status_code == 200 and good.data == content


def test_disabled_csv_does_not_consume_capability(csv_app, monkeypatch):
    client, broker, content = csv_app
    token = broker.publish(source_id=SOURCE, content=content)['token']
    monkeypatch.setenv('BALLOT_LENS_PUBLIC_CSV_DOWNLOAD_ENABLED', 'false')
    result = client.post(ENDPOINT, json={'token': token})
    assert result.status_code == 404
    assert result.headers['Cache-Control'] == 'no-store, no-cache, must-revalidate'
    monkeypatch.setenv('BALLOT_LENS_PUBLIC_CSV_DOWNLOAD_ENABLED', 'true')
    assert client.post(ENDPOINT, json={'token': token}).data == content


def test_unrelated_route_cache_policy_unchanged(csv_app):
    client, _, _ = csv_app
    response = client.get('/api/public/ballot-lens/registry')
    assert response.status_code == 200
    assert response.headers['Cache-Control'] == 'no-store, no-cache, must-revalidate'
    asset = client.get('/static/css/data_framework.css')
    assert asset.status_code == 200
    assert asset.headers['Cache-Control'] == 'public, max-age=31536000, immutable'
