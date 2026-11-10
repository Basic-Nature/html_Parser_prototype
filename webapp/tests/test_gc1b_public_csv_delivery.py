"""Synthetic security fixtures only; NOT real election extraction previews."""
from __future__ import annotations

import csv
import io
import pytest

from webapp.parser.services.public_csv_delivery import (
    OneUseCsvBroker, encode_finalized_preview, publish_from_public_result,
)

SOURCE = 'blsrc_v1_' + 'a' * 64


def synthetic_preview():
    return {
        'contract': 'ballot_lens_public_memory_preview_v1',
        'registry_source_id': SOURCE,
        'output_mode': 'MEMORY_PREVIEW_ONLY',
        'persistent_output': False,
        'headers': ['Precinct', 'Jane Doe - Election Day', 'Jane Doe - Provisional', 'Jane Doe - Total Votes', 'Grand Total'],
        'rows': [{'Precinct': 'Synthetic District', 'Jane Doe - Election Day': 2,
                  'Jane Doe - Provisional': 0, 'Jane Doe - Total Votes': None, 'Grand Total': 2}],
        'row_count': 1,
    }


def test_csv_preserves_real_zero_vs_missing_and_column_order():
    binary = encode_finalized_preview(synthetic_preview(), source_id=SOURCE)
    data = list(csv.reader(io.StringIO(binary.decode('utf-8-sig'))))
    assert data[0][-2:] == ['Jane Doe - Total Votes', 'Grand Total']
    assert data[1][-3:] == ['0', '', '2']


def test_injection_does_not_export_spreadsheet_formula():
    item = synthetic_preview()
    item['rows'][0]['Precinct'] = '=cmd()'
    item['rows'][0]['Jane Doe - Election Day'] = '@SUM(1,2)'
    value = encode_finalized_preview(item, source_id=SOURCE).decode('utf-8-sig')
    assert "'=cmd()" in value and "'@SUM(1,2)" in value


def test_rejects_fake_source_and_unknown_vote_columns():
    item = synthetic_preview()
    with pytest.raises(ValueError):
        encode_finalized_preview(item, source_id='blsrc_v1_' + 'b'*64)
    item['rows'][0]['Imaginary Candidate'] = 90
    with pytest.raises(ValueError):
        encode_finalized_preview(item, source_id=SOURCE)


def test_rejects_missing_or_empty_precinct():
    item = synthetic_preview()
    item['rows'][0]['Precinct'] = ''
    with pytest.raises(ValueError):
        encode_finalized_preview(item, source_id=SOURCE)


def test_one_use_and_expiry_and_capacity():
    tick = [0.0]
    broker = OneUseCsvBroker(clock=lambda: tick[0])
    token = broker.publish(source_id=SOURCE, content=b'csv')['token']
    assert broker.take(token) == b'csv'
    assert broker.take(token) is None
    token = broker.publish(source_id=SOURCE, content=b'csv')['token']
    tick[0] += 121
    assert broker.take(token) is None
    for _ in range(64):
        broker.publish(source_id=SOURCE, content=b'x')
    with pytest.raises(RuntimeError):
        broker.publish(source_id=SOURCE, content=b'x')


def test_feature_default_disabled(monkeypatch):
    monkeypatch.delenv('BALLOT_LENS_PUBLIC_CSV_DOWNLOAD_ENABLED', raising=False)
    assert publish_from_public_result({'contract':'ballot_lens_public_runtime_result_v1'}) is None


def test_result_cannot_be_published_from_unapproved_contract(monkeypatch):
    monkeypatch.setenv('BALLOT_LENS_PUBLIC_CSV_DOWNLOAD_ENABLED', 'true')
    item = {'contract': 'ballot_lens_public_runtime_result_v1', 'registry_source_id': SOURCE,
            'download_available': False, 'persistent_output': False,
            'terminal_status': 'completed', 'outputs': [synthetic_preview()]}
    assert publish_from_public_result({**item, 'outputs': []}) is None
    assert publish_from_public_result({**item, 'registry_source_id': 'bad'}) is None


def test_http_route_one_use_mime_and_no_store(monkeypatch):
    from flask import Flask
    from webapp.parser.routes.file_io_blueprint import create_file_io_blueprint
    from webapp.parser.services.public_csv_delivery import BROKER
    monkeypatch.setenv('BALLOT_LENS_PUBLIC_CSV_DOWNLOAD_ENABLED', '1')
    app = Flask('csv-contract-test')
    app.register_blueprint(create_file_io_blueprint())
    offer = BROKER.publish(source_id=SOURCE, content=b'Precinct,Grand Total\r\nSynthetic District,0\r\n')
    client = app.test_client()
    result = client.post('/api/public/csv-download', json={'token': offer['token']})
    assert result.status_code == 200
    assert result.data.startswith(b'Precinct,')
    assert result.headers['Cache-Control'].startswith('no-store')
    assert result.headers['X-Content-Type-Options'] == 'nosniff'
    assert client.post('/api/public/csv-download', json={'token': offer['token']}).status_code == 404


def test_http_route_fail_closed_on_origin_and_disabled(monkeypatch):
    from flask import Flask
    from webapp.parser.routes.file_io_blueprint import create_file_io_blueprint
    from webapp.parser.services.public_csv_delivery import BROKER
    app = Flask('csv-contract-test-origin')
    app.register_blueprint(create_file_io_blueprint())
    client = app.test_client()
    offer = BROKER.publish(source_id=SOURCE, content=b'Safe')
    monkeypatch.delenv('BALLOT_LENS_PUBLIC_CSV_DOWNLOAD_ENABLED', raising=False)
    assert client.post('/api/public/csv-download',json={'token':offer['token']}).status_code == 404
    monkeypatch.setenv('BALLOT_LENS_PUBLIC_CSV_DOWNLOAD_ENABLED','1')
    assert client.post('/api/public/csv-download',json={'token':offer['token']},headers={'Origin':'https://elsewhere.example'}).status_code == 404
    assert client.post('/api/public/csv-download',json={'token':offer['token']}).status_code == 200
