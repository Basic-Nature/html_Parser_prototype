"""B12 security tests: synthetic fixtures; real source covered by shadow runner."""
from __future__ import annotations

import csv
import io

import pytest

from webapp.parser.services.public_csv_delivery import BROKER, encode_finalized_preview
from webapp.parser.services.public_compact_csv_handoff import CompactCsvHandoff

SOURCE = 'blsrc_v1_' + 'a' * 64
DIGEST = 'b' * 64
HEADERS = ['Precinct', 'Grand Total', 'Jane Doe (DEM) - Election Day', 'Jane Doe (DEM) - Total Votes']
ROWS = [
    {'Precinct':'Synthetic 1','Grand Total':2,'Jane Doe (DEM) - Election Day':0,'Jane Doe (DEM) - Total Votes':None},
    {'Precinct':'Grand Total','Grand Total':2,'Jane Doe (DEM) - Election Day':0,'Jane Doe (DEM) - Total Votes':2},
]


def test_wide_csv_allows_grand_total_not_at_end_and_preserves_null_zero():
    p = {'contract':'ballot_lens_public_memory_preview_v1','registry_source_id':SOURCE,
         'output_mode':'MEMORY_PREVIEW_ONLY','persistent_output':False,
         'headers':HEADERS,'rows':ROWS,'row_count':len(ROWS)}
    values = list(csv.reader(io.StringIO(encode_finalized_preview(p,source_id=SOURCE).decode('utf-8-sig'))))
    assert values[0] == HEADERS
    assert values[1][-2:] == ['0','']
    assert values[2][-2:] == ['0','2']


def test_compact_requires_full_ack_scope_one_use_and_expiry(monkeypatch):
    monkeypatch.setenv('BALLOT_LENS_PUBLIC_CSV_DOWNLOAD_ENABLED','1')
    clock=[10.0]
    handoff=CompactCsvHandoff(clock=lambda:clock[0])
    key='syntheticSnapshot1';sid='sess_synthetic_1'
    assert handoff.stage(snapshot_id=key,session_id=sid,source_id=SOURCE,
                         source_sha256=DIGEST,headers=HEADERS,rows=ROWS)
    assert handoff.pending_count()==1
    assert handoff.finish(snapshot_id=key,session_id='sess_foreign',source_id=SOURCE) is None
    assert handoff.pending_count()==1
    offer=handoff.finish(snapshot_id=key,session_id=sid,source_id=SOURCE)
    assert offer and offer['registry_source_id']==SOURCE
    assert BROKER.take(offer['token']) is not None
    assert BROKER.take(offer['token']) is None
    assert handoff.finish(snapshot_id=key,session_id=sid,source_id=SOURCE) is None
    assert handoff.stage(snapshot_id='syntheticSnapshot2',session_id=sid,source_id=SOURCE,
                         source_sha256=DIGEST,headers=HEADERS,rows=ROWS)
    assert handoff.cancel(snapshot_id='syntheticSnapshot2',session_id='sess_foreign',source_id=SOURCE) is False
    assert handoff.cancel(snapshot_id='syntheticSnapshot2',session_id=sid,source_id=SOURCE) is True
    assert handoff.stage(snapshot_id='syntheticSnapshot3',session_id=sid,source_id=SOURCE,
                         source_sha256=DIGEST,headers=HEADERS,rows=ROWS)
    clock[0]+=601
    assert handoff.finish(snapshot_id='syntheticSnapshot3',session_id=sid,source_id=SOURCE) is None
    assert handoff.pending_count()==0


def test_disabled_and_bad_identity_fail_closed(monkeypatch):
    monkeypatch.delenv('BALLOT_LENS_PUBLIC_CSV_DOWNLOAD_ENABLED',raising=False)
    handoff=CompactCsvHandoff()
    assert handoff.stage(snapshot_id='syntheticSnapshot4',session_id='sess_synthetic',source_id=SOURCE,
                         source_sha256=DIGEST,headers=HEADERS,rows=ROWS) is False
    monkeypatch.setenv('BALLOT_LENS_PUBLIC_CSV_DOWNLOAD_ENABLED','1')
    with pytest.raises(ValueError):
        handoff.stage(snapshot_id='syntheticSnapshot4',session_id='untrusted',source_id=SOURCE,
                      source_sha256=DIGEST,headers=HEADERS,rows=ROWS)


def test_csv_request_rate_limit(monkeypatch):
    from webapp.parser.services import public_csv_delivery as d
    assert d.allow_public_csv_request('198.51.100.123',now=900.0)
    for _ in range(19):
        assert d.allow_public_csv_request('198.51.100.123',now=900.0)
    assert not d.allow_public_csv_request('198.51.100.123',now=900.0)
    assert d.allow_public_csv_request('198.51.100.123',now=961.0)
