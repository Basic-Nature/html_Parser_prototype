"""Reusable public CSV schema tests; no jurisdiction-specific production dispatch."""
from __future__ import annotations
import csv
import hashlib
import io
import pytest
from webapp.parser.services.public_registry_structured_csv import parse_approved_precinct_csv, PublicStructuredCSVError
from webapp.parser.services.public_registry_csv_dispatch import parse_runtime_approved_csv, ApprovedCSVDispatchError


def _source(headers,records):
    out=io.StringIO(newline='')
    writer=csv.DictWriter(out,fieldnames=headers)
    writer.writeheader();writer.writerows(records)
    return out.getvalue().encode('utf-8')


def test_generic_camel_case_headers_and_distinct_unknown_methods():
    hdr=['CandidateName','TOTAL_VOTES','LocalityName','LocalityCode','PrecinctName','PrecinctId','OfficeTitle','ElectionDate','Party']
    row={'CandidateName':'Candidate A','TOTAL_VOTES':'6','LocalityName':'COUNTY','LocalityCode':'001','PrecinctName':'Central','PrecinctId':'0101','OfficeTitle':'Member, United States Senate','ElectionDate':'2024-11-05','Party':'Independent'}
    payload=_source(hdr,[row]);shape=parse_approved_precinct_csv(payload,contest='Senate',year='2024',state='Virginia')
    assert shape.precinct_rows==1 and shape.rows[0]['Grand Total']==6
    assert shape.rows[0]['Candidate A (Independent) - Election Day'] is None
    assert shape.missing_method_cells==4


def test_fails_closed_for_wrong_year_or_missing_nonzero_precinct_id():
    hdr=['candidate','votes','locality','precinct id','contest','election year']
    row={'candidate':'Candidate A','votes':'5','locality':'County','precinct id':'100','contest':'President','election year':'2024'}
    shape=parse_approved_precinct_csv(_source(hdr,[row]),contest='President',year='2024')
    assert shape.precinct_rows==1
    with pytest.raises(PublicStructuredCSVError):
        parse_approved_precinct_csv(_source(hdr,[{**row,'election year':'2020'}]),contest='President',year='2024')
    with pytest.raises(PublicStructuredCSVError):
        parse_approved_precinct_csv(_source(hdr,[{**row,'precinct id':''}]),contest='President',year='2024')


def test_dispatch_is_only_server_approved_source_and_digest():
    from types import SimpleNamespace
    data=_source(['candidate','votes','locality','precinct','contest'],[{'candidate':'Candidate A','votes':'2','locality':'County','precinct':'P-1','contest':'Mayor'}])
    rt=SimpleNamespace(registry_source_id='approved-source-id',approved_target_url='https://example.gov/results.csv',source_projection={'state':'Example','year':'2024','contest':'Mayor'})
    shape=parse_runtime_approved_csv(rt,data,hashlib.sha256(data).hexdigest(),parse_approved_precinct_csv)
    assert shape.rows[-1]['Grand Total']==2
    with pytest.raises(ApprovedCSVDispatchError):
        parse_runtime_approved_csv(rt,data,'f'*64,parse_approved_precinct_csv)
