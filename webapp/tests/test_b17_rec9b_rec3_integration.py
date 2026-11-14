"""B17_REC9B_REC3 — synthetic integration contracts against isolated checkout.

No external source reads or election result previews. All rows in tests are
abstract labels and scalar test values; they are not real election results.
"""
from __future__ import annotations

import pytest
from types import SimpleNamespace


def test_shared_logic_exports_the_exact_pure_contract():
    from webapp.parser.utils import shared_logic as shared
    from webapp.parser.utils import shared_election_normalization as pure
    assert shared.preserve_jurisdiction is pure.preserve_jurisdiction
    assert shared.parse_explicit_vote_cell is pure.parse_explicit_vote_cell
    assert shared.candidate_method_matrix is pure.candidate_method_matrix
    assert shared.propose_source_route is pure.propose_source_route


def test_state_router_explicit_district_does_not_become_precinct():
    from webapp.parser.state_router import describe_explicit_result_jurisdiction
    observed=describe_explicit_result_jurisdiction(
        source_scope='statewide',source_state='Example State',
        explicit_row_fields={'election_district':'ED-3'},
        navigation_county_hint='suggested county')
    assert observed['row_jurisdiction']['precinct'] is None
    assert observed['row_jurisdiction']['county'] is None
    assert observed['row_jurisdiction']['election_district']=='ED-3'
    assert observed['navigation_county_hint_not_authoritative']=='suggested county'
    assert observed['row_granularity_from_explicit_labels']=='election_district'


def test_state_router_rejects_unknown_jurisdiction_field():
    from webapp.parser.state_router import describe_explicit_result_jurisdiction
    from webapp.parser.utils.shared_election_normalization import NormalizationContractError
    with pytest.raises(NormalizationContractError):
        describe_explicit_result_jurisdiction(source_scope='statewide',source_state='Example',
             explicit_row_fields={'unknown_geography':'X'})


def test_state_handler_default_metadata_is_unchanged():
    from webapp.parser.handlers.shared.state_handler_base import StateHandlerBase
    fake=SimpleNamespace(STATE_CODE='EX')
    md=StateHandlerBase.build_metadata(fake,
       {'title':'Office','state':'Example State','county':'Navigation Label'},[],[],{},
       {'url':'https://example.invalid/record'},'session')
    assert md['county']=='Navigation Label'
    assert 'source_jurisdiction_evidence' not in md


def test_state_handler_optin_retains_explicit_row_fields():
    from webapp.parser.handlers.shared.state_handler_base import StateHandlerBase
    fake=SimpleNamespace(STATE_CODE='EX')
    md=StateHandlerBase.build_metadata(fake,
       {'title':'Office','state':'Example State','county':'Navigation Label'},[],[],{},
       {'url':'https://example.invalid/record','source_scope':'statewide',
        'county':'Navigation Label', 'shared_normalization_opt_in':True,
        'explicit_row_jurisdiction':{'election_district':'3','precinct':'3-1'}},'session')
    evidence=md['source_jurisdiction_evidence']
    assert evidence['row_jurisdiction']['county'] is None
    assert evidence['row_jurisdiction']['precinct']=='3-1'
    assert evidence['row_jurisdiction']['election_district']=='3'
    assert evidence['navigation_county_hint_not_authoritative']=='Navigation Label'
    assert evidence['precinct_result_certified'] is False


def test_state_handler_optin_requires_explicit_mapping():
    from webapp.parser.handlers.shared.state_handler_base import StateHandlerBase
    from webapp.parser.utils.shared_election_normalization import NormalizationContractError
    with pytest.raises(NormalizationContractError):
        StateHandlerBase.build_metadata(SimpleNamespace(STATE_CODE='EX'),
          {'title':'Office'}, [], [], {}, {'shared_normalization_opt_in':True},'session')


def test_output_helper_preserves_zero_and_missing_and_all_methods():
    from webapp.parser.utils.output_utils import build_source_native_candidate_method_row
    row=build_source_native_candidate_method_row(
        precinct='P-001',candidates=['A','B'],methods=['Early','Election Day'],
        observed={('A','Early'):'0',('B','Election Day'):'25'})
    assert row['Precinct']=='P-001'
    assert row['A - Early']==0
    assert row['A - Election Day'] is None
    assert row['B - Early'] is None
    assert row['B - Election Day']==25
    assert len(row)==5
    assert 'Grand Total' not in row
    assert '% Precincts Reporting' not in row


def test_output_helper_refuses_unlabeled_precinct():
    from webapp.parser.utils.output_utils import build_source_native_candidate_method_row
    from webapp.parser.utils.shared_election_normalization import NormalizationContractError
    with pytest.raises(NormalizationContractError):
        build_source_native_candidate_method_row(precinct='',candidates=['A'],
                                                 methods=['Early'],observed={})


def test_registry_exact_source_metadata_is_proposal_only():
    from webapp.parser.handlers.registry import inspect_exact_source_route
    p=inspect_exact_source_route(registry_source_id='example-id',
       source_url='https://example.invalid/data.xlsx',catalog_format='CSV',
       catalog_state='Example State',catalog_scope='statewide')
    assert p['format_conflict_requires_review'] is True
    assert p['catalog_format_unchanged']=='CSV'
    assert p['route_activation_authorized'] is False
    assert p['precinct_granularity_verified'] is False


def _fake_enhanced_run(monkeypatch,*, optin: bool):
    import webapp.parser.handlers.shared.vendors.enhanced_voting as ev
    monkeypatch.setattr(ev,'scan_html_for_context',lambda **kw: {
        'state':'Example State','county':'Example County','year':'2024',
        'contests':[{'title':'Example Contest'}]})
    monkeypatch.setattr(ev,'select_contest_auto_first',lambda **kw: {'title':'Example Contest'})
    monkeypatch.setattr(ev,'run_navigation_recipe',lambda **kw: None)
    monkeypatch.setattr(ev,'robust_table_extraction',lambda **kw: {'headers':['Candidate'],'data':[]})
    class Coordinator:
        allow_duplicates=False
        def organize_and_enrich(self,*args,**kwargs): return args[0] if args else {}
    data={
       'state':'Example State','county':'Example County',
       'url':'https://example.invalid/elections',
    }
    if optin:
        data.update({'source_route_review_opt_in':True,'source_route_metadata':{
         'registry_source_id':'example-id',
         'source_url':'https://app.enhancedvoting.com/results/public/example',
         'catalog_format':'HTML','catalog_state':'Example State','catalog_scope':'county'}})
    return ev.parse_enhanced_voting(page=None,html_context=data,
                                     coordinator=Coordinator(),vendor='rockland')


def test_enhanced_voting_default_has_no_route_review(monkeypatch):
    result=_fake_enhanced_run(monkeypatch,optin=False)
    assert result[3]['vendor']=='rockland'
    assert 'source_route_review_only' not in result[3]


def test_enhanced_voting_review_does_not_change_vendor_or_rows(monkeypatch):
    result=_fake_enhanced_run(monkeypatch,optin=True)
    assert result[0]==['Candidate'] and result[1]==[]
    assert result[3]['vendor']=='rockland'
    assert result[3]['source_route_review_only']['vendor_hint_unverified']=='enhanced_voting'
    assert result[3]['source_route_review_only']['route_activation_authorized'] is False
