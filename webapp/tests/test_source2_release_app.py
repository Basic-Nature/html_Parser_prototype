"""Actual Flask app contract and real cached-source HTTP replay.
Fixture source metadata and authority are test-only; no production connection.
"""
from pathlib import Path
from types import SimpleNamespace as NS
import hashlib, json, os, pytest
from webapp.parser.services import source_registry_runtime as registry
from webapp.parser.services import source2_runtime_capabilities as scope
from webapp.parser.services import public_csv_delivery as csvs
from webapp.parser.services import public_compact_csv_handoff as handoff
from webapp.parser.services import public_compact_socket_adapter as adapter
from webapp.parser.services.public_compact_result_bridge import bind_compact_broker_to_runtime
from webapp.parser.services.public_contest_page_broker import PublicContestPageBroker

ROOT=Path(__file__).resolve().parents[2]
MD,UT=scope.MD,scope.UTAH
ENDPOINT='/api/public/ballot-lens/registry'
FIXTURE=Path(__file__).parent/'fixtures'/'source2_maryland_enrollment_identity.json'

def projection(sid, entry):
    return {'registry_source_id':sid,'registry_category':'curated',
            **{k:entry[k] for k in ('year','contest','state','scope','format')}}

@pytest.fixture
def release_app(monkeypatch):
    from webapp import Smart_Elections_Parser_Webapp as mod
    # Source identity metadata, not fabricated election votes. Primary is pulled
    # from committed legacy registry so actual ID and fields match production pilot.
    monkeypatch.setenv('SOURCE_REGISTRY_AUTHORITY_MODE','legacy_file')
    primary=next(x for x in registry.project_public_registry_sources(mod.URL_LIST_FILE) if x['registry_source_id']==UT)
    md=projection(MD,json.loads(FIXTURE.read_bytes())['exact_source_fields'])
    approved={UT:primary,MD:md}
    monkeypatch.setattr(registry,'project_public_registry_sources',lambda _:list(approved.values()))
    monkeypatch.setattr(registry,'resolve_public_registry_source',lambda _,sid:NS(registry_source_id=sid,registry_category='curated') if sid in approved else None)
    for k in (scope.LEGACY_COMPACT,scope.LEGACY_CSV,scope.MD_COMPACT,scope.MD_CSV,'BALLOT_LENS_PUBLIC_REGISTRY_ADDITIONAL_SOURCE_IDS','SOURCE_REGISTRY_MUTATIONS_ENABLED'):
        monkeypatch.delenv(k,raising=False)
    monkeypatch.setenv('BALLOT_LENS_PUBLIC_REGISTRY_PARSE_ENABLED','true')
    monkeypatch.setenv('BALLOT_LENS_PUBLIC_REGISTRY_PILOT_SOURCE_ID',UT)
    monkeypatch.setenv('SOURCE_REGISTRY_AUTHORITY_MODE','durable_db')
    monkeypatch.setitem(mod.app.config,'TESTING',True)
    with mod.app.test_client() as client:
        yield NS(client=client,approved=approved,mp=monkeypatch,mod=mod)

def activate_md(c):
    c.mp.setenv('BALLOT_LENS_PUBLIC_REGISTRY_ADDITIONAL_SOURCE_IDS',MD)
    c.mp.setenv(scope.MD_COMPACT,'true')
    c.mp.setenv(scope.MD_CSV,'true')

def test_actual_app_default_off_v1_v2(release_app):
    c=release_app
    a=c.client.get(ENDPOINT); b=c.client.get(ENDPOINT+'?contract=v2')
    assert a.status_code==b.status_code==200
    assert a.json['contract']=='ballot_lens_public_registry_v1' and a.json['execution_source_id']==UT
    assert b.json['contract']=='ballot_lens_public_registry_v2' and b.json['execution_source_ids']==[UT]
    assert 'execution_source_id' not in b.json and a.json['sources']==b.json['sources']
    assert all(set(x)=={'registry_source_id','registry_category','year','contest','state','scope','format'} for x in b.json['sources'])
    assert 'no-store' in b.headers['Cache-Control']
    assert c.client.get(ENDPOINT+'?contract=v3').status_code==400
    assert c.client.get(ENDPOINT+'?contract=v1&contract=v2').status_code==400
    assert c.client.post(ENDPOINT).status_code==405
    # No mutation endpoints added to this release, even with governance flag on.
    c.mp.setenv('SOURCE_REGISTRY_MUTATIONS_ENABLED','true')
    for path in ('/api/source-registry/v1/proposals','/api/source-registry/v1/proposals/invalid/review','/api/source-registry/v1/proposals/invalid/publish'):
        assert c.client.post(path,json={}).status_code in (404,405,503)

def test_actual_app_expansion_authority_revocation(release_app):
    c=release_app; activate_md(c)
    assert c.client.get(ENDPOINT+'?contract=v2').json['execution_source_ids']==[UT,MD]
    assert c.client.get(ENDPOINT).json['execution_source_id']==UT
    c.mp.setenv(scope.MD_COMPACT,'false')
    assert c.client.get(ENDPOINT+'?contract=v2').json['execution_source_ids']==[UT]
    c.mp.setenv(scope.MD_COMPACT,'true')
    c.approved.pop(MD)
    assert c.client.get(ENDPOINT+'?contract=v2').json['execution_source_ids']==[UT]
    c.approved.pop(UT)
    assert c.client.get(ENDPOINT+'?contract=v2').json['execution_source_ids']==[]

def test_actual_socket_admission_denies_missing_or_unavailable_primary(release_app):
    from webapp.parser import socket_ballot_lens_orchestration as sock
    c=release_app;activate_md(c);c.approved.pop(UT)
    denied=[]
    c.mp.setattr(sock,'_emit_public_registry_authority_status',lambda *a,**k:denied.append(k.get('reason_code')))
    def unexpected(*a,**k):pytest.fail('Socket session admission must not be reached')
    c.mp.setattr(sock,'_public_client_rate_key',unexpected)
    hooks={'logger':NS(warning=lambda *a,**k:None)}
    assert sock._initialize_public_registry_authority({'registry_source_id':MD},hooks) is None
    assert denied==['public_registry_authority_denied']
    def unhealthy(_,sid):
        if sid==UT:raise RuntimeError('test dependency outage')
        return NS(registry_source_id=sid,registry_category='curated')
    c.mp.setattr(registry,'resolve_public_registry_source',unhealthy)
    assert sock._initialize_public_registry_authority({'registry_source_id':MD},hooks) is None
    assert denied==['public_registry_authority_denied']*2

def test_real_cached_maryland_full_app_csv(release_app):
    c=release_app
    path=os.environ.get('EP_SOURCE2_MD_FIXTURE')
    if not path: pytest.skip('Exact official cached Maryland source unavailable; real-source acceptance pending')
    data=Path(path).read_bytes()
    assert hashlib.sha256(data).hexdigest()=='fa25069f8504691c03528c70c2a76240b06514ab54b491fd624229867018aacc'
    from webapp.parser.services.public_wide_precinct_bridge import parse_reviewed_wide_csv
    profile=(ROOT/'webapp/parser/services/reviewed_source_profiles/md2024_general_president.json').read_bytes()
    assert hashlib.sha256(profile).hexdigest()=='13b35e05ce17af2cb761e06ab08c6b56e50992570031787a9f31cfd0effc9162'
    r=parse_reviewed_wide_csv(data,json.loads(profile));assert len(r.rows)==1959 and len(r.headers)==220
    assert r.audit['derived_candidate_vote_sum']==3038334 and r.audit['candidate_count']==34
    assert sum(v is None for row in r.rows for v in row.values())==9795
    assert sum(type(v) is int and v==0 for row in r.rows for v in row.values())==341964
    activate_md(c)
    broker=csvs.OneUseCsvBroker(); c.mp.setattr(csvs,'BROKER',broker); c.mp.setattr(handoff,'BROKER',broker)
    from collections import OrderedDict
    c.mp.setattr(csvs,'_REQUEST_TIMES',OrderedDict())
    staged=handoff.CompactCsvHandoff();c.mp.setattr(handoff,'STAGED_COMPACT_CSV',staged)
    pages=PublicContestPageBroker(b'x'*32);c.mp.setattr(adapter,'broker_instance',lambda:pages)
    sid='sess_release_real_fixture_012345';socket='release_real_socket';origin='https://example.invalid'
    mgr=NS(resolve_socket=lambda s:sid if s==socket else None,get_metadata=lambda s:{'public_registry_session':True,'public_compact_registry_source_id':MD})
    peer=adapter.trusted_peer(manager=mgr,socket_sid=socket,source_id=MD,origin_allowed=True)
    rt=NS(source_projection={'state':'Maryland','year':'2024','contest':'President'},registry_source_id=MD)
    rt.enable_compact_capture=lambda f:setattr(rt,'capture',f)
    bind_compact_broker_to_runtime(rt,pages,peer)
    manifest=rt.capture(headers=r.headers,rows=r.rows,source_sha256=hashlib.sha256(data).hexdigest(),quarantine={'observations':0,'votes':0,'reason':None})
    results=[]
    def action(verb,body):
        adapter.process_action(action=verb,data=body,manager=mgr,request=NS(sid=socket,headers={'Origin':origin}),allowed_origins=[origin],emit=lambda e,v,room:results.append((e,v)))
    cursor=manifest['first_cursor'];last=None;matrix=[]
    assert manifest['manifest']['headers']==r.headers and manifest['manifest']['rows']==len(r.rows)
    # The actual broker contract decides completion. Verify all returned page data
    # independently against the source-native matrix before acknowledgement.
    for _ in range(100):
        action('next',{'cursor':cursor}); name,page=results[-1]
        assert name=='public_registry_compact_page',name
        frame=page['frame'];assert frame['p']==_
        matrix.extend(frame['d'])
        last=cursor
        cursor=frame['next']
        if cursor is None: break
    else: pytest.fail('Compact page count exceeded budget')
    assert len(matrix)==len(r.rows) and matrix==[[row[h] for h in r.headers] for row in r.rows]
    assert manifest['manifest']['page_count']==_+1==30
    action('finish',{'last_cursor':last});assert results[-1][0]==csvs.CSV_EVENT
    token=results[-1][1]['token']
    response=c.client.post(csvs.CSV_ENDPOINT,json={'token':token});assert response.status_code==200
    expected=csvs.encode_finalized_preview({'contract':'ballot_lens_public_memory_preview_v1','registry_source_id':MD,'output_mode':'MEMORY_PREVIEW_ONLY','persistent_output':False,'headers':r.headers,'rows':r.rows,'row_count':len(r.rows)},source_id=MD)
    assert response.data==expected
    assert hashlib.sha256(response.data).hexdigest()=='5d06f992bcc332975ab54c818b26a1325bace2a5efcf596439c4d3ebf87b3f1d'
    assert {'no-store','private','max-age=0'}<={x.strip().lower() for x in response.headers['Cache-Control'].split(',')}
    assert response.headers['X-Content-Type-Options']=='nosniff'
    assert c.client.post(csvs.CSV_ENDPOINT,json={'token':token}).status_code==404
