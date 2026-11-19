"""Synthetic security contract inputs only; no election preview or live source."""
from types import SimpleNamespace as NS
import concurrent.futures
import time
import threading
import pytest
from flask import Flask
from webapp.parser.services import source2_runtime_capabilities as scope
from webapp.parser.services import public_csv_delivery as csvs
from webapp.parser.services import public_compact_csv_handoff as handoff
from webapp.parser.services import public_compact_socket_adapter as adapter
from webapp.parser.services.public_compact_result_bridge import bind_compact_broker_to_runtime
from webapp.parser.services.public_contest_page_broker import PublicContestPageBroker
from webapp.parser.routes.file_io_blueprint import create_file_io_blueprint

MD,UT=scope.MD,scope.UTAH
FLAGS=(scope.LEGACY_COMPACT,scope.LEGACY_CSV,scope.MD_COMPACT,scope.MD_CSV)
PARSE='BALLOT_LENS_PUBLIC_REGISTRY_PARSE_ENABLED'
PRIMARY='BALLOT_LENS_PUBLIC_REGISTRY_PILOT_SOURCE_ID'
EXTRA='BALLOT_LENS_PUBLIC_REGISTRY_ADDITIONAL_SOURCE_IDS'
S='sess_contract_test_012345'; SID='socket_contract'; ORIGIN='https://example.invalid'
# Non-election contract sentinel, NOT extracted election data or preview evidence.
HEADERS=['Precinct','Grand Total']; ROWS=[{'Precinct':'TEST_SENTINEL','Grand Total':None}]

@pytest.fixture
def ctx(monkeypatch):
    for k in FLAGS:monkeypatch.delenv(k,raising=False)
    for k,v in {PARSE:'true',PRIMARY:UT,EXTRA:MD,'SOURCE_REGISTRY_AUTHORITY_MODE':'durable_db'}.items():monkeypatch.setenv(k,v)
    approved={MD,UT}
    def resolve(sid):
        return NS(registry_source_id=sid,registry_category='curated') if sid in approved else None
    monkeypatch.setattr(scope,'_resolve',resolve)
    broker=csvs.OneUseCsvBroker();monkeypatch.setattr(csvs,'BROKER',broker);monkeypatch.setattr(handoff,'BROKER',broker)
    staged=handoff.CompactCsvHandoff();monkeypatch.setattr(handoff,'STAGED_COMPACT_CSV',staged)
    pages=PublicContestPageBroker(b'x'*32);monkeypatch.setattr(adapter,'broker_instance',lambda:pages)
    monkeypatch.setattr(csvs,'allow_public_csv_request',lambda peer:True)
    app=Flask('source2-integration');app.register_blueprint(create_file_io_blueprint())
    return NS(monkeypatch=monkeypatch,approved=approved,broker=broker,staged=staged,pages=pages,client=app.test_client())

def flags(ctx,*names):
    for n in names:ctx.monkeypatch.setenv(n,'true')

def cap(s):return scope.capabilities_for_source(s)

def post(ctx,token,**kwargs):return ctx.client.post(csvs.CSV_ENDPOINT,json={'token':token},**kwargs)

@pytest.mark.parametrize('names,expected',[((),(False,False,False,False)),((scope.LEGACY_COMPACT,scope.LEGACY_CSV),(True,True,False,False)),((scope.MD_COMPACT,scope.MD_CSV),(False,False,True,True)),((scope.MD_CSV,),(False,False,False,False)),((scope.MD_COMPACT,),(False,False,True,False)),(FLAGS,(True,True,True,True))])
def test_source_isolation(ctx,names,expected):
    flags(ctx,*names);assert (cap(UT).compact,cap(UT).csv,cap(MD).compact,cap(MD).csv)==expected

@pytest.mark.parametrize('case',['revoked','parse_off','extra_removed','legacy_mode','resolver_error','primary_missing','wrong_category'])
def test_fail_closed(ctx,case):
    flags(ctx,scope.MD_COMPACT,scope.MD_CSV)
    if case=='revoked':ctx.approved.remove(MD)
    elif case=='parse_off':ctx.monkeypatch.setenv(PARSE,'false')
    elif case=='extra_removed':ctx.monkeypatch.delenv(EXTRA)
    elif case=='legacy_mode':ctx.monkeypatch.setenv('SOURCE_REGISTRY_AUTHORITY_MODE','legacy_file')
    elif case=='primary_missing':ctx.approved.remove(UT)
    elif case=='wrong_category':ctx.monkeypatch.setattr(scope,'_resolve',lambda s:NS(registry_source_id=s,registry_category='other'))
    else:
        def bad(s):raise RuntimeError('not public')
        ctx.monkeypatch.setattr(scope,'_resolve',bad)
    assert not cap(MD).compact and not cap(MD).csv

def test_http_source_recheck_and_headers(ctx):
    flags(ctx,scope.MD_COMPACT,scope.MD_CSV)
    t=ctx.broker.publish(source_id=MD,content=b'contract sentinel')['token']
    ctx.approved.remove(MD);assert post(ctx,t).status_code==404
    ctx.approved.add(MD);r=post(ctx,t)
    assert r.status_code==200 and r.data==b'contract sentinel'
    assert 'no-store' in r.headers['Cache-Control'] and 'private' in r.headers['Cache-Control']
    assert r.headers['X-Content-Type-Options']=='nosniff' and post(ctx,t).status_code==404

def test_http_utah_not_activated_by_md(ctx):
    flags(ctx,scope.MD_COMPACT,scope.MD_CSV)
    t=ctx.broker.publish(source_id=UT,content=b'sentinel')['token'];assert post(ctx,t).status_code==404
    flags(ctx,scope.LEGACY_CSV);assert post(ctx,t).status_code==200

def test_http_origin_reject_does_not_burn(ctx):
    flags(ctx,scope.LEGACY_CSV);t=ctx.broker.publish(source_id=UT,content=b'sentinel')['token']
    assert post(ctx,t,headers={'Origin':'https://evil.invalid'}).status_code==404
    assert post(ctx,t).status_code==200

def test_atomic_redemption(ctx):
    flags(ctx,scope.MD_COMPACT,scope.MD_CSV);t=ctx.broker.publish(source_id=MD,content=b'sentinel')['token']
    barrier=threading.Barrier(2)
    def auth(s):barrier.wait(timeout=3);return cap(s).csv
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        result=list(pool.map(lambda _:ctx.broker.take(t,authorize_source=auth),range(2)))
    assert result.count(b'sentinel')==1 and result.count(None)==1

def result_for(sid):
    return {'contract':'ballot_lens_public_runtime_result_v1','registry_source_id':sid,'persistent_output':False,'download_available':False,'terminal_status':'completed','outputs':[{'contract':'ballot_lens_public_memory_preview_v1','registry_source_id':sid,'output_mode':'MEMORY_PREVIEW_ONLY','persistent_output':False,'headers':HEADERS,'rows':ROWS,'row_count':len(ROWS)}]}

def test_md_direct_mint_forbidden(ctx):
    flags(ctx,*FLAGS);assert csvs.publish_from_public_result(result_for(MD)) is None
    assert csvs.publish_from_public_result(result_for(UT)) is not None

def mock_socket(ctx,sid=MD):
    mgr=NS(resolve_socket=lambda socket:S if socket==SID else None,get_metadata=lambda server:{'public_registry_session':True,'public_compact_registry_source_id':sid})
    peer=adapter.trusted_peer(manager=mgr,socket_sid=SID,source_id=sid,origin_allowed=True)
    runtime=NS(source_projection={'state':'Maryland','year':'2024','contest':'President'},registry_source_id=sid)
    runtime.enable_compact_capture=lambda capture:setattr(runtime,'capture',capture)
    bind_compact_broker_to_runtime(runtime,ctx.pages,peer)
    events=[]
    def action(which,data,socket=SID,origin=ORIGIN):
        adapter.process_action(action=which,data=data,manager=mgr,request=NS(sid=socket,headers={'Origin':origin}),allowed_origins=[ORIGIN],emit=lambda e,v,room:events.append((e,v,room)))
    return runtime,action,events

def capture(rt):return rt.capture(headers=HEADERS,rows=ROWS,source_sha256='a'*64,quarantine={'observations':0,'votes':0,'reason':None})

def test_actual_compact_bridge_socket_and_http(ctx):
    flags(ctx,scope.MD_COMPACT,scope.MD_CSV);rt,action,events=mock_socket(ctx);m=capture(rt);c=m['first_cursor']
    action('finish',{'last_cursor':c});assert events[-1][0]=='public_registry_compact_unavailable'
    action('next',{'cursor':c});assert events[-1][0]=='public_registry_compact_page'
    action('finish',{'last_cursor':c});assert [e[0] for e in events[-2:]]==['public_registry_compact_complete',csvs.CSV_EVENT]
    token=events[-1][1]['token'];r=post(ctx,token);assert r.status_code==200 and b'TEST_SENTINEL,' in r.data
    assert post(ctx,token).status_code==404

def test_compact_revocation_on_next(ctx):
    flags(ctx,scope.MD_COMPACT,scope.MD_CSV);rt,action,events=mock_socket(ctx);c=capture(rt)['first_cursor']
    ctx.approved.remove(MD);action('next',{'cursor':c});assert events[-1][0]=='public_registry_compact_unavailable'

def test_csv_disabled_after_stage(ctx):
    flags(ctx,scope.MD_COMPACT,scope.MD_CSV);rt,action,events=mock_socket(ctx);c=capture(rt)['first_cursor']
    action('next',{'cursor':c});ctx.monkeypatch.setenv(scope.MD_CSV,'false');action('finish',{'last_cursor':c})
    assert events[-1][0]=='public_registry_compact_complete' and all(e[0]!=csvs.CSV_EVENT for e in events)

def test_compact_revoked_before_capture(ctx):
    flags(ctx,scope.MD_COMPACT);rt,_,_=mock_socket(ctx);ctx.approved.remove(MD)
    with pytest.raises(RuntimeError,match='PUBLIC_COMPACT_AUTHORITY_UNAVAILABLE'):capture(rt)

def test_foreign_socket(ctx):
    flags(ctx,scope.MD_COMPACT);rt,action,events=mock_socket(ctx);c=capture(rt)['first_cursor']
    action('next',{'cursor':c},socket='other');assert events[-1][0]=='public_registry_compact_unavailable'

