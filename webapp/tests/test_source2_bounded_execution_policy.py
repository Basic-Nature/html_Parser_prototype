import pytest
from webapp.parser.auth.capability_policy import Capability, CapabilityPolicyError
from webapp.parser.services.public_ballot_lens_policy import (
    PUBLIC_REGISTRY_ADDITIONAL_SOURCE_ENV as EXTRA,
    PUBLIC_REGISTRY_PARSE_ENV as ENABLED,
    PUBLIC_REGISTRY_PILOT_SOURCE_ENV as PRIMARY,
    PublicBallotLensPolicyError, authorize_public_registry_parse,
    configured_public_registry_execution_source_ids as configured,
    projected_public_registry_execution_source_ids as projected,
)
from webapp.parser.utils.url_registry import public_registry_source_id_for_entry, normalize_url_for_match
import json, hashlib
from pathlib import Path

UT = 'blsrc_v1_267d575a499b910deacc0c4ee660e696aeeab9c162126fe17625239be15f615c'
MD = 'blsrc_v1_fd0d8be9659f0c22e4c03cca55b52742aa809ae8650729e63bc186235cb64837'
OTHER = 'blsrc_v1_' + 'a'*64
BASE = {PRIMARY:UT, ENABLED:'true', 'SOURCE_REGISTRY_AUTHORITY_MODE':'durable_db', 'BALLOT_LENS_PUBLIC_MARYLAND_COMPACT_PAGES_ENABLED':'true'}

def authorize(sid, env, resolved=True):
    return authorize_public_registry_parse({'registry_source_id':sid}, registry_source_resolved=resolved, environ=env)

def test_legacy_pilot():
    assert configured(BASE)==(UT,)
    assert authorize(UT,BASE)==(Capability.BALLOT_LENS_PUBLIC_REGISTRY_PARSE, UT)
    with pytest.raises(PublicBallotLensPolicyError): authorize(MD,BASE)

def test_two_exact_ids():
    env={**BASE,EXTRA:MD}
    assert configured(env)==(UT,MD)
    for sid in (UT,MD): assert authorize(sid,env)[1]==sid
    with pytest.raises(PublicBallotLensPolicyError): authorize(OTHER,env)

@pytest.mark.parametrize('extra',['','*','https://example.invalid/',UT,MD+','+OTHER,MD+' '+OTHER,MD.upper(),'x'*4096])
def test_invalid_extra_never_grants_authority(extra):
    env={**BASE,EXTRA:extra}
    assert configured(env)==(UT,)
    assert authorize(UT,env)[1]==UT
    with pytest.raises(PublicBallotLensPolicyError): authorize(MD,env)

@pytest.mark.parametrize('primary',['','*','https://example.invalid/'])
def test_no_valid_primary_no_expansion(primary):
    env={**BASE,PRIMARY:primary,EXTRA:MD}
    assert configured(env)==()
    with pytest.raises(PublicBallotLensPolicyError): authorize(MD,env)

@pytest.mark.parametrize('sid',[UT,MD])
def test_global_disable_and_unresolved_deny(sid):
    env={**BASE,EXTRA:MD}
    with pytest.raises(CapabilityPolicyError): authorize(sid,{**env,ENABLED:'false'})
    with pytest.raises(CapabilityPolicyError): authorize(sid,env,False)

def test_projection_intersection_order_and_disable():
    env={**BASE,EXTRA:MD}
    assert projected({UT,MD,OTHER},env)==(UT,MD)
    assert projected({UT,OTHER},env)==(UT,)
    assert projected({MD,OTHER},env)==()
    assert projected({OTHER},env)==()
    assert projected({UT,MD},{**env,ENABLED:'false'})==()

def test_browser_url_and_extra_keys_still_denied():
    env={**BASE,EXTRA:MD}
    for payload in ({'registry_source_id':'https://example.invalid/'}, {'registry_source_id':MD,'url':'https://example.invalid/'}, {'registry_source_id':MD,'session_id':'client'}, {'registry_source_id':[UT,MD]}):
        with pytest.raises(PublicBallotLensPolicyError):
            authorize_public_registry_parse(payload,registry_source_resolved=True,environ=env)

def test_reviewed_maryland_identity():
    p=Path(__file__).resolve().parent/'fixtures'/'source2_maryland_enrollment_identity.json'
    draft=json.loads(p.read_text(encoding='utf-8'))
    row=draft['exact_source_fields']
    # This is an ephemeral identity derivation fixture, not an approved database row.
    entry={**row,'section':'Curated','parser_eligible':True,'review_status':'approved'}
    assert public_registry_source_id_for_entry(entry)==MD
    assert normalize_url_for_match(row['url'])==draft['normalized_url']
    assert hashlib.sha256(row['url'].encode()).hexdigest()==draft['exact_url_sha256']
    assert public_registry_source_id_for_entry({**entry,'parser_eligible':False}) is None

@pytest.mark.parametrize('overrides', [
    {EXTRA:OTHER}, {'SOURCE_REGISTRY_AUTHORITY_MODE':'legacy_file'},
    {'BALLOT_LENS_PUBLIC_MARYLAND_COMPACT_PAGES_ENABLED':'false'},
    {PRIMARY:OTHER},
])
def test_extra_requires_reviewed_pair_durable_authority_and_compact(overrides):
    env={**BASE,EXTRA:MD,**overrides}
    assert configured(env)==(env[PRIMARY],)
    with pytest.raises(PublicBallotLensPolicyError): authorize(MD,env)
