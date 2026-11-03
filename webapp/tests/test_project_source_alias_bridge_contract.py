"""W24PUBPROJ1F synthetic SQLite source-alias handoff tests.

No production DB, network, parser, file upload, election data or mutation.
The service module is loaded from the shadow candidate directory, not installed.
"""
from __future__ import annotations

import hashlib
import importlib.util
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from webapp.parser.models.election_project import ElectionProject, ElectionProjectSourceRef
from webapp.parser.services.election_projects import ProjectError
from webapp.parser.utils.models import (
    Base, SourceRegistryBase, SourceRegistryAlias, SourceRegistryBinding,
    SourceRegistryRevision, SourceRegistrySource,
)

MODULE_FILE = Path(__file__).resolve().parents[1] / 'parser' / 'services' / 'project_source_alias_bridge.py'
spec = importlib.util.spec_from_file_location('w24pubproj1f_shadow_bridge', MODULE_FILE)
assert spec and spec.loader
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)

ALIAS = 'blsrc_v1_' + '7' * 64
URL = 'https://example.invalid/fixture-only-presidential-results.json'
OWNER = 'a' * 64


@pytest.fixture
def case():
    engine = create_engine('sqlite+pysqlite:///:memory:')
    Base.metadata.create_all(engine, tables=[
        ElectionProject.__table__, ElectionProjectSourceRef.__table__,
    ])
    SourceRegistryBase.metadata.create_all(engine, tables=[
        SourceRegistrySource.__table__, SourceRegistryRevision.__table__,
        SourceRegistryBinding.__table__, SourceRegistryAlias.__table__,
    ])
    with Session(engine) as db:
        project = ElectionProject(id=uuid.uuid4(), owner_key=OWNER,
            creation_key=uuid.uuid4(), title='Synthetic project', description='',
            lifecycle='active', row_version=3)
        source = SourceRegistrySource(id=uuid.uuid4(), lifecycle_state='active', row_version=1)
        db.add_all([project, source]); db.flush()
        revision = SourceRegistryRevision(id=uuid.uuid4(), source_id=source.id,
            revision_number=1, exact_url=URL, normalized_url=URL, host='example.invalid',
            url_sha256=hashlib.sha256(URL.encode()).hexdigest())
        db.add(revision); db.flush()
        binding = SourceRegistryBinding(id=uuid.uuid4(), source_id=source.id,
            current_revision_id=revision.id, year='2024', state='AZ', contest='President',
            scope='statewide', format='JSON', notes='', review_state='approved',
            parser_eligible=True, public_eligible=True, workflow_eligible=True,
            row_version=1)
        db.add(binding); db.flush()
        ref = ElectionProjectSourceRef(id=uuid.uuid4(), project_id=project.id,
            registry_binding_id=binding.id, registry_revision_id=revision.id)
        alias = SourceRegistryAlias(id=uuid.uuid4(), binding_id=binding.id,
            alias_type='legacy_blsrc_v1', alias_value=ALIAS, active=True)
        db.add_all([ref,alias]);db.flush()
        pub = SimpleNamespace(registry_source_id=ALIAS, registry_category='curated',
            url=URL, year='2024', state='AZ', contest='President',
            registry_scope='statewide', registry_format='JSON')
        data = dict(db=db,project=project,ref=ref,binding=binding,revision=revision,
            source=source,alias=alias,pub=pub,owner=OWNER,version=3)
        yield data
    engine.dispose()


def qualify(case, **kw):
    def resolve(_path, _alias):
        return case['pub']
    args=dict(project_id=case['project'].id,source_ref_id=case['ref'].id,
        expected_project_version=case['version'],owner_key=case['owner'],
        registry_path=Path('no-network-registry-fixture.txt'),public_resolver=resolve)
    args.update(kw)
    return bridge.resolve_project_source_alias(case['db'],**args)


def denied(case, status=None, **kwargs):
    with pytest.raises(ProjectError) as err:
        qualify(case,**kwargs)
    if status is not None: assert err.value.status==status


def test_valid_readonly_handoff_no_secret(case):
    out=qualify(case)
    assert out['contract']=='project_source_alias_handoff_v1'
    assert out['registry_source_id']==ALIAS
    assert out['project_id']==str(case['project'].id)
    assert out['source_ref_id']==str(case['ref'].id)
    assert out['project_row_version']==3
    assert out['source_label']['state']=='AZ'
    assert out['execution_authorized'] is False
    assert out['run_dispatched'] is False
    assert out['source_url_disclosed'] is False
    assert URL not in str(out)
    assert 'url' not in out and 'token' not in out
    assert 'execution_source_id' not in out


def test_wrong_owner_is_404(case): denied(case,404,owner_key='b'*64)

def test_missing_project_is_404(case): denied(case,404,project_id=uuid.uuid4())

def test_wrong_project_ref_is_denied(case): denied(case,409,source_ref_id=uuid.uuid4())

def test_bad_owner_key(case): denied(case,409,owner_key='short')

def test_archived_project(case):
    case['project'].lifecycle='archived';case['db'].flush(); denied(case,409)

def test_stale_project_version(case): denied(case,409,expected_project_version=2)

def test_boolean_project_version(case): denied(case,400,expected_project_version=True)

def test_missing_binding(case):
    case['ref'].registry_binding_id=uuid.uuid4();case['db'].flush();denied(case,409)

def test_missing_revision(case):
    case['ref'].registry_revision_id=uuid.uuid4();case['db'].flush();denied(case,409)

def test_rotated_current_revision(case):
    # Duplicate revision is unnecessary for negative query: mismatched UUID is denied.
    case['binding'].current_revision_id=uuid.uuid4();case['db'].flush();denied(case,409)

def test_revision_foreign_source(case):
    case['revision'].source_id=uuid.uuid4();case['db'].flush();denied(case,409)

def test_source_not_active(case):
    case['source'].lifecycle_state='quarantined';case['db'].flush();denied(case,409)

def test_binding_quarantined(case):
    case['binding'].review_state='quarantined'
    case['binding'].parser_eligible=False;case['binding'].public_eligible=False
    case['binding'].workflow_eligible=False;case['db'].flush();denied(case,409)

def test_binding_deprecated(case):
    case['binding'].review_state='deprecated'
    case['binding'].parser_eligible=False;case['binding'].public_eligible=False
    case['binding'].workflow_eligible=False;case['db'].flush();denied(case,409)

def test_not_public_eligible(case):
    case['binding'].public_eligible=False;case['db'].flush();denied(case,409)

def test_not_parser_eligible(case):
    case['binding'].public_eligible=False;case['binding'].parser_eligible=False
    case['binding'].workflow_eligible=False;case['db'].flush();denied(case,409)

def test_inactive_alias(case):
    case['alias'].active=False;case['db'].flush();denied(case,409)

def test_missing_legacy_alias_v2_only(case):
    case['alias'].alias_type='stable_blsrc_v2';case['db'].flush();denied(case,409)

def test_two_active_legacy_aliases(case):
    case['db'].add(SourceRegistryAlias(id=uuid.uuid4(), binding_id=case['binding'].id,
        alias_type='legacy_blsrc_v1',alias_value='blsrc_v1_'+'8'*64,active=True))
    case['db'].flush();denied(case,409)

def test_malformed_legacy_alias(case):
    case['alias'].alias_value='malformed';case['db'].flush();denied(case,409)

def test_runtime_public_projection_missing(case):
    denied(case,409,public_resolver=lambda p,a:None)

def test_runtime_registry_exception_fail_closed(case):
    def fail(*_): raise RuntimeError('synthetic runtime outage')
    denied(case,409,public_resolver=fail)

def test_runtime_url_mismatch(case):
    case['pub'].url='https://example.invalid/other-source';denied(case,409)

def test_runtime_alias_mismatch(case):
    case['pub'].registry_source_id='blsrc_v1_'+'9'*64;denied(case,409)

def test_runtime_unapproved_category(case):
    case['pub'].registry_category='unreviewed';denied(case,409)

def test_runtime_state_mismatch(case):
    case['pub'].state='TX';denied(case,409)

def test_runtime_scope_mismatch(case):
    case['pub'].registry_scope='county';denied(case,409)

def test_foreign_project_reference_not_exposed(case):
    other=ElectionProject(id=uuid.uuid4(),owner_key='b'*64,
        creation_key=uuid.uuid4(),title='Synthetic other',description='',
        lifecycle='active',row_version=1)
    case['db'].add(other);case['db'].flush()
    case['ref'].project_id=other.id;case['db'].flush()
    denied(case,409)
