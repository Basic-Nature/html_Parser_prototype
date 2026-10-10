"""J1 offline contract checks: no live DB, parser, remote or identity mutation."""
from __future__ import annotations
import os
import uuid
from datetime import date
import pytest
from sqlalchemy import Column, MetaData, Table, Uuid, String, Boolean, create_engine, event, select
from sqlalchemy.orm import Session
from webapp.parser.services import election_projects as svc
from webapp.parser.models.election_project import ElectionProject, ElectionProjectScope, ElectionProjectSourceRef
from webapp.parser.utils.models import Base, SourceRegistryBase

@pytest.fixture
def owner(monkeypatch):
    monkeypatch.setenv("ELECTION_PROJECT_OWNER_HMAC_KEY", "f" * 48)
    return svc.owner_key("cert:"+"a"*64)

@pytest.fixture
def db():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    @event.listens_for(engine, "connect")
    def fk_on(conn, _): conn.execute("PRAGMA foreign_keys=ON")
    # Registry is a separate authority/metadata collection. Use disposable
    # test-only tables in an independent MetaData; never mutate Base.metadata.
    registry_meta = MetaData()
    Table('source_registry_revisions', registry_meta,
      Column('id',Uuid(as_uuid=True),primary_key=True))
    Table('source_registry_bindings',registry_meta,
      Column('id',Uuid(as_uuid=True),primary_key=True),
      Column('current_revision_id',Uuid(as_uuid=True),nullable=False),
      Column('source_id',Uuid(as_uuid=True),nullable=False),
      Column('review_state',String(24),nullable=False),
      Column('public_eligible',Boolean,nullable=False),
      Column('year',String(16),nullable=False),
      Column('state',String(64),nullable=False),
      Column('contest',String(512),nullable=False),
      Column('scope',String(512),nullable=False),
      Column('format',String(64),nullable=False))
    registry_meta.create_all(engine)
    project_tables = [Base.metadata.tables[x] for x in
      ('election_projects', 'election_project_scopes', 'election_project_source_refs')]
    Base.metadata.create_all(engine, tables=project_tables)
    with Session(engine) as session: yield session
    engine.dispose()

def create_project(db,owner):
    obj=svc.create(db,owner,{"title":"AZ multi-contest audit", "idempotency_key":str(uuid.uuid4())})
    db.commit();return obj

def test_distinct_owner_isolation_and_idempotency(db,owner):
    key=str(uuid.uuid4())
    payload={"title":"Arizona audit", "idempotency_key":key}
    one=svc.create(db,owner,payload)
    assert svc.create(db,owner,payload)['id']==one['id']
    with pytest.raises(svc.ProjectError) as denied:
        svc.detail(db,uuid.UUID(one['id']),"other-owner")
    assert denied.value.status==404
    assert len(svc.projects(db,owner))==1
    with pytest.raises(svc.ProjectError):
        svc.create(db,owner,{**payload,"title":"Different"})

def test_multi_scope_updates_are_bounded_and_versioned(db,owner):
    obj=create_project(db,owner); pid=uuid.UUID(obj['id'])
    rows=[{"election_year":2024,"state_code":"AZ","contest":"President","granularity":"county"},
          {"election_year":2024,"state_code":"TX","contest":"Senate","granularity":"precinct"}]
    result=svc.replace_scopes(db,pid,owner,{"scopes":rows,"expected_version":1})
    assert result['row_version']==2 and len(result['scopes'])==2
    assert {r['state_code'] for r in result['scopes']}=={'AZ','TX'}
    with pytest.raises(svc.ProjectError) as err:
        svc.replace_scopes(db,pid,owner,{"scopes":rows,"expected_version":1})
    assert err.value.status==409
    with pytest.raises(svc.ProjectError):svc.validate_scopes([{"state_code":"Arizona"}])
    with pytest.raises(svc.ProjectError):svc.validate_scopes([{"election_year":2024,"election_date":"2025-11-05"}])
    with pytest.raises(svc.ProjectError):svc.validate_scopes([{"state_code":"AZ"}]*21)

def test_source_metadata_reference_no_execution_or_registry_write(db,owner):
    project=create_project(db,owner); pid=uuid.UUID(project['id']);binding=uuid.uuid4();revision=uuid.uuid4()
    db.execute(__import__('sqlalchemy').text("INSERT INTO source_registry_revisions(id) VALUES (:id)"),{'id':revision.hex})
    db.execute(__import__('sqlalchemy').text("INSERT INTO source_registry_bindings(id,current_revision_id,source_id,review_state,public_eligible,year,state,contest,scope,format) "
        "VALUES (:id,:rev,:sid,'approved',1,'2024','AZ','President','statewide','PDF')"),
        {'id':binding.hex,'rev':revision.hex,'sid':uuid.uuid4().hex})
    result=svc.add_source_ref(db,pid,owner,{"registry_binding_id":str(binding),"expected_version":1})
    assert result['row_version']==2 and result['source_refs'][0]['execution_authorized'] is False
    assert result['source_refs'][0]['canonical_publication'] is False
    assert svc.options(db)[0]['registry_binding_id']==str(binding)
    with pytest.raises(svc.ProjectError):
        svc.add_source_ref(db,pid,owner,{"registry_binding_id":str(uuid.uuid4()),"expected_version":2})
    assert len(svc.detail(db,pid,owner)['source_refs'])==1

def test_no_weak_owner_keys(monkeypatch):
    monkeypatch.delenv('ELECTION_PROJECT_OWNER_HMAC_KEY',raising=False)
    with pytest.raises(svc.ProjectError) as e:svc.owner_key('cert:some-id')
    assert e.value.status==503

def test_project_fixture_preserves_source_registry_metadata_isolation():
    names = {'source_registry_revisions', 'source_registry_bindings'}
    assert not (names & set(Base.metadata.tables))
    assert names <= set(SourceRegistryBase.metadata.tables)
    refs = Base.metadata.tables['election_project_source_refs']
    assert all(fk.column.table.name == 'election_projects' for fk in refs.foreign_keys)
