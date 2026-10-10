"""J2A offline eligibility contract: no socket, network, parser or production DB."""
from __future__ import annotations
import uuid
from datetime import datetime, timezone
from pathlib import Path
import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session
from webapp.parser.models.election_project import ElectionProject, ElectionProjectSourceRef
from webapp.parser.services import project_run_preflight as svc
from webapp.parser.services.election_projects import ProjectError
from webapp.parser.utils.models import (
    Base, SourceRegistryBase, SourceRegistryBinding,
    SourceRegistryRevision, SourceRegistrySource,
)

@pytest.fixture
def example(monkeypatch):
    engine = create_engine("sqlite+pysqlite:///:memory:")
    @event.listens_for(engine, "connect")
    def enable_fk(dbapi, _):
        dbapi.execute("PRAGMA foreign_keys=ON")
    SourceRegistryBase.metadata.create_all(engine, tables=[
        SourceRegistrySource.__table__, SourceRegistryRevision.__table__,
        SourceRegistryBinding.__table__,
    ])
    Base.metadata.create_all(engine, tables=[
        Base.metadata.tables[x] for x in (
            "election_projects", "election_project_scopes", "election_project_source_refs",
        )
    ])
    now = datetime.now(timezone.utc)
    ids = {key: uuid.uuid4() for key in ("project", "ref", "source", "rev", "binding", "workflow")}
    url = "https://example.gov/elections/2024/results.csv"
    with Session(engine) as db:
        db.add(SourceRegistrySource(
            id=ids["source"], lifecycle_state="active", row_version=1,
            created_at=now, updated_at=now,
        ))
        db.flush()  # Persist the Registry source before its FK-bound revision.
        db.add(SourceRegistryRevision(
            id=ids["rev"], source_id=ids["source"], revision_number=1,
            exact_url=url, normalized_url=url, host="example.gov",
            url_sha256="a" * 64, created_at=now,
        ))
        db.flush()
        db.add(SourceRegistryBinding(
            id=ids["binding"], source_id=ids["source"],
            current_revision_id=ids["rev"], year="2024", state="AZ",
            contest="President", scope="Pima", format="CSV", notes="",
            review_state="approved", parser_eligible=True,
            public_eligible=True, workflow_eligible=True,
            row_version=1, created_at=now, updated_at=now,
        ))
        db.add(ElectionProject(
            id=ids["project"], owner_key="owner_1", creation_key=uuid.uuid4(),
            title="Arizona audit", description="", lifecycle="active",
            row_version=3, created_at=now, updated_at=now,
        ))
        db.flush()
        db.add(ElectionProjectSourceRef(
            id=ids["ref"], project_id=ids["project"],
            registry_binding_id=ids["binding"], registry_revision_id=ids["rev"],
            created_at=now,
        ))
        db.commit()
        monkeypatch.setattr(svc, "assert_workflow_runtime_capability",
            lambda principal, cap: frozenset({"workflow_contributor"}))
        monkeypatch.setattr(svc, "_workflow_snapshot",
            lambda *args, **kwargs: (url, 5))
        values = dict(session=db, project_id=ids["project"],
            source_ref_id=ids["ref"], workflow_item_id=ids["workflow"],
            expected_project_version=3, owner_key="owner_1",
            principal="cert:" + "a" * 64, registry_path=Path("urls.txt"))
        yield db, ids, values, url
    engine.dispose()

def test_eligible_preview_contains_no_execution_authority(example):
    _, ids, values, url = example
    result = svc.project_run_preflight(**values)
    assert result["contract"] == "project_run_preflight_v1"
    assert result["eligible_for_confirmation_review"] is True
    assert result["confirmation_enabled"] is False
    assert result["execution_authorized"] is False
    assert result["run_dispatched"] is False
    assert result["source_url_disclosed"] is False
    assert url not in repr(result)
    assert "workflow_pass_id" not in result
    assert "execution_mode" not in result
    assert "confirmation_token" not in result
    assert result["registry_revision_id"] == str(ids["rev"])

def test_cross_owner_and_project_version_fail_closed(example):
    _, ids, values, _ = example
    with pytest.raises(ProjectError) as err:
        svc.project_run_preflight(**{**values, "owner_key": "other"})
    assert err.value.status == 404
    with pytest.raises(ProjectError) as err:
        svc.project_run_preflight(**{**values, "expected_project_version": 2})
    assert err.value.status == 409

def test_foreign_source_reference_denied(example):
    _, ids, values, _ = example
    with pytest.raises(ProjectError) as err:
        svc.project_run_preflight(**{**values, "source_ref_id": uuid.uuid4()})
    assert err.value.status == 403

def test_registry_revision_and_eligibility_recheck(example):
    db, ids, values, _ = example
    binding = db.get(SourceRegistryBinding, ids["binding"])
    # Install a valid newer revision before changing the binding pointer.
    new_revision_id = uuid.uuid4()
    replacement = SourceRegistryRevision(
        id=new_revision_id, source_id=ids["source"],
        revision_number=2, exact_url="https://example.gov/new.csv",
        normalized_url="https://example.gov/new.csv", host="example.gov",
        url_sha256="b" * 64, created_at=datetime.now(timezone.utc),
    )
    db.add(replacement)
    db.flush()
    binding.current_revision_id = new_revision_id
    db.flush()
    with pytest.raises(ProjectError) as err:
        svc.project_run_preflight(**values)
    assert err.value.status == 403
    binding.current_revision_id = ids["rev"]
    binding.workflow_eligible = False
    db.flush()
    with pytest.raises(ProjectError):
        svc.project_run_preflight(**values)

def test_workflow_context_and_archive_fail_closed(example, monkeypatch):
    db, ids, values, _ = example
    monkeypatch.setattr(svc, "_workflow_snapshot", lambda *a, **k: ("https://elsewhere.gov/file", 5))
    with pytest.raises(ProjectError):
        svc.project_run_preflight(**values)
    project = db.get(ElectionProject, ids["project"])
    project.lifecycle = "archived"
    db.flush()
    with pytest.raises(ProjectError) as err:
        svc.project_run_preflight(**values)
    assert err.value.status == 409
