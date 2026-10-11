"""J2B1NH-A focused regression tests: canonical-only coverage projection."""
from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.pool import StaticPool

from webapp.parser.services.canonical_coverage_reader import CoverageFilters, query_canonical_coverage
from webapp.parser.utils.models import (
    Base, CanonicalElectionRace, CanonicalElectionResult, CanonicalSourceArtifact,
)


def _fixture():
    engine = create_engine("sqlite+pysqlite:///:memory:", poolclass=StaticPool)
    Base.metadata.create_all(engine, tables=[
        CanonicalSourceArtifact.__table__, CanonicalElectionRace.__table__,
        CanonicalElectionResult.__table__,
    ])
    from sqlalchemy.orm import Session
    with Session(engine) as session:
        artifact = CanonicalSourceArtifact(id=uuid4(), artifact_role="seed", filename="private.csv", sha256="a"*64, provenance={})
        session.add(artifact)
        for idx, state in enumerate(("TX", "TX", "AZ")):
            race = CanonicalElectionRace(
                id=uuid4(), source_race_id=f"sr-{idx}", election_year=2024,
                state=state, contest="President", selected_dl_source="DL1" if idx != 1 else "DL2",
                payload_artifact_id=artifact.id, approval_artifact_id=artifact.id,
                verification_status="approved", qa_metadata={}, source_url="https://private.invalid/source",
            )
            session.add(race)
            session.flush()
            for n, j in enumerate(("County A", "County A", "County B")):
                session.add(CanonicalElectionResult(
                    id=uuid4(), race_id=race.id, source_row_index=n,
                    source_row_hash=f"{idx}{n}".ljust(64,"0"), source_jurisdiction_label=j,
                    jurisdiction_key=j, jurisdiction_name=j, jurisdiction_type="county",
                    aggregation_scope="jurisdiction", candidate=f"Candidate {n}",
                    total_votes=0 if n == 0 else 11,
                ))
        session.commit()
    return engine


def test_exact_race_grouping_and_unlinked_public_projection():
    engine = _fixture()
    value = query_canonical_coverage(engine, CoverageFilters(state="TX", year=2024, limit=2))
    assert value["contract"] == "canonical_coverage_v1"
    assert value["pagination"]["total_races"] == 2
    assert len({v["race_id"] for v in value["items"]}) == 2
    assert all(v["result_count"] == 3 and v["jurisdiction_count"] == 2 for v in value["items"])
    assert all(v["jurisdiction_scope_rows"] == 3 and v["precinct_scope_rows"] == 0 for v in value["items"])
    assert {v["selected_dl_source"] for v in value["items"]} == {"DL1", "DL2"}
    assert all(v["official_source_relationship"] == "not_established" for v in value["items"])
    assert "source_url" not in str(value)
    assert "private.csv" not in str(value)
    assert value["semantic_contract"]["selected_dl_lane"] == "chosen_payload_not_two_observer_agreement"


def test_exact_pagination_and_absence_not_zero_votes():
    engine = _fixture()
    page0 = query_canonical_coverage(engine, CoverageFilters(state="TX", limit=1, offset=0))
    page1 = query_canonical_coverage(engine, CoverageFilters(state="TX", limit=1, offset=1))
    assert page0["pagination"]["has_more"] is True
    assert page1["pagination"]["has_more"] is False
    assert page0["items"][0]["race_id"] != page1["items"][0]["race_id"]
    absent = query_canonical_coverage(engine, CoverageFilters(state="NY", year=2024))
    assert absent["items"] == []
    assert absent["pagination"]["total_races"] == 0
    assert absent["semantic_contract"]["missing_coverage"] == "not_published_in_this_view"


@pytest.mark.parametrize("limit,offset", [(0,0),(51,0),(1,-1),(1,5001)])
def test_pagination_guards(limit,offset):
    with pytest.raises(ValueError):
        query_canonical_coverage(_fixture(), CoverageFilters(limit=limit,offset=offset))


def test_public_coverage_capability_is_narrow():
    from webapp.parser.auth.capability_policy import (
        assert_public_read_surface, Capability, CapabilityPolicyError,
    )
    assert assert_public_read_surface("data_framework_canonical_coverage", "GET") == Capability.PUBLIC_READ
    with pytest.raises(CapabilityPolicyError):
        assert_public_read_surface("data_framework_canonical_coverage", "POST")
    with pytest.raises(CapabilityPolicyError):
        assert_public_read_surface("data_framework_canonical_coverage_delete", "GET")


def test_public_projection_contains_no_source_registry_or_artifact_identity():
    value = query_canonical_coverage(_fixture(), CoverageFilters(limit=5))
    forbidden = ("source_url", "registry_source_id", "payload_artifact_id",
                 "approval_artifact_id", "actor", "notes", "qa_metadata",
                 "parser_recipe", "cookie", "source_row_hash")
    for race in value["items"]:
        assert not (set(race) & set(forbidden))
