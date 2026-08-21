"""C2E 1.3 contracts for the canonical publication reader and frontend boundary."""

from __future__ import annotations

import inspect
from pathlib import Path

from sqlalchemy.orm import Session

from webapp.parser.services.canonical_election_reader import (
    CanonicalResultFilters,
    query_canonical_results,
)
from webapp.parser.utils.models import (
    CanonicalElectionRace,
    CanonicalElectionResult,
    CanonicalSourceArtifact,
    CanonicalVoteComponent,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_canonical_reader_preserves_zero_null_year_precision_and_signed_components(test_db_engine):
    with Session(test_db_engine) as session:
        payload = CanonicalSourceArtifact(
            artifact_role="payload",
            filename="c2e13-payload.xlsx",
            sha256="1" * 64,
            row_count=2,
            race_count=1,
            provenance={"test": "c2e13"},
        )
        approval = CanonicalSourceArtifact(
            artifact_role="approval",
            filename="c2e13-approval.xlsx",
            sha256="2" * 64,
            row_count=1,
            race_count=1,
            provenance={"test": "c2e13"},
        )
        session.add_all([payload, approval])
        session.flush()

        race = CanonicalElectionRace(
            source_race_id="C2E13-2024-ZZ-RACE",
            election_year=2024,
            election_date=None,
            date_precision="year",
            state="ZZ",
            contest="C2E 1.3 Contract Race",
            office_basic="Test Office",
            production_status="prod_loaded",
            selected_dl_source="DL1",
            source_url="https://example.invalid/c2e13",
            verification_status="verified",
            payload_artifact_id=payload.id,
            approval_artifact_id=approval.id,
            qa_metadata={"test": True},
        )
        session.add(race)
        session.flush()

        district_zero = CanonicalElectionResult(
            race_id=race.id,
            source_row_index=1,
            source_row_hash="a" * 64,
            source_jurisdiction_label="District 5",
            jurisdiction_key="ZZ|district|5",
            jurisdiction_name="District 5",
            jurisdiction_type="district",
            aggregation_scope="jurisdiction",
            precinct=None,
            ballot_candidate_name="Zero Candidate",
            candidate="Zero Candidate",
            ballot_party=None,
            party=None,
            fec_id=None,
            is_write_in=False,
            total_votes=0,
            source_url=None,
            provenance={"test": "zero-null"},
        )
        county_known = CanonicalElectionResult(
            race_id=race.id,
            source_row_index=2,
            source_row_hash="b" * 64,
            source_jurisdiction_label="Example County",
            jurisdiction_key="ZZ|county|example",
            jurisdiction_name="Example County",
            jurisdiction_type="county",
            aggregation_scope="jurisdiction",
            precinct=None,
            ballot_candidate_name="Known Candidate",
            candidate="Known Candidate",
            ballot_party="Example Party",
            party="EXP",
            fec_id=None,
            is_write_in=False,
            total_votes=12,
            source_url="https://example.invalid/result",
            provenance={"test": "known"},
        )
        session.add_all([district_zero, county_known])
        session.flush()

        session.add_all(
            [
                CanonicalVoteComponent(
                    result_id=district_zero.id,
                    vote_method="mail",
                    votes=-4,
                    source_column="Mail Votes",
                ),
                CanonicalVoteComponent(
                    result_id=district_zero.id,
                    vote_method="other",
                    votes=4,
                    source_column="Other Votes",
                ),
            ]
        )
        session.commit()

    rows = query_canonical_results(
        test_db_engine,
        CanonicalResultFilters(
            state="ZZ",
            year=2024,
            jurisdiction="District 5",
            limit=10,
        ),
        include_components=True,
    )

    assert len(rows) == 1
    row = rows[0]

    # A real zero remains zero.
    assert row["votes"] == 0
    assert row["total_votes"] == 0

    # Nullable evidence remains null; no exact date or precinct is invented.
    assert row["election_date"] is None
    assert row["precinct"] is None
    assert row["party"] is None

    # The year remains queryable even though exact election_date is NULL.
    assert row["year"] == 2024
    assert row["date_precision"] == "year"

    # A district must not be mislabeled as a county for compatibility.
    assert row["jurisdiction_name"] == "District 5"
    assert row["jurisdiction_type"] == "district"
    assert row["county"] is None

    # Signed source evidence must survive the publication adapter unchanged.
    assert row["vote_components"] == [
        {
            "vote_method": "mail",
            "votes": -4,
            "source_column": "Mail Votes",
        },
        {
            "vote_method": "other",
            "votes": 4,
            "source_column": "Other Votes",
        },
    ]

    county_rows = query_canonical_results(
        test_db_engine,
        CanonicalResultFilters(
            state="ZZ",
            year=2024,
            jurisdiction="Example County",
            limit=10,
        ),
    )
    assert county_rows[0]["county"] == "Example County"
    assert county_rows[0]["votes"] == 12


def test_canonical_reader_is_rollback_only_and_has_postgres_read_only_guard():
    source = inspect.getsource(query_canonical_results)

    assert "SET TRANSACTION READ ONLY" in source
    assert "transaction.rollback()" in source
    assert ".commit(" not in source


def test_canonical_endpoint_is_wired_without_warehouse_fallback():
    main_source = (REPO_ROOT / "webapp/Smart_Elections_Parser_Webapp.py").read_text(encoding="utf-8")
    route_source = (REPO_ROOT / "webapp/parser/routes/election_data_blueprint.py").read_text(encoding="utf-8")
    config_source = (REPO_ROOT / "webapp/parser/config.py").read_text(encoding="utf-8")

    assert "def api_ballotlens_database():" in main_source
    assert '"api_ballotlens_database": api_ballotlens_database' in main_source
    assert '"/api/ballotlens-database"' in route_source
    assert '"api_ballotlens_database"' in route_source
    assert 'DATA_API_URL = os.environ.get("DATA_API_URL", "/api/ballotlens-database")' in config_source

    handler_block = main_source.split("def api_ballotlens_database():", 1)[1].split(
        "def api_warehouse_election_results():",
        1,
    )[0]
    assert "FROM warehouse_election_results" not in handler_block
    assert "_get_warehouse_columns" not in handler_block
    assert "data_source != \"canonical\"" in handler_block


def test_frontend_uses_configured_canonical_api_and_preserves_null_semantics():
    data_framework = (REPO_ROOT / "webapp/static/js/data_framework.js").read_text(encoding="utf-8")
    ballot_lens = (REPO_ROOT / "webapp/static/js/ballot_lens_modern.js").read_text(encoding="utf-8")
    data_template = (REPO_ROOT / "webapp/templates/data_framework.html").read_text(encoding="utf-8")
    ballot_template = (REPO_ROOT / "webapp/templates/ballot_lens.html").read_text(encoding="utf-8")

    assert "'/api/ballotlens-database'" in data_framework
    assert "'/api/warehouse_election_results'" not in data_framework
    assert "'/api/ballotlens-database'" in ballot_lens
    assert "'/api/warehouse_election_results" not in ballot_lens

    assert 'data-api-url="{{ data_api_url }}"' in data_template
    assert 'id="ballotLensConfig"' in ballot_template
    assert 'data-api-url="{{ data_api_url }}"' in ballot_template

    # Null must not be normalized to zero at the ingestion boundary.
    assert "if (value == null || value === '') return null;" in data_framework
    assert "return Number.isFinite(num) ? num : null;" in data_framework
    assert "v == null ? 'NULL'" in data_framework

    # Known historical corruption patterns must stay removed.
    forbidden = (
        "parseNumeric(record['Total Votes'] ||",
        "votes: downVotes || presidentialVotes",
        "Number(row.votes || 0) || 0",
        "reduce((sum, val) => sum + val, 0) || 1",
    )
    for token in forbidden:
        assert token not in data_framework


def test_canonical_endpoint_preserves_principal_boundary_and_does_not_infer_null_reason():
    main_source = (REPO_ROOT / "webapp/Smart_Elections_Parser_Webapp.py").read_text(encoding="utf-8")
    handler_block = main_source.split("def api_ballotlens_database():", 1)[1].split(
        "def api_warehouse_election_results():",
        1,
    )[0]

    assert "principal, _, _ = get_request_principal()" in handler_block
    assert "if not principal and not ALLOW_DEV_NO_PRINCIPAL:" in handler_block
    assert 'return jsonify({"error": "Unauthorized"}), 403' in handler_block
    assert '"contract": "canonical_results_v1"' in handler_block
    assert '"null": "preserved_null"' in handler_block
    assert '"null_reason": "not_inferred"' in handler_block
    assert "unknown_or_not_present_at_this_granularity" not in handler_block
