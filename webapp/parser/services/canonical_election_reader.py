"""Read-only canonical election publication adapter.

C2E 1.3 contract:
- canonical_* tables are authoritative production reads
- warehouse_election_results is not a fallback
- NULL is preserved as NULL
- numeric zero remains numeric zero
- exact election dates are never synthesized from year-only evidence
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.sql import Select
from sqlalchemy.engine import Engine

from webapp.parser.utils.models import (
    CanonicalElectionRace,
    CanonicalElectionResult,
    CanonicalVoteComponent,
)


@dataclass(frozen=True)
class CanonicalResultFilters:
    """Validated read filters accepted by the canonical publication endpoint."""

    state: str | None = None
    year: int | None = None
    jurisdiction: str | None = None
    contest: str | None = None
    candidate: str | None = None
    party: str | None = None
    aggregation_scope: str | None = None
    limit: int = 500


@dataclass(frozen=True)
class CanonicalFacetFilters:
    """Scope filters for self-excluding canonical workbench facets."""

    state: str | None = None
    year: int | None = None
    jurisdiction: str | None = None
    contest: str | None = None


def _iso_or_none(value: date | datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _county_compatibility_value(
    jurisdiction_name: str | None,
    jurisdiction_type: str | None,
) -> str | None:
    """Populate legacy `county` only when canonical evidence says county."""

    if not jurisdiction_name or not jurisdiction_type:
        return None
    if jurisdiction_type.strip().lower() != "county":
        return None
    return jurisdiction_name


def _serialize_result(row: Any) -> dict[str, Any]:
    """Map a canonical join row to the stable publication JSON contract."""

    jurisdiction_name = row.jurisdiction_name
    jurisdiction_type = row.jurisdiction_type
    total_votes = row.total_votes

    return {
        "id": str(row.result_id),
        "race_id": str(row.race_id),
        "source_race_id": row.source_race_id,
        "source_row_index": row.source_row_index,
        "source_row_hash": row.source_row_hash,
        "state": row.state,
        "year": row.election_year,
        "election_year": row.election_year,
        "election_date": _iso_or_none(row.election_date),
        "date_precision": row.date_precision,
        "contest": row.contest,
        "office_basic": row.office_basic,
        "jurisdiction_key": row.jurisdiction_key,
        "jurisdiction_name": jurisdiction_name,
        "jurisdiction_type": jurisdiction_type,
        "source_jurisdiction_label": row.source_jurisdiction_label,
        "county": _county_compatibility_value(
            jurisdiction_name,
            jurisdiction_type,
        ),
        "aggregation_scope": row.aggregation_scope,
        "precinct": row.precinct,
        "candidate": row.candidate,
        "ballot_candidate_name": row.ballot_candidate_name,
        "party": row.party,
        "ballot_party": row.ballot_party,
        "fec_id": row.fec_id,
        "is_write_in": bool(row.is_write_in),
        # Compatibility alias for current UI consumers. It is intentionally an
        # exact alias, not `or 0`, so zero remains zero and future NULL remains NULL.
        "votes": total_votes,
        "total_votes": total_votes,
        "source_url": (
            row.result_source_url
            if row.result_source_url is not None
            else row.race_source_url
        ),
        "verification_status": row.verification_status,
        "verified_at": _iso_or_none(row.verified_at),
        "production_status": row.production_status,
        "selected_dl_source": row.selected_dl_source,
        "data_source": "canonical",
        "format": "canonical",
    }


def _build_result_statement(filters: CanonicalResultFilters) -> Select:
    result = CanonicalElectionResult
    race = CanonicalElectionRace

    stmt = (
        select(
            result.id.label("result_id"),
            result.race_id.label("race_id"),
            result.source_row_index,
            result.source_row_hash,
            result.source_jurisdiction_label,
            result.jurisdiction_key,
            result.jurisdiction_name,
            result.jurisdiction_type,
            result.aggregation_scope,
            result.precinct,
            result.ballot_candidate_name,
            result.candidate,
            result.ballot_party,
            result.party,
            result.fec_id,
            result.is_write_in,
            result.total_votes,
            result.source_url.label("result_source_url"),
            race.source_race_id,
            race.election_year,
            race.election_date,
            race.date_precision,
            race.state,
            race.contest,
            race.office_basic,
            race.source_url.label("race_source_url"),
            race.verification_status,
            race.verified_at,
            race.production_status,
            race.selected_dl_source,
        )
        .join(race, result.race_id == race.id)
    )

    if filters.state:
        stmt = stmt.where(func.lower(race.state) == filters.state.strip().lower())
    if filters.year is not None:
        # Intentionally use election_year. Year-only canonical races may have
        # election_date=NULL and must still remain queryable.
        stmt = stmt.where(race.election_year == int(filters.year))
    if filters.jurisdiction:
        stmt = stmt.where(
            func.lower(result.jurisdiction_name)
            == filters.jurisdiction.strip().lower()
        )
    if filters.contest:
        stmt = stmt.where(race.contest.ilike(f"%{filters.contest.strip()}%"))
    if filters.candidate:
        stmt = stmt.where(result.candidate.ilike(f"%{filters.candidate.strip()}%"))
    if filters.party:
        stmt = stmt.where(result.party.ilike(f"%{filters.party.strip()}%"))
    if filters.aggregation_scope:
        stmt = stmt.where(result.aggregation_scope == filters.aggregation_scope)

    return stmt.order_by(
        race.election_year.desc(),
        race.state.asc(),
        race.contest.asc(),
        result.jurisdiction_key.asc(),
        result.source_row_index.asc(),
    ).limit(max(1, min(1000, int(filters.limit))))


def _attach_components(
    conn: Any,
    items: list[dict[str, Any]],
) -> None:
    if not items:
        return

    result_ids = [UUID(item["id"]) for item in items]
    stmt = (
        select(
            CanonicalVoteComponent.result_id,
            CanonicalVoteComponent.vote_method,
            CanonicalVoteComponent.votes,
            CanonicalVoteComponent.source_column,
        )
        .where(CanonicalVoteComponent.result_id.in_(result_ids))
        .order_by(
            CanonicalVoteComponent.result_id.asc(),
            CanonicalVoteComponent.vote_method.asc(),
        )
    )

    grouped: dict[str, list[dict[str, Any]]] = {str(rid): [] for rid in result_ids}
    for row in conn.execute(stmt).mappings():
        grouped.setdefault(str(row["result_id"]), []).append(
            {
                "vote_method": row["vote_method"],
                "votes": row["votes"],
                "source_column": row["source_column"],
            }
        )

    for item in items:
        # Empty list means no component rows exist. It does not synthesize zeros.
        item["vote_components"] = grouped.get(item["id"], [])


def _apply_facet_scope_filters(
    stmt: Select,
    filters: CanonicalFacetFilters,
    *,
    exclude: str,
) -> Select:
    """Apply all active facet filters except the dimension being enumerated."""

    result = CanonicalElectionResult
    race = CanonicalElectionRace

    if exclude != "state" and filters.state:
        stmt = stmt.where(func.lower(race.state) == filters.state.strip().lower())
    if exclude != "year" and filters.year is not None:
        stmt = stmt.where(race.election_year == int(filters.year))
    if exclude != "jurisdiction" and filters.jurisdiction:
        stmt = stmt.where(
            func.lower(result.jurisdiction_name)
            == filters.jurisdiction.strip().lower()
        )
    if exclude != "contest" and filters.contest:
        stmt = stmt.where(race.contest.ilike(f"%{filters.contest.strip()}%"))

    return stmt


def _build_facet_statement(
    filters: CanonicalFacetFilters,
    dimension: str,
) -> Select:
    """Build one self-excluding distinct-value query for the workbench."""

    result = CanonicalElectionResult
    race = CanonicalElectionRace

    if dimension == "year":
        stmt = (
            select(race.election_year.label("year"))
            .select_from(result)
            .join(race, result.race_id == race.id)
            .where(race.election_year.is_not(None))
        )
        stmt = _apply_facet_scope_filters(stmt, filters, exclude="year")
        return stmt.distinct().order_by(race.election_year.desc())

    if dimension == "state":
        stmt = (
            select(race.state.label("state"))
            .select_from(result)
            .join(race, result.race_id == race.id)
            .where(race.state.is_not(None))
        )
        stmt = _apply_facet_scope_filters(stmt, filters, exclude="state")
        return stmt.distinct().order_by(race.state.asc())

    if dimension == "jurisdiction":
        stmt = (
            select(
                result.jurisdiction_name.label("jurisdiction_name"),
                result.jurisdiction_type.label("jurisdiction_type"),
            )
            .select_from(result)
            .join(race, result.race_id == race.id)
            .where(result.jurisdiction_name.is_not(None))
        )
        stmt = _apply_facet_scope_filters(stmt, filters, exclude="jurisdiction")
        return stmt.distinct().order_by(
            result.jurisdiction_name.asc(),
            result.jurisdiction_type.asc(),
        )

    if dimension == "contest":
        stmt = (
            select(race.contest.label("contest"))
            .select_from(result)
            .join(race, result.race_id == race.id)
            .where(race.contest.is_not(None))
        )
        stmt = _apply_facet_scope_filters(stmt, filters, exclude="contest")
        return stmt.distinct().order_by(race.contest.asc())

    raise ValueError(f"Unsupported canonical facet dimension: {dimension}")


def query_canonical_facets(
    engine: Engine,
    filters: CanonicalFacetFilters,
) -> dict[str, Any]:
    """Return complete self-excluding canonical facets in a read-only transaction."""

    statements = {
        "years": _build_facet_statement(filters, "year"),
        "states": _build_facet_statement(filters, "state"),
        "jurisdictions": _build_facet_statement(filters, "jurisdiction"),
        "contests": _build_facet_statement(filters, "contest"),
    }

    with engine.connect() as conn:
        transaction = conn.begin()
        try:
            if conn.dialect.name == "postgresql":
                conn.exec_driver_sql("SET TRANSACTION READ ONLY")

            years = [
                int(row.year)
                for row in conn.execute(statements["years"])
                if row.year is not None
            ]
            states = [
                str(row.state)
                for row in conn.execute(statements["states"])
                if row.state
            ]
            jurisdictions = [
                {
                    "name": str(row.jurisdiction_name),
                    "type": row.jurisdiction_type,
                }
                for row in conn.execute(statements["jurisdictions"])
                if row.jurisdiction_name
            ]
            contests = [
                str(row.contest)
                for row in conn.execute(statements["contests"])
                if row.contest
            ]

            return {
                "years": years,
                "states": states,
                "jurisdictions": jurisdictions,
                "contests": contests,
                "active_filters": {
                    "state": filters.state,
                    "year": filters.year,
                    "jurisdiction": filters.jurisdiction,
                    "contest": filters.contest,
                },
            }
        finally:
            # Facet discovery is read-only and never commits.
            transaction.rollback()


def query_canonical_results(
    engine: Engine,
    filters: CanonicalResultFilters,
    *,
    include_components: bool = False,
) -> list[dict[str, Any]]:
    """Execute a deterministic canonical read in a rollback-only transaction."""

    stmt = _build_result_statement(filters)

    with engine.connect() as conn:
        transaction = conn.begin()
        try:
            if conn.dialect.name == "postgresql":
                # Must occur before the first statement in the transaction.
                conn.exec_driver_sql("SET TRANSACTION READ ONLY")

            rows = conn.execute(stmt).all()
            items = [_serialize_result(row) for row in rows]

            if include_components:
                _attach_components(conn, items)

            return items
        finally:
            # The publication adapter never commits.
            transaction.rollback()
