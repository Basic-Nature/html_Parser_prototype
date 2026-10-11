"""Bounded public canonical race-coverage projection (J2B1NH-A).

Canonical race identity is an instance ID, not a publication-stable permalink.
No source registry, operational workflow, raw URL, or observer metadata is joined.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from sqlalchemy import case, func, select
from sqlalchemy.engine import Engine

from webapp.parser.utils.models import CanonicalElectionRace, CanonicalElectionResult


@dataclass(frozen=True, slots=True)
class CoverageFilters:
    state: str | None = None
    year: int | None = None
    contest: str | None = None
    limit: int = 20
    offset: int = 0


def _safe_verification_label(raw: Any) -> str:
    """Keep only bounded machine-readable persisted status tokens."""
    value = str(raw or "").strip().lower()
    return value if re.fullmatch(r"[a-z][a-z0-9_]{0,31}", value) else "unavailable"


def _iso(value: date | datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def query_canonical_coverage(engine: Engine, filters: CoverageFilters) -> dict[str, Any]:
    """Read current canonical publication grouped by *exact race UUID*.

    At most 50 race summaries are returned. Counting is done entirely by SQL;
    vote components are intentionally not joined, to avoid double counting.
    """
    if not 1 <= filters.limit <= 50 or not 0 <= filters.offset <= 5000:
        raise ValueError("coverage pagination outside allowed bounds")

    race = CanonicalElectionRace
    result = CanonicalElectionResult
    aggregates = (
        select(
            result.race_id.label("race_id"),
            func.count(result.id).label("result_count"),
            func.count(func.distinct(result.jurisdiction_key)).label("jurisdiction_count"),
            func.sum(case((result.aggregation_scope == "jurisdiction", 1), else_=0)).label("jurisdiction_scope_rows"),
            func.sum(case((result.aggregation_scope == "precinct", 1), else_=0)).label("precinct_scope_rows"),
        )
        .group_by(result.race_id)
        .subquery("canonical_race_coverage")
    )
    criterion = []
    if filters.state:
        criterion.append(func.lower(race.state) == filters.state.strip().lower())
    if filters.year is not None:
        criterion.append(race.election_year == filters.year)
    if filters.contest:
        criterion.append(race.contest.ilike(f"%{filters.contest.strip()}%"))

    scope = select(race.id).join(aggregates, aggregates.c.race_id == race.id).where(*criterion)
    query = (
        select(
            race.id.label("race_id"), race.state, race.election_year,
            race.election_date, race.date_precision, race.contest,
            race.verification_status, race.selected_dl_source,
            aggregates.c.result_count, aggregates.c.jurisdiction_count,
            aggregates.c.jurisdiction_scope_rows, aggregates.c.precinct_scope_rows,
        )
        .join(aggregates, aggregates.c.race_id == race.id)
        .where(*criterion)
        .order_by(race.election_year.desc(), race.state.asc(), race.contest.asc(), race.id.asc())
        .limit(filters.limit)
        .offset(filters.offset)
    )
    with engine.connect() as conn:
        transaction = conn.begin()
        try:
            if conn.dialect.name == "postgresql":
                conn.exec_driver_sql("SET TRANSACTION READ ONLY")
            total = int(conn.execute(select(func.count()).select_from(scope.subquery())).scalar_one())
            rows = conn.execute(query).all()
            items = [{
                "race_id": str(row.race_id),
                "state": row.state,
                "year": row.election_year,
                "contest": row.contest,
                "election_date": _iso(row.election_date),
                "date_precision": row.date_precision,
                "result_count": int(row.result_count),
                "jurisdiction_count": int(row.jurisdiction_count),
                "jurisdiction_scope_rows": int(row.jurisdiction_scope_rows),
                "precinct_scope_rows": int(row.precinct_scope_rows),
                "verification_status": _safe_verification_label(row.verification_status),
                "selected_dl_source": row.selected_dl_source if row.selected_dl_source in {"DL1", "DL2"} else None,
                "official_source_relationship": "not_established",
            } for row in rows]
            return {
                "contract": "canonical_coverage_v1",
                "authority": "canonical_production",
                "data_source": "canonical",
                "items": items,
                "pagination": {
                    "limit": filters.limit,
                    "offset": filters.offset,
                    "total_races": total,
                    "has_more": filters.offset + len(items) < total,
                },
                "active_filters": {
                    "state": filters.state,
                    "year": filters.year,
                    "contest": filters.contest,
                },
                "semantic_contract": {
                    "lineage": "not_inferred",
                    "precinct_null": "preserved",
                    "zero": "numeric_zero",
                    "missing_coverage": "not_published_in_this_view",
                    "selected_dl_lane": "chosen_payload_not_two_observer_agreement",
                    "race_id": "current_record_instance_only",
                    "no_warehouse_fallback": True,
                    "historical_seed_counts_not_live_authority": True,
                },
            }
        finally:
            transaction.rollback()
