from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

from sqlalchemy import func, inspect, select, text
from sqlalchemy.orm import Session


CANONICAL_AUTHORITY = {
    "kind": "canonical_publication",
    "canonical": True,
    "source": "postgresql",
    "read_only": True,
    "lineage_inferred": False,
}

LEGACY_WORKLIST_AUTHORITY = {
    "kind": "operational_worklist_legacy",
    "canonical": False,
    "source": "postgresql",
    "read_only": True,
    "lineage_inferred": False,
}

WORKFLOW_SCHEMA_REASON = "workflow_schema_not_provisioned"
WORKFLOW_FACET_AXES = (
    "year",
    "state",
    "jurisdiction",
    "jurisdiction_type",
    "contest",
    "lifecycle_state",
    "current_stage",
    "stage_condition",
    "priority",
)


def _set_session_read_only(session: Session) -> None:
    bind = session.get_bind()
    if str(bind.dialect.name or "").lower() == "postgresql":
        session.execute(text("SET TRANSACTION READ ONLY"))


def _set_connection_read_only(conn: Any) -> None:
    if str(conn.dialect.name or "").lower() == "postgresql":
        conn.exec_driver_sql("SET TRANSACTION READ ONLY")


def _table_exists(engine: Any, table_name: str) -> bool:
    return bool(inspect(engine).has_table(table_name))


def build_scope_directory_payload(
    scope_rows: list[tuple[Any, Any, Any]],
    years: list[Any],
    contests: list[Any],
) -> dict[str, Any]:
    """Build the compatibility directory without coercing districts into counties."""

    states: set[str] = set()
    counties: dict[str, set[str]] = defaultdict(set)
    jurisdictions: dict[str, dict[tuple[str, str | None], dict[str, Any]]] = defaultdict(dict)

    for state_raw, name_raw, type_raw in scope_rows:
        state = str(state_raw or "").strip()
        name = str(name_raw or "").strip()
        jurisdiction_type = (
            str(type_raw).strip() if type_raw is not None and str(type_raw).strip() else None
        )
        if not state or not name:
            continue

        states.add(state)
        key = (name, jurisdiction_type)
        jurisdictions[state][key] = {
            "name": name,
            "type": jurisdiction_type,
        }

        if jurisdiction_type and jurisdiction_type.lower() == "county":
            counties[state].add(name)

    states_list = sorted(states)
    counties_dict = {
        state: sorted(counties.get(state, set()))
        for state in states_list
    }
    jurisdictions_dict = {
        state: sorted(
            jurisdictions.get(state, {}).values(),
            key=lambda item: (
                str(item.get("name") or "").lower(),
                str(item.get("type") or "").lower(),
            ),
        )
        for state in states_list
    }

    normalized_years = sorted(
        {int(value) for value in years if value is not None},
        reverse=True,
    )
    normalized_contests = sorted(
        {str(value).strip() for value in contests if str(value or "").strip()}
    )

    total_counties = sum(len(values) for values in counties_dict.values())
    total_jurisdictions = sum(len(values) for values in jurisdictions_dict.values())

    return {
        "success": True,
        "available": True,
        "degraded": False,
        "contract": "canonical_scope_directory_v1",
        "authority": "canonical_production",
        "authority_detail": dict(CANONICAL_AUTHORITY),
        "states": states_list,
        # Compatibility surface: ONLY semantically confirmed county jurisdictions.
        "counties": counties_dict,
        "jurisdictions": jurisdictions_dict,
        "years": normalized_years,
        "contests": normalized_contests,
        "total_states": len(states_list),
        "total_counties": total_counties,
        "total_jurisdictions": total_jurisdictions,
        "semantic_contract": {
            "county_compatibility": "jurisdiction_type_equals_county_only",
            "districts_are_not_counties": True,
            "lineage": "not_inferred",
        },
    }


def read_public_scope_directory() -> dict[str, Any]:
    """Read state/jurisdiction scope directly from canonical production tables."""

    from webapp.parser.utils.db_utils import get_engine
    from webapp.parser.utils.models import (
        CanonicalElectionRace,
        CanonicalElectionResult,
    )

    engine = get_engine()
    result = CanonicalElectionResult
    race = CanonicalElectionRace

    scope_stmt = (
        select(
            race.state,
            result.jurisdiction_name,
            result.jurisdiction_type,
        )
        .select_from(result)
        .join(race, result.race_id == race.id)
        .where(
            race.state.is_not(None),
            result.jurisdiction_name.is_not(None),
        )
        .distinct()
        .order_by(
            race.state.asc(),
            result.jurisdiction_name.asc(),
            result.jurisdiction_type.asc(),
        )
    )
    years_stmt = (
        select(race.election_year)
        .select_from(result)
        .join(race, result.race_id == race.id)
        .where(race.election_year.is_not(None))
        .distinct()
        .order_by(race.election_year.desc())
    )
    contests_stmt = (
        select(race.contest)
        .select_from(result)
        .join(race, result.race_id == race.id)
        .where(race.contest.is_not(None))
        .distinct()
        .order_by(race.contest.asc())
    )

    with engine.connect() as conn:
        transaction = conn.begin()
        try:
            _set_connection_read_only(conn)
            scope_rows = list(conn.execute(scope_stmt).all())
            years = [row[0] for row in conn.execute(years_stmt).all()]
            contests = [row[0] for row in conn.execute(contests_stmt).all()]
            return build_scope_directory_payload(scope_rows, years, contests)
        finally:
            transaction.rollback()


def _parse_limit(raw: Mapping[str, Any] | None, default: int = 100) -> int:
    source = raw or {}
    try:
        value = int(str(source.get("limit") or default).strip())
    except (TypeError, ValueError):
        value = default
    return max(1, min(500, value))


def _legacy_worklist_unavailable() -> dict[str, Any]:
    return {
        "success": True,
        "available": False,
        "degraded": True,
        "reason": "legacy_worklist_schema_not_provisioned",
        "authority": dict(LEGACY_WORKLIST_AUTHORITY),
        "total": None,
        "records": [],
        "visibility": "public_projection",
        "semantic_contract": {
            "unavailable_count": "null",
            "zero": "numeric_zero_only",
            "identity_fields": "redacted",
        },
    }


def read_public_worklist(
    raw_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Read the legacy public Worklist projection from the central production engine."""

    from webapp.parser.models.election_data import DownloadRecord
    from webapp.parser.utils.db_utils import get_engine

    engine = get_engine()
    if not _table_exists(engine, DownloadRecord.__tablename__):
        return _legacy_worklist_unavailable()

    limit = _parse_limit(raw_params)
    params = raw_params or {}
    session = Session(bind=engine, autoflush=False, expire_on_commit=False)
    try:
        _set_session_read_only(session)
        query = session.query(DownloadRecord)

        state = str(params.get("state") or "").strip()
        if state:
            query = query.filter(DownloadRecord.state == state)

        year_raw = str(params.get("year") or "").strip()
        if year_raw:
            try:
                year_value: Any = int(year_raw)
            except ValueError:
                year_value = year_raw
            query = query.filter(DownloadRecord.year == year_value)

        status = str(params.get("status") or "").strip()
        if status:
            query = query.filter(DownloadRecord.workflow_status == status)

        total = int(query.count())
        rows = query.order_by(DownloadRecord.updated_at.desc()).limit(limit).all()
        records: list[dict[str, Any]] = []
        for row in rows:
            records.append({
                "id": row.id,
                "race_id": row.race_id,
                "year": row.year,
                "state": row.state,
                "county": row.county,
                "office": row.office,
                "source_url": row.source_url,
                "dl1_assigned_to": None,
                "dl1_status": row.dl1_status,
                "dl2_assigned_to": None,
                "dl2_status": row.dl2_status,
                "preqc_result": row.preqc_result,
                "qc1_assigned_to": None,
                "qc1_status": row.qc1_status,
                "qc1_selected_dl": None,
                "qc2_assigned_to": None,
                "qc2_status": row.qc2_status,
                "workflow_status": row.workflow_status,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                "visibility": "public_projection",
            })

        return {
            "success": True,
            "available": True,
            "degraded": False,
            "authority": dict(LEGACY_WORKLIST_AUTHORITY),
            "total": total,
            "records": records,
            "visibility": "public_projection",
            "semantic_contract": {
                "unavailable_count": "null",
                "zero": "numeric_zero_only",
                "identity_fields": "redacted",
            },
        }
    finally:
        session.rollback()
        session.close()


def _canonical_publication_counts(engine: Any) -> dict[str, int]:
    from webapp.parser.utils.models import (
        CanonicalElectionRace,
        CanonicalElectionResult,
    )

    with engine.connect() as conn:
        transaction = conn.begin()
        try:
            _set_connection_read_only(conn)
            result_count = int(
                conn.execute(
                    select(func.count()).select_from(CanonicalElectionResult)
                ).scalar_one()
            )
            race_count = int(
                conn.execute(
                    select(func.count()).select_from(CanonicalElectionRace)
                ).scalar_one()
            )
            return {
                "result_rows": result_count,
                "races": race_count,
            }
        finally:
            transaction.rollback()


def _unavailable_operational_stats(canonical_counts: dict[str, int]) -> dict[str, Any]:
    return {
        "success": True,
        "available": False,
        "degraded": True,
        "reason": "legacy_worklist_schema_not_provisioned",
        "authority": {
            "operational": dict(LEGACY_WORKLIST_AUTHORITY),
            "production_records": dict(CANONICAL_AUTHORITY),
        },
        "stats": {
            "total_races": None,
            "dl1_ready": None,
            "dl2_ready": None,
            "preqc_passed": None,
            "qc1_pending": None,
            "qc2_pending": None,
            "production_records": canonical_counts["result_rows"],
        },
        "canonical_publication": canonical_counts,
        "semantic_contract": {
            "unavailable_operational_values": "null",
            "zero": "numeric_zero_only",
            "canonical_and_operational_authorities_are_distinct": True,
        },
    }


def read_public_election_stats() -> dict[str, Any]:
    """Read operational counts centrally while keeping canonical publication distinct."""

    from webapp.parser.models.election_data import DownloadRecord
    from webapp.parser.utils.db_utils import get_engine

    engine = get_engine()
    canonical_counts = _canonical_publication_counts(engine)

    if not _table_exists(engine, DownloadRecord.__tablename__):
        return _unavailable_operational_stats(canonical_counts)

    session = Session(bind=engine, autoflush=False, expire_on_commit=False)
    try:
        _set_session_read_only(session)
        stats = {
            "total_races": int(session.query(func.count(DownloadRecord.id)).scalar() or 0),
            "dl1_ready": int(
                session.query(func.count(DownloadRecord.id))
                .filter(DownloadRecord.dl1_status == "ready_for_qc")
                .scalar()
                or 0
            ),
            "dl2_ready": int(
                session.query(func.count(DownloadRecord.id))
                .filter(DownloadRecord.dl2_status == "ready_for_qc")
                .scalar()
                or 0
            ),
            "preqc_passed": int(
                session.query(func.count(DownloadRecord.id))
                .filter(DownloadRecord.preqc_result == "passed")
                .scalar()
                or 0
            ),
            "qc1_pending": int(
                session.query(func.count(DownloadRecord.id))
                .filter(DownloadRecord.qc1_status == "pending")
                .scalar()
                or 0
            ),
            "qc2_pending": int(
                session.query(func.count(DownloadRecord.id))
                .filter(DownloadRecord.qc2_status == "pending")
                .scalar()
                or 0
            ),
            "production_records": canonical_counts["result_rows"],
        }
        return {
            "success": True,
            "available": True,
            "degraded": False,
            "authority": {
                "operational": dict(LEGACY_WORKLIST_AUTHORITY),
                "production_records": dict(CANONICAL_AUTHORITY),
            },
            "stats": stats,
            "canonical_publication": canonical_counts,
            "semantic_contract": {
                "unavailable_operational_values": "null",
                "zero": "numeric_zero_only",
                "canonical_and_operational_authorities_are_distinct": True,
            },
        }
    finally:
        session.rollback()
        session.close()


def _workflow_degraded_base(filters: Mapping[str, Any]) -> dict[str, Any]:
    from webapp.parser.services.workflow_reader import (
        WORKFLOW_AUTHORITY,
        WORKFLOW_READ_SCHEMA_VERSION,
    )

    return {
        "success": True,
        "available": False,
        "degraded": True,
        "reason": WORKFLOW_SCHEMA_REASON,
        "required_migration": "e7b2c4d91f60",
        "schema_version": WORKFLOW_READ_SCHEMA_VERSION,
        "authority": dict(WORKFLOW_AUTHORITY),
        "filters": dict(filters),
        "semantic_contract": {
            "unavailable_counts": "null",
            "zero": "numeric_zero_only",
            "canonical_fallback": False,
            "lineage_inferred": False,
        },
    }


def read_public_workflow_facets(
    raw_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from webapp.parser.services.workflow_reader import (
        parse_workflow_filters,
        read_workflow_facets,
    )
    from webapp.parser.utils.db_utils import SessionLocal, get_engine

    filters = parse_workflow_filters(raw_params)
    engine = get_engine()
    if not _table_exists(engine, "workflow_items"):
        return {
            **_workflow_degraded_base(filters),
            "facet_mode": "self_excluding",
            "axes": list(WORKFLOW_FACET_AXES),
            "facets": {axis: [] for axis in WORKFLOW_FACET_AXES},
        }

    session = SessionLocal()
    try:
        payload = read_workflow_facets(session, raw_params)
        return {
            **payload,
            "available": True,
            "degraded": False,
        }
    finally:
        session.rollback()
        session.close()


def read_public_workflow_stats(
    raw_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from webapp.parser.services.workflow_reader import (
        parse_workflow_filters,
        read_workflow_stats,
    )
    from webapp.parser.utils.db_utils import SessionLocal, get_engine

    filters = parse_workflow_filters(raw_params)
    engine = get_engine()
    if not _table_exists(engine, "workflow_items"):
        return {
            **_workflow_degraded_base(filters),
            "total": None,
            "action_counts": {
                "blocked": None,
                "ready_for_publication": None,
                "published": None,
            },
            "by_lifecycle_state": [],
            "by_current_stage": [],
            "by_stage_condition": [],
        }

    session = SessionLocal()
    try:
        payload = read_workflow_stats(session, raw_params)
        return {
            **payload,
            "available": True,
            "degraded": False,
        }
    finally:
        session.rollback()
        session.close()
