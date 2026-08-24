"""Read-only PostgreSQL authority for the ElectionPulse operational workflow plane.

The workflow_* tables describe assignments, independent passes, reconciliation,
reviews, blockers, and publication readiness. They are explicitly NONCANONICAL.

This module never writes election truth and never falls back to Google Sheets,
DB-Lite, warehouse compatibility data, or another UI surface.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Mapping
from uuid import UUID

from sqlalchemy import and_, func, or_, select, text
from sqlalchemy.orm import Session

from webapp.parser.utils.models import (
    WorkflowArtifactLink,
    WorkflowComparison,
    WorkflowDiscrepancy,
    WorkflowEvent,
    WorkflowItem,
    WorkflowPass,
    WorkflowReview,
)


WORKFLOW_READ_SCHEMA_VERSION = "workflow_read_v1"
WORKFLOW_AUTHORITY = {
    "kind": "operational_workflow",
    "canonical": False,
    "source": "postgresql",
    "read_only": True,
    "lineage_inferred": False,
}

_MAX_LIMIT = 500
_MAX_OFFSET = 1_000_000
_MAX_TEXT_FILTER = 512

_FILTER_KEYS = {
    "year",
    "state",
    "jurisdiction",
    "jurisdiction_type",
    "contest",
    "lifecycle_state",
    "current_stage",
    "stage_condition",
    "priority",
    "source_race_id",
    "canonical_linked",
    "search",
}

_FACET_AXES = (
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


class WorkflowReadValidationError(ValueError):
    """Raised when a read request contains invalid scope or pagination values."""


def _authority_payload() -> dict[str, Any]:
    return dict(WORKFLOW_AUTHORITY)


def _iso(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return _iso(value)


def _text_filter(
    raw: Any,
    *,
    field: str,
    max_len: int = _MAX_TEXT_FILTER,
) -> str | None:
    if raw is None:
        return None
    value = str(raw).strip()
    if not value:
        return None
    if len(value) > max_len:
        raise WorkflowReadValidationError(
            f"{field} exceeds maximum length {max_len}."
        )
    return value


def _int_filter(
    raw: Any,
    *,
    field: str,
    minimum: int,
    maximum: int,
) -> int | None:
    if raw is None or str(raw).strip() == "":
        return None
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise WorkflowReadValidationError(
            f"{field} must be an integer."
        ) from exc
    if value < minimum or value > maximum:
        raise WorkflowReadValidationError(
            f"{field} must be between {minimum} and {maximum}."
        )
    return value


def _bool_filter(raw: Any, *, field: str) -> bool | None:
    if raw is None or str(raw).strip() == "":
        return None
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise WorkflowReadValidationError(
        f"{field} must be true or false."
    )


def parse_workflow_filters(
    raw: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Normalize workflow filters without introducing county aliases."""

    source = raw or {}
    filters: dict[str, Any] = {}

    filters["year"] = _int_filter(
        source.get("year"),
        field="year",
        minimum=1700,
        maximum=2300,
    )
    filters["state"] = _text_filter(
        source.get("state"),
        field="state",
        max_len=64,
    )
    filters["jurisdiction"] = _text_filter(
        source.get("jurisdiction"),
        field="jurisdiction",
        max_len=256,
    )
    filters["jurisdiction_type"] = _text_filter(
        source.get("jurisdiction_type"),
        field="jurisdiction_type",
        max_len=32,
    )
    filters["contest"] = _text_filter(
        source.get("contest"),
        field="contest",
        max_len=256,
    )
    filters["lifecycle_state"] = _text_filter(
        source.get("lifecycle_state"),
        field="lifecycle_state",
        max_len=32,
    )
    filters["current_stage"] = _text_filter(
        source.get("current_stage"),
        field="current_stage",
        max_len=48,
    )
    filters["stage_condition"] = _text_filter(
        source.get("stage_condition"),
        field="stage_condition",
        max_len=32,
    )
    filters["priority"] = _int_filter(
        source.get("priority"),
        field="priority",
        minimum=0,
        maximum=1_000_000,
    )
    filters["source_race_id"] = _text_filter(
        source.get("source_race_id"),
        field="source_race_id",
        max_len=128,
    )
    filters["canonical_linked"] = _bool_filter(
        source.get("canonical_linked"),
        field="canonical_linked",
    )
    filters["search"] = _text_filter(
        source.get("search"),
        field="search",
        max_len=256,
    )

    return {
        key: value
        for key, value in filters.items()
        if value is not None
    }


def parse_pagination(
    raw: Mapping[str, Any] | None,
) -> tuple[int, int]:
    source = raw or {}
    limit = _int_filter(
        source.get("limit"),
        field="limit",
        minimum=1,
        maximum=_MAX_LIMIT,
    )
    offset = _int_filter(
        source.get("offset"),
        field="offset",
        minimum=0,
        maximum=_MAX_OFFSET,
    )
    return limit or 100, offset or 0


def _set_transaction_read_only(session: Session) -> None:
    """Fail closed toward writes for PostgreSQL-backed workflow reads."""

    bind = session.get_bind()
    dialect_name = str(bind.dialect.name or "").lower()
    if dialect_name == "postgresql":
        session.execute(text("SET TRANSACTION READ ONLY"))


def _filter_conditions(
    filters: Mapping[str, Any],
    *,
    exclude_axis: str | None = None,
) -> list[Any]:
    conditions: list[Any] = []

    if exclude_axis != "year" and "year" in filters:
        conditions.append(
            WorkflowItem.election_year == filters["year"]
        )
    if exclude_axis != "state" and "state" in filters:
        conditions.append(
            WorkflowItem.state == filters["state"]
        )
    if exclude_axis != "jurisdiction" and "jurisdiction" in filters:
        conditions.append(
            WorkflowItem.jurisdiction_name == filters["jurisdiction"]
        )
    if (
        exclude_axis != "jurisdiction_type"
        and "jurisdiction_type" in filters
    ):
        conditions.append(
            WorkflowItem.jurisdiction_type
            == filters["jurisdiction_type"]
        )
    if exclude_axis != "contest" and "contest" in filters:
        conditions.append(
            WorkflowItem.contest == filters["contest"]
        )
    if (
        exclude_axis != "lifecycle_state"
        and "lifecycle_state" in filters
    ):
        conditions.append(
            WorkflowItem.lifecycle_state
            == filters["lifecycle_state"]
        )
    if (
        exclude_axis != "current_stage"
        and "current_stage" in filters
    ):
        conditions.append(
            WorkflowItem.current_stage
            == filters["current_stage"]
        )
    if (
        exclude_axis != "stage_condition"
        and "stage_condition" in filters
    ):
        conditions.append(
            WorkflowItem.stage_condition
            == filters["stage_condition"]
        )
    if exclude_axis != "priority" and "priority" in filters:
        conditions.append(
            WorkflowItem.priority == filters["priority"]
        )
    if "source_race_id" in filters:
        conditions.append(
            WorkflowItem.source_race_id == filters["source_race_id"]
        )
    if "canonical_linked" in filters:
        if filters["canonical_linked"]:
            conditions.append(
                WorkflowItem.canonical_race_id.is_not(None)
            )
        else:
            conditions.append(
                WorkflowItem.canonical_race_id.is_(None)
            )
    if "search" in filters:
        pattern = f"%{filters['search']}%"
        conditions.append(
            or_(
                WorkflowItem.contest.ilike(pattern),
                WorkflowItem.jurisdiction_name.ilike(pattern),
                WorkflowItem.source_race_id.ilike(pattern),
                WorkflowItem.source_url.ilike(pattern),
            )
        )

    return conditions


def _where(statement: Any, conditions: list[Any]) -> Any:
    if conditions:
        statement = statement.where(and_(*conditions))
    return statement


def _serialize_item(item: WorkflowItem) -> dict[str, Any]:
    return {
        "id": str(item.id),
        "authority": "operational_workflow",
        "canonical_authority": False,
        "lifecycle_state": item.lifecycle_state,
        "current_stage": item.current_stage,
        "stage_condition": item.stage_condition,
        "priority": item.priority,
        "scope": {
            "election_year": item.election_year,
            "election_date": _iso(item.election_date),
            "state": item.state,
            "jurisdiction_name": item.jurisdiction_name,
            "jurisdiction_type": item.jurisdiction_type,
            "contest": item.contest,
            "office_basic": item.office_basic,
            "election_type": item.election_type,
            "source_race_id": item.source_race_id,
        },
        "source_url": item.source_url,
        "canonical_reference": {
            "race_id": (
                str(item.canonical_race_id)
                if item.canonical_race_id is not None
                else None
            ),
            "linked": item.canonical_race_id is not None,
            "lineage_inferred": False,
        },
        "blocker": {
            "reason_code": item.blocked_reason_code,
            "detail": item.blocker_detail,
        },
        "created_by_principal": item.created_by_principal,
        "workflow_metadata": _json_safe(item.workflow_metadata),
        "row_version": item.row_version,
        "created_at": _iso(item.created_at),
        "updated_at": _iso(item.updated_at),
    }


def _serialize_pass(row: WorkflowPass) -> dict[str, Any]:
    return {
        "id": str(row.id),
        "workflow_item_id": str(row.workflow_item_id),
        "pass_number": row.pass_number,
        "pass_label": row.pass_label,
        "revision_number": row.revision_number,
        "is_current": row.is_current,
        "status": row.status,
        "assigned_principal": row.assigned_principal,
        "source_evidence_ref": row.source_evidence_ref,
        "staging_batch_id": (
            str(row.staging_batch_id)
            if row.staging_batch_id is not None
            else None
        ),
        "candidate_check_status": row.candidate_check_status,
        "candidate_check_result": _json_safe(
            row.candidate_check_result
        ),
        "semantic_validation_status": (
            row.semantic_validation_status
        ),
        "semantic_validation_result": _json_safe(
            row.semantic_validation_result
        ),
        "started_at": _iso(row.started_at),
        "submitted_at": _iso(row.submitted_at),
        "superseded_at": _iso(row.superseded_at),
        "notes": row.notes,
        "created_at": _iso(row.created_at),
        "updated_at": _iso(row.updated_at),
    }


def _serialize_comparison(
    row: WorkflowComparison,
) -> dict[str, Any]:
    return {
        "id": str(row.id),
        "workflow_item_id": str(row.workflow_item_id),
        "left_pass_id": str(row.left_pass_id),
        "right_pass_id": str(row.right_pass_id),
        "comparison_version": row.comparison_version,
        "status": row.status,
        "strict_equality_passed": row.strict_equality_passed,
        "difference_count": row.difference_count,
        "difference_summary": _json_safe(row.difference_summary),
        "checked_at": _iso(row.checked_at),
        "checked_by_service_version": row.checked_by_service_version,
        "reviewed_by_principal": row.reviewed_by_principal,
        "reviewed_at": _iso(row.reviewed_at),
        "created_at": _iso(row.created_at),
    }


def _serialize_discrepancy(
    row: WorkflowDiscrepancy,
) -> dict[str, Any]:
    return {
        "id": str(row.id),
        "comparison_id": str(row.comparison_id),
        "workflow_item_id": str(row.workflow_item_id),
        "category": row.category,
        "semantic_key": _json_safe(row.semantic_key),
        "left_value": _json_safe(row.left_value),
        "right_value": _json_safe(row.right_value),
        "left_value_state": row.left_value_state,
        "right_value_state": row.right_value_state,
        "severity": row.severity,
        "resolution_status": row.resolution_status,
        "resolution_code": row.resolution_code,
        "resolution_notes": row.resolution_notes,
        "resolved_by_principal": row.resolved_by_principal,
        "resolved_at": _iso(row.resolved_at),
        "created_at": _iso(row.created_at),
    }


def _serialize_review(row: WorkflowReview) -> dict[str, Any]:
    return {
        "id": str(row.id),
        "workflow_item_id": str(row.workflow_item_id),
        "review_stage": row.review_stage,
        "reviewer_principal": row.reviewer_principal,
        "decision": row.decision,
        "selected_pass_id": (
            str(row.selected_pass_id)
            if row.selected_pass_id is not None
            else None
        ),
        "selected_staging_batch_id": (
            str(row.selected_staging_batch_id)
            if row.selected_staging_batch_id is not None
            else None
        ),
        "checklist_version": row.checklist_version,
        "checklist_result": _json_safe(row.checklist_result),
        "reason_codes": _json_safe(row.reason_codes),
        "notes": row.notes,
        "reviewed_at": _iso(row.reviewed_at),
    }


def _serialize_artifact(
    row: WorkflowArtifactLink,
) -> dict[str, Any]:
    return {
        "id": str(row.id),
        "workflow_item_id": str(row.workflow_item_id),
        "pass_id": (
            str(row.pass_id)
            if row.pass_id is not None
            else None
        ),
        "relation_type": row.relation_type,
        "artifact_type": row.artifact_type,
        "artifact_ref": row.artifact_ref,
        "artifact_sha256": row.artifact_sha256,
        "canonical_source_artifact_id": (
            str(row.canonical_source_artifact_id)
            if row.canonical_source_artifact_id is not None
            else None
        ),
        "staging_batch_id": (
            str(row.staging_batch_id)
            if row.staging_batch_id is not None
            else None
        ),
        "artifact_metadata": _json_safe(row.artifact_metadata),
        "created_at": _iso(row.created_at),
    }


def _serialize_event(row: WorkflowEvent) -> dict[str, Any]:
    return {
        "id": str(row.id),
        "workflow_item_id": str(row.workflow_item_id),
        "actor_type": row.actor_type,
        "actor_principal": row.actor_principal,
        "actor_service": row.actor_service,
        "event_type": row.event_type,
        "stage": row.stage,
        "prior_state": _json_safe(row.prior_state),
        "new_state": _json_safe(row.new_state),
        "related_pass_id": (
            str(row.related_pass_id)
            if row.related_pass_id is not None
            else None
        ),
        "related_comparison_id": (
            str(row.related_comparison_id)
            if row.related_comparison_id is not None
            else None
        ),
        "related_review_id": (
            str(row.related_review_id)
            if row.related_review_id is not None
            else None
        ),
        "related_staging_batch_id": (
            str(row.related_staging_batch_id)
            if row.related_staging_batch_id is not None
            else None
        ),
        "related_canonical_race_id": (
            str(row.related_canonical_race_id)
            if row.related_canonical_race_id is not None
            else None
        ),
        "reason_code": row.reason_code,
        "summary": row.summary,
        "event_metadata": _json_safe(row.event_metadata),
        "occurred_at": _iso(row.occurred_at),
    }


def read_workflow_items(
    session: Session,
    raw_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _set_transaction_read_only(session)
    filters = parse_workflow_filters(raw_params)
    limit, offset = parse_pagination(raw_params)
    conditions = _filter_conditions(filters)

    count_stmt = _where(
        select(func.count()).select_from(WorkflowItem),
        conditions,
    )
    total = int(session.execute(count_stmt).scalar_one())

    stmt = _where(
        select(WorkflowItem),
        conditions,
    ).order_by(
        WorkflowItem.priority.desc(),
        WorkflowItem.updated_at.asc(),
        WorkflowItem.id.asc(),
    ).limit(limit).offset(offset)

    items = [
        _serialize_item(row)
        for row in session.execute(stmt).scalars().all()
    ]

    return {
        "success": True,
        "schema_version": WORKFLOW_READ_SCHEMA_VERSION,
        "authority": _authority_payload(),
        "filters": _json_safe(filters),
        "items": items,
        "pagination": {
            "limit": limit,
            "offset": offset,
            "returned": len(items),
            "total": total,
            "has_more": offset + len(items) < total,
        },
    }


def read_workflow_item_detail(
    session: Session,
    item_id: UUID | str,
) -> dict[str, Any] | None:
    _set_transaction_read_only(session)

    try:
        normalized_id = (
            item_id if isinstance(item_id, UUID) else UUID(str(item_id))
        )
    except (TypeError, ValueError) as exc:
        raise WorkflowReadValidationError(
            "workflow item id must be a UUID."
        ) from exc

    item = session.get(WorkflowItem, normalized_id)
    if item is None:
        return None

    passes = session.execute(
        select(WorkflowPass)
        .where(WorkflowPass.workflow_item_id == normalized_id)
        .order_by(
            WorkflowPass.pass_number.asc(),
            WorkflowPass.revision_number.desc(),
        )
    ).scalars().all()

    comparisons = session.execute(
        select(WorkflowComparison)
        .where(WorkflowComparison.workflow_item_id == normalized_id)
        .order_by(WorkflowComparison.created_at.desc())
    ).scalars().all()

    discrepancies = session.execute(
        select(WorkflowDiscrepancy)
        .where(WorkflowDiscrepancy.workflow_item_id == normalized_id)
        .order_by(
            WorkflowDiscrepancy.resolution_status.asc(),
            WorkflowDiscrepancy.created_at.asc(),
        )
    ).scalars().all()

    reviews = session.execute(
        select(WorkflowReview)
        .where(WorkflowReview.workflow_item_id == normalized_id)
        .order_by(WorkflowReview.reviewed_at.asc())
    ).scalars().all()

    artifacts = session.execute(
        select(WorkflowArtifactLink)
        .where(WorkflowArtifactLink.workflow_item_id == normalized_id)
        .order_by(WorkflowArtifactLink.created_at.asc())
    ).scalars().all()

    events = session.execute(
        select(WorkflowEvent)
        .where(WorkflowEvent.workflow_item_id == normalized_id)
        .order_by(
            WorkflowEvent.occurred_at.asc(),
            WorkflowEvent.id.asc(),
        )
        .limit(500)
    ).scalars().all()

    return {
        "success": True,
        "schema_version": WORKFLOW_READ_SCHEMA_VERSION,
        "authority": _authority_payload(),
        "item": _serialize_item(item),
        "passes": [_serialize_pass(row) for row in passes],
        "comparisons": [
            _serialize_comparison(row) for row in comparisons
        ],
        "discrepancies": [
            _serialize_discrepancy(row) for row in discrepancies
        ],
        "reviews": [_serialize_review(row) for row in reviews],
        "artifacts": [_serialize_artifact(row) for row in artifacts],
        "events": [_serialize_event(row) for row in events],
    }


def _facet_rows(
    session: Session,
    filters: Mapping[str, Any],
    axis: str,
) -> list[dict[str, Any]]:
    conditions = _filter_conditions(
        filters,
        exclude_axis=axis,
    )

    if axis == "jurisdiction":
        stmt = select(
            WorkflowItem.jurisdiction_name,
            WorkflowItem.jurisdiction_type,
            func.count(WorkflowItem.id),
        ).group_by(
            WorkflowItem.jurisdiction_name,
            WorkflowItem.jurisdiction_type,
        ).order_by(
            WorkflowItem.jurisdiction_name.asc().nulls_first(),
            WorkflowItem.jurisdiction_type.asc().nulls_first(),
        )
        stmt = _where(stmt, conditions)
        return [
            {
                "value": {
                    "name": name,
                    "type": jurisdiction_type,
                },
                "count": int(count),
            }
            for name, jurisdiction_type, count
            in session.execute(stmt).all()
        ]

    columns = {
        "year": WorkflowItem.election_year,
        "state": WorkflowItem.state,
        "jurisdiction_type": WorkflowItem.jurisdiction_type,
        "contest": WorkflowItem.contest,
        "lifecycle_state": WorkflowItem.lifecycle_state,
        "current_stage": WorkflowItem.current_stage,
        "stage_condition": WorkflowItem.stage_condition,
        "priority": WorkflowItem.priority,
    }
    column = columns[axis]

    stmt = select(
        column,
        func.count(WorkflowItem.id),
    ).group_by(column).order_by(column.asc().nulls_first())
    stmt = _where(stmt, conditions)

    return [
        {
            "value": _json_safe(value),
            "count": int(count),
        }
        for value, count in session.execute(stmt).all()
    ]


def read_workflow_facets(
    session: Session,
    raw_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _set_transaction_read_only(session)
    filters = parse_workflow_filters(raw_params)

    facets = {
        axis: _facet_rows(session, filters, axis)
        for axis in _FACET_AXES
    }

    return {
        "success": True,
        "schema_version": WORKFLOW_READ_SCHEMA_VERSION,
        "authority": _authority_payload(),
        "facet_mode": "self_excluding",
        "filters": _json_safe(filters),
        "axes": list(_FACET_AXES),
        "facets": facets,
    }


def _group_counts(
    session: Session,
    column: Any,
    conditions: list[Any],
) -> list[dict[str, Any]]:
    stmt = select(
        column,
        func.count(WorkflowItem.id),
    ).group_by(column).order_by(column.asc().nulls_first())
    stmt = _where(stmt, conditions)
    return [
        {
            "value": _json_safe(value),
            "count": int(count),
        }
        for value, count in session.execute(stmt).all()
    ]


def read_workflow_stats(
    session: Session,
    raw_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _set_transaction_read_only(session)
    filters = parse_workflow_filters(raw_params)
    conditions = _filter_conditions(filters)

    total_stmt = _where(
        select(func.count()).select_from(WorkflowItem),
        conditions,
    )
    total = int(session.execute(total_stmt).scalar_one())

    blocked_stmt = _where(
        select(func.count()).select_from(WorkflowItem),
        conditions + [WorkflowItem.lifecycle_state == "blocked"],
    )
    ready_stmt = _where(
        select(func.count()).select_from(WorkflowItem),
        conditions + [
            WorkflowItem.lifecycle_state == "ready_for_publication"
        ],
    )
    published_stmt = _where(
        select(func.count()).select_from(WorkflowItem),
        conditions + [WorkflowItem.lifecycle_state == "published"],
    )

    return {
        "success": True,
        "schema_version": WORKFLOW_READ_SCHEMA_VERSION,
        "authority": _authority_payload(),
        "filters": _json_safe(filters),
        "total": total,
        "action_counts": {
            "blocked": int(
                session.execute(blocked_stmt).scalar_one()
            ),
            "ready_for_publication": int(
                session.execute(ready_stmt).scalar_one()
            ),
            "published": int(
                session.execute(published_stmt).scalar_one()
            ),
        },
        "by_lifecycle_state": _group_counts(
            session,
            WorkflowItem.lifecycle_state,
            conditions,
        ),
        "by_current_stage": _group_counts(
            session,
            WorkflowItem.current_stage,
            conditions,
        ),
        "by_stage_condition": _group_counts(
            session,
            WorkflowItem.stage_condition,
            conditions,
        ),
    }
