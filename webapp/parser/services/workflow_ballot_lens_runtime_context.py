"""Server-owned runtime context adapter for Workflow-bound Ballot Lens.

The browser supplies only governed Workflow item/pass identifiers and an
optimistic row version. This adapter resolves every execution-sensitive value
server-side from Workflow/canonical data plus the maintained URL registry.

Trust may come from:
* a prior governed Workflow QC1/QC2 approval chain for the exact source; or
* structured legacy canonical production provenance used to bootstrap the
  governed Workflow plane.

A verification-status string alone is never sufficient authority.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from webapp.parser.services.trusted_parser_source_policy import (
    PROVENANCE_LEGACY_PRODUCTION,
    PROVENANCE_WORKFLOW_QC,
)
from webapp.parser.services.workflow_ballot_lens_execution import (
    WorkflowBallotLensExecutionDenied,
    WorkflowBallotLensServerContext,
    validate_workflow_execution_request,
)
from webapp.parser.services.workflow_reviews import (
    WorkflowQCReviewConflict,
    load_workflow_publication_approval_authority,
)
from webapp.parser.utils.models import (
    CanonicalElectionRace,
    CanonicalSourceArtifact,
    CanonicalVerificationEvent,
    WorkflowItem,
    WorkflowPass,
)
from webapp.parser.utils.url_registry import load_url_registry


WORKFLOW_BALLOT_LENS_RUNTIME_CONTEXT_CONTRACT = (
    "workflow_ballot_lens_runtime_context_v1"
)


class WorkflowBallotLensRuntimeContextDenied(PermissionError):
    pass


def _deny(exc: Exception | None = None):
    error = WorkflowBallotLensRuntimeContextDenied(
        "Workflow Ballot Lens runtime context denied."
    )
    if exc is None:
        raise error
    raise error from exc


def _uuid(value: object) -> UUID:
    try:
        return value if isinstance(value, UUID) else UUID(str(value))
    except (TypeError, ValueError, AttributeError) as exc:
        _deny(exc)


def _text(value: object) -> str:
    return str(value or "").strip()


def _category(section: object) -> str:
    value = _text(section).casefold()
    if "deprecated" in value:
        return "deprecated"
    if "quarantine" in value:
        return "quarantine"
    if "backlog" in value or "legacy / unsorted backlog" in value:
        return "backlog"
    if "curated" in value:
        return "curated"
    return "unclassified"


def _metadata_compatible(entry: Mapping[str, object], item: WorkflowItem) -> bool:
    year = _text(entry.get("year"))
    state = _text(entry.get("state"))
    contest = _text(entry.get("contest"))

    if item.election_year is not None and year:
        try:
            if int(year) != int(item.election_year):
                return False
        except (TypeError, ValueError):
            return False
    if item.state and state and state.casefold() != _text(item.state).casefold():
        return False
    if (
        item.contest
        and contest
        and contest.casefold() != _text(item.contest).casefold()
    ):
        return False
    return True


def _exact_registry_state(
    item: WorkflowItem,
    *,
    registry_path: Path,
) -> dict[str, object]:
    source_url = _text(item.source_url)
    if not source_url:
        _deny()

    entries, _ = load_url_registry(registry_path)
    exact = [
        entry
        for entry in entries
        if _text(entry.get("url")) == source_url
    ]
    if not exact:
        _deny()

    compatible = [
        entry for entry in exact if _metadata_compatible(entry, item)
    ]
    if len(compatible) == 1:
        chosen = compatible[0]
    elif len(exact) == 1:
        chosen = exact[0]
    else:
        _deny()

    section = _text(chosen.get("section"))
    category = _category(section)
    review_status = _text(chosen.get("review_status")).casefold()
    quarantined = (
        review_status == "quarantined"
        or "quarantine" in section.casefold()
    )
    deprecated = (
        category == "deprecated"
        or "deprecated" in section.casefold()
    )

    return {
        "source_url": source_url,
        "exact_registry_identity": True,
        "registry_category": category,
        "review_status": review_status,
        "parser_eligible": chosen.get("parser_eligible") is True,
        "quarantined": quarantined,
        "deprecated": deprecated,
        "registry_line": chosen.get("line"),
        "registry_section": section,
    }


def _workflow_qc_evidence(
    session: Session,
    *,
    current_item: WorkflowItem,
    source_url: str,
) -> dict[str, object] | None:
    candidates = session.execute(
        select(WorkflowItem)
        .where(
            WorkflowItem.source_url == source_url,
            WorkflowItem.id != current_item.id,
        )
        .order_by(WorkflowItem.updated_at.desc(), WorkflowItem.id)
    ).scalars().all()

    for prior in candidates:
        try:
            authority = load_workflow_publication_approval_authority(
                session,
                prior.id,
                require_ready_state=False,
            )
        except WorkflowQCReviewConflict:
            continue

        prior_item = authority.get("item")
        selected = authority.get("selected_pass")
        qc1 = authority.get("qc1_review")
        qc2 = authority.get("qc2_review")
        event = authority.get("qc2_event")
        if (
            not isinstance(prior_item, WorkflowItem)
            or _text(prior_item.source_url) != source_url
            or selected is None
            or qc1 is None
            or qc2 is None
            or event is None
            or authority.get("open_discrepancy_count") != 0
        ):
            continue

        return {
            "source_url": source_url,
            "qc_backed": True,
            "provenance_class": PROVENANCE_WORKFLOW_QC,
            "prior_workflow_item_id": str(prior_item.id),
            "qc1_review_id": str(qc1.id),
            "qc2_review_id": str(qc2.id),
            "selected_pass_id": str(selected.id),
            "qc2_event_id": str(event.id),
        }
    return None


def _artifact_is_structured(
    artifact: CanonicalSourceArtifact | None,
    *,
    role_token: str,
) -> bool:
    if artifact is None:
        return False
    role = _text(artifact.artifact_role).casefold()
    digest = _text(artifact.sha256).casefold()
    provenance = artifact.provenance
    return bool(
        role_token in role
        and re.fullmatch(r"[0-9a-f]{64}", digest)
        and isinstance(provenance, Mapping)
        and bool(provenance)
    )


def _canonical_bootstrap_evidence(
    session: Session,
    *,
    item: WorkflowItem,
    source_url: str,
) -> dict[str, object] | None:
    if item.canonical_race_id is None:
        return None

    race = session.get(CanonicalElectionRace, item.canonical_race_id)
    if race is None or race.id != item.canonical_race_id:
        return None
    if (
        _text(race.source_url) != source_url
        or _text(race.production_status).casefold() != "prod_loaded"
        or _text(race.verification_status).casefold() != "verified"
        or race.verified_at is None
        or _text(race.selected_dl_source) not in {"DL1", "DL2"}
        or not isinstance(race.qa_metadata, Mapping)
        or not race.qa_metadata
        or race.payload_artifact_id == race.approval_artifact_id
    ):
        return None

    payload_artifact = session.get(
        CanonicalSourceArtifact,
        race.payload_artifact_id,
    )
    approval_artifact = session.get(
        CanonicalSourceArtifact,
        race.approval_artifact_id,
    )
    if not _artifact_is_structured(payload_artifact, role_token="payload"):
        return None
    if not _artifact_is_structured(approval_artifact, role_token="approval"):
        return None

    events = session.execute(
        select(CanonicalVerificationEvent).where(
            CanonicalVerificationEvent.race_id == race.id
        )
    ).scalars().all()
    matching_events = [
        event
        for event in events
        if (
            _text(event.stage)
            and _text(event.status)
            and _text(event.selected_dl_source) == _text(race.selected_dl_source)
            and isinstance(event.event_metadata, Mapping)
            and bool(event.event_metadata)
        )
    ]
    if not matching_events:
        return None

    event = matching_events[-1]
    return {
        "source_url": source_url,
        "qc_backed": True,
        "provenance_class": PROVENANCE_LEGACY_PRODUCTION,
        "canonical_race_id": str(race.id),
        "payload_artifact_id": str(payload_artifact.id),
        "approval_artifact_id": str(approval_artifact.id),
        "verification_event_id": str(event.id),
    }


def _qc_evidence(
    session: Session,
    *,
    item: WorkflowItem,
    source_url: str,
) -> dict[str, object]:
    workflow = _workflow_qc_evidence(
        session,
        current_item=item,
        source_url=source_url,
    )
    if workflow is not None:
        return workflow

    canonical = _canonical_bootstrap_evidence(
        session,
        item=item,
        source_url=source_url,
    )
    if canonical is not None:
        return canonical

    _deny()


def build_workflow_ballot_lens_server_context(
    session: Session,
    payload: Mapping[str, object] | None,
    *,
    registry_path: Path,
) -> WorkflowBallotLensServerContext:
    try:
        request = validate_workflow_execution_request(payload)
    except WorkflowBallotLensExecutionDenied as exc:
        _deny(exc)

    item_id = _uuid(request["workflow_item_id"])
    pass_id = _uuid(request["workflow_pass_id"])

    item = session.execute(
        select(WorkflowItem)
        .where(WorkflowItem.id == item_id)
        .with_for_update()
    ).scalar_one_or_none()
    if item is None:
        _deny()

    workflow_pass = session.execute(
        select(WorkflowPass)
        .where(
            WorkflowPass.id == pass_id,
            WorkflowPass.workflow_item_id == item.id,
        )
        .with_for_update()
    ).scalar_one_or_none()
    if workflow_pass is None:
        _deny()

    if (
        item.lifecycle_state,
        item.current_stage,
        item.stage_condition,
    ) != ("active", "independent_acquisition", "in_progress"):
        _deny()

    source_url = _text(item.source_url)
    if not source_url:
        _deny()

    registry_state = _exact_registry_state(
        item,
        registry_path=registry_path,
    )
    qc_evidence = _qc_evidence(
        session,
        item=item,
        source_url=source_url,
    )

    return WorkflowBallotLensServerContext(
        workflow_item_id=str(item.id),
        workflow_pass_id=str(workflow_pass.id),
        row_version=int(item.row_version),
        assigned_principal=_text(workflow_pass.assigned_principal),
        pass_is_current=workflow_pass.is_current is True,
        pass_status=_text(workflow_pass.status),
        source_url=source_url,
        qc_evidence=qc_evidence,
        registry_state=registry_state,
    )
