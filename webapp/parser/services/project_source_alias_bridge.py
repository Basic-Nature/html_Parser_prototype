"""J2 Project -> public Ballot Lens alias handoff (read-only, no execution).

A Project reference is not an execution credential. Every call rechecks the
owner and current Source Registry revision, then independently resolves the
published public alias. Never return exact URLs or parser authorization.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from webapp.parser.models.election_project import ElectionProject, ElectionProjectSourceRef
from webapp.parser.services.election_projects import ProjectError
from webapp.parser.utils.models import (
    SourceRegistryAlias,
    SourceRegistryBinding,
    SourceRegistryRevision,
    SourceRegistrySource,
)

_PUBLIC_ALIAS = re.compile(r"\Ablsrc_v1_[0-9a-f]{64}\Z")
_CONTRACT = "project_source_alias_handoff_v1"


def _deny() -> None:
    # Do not disclose whether a foreign ref, URL or alias exists.
    raise ProjectError("project_source_handoff_unavailable", 409)


def resolve_project_source_alias(
    session: Session,
    *,
    project_id: UUID,
    source_ref_id: UUID,
    expected_project_version: int,
    owner_key: str,
    registry_path: Path,
    public_resolver: Callable[[Path, str], object] | None = None,
) -> dict[str, object]:
    """Resolve an owner-controlled Project ref to a current public alias.

    The optional public_resolver is injectable for isolated tests only; the
    normal route always uses the configured live Source Registry read model.
    Neither path creates parser runs, source writes or execution credentials.
    """
    if not isinstance(project_id, UUID) or not isinstance(source_ref_id, UUID):
        _deny()
    if not isinstance(owner_key, str) or len(owner_key) != 64:
        _deny()
    if (type(expected_project_version) is not int or
            not 1 <= expected_project_version <= 2147483647):
        raise ProjectError("invalid_project_version")

    project = session.execute(select(ElectionProject).where(
        ElectionProject.id == project_id,
        ElectionProject.owner_key == owner_key,
    )).scalar_one_or_none()
    if project is None:
        raise ProjectError("project_not_found", 404)
    if project.lifecycle != "active":
        _deny()
    if project.row_version != expected_project_version:
        raise ProjectError("project_version_conflict", 409)

    source_ref = session.execute(select(ElectionProjectSourceRef).where(
        ElectionProjectSourceRef.id == source_ref_id,
        ElectionProjectSourceRef.project_id == project_id,
    )).scalar_one_or_none()
    if source_ref is None:
        _deny()

    binding = session.get(SourceRegistryBinding, source_ref.registry_binding_id)
    revision = session.get(SourceRegistryRevision, source_ref.registry_revision_id)
    if (binding is None or revision is None or
            binding.current_revision_id != source_ref.registry_revision_id or
            revision.source_id != binding.source_id or
            binding.review_state != "approved" or
            binding.public_eligible is not True or
            binding.parser_eligible is not True):
        _deny()

    source = session.get(SourceRegistrySource, binding.source_id)
    if source is None or source.lifecycle_state != "active":
        _deny()

    # A single current, active legacy alias is required. v2-only sources stay
    # ineligible until the public selection/runtime accepts v2 end to end.
    alias_rows = session.execute(select(SourceRegistryAlias).where(
        SourceRegistryAlias.binding_id == binding.id,
        SourceRegistryAlias.alias_type == "legacy_blsrc_v1",
        SourceRegistryAlias.active.is_(True),
    ).limit(2)).scalars().all()
    if len(alias_rows) != 1:
        _deny()
    alias = alias_rows[0].alias_value
    if not isinstance(alias, str) or not _PUBLIC_ALIAS.fullmatch(alias):
        _deny()

    # Defensive collision check despite the table's unique alias_value guard.
    matching = session.execute(select(SourceRegistryAlias.id).where(
        SourceRegistryAlias.alias_value == alias,
    ).limit(2)).all()
    if len(matching) != 1:
        _deny()

    if public_resolver is None:
        from webapp.parser.services.source_registry_runtime import resolve_public_registry_source
        public_resolver = resolve_public_registry_source
    try:
        published = public_resolver(registry_path, alias)
    except Exception:
        # A read-model outage is not proof that no election source exists.
        _deny()
    if published is None:
        _deny()

    # Compare exact URL only on the server. Never place it in returned JSON.
    # This also prevents an alias from a different public registry mode or
    # a stale legacy registry row from being used as a Project handoff.
    if (getattr(published, "registry_source_id", None) != alias or
            getattr(published, "registry_category", None) != "curated" or
            getattr(published, "url", None) != revision.exact_url):
        _deny()
    for key, value in (
        ("year", binding.year), ("state", binding.state),
        ("contest", binding.contest), ("registry_scope", binding.scope),
        ("registry_format", binding.format),
    ):
        if str(getattr(published, key, "")) != str(value):
            _deny()

    return {
        "contract": _CONTRACT,
        "project_id": str(project.id),
        "project_row_version": project.row_version,
        "source_ref_id": str(source_ref.id),
        "registry_source_id": alias,
        "source_label": {
            "year": binding.year, "state": binding.state,
            "contest": binding.contest, "scope": binding.scope,
            "format": binding.format,
        },
        "execution_authorized": False,
        "run_dispatched": False,
        "source_url_disclosed": False,
    }
