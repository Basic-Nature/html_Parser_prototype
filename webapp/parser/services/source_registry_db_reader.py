"""Read-only durable Source Registry adapter for shadow/durable read models.

This module performs no work at import time. Every DB operation starts a
read-only transaction on PostgreSQL and always rolls back/closes the Session.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from webapp.parser.utils.url_registry import normalize_url_for_match


SOURCE_REGISTRY_DB_READER_CONTRACT = "source_registry_db_reader_v1"


class SourceRegistryDbReadError(RuntimeError):
    pass


class SourceRegistryDbAmbiguity(SourceRegistryDbReadError):
    pass


def _category(review_state: object) -> str:
    value = str(review_state or "").strip().lower()
    if value == "approved":
        return "curated"
    if value == "backlog":
        return "backlog"
    if value == "quarantined":
        return "quarantine"
    return "unclassified"


def _review_status(review_state: object) -> str:
    return "quarantined" if str(review_state or "").strip().lower() == "quarantined" else "approved"


def _trusted_projection(binding: Any, revision: Any) -> dict[str, object]:
    scope = str(binding.scope or "")
    return {
        "year": str(binding.year or ""),
        "contest": str(binding.contest or ""),
        "state": str(binding.state or ""),
        "scope": scope,
        "format": str(binding.format or ""),
        "notes": str(binding.notes or ""),
        "url": str(revision.exact_url or ""),
        "county": None if not scope or scope in {"-", "statewide"} else scope,
        "registry_category": _category(binding.review_state),
        "review_status": _review_status(binding.review_state),
        "parser_eligible": binding.parser_eligible is True,
        "normalized_url": str(revision.normalized_url or ""),
    }


def _public_projection(alias: Any, binding: Any) -> dict[str, object]:
    return {
        "registry_source_id": str(alias.alias_value or ""),
        "year": str(binding.year or ""),
        "contest": str(binding.contest or ""),
        "state": str(binding.state or ""),
        "scope": str(binding.scope or ""),
        "format": str(binding.format or ""),
        "registry_category": "curated",
    }


def _public_sort_key(item: dict[str, object]) -> tuple[str, ...]:
    return (
        str(item.get("year") or ""),
        str(item.get("state") or ""),
        str(item.get("contest") or ""),
        str(item.get("scope") or ""),
        str(item.get("format") or ""),
        str(item.get("registry_source_id") or ""),
    )


def _trusted_sort_key(item: dict[str, object]) -> tuple[str, ...]:
    return (
        str(item.get("year") or ""),
        str(item.get("state") or ""),
        str(item.get("contest") or ""),
        str(item.get("scope") or ""),
        str(item.get("format") or ""),
        str(item.get("url") or ""),
        str(item.get("registry_category") or ""),
        str(item.get("notes") or ""),
    )


class SqlAlchemySourceRegistryDbReader:
    """Concrete read-only implementation of the durable reader protocol."""

    def __init__(self, *, engine_factory: Callable[[], Any] | None = None) -> None:
        self._engine_factory = engine_factory

    def _engine(self) -> Any:
        if self._engine_factory is not None:
            return self._engine_factory()
        from webapp.parser.utils.db_utils import get_engine
        return get_engine()

    def _read(self, callback: Callable[[Session], Any]) -> Any:
        session = Session(
            bind=self._engine(),
            autoflush=False,
            expire_on_commit=False,
        )
        try:
            bind = session.get_bind()
            if str(bind.dialect.name or "").lower() == "postgresql":
                session.execute(text("SET TRANSACTION READ ONLY"))
            return callback(session)
        finally:
            session.rollback()
            session.close()

    @staticmethod
    def _models():
        from webapp.parser.utils.models import (
            SourceRegistryAlias,
            SourceRegistryBinding,
            SourceRegistryRevision,
            SourceRegistrySource,
        )
        return (
            SourceRegistryAlias,
            SourceRegistryBinding,
            SourceRegistryRevision,
            SourceRegistrySource,
        )

    def list_public_sources(self) -> list[dict[str, object]]:
        Alias, Binding, _Revision, Source = self._models()

        def read(session: Session) -> list[dict[str, object]]:
            rows = session.execute(
                select(Alias, Binding)
                .join(Binding, Alias.binding_id == Binding.id)
                .join(Source, Binding.source_id == Source.id)
                .where(
                    Alias.alias_type == "legacy_blsrc_v1",
                    Alias.active.is_(True),
                    Binding.public_eligible.is_(True),
                    Binding.review_state == "approved",
                    Source.lifecycle_state == "active",
                )
            ).all()
            return sorted(
                [_public_projection(alias, binding) for alias, binding in rows],
                key=_public_sort_key,
            )

        return self._read(read)

    def resolve_public_source_alias(self, alias: str) -> dict[str, object] | None:
        Alias, Binding, _Revision, Source = self._models()
        wanted = str(alias or "").strip()

        def read(session: Session) -> dict[str, object] | None:
            rows = session.execute(
                select(Alias, Binding)
                .join(Binding, Alias.binding_id == Binding.id)
                .join(Source, Binding.source_id == Source.id)
                .where(
                    Alias.alias_value == wanted,
                    Alias.alias_type == "legacy_blsrc_v1",
                    Alias.active.is_(True),
                    Binding.public_eligible.is_(True),
                    Binding.review_state == "approved",
                    Source.lifecycle_state == "active",
                )
            ).all()
            if not rows:
                return None
            if len(rows) != 1:
                raise SourceRegistryDbAmbiguity(
                    "public alias resolved to multiple durable bindings"
                )
            return _public_projection(*rows[0])

        return self._read(read)

    def resolve_public_execution_source(self, alias: str) -> dict[str, object] | None:
        Alias, Binding, Revision, Source = self._models()
        wanted = str(alias or "").strip()

        def read(session: Session) -> dict[str, object] | None:
            rows = session.execute(
                select(Alias, Binding, Revision)
                .join(Binding, Alias.binding_id == Binding.id)
                .join(Revision, Binding.current_revision_id == Revision.id)
                .join(Source, Binding.source_id == Source.id)
                .where(
                    Alias.alias_value == wanted,
                    Alias.alias_type == "legacy_blsrc_v1",
                    Alias.active.is_(True),
                    Binding.public_eligible.is_(True),
                    Binding.review_state == "approved",
                    Source.lifecycle_state == "active",
                )
            ).all()
            if not rows:
                return None
            if len(rows) != 1:
                raise SourceRegistryDbAmbiguity(
                    "public execution alias resolved ambiguously"
                )
            alias_row, binding, revision = rows[0]
            result = _public_projection(alias_row, binding)
            result["url"] = str(revision.exact_url or "")
            return result

        return self._read(read)

    def list_trusted_exact_sources(self, source_url: str) -> list[dict[str, object]]:
        _Alias, Binding, Revision, Source = self._models()
        wanted = str(source_url or "")

        def read(session: Session) -> list[dict[str, object]]:
            rows = session.execute(
                select(Binding, Revision)
                .join(Revision, Binding.current_revision_id == Revision.id)
                .join(Source, Binding.source_id == Source.id)
                .where(Revision.exact_url == wanted)
            ).all()
            return sorted(
                [_trusted_projection(binding, revision) for binding, revision in rows],
                key=_trusted_sort_key,
            )

        return self._read(read)

    def resolve_trusted_exact_source(self, source_url: str) -> dict[str, object] | None:
        rows = self.list_trusted_exact_sources(source_url)
        if not rows:
            return None
        if len(rows) != 1:
            raise SourceRegistryDbAmbiguity(
                "exact URL maps to multiple durable semantic bindings"
            )
        return rows[0]

    def resolve_workflow_binding(self, source_url: str) -> dict[str, object] | None:
        _Alias, Binding, Revision, Source = self._models()
        wanted = str(source_url or "")

        def read(session: Session) -> dict[str, object] | None:
            rows = session.execute(
                select(Binding, Revision)
                .join(Revision, Binding.current_revision_id == Revision.id)
                .join(Source, Binding.source_id == Source.id)
                .where(
                    Revision.exact_url == wanted,
                    Binding.workflow_eligible.is_(True),
                    Binding.review_state == "approved",
                    Source.lifecycle_state == "active",
                )
            ).all()
            if not rows:
                return None
            if len(rows) != 1:
                raise SourceRegistryDbAmbiguity(
                    "workflow exact URL resolved ambiguously"
                )
            return _trusted_projection(*rows[0])

        return self._read(read)

    def list_trusted_registry_entries(self) -> list[dict[str, object]]:
        _Alias, Binding, Revision, Source = self._models()

        def read(session: Session) -> list[dict[str, object]]:
            rows = session.execute(
                select(Binding, Revision)
                .join(Revision, Binding.current_revision_id == Revision.id)
                .join(Source, Binding.source_id == Source.id)
            ).all()
            return sorted(
                [_trusted_projection(binding, revision) for binding, revision in rows],
                key=_trusted_sort_key,
            )

        return self._read(read)

    def resolve_legacy_url_compatibility(self, source_url: str) -> dict[str, object]:
        _Alias, Binding, Revision, _Source = self._models()
        wanted = normalize_url_for_match(source_url)
        if not wanted:
            return {
                "matched": False,
                "parser_eligible": False,
                "review_statuses": [],
                "normalized_urls": [],
            }

        def read(session: Session) -> dict[str, object]:
            rows = session.execute(
                select(Binding, Revision)
                .join(Revision, Binding.current_revision_id == Revision.id)
                .where(Revision.normalized_url == wanted)
            ).all()
            return {
                "matched": bool(rows),
                "parser_eligible": any(
                    binding.parser_eligible is True for binding, _ in rows
                ),
                "review_statuses": sorted({
                    _review_status(binding.review_state) for binding, _ in rows
                }),
                "normalized_urls": sorted({
                    str(revision.normalized_url or "") for _, revision in rows
                }),
            }

        return self._read(read)
