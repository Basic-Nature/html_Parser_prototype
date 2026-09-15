"""Source Registry read-model boundary.

The default remains legacy_file. shadow_db compares a caller-injected durable
reader while returning legacy authority. durable_db never silently falls back
to the file registry. Importing this module performs no DB or network access.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol

from webapp.parser.utils.url_registry import (
    find_url_registry_entries,
    list_public_registry_sources,
    load_url_registry,
    lookup_exact_registry_entry,
    resolve_public_registry_source,
    source_registry_authority_mode,
)

SOURCE_REGISTRY_READ_MODEL_CONTRACT = "source_registry_read_model_v1"


class SourceRegistryReadModelError(RuntimeError):
    pass


class SourceRegistryShadowMismatch(SourceRegistryReadModelError):
    pass


class SourceRegistryDbReader(Protocol):
    def list_public_sources(self) -> list[dict[str, object]]: ...
    def resolve_public_source_alias(self, alias: str) -> dict[str, object] | None: ...
    def resolve_trusted_exact_source(self, source_url: str) -> dict[str, object] | None: ...
    def resolve_workflow_binding(self, source_url: str) -> dict[str, object] | None: ...
    def list_trusted_registry_entries(self) -> list[dict[str, object]]: ...
    def resolve_legacy_url_compatibility(self, source_url: str) -> dict[str, object]: ...


def _public_projection(source: object) -> dict[str, object] | None:
    if source is None:
        return None
    if isinstance(source, Mapping):
        return dict(source)
    return {
        "registry_source_id": getattr(source, "registry_source_id"),
        "year": getattr(source, "year"),
        "contest": getattr(source, "contest"),
        "state": getattr(source, "state"),
        "scope": getattr(source, "registry_scope"),
        "format": getattr(source, "registry_format"),
        "registry_category": getattr(source, "registry_category"),
    }


def _trusted_projection(entry: object) -> dict[str, object] | None:
    if entry is None:
        return None
    if isinstance(entry, Mapping):
        return dict(entry)
    return {
        "year": getattr(entry, "year"),
        "contest": getattr(entry, "contest"),
        "state": getattr(entry, "state"),
        "scope": getattr(entry, "registry_scope"),
        "format": getattr(entry, "registry_format"),
        "notes": getattr(entry, "notes"),
        "url": getattr(entry, "url"),
        "registry_category": getattr(entry, "registry_category"),
    }


class SourceRegistryReadModel:
    def __init__(
        self,
        registry_path: str | Path,
        *,
        mode: str | None = None,
        db_reader: SourceRegistryDbReader | None = None,
        mismatch_recorder: Callable[[str, object, object], None] | None = None,
    ) -> None:
        self.registry_path = Path(registry_path)
        self.mode = source_registry_authority_mode() if mode is None else str(mode).strip()
        if self.mode not in {"legacy_file", "shadow_db", "durable_db"}:
            raise SourceRegistryReadModelError(
                f"unknown Source Registry authority mode: {self.mode!r}"
            )
        self.db_reader = db_reader
        self.mismatch_recorder = mismatch_recorder

    def _db(self) -> SourceRegistryDbReader:
        if self.db_reader is None:
            raise SourceRegistryReadModelError(
                "durable Source Registry reader is unavailable; no file fallback is allowed"
            )
        return self.db_reader

    def _compare(self, operation: str, legacy: object, durable: object) -> None:
        if legacy == durable:
            return
        if self.mismatch_recorder is not None:
            self.mismatch_recorder(operation, legacy, durable)
        raise SourceRegistryShadowMismatch(
            f"Source Registry shadow mismatch for {operation}"
        )

    def list_public_sources(self) -> list[dict[str, object]]:
        if self.mode == "durable_db":
            return list(self._db().list_public_sources())
        legacy = [
            {
                "registry_source_id": item.registry_source_id,
                "year": item.year,
                "contest": item.contest,
                "state": item.state,
                "scope": item.registry_scope,
                "format": item.registry_format,
                "registry_category": item.registry_category,
            }
            for item in list_public_registry_sources(self.registry_path)
        ]
        if self.mode == "shadow_db":
            durable = list(self._db().list_public_sources())
            self._compare("list_public_sources", legacy, durable)
        return legacy

    def resolve_public_source_alias(self, alias: str) -> dict[str, object] | None:
        if self.mode == "durable_db":
            return self._db().resolve_public_source_alias(alias)
        legacy = _public_projection(
            resolve_public_registry_source(self.registry_path, alias)
        )
        if self.mode == "shadow_db":
            durable = self._db().resolve_public_source_alias(alias)
            self._compare("resolve_public_source_alias", legacy, durable)
        return legacy

    def _legacy_exact_source(self, source_url: str) -> dict[str, object] | None:
        return _trusted_projection(
            lookup_exact_registry_entry(source_url, path=self.registry_path)
        )

    def resolve_trusted_exact_source(self, source_url: str) -> dict[str, object] | None:
        if self.mode == "durable_db":
            return self._db().resolve_trusted_exact_source(source_url)
        legacy = self._legacy_exact_source(source_url)
        if self.mode == "shadow_db":
            durable = self._db().resolve_trusted_exact_source(source_url)
            self._compare("resolve_trusted_exact_source", legacy, durable)
        return legacy

    def resolve_workflow_binding(self, source_url: str) -> dict[str, object] | None:
        if self.mode == "durable_db":
            return self._db().resolve_workflow_binding(source_url)
        entry = self._legacy_exact_source(source_url)
        legacy = None
        if entry and str(entry.get("registry_category") or "").lower() == "curated":
            legacy = dict(entry)
        if self.mode == "shadow_db":
            durable = self._db().resolve_workflow_binding(source_url)
            self._compare("resolve_workflow_binding", legacy, durable)
        return legacy

    def list_trusted_registry_entries(self) -> list[dict[str, object]]:
        if self.mode == "durable_db":
            return list(self._db().list_trusted_registry_entries())
        legacy, _ = load_url_registry(self.registry_path)
        if self.mode == "shadow_db":
            durable = list(self._db().list_trusted_registry_entries())
            self._compare("list_trusted_registry_entries", legacy, durable)
        return legacy

    def resolve_legacy_url_compatibility(self, source_url: str) -> dict[str, object]:
        if self.mode == "durable_db":
            return self._db().resolve_legacy_url_compatibility(source_url)
        matches = find_url_registry_entries(self.registry_path, source_url)
        legacy = {
            "matched": bool(matches),
            "parser_eligible": any(
                item.get("parser_eligible") is True for item in matches
            ),
            "review_statuses": sorted({
                str(item.get("review_status") or "") for item in matches
            }),
            "normalized_urls": sorted({
                str(item.get("normalized_url") or "") for item in matches
            }),
        }
        if self.mode == "shadow_db":
            durable = self._db().resolve_legacy_url_compatibility(source_url)
            self._compare("resolve_legacy_url_compatibility", legacy, durable)
        return legacy
