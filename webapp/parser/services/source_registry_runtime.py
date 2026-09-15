"""Compatibility runtime seam for Source Registry authority modes.

legacy_file preserves existing helper signatures and performs no DB access.
shadow_db compares durable reads but returns legacy authority. durable_db never
silently falls back; raw file-only diagnostics remain fail-closed until a
separate durable diagnostics contract is accepted.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from webapp.parser.services.source_registry_read_model import (
    SourceRegistryReadModel,
    SourceRegistryReadModelError,
    SourceRegistryShadowMismatch,
)
from webapp.parser.utils import url_registry as _legacy_registry

ContributorRegistryEntry = _legacy_registry.ContributorRegistryEntry
PublicRegistryResolutionError = _legacy_registry.PublicRegistryResolutionError
PublicRegistrySource = _legacy_registry.PublicRegistrySource


SOURCE_REGISTRY_RUNTIME_CONTRACT = "source_registry_runtime_v1"


def _payload_sha(value: object) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _default_mismatch_recorder(
    operation: str,
    legacy: object,
    durable: object,
) -> None:
    from webapp.parser.utils.logger_singleton import logger
    logger.error({
        "level": "ERROR",
        "type": "source_registry_shadow_mismatch",
        "message": "Source Registry shadow parity mismatch.",
        "operation": operation,
        "legacy_sha256": _payload_sha(legacy),
        "durable_sha256": _payload_sha(durable),
    })


def _compare(
    operation: str,
    legacy: object,
    durable: object,
    recorder: Callable[[str, object, object], None],
) -> None:
    if legacy == durable:
        return
    recorder(operation, legacy, durable)
    raise SourceRegistryShadowMismatch(
        f"Source Registry shadow mismatch for {operation}"
    )


def build_source_registry_read_model(
    registry_path: str | Path,
    *,
    mode: str | None = None,
    db_reader: object | None = None,
    mismatch_recorder: Callable[[str, object, object], None] | None = None,
) -> SourceRegistryReadModel:
    chosen = (
        _legacy_registry.source_registry_authority_mode()
        if mode is None
        else str(mode or "").strip().lower()
    )
    reader = db_reader
    if chosen in {"shadow_db", "durable_db"} and reader is None:
        from webapp.parser.services.source_registry_db_reader import (
            SqlAlchemySourceRegistryDbReader,
        )
        reader = SqlAlchemySourceRegistryDbReader()
    return SourceRegistryReadModel(
        registry_path,
        mode=chosen,
        db_reader=reader,
        mismatch_recorder=mismatch_recorder or _default_mismatch_recorder,
    )


def _category_from_legacy(entry: dict[str, object]) -> str:
    section = str(entry.get("section") or "").lower()
    if "quarantine" in section:
        return "quarantine"
    if "backlog" in section or "legacy / unsorted backlog" in section:
        return "backlog"
    if "curated" in section:
        return "curated"
    return "unclassified"


def _semantic_legacy_entry(entry: dict[str, object]) -> dict[str, object]:
    scope = str(entry.get("scope") or "")
    return {
        "year": str(entry.get("year") or ""),
        "contest": str(entry.get("contest") or ""),
        "state": str(entry.get("state") or ""),
        "scope": scope,
        "format": str(entry.get("format") or ""),
        "notes": str(entry.get("notes") or ""),
        "url": str(entry.get("url") or ""),
        "county": None if not scope or scope in {"-", "statewide"} else scope,
        "registry_category": _category_from_legacy(entry),
        "review_status": str(entry.get("review_status") or ""),
        "parser_eligible": entry.get("parser_eligible") is True,
        "normalized_url": str(entry.get("normalized_url") or ""),
    }


def _trusted_key(item: dict[str, object]) -> tuple[str, ...]:
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


def _public_execution_projection(
    source: PublicRegistrySource | None,
) -> dict[str, object] | None:
    if source is None:
        return None
    return {
        "registry_source_id": source.registry_source_id,
        "year": source.year,
        "contest": source.contest,
        "state": source.state,
        "scope": source.registry_scope,
        "format": source.registry_format,
        "registry_category": source.registry_category,
        "url": source.url,
    }


def _public_from_mapping(value: dict[str, object]) -> PublicRegistrySource:
    return PublicRegistrySource(
        registry_source_id=str(value.get("registry_source_id") or ""),
        year=str(value.get("year") or ""),
        contest=str(value.get("contest") or ""),
        state=str(value.get("state") or ""),
        registry_scope=str(value.get("scope") or ""),
        registry_format=str(value.get("format") or ""),
        registry_category=str(value.get("registry_category") or ""),
        url=str(value.get("url") or ""),
    )


def _contributor_projection(
    entry: ContributorRegistryEntry | None,
) -> dict[str, object] | None:
    if entry is None:
        return None
    return {
        "year": entry.year,
        "contest": entry.contest,
        "state": entry.state,
        "scope": entry.registry_scope,
        "format": entry.registry_format,
        "notes": entry.notes,
        "url": entry.url,
        "registry_category": entry.registry_category,
    }


def _contributor_from_mapping(
    value: dict[str, object],
) -> ContributorRegistryEntry:
    return ContributorRegistryEntry(
        year=str(value.get("year") or ""),
        contest=str(value.get("contest") or ""),
        state=str(value.get("state") or ""),
        registry_scope=str(value.get("scope") or ""),
        registry_format=str(value.get("format") or ""),
        notes=str(value.get("notes") or ""),
        url=str(value.get("url") or ""),
        registry_category=str(value.get("registry_category") or ""),
    )


def project_public_registry_sources(
    path: str | Path,
) -> list[dict[str, object]]:
    model = build_source_registry_read_model(path)
    return list(model.list_public_sources())


def resolve_public_registry_source(
    path: str | Path,
    registry_source_id: str,
) -> PublicRegistrySource | None:
    mode = _legacy_registry.source_registry_authority_mode()
    if mode == "legacy_file":
        return _legacy_registry.resolve_public_registry_source(path, registry_source_id)

    model = build_source_registry_read_model(path, mode=mode)
    model.resolve_public_source_alias(registry_source_id)

    legacy = _legacy_registry.resolve_public_registry_source(path, registry_source_id)
    reader = model.db_reader
    if reader is None or not hasattr(reader, "resolve_public_execution_source"):
        raise SourceRegistryReadModelError(
            "durable Source Registry execution resolver is unavailable"
        )
    durable = reader.resolve_public_execution_source(registry_source_id)

    if mode == "shadow_db":
        _compare(
            "resolve_public_execution_source",
            _public_execution_projection(legacy),
            durable,
            model.mismatch_recorder or _default_mismatch_recorder,
        )
        return legacy

    if durable is None:
        return None
    return _public_from_mapping(dict(durable))


def is_parser_eligible_url(
    path: str | Path,
    url: str,
) -> tuple[bool, str]:
    mode = _legacy_registry.source_registry_authority_mode()
    if mode == "legacy_file":
        return _legacy_registry.is_parser_eligible_url(path, url)

    model = build_source_registry_read_model(path, mode=mode)
    state = model.resolve_legacy_url_compatibility(url)
    matched = state.get("matched") is True
    eligible = state.get("parser_eligible") is True
    statuses = {
        str(value or "")
        for value in (state.get("review_statuses") or [])
    }
    if eligible:
        return True, "approved_registry"
    if matched and "quarantined" in statuses:
        return False, "registry_quarantined"
    if matched:
        return False, "registry_not_parser_eligible"
    return False, "url_not_in_approved_registry"


def lookup_exact_registry_entry(
    url: str,
    *,
    path: str | Path,
) -> ContributorRegistryEntry | None:
    mode = _legacy_registry.source_registry_authority_mode()
    if mode == "legacy_file":
        return _legacy_registry.lookup_exact_registry_entry(url, path=path)

    model = build_source_registry_read_model(path, mode=mode)
    reader = model.db_reader
    if reader is None or not hasattr(reader, "list_trusted_exact_sources"):
        raise SourceRegistryReadModelError(
            "durable Source Registry exact-source reader is unavailable"
        )

    durable_rows = [
        dict(item)
        for item in reader.list_trusted_exact_sources(url)
    ]
    legacy = _legacy_registry.lookup_exact_registry_entry(url, path=path)
    legacy_projection = _contributor_projection(legacy)

    comparable = [
        {
            "year": str(item.get("year") or ""),
            "contest": str(item.get("contest") or ""),
            "state": str(item.get("state") or ""),
            "scope": str(item.get("scope") or ""),
            "format": str(item.get("format") or ""),
            "notes": str(item.get("notes") or ""),
            "url": str(item.get("url") or ""),
            "registry_category": str(item.get("registry_category") or ""),
        }
        for item in durable_rows
    ]

    if mode == "shadow_db":
        if legacy_projection is None:
            _compare(
                "lookup_exact_registry_entry",
                [],
                comparable,
                model.mismatch_recorder or _default_mismatch_recorder,
            )
            return None
        if legacy_projection not in comparable:
            _compare(
                "lookup_exact_registry_entry",
                legacy_projection,
                comparable,
                model.mismatch_recorder or _default_mismatch_recorder,
            )
        return legacy

    if not comparable:
        return None
    if len(comparable) != 1:
        raise SourceRegistryReadModelError(
            "durable exact URL maps to multiple semantic bindings"
        )
    return _contributor_from_mapping(comparable[0])


def load_url_registry(
    path: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    entries, diagnostics = _legacy_registry.load_url_registry(path)
    mode = _legacy_registry.source_registry_authority_mode()
    if mode == "legacy_file":
        return entries, diagnostics

    model = build_source_registry_read_model(path, mode=mode)
    reader = model.db_reader
    if reader is None:
        raise SourceRegistryReadModelError(
            "durable Source Registry reader is unavailable"
        )

    if mode == "shadow_db":
        legacy_semantic = sorted(
            (_semantic_legacy_entry(dict(item)) for item in entries),
            key=_trusted_key,
        )
        durable_semantic = sorted(
            (dict(item) for item in reader.list_trusted_registry_entries()),
            key=_trusted_key,
        )
        _compare(
            "load_url_registry_semantic_projection",
            legacy_semantic,
            durable_semantic,
            model.mismatch_recorder or _default_mismatch_recorder,
        )
        # Shadow mode deliberately returns the exact legacy raw representation.
        return entries, diagnostics

    raise SourceRegistryReadModelError(
        "durable_db raw registry diagnostics are not yet an accepted "
        "replacement for file-specific section/line metadata"
    )
