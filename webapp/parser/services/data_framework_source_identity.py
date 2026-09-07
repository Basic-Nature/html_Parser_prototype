# Stable Source Evidence identity projection for Data Framework.
#
# This module is read-only and network-free. It never creates a registry
# identifier. It can only return an existing registry_source_id from a
# PublicRegistrySource already supplied by url_registry authority.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from webapp.parser.utils.url_registry import PublicRegistrySource


def _fold(value: Any) -> str:
    return str(value or "").strip().casefold()


def resolve_curated_registry_source_id(
    record: Mapping[str, Any],
    public_sources: Sequence[PublicRegistrySource],
    *,
    year: str | None = None,
) -> str | None:
    # Exact source URL text is mandatory. If the URL identifies one approved
    # public row, that existing row is authority. Multiple semantic rows sharing
    # the URL may be disambiguated only by available curated metadata.
    source_url = str(record.get("source_url") or "")
    if not source_url:
        return None

    candidates = [
        source
        for source in public_sources
        if source.url == source_url
    ]
    if len(candidates) == 1:
        return candidates[0].registry_source_id
    if not candidates:
        return None

    dimensions = (
        (_fold(record.get("state")), "state"),
        (_fold(year), "year"),
        (_fold(record.get("contest")), "contest"),
        (_fold(record.get("county")), "registry_scope"),
    )
    for expected, attribute in dimensions:
        if not expected:
            continue
        candidates = [
            source
            for source in candidates
            if _fold(getattr(source, attribute, "")) == expected
        ]
        if not candidates:
            return None

    if len(candidates) != 1:
        return None
    return candidates[0].registry_source_id
