# Schema-aware Smart Elections URL registry.
#
# This module is intentionally network-free. It classifies registry entries for
# parser eligibility but does not contact any URL.
#
# Security model:
# - seven-column registry rows outside a Quarantine section are parser-eligible;
# - Quarantine rows remain visible/auditable but are not parser-eligible;
# - malformed or legacy unstructured rows are not parser-eligible;
# - multiple semantic rows may share one normalized URL.

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


REGISTRY_FIELD_NAMES = (
    "year",
    "contest",
    "state",
    "scope",
    "format",
    "notes",
    "url",
)


def normalize_url_for_match(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""

    try:
        parts = urlsplit(raw)
    except Exception:
        return raw

    scheme = (parts.scheme or "").lower()
    host = (parts.hostname or "").lower()

    if not scheme or not host:
        return raw

    try:
        port = parts.port
    except ValueError:
        return raw

    if port is None:
        netloc = host
    elif (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
        netloc = host
    else:
        netloc = f"{host}:{port}"

    return urlunsplit((scheme, netloc, parts.path, parts.query, ""))


def load_url_registry(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    registry_path = Path(path)
    entries: list[dict[str, Any]] = []
    malformed: list[dict[str, Any]] = []
    current_section = "UNSECTIONED"

    if not registry_path.exists():
        return [], {
            "path": str(registry_path),
            "row_count": 0,
            "malformed_row_count": 0,
            "quarantine_row_count": 0,
            "parser_eligible_count": 0,
        }

    for line_number, raw_line in enumerate(
        registry_path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        stripped = raw_line.strip()

        if not stripped:
            continue

        if stripped.startswith("#"):
            if stripped.startswith("# ==="):
                current_section = stripped.lstrip("# ").strip()
            continue

        fields = raw_line.split("\t")
        if len(fields) != len(REGISTRY_FIELD_NAMES):
            malformed.append({
                "line": line_number,
                "field_count": len(fields),
                "section": current_section,
            })
            continue

        item = dict(zip(REGISTRY_FIELD_NAMES, (value.strip() for value in fields)))
        url = item["url"]

        try:
            parts = urlsplit(url)
        except Exception:
            parts = None

        structurally_valid = bool(
            parts
            and parts.scheme.lower() in {"http", "https"}
            and parts.hostname
        )

        quarantined = "quarantine" in current_section.lower()
        review_status = (
            "quarantined"
            if quarantined
            else ("approved" if structurally_valid else "invalid")
        )
        parser_eligible = bool(structurally_valid and not quarantined)

        scope = item["scope"]
        county = None
        if scope and scope not in {"-", "statewide"}:
            county = scope

        entries.append({
            **item,
            "county": county,
            "section": current_section,
            "line": line_number,
            "review_status": review_status,
            "parser_eligible": parser_eligible,
            "normalized_url": normalize_url_for_match(url),
        })

    return entries, {
        "path": str(registry_path),
        "row_count": len(entries),
        "malformed_row_count": len(malformed),
        "malformed_rows": malformed,
        "quarantine_row_count": sum(
            1 for entry in entries if entry["review_status"] == "quarantined"
        ),
        "parser_eligible_count": sum(
            1 for entry in entries if entry["parser_eligible"]
        ),
    }


def find_url_registry_entries(
    path: str | Path,
    url: str,
) -> list[dict[str, Any]]:
    target = normalize_url_for_match(url)
    if not target:
        return []

    entries, _ = load_url_registry(path)
    return [
        entry
        for entry in entries
        if entry.get("normalized_url") == target
    ]


def is_parser_eligible_url(
    path: str | Path,
    url: str,
) -> tuple[bool, str]:
    matches = find_url_registry_entries(path, url)

    if any(entry.get("parser_eligible") for entry in matches):
        return True, "approved_registry"

    if matches:
        if any(entry.get("review_status") == "quarantined" for entry in matches):
            return False, "registry_quarantined"
        return False, "registry_not_parser_eligible"

    return False, "url_not_in_approved_registry"

# ---------------------------------------------------------------------------
# W3 contributor exact-source projection
# ---------------------------------------------------------------------------
# The parser-facing registry helpers above intentionally retain their existing
# normalized matching semantics. Contributor source disclosure is stricter:
# the workflow item's stored source_url must match the maintained registry row
# byte-for-byte as text, and only the Curated section may be disclosed.

from dataclasses import dataclass


@dataclass(frozen=True)
class ContributorRegistryEntry:
    year: str
    contest: str
    state: str
    registry_scope: str
    registry_format: str
    notes: str
    url: str
    registry_category: str


def _contributor_registry_category(entry: dict[str, Any]) -> str:
    section = str(entry.get("section") or "").lower()
    if "quarantine" in section:
        return "quarantine"
    if "backlog" in section or "legacy / unsorted backlog" in section:
        return "backlog"
    if "curated" in section:
        return "curated"
    return "unclassified"



# ---------------------------------------------------------------------------
# Ballot Lens public registry-source projection
# ---------------------------------------------------------------------------
# Public execution is narrower than existing trusted parser eligibility:
# only reviewed Curated rows become anonymous execution authority.

PUBLIC_REGISTRY_SOURCE_ID_PREFIX = "blsrc_v1_"
PUBLIC_REGISTRY_CATEGORIES = frozenset({"curated"})
_PUBLIC_REGISTRY_SOURCE_ID_RE = re.compile(
    r"\Ablsrc_v1_[0-9a-f]{64}\Z"
)


class PublicRegistryResolutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class PublicRegistrySource:
    registry_source_id: str
    year: str
    contest: str
    state: str
    registry_scope: str
    registry_format: str
    registry_category: str
    url: str


def _is_public_registry_entry(entry: dict[str, Any]) -> bool:
    category = _contributor_registry_category(entry)
    return bool(
        entry.get("parser_eligible") is True
        and str(entry.get("review_status") or "") == "approved"
        and category in PUBLIC_REGISTRY_CATEGORIES
    )


def _public_registry_source_id_material(entry: dict[str, Any]) -> bytes:
    category = _contributor_registry_category(entry)
    values = ["blsrc_v1", category]
    values.extend(
        str(entry.get(field_name) or "")
        for field_name in REGISTRY_FIELD_NAMES
    )
    return "\x1f".join(values).encode("utf-8")


def public_registry_source_id_for_entry(
    entry: dict[str, Any],
) -> str | None:
    if not _is_public_registry_entry(entry):
        return None
    digest = hashlib.sha256(
        _public_registry_source_id_material(entry)
    ).hexdigest()
    return f"{PUBLIC_REGISTRY_SOURCE_ID_PREFIX}{digest}"


def _public_registry_source_from_entry(
    entry: dict[str, Any],
) -> PublicRegistrySource:
    registry_source_id = public_registry_source_id_for_entry(entry)
    if not registry_source_id:
        raise PublicRegistryResolutionError(
            "Registry entry is not approved for public parser execution."
        )
    return PublicRegistrySource(
        registry_source_id=registry_source_id,
        year=str(entry.get("year") or ""),
        contest=str(entry.get("contest") or ""),
        state=str(entry.get("state") or ""),
        registry_scope=str(entry.get("scope") or ""),
        registry_format=str(entry.get("format") or ""),
        registry_category=_contributor_registry_category(entry),
        url=str(entry.get("url") or ""),
    )


def list_public_registry_sources(
    path: str | Path,
) -> list[PublicRegistrySource]:
    entries, _ = load_url_registry(path)
    sources: list[PublicRegistrySource] = []
    seen_ids: set[str] = set()
    for entry in entries:
        if not _is_public_registry_entry(entry):
            continue
        source = _public_registry_source_from_entry(entry)
        if source.registry_source_id in seen_ids:
            raise PublicRegistryResolutionError(
                "Duplicate exact public registry source ID detected."
            )
        seen_ids.add(source.registry_source_id)
        sources.append(source)
    sources.sort(
        key=lambda item: (
            item.year,
            item.state,
            item.contest,
            item.registry_scope,
            item.registry_format,
            item.registry_source_id,
        )
    )
    return sources


def project_public_registry_sources(
    path: str | Path,
) -> list[dict[str, str]]:
    return [
        {
            "registry_source_id": source.registry_source_id,
            "year": source.year,
            "contest": source.contest,
            "state": source.state,
            "scope": source.registry_scope,
            "format": source.registry_format,
            "registry_category": source.registry_category,
        }
        for source in list_public_registry_sources(path)
    ]


def resolve_public_registry_source(
    path: str | Path,
    registry_source_id: str,
) -> PublicRegistrySource | None:
    wanted = str(registry_source_id or "").strip()
    if not _PUBLIC_REGISTRY_SOURCE_ID_RE.fullmatch(wanted):
        return None
    matches = [
        source
        for source in list_public_registry_sources(path)
        if source.registry_source_id == wanted
    ]
    if len(matches) > 1:
        raise PublicRegistryResolutionError(
            "Public registry source ID resolved ambiguously."
        )
    return matches[0] if matches else None


def lookup_exact_registry_entry(
    url: str,
    *,
    path: str | Path,
) -> ContributorRegistryEntry | None:
    """Return the exact maintained-registry row for contributor disclosure.

    This function deliberately does not call normalize_url_for_match().
    Contributor disclosure must remain bound to the exact source_url already
    frozen onto the governed workflow item.
    """

    wanted = str(url or "")
    if not wanted:
        return None

    entries, _ = load_url_registry(path)
    for entry in entries:
        if str(entry.get("url") or "") != wanted:
            continue
        return ContributorRegistryEntry(
            year=str(entry.get("year") or ""),
            contest=str(entry.get("contest") or ""),
            state=str(entry.get("state") or ""),
            registry_scope=str(entry.get("scope") or ""),
            registry_format=str(entry.get("format") or ""),
            notes=str(entry.get("notes") or ""),
            url=str(entry.get("url") or ""),
            registry_category=_contributor_registry_category(entry),
        )

    return None
