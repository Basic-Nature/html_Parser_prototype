from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from ..Context_Integration.Context_Library.constants import (
    LOCATION_ABBREVIATIONS,
    LOCATION_KEYWORDS,
    LOCATION_SYNONYM_MAP,
)
from .detect import is_location_header

# ---------------------------------------------------------------------------
# Shared helpers for identifying and synthesizing geographic divisions.
# Centralizing this logic keeps every format handler aligned with the
# location heuristics defined in Context_Library/constants.py.
# ---------------------------------------------------------------------------

_LOCATION_FORBIDDEN_SUBSTRINGS = {
    "candidate",
    "write",
    "attorney",
    "campaign",
    "ballot",
    "total",
    "vote",
    "counter",
    "scanner",
    "absentee",
    "military",
    "applicable",
    "inapplicable",
}

_BASE_PRIORITY_HEADERS: Tuple[str, ...] = (
    "Precinct",
    "Assembly District",
    "Election District",
)

_SHORT_LOCATION_TOKENS = {
    "ad",
    "ed",
    "wd",
    "pct",
    "prec",
    "ward",
    "precinct",
    "division",
    "div",
    "district",
    "borough",
    "boro",
    "city",
    "county",
    "town",
    "village",
    "muni",
    "municipality",
    "community",
    "neighborhood",
    "region",
    "zone",
    "sector",
    "area",
}

for abbr in LOCATION_ABBREVIATIONS.keys():
    token = re.sub(r"[^a-z0-9]+", "", abbr.lower())
    if token:
        _SHORT_LOCATION_TOKENS.add(token)


def _normalize_location_text(value: str) -> str:
    if not value:
        return ""
    lowered = value.lower()
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


@lru_cache(maxsize=1_024)
def _location_phrases() -> Tuple[str, ...]:
    phrases = set()
    for keyword in LOCATION_KEYWORDS:
        normalized = _normalize_location_text(keyword)
        if normalized:
            phrases.add(normalized)
    for alias, canonical in LOCATION_SYNONYM_MAP.items():
        normalized_alias = _normalize_location_text(alias)
        if normalized_alias:
            phrases.add(normalized_alias)
        normalized_canon = _normalize_location_text(canonical)
        if normalized_canon:
            phrases.add(normalized_canon)
    phrases.update(
        {
            "precinct",
            "ward",
            "assembly district",
            "election district",
            "community district",
            "council district",
            "congressional district",
            "judicial district",
            "senate district",
            "supervisorial district",
            "school district",
            "voting district",
            "borough",
            "county",
            "city",
            "town",
            "village",
            "division",
            "subdivision",
            "region",
            "zone",
            "area",
            "sector",
        }
    )
    return tuple(sorted({p for p in phrases if p}, key=len, reverse=True))


def is_strict_location_header(header: str | None) -> bool:
    """Return True if *header* clearly refers to a geographic division."""
    if not isinstance(header, str):
        return False
    normalized = _normalize_location_text(header)
    if not normalized:
        return False
    if any(bad in normalized for bad in _LOCATION_FORBIDDEN_SUBSTRINGS):
        return False
    tokens = normalized.split()
    if any(tok in _SHORT_LOCATION_TOKENS for tok in tokens):
        return True
    padded = f" {normalized} "
    for phrase in _location_phrases():
        if f" {phrase} " in padded:
            return True
    # Fall back to the broader heuristic when strings are extremely short.
    if len(normalized) <= 5 and is_location_header(header):
        return True
    return False


def collect_location_headers(
    headers: Iterable[Any],
    *,
    ensure_precinct: bool = True,
    extra_headers: Sequence[str] | None = None,
) -> List[str]:
    """Return an ordered list of headers that look like location columns."""
    ordered: List[str] = []
    seen = set()

    if ensure_precinct:
        ordered.append("Precinct")
        seen.add("Precinct")

    for base in _BASE_PRIORITY_HEADERS[1:]:
        if base in headers and base not in seen:
            ordered.append(base)
            seen.add(base)

    for extra in extra_headers or []:
        if not isinstance(extra, str):
            continue
        stripped = extra.strip()
        if stripped and stripped not in seen:
            ordered.append(stripped)
            seen.add(stripped)

    for header in headers:
        if not isinstance(header, str):
            continue
        stripped = header.strip()
        if not stripped or stripped in seen:
            continue
        if is_strict_location_header(stripped):
            ordered.append(stripped)
            seen.add(stripped)

    return ordered


def format_location_fragment(header: str, value: Any) -> str:
    """Generate a human-readable fragment (e.g., ``AD 65``)."""
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    low = header.lower()
    if "assembly" in low and "district" in low:
        prefix = "AD"
    elif "election" in low and "district" in low:
        prefix = "ED"
    elif "community" in low and "district" in low:
        prefix = "Community District"
    elif "council" in low and "district" in low:
        prefix = "Council District"
    elif "congressional" in low and "district" in low:
        prefix = "Congressional District"
    elif "judicial" in low and "district" in low:
        prefix = "Judicial District"
    elif "senate" in low and "district" in low:
        prefix = "Senate District"
    elif "supervisorial" in low and "district" in low:
        prefix = "Supervisorial District"
    elif "school" in low and "district" in low:
        prefix = "School District"
    elif "ward" in low:
        prefix = "Ward"
    elif "division" in low:
        prefix = "Division"
    elif "borough" in low:
        prefix = "Borough"
    elif "county" in low:
        prefix = "County"
    elif "city" in low:
        prefix = "City"
    elif "town" in low:
        prefix = "Town"
    elif "village" in low:
        prefix = "Village"
    elif "precinct" in low:
        prefix = "Precinct"
    elif "district" in low:
        prefix = "District"
    else:
        prefix = header.strip()
    return f"{prefix} {text}".strip()


def attach_precinct_column(
    headers: Sequence[str] | None,
    rows: Sequence[Dict[str, Any]] | None,
    *,
    location_headers: Sequence[str] | None = None,
    column_name: str = "Precinct",
) -> Tuple[List[str], List[Dict[str, Any]], bool]:
    """Ensure a canonical precinct column exists and is populated when possible."""

    def _normalize_header(token: str | None) -> str:
        return (token or "").strip().lower()

    def _has_value(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        return bool(str(value).strip())

    def _dedupe_fragments(parts: List[str]) -> List[str]:
        ordered: List[str] = []
        seen = set()
        for part in parts:
            key = part.lower()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(part)
        return ordered

    working_headers = list(headers or [])
    working_rows = [dict(row or {}) for row in (rows or [])]

    canonical_label = (column_name or "Precinct").strip() or "Precinct"
    canonical_key = canonical_label.lower()

    existing_idx = next(
        (idx for idx, header in enumerate(working_headers)
         if isinstance(header, str) and header.strip().lower() == canonical_key),
        None,
    )

    column_added = False
    if existing_idx is None:
        working_headers.insert(0, canonical_label)
        target_header = canonical_label
        column_added = True
    else:
        target_header = working_headers[existing_idx]

    sanitized_locations = [
        header.strip()
        for header in (location_headers or [])
        if isinstance(header, str) and header.strip()
    ]
    if not sanitized_locations:
        sanitized_locations = collect_location_headers(working_headers, ensure_precinct=True)

    normalized_seen = set()
    ordered_locations: List[str] = []
    for header in sanitized_locations:
        norm = _normalize_header(header)
        if not header or norm in normalized_seen:
            continue
        normalized_seen.add(norm)
        ordered_locations.append(header)

    source_headers = [
        header for header in ordered_locations
        if _normalize_header(header) != canonical_key
    ]

    attached = column_added
    if not working_rows:
        return working_headers, working_rows, attached

    for row in working_rows:
        if _has_value(row.get(target_header)):
            continue

        fragments: List[str] = []
        for header in source_headers:
            if header not in row:
                continue
            fragment = format_location_fragment(header, row.get(header))
            if fragment:
                fragments.append(fragment)
        fragments = _dedupe_fragments(fragments)

        if fragments:
            row[target_header] = " / ".join(fragments)
            attached = True
        else:
            row.setdefault(target_header, "")

    return working_headers, working_rows, attached



