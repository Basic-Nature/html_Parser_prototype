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
    headers: List[str],
    rows: List[Dict[str, Any]],
    *,
    location_headers: Sequence[str] | None = None,
    column_name: str = "Precinct",
) -> Tuple[List[str], List[Dict[str, Any]], bool]:
    """Ensure ``column_name`` exists, synthesizing it from other location fields."""
    if not rows:
        return headers, rows, False

    working_headers = list(headers)
    detected = [h for h in (location_headers or []) if isinstance(h, str) and h.strip()]
    if not detected:
        detected = collect_location_headers(working_headers, ensure_precinct=(column_name == "Precinct"))

    candidate_headers: List[str] = []
    for header in detected:
        if header == column_name or header in working_headers:
            if header not in candidate_headers:
                candidate_headers.append(header)

    if column_name not in candidate_headers:
        candidate_headers.insert(0, column_name)

    added_any = False
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get(column_name):
            added_any = True
            continue
        fragments: List[str] = []
        for header in candidate_headers:
            if header == column_name:
                continue
            value = row.get(header)
            fragment = format_location_fragment(header, value)
            if fragment:
                fragments.append(fragment)
        if fragments:
            deduped = list(dict.fromkeys(fragments))
            row[column_name] = " / ".join(deduped)
            added_any = True

    if added_any and column_name not in working_headers:
        working_headers = [column_name] + [h for h in working_headers if h != column_name]

    return working_headers, rows, added_any
