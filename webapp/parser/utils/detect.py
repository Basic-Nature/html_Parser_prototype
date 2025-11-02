"""
detect.py
Detection heuristics, NER-assisted column detection, entity/dataclass models,
percent/location detection, harmonization, metrics, shared regex & numeric parsing.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from ..Context_Integration.Context_Library.constants import (
    BALLOT_TYPES,
    BALLOT_TYPES_SORT_ORDER,
    CANDIDATE_KEYWORDS,
    LOCATION_ABBREVIATIONS,
    LOCATION_KEYWORDS,
    PERCENT_KEYWORDS,
    TOTAL_KEYWORDS,
)
from .logger_singleton import logger
from .shared_logic import (
    safe_add,
    safe_append,
    safe_get,
    safe_items,
    safe_keys,
    safe_lower,
    safe_strip,
    safe_translate,
)

# ---------- Central Regex / Patterns ----------
PERCENT_REPORTED_RE = re.compile(r"(\d{1,3})\s*%[\s\-]*reported", re.I)
NUMBER_LIKE_RE = re.compile(r"^-?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?$")
NAME_LIKE_RE = re.compile(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}$")

# ---------- Metrics ----------
def emit_metric(name: str, **labels):
    try:
        logger.info({"metric": name, "labels": labels})
    except Exception:
        pass

# ---------- Dataclasses ----------
@dataclass
class EntityInfo:
    people: set = field(default_factory=set)
    locations: set = field(default_factory=set)
    ballot_types: set = field(default_factory=set)
    numbers: set = field(default_factory=set)
    row_entities: list = field(default_factory=list)
    ml_confidence: float | None = None
    association_log: Any | None = None
    segments: Any | None = None
    panels: Any | None = None
    location_column: str | None = None
    percent_column: str | None = None

@dataclass
class StructureInfo:
    location_header: str | None = None
    percent_header: str | None = None
    candidate_headers: List[str] = field(default_factory=list)
    ballot_types_headers: List[str] = field(default_factory=list)
    total_header: str | None = None
    verified: bool = False

# ---------- Normalization ----------
@lru_cache(maxsize=100_000)
def _norm(s: str) -> str:
    s = s.strip().lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii")
    s = re.sub(r"\s+"," ", s)
    return s

def normalize_text(s: str) -> str:
    return _norm(str(s))

def normalize_for_matching(text) -> str:
    t = safe_lower(safe_strip(text))
    table = str.maketrans('', '', r"""!"#$%&'()*+,./:;<=>?@[\]^_`{|}~""")
    return safe_translate(t, table)

# ---------- Percent / Heading ----------
def extract_percent_reported_from_heading(heading: str) -> str:
    if not heading:
        return ""
    m = PERCENT_REPORTED_RE.search(heading)
    if m:
        return f"{int(m.group(1))}%"
    if re.search(r"\bfully\s+reported\b", heading, re.I):
        return "100%"
    return ""

# ---------- Location / Percent Detection ----------
def _is_percent_header(h: str) -> bool:
    nh = normalize_header(h or "")
    if nh in {normalize_header(p) for p in PERCENT_KEYWORDS}:
        return True
    return "%" in (h or "").lower() or "reported" in (h or "").lower()

def _should_exclude_as_location(h: str) -> bool:
    return normalize_text(h) in {"contest","race","rawjson","raw json","json","data"}

def _is_bad_location_fallback(h: str) -> bool:
    if _is_percent_header(h):
        return True
    s = (h or "").strip().lower()
    return bool(re.fullmatch(r"(col(umn)?\s*\d+|\d+)", s))

def is_location_header(header) -> bool:
    hn = normalize_for_matching(header)
    for kw in LOCATION_KEYWORDS:
        if kw in hn:
            return True
    if hn in LOCATION_ABBREVIATIONS:
        return True
    return False

def dynamic_detect_location_header(headers: List[str], coordinator=None) -> Tuple[str, str, str]:
    try:
        lib = getattr(coordinator, "library", {})
        loc_patterns = set(safe_get(lib, "location_patterns", [])) or set(LOCATION_KEYWORDS)
        pct_patterns = set(safe_get(lib, "percent_patterns", [])) or set(PERCENT_KEYWORDS)
    except Exception:
        loc_patterns = set(LOCATION_KEYWORDS)
        pct_patterns = set(PERCENT_KEYWORDS)

    norm_headers = [normalize_text(h) for h in headers or []]
    percent_header: str | None = None
    location_header: str | None = None
    location_entity = None

    for idx, normalized in enumerate(norm_headers):
        if any(normalize_text(p) == normalized for p in pct_patterns) or _is_percent_header(headers[idx]):
            percent_header = headers[idx]
            break

    if percent_header is None:
        for idx, normalized in enumerate(norm_headers):
            if any(normalize_text(p) in normalized for p in pct_patterns) or _is_percent_header(headers[idx]):
                percent_header = headers[idx]
                break

    for idx, normalized in enumerate(norm_headers):
        if (
            any(normalize_text(p) == normalized for p in loc_patterns)
            and not _is_percent_header(headers[idx])
            and not _should_exclude_as_location(headers[idx])
        ):
            location_header = headers[idx]
            break

    if location_header is None:
        for idx, normalized in enumerate(norm_headers):
            if (
                any(normalize_text(p) in normalized for p in loc_patterns)
                and not _is_percent_header(headers[idx])
                and not _should_exclude_as_location(headers[idx])
            ):
                location_header = headers[idx]
                break

    if (
        location_header
        and percent_header
        and normalize_header(location_header) == normalize_header(percent_header)
    ):
        location_header = None

    return location_header, percent_header, location_entity

# ---------- Candidate Column Detection ----------
def detect_candidate_column(
    headers: List[str],
    data: List[Dict[str, Any]],
    coordinator=None,
) -> Optional[str]:
    if not headers:
        return None

    normalized_keywords = {normalize_header(k) for k in CANDIDATE_KEYWORDS}

    for header in headers:
        if normalize_header(header) in normalized_keywords:
            return header

    # NER on headers
    if coordinator and hasattr(coordinator, "extract_entities"):
        for header in headers:
            try:
                entities = coordinator.extract_entities(header)
                if any(label == "PERSON" for _, label in entities):
                    return header
            except Exception:
                pass

    samples = data[: min(40, len(data))]

    for header in headers:
        hits = 0
        seen = 0
        for row in samples:
            value = safe_get(row, header, "")
            if not isinstance(value, str) or not value:
                continue
            seen += 1
            if coordinator and hasattr(coordinator, "extract_entities"):
                try:
                    entities = coordinator.extract_entities(value)
                    if any(label == "PERSON" for _, label in entities):
                        hits += 1
                except Exception:
                    pass
        if seen and hits / seen >= 0.35:
            return header

    for header in headers:
        hits = 0
        count = 0
        for row in samples:
            value = safe_get(row, header, "")
            if not isinstance(value, str) or not value:
                continue
            count += 1
            if NAME_LIKE_RE.match(value.strip()):
                hits += 1
        if count and hits / count >= 0.35:
            return header
    return None

# ---------- Entity Annotation (Light) ----------
def nlp_entity_annotate_table(headers, data, context=None, coordinator=None):
    info = EntityInfo()
    if not coordinator:
        return headers, data, info.__dict__
    for h in headers:
        try:
            ents = coordinator.extract_entities(h)
            for ent, label in ents:
                if label == "PERSON":
                    safe_add(info.people, ent)
                elif label in {"GPE", "LOC", "FAC"}:
                    safe_add(info.locations, ent)
        except Exception:
            pass
    for row in data:
        row_ents = {"people": set(), "locations": set(), "ballot_types": set(), "numbers": set()}
        for h, v in safe_items(row):
            if not v:
                continue
            if isinstance(v, str) and NUMBER_LIKE_RE.match(v.replace(",", "")):
                safe_add(info.numbers, v)
                safe_add(row_ents["numbers"], v)
            if any(bt.lower() in h.lower() for bt in BALLOT_TYPES):
                safe_add(info.ballot_types, h)
                safe_add(row_ents["ballot_types"], h)
            if coordinator and isinstance(v, str):
                try:
                    ents = coordinator.extract_entities(v)
                    for ent, label in ents:
                        if label == "PERSON":
                            safe_add(info.people, ent)
                            safe_add(row_ents["people"], ent)
                        elif label in {"GPE", "LOC", "FAC"}:
                            safe_add(info.locations, ent)
                            safe_add(row_ents["locations"], ent)
                except Exception:
                    pass
        safe_append(info.row_entities, row_ents)
    return headers, data, info.__dict__

# ---------- Harmonization ----------
def harmonize_headers_and_data(
    headers: List[str],
    data: List[Dict[str, Any]],
    context: dict | None = None,
):
    headers = headers or []
    data = data or []

    all_headers = {h for h in headers if h}
    for row in data:
        all_headers.update(safe_keys(row))

    percent_val = None
    if any("Percent Reported" in safe_keys(row) for row in data):
        all_headers.add("Percent Reported")
        for row in data:
            percent_val = percent_val or safe_get(row, "Percent Reported", "") or None

    if context and safe_get(context, "percent_reported"):
        all_headers.add("Percent Reported")
        percent_val = safe_get(context, "percent_reported", percent_val)

    seen: set[str] = set()
    ordered: list[str] = []
    for header in headers:
        if header in all_headers and header not in seen:
            ordered.append(header)
            seen.add(header)
    for header in all_headers:
        if header not in seen:
            ordered.append(header)
            seen.add(header)

    loc_col = next((h for h in ordered if is_location_header(h)), None)
    if loc_col and loc_col != "Precinct":
        ordered = ["Precinct" if h == loc_col else h for h in ordered]
        for row in data:
            row["Precinct"] = row.pop(loc_col, row.get("Precinct", ""))
        loc_col = "Precinct"

    cand_col = next((h for h in ordered if any(k in h.lower() for k in CANDIDATE_KEYWORDS)), None)
    ballot_cols = [h for h in ordered if any(bt in h.lower() for bt in BALLOT_TYPES)]

    harmonized: list[dict] = []
    dedup: set[tuple] = set()
    for row in data:
        full = {h: safe_get(row, h, "") for h in ordered}
        if "Percent Reported" in ordered and not full.get("Percent Reported") and percent_val:
            full["Percent Reported"] = percent_val
        if loc_col and cand_col and full.get(loc_col) and full.get(cand_col):
            key = (
                full.get(loc_col),
                full.get(cand_col),
                *[full.get(b, "") for b in ballot_cols],
            )
            if key in dedup:
                continue
            dedup.add(key)
        harmonized.append(full)

    keep = [h for h in ordered if (h in headers) or any(r.get(h) not in ("", None) for r in harmonized)]
    if not keep:
        keep = ordered

    cand_cols = [h for h in keep if any(k in h.lower() for k in CANDIDATE_KEYWORDS)]
    bt_cols = [h for h in keep if any(bt in h.lower() for bt in BALLOT_TYPES)]

    def _bt_sort_key(col: str) -> tuple[int, int]:
        low = (col or "").strip().lower()
        for index, canon in enumerate(BALLOT_TYPES_SORT_ORDER):
            if low == canon.lower():
                return (0, index)
        for index, canon in enumerate(BALLOT_TYPES_SORT_ORDER):
            if canon.lower() in low:
                return (1, index)
        return (2, len(BALLOT_TYPES_SORT_ORDER))

    bt_cols = sorted(list(dict.fromkeys(bt_cols)), key=_bt_sort_key)

    final: list[str] = []
    if "Precinct" in keep:
        final.append("Precinct")
    final.extend(list(dict.fromkeys(cand_cols + bt_cols)))
    remainder = [h for h in keep if h not in {"Precinct", *cand_cols, *bt_cols}]
    final.extend(remainder)

    seen_final: set[str] = set()
    deduped_final: list[str] = []
    for header in final:
        lower = header.lower()
        if lower in seen_final:
            continue
        seen_final.add(lower)
        deduped_final.append(header)

    result_rows = [{h: row.get(h, "") for h in deduped_final} for row in harmonized]
    return deduped_final, result_rows

# ---------- Header Utilities ----------
def find_best_header(headers, keywords):
    lowered_headers = [safe_lower(h) for h in headers]
    for keyword in keywords:
        lowered_keyword = safe_lower(keyword)
        for index, lowered_header in enumerate(lowered_headers):
            if lowered_keyword in lowered_header:
                return headers[index]
    for keyword in keywords:
        lowered_keyword = safe_lower(keyword)
        matches = difflib.get_close_matches(lowered_keyword, lowered_headers, n=1, cutoff=0.7)
        if matches:
            return headers[lowered_headers.index(matches[0])]
    return None

def is_likely_header(row_cells: List[str]) -> bool:
    known = {
        *(k.lower() for k in CANDIDATE_KEYWORDS),
        *(k.lower() for k in LOCATION_KEYWORDS),
        *(k.lower() for k in PERCENT_KEYWORDS),
        *(k.lower() for k in TOTAL_KEYWORDS),
        "votes",
        "percent",
        "district",
        "party",
        "candidate",
    }
    matches = sum(1 for cell in row_cells if any(keyword in cell.lower() for keyword in known))
    return matches >= 2

# ---------- Numeric Parsing ----------
def parse_numeric(val: Any) -> Tuple[Optional[int], bool]:
    if val is None:
        return None, False

    text = str(val).strip()
    pct = text.endswith("%")
    normalized = text.replace("%", "").replace(",", "")
    if normalized.replace(".", "", 1).isdigit():
        try:
            return int(float(normalized)), pct
        except Exception:
            return None, pct
    return None, pct

# ---------- Table Data (simple) ----------
def extract_table_data(table, coordinator=None, structure_info=None):
    from .browser_utils import safe_count, safe_inner_text, safe_locator, safe_nth
    headers: list[str] = []
    rows: list[dict] = []
    head_cells = safe_locator(table, "thead tr th", logger)
    if safe_count(head_cells, logger) == 0:
        first_row = safe_nth(safe_locator(table, "tr", logger), 0, logger)
        head_cells = safe_locator(first_row, "th,td", logger) if first_row else []
    for index in range(safe_count(head_cells, logger)):
        text = safe_inner_text(safe_nth(head_cells, index, logger), logger).strip()
        headers.append(text or f"Column {index + 1}")
    body_rows = safe_locator(table, "tbody tr", logger)
    if safe_count(body_rows, logger) == 0:
        body_rows = safe_locator(table, "tr", logger)
    for row_index in range(safe_count(body_rows, logger)):
        row_locator = safe_nth(body_rows, row_index, logger)
        cells = safe_locator(row_locator, "td,th", logger)
        if safe_count(cells, logger) == 0:
            continue
        row_data: dict[str, str] = {}
        for cell_index in range(safe_count(cells, logger)):
            if cell_index < len(headers):
                row_data[headers[cell_index]] = safe_inner_text(
                    safe_nth(cells, cell_index, logger),
                    logger,
                ).strip()
        if any(row_data.values()):
            rows.append(row_data)
    return headers, rows, {}

# Safe constants wiring
try:
    from ..Context_Integration.Context_Library.constants import (
        HEADER_SYNONYM_MAP,
        PERCENT_KEYWORDS,
        TOTAL_KEYWORDS,
    )
except Exception:
    HEADER_SYNONYM_MAP = {}
    TOTAL_KEYWORDS = {"total", "total vote", "grand total"}
    PERCENT_KEYWORDS = {"percent reported", "% precincts reporting", "% reported", "precincts reporting"}

def normalize_header(h: Any) -> str:
    s = ("" if h is None else str(h)).strip()
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s)
    low = s.lower()

    # synonyms (Candidate, Precinct, Party, etc.)
    if low in HEADER_SYNONYM_MAP:
        return HEADER_SYNONYM_MAP[low]

    # totals normalization
    if any(t in low for t in ("grand total",)):
        return "Grand Total"
    if low in TOTAL_KEYWORDS or low in {"total", "total vote"}:
        return "Total Vote"

    # percent reported normalization
    if low in {p.lower() for p in PERCENT_KEYWORDS}:
        return "Percent Reported"

    return s

def dedupe_headers_with_suffix(headers: List[str]) -> List[str]:
    out, seen = [], set()
    for i, h in enumerate(headers or []):
        hs = normalize_header(h) or f"Column {i+1}"
        base = hs
        k = 2
        while hs.lower() in seen:
            hs = f"{base}_{k}"
            k += 1
        seen.add(hs.lower())
        out.append(hs)
    return out

def is_total_column(label: str) -> bool:
    low = (label or "").strip().lower()
    return (low in {t.lower() for t in TOTAL_KEYWORDS}) or low in {"total", "total vote", "grand total"}

__all__ = [
    "emit_metric",
    "EntityInfo",
    "StructureInfo",
    "normalize_text",
    "normalize_header",
    "normalize_for_matching",
    "extract_percent_reported_from_heading",
    "is_location_header",
    "dynamic_detect_location_header",
    "detect_candidate_column",
    "nlp_entity_annotate_table",
    "harmonize_headers_and_data",
    "find_best_header",
    "is_likely_header",
    "parse_numeric",
    "extract_table_data",
    "dedupe_headers_with_suffix",
    "is_total_column",
]
