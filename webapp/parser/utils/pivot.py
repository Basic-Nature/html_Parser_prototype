"""
pivot.py
Wide table pivot with:
- Detector integration
- Fast path for already-wide tables
- Strict header ordering (by candidate total desc then alpha)
- Party column inclusion (Candidate - Party)
- Per-candidate % Vote (recomputed)
- Cumulative running Vote / % (Candidate - Cumulative Vote / Candidate - Cumulative %)
- Division Type column via STATE_TO_DIVISION_TYPE_MAP
- Optional aggregate "All Precincts" summary row

Context / flags (defaults shown):
  strict_header_order: True
  include_party_in_wide: True
  include_candidate_percent: True
  include_cumulative_columns: True
  include_all_precincts_row: True
  include_division_type_column: True
  state: (two-letter or full name; used to derive division type)
"""

from __future__ import annotations

import hashlib
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import orjson

from ..Context_Integration.Context_Library.constants import (
    BALLOT_TYPES,
    BALLOT_TYPES_SORT_ORDER,
    CANDIDATE_BALLOT_SPLIT_PATTERN,
    DIVISION_HEURISTIC_TERMS,
    DIVISION_SUFFIXES,
    KNOWN_COUNTY_TO_PRECINCTS_MAP,
    LOCATION_ABBREVIATIONS,
    LOCATION_KEYWORDS,
    LOCATION_SYNONYM_MAP,
    PARTY_NORMALIZATION_MAP,
    STATE_TO_DIVISION_TYPE_MAP,
    TOTAL_KEYWORDS,
    canonical_ballot_group,
    normalize_party_label,
)
from .detect import dynamic_detect_location_header, normalize_header, parse_numeric
from .logger_singleton import logger
from .shared_logic import safe_get, safe_strip

_CAND_BT_RE = re.compile(CANDIDATE_BALLOT_SPLIT_PATTERN, re.UNICODE)
_TOTAL_LIKE = {normalize_header(t) for t in TOTAL_KEYWORDS} | {
    normalize_header("grand total"),
    normalize_header("total vote")
}
_BALLOT_TYPE_NORM_SET = {normalize_header(bt) for bt in BALLOT_TYPES} | {
    normalize_header(bt) for bt in BALLOT_TYPES_SORT_ORDER
}

_KNOWN_PARTY_NORMALS = {
    str(val).strip().lower()
    for val in PARTY_NORMALIZATION_MAP.values()
    if isinstance(val, str) and val.strip()
}

# --- Added natural sort + division heuristics (place here with other global regex/constants) ---
_SPLIT_NUM_RE = re.compile(r'(\d+)')


_NUMERIC_TRAILING_TYPES = {
    "precinct",
    "election district",
    "assembly district",
    "senate district",
    "congressional district",
    "school district",
    "township",
    "ward",
    "district",
    "division",
}


_DIVISION_TYPE_PRIORITY = (
    "precinct",
    "ward",
    "election district",
    "assembly district",
    "senate district",
    "congressional district",
    "school district",
    "township",
    "borough",
    "city",
    "town",
    "village",
    "county",
    "parish",
    "municipality",
    "district",
    "division",
    "region",
)


def _token_to_pattern(token: str) -> str:
    token = token.strip().lower()
    escaped = re.escape(token)
    escaped = escaped.replace(r"\ ", r"\s+")
    escaped = escaped.replace(r"\.", r"\.?")
    return escaped


def _build_division_token_patterns() -> list[tuple[re.Pattern, str]]:
    base_types = {dtype.lower() for _, dtype in DIVISION_HEURISTIC_TERMS}
    base_types.update(
        {
            "precinct",
            "election district",
            "assembly district",
            "senate district",
            "congressional district",
            "school district",
            "township",
            "ward",
        }
    )

    token_map: dict[str, set[str]] = {dtype: {dtype} for dtype in base_types}

    for kw in LOCATION_KEYWORDS:
        norm_kw = kw.strip().lower()
        if norm_kw in token_map:
            token_map[norm_kw].add(norm_kw)

    for alias, canonical in LOCATION_SYNONYM_MAP.items():
        canonical_norm = canonical.strip().lower()
        alias_norm = alias.strip().lower()
        if canonical_norm in token_map:
            token_map[canonical_norm].add(alias_norm)

    for abbr, expansions in LOCATION_ABBREVIATIONS.items():
        abbr_norm = abbr.strip().lower()
        for expansion in expansions:
            exp_norm = expansion.strip().lower()
            if exp_norm in token_map:
                token_map[exp_norm].add(abbr_norm)

    priority_map = {dtype: idx for idx, dtype in enumerate(_DIVISION_TYPE_PRIORITY)}
    patterns: list[tuple[re.Pattern, str]] = []
    for dtype, tokens in token_map.items():
        clean_tokens = [_token_to_pattern(tok) for tok in tokens if tok]
        if not clean_tokens:
            continue
        clean_tokens.sort(key=len, reverse=True)
        joined = "|".join(clean_tokens)
        if dtype in _NUMERIC_TRAILING_TYPES:
            regex = rf"\b(?:{joined})(?:\s*[-#:]?\s*\d+[A-Za-z]?)?\b"
        else:
            regex = rf"\b(?:{joined})\b"
        patterns.append((re.compile(regex, re.I), dtype))

    def _pattern_sort_key(item: tuple[re.Pattern, str]) -> tuple[int, int, str]:
        dtype = item[1]
        pr = priority_map.get(dtype, len(priority_map))
        return (pr, -len(dtype), dtype)

    patterns.sort(key=_pattern_sort_key)
    return patterns


_DIVISION_TOKEN_PATTERNS = _build_division_token_patterns()

# ---- small numeric helper & slight micro-optimizations ----
_NUM_CLEAN_RE = re.compile(r"[,\s%]+")

def _coerce_int(val) -> int:
    """Fast numeric coerce; returns 0 if not clean int."""
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val) if val.is_integer() else 0
    if isinstance(val, str):
        s = _NUM_CLEAN_RE.sub("", val.strip())
        if s and (s.isdigit() or (s.startswith("-") and s[1:].isdigit())):
            try:
                return int(s)
            except Exception:
                return 0
    return 0

# (Optional) cache normalized header strings to avoid recomputing many times
def _normalized_header_cache(headers: List[str]) -> dict:
    return {h: normalize_header(h) for h in headers}

def _natural_key(s: str):
    if s is None:
        return []
    if s.lower().strip() in ("all precincts", "all districts", "total", "overall"):
        return [float('inf')]
    parts = _SPLIT_NUM_RE.split(s.lower())
    key = []
    for p in parts:
        key.append(int(p) if p.isdigit() else p)
    return key

def _sort_precincts(precincts: List[str], context: dict):
    mode = context.get("precinct_sort", "natural")
    aggregate_last = context.get("aggregate_last", True)
    aggs = {"all precincts", "all districts", "total", "overall"}
    if mode == "alpha":
        ordered = sorted(precincts, key=lambda x: x.lower())
    elif mode == "numeric":
        if all(p.isdigit() for p in precincts if p):
            ordered = sorted(precincts, key=lambda x: int(x))
        else:
            ordered = sorted(precincts, key=_natural_key)
    else:
        ordered = sorted(precincts, key=_natural_key)
    if aggregate_last and any(p.lower() in aggs for p in ordered):
        core = [p for p in ordered if p.lower() not in aggs]
        agg = [p for p in ordered if p.lower() in aggs]
        return core + agg
    return ordered

def _infer_division_type_by_suffix(original: str) -> str:
    if not original:
        return ""
    low = original.lower().strip()
    for term, dtype in DIVISION_HEURISTIC_TERMS:
        if f" {term}" in f" {low} " or low.endswith(term):
            return dtype
    return ""

def _extract_municipality(precinct: str) -> str:
    """
    Heuristic municipality extractor from a precinct label.
    Examples:
      'Albion - 01'          -> 'Albion'
      'Springfield 12'       -> 'Springfield'
      'Town of Dover 3'      -> 'Dover'
      'Village of Rome - 4'  -> 'Rome'
    """
    if not precinct:
        return ""
    p = str(precinct).strip()
    # Split on dash first
    if " - " in p:
        left = p.split(" - ", 1)[0].strip()
        if left:
            p = left
    # Remove common tokens
    p2 = re.sub(r"\b(Town|City|Village|Ward|Dist(rict)?|Pct|Precinct|Borough|Twp|Township|County|Parish|of)\b\.?", "", p, flags=re.I)
    # Trim trailing numbers
    p2 = re.sub(r"\d+$", "", p2).strip()
    # Collapse spaces
    p2 = re.sub(r"\s{2,}", " ", p2)
    return p2 or ""

def _numeric_ratio(values) -> float:
    non = [v for v in values if v not in ("", None)]
    if not non:
        return 0.0
    num = 0
    for v in non:
        s = str(v).replace(",", "").replace("%", "").strip()
        if s.replace(".", "", 1).lstrip("-").isdigit():
            num += 1
    return num / len(non)

def _is_numeric_column(h: str, data: List[Dict[str, Any]]) -> bool:
    vals = [safe_get(r, h, "") for r in data]
    return _numeric_ratio(vals) >= 0.5

def _fast_path_already_wide(headers: List[str], data: List[Dict[str, Any]]):
    candidate_map: Dict[str, Set[str]] = {}
    has_pattern = False
    numeric_cache: Dict[str, bool] = {}
    for h in headers:
        if h in ("Precinct", "Percent Reported", "Grand Total", "Division Type"):
            continue
        m = _CAND_BT_RE.match(h)
        if not m:
            continue
        cand = m.group("cand").strip()
        bt = m.group("bt").strip()
        if not cand or not bt:
            continue
        cand_norm = normalize_header(cand)
        if cand_norm in _BALLOT_TYPE_NORM_SET or cand_norm in _TOTAL_LIKE:
            continue
        candidate_map.setdefault(cand, set()).add(bt)
        has_pattern = True
    if not has_pattern or len(candidate_map) < 1:
        return None
    pattern_cols = [h for h in headers if _CAND_BT_RE.match(h)]
    if not pattern_cols:
        return None
    numeric_like = 0
    for h in pattern_cols:
        if h not in numeric_cache:
            numeric_cache[h] = _is_numeric_column(h, data)
        if numeric_cache[h]:
            numeric_like += 1
    if numeric_like == 0:
        return None
    needs_total = "Grand Total" not in headers
    if needs_total:
        for row in data:
            gt = 0
            for h in pattern_cols:
                v = row.get(h)
                if isinstance(v, (int, float)):
                    gt += int(v)
                elif isinstance(v, str):
                    s = v.replace(",", "").strip()
                    if s.isdigit():
                        gt += int(s)
            row["Grand Total"] = gt
        headers = headers + ["Grand Total"]
    logger.info(f"[PIVOT] Fast path wide detected: candidates={len(candidate_map)} pattern_cols={len(pattern_cols)}")
    return headers, data

def debug_dump_pivot_state(tag: str, headers: List[str], data: List[Dict[str, Any]], limit: int = 3):
    sample = data[:limit]
    logger.debug({"pivot_debug": tag, "headers": headers, "row_samples": sample})

def _strip_party_fragment(label: str, party_hint: str | None = None) -> str:
    if not label:
        return label
    text = str(label).strip()
    hint_norm = normalize_party_label(party_hint) if party_hint else ""
    hint_low = hint_norm.lower() if hint_norm else ""

    def _matches(fragment: str) -> bool:
        if not fragment:
            return False
        cleaned = re.sub(r"[-–—/,|:]", " ", fragment).strip()
        normalized = normalize_party_label(fragment)
        if not normalized and cleaned:
            normalized = normalize_party_label(cleaned)
        normalized = (normalized or "").strip()
        if not normalized:
            return False
        norm_low = normalized.lower()
        if norm_low not in _KNOWN_PARTY_NORMALS:
            return False
        if hint_low and norm_low != hint_low:
            return False
        return True

    tokens = text.split()
    changed = True
    while tokens and changed:
        changed = False
        max_width = min(4, len(tokens))
        for width in range(max_width, 0, -1):
            tail_tokens = tokens[-width:]
            tail = " ".join(tail_tokens)
            if _matches(tail):
                candidate_tokens = tokens[:-width]
                candidate = " ".join(candidate_tokens).strip()
                if candidate:
                    tokens = candidate_tokens
                    text = candidate
                    changed = True
                break

    sep_match = re.search(r"(.+?)[\s]*([-–—/,|:])\s*([A-Za-z'. ]+)$", text)
    if sep_match and _matches(sep_match.group(3)):
        text = sep_match.group(1).strip()

    text = text.rstrip("-–—/,|: ").strip()
    return text or label.strip()


def _normalize_candidate_label(raw: str) -> str:
    """
    Strip leading short uppercase party token (<=4 chars) & drop trailing party parens when recognizable.
    DEM Jane Doe -> Jane Doe
    WOR Jane Doe (Democratic) -> Jane Doe
    """
    if not raw:
        return raw
    label = raw.strip()
    parts = label.split()
    if parts and len(parts[0]) <= 4 and parts[0].isupper():
        remainder = " ".join(parts[1:]).strip()
        label = remainder or label

    match = re.search(r"\(([^)]+)\)\s*$", label)
    if match:
        inner = normalize_party_label(match.group(1))
        if inner:
            label = label[: match.start()].strip()

    label = _strip_party_fragment(label)
    label = re.sub(r"\s{2,}", " ", label).strip()

    return label or raw

def _collect_ballot_types(headers: List[str], data: List[Dict[str, Any]], detector=None, candidate_col: Optional[str] = None):
    if detector:
        bt = detector.detect_ballot_types(headers, data)
        if bt:
            ordered = [b for b in BALLOT_TYPES_SORT_ORDER if b in bt]
            for b in bt:
                if b not in ordered:
                    ordered.append(b)
            return ordered
    bt_set = set()
    norm_sorted = [normalize_header(bt2) for bt2 in BALLOT_TYPES_SORT_ORDER]
    for h in headers:
        nh = normalize_header(h)
        if nh in norm_sorted:
            bt_set.add(h)
    if not bt_set:
        for h in headers:
            if h in ("Precinct", "Percent Reported", candidate_col, "Division Type", "Party"):
                continue
            nh = normalize_header(h)
            if nh in _TOTAL_LIKE:
                continue
            if any(b.lower() == h.lower() for b in BALLOT_TYPES):
                bt_set.add(h)
                continue
            if _is_numeric_column(h, data):
                bt_set.add(h)
    bt_set = {b for b in bt_set if normalize_header(b) not in _TOTAL_LIKE}
    ordered = [b for b in BALLOT_TYPES_SORT_ORDER if b in bt_set]
    for b in sorted(bt_set):
        if b not in ordered:
            ordered.append(b)
    return ordered

def _derive_party_map(candidate_col: Optional[str], data: List[Dict[str, Any]]) -> Dict[str, str]:
    if not candidate_col:
        return {}
    party_map: Dict[str, str] = {}
    for r in data:
        cand = safe_strip(safe_get(r, candidate_col, "")) or ""
        if not cand:
            continue
        party = safe_strip(safe_get(r, "Party", "")) or ""
        if not party:
            continue
        party_map.setdefault(cand, party)
    return party_map

def _normalize_division_name(name: str) -> str:
    n = name.strip().lower()
    for suf in DIVISION_SUFFIXES:
        if n.endswith(suf):
            n = n.removesuffix(suf)
            break
    return n

def _division_type_for(division: str, state: str | None) -> str:
    if not division:
        return ""
    if state:
        s = _normalize_state_key(state)
        div_norm = _normalize_division_name(division)
        state_map = STATE_TO_DIVISION_TYPE_MAP.get(s)
        if state_map:
            t = state_map.get(div_norm)
            if t:
                return t
            t2 = state_map.get(division.strip().lower())
            if t2:
                return t2
    return _infer_division_type_by_suffix(division)

def _s(x) -> str:
    """Coerce any segment to a safe string for header/label building."""
    try:
        return x if isinstance(x, str) else str(x)
    except Exception:
        return ""

def _safe_col_name(*parts) -> str:
    """
    Build stable column names by joining non-empty segments with ' - '.
    Ensures every segment is a string; prevents str+int TypeError during pivot.
    """
    segs = [_s(p) for p in parts if p not in (None, "")]
    return " - ".join(segs)

def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def _normalize_state_key(state: str | None) -> str:
    """Normalize arbitrary state strings to keys compatible with STATE_TO_DIVISION_TYPE_MAP."""
    if not state:
        return ""
    key = re.sub(r"[^a-z0-9]+", "_", state.strip().lower())
    key = re.sub(r"_+", "_", key).strip("_")
    return key


_COUNTY_DESIGNATOR_PATTERN = re.compile(r"\b(county|parish|borough|municipality|city|township|district)\b", re.I)


def _normalize_county_key(county: str | None) -> str:
    if not county:
        return ""
    key = _norm_text(county)
    key = _COUNTY_DESIGNATOR_PATTERN.sub("", key)
    key = re.sub(r"\s{2,}", " ", key).strip()
    return key


def _lookup_precinct_aliases_for_county(county: str | None) -> list[str]:
    if not county:
        return []
    norm_target = _normalize_county_key(county)
    if not norm_target:
        return []
    for raw_key, values in KNOWN_COUNTY_TO_PRECINCTS_MAP.items():
        if _normalize_county_key(raw_key) == norm_target:
            return values or []
    return []

def _detect_division_type_for_precinct(loc: str, state: str | None, context: dict) -> str:
    """
    Infer a stable division type for a given Precinct label:
      - If matches known municipality of the provided county -> 'municipality'
      - If looks like ED/Ward/District -> 'district'
      - If equals a known division in state map -> that division type
      - 'All Precincts' -> 'aggregate'
      - Else: suffix heuristic fallback
    """
    if not loc:
        return ""
    low = _norm_text(loc)
    if low in {"all precincts", "all districts", "overall", "total"}:
        return "aggregate"

    for pattern, dtype in _DIVISION_TOKEN_PATTERNS:
        if pattern.search(low):
            return dtype

    county_aliases = _lookup_precinct_aliases_for_county(context.get("county") or context.get("County"))
    if county_aliases:
        for muni in county_aliases:
            mlow = _norm_text(muni)
            if not mlow:
                continue
            if low == mlow:
                return "municipality"
        for muni in county_aliases:
            mlow = _norm_text(muni)
            if not mlow:
                continue
            if low.startswith(mlow + " ") or (" " + mlow + " ") in (" " + low + " "):
                return "municipality"

    wide_tokens = (
        ("countywide", "county"),
        ("citywide", "city"),
        ("boroughwide", "borough"),
        ("townwide", "town"),
        ("villagewide", "village"),
    )
    for token, dtype in wide_tokens:
        if token in low:
            return dtype

    if re.search(r"\bdistrict\b", low, flags=re.I):
        return "district"

    county_norm = _normalize_county_key(context.get("county") or context.get("County"))
    if county_norm and (low == county_norm or low.startswith(county_norm + " ")):
        county_orig = context.get("county") or context.get("County") or county_norm
        return str(county_orig)

    if state:
        smap = STATE_TO_DIVISION_TYPE_MAP.get(_normalize_state_key(state))
        if smap:
            base = re.sub(r"\d+.*$", "", low).strip()
            if base in smap:
                return smap[base]
            if low in smap:
                return smap[low]

    return _division_type_for(loc, state)

def _detect_division_name_for_precinct(loc: str, state: str | None, context: dict) -> str:
    """
    Resolve a human-readable Division Name for the row:
      - If looks like ED/ward/district: prefer the municipality name
        (via _extract_municipality), validated against county’s known list.
      - If matches a municipality name prefix in KNOWN_COUNTY_TO_PRECINCTS_MAP for the given county, use that muni.
      - If the label equals/starts with the county name, return the county.
      - Else, if STATE_TO_DIVISION_TYPE_MAP recognizes a base name, use that base.
      - Aggregates ('All Precincts', etc.) return 'All'.
      - Fallback: return extracted municipality or the raw loc.
    """
    if not loc:
        return ""
    low = _norm_text(loc)
    if low in {"all precincts", "all districts", "overall", "total"}:
        return "All"

    county_norm = _normalize_county_key(context.get("county") or context.get("County"))
    if county_norm and (low == county_norm or low.startswith(county_norm + " ")):
        county_orig = context.get("county") or context.get("County") or county_norm
        return str(county_orig)

    muni_guess = _extract_municipality(loc)
    muni_guess_low = _norm_text(muni_guess)

    county_aliases = _lookup_precinct_aliases_for_county(context.get("county") or context.get("County"))
    if county_aliases and muni_guess_low:
        for canonical in county_aliases:
            if _norm_text(canonical) == muni_guess_low:
                return canonical
        for canonical in county_aliases:
            cn_low = _norm_text(canonical)
            if muni_guess_low.startswith(cn_low) or cn_low.startswith(muni_guess_low):
                return canonical

    if county_aliases:
        for canonical in county_aliases:
            cn_low = _norm_text(canonical)
            if low == cn_low or low.startswith(cn_low + " ") or (" " + cn_low + " ") in (" " + low + " "):
                return canonical

    if state:
        smap = STATE_TO_DIVISION_TYPE_MAP.get(_normalize_state_key(state)) or {}
        base = re.sub(r"\d+.*$", "", low).strip()
        if base in smap:
            return " ".join(w.capitalize() for w in base.split())

    if muni_guess:
        return muni_guess
    return loc


def _normalize_party_value(raw: str | None) -> str:
    party = normalize_party_label(raw) if raw else ""
    if party:
        return party
    return (raw or "").strip().title()


def _extract_party_from_label(label: str) -> str:
    if not label:
        return ""
    match = re.search(r"\(([^)]+)\)\s*$", label)
    if match:
        inner = match.group(1)
        normalized = normalize_party_label(inner)
        if normalized:
            return normalized
    tail_match = re.search(r"([-–—/,|:])\s*([A-Za-z'. ]+)$", label)
    if tail_match:
        tail = tail_match.group(2)
        normalized = normalize_party_label(tail) or normalize_party_label(re.sub(r"[-–—/,|:]", " ", tail))
        if normalized:
            return normalized
    parts = label.split()
    if parts:
        for width in range(1, min(4, len(parts)) + 1):
            tail = " ".join(parts[-width:])
            if len(parts) > width:
                normalized = normalize_party_label(tail)
                if normalized:
                    return normalized
        prefix = parts[0]
        if prefix.isupper() and len(prefix) <= 4:
            normalized = normalize_party_label(prefix)
            if normalized:
                return normalized
    return ""


def _candidate_display_and_key(label: str, party_hint: str) -> tuple[str, str]:
    base = _normalize_candidate_label(label)
    party_norm = _normalize_party_value(party_hint)
    candidate = base
    match = re.search(r"\(([^)]+)\)\s*$", candidate)
    if match:
        inner = normalize_party_label(match.group(1))
        if inner:
            candidate = candidate[: match.start()].strip()
    if party_norm and candidate.lower().endswith(f"({party_norm.lower()})"):
        candidate = candidate[: -(len(party_norm) + 2)].strip()
    candidate = _strip_party_fragment(candidate, party_norm)
    candidate = candidate.rstrip("-–—/,|: ").strip()
    candidate = re.sub(r"\s{2,}", " ", candidate).strip()
    key = candidate.lower()
    if not key:
        key = base.lower()
    return candidate or base or label, key or label.lower()


def _normalize_ballot_suffix(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (label or "").lower()).strip()


def _map_ballot_suffix(suffix: str) -> tuple[str | None, str | None]:
    norm = _normalize_ballot_suffix(suffix)
    if not norm:
        return None, None
    if any(token in norm for token in ("election day", "eday", "day of")):
        return "Election Day Votes", None
    if any(token in norm for token in ("early", "advance", "in person early", "pre poll")):
        return "Early Votes", None
    if any(token in norm for token in ("mail", "absentee", "vote by mail", "vb m", "mail in", "mail-in", "military", "overseas", "postal")):
        return "Mail In Votes", None
    if "provisional" in norm or "question" in norm or "affidavit" in norm or "challenged" in norm:
        return "Provisional Votes", None
    if "uncategorized" in norm or "unassigned" in norm or "unreported" in norm:
        return "Uncategorized Votes", None
    if norm in {"total", "total vote", "total votes", "votes total"}:
        return None, None
    clean = re.sub(r"\s+", " ", suffix or "").strip()
    if not clean:
        return None, None
    return None, f"{clean} Votes"


def _pluralize_division(label: str) -> str:
    if not label:
        return "All Divisions"
    if label.endswith("y") and len(label) > 1 and label[-2] not in "aeiou":
        return label[:-1] + "ies"
    if label.endswith("s"):
        return label
    return label + "s"


def _choose_division_header(rows: List[Dict[str, Any]], context: dict | None) -> tuple[str, bool]:
    context = context or {}
    types: List[str] = []
    for row in rows or []:
        dtype = (row.get("Division Type") or "").strip().lower()
        if dtype and dtype != "aggregate" and dtype not in types:
            types.append(dtype)
    if types:
        header = types[0].title()
        return header or "Division", len(types) > 1
    fallback = (context.get("jurisdiction_type") or "").strip()
    state_hint = context.get("state") or context.get("State")
    guessed: List[str] = []
    for row in rows or []:
        loc_label = row.get("Division Name") or row.get("Precinct")
        if not loc_label:
            continue
        guess = _detect_division_type_for_precinct(loc_label, state_hint, context)
        if guess and guess not in ("", "aggregate") and guess not in guessed:
            guessed.append(guess)
        if len(guessed) > 1:
            break

    if guessed:
        header_name = (fallback or guessed[0]).title()
        return header_name, len(guessed) > 1

    if fallback:
        return fallback.title(), False

    return "Precinct", False

def pivot_to_wide(
    headers: List[str],
    data: List[Dict[str, Any]],
    entity_info: Dict[str, Any],
    coordinator=None,
    context: dict | None = None
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Enhanced wide pivot:
      - Cross-endorsement merge (merge_cross_endorsements)
      - Robust ballot type consolidation
      - Local + cumulative % calculations
      - Natural precinct sort
      - Fallback if already wide
      - Party inference preserved when labels normalized
    """
    context = context or {}
    headers = headers or []
    data = data or []
    if not data:
        # Early bail: nothing to pivot
        return headers or ["Precinct"], data
    detector = (context.get("detector") or entity_info.get("detector")) if entity_info else context.get("detector")
    hdr_norm_map = _normalized_header_cache(headers)

    # Flags
    strict_order = context.get("strict_header_order", True)
    include_party = context.get("include_party_in_wide", True)
    include_candidate_pct = context.get("include_candidate_percent", True)
    include_cumulative = context.get("include_cumulative_columns", True)
    include_all_row = context.get("include_all_precincts_row", True)
    include_division_type = context.get("include_division_type_column", True)
    include_division_name = context.get("include_division_name_column", False)  # <-- new flag
    merge_cross = context.get("merge_cross_endorsements", True)

    state = context.get("state")
    if not state and entity_info:
        state = entity_info.get("state")
    if state and "state" not in context:
        context["state"] = state
    county_ctx = (context.get("county") or (entity_info or {}).get("county") if entity_info else None)
    if county_ctx:
        context["county"] = county_ctx

    # Location / percent headers
    location_header = (entity_info or {}).get("location_column") or (entity_info or {}).get("location_header")
    percent_header = (entity_info or {}).get("percent_column") or (entity_info or {}).get("percent_header")

    if coordinator and (not location_header or not percent_header):
        det_loc, det_pct, _ = dynamic_detect_location_header(headers, coordinator)
        location_header = location_header or det_loc
        percent_header = percent_header or det_pct

    # Ensure Precinct column
    if isinstance(location_header, str) and location_header.strip().lower() in {"", "none"}:
        location_header = None
    if not location_header:
        location_header = "Precinct"
        default_loc = safe_get(context, "location_value", "") or safe_get(context, "contest", "") or "All"
        for r in data:
            if "Precinct" not in r:
                r["Precinct"] = default_loc

    location_alias = (location_header or "Precinct").strip()
    if location_header != "Precinct" and location_header:
        original_location_header = location_header
        headers = ["Precinct" if h == original_location_header else h for h in headers]
        for r in data:
            r["Precinct"] = r.pop(original_location_header, r.get("Precinct", "")) or r.get("Precinct", "")
        location_header = "Precinct"
        location_alias = original_location_header
    else:
        location_alias = "Precinct"

    if location_alias == "Precinct":
        best_alias = None
        best_score = float("inf")
        alias_counts: Dict[str, int] = {}
        county_hint = context.get("county") or context.get("County")
        map_alias = None
        if state and county_hint:
            map_alias = _division_type_for(county_hint, state)
            if map_alias in ("", None, "precinct"):
                map_alias = None

        priority_map = {dtype: idx for idx, dtype in enumerate(_DIVISION_TYPE_PRIORITY)}

        for sample_row in data:
            loc_val = safe_get(sample_row, "Precinct", "")
            if not loc_val:
                continue
            guess = _detect_division_type_for_precinct(loc_val, state, context)
            if not guess or guess in {"", "aggregate", "precinct", "county"}:
                continue
            alias_counts[guess] = alias_counts.get(guess, 0) + 1
            score = priority_map.get(guess, len(priority_map))
            if score < best_score:
                best_alias = guess
                best_score = score
            elif score == best_score and best_alias:
                if alias_counts[guess] > alias_counts.get(best_alias, 0):
                    best_alias = guess

        if best_alias:
            location_alias = best_alias.title()
        elif map_alias:
            location_alias = map_alias.title()

    location_alias = (location_alias or "Precinct").strip()
    if location_alias:
        location_alias = location_alias.replace("_", " ")
        if location_alias.lower() == "precinct":
            location_alias = "Precinct"
        else:
            location_alias = " ".join(part.capitalize() for part in location_alias.split())
    else:
        location_alias = "Precinct"

    aggregate_label_value = f"All {_pluralize_division(location_alias or 'Precinct')}"


    # Normalize percent header
    if percent_header and normalize_header(percent_header) != normalize_header("Percent Reported"):
        for r in data:
            if percent_header in r and "Percent Reported" not in r:
                r["Percent Reported"] = r.pop(percent_header)
        headers = ["Percent Reported" if h == percent_header else h for h in headers]
        percent_header = "Percent Reported"

    # Fast path if already wide
    fast = _fast_path_already_wide(headers, data)
    if fast:
        wh, wd = fast
        # Division Type/Name enrichment for already-wide tables
        if include_division_type and "Division Type" not in wh:
            for row in wd:
                row["Division Type"] = _detect_division_type_for_precinct(row.get("Precinct", ""), state, context)
            wh.insert(1, "Division Type")
        if include_division_name and "Division Name" not in wh:
            for row in wd:
                row["Division Name"] = _detect_division_name_for_precinct(row.get("Precinct", ""), state, context)
            wh.insert(2, "Division Name")
        if include_all_row:
            summary = {"Precinct": aggregate_label_value}
            if "Percent Reported" in wh:
                summary["Percent Reported"] = ""
            if "Division Type" in wh:
                summary["Division Type"] = "aggregate"
            for h in wh:
                if h in ("Precinct", "Percent Reported", "Division Type"):
                    continue
                if h.endswith("% Vote"):
                    summary[h] = ""
                    continue
                total = 0
                for r in wd:
                    iv, _ = parse_numeric(r.get(h, ""))
                    if iv is not None:
                        total += iv
                summary[h] = str(total)
            wd.append(summary)
        if location_alias != "Precinct" and "Precinct" in wh:
            idx = wh.index("Precinct")
            wh[idx] = location_alias
            for row in wd:
                row[location_alias] = row.pop("Precinct", "")
        return wh, wd

    # ---------------- Candidate column detection ----------------
    candidate_col = None
    if detector:
        try:
            candidate_col = detector.detect_candidate_column(headers, data)
        except Exception:
            candidate_col = None
    if not candidate_col:
        for h in headers:
            if normalize_header(h) == "candidate":
                candidate_col = h
                break
    if not candidate_col:
        # heuristic: first non-Precinct, non-percent, non-total textual column
        for h in headers:
            hn = normalize_header(h)
            if h in ("Precinct", "Percent Reported", "Division Type", "Party"):
                continue
            if hn in _TOTAL_LIKE:
                continue
            candidate_col = h
            break

    debug_dump_pivot_state("pre_wide", headers, data, limit=2)

    # If still none & pattern columns exist treat as already wide
    if not candidate_col and any(_CAND_BT_RE.match(h) for h in headers):
        fast_wide = _fast_path_already_wide(headers, data)
        if fast_wide:
            wh, wd = fast_wide
            return wh, wd

    # ---------------- Aggregate candidates (with cross endorsement normalization) -------------
    candidates_totals: Dict[str, int] = {}
    raw_to_norm: Dict[str, str] = {}
    if candidate_col:
        for r in data:
            raw_c = safe_strip(safe_get(r, candidate_col, "")) or ""
            if not raw_c:
                continue
            norm_c = _normalize_candidate_label(raw_c) if merge_cross else raw_c
            raw_to_norm[raw_c] = norm_c
            row_total = 0
            for k, v in r.items():
                if k in (candidate_col, "Precinct", "Percent Reported", "Party", "Division Type"):
                    continue
                nh = normalize_header(k)
                if nh in _TOTAL_LIKE or any(bt.lower() in k.lower() for bt in BALLOT_TYPES):
                    if isinstance(v, (int, float)):
                        row_total += int(v)
                    elif isinstance(v, str):
                        sv = v.replace(",", "").replace("%", "").strip()
                        if sv.isdigit():
                            row_total += int(sv)
            candidates_totals[norm_c] = candidates_totals.get(norm_c, 0) + row_total

    ballot_types = _collect_ballot_types(headers, data, detector=detector, candidate_col=candidate_col)
    party_map = _derive_party_map(candidate_col, data) if include_party else {}
    raw_party_present = include_party and (
        "Party" in headers or any("Party" in r for r in data)
    )
    party_column_requested = include_party and (raw_party_present or bool(party_map))

    # ---------------- Fallback: unstructured numeric table ----------------
    if not candidates_totals and not ballot_types:
        wide_headers = ["Precinct"]
        if include_division_type:
            wide_headers.append("Division Type")
        if include_division_name:
            wide_headers.append("Division Name")
        if percent_header:
            wide_headers.append("Percent Reported")
        wide_headers.append("Grand Total")
        loc_values = {safe_get(r, "Precinct", "") for r in data if safe_get(r, "Precinct", "")} or {"All"}
        rows_out = []
        for loc in _sort_precincts(list(loc_values), context):
            row = {"Precinct": loc}
            if include_division_type:
                row["Division Type"] = _division_type_for(loc, state)
            if percent_header:
                row["Percent Reported"] = ""
            grand = 0
            for r in data:
                if safe_get(r, "Precinct", "") != loc:
                    continue
                for h in headers:
                    if h in ("Precinct", "Percent Reported", "Division Type"):
                        continue
                    iv, _ = parse_numeric(r.get(h, ""))
                    if iv is not None:
                        grand += iv
                if percent_header and "Percent Reported" in r and not row.get("Percent Reported"):
                    row["Percent Reported"] = r["Percent Reported"]
            row["Grand Total"] = str(grand)
            rows_out.append(row)
        if include_all_row and rows_out and aggregate_label_value not in {r["Precinct"] for r in rows_out}:
            agg = {"Precinct": aggregate_label_value}
            if include_division_type:
                agg["Division Type"] = "aggregate"
            if percent_header:
                agg["Percent Reported"] = ""
            agg["Grand Total"] = str(sum(int(r["Grand Total"]) for r in rows_out if r.get("Grand Total", "").isdigit()))
            rows_out.append(agg)
        logger.info("[PIVOT] Fallback simple wide applied (no candidates detected).")
        if location_alias != "Precinct":
            wide_headers[0] = location_alias
            for row in rows_out:
                row[location_alias] = row.pop("Precinct", "")
        return wide_headers, rows_out

    # ---------------- Order candidates ----------------
    candidate_names = list(candidates_totals.keys())
    if strict_order and candidates_totals:
        candidate_names.sort(key=lambda c: (-candidates_totals.get(c, 0), c.lower()))
    else:
        candidate_names.sort()

    # ---------------- Build header list ----------------
    wide_headers: List[str] = ["Precinct"]
    if include_division_type:
        wide_headers.append("Division Type")
    if include_division_name:
        insert_at = 2 if include_division_type else 1
        wide_headers.insert(insert_at, "Division Name")
    if "Percent Reported" in headers:
        wide_headers.append("Percent Reported")

    for cand in candidate_names:
        if party_column_requested:
            wide_headers.append(_safe_col_name(cand, "Party"))
        for bt in ballot_types:
            wide_headers.append(_safe_col_name(cand, bt))
        wide_headers.append(_safe_col_name(cand, "Total Vote"))
        if include_candidate_pct:
            wide_headers.append(_safe_col_name(cand, "% Vote"))
        if include_cumulative:
            wide_headers.append(_safe_col_name(cand, "Cumulative Vote"))
            if include_candidate_pct:
                wide_headers.append(_safe_col_name(cand, "Cumulative %"))
    wide_headers.append("Grand Total")

    # ---------------- Build rows ----------------
    loc_values = _sort_precincts(
        list({safe_get(r, "Precinct", "") for r in data if safe_get(r, "Precinct", "")} or ["All"]),
        context
    )
    running_totals = {c: 0 for c in candidate_names}
    overall_grand_total = 0
    out_rows: List[Dict[str, Any]] = []
    division_type_cache: Dict[str, str] = {}
    division_name_cache: Dict[str, str] = {}
    column_sums: defaultdict[str, int] = defaultdict(int)


    for loc in loc_values:
        out = {h: "" for h in wide_headers}
        out["Precinct"] = loc
        if include_division_type:
            if loc not in division_type_cache:
                division_type_cache[loc] = _detect_division_type_for_precinct(loc, state, context)
            out["Division Type"] = division_type_cache[loc]
        if include_division_name:
            if loc not in division_name_cache:
                division_name_cache[loc] = _detect_division_name_for_precinct(loc, state, context)
            out["Division Name"] = division_name_cache[loc]
        if "Percent Reported" in wide_headers:
            for r in data:
                if safe_get(r, "Precinct", "") == loc and safe_get(r, "Percent Reported", ""):
                    out["Percent Reported"] = r.get("Percent Reported")
                    break

        grand_total_loc = 0
        cand_totals_local: Dict[str, int] = {}

        for r in data:
            if safe_get(r, "Precinct", "") != loc:
                continue
            raw_c = safe_strip(safe_get(r, candidate_col, "")) if candidate_col else ""
            if not raw_c:
                continue
            norm_c = raw_to_norm.get(raw_c, raw_c)
            if norm_c not in candidate_names:
                continue
            cand_row_total = 0
            for bt in ballot_types:
                val = r.get(bt)
                if val in (None, "", "-"):
                    for k, v in r.items():
                        if (hdr_norm_map.get(k) or normalize_header(k)) == (hdr_norm_map.get(bt) or normalize_header(bt)):
                            val = v
                            break
                vc = _coerce_int(val)
                if vc:
                    bt_col = _safe_col_name(norm_c, bt)
                    current = _coerce_int(out.get(bt_col, 0))
                    out[bt_col] = current + vc
                    cand_row_total += vc
            # explicit total override
            for k, v in r.items():
                if (hdr_norm_map.get(k) or normalize_header(k)) in _TOTAL_LIKE:
                    ov = _coerce_int(v)
                    if ov:
                        cand_row_total = ov
                        break

            total_vote_col = _safe_col_name(norm_c, "Total Vote")
            out[total_vote_col] = _coerce_int(out.get(total_vote_col, 0)) + cand_row_total
            cand_totals_local[norm_c] = cand_totals_local.get(norm_c, 0) + cand_row_total
            grand_total_loc += cand_row_total

            if party_column_requested:
                party_val = party_map.get(raw_c) or party_map.get(norm_c)
                party_col = _safe_col_name(norm_c, "Party")
                if party_col in out and party_val and not out[party_col]:
                    out[party_col] = party_val

        # Local %
        if include_candidate_pct and grand_total_loc:
            for cand in candidate_names:
                ct = cand_totals_local.get(cand, 0)
                out[_safe_col_name(cand, "% Vote")] = f"{(ct / grand_total_loc) * 100:.2f}%"

        # Cumulative
        if include_cumulative:
            overall_grand_total += grand_total_loc
            for cand in candidate_names:
                running_totals[cand] += cand_totals_local.get(cand, 0)
                out[_safe_col_name(cand, "Cumulative Vote")] = running_totals[cand]
                if include_candidate_pct and overall_grand_total:
                    out[_safe_col_name(cand, "Cumulative %")] = f"{(running_totals[cand] / overall_grand_total) * 100:.2f}%"

        out["Grand Total"] = grand_total_loc

        # Column sums (only numeric)
        for k, v in out.items():
            if k in ("Precinct", "Division Type", "Division Name", "Percent Reported"):  # <-- skip Division Name too
                continue
            iv = _coerce_int(v)
            column_sums[k] += iv

        out_rows.append(out)

    # Aggregate row (single pass sums)
    if include_all_row and out_rows:
        agg = {h: "" for h in wide_headers}
        agg["Precinct"] = aggregate_label_value
        if include_division_type:
            agg["Division Type"] = "aggregate"
        if include_division_name:
            agg["Division Name"] = "All"
        grand_total_all = column_sums.get("Grand Total", 0)

        for cand in candidate_names:
            total_col = _safe_col_name(cand, "Total Vote")
            agg[total_col] = column_sums.get(total_col, 0)
            if include_candidate_pct:
                if grand_total_all:
                    agg[_safe_col_name(cand, "% Vote")] = f"{(agg[total_col] / grand_total_all) * 100:.2f}%"
                else:
                    agg[_safe_col_name(cand, "% Vote")] = ""
            if include_cumulative:
                agg[_safe_col_name(cand, "Cumulative Vote")] = agg[total_col]
                if include_candidate_pct:
                    agg[_safe_col_name(cand, "Cumulative %")] = agg[_safe_col_name(cand, "% Vote")]
            if party_column_requested:
                party_col = _safe_col_name(cand, "Party")
                if party_col in wide_headers:
                    # Take first non-empty from rows
                    for r in out_rows:
                        if r.get(party_col):
                            agg[party_col] = r[party_col]
                            break
            for bt in ballot_types:
                bt_col = _safe_col_name(cand, bt)
                agg[bt_col] = column_sums.get(bt_col, 0)

        agg["Grand Total"] = grand_total_all
        out_rows.append(agg)

    if location_alias != "Precinct":
        if wide_headers:
            wide_headers[0] = location_alias
        for row in out_rows:
            row[location_alias] = row.pop("Precinct", "")

    logger.info(f"[PIVOT] wide rows={len(out_rows)} cols={len(wide_headers)} candidates={len(candidate_names)} bt={len(ballot_types)}")
    if not candidate_names:
        logger.warning("[PIVOT] No candidates detected – verify headers and candidate column extraction.")
    return wide_headers, out_rows


def transform_wide_to_smart_standard(
    headers: List[str],
    rows: List[Dict[str, Any]],
    context: dict | None = None
) -> tuple[List[str], List[Dict[str, Any]], bool]:
    context = context or {}
    if not headers or not rows:
        return headers, rows, False

    has_precinct = "Precinct" in headers or any("Precinct" in r for r in rows)
    has_division_name = "Division Name" in headers or any("Division Name" in r for r in rows)

    location_field: str | None = None
    if has_division_name:
        location_field = "Division Name"
    elif has_precinct:
        location_field = "Precinct"
    else:
        candidate_headers: List[str] = []
        ignore_norms = {
            normalize_header("division type"),
            normalize_header("percent reported"),
            normalize_header("grand total"),
            normalize_header("county"),
        }
        for h in headers:
            if not isinstance(h, str):
                continue
            if " - " in h:
                candidate_headers.append(h)
                continue
            nh = normalize_header(h)
            if nh in ignore_norms:
                continue
            if nh == normalize_header("precinct"):
                location_field = h
                break
            if location_field is None:
                location_field = h
        if location_field is None and candidate_headers:
            # Inspect first row for a non-candidate key
            sample = rows[0] if rows else {}
            if isinstance(sample, dict):
                for key in sample.keys():
                    if " - " in key:
                        continue
                    nh = normalize_header(key)
                    if nh in ignore_norms:
                        continue
                    location_field = key
                    break

    if not location_field:
        return headers, rows, False

    candidate_map: Dict[str, Dict[str, Any]] = {}
    for header in headers:
        if " - " not in header:
            continue
        base, suffix = header.split(" - ", 1)
        suffix_norm = suffix.strip().lower()
        info = candidate_map.setdefault(base, {"party": None, "total": None, "ballots": {}, "raw": base})
        if suffix_norm == "party":
            info["party"] = header
        elif suffix_norm in ("total vote", "total"):
            info["total"] = header
        elif "cumulative" in suffix_norm or suffix_norm.endswith("% vote") or suffix_norm.startswith("%"):
            continue
        else:
            info["ballots"][suffix] = header

    candidate_map = {k: v for k, v in candidate_map.items() if v.get("total") or v.get("ballots")}
    if not candidate_map:
        return headers, rows, False

    division_header, include_division_type = _choose_division_header(rows, context)

    county_value = (context.get("county") or context.get("County") or "").strip()
    county_value = county_value.title() if county_value else ""

    transformed_rows: List[Dict[str, Any]] = []
    standard_usage: Set[str] = set()
    extra_usage: Set[str] = set()
    mismatches: List[dict] = []

    for row in rows:
        location_raw = safe_strip(row.get(location_field, "") or row.get("Precinct", "")) or ""
        if not location_raw:
            continue
        division_type_value = safe_strip(row.get("Division Type", ""))
        if include_division_type and not division_type_value:
            division_type_value = division_header.lower()
        if not include_division_type:
            division_type_value = ""

        loc_lower = location_raw.lower()
        if loc_lower.startswith("all ") and division_header:
            location_value = f"All {_pluralize_division(division_header)}"
        else:
            location_value = location_raw

        grouped_variants: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for cand_label, info in candidate_map.items():
            total_col = info.get("total")
            ballots = info.get("ballots", {})
            party_col = info.get("party")

            total_val = _coerce_int(row.get(total_col, 0)) if total_col else 0
            ballot_values: Dict[str, int] = {}
            for suffix, col in ballots.items():
                val = _coerce_int(row.get(col, 0))
                if val:
                    ballot_values[suffix] = val
            cand_norm = re.sub(r"\s+", " ", (cand_label or "").replace("-", " ")).strip().lower()
            is_write_in_candidate = cand_norm.startswith("write in") or cand_norm == "writein"

            if not total_val and not ballot_values and not is_write_in_candidate:
                continue

            party_val_raw = safe_strip(row.get(party_col, "")) if party_col else ""
            party_val = _normalize_party_value(party_val_raw)
            if not party_val:
                fallback = _extract_party_from_label(cand_label)
                party_val = fallback or party_val

            display_name, base_key = _candidate_display_and_key(cand_label, party_val)

            if is_write_in_candidate:
                display_name = "Write-In"
                base_key = "write-in"
                party_val = "NA"

            standard_values: Dict[str, int] = defaultdict(int)
            extra_values: Dict[str, int] = defaultdict(int)
            explicit_uncategorized: Optional[int] = None

            for suffix, value in ballot_values.items():
                std_col, extra_col = _map_ballot_suffix(suffix)
                if std_col:
                    if std_col == "Uncategorized Votes":
                        explicit_uncategorized = (explicit_uncategorized or 0) + value
                    else:
                        standard_values[std_col] += value
                elif extra_col:
                    extra_values[extra_col] += value

            sum_standard = sum(standard_values.values())
            sum_extra = sum(extra_values.values())
            uncategorized = explicit_uncategorized or 0

            if explicit_uncategorized is None and total_val:
                remainder = total_val - (sum_standard + sum_extra)
                if remainder < 0:
                    remainder = 0
                uncategorized = remainder

            if not total_val and (sum_standard or sum_extra or uncategorized):
                total_val = sum_standard + sum_extra + uncategorized

            calculated_total = sum_standard + sum_extra + uncategorized

            if total_val and calculated_total and abs(calculated_total - total_val) > 0:
                mismatches.append({
                    "location": location_value,
                    "candidate": display_name,
                    "reported_total": total_val,
                    "calculated_total": calculated_total,
                })

            standard_usage.update({col for col, val in standard_values.items() if val})
            if uncategorized:
                standard_usage.add("Uncategorized Votes")
            extra_usage.update({col for col, val in extra_values.items() if val})

            variant = {
                "display": display_name,
                "party": party_val,
                "standard": dict(standard_values),
                "extra": dict(extra_values),
                "uncategorized": uncategorized,
                "calculated_total": calculated_total,
                "reported_total": total_val,
                "write_in": is_write_in_candidate,
            }

            grouped_variants[base_key].append(variant)

        for variants in grouped_variants.values():
            if not variants:
                continue

            def _signature(entry: Dict[str, Any]) -> tuple:
                sig: List[tuple] = [("uncategorized", entry["uncategorized"]), ("total", entry["calculated_total"])]
                sig.extend(sorted(entry["standard"].items()))
                sig.extend(sorted(entry["extra"].items()))
                return tuple(sig)

            signatures = {_signature(v) for v in variants}

            merged_variants: List[Dict[str, Any]]
            if len(signatures) == 1:
                base_variant = variants[0].copy()
                parties = [v["party"] for v in variants if v.get("party")]
                if parties:
                    unique = []
                    for p in parties:
                        if p not in unique:
                            unique.append(p)
                    base_variant["party"] = " / ".join(unique)
                base_variant["write_in"] = any(v["write_in"] for v in variants)
                merged_variants = [base_variant]
            else:
                merged_variants = variants

            for variant in merged_variants:
                row_out: Dict[str, Any] = {
                    "County": county_value,
                    division_header: location_value or "",
                    "Ballot Candidate Name": variant["display"],
                    "Ballot Party": variant.get("party", ""),
                    "Uncategorized Votes": variant.get("uncategorized", 0),
                    "Calculated Total Votes": variant.get("calculated_total", 0),
                    "Write-In": "TRUE" if variant.get("write_in") else "FALSE",
                }
                if include_division_type:
                    row_out["Division Type"] = division_type_value or division_header.lower()

                for col, val in variant["standard"].items():
                    if val:
                        row_out[col] = val
                for col, val in variant["extra"].items():
                    if val:
                        row_out[col] = val

                transformed_rows.append(row_out)

    if transformed_rows and include_division_type:
        observed_types = [safe_strip(r.get("Division Type", "")) for r in transformed_rows]
        unique_types = [t for t in dict.fromkeys(observed_types) if t]
        if len(unique_types) <= 1:
            primary_type = unique_types[0] if unique_types else ""
            for r in transformed_rows:
                r.pop("Division Type", None)
            include_division_type = False
            meta = context.setdefault("smart_standard_metadata", {})
            meta["primary_division_type"] = primary_type

    if not transformed_rows:
        return headers, rows, False

    standard_order = ["Early Votes", "Election Day Votes", "Mail In Votes", "Provisional Votes"]
    ordered_standard = [col for col in standard_order if col in standard_usage]
    extra_columns = sorted(extra_usage)

    output_headers: List[str] = ["County", division_header]
    if include_division_type:
        output_headers.append("Division Type")
    output_headers.extend(["Ballot Candidate Name", "Ballot Party", "Uncategorized Votes"])
    output_headers.extend(ordered_standard)
    output_headers.extend(extra_columns)
    output_headers.extend(["Calculated Total Votes", "Write-In"])

    if mismatches:
        context.setdefault("smart_standard_warnings", []).extend(mismatches[:50])

    context.setdefault("smart_standard_metadata", {})["ballot_columns"] = {
        "standard": ordered_standard,
        "extra": extra_columns,
        "division_header": division_header,
    }

    county_values = [row.get("County") for row in transformed_rows if row.get("County") not in (None, "")]
    unique_counties = list(dict.fromkeys(county_values))
    metadata = context.setdefault("smart_standard_metadata", {})

    if len(unique_counties) == 1:
        metadata["single_county"] = unique_counties[0]
        output_headers = [col for col in output_headers if col != "County"]
        for row in transformed_rows:
            row.pop("County", None)
    else:
        for row in transformed_rows:
            if row.get("County") is None:
                row["County"] = ""

    for row in transformed_rows:
        for col in output_headers:
            row.setdefault(col, "")
        if include_division_type and "Division Type" in output_headers and not row.get("Division Type"):
            row["Division Type"] = division_header.lower()

    context["smart_standard_applied"] = True

    return output_headers, transformed_rows, True

# --- Single-row RawJSON expansion helper (if not already present) ---
def expand_single_rawjson_row(headers: List[str], rows: List[Dict[str, Any]], context: dict | None = None):
    """
    If exactly one row contains a RawJSON blob (contest-level) expand it
    into precinct-level wide format early (before generic pivot).
    """
    context = context or {}
    if not headers or "RawJSON" not in headers or len(rows) != 1:
        return headers, rows
    try:
        from .pivot import (
            pivot_candidate_groups_from_rawjson,  # self-import-safe if structure unchanged
        )
    except Exception:
        # Fallback: local reference if same module
        pass
    try:
        cg_headers, cg_rows = pivot_candidate_groups_from_rawjson(headers, rows, context=context, drop_rawjson=False)
        if cg_headers and cg_rows:
            context["rawjson_expanded_early"] = True
            return cg_headers, cg_rows
    except Exception as e:
        context["rawjson_expand_error"] = str(e)
    return headers, rows

# ---------------- RawJSON candidate groups pivot fix -----------------

def _norm_key(s: str) -> str:
    if s is None:
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())

def _build_colmap(headers: list[str]) -> dict[str, str]:
    # normalized -> actual
    return {_norm_key(h): h for h in (headers or [])}

def _read_ndjson_record(raw_path: str, raw_id: int):
    """
    Resolve a single NDJSON record by ordinal index (1-based or 0-based tolerant).
    Returns a parsed object or None.
    """
    if not raw_path:
        return None
    try:
        path = os.path.abspath(raw_path)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            lines = f.read().splitlines()
        idxs = []
        try:
            rid = int(raw_id)
            idxs = [rid, rid - 1] if rid > 0 else [0]
        except Exception:
            idxs = []
        for idx in idxs:
            if 0 <= idx < len(lines):
                line = lines[idx].decode("utf-8", errors="ignore").strip()
                if line:
                    try:
                        return orjson.loads(line)
                    except Exception:
                        continue
        for line_b in lines:
            try:
                obj = orjson.loads(line_b)
                if isinstance(obj, dict) and str(obj.get("id", "")).strip() == str(raw_id).strip():
                    return obj
            except Exception:
                continue
    except Exception:
        return None
    return None

def _pick_contest_from_obj(obj, context: dict, rows: list[dict]) -> dict | None:
    """
    Return a single contest dict that has ballotOptions.
    Accepts:
      - direct contest dict with ballotOptions
      - dict with 'contests' (list)
      - list of contests
    Uses context['contest'] and any 'Contest' cell in rows to match by name (case-insensitive).
    """
    def _name(obj: Any) -> str:
        if not isinstance(obj, dict):
            return ""
        return (obj.get("name") or "").strip().lower()
    want = (str(context.get("contest") or "")).strip().lower()
    row_want = ""
    for r in rows or []:
        v = r.get("Contest") or r.get("contest") or ""
        if v:
            row_want = str(v).strip().lower()
            break

    def _match(candidate: Any) -> bool:
        if not isinstance(candidate, dict):
            return False
        if not candidate.get("ballotOptions"):
            return False
        if want and _name(candidate) == want:
            return True
        if row_want and _name(candidate) == row_want:
            return True
        return not (want or row_want)

    # direct
    if isinstance(obj, dict) and obj.get("ballotOptions"):
        return obj
    # dict with list
    if isinstance(obj, dict):
        for key in ("contests", "races", "Contest", "Races"):
            lst = obj.get(key)
            if isinstance(lst, list):
                for c in lst:
                    if _match(c):
                        return c
                # fallback first contest with ballotOptions
                for c in lst:
                    if isinstance(c, dict) and c.get("ballotOptions"):
                        return c
    # list of contests
    if isinstance(obj, list):
        for c in obj:
            if _match(c):
                return c
        for c in obj:
            if isinstance(c, dict) and c.get("ballotOptions"):
                return c
    return None

def _load_contest_from_rows(headers: list[str], data: list[dict], context: dict | None = None) -> dict | None:
    """
    Try to load a contest object from:
      1) inline RawJSON column
      2) RawJSONPath + RawId pointer into NDJSON
    Returns parsed dict or None.
    """
    # header map (case/format-insensitive)
    context = context or {}
    colmap = _build_colmap(headers)
    rawjson_col = colmap.get(_norm_key("RawJSON"))
    rawpath_col = colmap.get(_norm_key("RawJSONPath"))
    rawid_col = colmap.get(_norm_key("RawId"))

    # 1) inline RawJSON
    if rawjson_col:
        for row in data:
            val = row.get(rawjson_col)
            if isinstance(val, str) and val.strip():
                try:
                    obj = orjson.loads(val)
                    c = _pick_contest_from_obj(obj, context, data)
                    if c:
                        return c
                except Exception:
                    pass
            if isinstance(val, dict):
                c = _pick_contest_from_obj(val, context, data)
                if c:
                    return c

    # 2) pointer: RawJSONPath + RawId
    if rawpath_col and rawid_col:
        for row in data:
            path = row.get(rawpath_col)
            rid = row.get(rawid_col)
            obj = _read_ndjson_record(path, rid)
            c = _pick_contest_from_obj(obj, context, data) if obj is not None else None
            if c:
                return c
    return None

def pivot_candidate_groups_from_rawjson(
    headers: List[str],
    data: List[Dict[str, Any]],
    context: dict | None = None,
    drop_rawjson: bool = True
) -> tuple[List[str] | None, List[Dict[str, Any]] | None]:
    """
    Expand a contest RawJSON payload into a wide precinct table.
    Accepts:
      - inline RawJSON column
      - RawJSONPath + RawId pointer into NDJSON
      - optional context['rawjson_obj'] (dict/JSON-string) or
        context['rawjson_path'] + context['raw_id'] fallback
    Case-insensitive header detection and robust contest picking.
    """
    context = context or {}
    if not headers or not data:
        return None, None

    # Detect presence of RawJSON (inline or pointer) case-insensitively,
    # but also allow explicit context overrides.
    normset = {_norm_key(h) for h in headers}
    has_raw_cols = bool({_norm_key("RawJSON"), _norm_key("RawJSONPath")} & normset)
    has_ctx_hint = bool(context.get("rawjson_obj") or (context.get("rawjson_path") and context.get("raw_id")))
    if not (has_raw_cols or has_ctx_hint):
        logger.debug({"level": "DEBUG", "type": "pivot", "message": "RawJSON pivot: no rawjson headers found and no context hint."})
        return None, None

    # Resolve contest object from multiple sources
    contest: dict | None = None

    # 1) From context rawjson_obj (dict or JSON string)
    rawjson_obj = context.get("rawjson_obj")
    if rawjson_obj and contest is None:
        try:
            obj = orjson.loads(rawjson_obj) if isinstance(rawjson_obj, (str, bytes, bytearray)) else rawjson_obj
            contest = _pick_contest_from_obj(obj, context, data)
        except Exception:
            pass

    # 2) From context rawjson_path/raw_id (explicit)
    if contest is None and context.get("rawjson_path") and context.get("raw_id") is not None:
        try:
            obj = _read_ndjson_record(context.get("rawjson_path"), context.get("raw_id"))
            if obj is not None:
                contest = _pick_contest_from_obj(obj, context, data)
        except Exception:
            pass

    # 3) From headers (inline or pointer columns)
    if contest is None:
        contest = _load_contest_from_rows(headers, data, context=context)

    if not (isinstance(contest, dict) and contest.get("ballotOptions")):
        logger.debug({"level": "DEBUG", "type": "pivot", "message": "RawJSON pivot: contest not loaded from inline/pointer/context."})
        return None, None

    ballot_options = contest.get("ballotOptions") or []
    if not ballot_options:
        return None, None

    # Contest-level reporting % (fallback for rows without status)
    precincts_participating = contest.get("precinctsParticipating")
    precincts_reporting = contest.get("precinctsReporting")
    contest_reporting_percent = None
    try:
        if precincts_participating and precincts_reporting is not None and precincts_participating > 0:
            pct_val = (precincts_reporting / precincts_participating) * 100.0
            # keep as simple string (no % sign) to align with rest of pipeline
            contest_reporting_percent = f"{pct_val:.1f}".rstrip("0").rstrip(".")
    except Exception:
        pass

    # Collect all canonical subgroup names (from option-level and precinct-level results)
    group_set = set()
    for opt in ballot_options:
        for gr in (opt.get("groupResults") or []):
            group_set.add(canonical_ballot_group(gr.get("groupName", "")))
        for pr in (opt.get("precinctResults") or []):
            for gr in (pr.get("groupResults") or []):
                group_set.add(canonical_ballot_group(gr.get("groupName", "")))

    # Stable group ordering (Election Day first heuristic, then alpha)
    group_order = []
    for pref in ("Election Day", "Early Voting", "Early In-Person", "Absentee", "Absentee Mail", "Mail", "Provisional"):
        for g in list(group_set):
            if g and g not in group_order and g.lower() == pref.lower():
                group_order.append(g)
    for g in sorted(group_set):
        if g and g not in group_order:
            group_order.append(g)

    # Containers
    precinct_rows: dict[str, dict] = {}
    candidate_meta: list[dict] = []

    def _safe_int(x):
        return int(x) if isinstance(x, (int, float)) else _coerce_int(x)

    # Build candidate + precinct distributions
    for opt in ballot_options:
        raw_name = (opt.get("name") or "").strip()
        party_raw = opt.get("politicalParty")
        party = normalize_party_label(party_raw)
        short_name = raw_name
        label = f"{party}: {short_name}" if party else short_name
        candidate_meta.append({
            "label": label,
            "party": party,
            "voteCount": _safe_int(opt.get("voteCount"))
        })

        # Precinct results per candidate
        pr_list = opt.get("precinctResults") or []
        for pr in pr_list:
            pname = pr.get("name") or f"Precinct {pr.get('id')}"
            prow = precinct_rows.setdefault(
                pname,
                {
                    "Precinct": pname,
                    "Percent Reported": "",
                    "_cand_group": {},   # label -> {groupName: count}
                    "_cand_totals": {}   # label -> total count
                }
            )
            # Reporting status (best-effort)
            status = (pr.get("reportingStatus") or "").lower()
            if not prow.get("Percent Reported"):
                if "fully" in status or "reported" in status:
                    prow["Percent Reported"] = "100"
                elif contest_reporting_percent:
                    prow["Percent Reported"] = contest_reporting_percent

            # Per-group counts within this precinct result
            per_group_counts = {g: 0 for g in group_order} if group_order else {}
            has_any_group = False
            for gr in (pr.get("groupResults") or []):
                gcanon = canonical_ballot_group(gr.get("groupName", ""))
                if gcanon:
                    per_group_counts.setdefault(gcanon, 0)
                    per_group_counts[gcanon] += _safe_int(gr.get("voteCount"))
                    has_any_group = True

            # If no subgroup breakdown but a total exists, treat as total only
            if not has_any_group and _safe_int(pr.get("voteCount")):
                prow["_cand_totals"][label] = prow["_cand_totals"].get(label, 0) + _safe_int(pr.get("voteCount"))
            else:
                cgmap = prow["_cand_group"].setdefault(label, {g: 0 for g in group_order})
                for g, v in per_group_counts.items():
                    cgmap[g] = cgmap.get(g, 0) + v
                    prow["_cand_totals"][label] = prow["_cand_totals"].get(label, 0) + v

    # Edge-case fallback: if there are no precinct rows, build a single All Precincts row from option-level totals
    if not precinct_rows:
        # Try to derive groups from option-level groupResults (if available)
        group_counts_by_label = {}
        for opt in ballot_options:
            raw_name = (opt.get("name") or "").strip()
            party_raw = opt.get("politicalParty")
            party = normalize_party_label(party_raw)
            short_name = raw_name
            label = f"{party}: {short_name}" if party else short_name

            # Sum groupResults if available; else use option voteCount
            gmap = defaultdict(int)
            has_any_group = False
            for gr in (opt.get("groupResults") or []):
                gcanon = canonical_ballot_group(gr.get("groupName", ""))
                if gcanon:
                    gmap[gcanon] += _safe_int(gr.get("voteCount"))
                    has_any_group = True
            total = sum(gmap.values()) if has_any_group else _safe_int(opt.get("voteCount"))
            group_counts_by_label[label] = (dict(gmap), total)

        # If still nothing, bail out
        if not group_counts_by_label:
            return None, None

        # Construct a synthetic 'All Precincts' row
        synthetic = {
            "Precinct": "All Precincts",
            "Percent Reported": contest_reporting_percent or "",
            "_cand_group": {},
            "_cand_totals": {}
        }
        for meta in candidate_meta:
            lbl = meta["label"]
            gmap, total = group_counts_by_label.get(lbl, ({}, 0))
            if gmap:
                synthetic["_cand_group"][lbl] = {g: _safe_int(v) for g, v in gmap.items()}
            if total:
                synthetic["_cand_totals"][lbl] = _safe_int(total)
        precinct_rows["All Precincts"] = synthetic

    # Headers
    base_headers = ["Precinct", "Percent Reported"]
    cand_headers: list[str] = []
    for meta in candidate_meta:
        lbl = meta["label"]
        # subgroup columns (keep stable order)
        for g in group_order:
            cand_headers.append(f"{lbl} - {g}")
        cand_headers.append(f"{lbl} - Total Vote")
        cand_headers.append(f"{lbl} - % Vote")

    final_headers = base_headers + cand_headers + ["Grand Total"]

    # Municipality column (optional)
    add_muni_col = context.get("add_municipality_column")
    auto_muni = context.get("auto_detect_municipality", True) and not add_muni_col
    municipalities_detected = set()
    muni_map: dict[str, str] = {}

    if auto_muni:
        for pname in precinct_rows.keys():
            m = _extract_municipality(pname)
            if m:
                municipalities_detected.add(m)
        if len(municipalities_detected) >= 3:
            add_muni_col = True

    if add_muni_col:
        for pname in precinct_rows.keys():
            muni_map[pname] = _extract_municipality(pname)
        final_headers = ["Municipality"] + final_headers

    # Optional Division Name column
    include_division_name = bool(context.get("include_division_name_column"))
    if include_division_name:
        insert_after = "Municipality" if "Municipality" in final_headers else "Precinct"
        idx = final_headers.index(insert_after) + 1
        if "Division Name" not in final_headers:
            final_headers.insert(idx, "Division Name")

    # Optional hierarchical headers (metadata only)
    if context.get("produce_hierarchical_headers"):
        top = []
        second = []
        if add_muni_col:
            top += ["", ""]                       # Municipality, Precinct
            second += ["Municipality", "Precinct"]
            top.append("")                        # Percent Reported
            second.append("Percent Reported")
        else:
            top += [""]
            second += ["Precinct"]
            top.append("")
            second.append("Percent Reported")
        for meta in candidate_meta:
            lbl = meta["label"]
            span = len(group_order) + 2  # groups + Total Vote + % Vote
            top.extend([lbl] * span)
            second.extend(group_order + ["Total Vote", "% Vote"])
        top.append("")
        second.append("Grand Total")
        context["hierarchical_headers"] = [top, second]

    # Build rows (natural precinct sort)
    out_rows: list[dict] = []
    for pname in sorted(precinct_rows.keys(), key=_natural_key):
        prow = precinct_rows[pname]
        row_out = {h: "" for h in final_headers}
        if add_muni_col:
            row_out["Municipality"] = muni_map.get(pname, "")
        row_out["Precinct"] = pname
        if include_division_name:
            state = (context or {}).get("state")
            row_out["Division Name"] = _detect_division_name_for_precinct(pname, state, context)
        if prow.get("Percent Reported"):
            row_out["Percent Reported"] = prow["Percent Reported"]

        grand_total = 0
        # Fill candidate subgroup + totals
        for meta in candidate_meta:
            lbl = meta["label"]
            group_map = prow.get("_cand_group", {}).get(lbl, {})
            cand_total = _safe_int(prow.get("_cand_totals", {}).get(lbl, 0))
            # subgroups
            for g in group_order:
                if g in group_map:
                    row_out[f"{lbl} - {g}"] = _safe_int(group_map[g])
            # totals
            row_out[f"{lbl} - Total Vote"] = cand_total
            grand_total += cand_total

        # Local candidate % if grand total present
        if grand_total > 0:
            for meta in candidate_meta:
                lbl = meta["label"]
                tv = _safe_int(prow.get("_cand_totals", {}).get(lbl, 0))
                pct_col = f"{lbl} - % Vote"
                row_out[pct_col] = f"{(tv / grand_total) * 100:.3f}".rstrip("0").rstrip(".")

        row_out["Grand Total"] = grand_total
        out_rows.append(row_out)

    # Municipality aggregates (optional)
    if add_muni_col and context.get("add_municipality_aggregate"):
        muni_agg: dict[str, dict] = {}
        numeric_cols = [h for h in final_headers if h not in ("Municipality", "Precinct", "Percent Reported", "Division Name")]
        for row in out_rows:
            m = row.get("Municipality", "")
            if not m:
                continue
            agg = muni_agg.setdefault(m, {h: "" for h in final_headers})
            agg["Municipality"] = m
            agg["Precinct"] = f"{m} (Aggregate)"
            if include_division_name:
                agg["Division Name"] = "All"
            for col in numeric_cols:
                v = row.get(col, "")
                if isinstance(v, (int, float)):
                    agg[col] = (agg.get(col) or 0) + int(v)
                else:
                    iv = _coerce_int(v)
                    if iv:
                        agg[col] = (agg.get(col) or 0) + iv
        # Recompute % for muni aggregates
        for mrow in muni_agg.values():
            gt = _coerce_int(mrow.get("Grand Total", 0))
            if not gt:
                continue
            for meta in candidate_meta:
                tvc = _coerce_int(mrow.get(f"{meta['label']} - Total Vote", 0))
                pct_col = f"{meta['label']} - % Vote"
                mrow[pct_col] = f"{(tvc / gt) * 100:.3f}".rstrip("0").rstrip(".")
        out_rows.extend(sorted(muni_agg.values(), key=lambda r: r["Municipality"]))

    # Optional "All Precincts" aggregate row
    if context.get("rawjson_include_all_precincts", True) and out_rows:
        agg = {h: "" for h in final_headers}
        agg["Precinct"] = "All Precincts"
        if include_division_name:
            agg["Division Name"] = "All"
        agg_gt = 0
        for h in final_headers:
            if h in ("Precinct", "Percent Reported", "Municipality", "Division Name"):
                continue
            # Only sum numeric cols and candidate group/total cols
            if h.endswith("Total Vote") or h == "Grand Total" or re.search(r" - (Election Day|Early|Absentee|Vote)$", h):
                s = sum(_coerce_int(r.get(h, 0)) for r in out_rows)
                agg[h] = s
                if h == "Grand Total":
                    agg_gt = s
        # Recompute candidate % for aggregate
        if agg_gt:
            for meta in candidate_meta:
                pct_col = f"{meta['label']} - % Vote"
                tv_col = f"{meta['label']} - Total Vote"
                tv = _coerce_int(agg.get(tv_col, 0))
                agg[pct_col] = f"{(tv / agg_gt) * 100:.3f}".rstrip("0").rstrip(".")
        out_rows.append(agg)

    # Context enrichment
    try:
        total_votes_all = sum(_coerce_int(meta["voteCount"]) for meta in candidate_meta)
        ranking = []
        for meta in candidate_meta:
            tv = _coerce_int(meta["voteCount"])
            pct = (tv / total_votes_all * 100.0) if total_votes_all else 0.0
            ranking.append({
                "label": meta["label"],
                "party": meta["party"],
                "total_votes": tv,
                "percent": round(pct, 3)
            })
        ranking.sort(key=lambda r: (-r["total_votes"], r["label"].lower()))
        context["rawjson_enrichment"] = {
            "contest_name": contest.get("name"),
            "candidate_count": len(candidate_meta),
            "ranking": ranking,
            "winner": ranking[0] if ranking else None
        }
    except Exception:
        pass

    # Mark applied
    context["rawjson_pivot_applied"] = True
    context["pivot_modes"] = context.get("pivot_modes", []) + ["rawjson_candidate_groups"]
    try:
        context["structure_hash"] = hashlib.sha256("|".join(final_headers).encode("utf-8")).hexdigest()[:16]
    except Exception:
        pass

    # Drop pointer/inline columns, regardless of casing, if requested
    if drop_rawjson:
        to_drop = {_norm_key("RawJSON"), _norm_key("RawJSONPath"), _norm_key("RawId")}
        final_headers = [h for h in final_headers if _norm_key(h) not in to_drop]
        for r in out_rows:
            for k in list(r.keys()):
                if _norm_key(k) in to_drop:
                    r.pop(k, None)

    return final_headers, out_rows