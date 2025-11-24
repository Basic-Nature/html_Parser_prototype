from __future__ import annotations

"""Shared helpers for PDF table heuristics and text parsing.

This module extracts reusable utilities from the PDF handler so other format
handlers or diagnostic tools can benefit without re-implementing the same
logic. The functions here are intentionally lightweight and free of pipeline
side effects (logging, file I/O, etc.).
"""

import os
import re
from collections import Counter
from typing import Any, Iterable, Sequence

from ..Context_Integration.Context_Library.constants import (
    BALLOT_TYPES,
    PARTY_KEYWORDS,
    CONTEST_KEYWORDS,
    normalize_party_label,
)
from .header_utils import collapse_multiline_header


_RECON_DEBUG_EVENTS: list[dict] = []


def _recon_debug_enabled() -> bool:
    flag = os.environ.get("SMART_ELECTIONS_RECON_DEBUG")
    if not flag:
        return False
    low = flag.strip().lower()
    return low not in {"0", "false", "no", "off"}


def _record_recon_event(event: dict | None) -> None:
    if not event or not _recon_debug_enabled():
        return
    _RECON_DEBUG_EVENTS.append(event)


def consume_reconstruction_debug_events() -> list[dict]:
    if not _RECON_DEBUG_EVENTS:
        return []
    events = list(_RECON_DEBUG_EVENTS)
    _RECON_DEBUG_EVENTS.clear()
    return events


_NUM_RE = re.compile(r"(?P<num>\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
_PCT_RE = re.compile(r"(?P<pct>\d{1,3}(?:\.\d+)?)\s*%")

_HEADER_STOPWORDS = {
    "total",
    "totals",
    "vote",
    "votes",
    "ballot",
    "ballots",
    "absentee",
    "mail",
    "early",
    "provisional",
    "grand",
    "report",
    "summary",
    "precinct",
    "precincts",
    "district",
    "districts",
    "ward",
    "wards",
    "registered",
    "turnout",
    "percent",
    "percentage",
    "pct",
    "%",
    "counted",
    "cast",
    "election",
    "day",
    "poll",
    "party",
}

_PARTY_TERMS = {str(p).lower() for p in PARTY_KEYWORDS if isinstance(p, str)} | {
    "dem",
    "democrat",
    "democratic",
    "rep",
    "republican",
    "gop",
    "ind",
    "independent",
    "lib",
    "libertarian",
    "green",
    "nonpartisan",
    "np",
}

_NUMERIC_TOKEN_RE = re.compile(r"^\s*[-+]?\d[\d,]*(?:\.\d+)?\s*(?:%+)?\s*$")
_PARTY_KEY_PATTERN = re.compile(r"\((?P<code>[A-Za-z]{1,4})\)\s*(?P<label>[A-Za-z][A-Za-z .&'/-]*)")
_PARTY_EQUALS_PATTERN = re.compile(r"\b(?P<code>[A-Za-z]{1,4})\b\s*=\s*(?P<label>[A-Za-z][A-Za-z .&'/-]*)")

_DISTRICT_NUM_RE = re.compile(r"^(?P<num>\d{1,3})(?:st|nd|rd|th)?$", re.I)
_DISTRICT_KEYWORDS = {"district", "dist"}
_DISTRICT_HEADING_EXCLUDE = {"total", "totals", "summary", "report", "results"}


def detect_district_heading(text: str) -> tuple[bool, str | None, str | None]:
    """Identify district boundary headings and return (matched, district_number, display_label)."""
    if not isinstance(text, str):
        return False, None, None
    cleaned = re.sub(r"\s{2,}", " ", text.strip())
    if not cleaned:
        return False, None, None
    lowered = cleaned.lower()
    if "district" not in lowered and "dist" not in lowered:
        return False, None, None
    if any(token in lowered for token in _DISTRICT_HEADING_EXCLUDE):
        return False, None, None
    if "districts" in lowered:
        return False, None, None

    tokens = [tok.strip(".,;:()[]{}#") for tok in cleaned.replace("-", " ").split()]
    if not tokens:
        return False, None, None

    def _parse_number(token: str) -> str | None:
        candidate = token.strip().lower().strip(".,;:()[]{}#")
        match = _DISTRICT_NUM_RE.match(candidate)
        if match:
            return match.group("num")
        return None

    district_positions: list[int] = []
    for idx, token in enumerate(tokens):
        normalized = token.lower().strip(".,;:()[]{}")
        if normalized in _DISTRICT_KEYWORDS or normalized.startswith("district"):
            district_positions.append(idx)

    if not district_positions:
        return False, None, None

    district_number: str | None = None
    for pos in district_positions:
        # Look for number immediately before the keyword
        if pos > 0:
            candidate = _parse_number(tokens[pos - 1])
            if candidate:
                district_number = candidate
        if district_number:
            break
        # Look for number after the keyword (skip "No." / "Number" / "#")
        if pos + 1 < len(tokens):
            look_idx = pos + 1
            if tokens[look_idx].lower() in {"no", "number", "#"} and look_idx + 1 < len(tokens):
                look_idx += 1
            candidate = _parse_number(tokens[look_idx])
            if candidate:
                district_number = candidate
        if district_number:
            break

    if not district_number:
        return False, None, None

    try:
        district_number = str(int(district_number))
    except Exception:
        district_number = district_number.lstrip("0") or district_number

    display_label = cleaned.strip("-: ")
    return True, district_number, display_label


def build_contest_regex(keywords: Iterable[str]) -> re.Pattern:
    parts: list[str] = []
    for phrase in keywords or []:  # type: ignore[arg-type]
        if not isinstance(phrase, str) or not phrase.strip():
            continue
        toks = re.split(r"\s+", phrase.strip().lower())
        escaped = []
        for token in toks:
            token = re.escape(token)
            token = token.replace(r"\.", r"\.?")
            token = token.replace(r"\-", r"[-\s]?")
            escaped.append(token)
        pattern = r"(?:[\s\-_\/]*?)".join(escaped)
        pattern = rf"(?<![A-Za-z0-9]){pattern}(?![A-Za-z0-9])"
        parts.append(pattern)
    if not parts:
        return re.compile(r"(?!x)x", re.I)
    return re.compile("|".join(parts), re.I)


CONTEST_TITLE_REGEX = build_contest_regex(CONTEST_KEYWORDS or [])


def normalize_text_token(s: str) -> str:
    text = (s or "").lower().strip()
    text = re.sub(r"[\s\-_\/]+", " ", text)
    return re.sub(r"[^a-z0-9 %]+", "", text)


def token_set(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))


def header_signature(label: str) -> set[str]:
    collapsed = collapse_multiline_header(label or "")
    tokens = set(re.findall(r"[a-z0-9]+", collapsed.lower()))
    return {tok for tok in tokens if tok and tok not in _HEADER_STOPWORDS}


def looks_like_candidate_header(label: str) -> bool:
    if not isinstance(label, str) or not label.strip():
        return False
    signature = header_signature(label)
    if not signature:
        return "(" in label and ")" in label
    if len(signature) == 1 and not ("(" in label and ")" in label):
        return False
    letters = sum(1 for ch in label if ch.isalpha())
    if letters < 4:
        return False
    return True


def compute_header_richness(candidate_headers: Sequence[str]) -> dict[str, float]:
    headers = list(candidate_headers or [])
    if not headers:
        return {
            "parentheses_ratio": 0.0,
            "multi_token_ratio": 0.0,
            "party_ratio": 0.0,
            "avg_length_norm": 0.0,
            "richness": 0.0,
        }
    total = len(headers)
    parentheses_ratio = sum(1 for h in headers if "(" in h and ")" in h) / total
    multi_token_ratio = sum(1 for h in headers if len(header_signature(h)) >= 2) / total
    party_ratio = sum(1 for h in headers if any(term in h.lower() for term in _PARTY_TERMS)) / total
    avg_length_norm = min(1.0, sum(len(h) for h in headers) / (total * 24.0))
    richness = min(1.0, 0.4 * parentheses_ratio + 0.3 * multi_token_ratio + 0.2 * party_ratio + 0.1 * avg_length_norm)
    return {
        "parentheses_ratio": round(parentheses_ratio, 4),
        "multi_token_ratio": round(multi_token_ratio, 4),
        "party_ratio": round(party_ratio, 4),
        "avg_length_norm": round(avg_length_norm, 4),
        "richness": round(richness, 4),
    }


def is_numeric_like(token: str) -> bool:
    if not isinstance(token, str):
        return False
    text = token.strip()
    if not text:
        return False
    if any(ch.isalpha() for ch in text):
        return False
    return bool(_NUMERIC_TOKEN_RE.match(text))


def normalize_numeric_token(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    if text.endswith("%"):
        return text.replace(" ", "")
    return text.replace(",", "").replace(" ", "")


def compute_numeric_fill(rows: Sequence[dict], candidate_headers: Sequence[str]) -> float:
    total_cells = len(rows or []) * len(candidate_headers or [])
    if total_cells == 0:
        return 0.0
    filled = 0
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        for header in candidate_headers:
            value = row.get(header, "")
            if value is None:
                continue
            if isinstance(value, (int, float)):
                filled += 1
                continue
            value_str = str(value).strip()
            if not value_str:
                continue
            if is_numeric_like(value_str):
                filled += 1
    return round(filled / total_cells, 4)


def evaluate_table_candidate_quality(headers: Sequence[str], rows: Sequence[dict], contest_title: str) -> dict[str, object]:
    headers_list = list(headers or [])
    rows_list = list(rows or [])
    candidate_headers: list[str] = []
    seen: set[str] = set()
    for header in headers_list:
        if header in seen:
            continue
        if looks_like_candidate_header(header):
            candidate_headers.append(header)
            seen.add(header)
    if not candidate_headers:
        fallback_terms = (
            "candidate",
            "name",
            "party",
            "vote",
            "votes",
            "total",
            "absentee",
            "mail",
            "early",
            "provisional",
            "percent",
            "election",
        )
        location_terms = (
            "ward",
            "precinct",
            "district",
            "county",
            "town",
            "city",
            "parish",
        )
        for header in headers_list:
            if not isinstance(header, str):
                continue
            low = header.lower()
            if any(loc in low for loc in location_terms):
                continue
            if any(term in low for term in fallback_terms):
                if header not in seen:
                    candidate_headers.append(header)
                    seen.add(header)
    if not candidate_headers:
        candidate_headers = [h for h in headers_list if isinstance(h, str) and h.strip()]
    richness_metrics = compute_header_richness(candidate_headers)
    numeric_fill = compute_numeric_fill(rows_list, candidate_headers)
    row_density = 0.0
    if candidate_headers:
        row_density = min(1.0, len(rows_list) / max(1, len(candidate_headers)))
    title_tokens = token_set(contest_title)
    table_tokens: set[str] = set()
    for header in headers_list[:6]:
        table_tokens |= header_signature(header)
    if rows_list:
        sample_row = rows_list[0]
        if isinstance(sample_row, dict):
            for value in list(sample_row.values())[:6]:
                table_tokens |= header_signature(str(value))
    alignment = 0.0
    if title_tokens and table_tokens:
        alignment = min(1.0, len(title_tokens & table_tokens) / len(title_tokens))
    score = min(
        1.0,
        0.45 * richness_metrics["richness"]
        + 0.3 * numeric_fill
        + 0.15 * row_density
        + 0.1 * alignment,
    )
    return {
        "score": round(score, 4),
        "rows": len(rows_list),
        "candidate_columns": len(candidate_headers),
        "details": {
            **richness_metrics,
            "numeric_fill": round(numeric_fill, 4),
            "row_density": round(row_density, 4),
            "contest_alignment": round(alignment, 4),
        },
    }


def find_best_header_match(source: str, targets: Sequence[str]) -> str | None:
    signature = header_signature(source)
    if not signature:
        return next((t for t in targets if t and t.strip().lower() == source.strip().lower()), None)
    best = (0.0, None)
    for target in targets:
        if not isinstance(target, str):
            continue
        target_sig = header_signature(target)
        if not target_sig:
            continue
        inter = len(signature & target_sig)
        union = len(signature | target_sig) or 1
        score = inter / union
        if score > best[0]:
            best = (score, target)
    if best[0] >= 0.45:
        return best[1]
    return None


def normalize_anchor_value(value) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip()).lower()


def merge_camelot_with_text(
    camelot_table: dict,
    text_headers: Sequence[str],
    text_rows: Sequence[dict],
) -> tuple[list[str], list[dict]] | None:
    if not camelot_table or not text_headers or not text_rows:
        return None
    camelot_headers = list(camelot_table.get("headers") or [])
    camelot_rows = list(camelot_table.get("rows") or [])
    if not camelot_headers or not camelot_rows:
        return None

    header_map: dict[str, str] = {}
    for ch in camelot_headers:
        best = find_best_header_match(ch, text_headers)
        if best:
            header_map[ch] = best

    anchor_header = camelot_headers[0]
    anchor_text_header = header_map.get(anchor_header)
    if not anchor_text_header and text_headers:
        anchor_text_header = text_headers[0]
    if not anchor_text_header:
        return None

    text_index = {
        normalize_anchor_value(row.get(anchor_text_header)): row
        for row in text_rows
        if isinstance(row, dict)
    }
    camelot_index = {
        normalize_anchor_value(row.get(anchor_header)): row
        for row in camelot_rows
        if isinstance(row, dict)
    }

    merged_rows: list[dict] = []
    seen_keys: set[str] = set()
    for key, text_row in text_index.items():
        if not key and camelot_index:
            continue
        camelot_row = camelot_index.get(key)
        merged_row: dict = {}
        for ch in camelot_headers:
            text_key = header_map.get(ch)
            value = ""
            if camelot_row and camelot_row.get(ch) not in (None, ""):
                value = camelot_row.get(ch)
            elif text_key and text_row.get(text_key) not in (None, ""):
                value = text_row.get(text_key)
            elif text_row.get(ch) not in (None, ""):
                value = text_row.get(ch)
            merged_row[ch] = value if value is not None else ""
        merged_rows.append(merged_row)
        seen_keys.add(key)

    for key, camelot_row in camelot_index.items():
        if key in seen_keys:
            continue
        merged_rows.append({ch: camelot_row.get(ch, "") for ch in camelot_headers})

    return camelot_headers, merged_rows


def best_title_match_idx(
    lines: Sequence[str],
    selected_title: str,
    contest_regex: re.Pattern | None = None,
) -> int:
    if not selected_title:
        return -1
    sel_tok = token_set(selected_title)
    best = (-1.0, -1)
    scan_limit = min(len(lines), 5000)
    for i, line in enumerate(lines[:scan_limit]):
        lt = token_set(line)
        if not lt:
            continue
        inter = len(sel_tok & lt)
        union = len(sel_tok | lt) or 1
        jacc = inter / union
        if contest_regex and contest_regex.search((line or "").lower()):
            jacc += 0.05  # tiny boost for lines that look like contest headings
        if jacc > best[0]:
            best = (jacc, i)
    return best[1]


def extract_contest_block(
    lines: Sequence[str],
    selected_title: str,
    contest_regex: re.Pattern | None = None,
    *,
    line_records: Sequence[dict] | None = None,
    include_metadata: bool = False,
) -> list[str] | tuple[list[str], dict]:
    if not lines:
        result: list[str] | tuple[list[str], dict]
        if include_metadata:
            result = ([], {
                "selected_title": selected_title,
                "heading_index": None,
                "line_slice": None,
                "line_count": 0,
                "page_range": None,
                "pages": [],
                "termination_reason": "no_lines",
            })
        else:
            result = []
        return result

    regex = contest_regex or CONTEST_TITLE_REGEX
    start_idx = best_title_match_idx(lines, selected_title, regex)
    if start_idx < 0:
        if include_metadata:
            return [], {
                "selected_title": selected_title,
                "heading_index": None,
                "line_slice": None,
                "line_count": 0,
                "page_range": None,
                "pages": [],
                "termination_reason": "heading_not_found",
            }
        return []

    block: list[str] = []
    block_indices: list[int] = []
    blanks = 0
    limit = min(len(lines), start_idx + 800)
    termination_reason = "end_of_document"

    last_page = None
    if line_records and 0 <= start_idx < len(line_records):
        last_page = line_records[start_idx].get("page")

    for rel_idx, raw in enumerate(lines[start_idx + 1 : limit], start=1):
        global_idx = start_idx + rel_idx
        record = None
        if line_records and 0 <= global_idx < len(line_records):
            record = line_records[global_idx]
            current_page = record.get("page")
            if current_page != last_page:
                blanks = 0
                last_page = current_page

        text = (raw or "").strip()
        low = text.lower()
        if not text:
            blanks += 1
            if blanks >= 3 and len(block) >= 2:
                termination_reason = "blank_gap"
                break
            continue

        blanks = 0
        if regex.search(low) and len(block) >= 2:
            termination_reason = "next_heading"
            break

        is_district, _district_num, _district_label = detect_district_heading(text)
        if block and is_district and len(block) >= 2:
            termination_reason = "district_heading"
            break

        block.append(text)
        block_indices.append(global_idx)

    if include_metadata:
        start_line = block_indices[0] if block_indices else None
        end_line = block_indices[-1] if block_indices else None

        page_values: list[int] = []
        page_offsets: dict[int, list[int]] = {}

        if line_records:
            for idx in block_indices:
                if 0 <= idx < len(line_records):
                    page_value = line_records[idx].get("page")
                    if isinstance(page_value, int):
                        page_values.append(page_value)
                        span = page_offsets.setdefault(page_value, [None, None])
                        if span[0] is None or idx < span[0]:
                            span[0] = idx
                        if span[1] is None or idx > span[1]:
                            span[1] = idx

        pages_sorted = sorted(dict.fromkeys(page_values)) if page_values else []
        page_range = [pages_sorted[0], pages_sorted[-1]] if pages_sorted else None

        metadata = {
            "selected_title": selected_title,
            "heading_index": start_idx,
            "line_slice": [start_line, end_line] if start_line is not None else None,
            "line_count": len(block_indices),
            "pages": pages_sorted,
            "page_range": page_range,
            "page_offsets": {
                page: offsets for page, offsets in page_offsets.items() if offsets[0] is not None
            },
            "termination_reason": termination_reason,
        }
        return block, metadata

    return block


def parse_candidate_line(line: str, ballot_types: Sequence[str]) -> dict | None:
    if not line or sum(ch.isalpha() for ch in line) < 3:
        return None
    try:
        month_pattern = r"^\s*(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
        if re.match(month_pattern + r"\b", line.strip(), re.I):
            if re.search(r"\b(election|primary|abstract|results|ballot|statement|return)\b", line, re.I):
                return None
            if re.match(month_pattern + r"[\s,]+\d{1,2}(?:st|nd|rd|th)?[\s,]+\d{4}\s*$", line.strip(), re.I):
                return None
            if re.match(month_pattern + r"[\s,]+\d{4}\s*$", line.strip(), re.I):
                return None
    except Exception:
        pass
    pct_match = _PCT_RE.search(line)
    pct_val = f"{pct_match.group('pct')}%" if pct_match else None
    nums = [m.group('num') for m in _NUM_RE.finditer(line)]
    if not nums:
        return None

    alias = {
        "ed": "Election Day",
        "electionday": "Election Day",
        "inperson": "Election Day",
        "early": "Early Voting",
        "early voting": "Early Voting",
        "absentee": "Absentee",
        "mail": "Absentee",
        "by mail": "Absentee",
        "provisional": "Provisional",
        "advance": "Early Voting",
        "total": "Total Vote",
    }

    parts = line.split()
    first_num_idx = None
    for idx, part in enumerate(parts):
        if _NUM_RE.fullmatch(part.strip(",%")):
            first_num_idx = idx
            break

    party = None
    m_paren = re.search(r"\(([^)]+)\)", line)
    if m_paren:
        party = normalize_party_label(m_paren.group(1))
    else:
        trailing = parts[: first_num_idx] if first_num_idx else parts
        if trailing:
            tail = trailing[-1].strip(" ,;")
            if len(tail) <= 12 and any(x in tail.lower() for x in ("dem", "rep", "green", "ind", "wf", "conserv", "lib")):
                party = normalize_party_label(tail)

    name_region = " ".join(parts[: first_num_idx] or parts).strip()
    name_region = re.sub(r"\([^)]*\)", "", name_region).strip()
    name = re.sub(r"\s{2,}", " ", name_region).strip(" -:\t")

    row = {"Candidate": name}
    if party:
        row["Party"] = normalize_party_label(party)

    assigned: dict[str, int] = {}
    norm_line = normalize_text_token(line)
    for bt in ballot_types or []:
        key = normalize_text_token(bt).replace("_", " ")
        if key and key in norm_line:
            idx = norm_line.find(key)
            tail = norm_line[idx:]
            m_val = _NUM_RE.search(tail)
            if m_val:
                assigned[bt] = int(m_val.group("num").replace(",", ""))
    for key, value in alias.items():
        if key in norm_line and value not in assigned:
            idx = norm_line.find(key)
            tail = norm_line[idx:]
            m_val = _NUM_RE.search(tail)
            if m_val:
                assigned[value] = int(m_val.group("num").replace(",", ""))

    total_val = None
    if not assigned:
        try:
            total_val = int(nums[-1].replace(",", ""))
        except Exception:
            total_val = None
    else:
        total_keys = [k for k in assigned.keys() if "total" in k.lower()]
        if total_keys:
            total_val = assigned.get(total_keys[0])
        else:
            total_val = sum(assigned.values()) if assigned else None

    for key, value in assigned.items():
        row[key] = value
    if total_val is not None:
        row["Total Vote"] = total_val
    if pct_val:
        row["% Vote"] = pct_val
    if "Total Vote" not in row and not assigned:
        return None
    return row


def extract_candidate_totals_from_lines(
    lines: Sequence[str],
    selected_title: str,
    ballot_types: Sequence[str] | None = None,
    contest_regex: re.Pattern | None = None,
) -> tuple[list[str], list[dict]]:
    block = extract_contest_block(lines, selected_title, contest_regex)
    if not block:
        return [], []
    ballot_list = list(ballot_types or BALLOT_TYPES or [])
    if not ballot_list:
        ballot_list = ["Election Day", "Early Voting", "Absentee", "Provisional"]
    rows: list[dict] = []
    present_cols = {"Candidate", "Party"}
    for line in block:
        row = parse_candidate_line(line, ballot_list)
        if row:
            rows.append(row)
            present_cols.update(row.keys())
    if not rows:
        return [], []
    headers = ["Candidate"]
    if "Party" in present_cols:
        headers.append("Party")
    for group in ballot_list:
        if group in present_cols:
            headers.append(group)
    if "Total Vote" in present_cols:
        headers.append("Total Vote")
    if "% Vote" in present_cols:
        headers.append("% Vote")
    normalized_rows = [{h: rec.get(h, "") for h in headers} for rec in rows]
    return headers, normalized_rows


def split_ws_blocks(s: str) -> list[str]:
    """Split a line into cells using multi-space, tab, or comma separators.

    Numeric thousands separators are removed prior to splitting so values like
    "23,476" stay in a single cell.
    """
    text = (s or "").strip()
    if not text:
        return []
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    cells = re.split(r"\s{2,}|\t|,", text)
    return [c.strip() for c in cells if c.strip()]


def is_bad_header_line(line: str) -> bool:
    if not isinstance(line, str):
        return True
    text = line.strip()
    if not text:
        return True
    low = text.lower()
    bad_tokens = (
        "statement and return",
        "printed as of",
        "page",
        "of",
        "total applicable ballots",
        "public counter",
        "manually counted emergency",
        "absentee / military",
        "unrecorded",
        "affidavit",
        "less - inapplicable",
        "vote for",
        "page",
    )
    if any(bt in low for bt in bad_tokens):
        if "vote for" in low and len(text) < 40:
            pass
        else:
            return True
    cells = split_ws_blocks(text)
    if len(cells) > 12:
        return True
    if any(len(c) > 80 for c in cells):
        return True
    digits = sum(ch.isdigit() for ch in text)
    if digits and digits / max(1, len(text)) > 0.35:
        return True
    return False


def table_looks_bad(headers: Sequence[str], rows: Sequence[dict]) -> bool:
    if not headers:
        return True
    if len(rows) <= 3:
        return True
    lowered = [h.lower() for h in headers if isinstance(h, str)]
    boiler = ("statement and return", "printed as of", "total applicable ballots", "page ")
    if any(any(b in h for b in boiler) for h in lowered):
        return True
    if any(len(h) > 80 for h in headers if isinstance(h, str)):
        return True
    if any((sum(ch.isdigit() for ch in h) / max(1, len(h))) > 0.35 for h in headers if isinstance(h, str)):
        return True
    return False


def find_header_line(lines: Sequence[str], hints: set[str], max_scan: int = 400) -> tuple[list[str], int]:
    best = (-1, -1, [])
    for idx, line in enumerate(lines[:max_scan]):
        if is_bad_header_line(line):
            continue
        cells = split_ws_blocks(line)
        if len(cells) > 12:
            continue
        if len(cells) >= 2:
            score = sum(1 for h in hints if h in line.lower())
            if score > best[0]:
                best = (score, idx, cells)
    if best[1] >= 0:
        return best[2], best[1]
    for idx, line in enumerate(lines[:max_scan]):
        if is_bad_header_line(line):
            continue
        cells = split_ws_blocks(line)
        if 3 <= len(cells) <= 12:
            return cells, idx
    return [], -1


def extract_table_by_whitespace(lines: Sequence[str], start_idx: int, headers: Sequence[str]) -> list[dict]:
    data: list[dict] = []
    min_cols = max(2, len(headers))
    for raw in lines[start_idx + 1 :]:
        if not (raw or "").strip():
            if data:
                break
            continue
        cells = split_ws_blocks(raw)
        if len(cells) < min_cols:
            if len(headers) == 1 and len(cells) == 1:
                data.append({headers[0]: cells[0]})
            else:
                if data:
                    break
                continue
        else:
            if len(cells) > len(headers):
                cells = list(cells[: len(headers) - 1]) + [" ".join(cells[len(headers) - 1 :])]
            row = dict(zip(headers, cells))
            data.append(row)
    return data


def matches_anchor_header(raw: str) -> bool:
    if not raw:
        return False
    text = raw.strip().lower()
    if not text:
        return False
    anchors = {
        "county",
        "precinct",
        "municipality",
        "ward",
        "district",
        "city",
        "town",
        "township",
        "borough",
        "parish",
        "county totals",
        "precinct totals",
        "precincts",
    }
    if text in anchors:
        return True
    return any(text.endswith(f" {anchor}") for anchor in anchors)


def _looks_like_vertical_stub(line: str) -> bool:
    if not isinstance(line, str):
        return False
    text = line.strip()
    if not text:
        return False
    low = text.lower()
    if any(ch.isdigit() for ch in text):
        return False
    if "|" in text or ":" in text:
        return False
    if any(keyword in low for keyword in ("abstract", "election", "november", "return", "official results")):
        return False
    return True


def _merge_token_fragments(tokens: Sequence[str]) -> list[str]:
    merged: list[str] = []
    for raw in tokens or []:
        text = re.sub(r"\s{2,}", " ", (raw or "").strip())
        if not text:
            continue
        if text.startswith("(") and text.endswith(")") and merged:
            merged[-1] = f"{merged[-1]} {text}"
            continue
        merged.append(text)
    return merged


def _clean_candidate_stub(label: str) -> str:
    text = (label or "").strip()
    if not text:
        return ""
    text = text.replace("**", "").strip("* ")
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+\(\s+", " (", text)
    return text.strip(" -")


def _compose_vertical_headers(
    anchor_label: str,
    pre_tokens: Sequence[str],
    post_tokens: Sequence[str],
) -> list[str]:
    pre: list[str] = []
    for token in pre_tokens:
        cleaned = _clean_candidate_stub(token)
        if cleaned:
            pre.append(cleaned)
    post: list[str] = []
    for token in post_tokens:
        cleaned = _clean_candidate_stub(token)
        if cleaned:
            post.append(cleaned)
    total_candidates = max(len(pre), len(post))
    if total_candidates <= 1:
        return []
    if len(pre) > total_candidates:
        pre = pre[-total_candidates:]
    if len(post) > total_candidates:
        post = post[-total_candidates:]
    headers: list[str] = [anchor_label]
    for idx in range(total_candidates):
        left_raw = post[idx] if idx < len(post) else ""
        right_raw = pre[idx] if idx < len(pre) else ""
        left = left_raw.strip()
        right = right_raw.strip()
        base = left
        party = ""
        if left:
            m = re.match(r"^(.*?)(\s*\([^)]*\))$", left)
            if m:
                left = m.group(1).strip()
                party = m.group(2).strip()
            else:
                left = left.strip()
        if right:
            if not left:
                base = right
            elif right.lower() not in left.lower():
                base = f"{left} {right}".strip()
            else:
                base = left
        else:
            base = left or right
        if not base:
            base = left or right
        combined = base or left_raw or right_raw
        if party and party not in combined:
            combined = f"{combined} {party}".strip()
        combined = re.sub(r"\s{2,}", " ", combined or "").strip()
        if not combined:
            combined = f"Candidate {idx + 1}"
        headers.append(combined)
    return headers


def _gather_vertical_rows(
    cleaned: Sequence[str],
    start_idx: int,
    anchor_label: str,
    headers: Sequence[str],
    contest_regex: re.Pattern | None,
) -> list[dict]:
    rows: list[dict] = []
    idx = max(0, start_idx)
    candidate_count = max(0, len(headers) - 1)
    while idx < len(cleaned):
        line = (cleaned[idx] or "").strip()
        if not line:
            idx += 1
            continue
        low = line.lower()
        if contest_regex and contest_regex.search(low):
            break
        if matches_anchor_header(line) and line.lower() != anchor_label.lower():
            break
        if is_numeric_like(line):
            idx += 1
            continue
        lookahead = cleaned[idx + 1] if idx + 1 < len(cleaned) else ""
        if not is_numeric_like(lookahead):
            idx += 1
            continue
        location = line
        idx += 1
        values: list[str] = []
        while idx < len(cleaned) and len(values) < candidate_count:
            token = (cleaned[idx] or "").strip()
            if is_numeric_like(token):
                values.append(token)
                idx += 1
                continue
            if token in {"", "-", "--", "—"}:
                values.append("")
                idx += 1
                continue
            break
        if len(values) < candidate_count:
            values.extend([""] * (candidate_count - len(values)))
        row = {anchor_label: location}
        for header, value in zip(headers[1:], values):
            row[header] = value
        rows.append(row)
    return rows


def _reconstruct_vertical_table(
    cleaned: Sequence[str],
    anchor_idx: int,
    contest_regex: re.Pattern | None,
) -> tuple[list[str], list[dict]]:
    anchor_label = cleaned[anchor_idx]
    _record_recon_event({
        "phase": "vertical_attempt",
        "anchor_label": anchor_label,
        "anchor_index": anchor_idx,
    })
    window_start = max(0, anchor_idx - 20)
    pre_window = cleaned[window_start:anchor_idx]
    trailing_block: list[str] = []
    pre_tokens: list[str] = []
    for line in pre_window:
        if _looks_like_vertical_stub(line):
            trailing_block.append(line)
        else:
            trailing_block = []
        if trailing_block:
            pre_tokens = list(trailing_block)
    pre_tokens = _merge_token_fragments(pre_tokens)
    if contest_regex:
        filtered: list[str] = []
        for token in pre_tokens:
            if not contest_regex.search(token.lower()):
                filtered.append(token)
        pre_tokens = filtered
    post_tokens_raw: list[str] = []
    idx = anchor_idx + 1
    while idx < len(cleaned):
        token = (cleaned[idx] or "").strip()
        if not token:
            if post_tokens_raw:
                break
            idx += 1
            continue
        low = token.lower()
        if contest_regex and contest_regex.search(low):
            break
        if matches_anchor_header(token) and token.lower() != anchor_label.lower():
            break
        next_line = cleaned[idx + 1] if idx + 1 < len(cleaned) else ""
        if is_numeric_like(next_line):
            break
        if is_numeric_like(token):
            break
        post_tokens_raw.append(token)
        idx += 1
        if len(post_tokens_raw) >= 16:
            break
    post_tokens = _merge_token_fragments(post_tokens_raw)
    headers = _compose_vertical_headers(anchor_label, pre_tokens, post_tokens)
    if not headers:
        _record_recon_event({
            "phase": "vertical_headers_empty",
            "anchor_label": anchor_label,
        })
        return [], []
    rows = _gather_vertical_rows(cleaned, idx, anchor_label, headers, contest_regex)
    if len(rows) < 3:
        _record_recon_event({
            "phase": "vertical_rows_insufficient",
            "anchor_label": anchor_label,
            "row_count": len(rows),
        })
        return [], []
    _record_recon_event({
        "phase": "vertical_success",
        "anchor_label": anchor_label,
        "row_count": len(rows),
        "header_count": len(headers),
    })
    return headers, rows


def _combine_header_rows(rows: Sequence[Sequence[str]]) -> list[str]:
    if not rows:
        return []
    combined = [re.sub(r"\s{2,}", " ", cell).strip() for cell in rows[0]]
    for next_row in rows[1:]:
        normalized = [re.sub(r"\s{2,}", " ", cell).strip() for cell in next_row]
        if len(normalized) < len(combined):
            normalized = list(normalized) + [""] * (len(combined) - len(normalized))
        elif len(normalized) > len(combined):
            combined = combined + [""] * (len(normalized) - len(combined))
        merged: list[str] = []
        for existing, addition in zip(combined, normalized):
            if not addition:
                merged.append(existing.strip())
            elif not existing:
                merged.append(addition.strip())
            else:
                merged.append(f"{existing} {addition}".strip())
        combined = merged
    return [cell.strip() for cell in combined]


def reconstruct_columnar_block(lines: Sequence[str], contest_regex: re.Pattern | None = None) -> tuple[list[str], list[dict]]:
    cleaned = [(line or "").strip() for line in lines if (line or "").strip()]
    if len(cleaned) <= 5:
        return [], []
    regex = contest_regex or CONTEST_TITLE_REGEX

    _record_recon_event({
        "phase": "start_columnar_reconstruction",
        "line_count": len(cleaned),
    })

    aggregated_headers: list[str] | None = None
    aggregated_rows: list[dict] = []

    def _update_aggregated(headers: list[str], rows: list[dict]) -> None:
        nonlocal aggregated_headers, aggregated_rows
        if not rows:
            return
        if not aggregated_headers or len(rows) > len(aggregated_rows):
            aggregated_headers = headers
            aggregated_rows = list(rows)
        elif aggregated_headers == headers:
            aggregated_rows.extend(rows)

    def _likely_location_line(start_idx: int) -> bool:
        numeric_seen = 0
        lookahead_limit = start_idx + 12
        probe_idx = start_idx
        while probe_idx < len(cleaned) and probe_idx < lookahead_limit:
            probe = cleaned[probe_idx]
            if not probe:
                probe_idx += 1
                continue
            cells = split_ws_blocks(probe)
            if not cells:
                probe_idx += 1
                continue
            if all(is_numeric_like(cell) for cell in cells):
                numeric_seen += 1
                if numeric_seen >= 2:
                    return True
                probe_idx += 1
                continue
            return False
        return numeric_seen >= 2

    def _clean_header_token(token: str) -> str:
        text = (token or "").strip()
        if not text:
            return ""
        text = re.sub(r"^\*+", "", text)
        text = re.sub(r"\*+$", "", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

    def _merge_parenthetical_tokens(tokens: list[str]) -> list[str]:
        merged: list[str] = []
        for tok in tokens:
            clean = _clean_header_token(tok)
            if not clean:
                continue
            if re.fullmatch(r"\([^)]{1,24}\)", clean) and merged:
                merged[-1] = f"{merged[-1]} {clean}".strip()
            else:
                merged.append(clean)
        return merged

    def _combine_candidate_tokens(post_token: str, pre_token: str) -> str:
        post_clean = _clean_header_token(post_token)
        pre_clean = _clean_header_token(pre_token)
        if not post_clean and not pre_clean:
            return ""
        if not pre_clean:
            return post_clean
        if not post_clean:
            return pre_clean
        if pre_clean.lower().startswith("misc"):
            return "Misc."
        base = post_clean
        party = ""
        party_match = re.search(r"\s*\([^)]*\)\s*$", base)
        if party_match:
            party = party_match.group(0).strip()
            base = base[: party_match.start()].strip()
        if pre_clean and pre_clean.lower() not in base.lower():
            base = f"{base} {pre_clean}".strip()
        if party:
            return f"{base} {party}".strip()
        return base

    def _dedupe_preserve_order(tokens: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for tok in tokens:
            key = tok.lower()
            if not tok or key in seen:
                continue
            seen.add(key)
            ordered.append(tok)
        return ordered

    def _looks_like_party_definition(token: str) -> bool:
        text = (token or "").strip()
        if not text:
            return False
        lowered = text.lower()
        if lowered in {"party", "party preference", "party affiliation"}:
            return True
        if lowered.startswith("party codes") or lowered.startswith("party preference"):
            return True
        if _PARTY_KEY_PATTERN.search(text) or _PARTY_EQUALS_PATTERN.search(text):
            return True
        cells = [cell.lower() for cell in split_ws_blocks(text)]
        if 1 < len(cells) <= 6:
            alpha_cells = [cell for cell in cells if cell.isalpha()]
            if alpha_cells and all(cell in _PARTY_TERMS or len(cell) <= 3 for cell in alpha_cells):
                return True
        return False

    max_anchor_scan = min(len(cleaned) - 3, 800)
    idx = 0
    while idx < max_anchor_scan:
        raw = cleaned[idx]
        if not matches_anchor_header(raw):
            idx += 1
            continue

        anchor_label = cleaned[idx]
        scan_idx = idx + 1
        header_rows: list[list[str]] = []
        _record_recon_event({
            "phase": "anchor_detected",
            "anchor_label": anchor_label,
            "anchor_index": idx,
        })

        while scan_idx < len(cleaned):
            token = cleaned[scan_idx]
            if matches_anchor_header(token):
                break
            if regex.search(token.lower()):
                break
            cells = split_ws_blocks(token)
            if not cells:
                scan_idx += 1
                continue
            if len(cells) == 1 and not is_numeric_like(cells[0]) and _likely_location_line(scan_idx + 1):
                break
            numeric_cells = sum(1 for cell in cells if is_numeric_like(cell))
            if numeric_cells >= max(1, len(cells) - 1):
                break
            header_rows.append(list(cells))
            scan_idx += 1
            if len(header_rows) >= 12:
                break

        if not header_rows:
            idx = max(idx + 1, scan_idx)
            continue

        pre_tokens: list[str] = []
        look_idx = idx - 1
        while look_idx >= 0 and len(pre_tokens) < 12:
            prev_line = cleaned[look_idx]
            if not prev_line.strip():
                if pre_tokens:
                    break
                look_idx -= 1
                continue
            if matches_anchor_header(prev_line):
                break
            if regex.search(prev_line.lower()):
                break
            if is_bad_header_line(prev_line):
                break
            candidate_cells = split_ws_blocks(prev_line)
            if not candidate_cells:
                look_idx -= 1
                continue
            if any(is_numeric_like(cell) for cell in candidate_cells):
                break
            token_clean = _clean_header_token(prev_line)
            if token_clean:
                pre_tokens.insert(0, token_clean)
            look_idx -= 1

        all_singletons = all(len(row) == 1 for row in header_rows)
        if all_singletons:
            post_tokens = [_clean_header_token(row[0]) for row in header_rows if row and row[0].strip()]
            post_tokens = _merge_parenthetical_tokens(post_tokens)
            candidate_headers = list(post_tokens)
            if pre_tokens and len(candidate_headers) + 1 == len(pre_tokens):
                missing = [tok for tok in pre_tokens if tok.lower().startswith("misc")]
                if missing and missing[0].lower() not in {h.lower() for h in candidate_headers}:
                    candidate_headers.append(missing[0])
            if pre_tokens and len(candidate_headers) == len(pre_tokens):
                combined = []
                for post_token, pre_token in zip(candidate_headers, pre_tokens):
                    combined.append(_combine_candidate_tokens(post_token, pre_token))
                candidate_headers = combined
            candidate_headers = _dedupe_preserve_order(candidate_headers)
            _record_recon_event({
                "phase": "header_rows_singletons",
                "anchor_label": anchor_label,
                "raw_header_rows": header_rows,
                "pre_header_tokens": pre_tokens,
                "derived_headers": candidate_headers,
            })
        else:
            candidate_headers = _combine_header_rows(header_rows)
            candidate_headers = [_clean_header_token(h) for h in candidate_headers if _clean_header_token(h)]
            candidate_headers = _dedupe_preserve_order(candidate_headers)
            if pre_tokens and len(candidate_headers) < len(pre_tokens):
                extras = [tok for tok in pre_tokens if tok.lower() not in {h.lower() for h in candidate_headers}]
                candidate_headers.extend(extras)
                candidate_headers = _dedupe_preserve_order(candidate_headers)
            _record_recon_event({
                "phase": "header_rows_combined",
                "anchor_label": anchor_label,
                "raw_header_rows": header_rows,
                "pre_header_tokens": pre_tokens,
                "combined_headers": candidate_headers,
            })

        if len(candidate_headers) <= 1:
            _record_recon_event({
                "phase": "header_rows_insufficient",
                "anchor_label": anchor_label,
                "candidate_headers": candidate_headers,
                "reason": "<=1 headers after combine",
            })
            alt_headers, alt_rows = _reconstruct_vertical_table(cleaned, idx, regex)
            if alt_headers and alt_rows:
                _update_aggregated(alt_headers, alt_rows)
            idx = max(idx + 1, scan_idx)
            continue

        location_header = anchor_label
        raw_rows: list[dict[str, Any]] = []
        current_location: str | None = None
        current_values: list[str] = []
        active_subcontest_label: str | None = None
        active_subcontest_number: str | None = None
        stop_event: dict | None = None

        def _flush_current_row(reason: str | None = None) -> None:
            nonlocal current_location, current_values, raw_rows
            if current_location is None:
                return
            if current_values:
                raw_rows.append({
                    "location": current_location,
                    "values": list(current_values),
                    "subcontest_label": active_subcontest_label,
                    "subcontest_number": active_subcontest_number,
                })
            elif reason:
                _record_recon_event({
                    "phase": "row_flush_no_values",
                    "anchor_label": anchor_label,
                    "location": current_location,
                    "reason": reason,
                })
            current_location = None
            current_values = []

        while scan_idx < len(cleaned):
            token = cleaned[scan_idx]
            if matches_anchor_header(token) and token.lower() != anchor_label.lower():
                stop_event = {
                    "reason": "next_anchor",
                    "line": token,
                    "line_index": scan_idx,
                }
                _flush_current_row("next_anchor")
                break
            token_is_district, district_number, district_label = detect_district_heading(token)
            if (
                regex.search(token.lower())
                and len(raw_rows) >= 2
                and not _looks_like_party_definition(token)
                and not token_is_district
            ):
                stop_event = {
                    "reason": "next_contest_heading",
                    "line": token,
                    "line_index": scan_idx,
                }
                _flush_current_row("next_contest")
                break

            if token_is_district:
                _record_recon_event({
                    "phase": "district_boundary_detected",
                    "anchor_label": anchor_label,
                    "district_label": district_label,
                    "district_number": district_number,
                    "line_index": scan_idx,
                })
                _flush_current_row("district_boundary")
                current_location = None
                current_values = []
                active_subcontest_label = district_label or token.strip()
                active_subcontest_number = district_number
                scan_idx += 1
                continue

            cells = split_ws_blocks(token)
            if not cells:
                scan_idx += 1
                continue

            if _looks_like_party_definition(token):
                scan_idx += 1
                continue

            numeric_mask = [is_numeric_like(cell) for cell in cells]
            text_cells = [cell for cell, flag in zip(cells, numeric_mask) if not flag]
            numeric_cells = [cell for cell, flag in zip(cells, numeric_mask) if flag]

            if len(text_cells) == len(cells) and len(cells) == 1:
                _flush_current_row("new_location")
                current_location = text_cells[0]
                scan_idx += 1
                continue

            if text_cells and numeric_cells:
                _flush_current_row("inline_location")
                current_location = text_cells[0]
                trailing_numeric = []
                start_idx = cells.index(text_cells[0]) + 1
                for cell in cells[start_idx:]:
                    if is_numeric_like(cell):
                        trailing_numeric.append(cell)
                current_values = trailing_numeric
                scan_idx += 1
                continue

            if numeric_cells:
                if current_location:
                    current_values.extend(numeric_cells)
                scan_idx += 1
                continue

            if text_cells:
                _flush_current_row("compound_location")
                current_location = " ".join(text_cells)
                scan_idx += 1
                continue

            scan_idx += 1

        _flush_current_row("end_of_block")

        if not raw_rows:
            _record_recon_event({
                "phase": "data_rows_insufficient",
                "anchor_label": anchor_label,
                "row_count": 0,
            })
            idx = max(idx + 1, scan_idx)
            continue

        row_lengths = [len(entry.get("values") or []) for entry in raw_rows if entry.get("values")]
        if not row_lengths:
            _record_recon_event({
                "phase": "data_rows_insufficient",
                "anchor_label": anchor_label,
                "row_count": 0,
                "reason": "no_numeric_values",
            })
            idx = max(idx + 1, scan_idx)
            continue

        length_counter = Counter(row_lengths)
        candidate_count = max(length_counter.items(), key=lambda kv: (kv[1], kv[0]))[0]
        _record_recon_event({
            "phase": "candidate_value_span",
            "anchor_label": anchor_label,
            "length_frequency": dict(length_counter),
            "selected_length": candidate_count,
        })

        if candidate_headers and len(candidate_headers) != candidate_count:
            _record_recon_event({
                "phase": "header_value_count_mismatch",
                "anchor_label": anchor_label,
                "header_count": len(candidate_headers),
                "value_count": candidate_count,
            })
            if len(candidate_headers) < candidate_count:
                extras: list[str] = []
                lower_existing = {h.lower() for h in candidate_headers}
                for token in pre_tokens:
                    cleaned_token = _clean_header_token(token)
                    if cleaned_token and cleaned_token.lower() not in lower_existing:
                        extras.append(cleaned_token)
                        lower_existing.add(cleaned_token.lower())
                    if len(candidate_headers) + len(extras) >= candidate_count:
                        break
                while len(candidate_headers) + len(extras) < candidate_count:
                    extras.append(f"Candidate {len(candidate_headers) + len(extras) + 1}")
                candidate_headers.extend(extras[: candidate_count - len(candidate_headers)])
            else:
                candidate_headers = candidate_headers[:candidate_count]

        data_lines: list[list[str]] = []
        padded_rows = 0
        overflow_rows = 0
        numeric_location_rows = 0
        month_tokens = {
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        }

        filtered_entries: list[dict[str, Any]] = []

        for entry in raw_rows:
            values = entry.get("values") or []
            if not values:
                continue
            location_text = entry.get("location", "")
            loc_clean = (location_text or "").strip()
            if loc_clean and any(month in loc_clean.lower() for month in month_tokens):
                _record_recon_event({
                    "phase": "row_footer_skip",
                    "anchor_label": anchor_label,
                    "location": loc_clean,
                    "reason": "month_token_detected",
                })
                continue
            row_values = list(values)
            if len(row_values) < candidate_count:
                padded_rows += 1
                _record_recon_event({
                    "phase": "row_value_padding",
                    "anchor_label": anchor_label,
                    "original_length": len(row_values),
                    "candidate_count": candidate_count,
                    "row_sample": [location_text] + row_values,
                })
                row_values.extend([""] * (candidate_count - len(row_values)))
            elif len(row_values) > candidate_count:
                overflow = row_values[candidate_count - 1 :]
                overflow_rows += 1
                _record_recon_event({
                    "phase": "row_value_overflow",
                    "anchor_label": anchor_label,
                    "original_length": len(row_values),
                    "candidate_count": candidate_count,
                    "overflow_joined": " ".join(overflow),
                    "row_sample": [location_text] + row_values,
                })
                row_values = row_values[: candidate_count - 1] + [" ".join(overflow)]
            if loc_clean and is_numeric_like(loc_clean):
                numeric_location_rows += 1
            normalized_values = row_values[:candidate_count]
            data_lines.append([loc_clean] + normalized_values)
            filtered_entries.append({
                "location": loc_clean,
                "values": normalized_values,
                "subcontest_label": entry.get("subcontest_label"),
                "subcontest_number": entry.get("subcontest_number"),
            })

        min_rows_required = 3
        if candidate_count >= 2:
            min_rows_required = 2
        if len(data_lines) < min_rows_required:
            _record_recon_event({
                "phase": "data_rows_insufficient",
                "anchor_label": anchor_label,
                "row_count": len(data_lines),
                "reason": f"<{min_rows_required} rows after normalization",
            })
            idx = max(idx + 1, scan_idx)
            continue

        if stop_event:
            stop_event = {
                "phase": "data_scan_stopped",
                "anchor_label": anchor_label,
                "rows_collected": len(data_lines),
                **stop_event,
            }
            _record_recon_event(stop_event)

        rows: list[dict] = []
        candidate_count = len(candidate_headers)
        for entry in filtered_entries:
            location = entry.get("location", "")
            values = list(entry.get("values") or [])
            if len(values) < candidate_count:
                values.extend([""] * (candidate_count - len(values)))
            row = {location_header: location}
            for header, value in zip(candidate_headers, values):
                row[header] = value
            if entry.get("subcontest_label"):
                row["_subcontest_label"] = entry["subcontest_label"]
            if entry.get("subcontest_number"):
                row["_subcontest_number"] = entry["subcontest_number"]
            rows.append(row)

        reject_reason: str | None = None
        if padded_rows >= max(1, len(rows) // 2):
            reject_reason = "too_many_padded_rows"
        elif numeric_location_rows >= max(1, len(rows) // 2):
            reject_reason = "numeric_locations"

        if reject_reason:
            _record_recon_event({
                "phase": "row_quality_reject",
                "anchor_label": anchor_label,
                "row_count": len(rows),
                "padded_rows": padded_rows,
                "numeric_location_rows": numeric_location_rows,
                "reason": reject_reason,
            })
            alt_headers, alt_rows = _reconstruct_vertical_table(cleaned, idx, regex)
            if alt_headers and alt_rows:
                _update_aggregated(alt_headers, alt_rows)
        else:
            headers = [location_header] + candidate_headers
            _record_recon_event({
                "phase": "reconstruction_success",
                "anchor_label": anchor_label,
                "row_count": len(rows),
                "candidate_count": candidate_count,
            })
            _update_aggregated(headers, rows)

        idx = max(idx + 1, scan_idx)

    if aggregated_headers and aggregated_rows:
        return aggregated_headers, aggregated_rows
    return [], []


def extract_party_lookup_from_lines(lines: Sequence[str] | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not lines:
        return mapping
    for raw in lines:
        if not raw:
            continue
        for match in _PARTY_KEY_PATTERN.finditer(raw):
            code = match.group("code").upper()
            label = normalize_party_label(match.group("label"))
            if code and label:
                mapping.setdefault(code, label)
        for match in _PARTY_EQUALS_PATTERN.finditer(raw):
            code = match.group("code").upper()
            label = normalize_party_label(match.group("label"))
            if code and label:
                mapping.setdefault(code, label)
    return mapping


def parse_candidate_header_with_party(header: str, party_lookup: dict[str, str]) -> tuple[str, str, dict]:
    info: dict[str, str] = {
        "source_header": header,
    }
    m = _PARTY_KEY_PATTERN.search(header or "")
    party_code = None
    party_label = None
    if m:
        party_code = m.group("code").upper()
        party_label = normalize_party_label(m.group("label"))
    else:
        equals = _PARTY_EQUALS_PATTERN.search(header or "")
        if equals:
            party_code = equals.group("code").upper()
            party_label = normalize_party_label(equals.group("label"))
    if not party_code:
        simple = re.search(r"\((?P<code>[A-Za-z]{1,4})\)", header or "")
        if simple:
            code = simple.group("code").upper()
            if code.isalpha():
                party_code = code
    if party_code and not party_label:
        party_label = normalize_party_label(party_code)
    if not party_label and party_code and party_code in party_lookup:
        party_label = party_lookup.get(party_code)
    tokens = token_set(header)
    for token in list(tokens):
        if len(token) <= 1:
            continue
        if token.upper() in party_lookup and not party_code:
            party_code = token.upper()
            party_label = party_lookup[token.upper()]
            break
    candidate_label = header
    if party_code:
        info["party_code"] = party_code
    if party_label:
        info["party_label"] = party_label
    cleaned = re.sub(r"\(([^)]*)\)", "", header or "").strip()
    if cleaned:
        candidate_label = cleaned
    info["candidate_label"] = candidate_label
    return candidate_label, party_label or "", info


def coerce_vote_value_for_reconstruction(value):
    if isinstance(value, (int, float)):
        try:
            return int(value)
        except Exception:
            return value
    if isinstance(value, str):
        text = value.strip()
        if not text or text.upper() in {"NA", "N/A", "--", "—"}:
            return ""
        digits = text.replace(",", "").replace(" ", "")
        if digits.isdigit():
            try:
                return int(digits)
            except Exception:
                return digits
        try:
            return int(float(digits))
        except Exception:
            return text
    return value


__all__ = [
    "CONTEST_TITLE_REGEX",
    "build_contest_regex",
    "best_title_match_idx",
    "coerce_vote_value_for_reconstruction",
    "compute_header_richness",
    "compute_numeric_fill",
    "evaluate_table_candidate_quality",
    "extract_candidate_totals_from_lines",
    "extract_contest_block",
    "extract_party_lookup_from_lines",
    "extract_table_by_whitespace",
    "detect_district_heading",
    "find_best_header_match",
    "find_header_line",
    "header_signature",
    "is_bad_header_line",
    "is_numeric_like",
    "looks_like_candidate_header",
    "matches_anchor_header",
    "merge_camelot_with_text",
    "normalize_anchor_value",
    "normalize_text_token",
    "normalize_numeric_token",
    "parse_candidate_header_with_party",
    "parse_candidate_line",
    "reconstruct_columnar_block",
    "split_ws_blocks",
    "table_looks_bad",
    "token_set",
]
