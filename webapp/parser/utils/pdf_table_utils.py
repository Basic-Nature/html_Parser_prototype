from __future__ import annotations

"""Shared helpers for PDF table heuristics and text parsing.

This module extracts reusable utilities from the PDF handler so other format
handlers or diagnostic tools can benefit without re-implementing the same
logic. The functions here are intentionally lightweight and free of pipeline
side effects (logging, file I/O, etc.).
"""

import re
from typing import Iterable, Sequence

from ..Context_Integration.Context_Library.constants import (
    BALLOT_TYPES,
    PARTY_KEYWORDS,
    CONTEST_KEYWORDS,
    normalize_party_label,
)
from .header_utils import collapse_multiline_header


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
) -> list[str]:
    if not lines:
        return []
    regex = contest_regex or CONTEST_TITLE_REGEX
    start_idx = best_title_match_idx(lines, selected_title, regex)
    if start_idx < 0:
        return []
    block: list[str] = []
    blanks = 0
    limit = min(len(lines), start_idx + 800)
    for raw in lines[start_idx + 1 : limit]:
        text = (raw or "").strip()
        low = text.lower()
        if not text:
            blanks += 1
            if blanks >= 3 and len(block) >= 2:
                break
            continue
        blanks = 0
        if regex.search(low) and len(block) >= 2:
            break
        block.append(text)
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
    cells = re.split(r"\s{2,}|\t|,", (s or "").strip())
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
        return [], []
    rows = _gather_vertical_rows(cleaned, idx, anchor_label, headers, contest_regex)
    if len(rows) < 3:
        return [], []
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

    max_anchor_scan = min(len(cleaned) - 3, 800)
    for idx in range(max_anchor_scan):
        raw = cleaned[idx]
        if not matches_anchor_header(raw):
            continue

        anchor_label = cleaned[idx]
        scan_idx = idx + 1
        header_rows: list[list[str]] = []

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
            numeric_cells = sum(1 for cell in cells if is_numeric_like(cell))
            if numeric_cells >= max(1, len(cells) - 1):
                break
            header_rows.append(list(cells))
            scan_idx += 1
            if len(header_rows) >= 3:
                break

        if not header_rows:
            continue

        candidate_headers = _combine_header_rows(header_rows)
        candidate_headers = [h for h in candidate_headers if h]
        if len(candidate_headers) <= 1:
            alt_headers, alt_rows = _reconstruct_vertical_table(cleaned, idx, regex)
            if alt_headers and alt_rows:
                return alt_headers, alt_rows
            continue

        location_header = anchor_label
        data_lines: list[list[str]] = []
        while scan_idx < len(cleaned):
            token = cleaned[scan_idx]
            if matches_anchor_header(token) and token.lower() != anchor_label.lower():
                break
            if regex.search(token.lower()) and len(data_lines) >= 2:
                break
            cells = split_ws_blocks(token)
            if not cells:
                if data_lines:
                    break
                scan_idx += 1
                continue
            numeric_cells = sum(1 for cell in cells[1:] if is_numeric_like(cell))
            if numeric_cells == 0 and data_lines:
                break
            if numeric_cells == 0:
                scan_idx += 1
                continue
            data_lines.append(cells)
            scan_idx += 1
        if len(data_lines) < 3:
            continue

        rows: list[dict] = []
        candidate_count = len(candidate_headers)
        for cells in data_lines:
            if not cells:
                continue
            location = cells[0]
            values = list(cells[1:])
            if len(values) < candidate_count:
                values.extend([""] * (candidate_count - len(values)))
            elif len(values) > candidate_count:
                overflow = values[candidate_count - 1 :]
                values = values[: candidate_count - 1] + [" ".join(overflow)]
            row = {location_header: location}
            for header, value in zip(candidate_headers, values):
                row[header] = value
            rows.append(row)

        if len(rows) >= 3:
            headers = [location_header] + candidate_headers
            return headers, rows

        if len(rows) < 3:
            alt_headers, alt_rows = _reconstruct_vertical_table(cleaned, idx, regex)
            if alt_headers and alt_rows:
                return alt_headers, alt_rows

    return [], []


def extract_party_lookup_from_lines(lines: Sequence[str] | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not lines:
        return mapping
    for raw in lines:
        if not raw:
            continue
        m = _PARTY_KEY_PATTERN.search(raw)
        if m:
            mapping[m.group("code").upper()] = normalize_party_label(m.group("label"))
            continue
        m = _PARTY_EQUALS_PATTERN.search(raw)
        if m:
            mapping[m.group("code").upper()] = normalize_party_label(m.group("label"))
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
    if party_label:
        info["party_inference"] = "lookup"
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
