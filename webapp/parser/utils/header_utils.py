from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Tuple

from .detect import dedupe_headers_with_suffix, normalize_header
from .salvage import normalize_ballot_column_name


def build_candidate_group_hierarchical(flat_headers: List[str]) -> Dict[str, List[List[str]]]:
    """
    Convert flattened headers like 'Democrat: Jane Doe - Election Day'
    into two header rows:
      Row1: 'Democrat: Jane Doe' (repeated)
      Row2: 'Election Day'
    Base columns (Precinct, Total Ballots Reported, Percent Reported) keep blank second row.
    Returns {"rows": [row1, row2], "style_hint": "candidate_group_pivot_v1"}
    """
    row1 = []
    row2 = []
    for h in flat_headers:
        if h in ("Precinct", "Total Ballots Reported", "Percent Reported"):
            row1.append(h)
            row2.append("")
            continue
        if " - " in h:
            cand, group = h.rsplit(" - ", 1)
            if group == "Total Reported":
                group = "Total"
            row1.append(cand)
            row2.append(group)
        else:
            row1.append(h)
            row2.append("")
    return {"rows": [row1, row2], "style_hint": "candidate_group_pivot_v1"}

def normalize_headers_list(headers: List[str]) -> List[str]:
    """
    Normalize and dedupe a list of header labels using constants-aware normalizer.
    """
    headers = [normalize_header(h) for h in (headers or [])]
    headers = [normalize_ballot_column_name(h) for h in headers]
    return dedupe_headers_with_suffix(headers)


def _clean_header_fragment(token: str) -> str:
    cleaned = (token or "").strip()
    if not cleaned:
        return ""
    cleaned = cleaned.strip("*")
    cleaned = cleaned.replace("**", " ")
    cleaned = cleaned.replace("*", "")
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def _assemble_header_label(top_fragment: str, bottom_fragment: str) -> str:
    top_fragment = _clean_header_fragment(top_fragment)
    bottom_fragment = _clean_header_fragment(bottom_fragment)
    if not bottom_fragment and not top_fragment:
        return ""

    if bottom_fragment:
        match = re.search(r"(\([^)]*\))\s*$", bottom_fragment)
        suffix = ""
        if match:
            suffix = match.group(1)
            bottom_core = bottom_fragment[: match.start()].strip()
        else:
            bottom_core = bottom_fragment
        pieces: List[str] = []
        if bottom_core:
            pieces.append(bottom_core)
        if top_fragment and top_fragment.lower() not in bottom_core.lower():
            pieces.append(top_fragment)
        label = " ".join(pieces).strip()
        if suffix:
            label = f"{label} {suffix}".strip()
        return re.sub(r"\s{2,}", " ", label)

    return top_fragment


def compact_header_tokens(
    tokens: List[str],
    candidate_count: int,
    prior_tokens: List[str] | None = None,
) -> List[str]:
    if candidate_count <= 0:
        return tokens

    cleaned_tokens = [_clean_header_fragment(t) for t in tokens if _clean_header_fragment(t)]
    cleaned_prior = [_clean_header_fragment(t) for t in (prior_tokens or []) if _clean_header_fragment(t)]

    if not cleaned_tokens and not cleaned_prior:
        return []

    if cleaned_prior:
        if len(cleaned_prior) > candidate_count:
            cleaned_prior = cleaned_prior[-candidate_count:]
        while len(cleaned_prior) < candidate_count:
            cleaned_prior.insert(0, "")

    if not cleaned_tokens:
        cleaned_tokens = [""] * candidate_count

    if len(cleaned_tokens) <= candidate_count and not cleaned_prior:
        while len(cleaned_tokens) < candidate_count:
            cleaned_tokens.append("")
        return cleaned_tokens

    if cleaned_prior and len(cleaned_tokens) <= candidate_count:
        top = cleaned_prior
        bottom = cleaned_tokens
    else:
        top = cleaned_prior or cleaned_tokens[:candidate_count]
        bottom = cleaned_tokens if cleaned_prior else cleaned_tokens[candidate_count:]

    while len(top) < candidate_count:
        top.append("")

    merged_bottom: List[str] = []
    for token in bottom:
        if token.startswith("(") and merged_bottom:
            merged_bottom[-1] = f"{merged_bottom[-1]} {token}".strip()
        else:
            merged_bottom.append(token)

    while len(merged_bottom) < candidate_count:
        merged_bottom.append("")

    combined: List[str] = []
    for idx in range(candidate_count):
        top_fragment = top[idx] if idx < len(top) else ""
        bottom_fragment = merged_bottom[idx] if idx < len(merged_bottom) else ""
        combined.append(_assemble_header_label(top_fragment, bottom_fragment))

    if len(merged_bottom) > candidate_count:
        for extra in merged_bottom[candidate_count:]:
            if not extra:
                continue
            combined[-1] = _assemble_header_label(combined[-1], extra)

    return combined


def collapse_multiline_header(raw: str | None) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    if "\n" not in text and "\r" not in text:
        return _clean_header_fragment(text)

    parts = [
        _clean_header_fragment(part)
        for part in re.split(r"[\r\n]+", text)
        if _clean_header_fragment(part)
    ]
    if not parts:
        return _clean_header_fragment(text)
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return _assemble_header_label(parts[0], parts[1])

    top_fragment = " ".join(parts[:-1])
    bottom_fragment = parts[-1]
    return _assemble_header_label(top_fragment, bottom_fragment)


def _register_header_mapping(mapping: Dict[str, str], key: str, label: str) -> None:
    mapping[key] = label
    stripped = key.strip()
    if stripped:
        mapping[stripped] = label


def normalize_table_headers(
    headers: Iterable[str | None],
    rows: List[Dict[str, Any]],
) -> Tuple[List[str], List[Dict[str, Any]]]:
    normalized_headers: List[str] = []
    header_mapping: Dict[str, str] = {}
    seen: Dict[str, int] = {}

    for idx, raw_header in enumerate(headers):
        collapsed = collapse_multiline_header(raw_header)
        if not collapsed:
            collapsed = f"Column {idx + 1}"
        base_key = collapsed.lower()
        count = seen.get(base_key, 0) + 1
        seen[base_key] = count
        final_label = collapsed if count == 1 else f"{collapsed}_{count}"
        normalized_headers.append(final_label)

        canonical = (raw_header or "").strip()
        _register_header_mapping(header_mapping, canonical, final_label)
        if isinstance(raw_header, str):
            _register_header_mapping(header_mapping, raw_header, final_label)
        _register_header_mapping(header_mapping, collapsed, final_label)

    normalized_rows: List[Dict[str, Any]] = []
    for row in rows:
        remapped: Dict[str, Any] = {}
        for key, value in row.items():
            key_str = str(key or "")
            mapped = header_mapping.get(key_str)
            if mapped is None:
                mapped = header_mapping.get(key_str.strip())
            if mapped is None:
                fallback = collapse_multiline_header(key_str) or f"Column {len(normalized_headers) + 1}"
                base_key = fallback.lower()
                count = seen.get(base_key, 0) + 1
                seen[base_key] = count
                mapped = fallback if count == 1 else f"{fallback}_{count}"
                header_mapping[key_str] = mapped
                header_mapping[key_str.strip()] = mapped
                if mapped not in normalized_headers:
                    normalized_headers.append(mapped)
            remapped[mapped] = value
        normalized_rows.append(remapped)

    return normalized_headers, normalized_rows

# If there is an existing function doing similar, keep it but route through ours:
try:
    original_normalize_headers  # type: ignore[name-defined]
except NameError:
    pass
else:
    def original_normalize_headers(headers: List[str]) -> List[str]:  # pyright: ignore[reportRedeclaration]
        return normalize_headers_list(headers)

__all__ = [
    "build_candidate_group_hierarchical",
    "normalize_headers_list",
    "compact_header_tokens",
    "collapse_multiline_header",
    "normalize_table_headers",
]