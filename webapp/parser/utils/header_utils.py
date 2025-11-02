from __future__ import annotations

from typing import Dict, List

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
]