from __future__ import annotations

from typing import List, Tuple, Dict

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