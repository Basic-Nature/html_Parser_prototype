from __future__ import annotations

import re
from collections import Counter
from typing import Sequence, Tuple

from ..utils.shared_logic import normalize_county_name, normalize_state_name
from .Context_Library.constants import KNOWN_STATE_TO_COUNTY_MAP


def infer_county_from_lines(
    state_hint: str | None,
    lines: Sequence[str] | None,
    *,
    max_lines: int = 4000,
    require_keyword: bool = True,
) -> Tuple[str | None, int]:
    """Infer a county for the provided state by scanning sanitized OCR lines.

    Args:
        state_hint: Raw or normalized state identifier ("minnesota", "mn", etc.).
        lines: Sequence of OCR/text lines to scan.
        max_lines: Guardrail to avoid scanning unbounded documents.
        require_keyword: When True, only consider lines that explicitly contain
            the word "county" (helps reduce false-positives in dense tables).

    Returns:
        Tuple of (normalized_county_name, hit_count). When inference fails the
        tuple defaults to (None, 0).
    """
    state_normalized = normalize_state_name(state_hint)
    if not state_normalized:
        return None, 0

    county_list = KNOWN_STATE_TO_COUNTY_MAP.get(state_normalized)
    if not county_list:
        return None, 0

    normalized_lookup = {
        normalize_county_name(county) or "": county
        for county in county_list
    }
    normalized_lookup = {key: value for key, value in normalized_lookup.items() if key}
    if not normalized_lookup:
        return None, 0

    hits: Counter[str] = Counter()
    for idx, line in enumerate(lines or []):
        if idx >= max_lines:
            break
        lowered = (line or "").lower()
        if require_keyword and "county" not in lowered:
            continue
        normalized_line = " " + re.sub(r"[^a-z ]+", " ", lowered) + " "
        normalized_line = re.sub(r"\s+", " ", normalized_line)
        for norm in normalized_lookup.keys():
            if len(norm) < 3:
                continue
            token = f" {norm} "
            if token in normalized_line:
                hits[norm] += 1

    if not hits:
        return None, 0

    best_norm, best_hits = hits.most_common(1)[0]
    return best_norm, best_hits
