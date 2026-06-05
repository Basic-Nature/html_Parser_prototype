"""Simple selector scoring utilities for navigation candidate ranking."""
from __future__ import annotations

from typing import List


def score_selector_candidate(
    candidate_label: str,
    contest_title: str,
    keywords: List[str] | None = None,
) -> float:
    score = 0.0
    label = (candidate_label or "").strip().lower()
    contest = (contest_title or "").strip().lower()

    if contest and contest in label:
        score += 2.0

    if keywords:
        for keyword in keywords:
            if keyword.strip().lower() in label:
                score += 1.0

    return score
