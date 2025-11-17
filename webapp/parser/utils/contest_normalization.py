"""Utilities for normalizing contest titles (referenda, propositions, etc.)."""

from __future__ import annotations

import re
from typing import Optional, Tuple

__all__ = ["normalize_contest_label"]

_REFERENDUM_KEYWORDS = (
    "amendment",
    "proposition",
    "measure",
    "referendum",
    "initiative",
    "question",
    "issue",
    "proposal",
    "bond",
)
_SHALL_TOKEN_RE = re.compile(r"\bshall\b", re.I)
_SEP_RE = re.compile(r"[\s\-–—:]+")


def _split_referendum_title(text: str) -> Tuple[str, Optional[str]]:
    """Split a referendum-style title into a short label and question text."""

    text = text.strip()
    if not text:
        return "", None

    colon_index = text.find(":")
    if colon_index != -1:
        label = text[:colon_index].strip(" -–—")
        question = text[colon_index + 1 :].strip()
        if label:
            return label, question or None

    shall_match = _SHALL_TOKEN_RE.search(text)
    if shall_match and shall_match.start() > 0:
        label = text[:shall_match.start()].strip(" -–—")
        question = text[shall_match.start():].strip()
        if label:
            return label, question or None

    if "?" in text:
        first_clause, remainder = text.split("?", 1)
        candidate_label = first_clause.strip(" -–—")
        lowered = candidate_label.lower()
        if any(keyword in lowered for keyword in _REFERENDUM_KEYWORDS):
            question = (first_clause + "?" + remainder).strip()
            return candidate_label or text, question or None

    return text, None


def _normalize_candidate_label(raw: str) -> str:
    """Condense sequences of separators to make comparisons consistent."""

    return _SEP_RE.sub(" ", (raw or "").strip())


def normalize_contest_label(
    raw_title: Optional[str],
    *,
    short_title: Optional[str] = None,
) -> Tuple[str, Optional[str], Optional[str]]:
    """Normalize contest titles, splitting referenda into label/question pairs.

    Returns a tuple of ``(normalized_label, question_text, raw_label)`` where
    ``raw_label`` preserves the original title if available.
    """

    short_candidate = (short_title or "").strip()
    raw_candidate = (raw_title or "").strip()
    working = short_candidate or raw_candidate

    if not working:
        return "", None, raw_candidate or None

    normalized_label, question = _split_referendum_title(working)
    normalized_label = normalized_label or working
    normalized_label = _normalize_candidate_label(normalized_label)

    if question:
        question = question.strip()
    else:
        if short_candidate and _normalize_candidate_label(short_candidate) != normalized_label:
            question = raw_candidate if raw_candidate and _normalize_candidate_label(raw_candidate) != normalized_label else short_candidate
        elif raw_candidate and _normalize_candidate_label(raw_candidate) != normalized_label:
            question = raw_candidate

    if question:
        question = question.strip()

    return normalized_label, question or None, raw_candidate or (short_candidate or None)
