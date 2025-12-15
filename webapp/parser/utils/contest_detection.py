from __future__ import annotations

"""Shared contest detection helpers for format handlers."""

import os
import re
from collections import Counter
from typing import Iterable, Pattern

from ..Context_Integration.Context_Library.constants import CONTEST_KEYWORDS

__all__ = [
    "CONTEST_PATTERN",
    "detect_contest_titles_from_text",
    "gather_lines_for_contest_detection",
]


def _build_contest_regex(keywords: Iterable[str]) -> Pattern[str]:
    parts: list[str] = []
    for phrase in keywords or []:
        if not isinstance(phrase, str) or not phrase.strip():
            continue
        tokens = re.split(r"\s+", phrase.strip().lower())
        formatted: list[str] = []
        for token in tokens:
            escaped = re.escape(token)
            escaped = escaped.replace(r"\.", r"\.?")
            escaped = escaped.replace(r"\-", r"[-\s]?")
            formatted.append(escaped)
        if not formatted:
            continue
        pattern = r"(?:[\s\-_\/]*?)".join(formatted)
        parts.append(rf"(?<![A-Za-z0-9]){pattern}(?![A-Za-z0-9])")
    if not parts:
        return re.compile(r"(?!x)x", re.I)
    return re.compile("|".join(parts), re.I)


CONTEST_PATTERN = _build_contest_regex(CONTEST_KEYWORDS)
CONTEST_NAME_REGEX = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s*\([^)]*\))?)\b")

_LOUD_CONTEST_TOKENS = {
    "abstract",
    "ballot",
    "ballots",
    "city",
    "counties",
    "county",
    "district",
    "districts",
    "general",
    "official",
    "precinct",
    "precincts",
    "report",
    "state",
    "states",
    "statement",
    "summary",
    "town",
    "township",
    "village",
    "ward",
    "wards",
}

_SHORT_TITLE_ALLOWLIST = {
    "attorney",
    "auditor",
    "controller",
    "governor",
    "judge",
    "mayor",
    "president",
    "sheriff",
    "treasurer",
}


def _should_drop_contest_title(title: str) -> tuple[bool, str | None]:
    clean = (title or "").strip()
    if not clean:
        return True, "empty"
    alpha = sum(ch.isalpha() for ch in clean)
    if alpha < 4:
        return True, "low_alpha"
    words = re.findall(r"[A-Za-z']+", clean.lower())
    if not words:
        return True, "no_words"
    if len(words) == 1 and words[0] not in _SHORT_TITLE_ALLOWLIST:
        return True, "single_generic"
    non_generic = [word for word in words if word not in _LOUD_CONTEST_TOKENS]
    if not non_generic:
        return True, "generic_only"
    return False, None


def detect_contest_titles_from_text(
    lines: Iterable[str] | None,
    pdf_path: str | None = None,
    *,
    diagnostics: dict | None = None,
) -> list[str]:
    """Extract contest-like titles from OCR/text lines with optional diagnostics."""

    diag_bucket = diagnostics if isinstance(diagnostics, dict) else None
    titles: list[str] = []
    kept_samples: list[str] = []
    drop_samples: list[dict[str, str]] = []
    drop_reason_counts: Counter[str] = Counter()
    raw_candidate_total = 0
    sample_limit = 12

    def _format_match(match: str) -> str:
        parts = match.split()
        if len(parts) >= 3 and parts[-1].isdigit():
            location = parts[-2]
            contest = " ".join(parts[:-2])
            if contest and location:
                return f"{contest} ({location})"
        return match

    def _record(candidate_title: str, origin: str) -> None:
        nonlocal raw_candidate_total
        candidate = (candidate_title or "").strip()
        if not candidate:
            return
        raw_candidate_total += 1
        drop, reason = _should_drop_contest_title(candidate)
        if drop:
            if diag_bucket is not None:
                cause = reason or "filtered"
                drop_reason_counts[cause] += 1
                if len(drop_samples) < sample_limit:
                    drop_samples.append({
                        "title": candidate,
                        "reason": cause,
                        "origin": origin,
                    })
            return
        titles.append(candidate)
        if diag_bucket is not None and len(kept_samples) < sample_limit:
            kept_samples.append(candidate)

    scanned_lines = list(lines or [])
    filename_hint: str | None = None
    if pdf_path:
        filename_line = os.path.basename(pdf_path).replace(".pdf", "")
        filename_hint = filename_line
        if filename_line:
            scanned_lines.append(filename_line)
            if CONTEST_PATTERN.search(filename_line):
                matches = CONTEST_NAME_REGEX.findall(filename_line)
                for match in matches:
                    _record(_format_match(match), "filename")

    for idx, line in enumerate(scanned_lines):
        text = (line or "").strip()
        if not text:
            continue
        if not CONTEST_PATTERN.search(text):
            continue
        matches = CONTEST_NAME_REGEX.findall(text)
        if not matches:
            _record(text, f"line:{idx}")
            continue
        for match in matches:
            _record(_format_match(match), f"line:{idx}")

    if diag_bucket is not None:
        diag_bucket.update({
            "lines_scanned": len(lines or []),
            "raw_candidates": raw_candidate_total,
            "kept_candidates": len(titles),
            "drop_reasons": dict(drop_reason_counts),
            "sample_kept": kept_samples,
            "sample_dropped": drop_samples,
        })
        if filename_hint:
            diag_bucket["filename_line"] = filename_hint

    return titles


def gather_lines_for_contest_detection(
    headers: Iterable[str] | None,
    rows: Iterable[dict] | None,
    *,
    limit_rows: int = 60,
    max_cell_chars: int = 200,
) -> list[str]:
    """Build a compact list of textual lines from tabular data for contest detection."""

    lines: list[str] = []
    if headers:
        for header in headers:
            header_text = str(header or "").strip()
            if header_text:
                lines.append(header_text[:max_cell_chars])

    if not rows:
        return lines

    # Iterate rows conservatively to avoid huge memory usage.
    for idx, row in enumerate(rows):
        if idx >= limit_rows:
            break
        if isinstance(row, dict):
            tokens: list[str] = []
            iterable_headers = list(headers or row.keys())
            for header in iterable_headers:
                value = row.get(header)
                if value is None:
                    continue
                cell = str(value).strip()
                if cell:
                    tokens.append(cell[:max_cell_chars])
            fused = " ".join(tokens)
        else:
            fused = str(row or "").strip()[:max_cell_chars]
        if fused:
            lines.append(fused)

    return lines
