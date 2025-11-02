"""
detector.py
Encapsulated column/entity detection + scoring used by refactored table_core.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional

from ..Context_Integration.Context_Library.constants import (
    BALLOT_TYPES,
    CANDIDATE_KEYWORDS,
    LOCATION_ABBREVIATIONS,
    LOCATION_KEYWORDS,
    PERCENT_KEYWORDS,
)
from .shared_logic import safe_add

_NAME_LIKE_RE = re.compile(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}$")
_PERCENT_INLINE_RE = re.compile(r"(\d{1,3})\s*%")

@lru_cache(maxsize=100_000)
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s).strip().lower())
    s = re.sub(r"\s+", " ", s)
    return s

def _numeric_like(val: str) -> bool:
    if val is None:
        return False
    sanitized = str(val).replace(",", "").strip()
    return bool(re.fullmatch(r"-?\d+(?:\.\d+)?%?", sanitized))

@dataclass
class EntityAnnotation:
    people: set = field(default_factory=set)
    locations: set = field(default_factory=set)
    ballot_types: set = field(default_factory=set)
    numbers: set = field(default_factory=set)

class Detector:
    def __init__(self, coordinator=None):
        self.coordinator = coordinator

    # ----------- Normalization ----------- #
    def norm(self, s: str) -> str:
        return _norm(s)

    # ----------- Percent ----------- #
    def is_percent_header(self, h: str) -> bool:
        hn = self.norm(h or "")
        if hn in {_norm(p) for p in PERCENT_KEYWORDS}:
            return True
        return "%" in (h or "").lower() or "reported" in (h or "").lower()

    def extract_percent_inline(self, text: str) -> str:
        if not text:
            return ""
        m = _PERCENT_INLINE_RE.search(text)
        if m:
            return f"{m.group(1)}%"
        if re.search(r"\bfully\s+reported\b", text, re.I):
            return "100%"
        return ""

    # ----------- Location ----------- #
    def is_location_header(self, h: str) -> bool:
        hn = self.norm(h)
        if any(kw in hn for kw in (self.norm(k) for k in LOCATION_KEYWORDS)):
            return True
        if hn in LOCATION_ABBREVIATIONS:
            return True
        return False

    def detect_location_header(self, headers: List[str]) -> Optional[str]:
        for h in headers:
            if self.is_location_header(h) and not self.is_percent_header(h):
                return h
        norm_loc = [self.norm(k) for k in LOCATION_KEYWORDS]
        for h in headers:
            score = max(difflib.SequenceMatcher(None, self.norm(h), loc).ratio() for loc in norm_loc)
            if score >= 0.85 and not self.is_percent_header(h):
                return h
        for h in headers:
            if not self.is_percent_header(h):
                return h
        return None

    def detect_percent_header(self, headers: List[str]) -> Optional[str]:
        for h in headers:
            if self.is_percent_header(h):
                return h
        return None

    # ----------- Candidate ----------- #
    def detect_candidate_column(
        self,
        headers: List[str],
        data: List[Dict[str, Any]],
    ) -> Optional[str]:
        if not headers:
            return None

        normalized_keywords = {self.norm(k) for k in CANDIDATE_KEYWORDS}
        for header in headers:
            if self.norm(header) in normalized_keywords:
                return header

        if self.coordinator and hasattr(self.coordinator, "extract_entities"):
            for header in headers:
                try:
                    entities = self.coordinator.extract_entities(header)
                    if any(label == "PERSON" for _, label in entities):
                        return header
                except Exception:
                    pass

        samples = data[: min(50, len(data))]

        if self.coordinator and hasattr(self.coordinator, "extract_entities"):
            for header in headers:
                hits = 0
                seen = 0
                for row in samples:
                    value = row.get(header, "")
                    if not isinstance(value, str) or not value:
                        continue
                    seen += 1
                    try:
                        entities = self.coordinator.extract_entities(value)
                        if any(label == "PERSON" for _, label in entities):
                            hits += 1
                    except Exception:
                        pass
                if seen and hits / seen >= 0.35:
                    return header

        for header in headers:
            hits = 0
            count = 0
            for row in samples:
                value = row.get(header, "")
                if not isinstance(value, str) or not value:
                    continue
                count += 1
                if _NAME_LIKE_RE.match(value.strip()):
                    hits += 1
            if count and hits / count >= 0.35:
                return header

        for header in headers:
            lowered = header.lower()
            if lowered in {"precinct", "district"}:
                continue
            if self.is_percent_header(header):
                continue
            return header
        return None

    # ----------- Ballot Types ----------- #
    def detect_ballot_types(self, headers: List[str], data: List[Dict[str, Any]]) -> List[str]:
        ballot_type_headers: list[str] = []
        known = {self.norm(b) for b in BALLOT_TYPES}
        for header in headers:
            if self.norm(header) in known:
                ballot_type_headers.append(header)
        if ballot_type_headers:
            return ballot_type_headers
        for header in headers:
            if header.lower() in {"precinct", "candidate", "percent reported"}:
                continue
            values = [row.get(header, "") for row in data]
            non_empty = [value for value in values if value not in ("", None)]
            if not non_empty:
                continue
            numeric = sum(1 for value in non_empty if _numeric_like(str(value)))
            if numeric / len(non_empty) >= 0.5:
                ballot_type_headers.append(header)
        return ballot_type_headers

    # ----------- Entity Annotation ----------- #
    def annotate_entities(self, headers: List[str], data: List[Dict[str, Any]]) -> EntityAnnotation:
        annotation = EntityAnnotation()
        for header in headers:
            if any(bt.lower() in header.lower() for bt in BALLOT_TYPES):
                safe_add(annotation.ballot_types, header)

        for row in data:
            for header, value in row.items():
                if not value:
                    continue
                if isinstance(value, str) and _numeric_like(value):
                    safe_add(annotation.numbers, value)
                if self.is_location_header(header):
                    safe_add(annotation.locations, value)

        if self.coordinator and hasattr(self.coordinator, "extract_entities"):
            candidate_headers = {
                header
                for header in headers
                if self.norm(header) in {self.norm(k) for k in CANDIDATE_KEYWORDS}
            }
            candidates: set[str] = set()
            for header in candidate_headers:
                for row in data:
                    value = row.get(header)
                    if isinstance(value, str):
                        candidates.add(value)
            for name in list(candidates)[:200]:
                try:
                    entities = self.coordinator.extract_entities(name)
                    if any(label == "PERSON" for _, label in entities):
                        safe_add(annotation.people, name)
                except Exception:
                    pass

        return annotation