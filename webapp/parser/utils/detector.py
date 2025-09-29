"""
detector.py
Encapsulated column/entity detection + scoring used by refactored table_core.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import re, unicodedata, difflib
from functools import lru_cache

from .shared_logic import safe_get, safe_add
from .logger_singleton import logger
from ..Context_Integration.Context_Library.constants import (
    LOCATION_KEYWORDS, LOCATION_ABBREVIATIONS, PERCENT_KEYWORDS,
    CANDIDATE_KEYWORDS, BALLOT_TYPES, TOTAL_KEYWORDS
)

_NAME_LIKE_RE = re.compile(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}$")
_PERCENT_INLINE_RE = re.compile(r"(\d{1,3})\s*%")

@lru_cache(maxsize=100_000)
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s).strip().lower())
    s = re.sub(r"\s+", " ", s)
    return s

def _numeric_like(val: str) -> bool:
    if val is None: return False
    s = str(val).replace(",", "").strip()
    return bool(re.fullmatch(r"-?\d+(?:\.\d+)?%?", s))

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
    def norm(self, s: str) -> str: return _norm(s)

    # ----------- Percent ----------- #
    def is_percent_header(self, h: str) -> bool:
        hn = self.norm(h or "")
        if hn in {_norm(p) for p in PERCENT_KEYWORDS}: return True
        return "%" in (h or "").lower() or "reported" in (h or "").lower()

    def extract_percent_inline(self, text: str) -> str:
        if not text: return ""
        m = _PERCENT_INLINE_RE.search(text)
        if m: return f"{m.group(1)}%"
        if re.search(r"\bfully\s+reported\b", text, re.I):
            return "100%"
        return ""

    # ----------- Location ----------- #
    def is_location_header(self, h: str) -> bool:
        hn = self.norm(h)
        if any(kw in hn for kw in (self.norm(k) for k in LOCATION_KEYWORDS)): return True
        if hn in LOCATION_ABBREVIATIONS: return True
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
    def detect_candidate_column(self, headers: List[str], data: List[Dict[str, Any]]) -> Optional[str]:
        if not headers: return None
        norm_kw = {self.norm(k) for k in CANDIDATE_KEYWORDS}
        for h in headers:
            if self.norm(h) in norm_kw: return h
        # NER on headers
        if self.coordinator and hasattr(self.coordinator, "extract_entities"):
            for h in headers:
                try:
                    ents = self.coordinator.extract_entities(h)
                    if any(lbl == "PERSON" for _, lbl in ents):
                        return h
                except Exception:
                    pass
        samples = data[:min(50, len(data))]
        # NER on values
        if self.coordinator and hasattr(self.coordinator, "extract_entities"):
            for h in headers:
                hits = seen = 0
                for r in samples:
                    v = r.get(h, "")
                    if not isinstance(v, str) or not v: continue
                    seen += 1
                    try:
                        ents = self.coordinator.extract_entities(v)
                        if any(lbl == "PERSON" for _, lbl in ents):
                            hits += 1
                    except Exception:
                        pass
                if seen and hits / seen >= 0.35:
                    return h
        # Pattern heuristic
        for h in headers:
            hits = cnt = 0
            for r in samples:
                v = r.get(h, "")
                if not isinstance(v, str) or not v: continue
                cnt += 1
                if _NAME_LIKE_RE.match(v.strip()): hits += 1
            if cnt and hits / cnt >= 0.35:
                return h
        # Fallback
        for h in headers:
            hl = h.lower()
            if hl in ("precinct", "district"): continue
            if self.is_percent_header(h): continue
            return h
        return None

    # ----------- Ballot Types ----------- #
    def detect_ballot_types(self, headers: List[str], data: List[Dict[str, Any]]) -> List[str]:
        bt = []
        known = {self.norm(b) for b in BALLOT_TYPES}
        for h in headers:
            if self.norm(h) in known:
                bt.append(h)
        if bt: return bt
        for h in headers:
            if h.lower() in ("precinct", "candidate", "percent reported"): continue
            vals = [r.get(h, "") for r in data]
            non = [v for v in vals if v not in ("", None)]
            if not non: continue
            numeric = sum(1 for v in non if _numeric_like(str(v)))
            if numeric / len(non) >= 0.5:
                bt.append(h)
        return bt

    # ----------- Entity Annotation ----------- #
    def annotate_entities(self, headers: List[str], data: List[Dict[str, Any]]) -> EntityAnnotation:
        ann = EntityAnnotation()
        for h in headers:
            if any(bt.lower() in h.lower() for bt in BALLOT_TYPES):
                safe_add(ann.ballot_types, h)
        # crude scanning
        for r in data:
            for h, v in r.items():
                if not v: continue
                if isinstance(v, str) and _numeric_like(v):
                    safe_add(ann.numbers, v)
                if self.is_location_header(h):
                    safe_add(ann.locations, v)
        # optional NER on candidate-like values
        if self.coordinator and hasattr(self.coordinator, "extract_entities"):
            candidates = set()
            cand_headers = [h for h in headers if self.norm(h) in {self.norm(k) for k in CANDIDATE_KEYWORDS}]
            for ch in cand_headers:
                for r in data:
                    val = r.get(ch)
                    if isinstance(val, str): candidates.add(val)
            for name in list(candidates)[:200]:
                try:
                    ents = self.coordinator.extract_entities(name)
                    if any(lbl == "PERSON" for _, lbl in ents):
                        safe_add(ann.people, name)
                except Exception:
                    pass
        return ann