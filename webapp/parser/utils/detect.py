"""
detect.py
Detection heuristics, NER-assisted column detection, entity/dataclass models,
percent/location detection, harmonization, metrics, shared regex & numeric parsing.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import re, unicodedata, difflib, time
from functools import lru_cache

from .shared_logic import (
    safe_get, safe_lower, safe_strip, safe_translate, safe_values,
    safe_items, safe_append, safe_add, safe_keys
)
from .logger_singleton import logger
from ..Context_Integration.Context_Library.constants import (
    LOCATION_KEYWORDS, LOCATION_ABBREVIATIONS, PERCENT_KEYWORDS,
    CANDIDATE_KEYWORDS, BALLOT_TYPES, BALLOT_TYPES_SORT_ORDER,
    TOTAL_KEYWORDS, MISC_FOOTER_KEYWORDS, PARTY_KEYWORDS
)

# ---------- Central Regex / Patterns ----------
PERCENT_REPORTED_RE = re.compile(r"(\d{1,3})\s*%[\s\-]*reported", re.I)
NUMBER_LIKE_RE = re.compile(r"^-?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?$")
NAME_LIKE_RE = re.compile(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}$")

# ---------- Metrics ----------
def emit_metric(name: str, **labels):
    try:
        logger.info({"metric": name, "labels": labels})
    except Exception:
        pass

# ---------- Dataclasses ----------
@dataclass
class EntityInfo:
    people: set = field(default_factory=set)
    locations: set = field(default_factory=set)
    ballot_types: set = field(default_factory=set)
    numbers: set = field(default_factory=set)
    row_entities: list = field(default_factory=list)
    ml_confidence: float | None = None
    association_log: Any | None = None
    segments: Any | None = None
    panels: Any | None = None
    location_column: str | None = None
    percent_column: str | None = None

@dataclass
class StructureInfo:
    location_header: str | None = None
    percent_header: str | None = None
    candidate_headers: List[str] = field(default_factory=list)
    ballot_types_headers: List[str] = field(default_factory=list)
    total_header: str | None = None
    verified: bool = False

# ---------- Normalization ----------
@lru_cache(maxsize=100_000)
def _norm(s: str) -> str:
    s = s.strip().lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii")
    s = re.sub(r"\s+"," ", s)
    return s

def normalize_text(s: str) -> str:
    return _norm(str(s))

def normalize_header(h: str) -> str:
    return _norm(str(h))

def normalize_for_matching(text) -> str:
    t = safe_lower(safe_strip(text))
    table = str.maketrans('', '', r"""!"#$%&'()*+,./:;<=>?@[\]^_`{|}~""")
    return safe_translate(t, table)

# ---------- Percent / Heading ----------
def extract_percent_reported_from_heading(heading: str) -> str:
    if not heading: return ""
    m = PERCENT_REPORTED_RE.search(heading)
    if m: return f"{int(m.group(1))}%"
    if re.search(r"\bfully\s+reported\b", heading, re.I): return "100%"
    return ""

# ---------- Location / Percent Detection ----------
def _is_percent_header(h: str) -> bool:
    nh = normalize_header(h or "")
    if nh in {normalize_header(p) for p in PERCENT_KEYWORDS}: return True
    return "%" in (h or "").lower() or "reported" in (h or "").lower()

def _should_exclude_as_location(h: str) -> bool:
    return normalize_text(h) in {"contest","race","rawjson","raw json","json","data"}

def _is_bad_location_fallback(h: str) -> bool:
    if _is_percent_header(h): return True
    s = (h or "").strip().lower()
    return bool(re.fullmatch(r"(col(umn)?\s*\d+|\d+)", s))

def is_location_header(header) -> bool:
    hn = normalize_for_matching(header)
    for kw in LOCATION_KEYWORDS:
        if kw in hn:
            return True
    if hn in LOCATION_ABBREVIATIONS:
        return True
    return False

def dynamic_detect_location_header(headers: List[str], coordinator=None) -> Tuple[str,str,str]:
    try:
        lib = getattr(coordinator, "library", {})
        loc_patterns = set(safe_get(lib, "location_patterns", [])) or set(LOCATION_KEYWORDS)
        pct_patterns = set(safe_get(lib, "percent_patterns", [])) or set(PERCENT_KEYWORDS)
    except Exception:
        loc_patterns, pct_patterns = set(LOCATION_KEYWORDS), set(PERCENT_KEYWORDS)
    norm_headers=[normalize_text(h) for h in headers or []]
    location_header = percent_header = location_entity = None
    # percent
    for i,h in enumerate(norm_headers):
        if any(normalize_text(p)==h for p in pct_patterns) or _is_percent_header(headers[i]):
            percent_header=headers[i]; break
    if not percent_header:
        for i,h in enumerate(norm_headers):
            if any(normalize_text(p) in h for p in pct_patterns) or _is_percent_header(headers[i]):
                percent_header=headers[i]; break
    # location
    for i,h in enumerate(norm_headers):
        if any(normalize_text(p)==h for p in loc_patterns) and not _is_percent_header(headers[i]) and not _should_exclude_as_location(headers[i]):
            location_header=headers[i]; break
    if not location_header:
        for i,h in enumerate(norm_headers):
            if any(normalize_text(p) in h for p in loc_patterns) and not _is_percent_header(headers[i]) and not _should_exclude_as_location(headers[i]):
                location_header=headers[i]; break
    if location_header and percent_header and normalize_header(location_header)==normalize_header(percent_header):
        location_header=None
    return location_header, percent_header, location_entity

# ---------- Candidate Column Detection ----------
def detect_candidate_column(headers: List[str], data: List[Dict[str,Any]], coordinator=None) -> Optional[str]:
    if not headers: return None
    norm_kw={normalize_header(k) for k in CANDIDATE_KEYWORDS}
    for h in headers:
        if normalize_header(h) in norm_kw: return h
    # NER on headers
    if coordinator and hasattr(coordinator,"extract_entities"):
        for h in headers:
            try:
                ents = coordinator.extract_entities(h)
                if any(lbl=="PERSON" for _,lbl in ents):
                    return h
            except Exception:
                pass
    # sample value NER
    samples=data[:min(40,len(data))]
    for h in headers:
        hits=seen=0
        for r in samples:
            v=safe_get(r,h,"")
            if not isinstance(v,str) or not v: continue
            seen+=1
            if coordinator and hasattr(coordinator,"extract_entities"):
                try:
                    ents=coordinator.extract_entities(v)
                    if any(lbl=="PERSON" for _,lbl in ents): hits+=1
                except Exception:
                    pass
        if seen and hits/seen>=0.35: return h
    # simple pattern
    for h in headers:
        hits=cnt=0
        for r in samples:
            v=safe_get(r,h,"")
            if not isinstance(v,str) or not v: continue
            cnt+=1
            if NAME_LIKE_RE.match(v.strip()): hits+=1
        if cnt and hits/cnt>=0.35:
            return h
    return None

# ---------- Entity Annotation (Light) ----------
def nlp_entity_annotate_table(headers, data, context=None, coordinator=None):
    info=EntityInfo()
    if not coordinator:
        return headers, data, info.__dict__
    for h in headers:
        try:
            ents = coordinator.extract_entities(h)
            for ent,label in ents:
                if label=="PERSON": safe_add(info.people, ent)
                elif label in {"GPE","LOC","FAC"}: safe_add(info.locations, ent)
        except Exception:
            pass
    for row in data:
        row_ents={"people":set(),"locations":set(),"ballot_types":set(),"numbers":set()}
        for h,v in safe_items(row):
            if not v: continue
            if isinstance(v,str) and NUMBER_LIKE_RE.match(v.replace(",","")):
                safe_add(info.numbers, v)
                safe_add(row_ents["numbers"], v)
            if any(bt.lower() in h.lower() for bt in BALLOT_TYPES):
                safe_add(info.ballot_types, h)
                safe_add(row_ents["ballot_types"], h)
            if coordinator and isinstance(v,str):
                try:
                    ents=coordinator.extract_entities(v)
                    for ent,label in ents:
                        if label=="PERSON": safe_add(info.people, ent); safe_add(row_ents["people"], ent)
                        elif label in {"GPE","LOC","FAC"}: safe_add(info.locations, ent); safe_add(row_ents["locations"], ent)
                except Exception:
                    pass
        safe_append(info.row_entities, row_ents)
    return headers, data, info.__dict__

# ---------- Harmonization ----------
def harmonize_headers_and_data(headers: List[str], data: List[Dict[str,Any]], context: dict | None = None):
    headers=headers or []; data=data or []
    all_headers=set(h for h in headers if h)
    for r in data: all_headers.update(safe_keys(r))
    percent_val=None
    if any("Percent Reported" in safe_keys(r) for r in data):
        all_headers.add("Percent Reported")
        for r in data:
            percent_val=percent_val or safe_get(r,"Percent Reported","") or None
    if context and safe_get(context,"percent_reported"):
        all_headers.add("Percent Reported")
        percent_val=safe_get(context,"percent_reported",percent_val)
    seen=set()
    ordered=[h for h in headers if h in all_headers and not (h in seen or seen.add(h))]
    for h in all_headers:
        if h not in seen: ordered.append(h); seen.add(h)
    # normalize location header
    loc_col = next((h for h in ordered if is_location_header(h)), None)
    if loc_col and loc_col!="Precinct":
        ordered=["Precinct" if h==loc_col else h for h in ordered]
        for r in data:
            r["Precinct"]=r.pop(loc_col, r.get("Precinct",""))
        loc_col="Precinct"
    cand_col = next((h for h in ordered if any(k in h.lower() for k in CANDIDATE_KEYWORDS)), None)
    ballot_cols=[h for h in ordered if any(bt in h.lower() for bt in BALLOT_TYPES)]
    harmonized=[]
    dedup=set()
    for r in data:
        full={h: safe_get(r,h,"") for h in ordered}
        if "Percent Reported" in ordered and not full.get("Percent Reported") and percent_val:
            full["Percent Reported"]=percent_val
        if loc_col and cand_col and full.get(loc_col) and full.get(cand_col):
            key=(full.get(loc_col), full.get(cand_col))+tuple(full.get(b,"") for b in ballot_cols)
            if key in dedup: continue
            dedup.add(key)
        harmonized.append(full)
    keep=[h for h in ordered if (h in headers) or any(r.get(h) not in ("",None) for r in harmonized)]
    if not keep: keep=ordered
    cand_cols=[h for h in keep if any(k in h.lower() for k in CANDIDATE_KEYWORDS)]
    bt_cols=[h for h in keep if any(bt in h.lower() for bt in BALLOT_TYPES)]
    final=[]
    if "Precinct" in keep: final.append("Precinct")
    final+=sorted(set(cand_cols+bt_cols))
    final+=[h for h in keep if h not in set(["Precinct"]+cand_cols+bt_cols)]
    # uniq
    s=set(); final=[h for h in final if not (h in s or s.add(h))]
    return final, [{h: r.get(h,"") for h in final} for r in harmonized]

# ---------- Header Utilities ----------
def find_best_header(headers, keywords):
    hl=[safe_lower(h) for h in headers]
    for kw in keywords:
        kwl=safe_lower(kw)
        for i,h in enumerate(hl):
            if kwl in h: return headers[i]
    for kw in keywords:
        kwl=safe_lower(kw)
        matches=difflib.get_close_matches(kwl, hl, n=1, cutoff=0.7)
        if matches:
            return headers[hl.index(matches[0])]
    return None

def is_likely_header(row_cells: List[str]) -> bool:
    known=set([*(k.lower() for k in CANDIDATE_KEYWORDS),
               *(k.lower() for k in LOCATION_KEYWORDS),
               *(k.lower() for k in PERCENT_KEYWORDS),
               *(k.lower() for k in TOTAL_KEYWORDS),
               "votes","percent","district","party","candidate"])
    return sum(1 for c in row_cells if any(k in c.lower() for k in known))>=2

# ---------- Numeric Parsing ----------
def parse_numeric(val: Any) -> Tuple[Optional[int], bool]:
    if val is None: return None, False
    s=str(val).strip()
    pct=s.endswith("%")
    s=s.replace("%","").replace(",","")
    if s.replace(".","",1).isdigit():
        try:
            return int(float(s)), pct
        except Exception:
            return None, pct
    return None, pct

# ---------- Table Data (simple) ----------
def extract_table_data(table, coordinator=None, structure_info=None):
    from .browser_utils import safe_locator, safe_nth, safe_count, safe_inner_text
    headers=[]
    rows=[]
    head_cells = safe_locator(table, "thead tr th", logger)
    if safe_count(head_cells,logger)==0:
        first_row = safe_nth(safe_locator(table,"tr", logger),0,logger)
        head_cells = safe_locator(first_row,"th,td", logger) if first_row else []
    for i in range(safe_count(head_cells, logger)):
        txt = safe_inner_text(safe_nth(head_cells, i, logger), logger).strip()
        headers.append(txt or f"Column {i+1}")
    body_rows = safe_locator(table, "tbody tr", logger)
    if safe_count(body_rows, logger)==0:
        body_rows = safe_locator(table,"tr",logger)
    for i in range(safe_count(body_rows, logger)):
        row_locator = safe_nth(body_rows,i,logger)
        cells = safe_locator(row_locator,"td,th",logger)
        if safe_count(cells, logger)==0: continue
        r={}
        for j in range(safe_count(cells, logger)):
            if j < len(headers):
                r[headers[j]] = safe_inner_text(safe_nth(cells,j,logger), logger).strip()
        if any(v for v in r.values()):
            rows.append(r)
    return headers, rows, {}
