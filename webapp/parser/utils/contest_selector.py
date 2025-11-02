from __future__ import annotations

import json
import math

# Contest selection and filtering utilities (refactored)
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from difflib import get_close_matches
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from ..Context_Integration.Context_Library.constants import (
    CONTEST_KEYWORDS,
    CONTEST_TITLE_KEYWORDS,
    ELECTION_TYPE_REGEX_MAP,
    ELECTION_TYPES,
    OFFICE_KEYWORDS,
)
from .logger_singleton import logger, prompt
from .shared_logic import (
    normalize_county_name,
    normalize_state_name,
    safe_capitalize,
    safe_get,
    safe_lower,
    safe_model_encode,
    safe_strip,
)
from .user_prompt import PromptCancelled

# Some deployments may not expose optional constants
try:
    from ..Context_Integration.Context_Library.constants import CONTEST_TITLE_SKIP_PHRASES
except Exception:
    CONTEST_TITLE_SKIP_PHRASES = set()

# Optional NLP normalization (safe if NLTK missing)
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    try:
        STOPWORDS = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords')
        STOPWORDS = set(stopwords.words('english'))
    STEMMER = PorterStemmer()
except ImportError:
    STEMMER = None
    STOPWORDS = set()

LOG_SCOPE = "contest_selector"

if TYPE_CHECKING:
    from ..Context_Integration.context_coordinator import ContextCoordinator
   
# ================================================
# Data model for structured output (optional JSON)
# ================================================
@dataclass
class ContestRecord:
    title: str
    year: int | None = None
    jurisdiction: str | None = None
    level: str | None = None
    type_: str | None = None
    canonical_key: str | None = None
    cluster_id: int | None = None
    source: str | None = None
    confidence: float | None = None
    session_id: str | None = None 
    
# ------------------ Normalization helpers (keep existing _norm_key / _tokens / _jaccard) ------------------

def _extract_year_tokens(title: str) -> list[int]:
    return [int(y) for y in re.findall(r"\b(19|20)\d{2}\b", title or "") if 1800 < int(y) < 2100]

def _strip_years(title: str) -> str:
    return re.sub(r"\b(19|20)\d{2}\b", "", title or "").strip()

def _base_canonical_key(title: str) -> str:
    """
    Canonical key ignoring year tokens & punctuation for duplicate collapse.
    """
    t = _strip_years(title)
    return _norm_key(t)

# --------------------------------------------------------
# Keyword / text-based expansion of potential contest titles
# --------------------------------------------------------
def _expand_contests_from_context(context: dict | None, base_titles: list[str]) -> list[str]:
    """
    Mine additional potential contest titles from raw textual artifacts in context.
    Sources (if present):
      - context['page_text']
      - context['page'] (list/iterable of strings)
      - context['raw_text']
      - context['ocr_lines']
    Uses keyword presence & minimum word length heuristics.
    """
    if not context:
        return []
    seen_norm = {_norm_key(t) for t in base_titles if isinstance(t, str)}
    out = []
    containers = []
    for k in ("page_text", "raw_text"):
        v = context.get(k)
        if isinstance(v, str):
            containers.append(v.splitlines())
    for k in ("page", "ocr_lines"):
        v = context.get(k)
        if isinstance(v, (list, tuple)):
            containers.append(v)
    lines: list[str] = []
    for c in containers:
        for ln in c:
            if isinstance(ln, str):
                lines.append(ln.strip())
    if not lines:
        return []
    office_terms = {kw for kw, _cat in OFFICE_KEYWORDS}
    core_kw = {*(k.lower() for k in CONTEST_KEYWORDS),
               *(k.lower() for k in CONTEST_TITLE_KEYWORDS),
               *(t.lower() for t in office_terms)}
    skip_phr = {s.lower() for s in (CONTEST_TITLE_SKIP_PHRASES or set())}
    for ln in lines:
        if not ln or len(ln) < 4:
            continue
        raw = ln.strip()
        low = raw.lower()
        if any(sp in low for sp in skip_phr):
            continue
        tokens = re.findall(r"[a-z0-9']+", low)
        if len(tokens) < 2:
            continue
        if not any(t in core_kw for t in tokens):
            continue
        cleaned = sanitize_title(raw)
        if not cleaned or len(cleaned.split()) < 2:
            continue
        nk = _norm_key(cleaned)
        if not nk or nk in seen_norm:
            continue
        seen_norm.add(nk)
        out.append(cleaned)
    return out

def _merge_expanded_contests(original: list[dict], extra_titles: list[str]) -> list[dict]:
    """
    Merge extra titles (list[str]) into existing contest dict list without duplicates.
    """
    if not extra_titles:
        return original
    seen_norm = set()
    for c in original:
       seen_norm.add(_norm_key(safe_get(c, "title", "")))
    for t in extra_titles:
        nk = _norm_key(t)
        if nk and nk not in seen_norm:
            seen_norm.add(nk)
            original.append({"title": t})
    return original

# ================================================
# Clustering & scoring
# ================================================
def _cluster_titles_by_base(titles: list[str], jaccard_thresh=0.82) -> list[list[str]]:
    # Similar to earlier clustering but using canonical base key
    groups: list[list[str]] = []
    for t in titles:
        base = _base_canonical_key(t)
        tk = _tokens(base)
        matched = False
        for g in groups:
            g_base = _base_canonical_key(g[0])
            if _jaccard(tk, _tokens(g_base)) >= jaccard_thresh:
                g.append(t)
                matched = True
                break
        if not matched:
            groups.append([t])
    return groups

def _pick_rep_title(cluster: list[str]) -> str:
    if not cluster:
        return ""
    # Heuristic: prefer the one with year if others lack; else shortest
    with_year = [c for c in cluster if _extract_year_tokens(c)]
    if len(with_year) == 1:
        return with_year[0]
    if with_year:
        # choose shortest of with_year
        return sorted(with_year, key=lambda x: (len(_strip_years(x)), len(x)))[0]
    return sorted(cluster, key=lambda x: (len(_strip_years(x)), len(x)))[0]

def _score_title(coordinator, title: str, meta: dict) -> float:
    if not coordinator or not hasattr(coordinator, "score_header"):
        return 0.0
    try:
        return float(coordinator.score_header(title, meta) or 0.0)
    except Exception:
        return 0.0

# ================================================
# Logging utilities (chunk to avoid truncation)
# ================================================
def _chunk_log_options(options: list[str], session_id: str | None, chunk_size=60):
    for i in range(0, len(options), chunk_size):
        logger.info({
            "level": "INFO",
            "type": "selector",
            "message": "[SELECTOR] Contest options chunk",
            "session_id": session_id,
            "range": f"{i}-{min(i+chunk_size-1, len(options)-1)}",
            "options": options[i:i+chunk_size]
        })

# --------------------------------------------------------
# Paginated render utility
# --------------------------------------------------------
def _render_paginated_contest_menu(
    candidates: list[dict],
    page: int,
    page_size: int,
    allow_multiple: bool
) -> tuple[str, int]:
    """
    Build a page of contest options; returns (menu_text, total_pages).
    """
    total = len(candidates)
    total_pages = max(1, math.ceil(total / page_size))
    page = max(1, min(page, total_pages))
    start = (page - 1) * page_size
    end = min(start + page_size, total)
    lines = [
        f"Available contests (page {page}/{total_pages}, showing {end - start} of {total}):",
        "Commands: next | prev | page <n> | /search <term> | all | indices (e.g. 0,2 5) | substring | cancel"
    ]
    for idx in range(start, end):
        c = candidates[idx]
        t = safe_get(c, "title", "")
        y = safe_get(c, "year")
        typ = safe_get(c, "type_", "")
        meta = []
        if y:
            meta.append(str(y))
        if typ:
            meta.append(typ)
        meta_str = f" ({', '.join(meta)})" if meta else ""
        lines.append(f"[{idx}] {t}{meta_str}")
    if allow_multiple:
        lines.append("Tip: enter 'all' to select every visible contest.")
    return "\n".join(lines), total_pages

# -------------------------
# Structured logging helper
# -------------------------
def _log(level: str, type_: str, message: str, session_id: Optional[str] = None, payload: Optional[dict] = None):
    entry = {
        "level": level.upper(),
        "type": type_,
        "scope": LOG_SCOPE,
        "message": message,
        "session_id": session_id
    }
    if payload is not None:
        entry["payload"] = payload
    # Route by level
    lvl = level.lower()
    if lvl == "debug":
        logger.debug(entry)
    elif lvl == "warning":
        logger.warning(entry)
    elif lvl == "error":
        logger.error(entry)
    else:
        logger.info(entry)

# -------------------------
# Utilities
# -------------------------

def _norm_key(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r'[^a-z0-9 ]+', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def _tokens(s: str) -> set[str]:
    return set(re.findall(r'[a-z0-9]+', (s or "").lower()))

def _jaccard(a: set[str], b: set[str]) -> float:
    inter = len(a & b)
    union = max(1, len(a | b))
    return inter / union

def _cluster_titles(titles: List[str], thresh: float = 0.80) -> List[list[str]]:
    toks = [(_tokens(t), t) for t in titles if t]
    clusters: List[list[Tuple[set[str], str]]] = []
    for tk, t in toks:
        placed = False
        for c in clusters:
            # compare to cluster centroid (first item)
            if _jaccard(tk, c[0][0]) >= thresh:
                c.append((tk, t))
                placed = True
                break
        if not placed:
            clusters.append([(tk, t)])
    # return just titles
    return [[t for _, t in c] for c in clusters]

def _pick_rep(titles: List[str]) -> str:
    if not titles:
        return ""
    # Prefer shortest non-empty
    titles_s = sorted([t for t in titles if isinstance(t, str)], key=lambda x: (len(x.strip()), x.lower()))
    return titles_s[0] if titles_s else ""

def _build_effective_list(contests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    titles = []
    for c in (contests or []):
        t = c.get("title") if isinstance(c, dict) else str(c)
        if not t:
            continue
        titles.append(t)
    # Deduplicate by normalized key
    seen = set()
    uniq = []
    for t in titles:
        k = _norm_key(t)
        if k not in seen:
            seen.add(k)
            uniq.append(t)
    # Cluster near-duplicates; pick representative per cluster
    clusters = _cluster_titles(uniq, 0.85)
    reps = [{"title": _pick_rep(cl)} for cl in clusters if cl]
    return reps

def is_markup_like(text: str) -> bool:
    if not isinstance(text, str):
        return False
    t = text.strip().lower()
    if not t:
        return False
    # HTML/data-uri markers and obvious non-contest artifacts
    markup_tokens = ("<img", "<div", "<span", "<html", "<svg", "data:image/", "<p", "<table")
    return any(tok in t for tok in markup_tokens)

def sanitize_title(title: Any) -> str:
    """Trim, strip boilerplate artifacts, and remove obvious markup-like content."""
    if not isinstance(title, str):
        return ""
    s = safe_strip(title)
    # Trim ridiculous base64 chunks if present
    s = re.sub(r"data:image/[a-zA-Z]+;base64,[A-Za-z0-9+/=\s]+", "[image]", s, flags=re.IGNORECASE)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    # Drop most markup at the edges
    s = re.sub(r"^<[^>]+>", "", s)
    s = re.sub(r"<[^>]+>$", "", s)
    return s

def _remove_boilerplate(text: str) -> str:
    patterns = [
        r'\s*[\r\n]*Vote for \d+\s*',
        r'\s*[\r\n]*Select \d+\s*',
        r'\s*[\r\n]*Choose \d+\s*',
        r'\s*[\r\n]*Pick \d+\s*',
        r'\s*[\r\n]*Ballot Item \d+\s*',
        r'\s*[\r\n]*Ballot Position \d+\s*',
        r'\s*[\r\n]*For Election Use Only\s*',
        r'\s*[\r\n]*Unofficial Results\s*',
        r'\s*[\r\n]*Summary\s*',
        r'\s*[\r\n]*Results by Election District\s*',
    ]
    patterns += [pat for pat, _ in ELECTION_TYPE_REGEX_MAP if any(x in pat for x in ("vote", "select", "ballot"))]
    patterns += [rf'\b{re.escape(kw)}\b' for kw, _ in OFFICE_KEYWORDS]
    for pat in patterns:
        text = re.sub(pat, '', text, flags=re.IGNORECASE)
    return text

def _remove_keywords(text: str, keywords) -> str:
    for kw in keywords:
        text = re.sub(rf'\b{re.escape(kw)}(\'s|s)?\b', '', text, flags=re.IGNORECASE)
    return text

def _stem_and_remove_stopwords(text: str) -> str:
    if not STEMMER or not STOPWORDS:
        return text
    words = re.findall(r'\w+', text, flags=re.UNICODE)
    stemmed = [STEMMER.stem(w) for w in words if safe_lower(w) not in STOPWORDS]
    return ' '.join(stemmed)

def normalize_contest(title: str, advanced: bool = False) -> str:
    if not title:
        return ""
    title = safe_strip(title)
    title = _remove_boilerplate(title)
    title = _remove_keywords(title, CONTEST_KEYWORDS)
    title = _remove_keywords(title, [kw for kw, _ in OFFICE_KEYWORDS])
    title = re.sub(r'^[\d\W]+|[\d\W]+$', '', title, flags=re.UNICODE)
    title = re.sub(r'\s+', ' ', title).strip().lower()
    if advanced:
        title = _stem_and_remove_stopwords(title)
    assert isinstance(title, str), f"normalize_contest returned non-str: {type(title)}"
    return title

def extract_year_from_title(title) -> Optional[int]:
    if not title:
        return None
    # Avoid reading “years” from markup-like strings
    if is_markup_like(title):
        return None
    years = [int(y) for y in re.findall(r"(19|20)\d{2}", title)]
    if not years:
        return None
    title_lower = safe_lower(title)
    type_positions = []
    for t in ELECTION_TYPES:
        for m in re.finditer(re.escape(t), title_lower):
            type_positions.append((m.start(), t))
    if not type_positions:
        return max(years)
    best_year = None
    min_distance = float("inf")
    for y in years:
        y_match = re.search(str(y), title)
        if not y_match:
            continue
        y_pos = y_match.start()
        for pos, _t in type_positions:
            dist = abs(y_pos - pos)
            if dist < min_distance:
                min_distance = dist
                best_year = y
    return best_year if best_year else max(years)

def infer_election_type(title, context, contest, all_contests, coordinator) -> Optional[str]:
    if not title:
        return None
    if is_markup_like(title):
        return None
    title_lower = safe_lower(title)

    # 1) Regex/keyword patterns
    for pattern, forced_type in ELECTION_TYPE_REGEX_MAP:
        if re.search(pattern, title_lower):
            if forced_type:
                return forced_type
            match = re.search(pattern, title_lower)
            if match and match.lastindex is not None:
                return safe_capitalize(match.group(1))
            elif match:
                return safe_capitalize(match.group(0))

    # 2) Fuzzy to known types
    close = get_close_matches(title_lower, [safe_lower(t or "") for t in ELECTION_TYPES], n=1, cutoff=0.8)
    if close:
        return safe_capitalize(close[0])

    # 3) NER entity
    try:
        if coordinator:
            ents = coordinator.extract_entities(title)
            for ent, label in ents:
                if label == "EVENT" and safe_lower(ent or "") in [safe_lower(et or "") for et in ELECTION_TYPES]:
                    return safe_capitalize(ent)
    except Exception:
        pass

    # 4) Context override
    if context and safe_get(context, "type_"):
        return safe_capitalize(safe_get(context, "type_") or "")

    # 5) Most common within same year/county
    year = safe_get(contest, "year")
    county = safe_get(contest, "county")
    type_counts = defaultdict(int)
    for c in all_contests:
        if safe_get(c, "year") == year and safe_lower(safe_get(c, "county") or "") == safe_lower(county or ""):
            t = safe_lower(safe_get(c, "type_") or "")
            if t:
                type_counts[t] += 1
    if type_counts:
        most_common = max(type_counts.items(), key=lambda x: x[1])[0]
        return safe_capitalize(most_common or "")

    # 6) Office keyword hints
    for kw, typ in OFFICE_KEYWORDS:
        if kw in title_lower:
            return typ

    return None

def ensure_contest(contest) -> Dict[str, Any]:
    if not isinstance(contest, dict) or not contest:
        return {"title": str(contest)}
    title = contest.get("title")
    if title and isinstance(title, str) and title.strip():
        return contest
    for alt in ("name", "contest_name", "label"):
        alt_val = contest.get(alt)
        if alt_val and isinstance(alt_val, str) and alt_val.strip():
            contest["title"] = alt_val
            return contest
    contest["title"] = str(contest)
    return contest

# -------------------------
# ML verification
# -------------------------
def ml_verify_contest(
    contest: Dict[str, Any],
    coordinator: "ContextCoordinator",
    context: dict,
    threshold: float = 0.75
) -> bool:
    """
    ML/NER contest verification with graceful offline fallback.
    """
    from ..Context_Integration.context_coordinator import ContextCoordinator
    coordinator = coordinator or ContextCoordinator()

    title = sanitize_title(safe_strip(contest.get("title", "")))
    if not title or is_markup_like(title):
        _log("debug", "selector", "Filtered markup-like or empty title", payload={"title": title})
        return False

    year = safe_strip(contest.get("year", ""))
    ctype = safe_strip(contest.get("type_", ""))

    # Year score
    year_score = 0.0
    if year and re.match(r"^(19|20)\d{2}$", str(year)):
        year_score = 1.0
    else:
        try:
            entities = coordinator.extract_entities(title)
            for ent, label in entities:
                if label == "DATE" and re.match(r"^(19|20)\d{2}$", ent):
                    year_score = 0.9
                    break
        except Exception:
            pass

    # Type score
    ctype_norm = safe_lower(ctype).replace("election", "").strip()
    type_score = 0.0
    try:
        known_types = [safe_lower(t or "") for t in coordinator.get_election_types()]
    except Exception:
        known_types = [safe_lower(t or "") for t in ELECTION_TYPES]

    if ctype:
        if any(t in ctype_norm for t in known_types):
            type_score = 1.0
        elif any(safe_lower(v) in ctype_norm for v in ELECTION_TYPES):
            type_score = 1.0
        else:
            for pattern, forced_type in ELECTION_TYPE_REGEX_MAP:
                match = re.search(pattern, ctype_norm)
                if match:
                    type_score = 0.9
                    break
            if type_score == 0.0 and ctype_norm in {"judicial", "proposition", "amendment", "state legislature", "federal legislature"}:
                type_score = 0.8
            elif type_score == 0.0 and any(x in ctype_norm for x in ["general", "primary", "presidential", "special", "runoff"]):
                type_score = 0.8

    # Semantic fallback (safe when model unavailable)
    best_sim = 0.0
    try:
        model = getattr(coordinator, "_semantic_model", None)
    except Exception:
        model = None

    if type_score == 0.0 and model is not None and hasattr(model, "encode"):
        try:
            ctype_emb = safe_model_encode(model, [ctype_norm])
            known_embs = safe_model_encode(model, known_types)
            if ctype_emb is not None and known_embs is not None and len(ctype_emb) and len(known_embs):
                for idx, t in enumerate(known_types):
                    if not (isinstance(ctype_emb[0], np.ndarray) and isinstance(known_embs[idx], np.ndarray)):
                        continue
                    sim = float(np.dot(ctype_emb[0], known_embs[idx]) / (np.linalg.norm(ctype_emb[0]) * np.linalg.norm(known_embs[idx]) + 1e-8))
                    if sim > best_sim:
                        best_sim = sim
        except Exception:
            pass
        if best_sim > 0.7:
            type_score = 0.7

    # Title/NER boost
    ner_boost = 0.0
    office_found = False
    try:
        entities = coordinator.extract_entities(title)
        for ent, label in entities:
            if label in {"ORG", "PERSON", "EVENT", "NORP", "FAC", "GPE"}:
                ner_boost += 0.15
                office_found = True
            if any(safe_lower(kw) in safe_lower(ent) for kw, _ in OFFICE_KEYWORDS):
                ner_boost += 0.2
                office_found = True
    except Exception:
        pass

    title_kw_hit = any(safe_lower(kw or "") in safe_lower(title or "") for kw in CONTEST_KEYWORDS)
    title_score = 1.0 if (title_kw_hit or office_found) else 0.0

    # Coordinator score (safe)
    try:
        ml_score = coordinator.score_header(title, context)
        if isinstance(ml_score, str):
            ml_score = 0.0
    except Exception:
        ml_score = 0.0
    # Fuzzy boost (safe)
    fuzzy_boost = 0.0
    try:
        # Guard: only call fuzzy_score when both strings are non-empty
        if hasattr(coordinator, "fuzzy_score") and title and ctype:
            fz = coordinator.fuzzy_score(title, ctype)
            if isinstance(fz, (float, int)):
                fuzzy_boost = float(fz) * 0.1
    except Exception:
        pass

    # Context boost
    context_boost = 0.0
    if isinstance(context, dict):
        ctx_year = safe_get(context, "year")
        ctx_type = safe_lower(safe_get(context, "type_", ""))
        if ctx_year and str(ctx_year) == str(year):
            context_boost += 0.1
        if ctx_type and ctx_type in ctype_norm:
            context_boost += 0.1

    score = 0.35 * year_score + 0.25 * type_score + 0.2 * title_score + 0.1 * ml_score + fuzzy_boost + ner_boost + context_boost

    # Allow slightly lower threshold when no semantic model is available
    dynamic_threshold = threshold
    if model is None:
        dynamic_threshold = min(threshold, 0.65)

    if year_score == 1.0 and title_score == 1.0 and score >= 0.55:
        return True
    return score >= dynamic_threshold

def feedback_loop_verify_contests(
    contests: List[Dict[str, Any]],
    coordinator: "ContextCoordinator",
    context: dict,
    max_loops: int = 2,
    threshold: float = 0.8,
    session_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Light feedback loop with graceful fallback and structured logs.
    """
    from ..Context_Integration.context_coordinator import ContextCoordinator
    coordinator = coordinator or ContextCoordinator()
    verified = []
    for loop in range(max_loops):
        _log("warning", "selector", f"Feedback loop {loop+1}: verifying contests", session_id=session_id,
             payload={"candidates": len(contests), "threshold": threshold})
        verified = [c for c in contests if ml_verify_contest(c, coordinator, context, threshold=threshold)]
        if verified:
            _log("info", "selector", f"Feedback loop {loop+1}: verified={len(verified)}", session_id=session_id)
            break
        # Lower threshold a bit each loop
        threshold = max(0.6, threshold - 0.1)

    if verified:
        return verified

    # Fallback: pick strong titles (lengthy) or decent coordinator score
    fallbacks = []
    for c in contests:
        title = safe_get(c, "title", "")
        if isinstance(title, str) and len(title) > 12 and not is_markup_like(title):
            fallbacks.append(c)
            continue
        try:
            score = coordinator.score_header(title, context)
            if isinstance(score, (float, int)) and score > 0.6:
                fallbacks.append(c)
        except Exception:
            pass

    if fallbacks:
        _log("info", "selector", "Fallback: selecting by title/semantic score", session_id=session_id,
             payload={"selected": len(fallbacks)})
        return fallbacks

    return []

def resolve_selection_context(
    coordinator=None,
    context: dict | None = None,
    fallback_filename: str | None = None,
    allow_filename_infer: bool = True
) -> tuple[str | None, str | None, int | None]:
    """
    Infer (state, county, year) from (in order):
      - explicit context fields
      - first enriched contest in coordinator
      - dynamic_state_county_detection (if available & raw html in context)
      - filename tokens (STATE / <name>county / 4-digit year)
    """
    ctx = context or {}
    state = normalize_state_name(safe_get(ctx, "state")) or None
    county = normalize_county_name(safe_get(ctx, "county")) or None
    year = safe_get(ctx, "year")
    if isinstance(year, str) and year.isdigit():
        year = int(year)
    if not isinstance(year, int):
        year = None

    # From coordinator contests
    try:
        if (not state or not county or not year) and coordinator and hasattr(coordinator, "get_contests"):
            contests = coordinator.get_contests()
            if contests:
                c0 = contests[0]
                state = state or normalize_state_name(safe_get(c0, "state"))
                county = county or normalize_county_name(safe_get(c0, "county"))
                y = safe_get(c0, "year")
                if not year and isinstance(y, int):
                    year = y
    except Exception:
        pass

    # Filename inference (format handlers)
    if allow_filename_infer and fallback_filename:
        base = fallback_filename.lower()
        parts = base.replace(".json", "").replace(".csv", "").replace(".pdf", "").split("_")
        for p in parts:
            if not state and len(p) == 2 and p.isalpha():
                state = p.upper()
            if "county" in p and not county:
                county = (p.replace("county", "").strip() + " County").title()
            if not year:
                m = re.search(r"(19|20)\d{2}", p)
                if m:
                    try:
                        year = int(m.group(0))
                    except Exception:
                        pass

    return state, county, year

def select_contest_auto_first(
    *,
    coordinator=None,
    context: dict | None = None,
    allow_multiple: bool = False,
    session_id: str | None = None,
    force_interactive: bool = False,
    auto_confidence_threshold: float = 0.93,
    page_size: int = 30,
    prefer_year_match: bool = True,
    return_mode: str = "objects"
) -> Optional[List[Dict[str, Any]]]:
    """
    Wrapper:
      1. Resolve (state, county, year)
      2. Attempt non-interactive selection
      3. If result empty OR >1 and interactive needed, fallback to interactive select_contest
      4. Always returns list[dict] (or None if user cancels)
    """
    ctx = context or {}
    filename = safe_get(ctx, "input_file") or safe_get(ctx, "source_file") or safe_get(ctx, "source") or None
    state, county, year = resolve_selection_context(
        coordinator=coordinator,
        context=ctx,
        fallback_filename=filename
    )

    # Non-interactive attempt
    auto = select_contest_noninteractive(
        coordinator=coordinator,
        context=ctx,
        state=state,
        county=county,
        year=year,
        session_id=session_id,
        prefer_year_match=prefer_year_match,
        return_mode="objects"
    )
    auto_list = auto if isinstance(auto, list) else []
    # If we got exactly one or user forbids interactive, return it
    if not force_interactive and auto_list:
        if len(auto_list) == 1:
            return auto_list
        # If top has very high confidence, accept
        try:
            top_conf = float(safe_get(auto_list[0], "confidence") or 0.0)
        except Exception:
            top_conf = 0.0
        if top_conf >= auto_confidence_threshold and not allow_multiple:
            return [auto_list[0]]

    # Fallback to interactive
    return select_contest(
        coordinator=coordinator,
        state=state,
        county=county,
        year=year,
        session_id=session_id,
        context=ctx,
        allow_multiple=allow_multiple,
        force_interactive=force_interactive,
        page_size=page_size,
        auto_when_confident=True,
        auto_confidence_threshold=auto_confidence_threshold,
        return_mode="objects"
    )

# ================================================
# Non-interactive selection (auto strategy)
# ================================================
def select_contest_noninteractive(
    *,
    coordinator=None,
    context: dict | None = None,
    state: str | None = None,
    county: str | None = None,
    year: int | None = None,
    session_id: str | None = None,
    prefer_year_match: bool = True,
    return_mode: str = "objects"  # 'objects' | 'json' | 'titles'
) -> list[dict] | str | list[str]:
    """
    Attempt automatic contest selection WITHOUT user interaction.
    - Expands & clusters contests
    - Scores with coordinator if available
    - Picks top cluster representative
    - Optionally filters by explicit year if provided
    """
    context = context or {}
    selector_data = context.get("selector_data") or {}
    base_contests = selector_data.get("contests") or []
    base_titles = [safe_get(c, "title", "") for c in base_contests]
    extra = _expand_contests_from_context(context, base_titles)
    if extra:
        base_contests = _merge_expanded_contests(base_contests, extra)
    titles = [safe_get(c, "title", "") for c in base_contests if safe_get(c, "title")]
    titles = list(dict.fromkeys(titles))  # preserve order unique

    if not titles:
        return [] if return_mode != "json" else "[]"

    clusters = _cluster_titles_by_base(titles)
    reps = []
    for idx, cl in enumerate(clusters):
        rep = _pick_rep_title(cl)
        yrs = _extract_year_tokens(rep)
        rep_year = yrs[0] if yrs else None
        score = _score_title(coordinator, rep, {"state": state, "county": county, "year": year})
        reps.append(ContestRecord(
            title=rep,
            year=rep_year,
            jurisdiction=county,
            level=None,
            type_=None,
            canonical_key=_base_canonical_key(rep),
            cluster_id=idx,
            source="auto",
            confidence=score,
            session_id=session_id
        ))

    # Optional year preference
    if prefer_year_match and year:
        year_matches = [r for r in reps if r.year == year]
        if year_matches:
            reps = year_matches

    # Sort by confidence desc then shorter stripped title
    reps.sort(key=lambda r: (-float(r.confidence or 0.0), len(_strip_years(r.title)), r.title.lower()))

    if return_mode == "json":
        return json.dumps([asdict(r) for r in reps], ensure_ascii=False)
    if return_mode == "titles":
        return [r.title for r in reps]
    return [asdict(r) for r in reps]

# -------------------------
# Core selection
# -------------------------
def select_contest(
    coordinator=None,
    state: str | None = None,
    county: str | None = None,
    year: int | None = None,
    session_id: str | None = None,
    context: dict | None = None,
    allow_multiple: bool = False,
    prompt_message: str = "[PROMPT] Select contest (index(es), /search <term>, next, prev, page <n>, 'all', or 'cancel'): ",
    force_interactive: bool = False,
    disable_ml_verify: bool = False,
    page_size: int = 30,
    max_search_results: int = 400,
    return_mode: str = "objects",
    noninteractive_if_single: bool = True,
    auto_when_confident: bool = True,
    auto_confidence_threshold: float = 0.93
) -> Optional[List[Dict[str, Any]]]:
    """
    Adaptive contest selector with webapp prompt integration.
    Uses prompt.prompt_input (non-blocking for web frontend) instead of raw input().
    Returns list[dict] or None if user cancels.
    """
    context = context or {}
    selector_data = context.get("selector_data") or {}
    base_contests = selector_data.get("contests") or []
    base_titles = [safe_get(c, "title", "") for c in base_contests]
    expanded = _expand_contests_from_context(context, base_titles)
    if expanded:
        base_contests = _merge_expanded_contests(base_contests, expanded)

    raw_titles = [safe_get(c, "title", "") for c in base_contests if safe_get(c, "title")]
    raw_titles = list(dict.fromkeys(raw_titles))
    if not raw_titles:
        return [] if return_mode != "json" else "[]"

    clusters = _cluster_titles_by_base(raw_titles)
    candidates: list[ContestRecord] = []
    for idx, cl in enumerate(clusters):
        rep = _pick_rep_title(cl)
        yrs = _extract_year_tokens(rep)
        rep_year = yrs[0] if yrs else None
        score = _score_title(coordinator, rep, {"state": state, "county": county, "year": year})
        candidates.append(ContestRecord(
            title=rep,
            year=rep_year,
            jurisdiction=county,
            canonical_key=_base_canonical_key(rep),
            cluster_id=idx,
            source="cluster_rep",
            confidence=score,
            session_id=session_id
        ))

    if year:
        year_pref = [c for c in candidates if c.year == year]
        if year_pref:
            candidates = year_pref

    # Detect webapp mode and disable pagination (show all in one page)
    is_webapp = bool((context or {}).get("webapp") or (context or {}).get("web_use_full_menu")) \
                or str(getattr(prompt, "mode", "")).lower() == "webapp"
    if is_webapp:
        page_size = max(page_size, len(candidates) or 0)
        # Also simplify the prompt text for webapp (no paging commands)
        prompt_message = "[PROMPT] Select contest index(es) or 'cancel': "

    if noninteractive_if_single and len(candidates) == 1 and not force_interactive:
        result = [asdict(candidates[0])]
        if return_mode == "json":
            return json.dumps(result, ensure_ascii=False)
        if return_mode == "titles":
            return [result[0]["title"]]
        return result

    if auto_when_confident and not force_interactive:
        top_sorted = sorted(candidates, key=lambda r: -float(r.confidence or 0.0))
        if top_sorted and (top_sorted[0].confidence or 0.0) >= auto_confidence_threshold:
            result = [asdict(top_sorted[0])]
            if return_mode == "json":
                return json.dumps(result, ensure_ascii=False)
            if return_mode == "titles":
                return [result[0]["title"]]
            return result

    # Interactive path
    candidates = sorted(
        candidates,
        key=lambda r: (-float(r.confidence or 0.0), len(_strip_years(r.title)), r.title.lower())
    )
    titles = [c.title for c in candidates]
    structured_options = []
    for idx, c in enumerate(candidates):
        meta_parts = []
        if c.year:
            meta_parts.append(str(c.year))
        if c.confidence is not None:
            meta_parts.append(f"conf={c.confidence:.2f}")
        structured_options.append({
            "index": idx,
            "label": c.title,
            "meta": ", ".join(meta_parts) if meta_parts else ""
        })

    logger.info({
        "level": "INFO",
        "type": "contest_options",
        "message": f"Emitting {len(structured_options)} contest options",
        "session_id": session_id,
        "options": structured_options,
        "total_count": len(structured_options),
        "context": {
            "state": state,
            "county": county,
            "year": year,
            "source": safe_get(context, "source") or safe_get(context, "input_file"),
            "handler": safe_get(context, "handler"),
            "input_file": safe_get(context, "input_file")
        }
    })

    legacy_lines = ["Available contests:"]
    for o in structured_options:
        legacy_lines.append(f"[{o['index']}] {o['label']}" + (f" ({o['meta']})" if o['meta'] else ""))
    logger.info({
        "level": "INFO",
        "type": "selector",
        "message": "\n".join(legacy_lines),
        "session_id": session_id
    })
    page = 1

    def build_page_options(pg: int) -> tuple[list[str], int]:
        total = len(candidates)
        total_pages = max(1, math.ceil(total / page_size))
        pg = max(1, min(pg, total_pages))
        start = (pg - 1) * page_size
        end = min(start + page_size, total)
        opts = []
        for idx in range(start, end):
            c = candidates[idx]
            meta = []
            if c.year:
                meta.append(str(c.year))
            if c.confidence is not None:
                meta.append(f"conf={c.confidence:.2f}")
            opts.append(f"[{idx}] {c.title}" + (f" ({', '.join(meta)})" if meta else ""))
        return opts, total_pages

    selected: list[ContestRecord] = []
    prompted_once = False

    while True:
        page_options, total_pages = build_page_options(page)

        # Suppress confusing page logs in webapp mode
        if not is_webapp:
            logger.info({
                "level": "INFO",
                "type": "selector_menu",
                "message": f"Contests page {page}/{total_pages} (total={len(candidates)})",
                "session_id": session_id
            })

        prompt_ctx = {
            "kind": "contest",
            "page": page,
            "total_pages": total_pages,
            "count": len(candidates)
        }
        # Attach options once; with webapp mode page_size == total, this is the full list
        if not prompted_once:
            prompt_ctx["options"] = page_options

        try:
            user_in = prompt.prompt_input(
                prompt_message,
                session_id=session_id,
                context=prompt_ctx,
                allow_cancel=True
            ).strip()
            prompted_once = True
        except PromptCancelled:
            return None
        except Exception:
            user_in = ""

        if not user_in:
            selected = [candidates[0]]
            break

        lowered_input = user_in.lower()

        if lowered_input in {"cancel", "quit", "q", "exit"}:
            return None

        if allow_multiple and lowered_input in {"all", "*"}:
            selected = candidates
            break

        # Keep paging commands for CLI; they are harmless in webapp (single-page)
        if lowered_input in {"next", "n"}:
            page = page + 1 if page < total_pages else 1
            prompted_once = False
            continue

        if lowered_input in {"prev", "p", "previous"}:
            page = page - 1 if page > 1 else total_pages
            prompted_once = False
            continue

        if lowered_input.startswith("page "):
            try:
                req = int(lowered_input.split()[1])
                if 1 <= req <= total_pages:
                    page = req
                    prompted_once = False
            except Exception:
                pass
            continue

        if lowered_input.startswith("/search"):
            term = user_in.split(" ", 1)[1].strip() if " " in user_in else ""
            if not term:
                logger.warning({"level": "WARNING", "type": "selector", "message": "Empty search term", "session_id": session_id})
                continue
            term_l = term.lower()
            idxs = [i for i, t in enumerate(titles) if term_l in t.lower()]
            if not idxs:
                logger.warning({"level": "WARNING", "type": "selector", "message": f"No matches for '{term}'", "session_id": session_id})
                continue
            candidates = [candidates[i] for i in idxs[:max_search_results]]
            titles = [c.title for c in candidates]
            page = 1
            prompted_once = False
            # In webapp, also re-emit the full updated options so modal refreshes naturally
            if is_webapp:
                new_opts = []
                for i, c in enumerate(candidates):
                    meta_parts = []
                    if c.year:
                        meta_parts.append(str(c.year))
                    if c.confidence is not None:
                        meta_parts.append(f"conf={c.confidence:.2f}")
                    new_opts.append({"index": i, "label": c.title, "meta": ", ".join(meta_parts) if meta_parts else ""})
                logger.info({
                    "level": "INFO",
                    "type": "contest_options",
                    "message": f"Emitting {len(new_opts)} contest options",
                    "session_id": session_id,
                    "options": new_opts,
                    "total_count": len(new_opts),
                    "context": {
                        "state": state, "county": county, "year": year,
                        "source": safe_get(context, "source") or safe_get(context, "input_file"),
                        "handler": safe_get(context, "handler"),
                        "input_file": safe_get(context, "input_file")
                    }
                })
            continue

        parts = [p for p in re.split(r"[,\s]+", user_in) if p]
        if parts and all(p.isdigit() for p in parts):
            idxs = []
            for p in parts:
                i = int(p)
                if 0 <= i < len(candidates):
                    idxs.append(i)
            if not idxs:
                continue
            if not allow_multiple:
                idxs = [idxs[0]]
            selected = [candidates[i] for i in sorted(set(idxs))]
            break

        substr = [i for i, t in enumerate(titles) if user_in.lower() in t.lower()]
        if substr:
            if not allow_multiple:
                substr = [substr[0]]
            selected = [candidates[i] for i in substr]
            break

        import difflib
        fuzz = difflib.get_close_matches(user_in, titles, n=(5 if allow_multiple else 1), cutoff=0.45)
        if fuzz:
            idxs = [titles.index(f) for f in fuzz]
            if not allow_multiple:
                idxs = [idxs[0]]
            selected = [candidates[i] for i in idxs]
            break

        logger.warning({"level": "WARNING", "type": "selector", "message": "No match; try again.", "session_id": session_id})

    result_objs = [asdict(r) for r in selected]
    if return_mode == "json":
        return json.dumps(result_objs, ensure_ascii=False)
    if return_mode == "titles":
        return [r["title"] for r in result_objs]
    return result_objs