from __future__ import annotations
# Contest selection and filtering utilities (refactored)
import re
from collections import defaultdict
from difflib import get_close_matches
from typing import TYPE_CHECKING, List, Dict, Any, Optional

import numpy as np

from .logger_singleton import logger, prompt
from .user_prompt import PromptCancelled
from .shared_logic import (
    normalize_state_name, normalize_county_name, _sync_type_and_election_types,
    safe_get, safe_items, safe_lower, safe_split, safe_capitalize, safe_strip,
    safe_model_encode
)
from ..Context_Integration.Context_Library.constants import (
    ELECTION_TYPES, CONTEST_KEYWORDS, KNOWN_COUNTY_TO_PRECINCTS_MAP,
    ELECTION_TYPE_REGEX_MAP, OFFICE_KEYWORDS
)

# Some deployments may not expose optional constants
try:
    from ..Context_Integration.Context_Library.constants import CONTEST_TITLE_SKIP_PHRASES
except Exception:
    CONTEST_TITLE_SKIP_PHRASES = set()

# Optional NLP normalization (safe if NLTK missing)
try:
    from nltk.stem import PorterStemmer
    from nltk.corpus import stopwords
    import nltk
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
    detected_type = None
    try:
        known_types = [safe_lower(t or "") for t in coordinator.get_election_types()]
    except Exception:
        known_types = [safe_lower(t or "") for t in ELECTION_TYPES]

    if ctype:
        if any(t in ctype_norm for t in known_types):
            type_score = 1.0
            detected_type = ctype
        elif any(safe_lower(v) in ctype_norm for v in ELECTION_TYPES):
            type_score = 1.0
            detected_type = ctype
        else:
            for pattern, forced_type in ELECTION_TYPE_REGEX_MAP:
                match = re.search(pattern, ctype_norm)
                if match:
                    type_score = 0.9
                    detected_type = forced_type if forced_type else match.group(0)
                    break
            if type_score == 0.0 and ctype_norm in {"judicial", "proposition", "amendment", "state legislature", "federal legislature"}:
                type_score = 0.8
                detected_type = ctype_norm
            elif type_score == 0.0 and any(x in ctype_norm for x in ["general", "primary", "presidential", "special", "runoff"]):
                type_score = 0.8
                detected_type = ctype_norm

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
                        detected_type = t
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

# -------------------------
# Core selection
# -------------------------
def select_contest(
    coordinator: "ContextCoordinator",
    state=None,
    county=None,
    year=None,
    session_id=None,
    context=None,
    prompt_message="[PROMPT] Select contest (index, comma-separated indices, text, or 'cancel'): ",
    allow_multiple=True,
    log_func=None,
    *,
    force_interactive: bool = False,
    disable_ml_verify: bool = False,
) -> Optional[List[Dict[str, Any]]]:
    """
    Centralized contest selection with:
    - Handler-injected selector_data support
    - Soft/strict filtering
    - Optional ML/NER verification (skipped offline or when disabled)
    - Always-offer menu when force_interactive=True and >1 candidates
    - Returns a list of dicts: [{"title": "..."}] or [] if canceled/no selection
    """
    from ..Context_Integration.context_coordinator import ContextCoordinator
    coordinator = coordinator or ContextCoordinator()

    # Gather selector data
    selector_data = (context or {}).get("selector_data") if isinstance(context, dict) else None
    if not selector_data:
        selector_data = coordinator.get_for_selector()

    norm_state = normalize_state_name(state)
    norm_county = normalize_county_name(county)
    contests = safe_get(selector_data, "contests", []) or []
    noisy_patterns = set(safe_get(selector_data, "noisy_patterns", []) or [])

    # Sanitize/normalize titles and drop obvious noise
    cleaned_contests = []
    for c in contests:
        c = ensure_contest(c)
        t = sanitize_title(safe_get(c, "title", ""))
        if not t or is_markup_like(t):
            continue
        low = t.lower()
        if any(skip in low for skip in noisy_patterns):
            continue
        if CONTEST_TITLE_SKIP_PHRASES and any(s in low for s in {s.lower() for s in CONTEST_TITLE_SKIP_PHRASES}):
            continue
        c["title"] = t
        cleaned_contests.append(c)

    # Deduplicate by normalized title
    seen = set()
    deduped = []
    for c in cleaned_contests:
        key = (normalize_contest(safe_get(c, "title", "")), str(safe_get(c, "year", "")), safe_get(c, "type_", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)

    # Enrich missing metadata to avoid empty-input warnings during verification
    for c in deduped:
        try:
            title = safe_get(c, "title", "")
            if title:
                if not safe_get(c, "year"):
                    c["year"] = extract_year_from_title(title)
                if not safe_get(c, "type_"):
                    inferred = infer_election_type(title, context, c, deduped, coordinator)
                    if inferred:
                        c["type_"] = inferred
        except Exception:
            pass

    # Attach session_id
    for c in deduped:
        if session_id is not None:
            c["session_id"] = session_id

    _log("debug", "selector", "Initial candidate contests", session_id=session_id,
         payload={"count": len(deduped), "state": norm_state, "county": norm_county, "year": year})

    if not deduped:
        _log("warning", "selector", "No valid contests detected after sanitization", session_id=session_id)
        return []

    # Soft-vs-strict filter
    any_year_present = any(safe_get(c, "year") for c in deduped)
    any_type_present = any(safe_get(c, "type_") for c in deduped)

    filtered = []
    fallbacks = []
    for c in deduped:
        title = safe_get(c, "title", "")
        if not title:
            continue
        if norm_state and normalize_state_name(safe_get(c, "state", "")) not in (None, "", norm_state):
            fallbacks.append(c)
            continue
        if any_year_present and not safe_get(c, "year"):
            fallbacks.append(c)
            continue
        if any_type_present and not safe_get(c, "type_"):
            fallbacks.append(c)
            continue
        filtered.append(c)

    if not filtered:
        filtered = fallbacks or deduped

    _log("debug", "selector", "After soft/strict filtering", session_id=session_id,
         payload={"kept": len(filtered), "fallbacks": len(fallbacks)})

    # Determine offline/verify flags
    offline_mode = False
    try:
        # Heuristic: no semantic model attribute or explicitly False => offline
        offline_mode = not bool(getattr(coordinator, "has_semantic_model", False))
    except Exception:
        offline_mode = True

    candidates = filtered
    if not disable_ml_verify and not offline_mode:
        verify_ctx = {
            "state": norm_state,
            "county": norm_county,
            "year": year,
            "contests": filtered,
            "url": getattr(coordinator, "last_url", None) if hasattr(coordinator, "last_url") else None,
            "session_id": session_id,
            "type_": safe_get(context, "type_", "") if isinstance(context, dict) else ""
        }
        verified = [c for c in filtered if ml_verify_contest(c, coordinator, verify_ctx, threshold=0.75)]
        if not verified:
            verified = feedback_loop_verify_contests(filtered, coordinator, verify_ctx, session_id=session_id)
        if verified:
            candidates = verified
        else:
            _log("warning", "selector", "No contests passed verification; using filtered list.", session_id=session_id)

    # Single-contest fast-path if not forcing interactive
    if len(candidates) == 1 and not force_interactive:
        only = ensure_contest(candidates[0])
        _log("info", "selector", "Auto-selected single contest.", session_id=session_id,
             payload={"title": safe_get(only, "title", "")})
        if log_func:
            log_func(f"[CONTEST] Auto-selected: {safe_get(only, 'title', '')}")
        return [only]

    # Render a clear menu and accept indices or fuzzy text
    titles = [safe_get(c, "title", "") for c in candidates]
    lines = ["Available contests:"]
    for i, t in enumerate(titles):
        y = safe_get(candidates[i], "year")
        typ = safe_get(candidates[i], "type_", "")
        suffix = []
        if y:
            suffix.append(str(y))
        if typ:
            suffix.append(typ)
        suffix_str = f" ({', '.join(suffix)})" if suffix else ""
        lines.append(f"[{i}] {t}{suffix_str}")
    menu = "\n".join(lines)

    # Log menu with both selector (for analytics) and input (for terminal rendering)
    _log("info", "selector", menu, session_id=session_id)
    logger.info({
        "level": "INFO",
        "type": "input",
        "message": menu,
        "session_id": session_id
    })
    
    try:
        user_input = prompt.prompt_input(
            prompt_message,
            default=("0" if not allow_multiple else "all"),
            validator=lambda x: True,  # free-form; we sanitize below
            allow_cancel=True,
            header="CONTEST SELECTION",
            log_func=log_func,
            session_id=session_id,
            context={
                "count": len(titles),
                "state": state, "county": county, "year": year,
                "force_interactive": force_interactive,
                "offline_mode": offline_mode
            }
        )
    except PromptCancelled:
        _log("warning", "prompt", "Contest selection cancelled by user.", session_id=session_id)
        if log_func:
            log_func("[CONTEST] User cancelled contest selection.")
        return []

    choice = (user_input or "").strip()
    if not choice:
        # Default to first if not multiple
        if not allow_multiple and titles:
            return [ensure_contest(candidates[0])]
        return []

    if choice.lower() == "cancel":
        return []

    # Parse indices
    if allow_multiple and choice.lower() in {"all", "*"}:
        indices = list(range(len(titles)))
    else:
        # Parse indices
        indices = []
        parts = [p.strip() for p in re.split(r"[,\s]+", choice) if p.strip()]
        if all(p.isdigit() for p in parts):
            for p in parts:
                i = int(p)
                if 0 <= i < len(titles):
                    indices.append(i)
            if not allow_multiple and indices:
                indices = [indices[0]]
        else:
            # Fuzzy by text
            import difflib
            n = (5 if allow_multiple else 1)
            choices = difflib.get_close_matches(choice, titles, n=n, cutoff=0.45)
            if not choices:
                # containment fallback
                low = choice.lower()
                choices = [t for t in titles if low in t.lower()]
                if not allow_multiple and choices:
                    choices = [choices[0]]
            indices = [titles.index(t) for t in choices if t in titles]
            if not allow_multiple and indices:
                indices = [indices[0]]

    if not indices:
        _log("warning", "prompt", "No valid contest selection; defaulting to first.", session_id=session_id)
        if titles:
            return [ensure_contest(candidates[0])]
        return []

    selected = [ensure_contest(_sync_type_and_election_types(candidates[i]) or candidates[i]) for i in sorted(set(indices))]
    # Attach session_id
    if session_id is not None:
        for c in selected:
            try:
                c["session_id"] = session_id
            except Exception:
                pass

    # Log and return robust list of dicts; never return a bare bool/str
    try:
        _log("info", "selector", "Contest selection complete.", session_id=session_id,
             payload={"selected": [safe_get(c, "title", "") for c in selected]})
        if log_func:
            log_func(f"[CONTEST] User selected contests: {[safe_get(c, 'title', '') for c in selected]}")
    except Exception:
        pass
    
    return selected