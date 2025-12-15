"""
context_coordinator.py

Production-grade Context Coordinator for Election Data Pipeline

- Orchestrates advanced context analysis, NLP, and ML integrity checks.
- Bridges between spaCy (NLP), context_organizer (DOM/ML), and downstream consumers (selectors, handlers, routers).
- Provides robust, dynamic, and cache-aware access to contests, buttons, panels, tables, candidates, precincts, etc.
- Ensures all data is validated, deduplicated, and anomaly-checked before output.
"""
from __future__ import annotations

import difflib
import numbers
import os
import re
import subprocess
import threading
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import orjson
from rapidfuzz import fuzz, process
from sklearn.preprocessing import LabelEncoder

from ..config import BATCH_MAX_WORKERS, CONTEXT_LIBRARY_PATH, LOG_DIR, PROJECT_ROOT
from ..handlers.batch_handler import BatchProcessor
from ..services.election_data_services import ElectionDataService
from ..utils.browser_utils import (
    safe_click,
    safe_count,
    safe_evaluate,
    safe_get_attribute,
    safe_inner_text,
    safe_is_enabled,
    safe_is_visible,
    safe_locator,
    safe_nth,
    safe_wait_for_timeout,
    scan_buttons_with_progress,
)
from ..utils.html_scanner import (
    deduplicate_pattern_kb,
    get_segment_embedding,
    load_pattern_kb,
)
from ..utils.logger_singleton import logger
from ..utils.model_registry import ModelRegistry
from ..utils.shared_logic import (
    sync_type_and_election_types,
    keyphrase_match,
    normalize_county_name,
    normalize_state_name,
    safe_append,
    safe_endswith,
    safe_filename,
    safe_get,
    safe_get_first,
    safe_isupper,
    safe_items,
    safe_lower,
    safe_model_encode,
    safe_replace,
    safe_similarity,
    safe_startswith,
    safe_strip,
    safe_tolist,
)
from ..utils.spacy_utils import extract_dates, extract_entities, extract_locations
from .Context_Library.constants import (
    BALLOT_TYPES,
    BUTTON_TAGS,
    ELECTION_TYPES,
    KNOWN_COUNTY_TO_PRECINCTS_MAP,
    KNOWN_STATE_TO_COUNTY_MAP,
    LOCATION_KEYWORDS,
    PANEL_TAGS,
    PARTY_KEYWORDS,
    STATE_ABBR,
    STATE_MODULE_MAP,
    STATE_TAGS,
    TABLE_TAGS,
)
from .context_organizer import ContextOrganizer
from .Integrity_check import (
    advanced_cross_field_validation,
    detect_anomalies_with_ml,
    election_integrity_checks,
    monitor_db_for_alerts,
    print_integrity_summary,
)
from .librarian import atomic_write_json, clean_for_json


def get_semantic_score(model, text1, text2) -> float:
    """
    Compute semantic similarity between two strings using SentenceTransformer.
    Handles tensor/list conversion and logs errors gracefully.
    """
    # Type and value checks
    if model is None:
        if logger:
            logger.error("[get_semantic_score] Model is None.")
        return 0.0
    if not isinstance(text1, str) or not isinstance(text2, str) or not text1 or not text2:
        if logger:
            logger.error(
                "[get_semantic_score] Invalid input types: text1=%s, text2=%s",
                type(text1),
                type(text2),
            )
        return 0.0
    try:
        emb1 = safe_model_encode(model, text1, convert_to_tensor=True, show_progress_bar=False)
        emb2 = safe_model_encode(model, text2, convert_to_tensor=True, show_progress_bar=False)
        if logger:
            logger.debug("Type of emb1: %s, Type of emb2: %s", type(emb1), type(emb2))
        from sentence_transformers import util
        cos_sim = util.pytorch_cos_sim(emb1, emb2)
        # Defensive extraction
        if hasattr(cos_sim, "item"):
            val = cos_sim.item()
        elif hasattr(cos_sim, "numpy"):
            arr = cos_sim.numpy()
            val = float(arr.flatten()[0]) if arr.size > 0 else 0.0
        elif isinstance(cos_sim, (list, tuple, np.ndarray)):
            val = float(cos_sim[0][0]) if cos_sim and cos_sim[0] else 0.0
        else:
            if logger:
                logger.error("[get_semantic_score] Unexpected cos_sim type: %s", type(cos_sim))
            val = 0.0
        # Final type check
        if not isinstance(val, (float, int)):
            if logger:
                logger.error("[get_semantic_score] Non-numeric similarity value: %s", val)
            return 0.0
        return float(val)
    except Exception as e:
        if logger:
            logger.error("[get_semantic_score] Error: %s", e)
        return 0.0

def merge_and_rank_candidates(
    memory_candidates, dom_candidates, context, keywords, model,
    fuzzy_weight=0.3, semantic_weight=0.3, context_weight=0.2, hierarchy_weight=0.2
) -> List[Dict[str, Any]]:
    """
    Merge memory and DOM candidates, deduplicate, and rank by combined fuzzy and semantic score.
    """
    seen = set()
    all_candidates = []
    for cand in memory_candidates + dom_candidates:
        cand_dict = cand if isinstance(cand, dict) else {}
        if not safe_get(cand_dict, "label"):
            continue
        key = (safe_get(cand_dict, "label", ""), safe_get(cand_dict, "selector", ""))
        if key not in seen:
            seen.add(key)
            all_candidates.append(cand_dict)

    context_dict = context if isinstance(context, dict) else {}
    contest_obj = safe_get(context_dict, "contest", {})
    contest_title = safe_get(contest_obj, "title", "") if isinstance(contest_obj, dict) else str(contest_obj)
    context_str = " ".join([
        contest_title,
        str(safe_get(context_dict, "year", "")),
        str(safe_get(context_dict, "type_", "")),
        str(safe_get(context_dict, "county", "")),
        str(safe_get(context_dict, "state", "")),
    ]).strip()

    expected_class = safe_get(context_dict, "expected_class", "")
    expected_tag = safe_get(context_dict, "expected_tag", "")

    for cand in all_candidates:
        if not isinstance(cand, dict):
            continue
        label = safe_get(cand, "label", "") or ""
        # Strong full-string match
        full_match = int(safe_lower(label.strip()) == safe_lower(contest_title.strip()))
        # Keyphrase-aware match
        keyphrase_score = 0.0
        for kw in (keywords or []):
            if keyphrase_match(label, kw, min_words=2, fuzzy_cutoff=0.85) or keyphrase_match(label, kw, min_words=2, fuzzy_cutoff=0.85):
                keyphrase_score = 1.0
                break
        # Fuzzy/semantic as fallback
        fuzzy_scores = [
            difflib.SequenceMatcher(None, safe_lower(kw), safe_lower(label)).ratio()
            for kw in (keywords or [])
        ]
        fuzzy_score = max(fuzzy_scores) if fuzzy_scores else 0.0
        semantic_score = get_semantic_score(model, context_str, label)
        # Context proximity
        context_heading = safe_get(cand, "context_heading", "")
        context_proximity = 0.0
        if context_heading and contest_title:
            context_proximity = get_semantic_score(model, contest_title, context_heading)
        # Hierarchy/class/tag bonus
        hierarchy_score = 0.0
        cand_class = safe_get(cand, "class", "")
        cand_tag = safe_get(cand, "tag", "")
        if expected_class and (expected_class in cand_class or expected_class in safe_lower(cand_class)):
            hierarchy_score += 0.5
        if expected_tag and (expected_tag == cand_tag or expected_tag in safe_lower(cand_tag)):
            hierarchy_score += 0.5
        if full_match:
            hierarchy_score += 1.0
        cand["keyphrase_score"] = keyphrase_score
        cand["fuzzy_score"] = fuzzy_score
        cand["semantic_score"] = semantic_score
        cand["context_proximity"] = context_proximity
        cand["hierarchy_score"] = hierarchy_score
        cand["combined_score"] = (
            0.4 * keyphrase_score +  # prioritize keyphrase match
            fuzzy_weight * fuzzy_score +
            semantic_weight * semantic_score +
            context_weight * context_proximity +
            hierarchy_weight * hierarchy_score
        )

    all_candidates = [c for c in all_candidates if isinstance(c, dict)]
    all_candidates.sort(
        key=lambda c: (
            safe_get(c, "combined_score", 0),
            safe_get(c, "is_visible", False),
            safe_get(c, "is_clickable", False)
        ),
        reverse=True
    )
    return all_candidates

def dynamic_state_county_detection(context, html, debug=False) -> tuple:
    """
    Robustly detect county (first) and state (second) using all available clues and cross-referencing.
    Utilizes context fields, contest titles, URL, and canonical librarian mappings.
    Returns (county, state, handler_path, detection_log)
    """
    # Lightweight in-function caches for large lookups
    if not hasattr(dynamic_state_county_detection, "_lookup_cache"):
        dynamic_state_county_detection._lookup_cache = {}
    cache = dynamic_state_county_detection._lookup_cache
    detection_log = []
    state_to_county = KNOWN_STATE_TO_COUNTY_MAP
    county_to_precinct = KNOWN_COUNTY_TO_PRECINCTS_MAP
    state_module_map = STATE_MODULE_MAP
    # Cache normalized lists/sets used multiple times
    cache_key = "norm_sets"
    if cache_key in cache:
        known_states, all_counties, all_precincts = cache[cache_key]
    else:
        known_states = set(state_to_county.keys())
        state_to_county_values = state_to_county.values() if isinstance(state_to_county, dict) else state_to_county
        all_counties = {normalize_county_name(c) for counties in state_to_county_values for c in counties}
        county_to_precinct_values = county_to_precinct.values() if isinstance(county_to_precinct, dict) else county_to_precinct
        all_precincts = {normalize_county_name(d) for precincts in county_to_precinct_values for d in precincts}
        cache[cache_key] = (known_states, all_counties, all_precincts)

    known_states = set(state_to_county.keys())
    state_to_county_values = state_to_county.values() if isinstance(state_to_county, dict) else state_to_county
    all_counties = {normalize_county_name(c) for counties in state_to_county_values for c in counties}

    county_to_precinct_values = county_to_precinct.values() if isinstance(county_to_precinct, dict) else county_to_precinct
    all_precincts = {normalize_county_name(d) for precincts in county_to_precinct_values for d in precincts}

    entities_cache = None

    def _get_entities():
        nonlocal entities_cache
        if entities_cache is None:
            try:
                entities_cache = extract_entities(html) if html else []
            except Exception:
                entities_cache = []
        return entities_cache or []

    def _precinct_to_county(target: str | None) -> str | None:
        if not target:
            return None
        for c, precincts in county_to_precinct.items():
            if not isinstance(precincts, list):
                continue
            normalized_precincts = {normalize_county_name(x) for x in precincts}
            if target in normalized_precincts:
                return normalize_county_name(c)
        return None

    # --- 1. Try context fields directly (normalize and validate) ---
    if not isinstance(context, dict) or not context:
        context = {}
    raw_county = safe_get(context, "county", None)
    raw_state = safe_get(context, "state", None)
    session_id = safe_get(context, "session_id", None)
    county = normalize_county_name(raw_county) if raw_county else None
    state = normalize_state_name(raw_state) if raw_state else None

    # Validate county: is it a real county, or a precinct?
    if county:
        if county in all_counties:
            detection_log.append(f"County found in context: {county} (validated as county)")
            detection_log.append("[SOURCE] context:county")
            # Webapp-friendly structured log with a message (avoid empty message field)
            logger.info({
                "level": "INFO",
                "type": "router",
                "message": f"[Context Detection] County found in context: {county} (validated as county)",
                "session_id": session_id
            })
        elif county in all_precincts:
            # Map up to parent county
            parent_county = None
            for c, precincts in county_to_precinct.items():
                if not isinstance(precincts, list):
                    continue
                if county in {normalize_county_name(d) for d in precincts}:
                    parent_county = normalize_county_name(c)
                    break
            if parent_county:
                detection_log.append(f"County '{county}' found in context, but is a precinct. Mapped to parent county '{parent_county}'.")
                logger.info({
                    "level": "INFO",
                    "type": "router",
                    "message": f"[Context Detection] County '{county}' mapped to parent '{parent_county}'",
                    "session_id": session_id
                })               
                county = parent_county
                detection_log.append("[SOURCE] context:precinct->county")
            else:
                detection_log.append(f"County '{county}' found in context, but is a precinct with no parent mapping.")
                logger.info({
                    "level": "INFO",
                    "type": "router",
                    "message": f"[Context Detection] County '{county}' is a precinct with no parent mapping",
                    "session_id": session_id
                })
        else:
            detection_log.append(f"County '{county}' found in context, but not recognized as county or precinct.")
            logger.info({
                "level": "INFO",
                "type": "router",
                "message": f"[Context Detection] County '{county}' not recognized as county or precinct",
                "session_id": session_id
            })
            county = None

    # Validate state: is it a real state?
    if state:
        if state in known_states:
            detection_log.append(f"State found in context: {state} (validated as state)")
            detection_log.append("[SOURCE] context:state")
            logger.info({
                "level": "INFO",
                "type": "router",
                "message": f"[Context Detection] State found in context: {state} (validated as state)",
                "session_id": session_id
            })
        else:
            # Try to map via state_module_map (handle abbreviations and fuzzy)
            mapped_state = state_module_map.get(state)
            if not mapped_state:
                # Try abbreviation
                abbr = safe_lower(state)
                mapped_state = STATE_ABBR.get(abbr)
                if mapped_state:
                    detection_log.append(f"State '{state}' mapped from abbreviation to '{mapped_state}'.")
                    logger.info({
                        "level": "INFO",
                        "type": "router",
                        "message": f"[Context Detection] State '{state}' mapped from abbreviation to '{mapped_state}'.",
                        "session_id": session_id
                    })
            if mapped_state:
                state = normalize_state_name(mapped_state)
                detection_log.append(f"State '{state}' found in context, mapped via state_module_map/abbr.")
                logger.info({
                    "level": "INFO",
                    "type": "router",
                    "message": f"[Context Detection] State '{state}' mapped via state_module_map/abbr.",
                    "session_id": session_id
                })
                detection_log.append("[SOURCE] context:mapped_state")
            else:
                # Fuzzy match as last resort
                match = difflib.get_close_matches(state, known_states, n=1, cutoff=0.8)
                if match:
                    state = safe_get_first(match, "state_match", None, logger)
                    detection_log.append(f"State '{state}' fuzzy-matched from context.")
                    logger.info({
                        "level": "INFO",
                        "type": "router",
                        "message": f"[Context Detection] State '{state}' fuzzy-matched from context.",
                        "session_id": session_id
                    })
                else:
                    detection_log.append(f"State '{state}' found in context, but not recognized.")
                    logger.info({
                        "level": "INFO",
                        "type": "router",
                        "message": f"[Context Detection] State '{state}' found in context, but not recognized.",
                        "session_id": session_id
                    })
                    state = None

    # --- 2. Try to extract county from URL ---
    url = safe_get(context, "url", "")
    if not county and url:
        url_lower = safe_lower(url)
        # Exact match
        for c in all_counties:
            if c in url_lower:
                county = c
                detection_log.append(f"County '{county}' detected from URL.")
                detection_log.append("[SOURCE] url:county_exact")
                logger.info({
                    "level": "INFO",
                    "type": "router",
                    "message": f"[Context Detection] County '{county}' detected from URL.",
                    "session_id": session_id
                })
                break
        # precinct in URL
        if not county:
            for d in all_precincts:
                if d in url_lower:
                    for c, precincts in county_to_precinct.items():
                        if not isinstance(precincts, list):
                            continue
                        if d in {normalize_county_name(x) for x in precincts}:
                            county = normalize_county_name(c)
                            detection_log.append(f"precinct '{d}' detected from URL, mapped to county '{county}'")
                            detection_log.append("[SOURCE] url:precinct->county")
                            logger.info({
                                "level": "INFO",
                                "type": "router",
                                "message": f"[Context Detection] Precinct '{d}' detected from URL, mapped to county '{county}'.",
                                "session_id": session_id
                            })
                            break
                    if county:
                        break
        # Fuzzy match county in URL
        if not county:
            url_tokens = re.split(r"[\W_]+", url_lower)
            matches = difflib.get_close_matches(" ".join(url_tokens), all_counties, n=1, cutoff=0.7)
            if matches:
                county = safe_get_first(matches, "county_match", None, logger)
                detection_log.append(f"County '{county}' fuzzy-matched from URL tokens.")
                detection_log.append("[SOURCE] url:county_fuzzy")
                logger.info({
                    "level": "INFO",
                    "type": "router",
                    "message": f"[Context Detection] County '{county}' fuzzy-matched from URL tokens.",
                    "session_id": session_id
                })
            else:
                matches = difflib.get_close_matches(" ".join(url_tokens), all_precincts, n=1, cutoff=0.7)
                if matches:
                    for c, precincts in county_to_precinct.items():
                        if not isinstance(precincts, list):
                            continue
                        match_val = safe_get_first(matches, "precinct_match", None, logger)
                        if match_val in {normalize_county_name(x) for x in precincts}:
                            county = normalize_county_name(c)
                            detection_log.append(f"precinct '{match_val}' fuzzy-matched from URL tokens, mapped to county '{county}'")
                            detection_log.append("[SOURCE] url:precinct->county")
                            logger.info({
                                "level": "INFO",
                                "type": "router",
                                "message": f"[Context Detection] Precinct '{match_val}' fuzzy-matched from URL tokens, mapped to county '{county}'.",
                                "session_id": session_id
                            })
                            break

    # --- 3. Try to extract county from contest titles ---
    contests = safe_get(context, "contests", [])
    if not county and contests:
        for contest in contests:
            if not isinstance(contest, dict):
                continue
            title = safe_get(contest, "title", "")
            title_lower = safe_lower(title)
            for c in all_counties:
                if re.search(rf"\b{re.escape(c)}\b", title_lower):
                    county = c
                    detection_log.append(f"County '{county}' detected from contest title: '{title}'")
                    logger.info({
                        "level": "INFO",
                        "type": "router",
                        "message": f"[Context Detection] County '{county}' detected from contest title: '{title}'.",
                        "session_id": session_id
                    })
                    break
            if county:
                break
            for d in all_precincts:
                if re.search(rf"\b{re.escape(d)}\b", title_lower):
                    for c, precincts in county_to_precinct.items():
                        if not isinstance(precincts, list):
                            continue
                        if d in {normalize_county_name(x) for x in precincts}:
                            county = normalize_county_name(c)
                            detection_log.append(f"precinct '{d}' detected from contest title: '{title}', mapped to county '{county}'")
                            logger.info({
                                "level": "INFO",
                                "type": "router",
                                "message": f"[Context Detection] Precinct '{d}' detected from contest title: '{title}', mapped to county '{county}'.",
                                "session_id": session_id
                            })
                            break
                    if county:
                        break
            if county:
                break

    # --- 4. Try to extract county from HTML using NLP entities ---
    if not county and html:
        entities = _get_entities()
        gpe_entities = [normalize_county_name(ent) for ent, label in entities if label in ("GPE", "LOC")]
        county_hits: Counter[str] = Counter()
        precinct_hits: Counter[str] = Counter()
        for ent in gpe_entities:
            if ent in all_counties:
                county_hits[ent] += 1
            elif ent in all_precincts:
                precinct_hits[ent] += 1

        if county_hits:
            top_county, hits = county_hits.most_common(1)[0]
            min_hits = 1 if len(county_hits) == 1 else 2
            if hits >= min_hits:
                county = top_county
                detection_log.append(f"County '{county}' selected from HTML NLP entities (hits={hits}).")
                detection_log.append("[SOURCE] nlp:county_majority")
                logger.info({
                    "level": "INFO",
                    "type": "router",
                    "message": f"[Context Detection] County '{county}' selected from HTML NLP entities (hits={hits}).",
                    "session_id": session_id
                })

        if not county and precinct_hits:
            top_precinct, hits = precinct_hits.most_common(1)[0]
            mapped = _precinct_to_county(top_precinct)
            if mapped:
                county = mapped
                detection_log.append(
                    f"precinct '{top_precinct}' selected from HTML NLP entities (hits={hits}), mapped to county '{county}'"
                )
                detection_log.append("[SOURCE] nlp:precinct_majority")
                logger.info({
                    "level": "INFO",
                    "type": "router",
                    "message": (
                        f"[Context Detection] Precinct '{top_precinct}' selected from HTML NLP entities (hits={hits}), mapped to '{county}'."
                    ),
                    "session_id": session_id
                })

        if not county:
            for ent in gpe_entities:
                if ent in all_counties:
                    county = ent
                    detection_log.append(f"County '{county}' detected from HTML NLP entity.")
                    detection_log.append("[SOURCE] nlp:county")
                    logger.info({
                        "level": "INFO",
                        "type": "router",
                        "message": f"[Context Detection] County '{county}' detected from HTML NLP entity.",
                        "session_id": session_id
                    })
                    break
                elif ent in all_precincts:
                    mapped = _precinct_to_county(ent)
                    if mapped:
                        county = mapped
                        detection_log.append(f"precinct '{ent}' detected from HTML NLP entity, mapped to county '{county}'")
                        detection_log.append("[SOURCE] nlp:precinct->county")
                        logger.info({
                            "level": "INFO",
                            "type": "router",
                            "message": f"[Context Detection] Precinct '{ent}' detected from HTML NLP entity, mapped to county '{county}'.",
                            "session_id": session_id
                        })
                        break

    # --- 5. Now try to detect state, using county if found ---
    if not state and county:
        for s, counties in state_to_county.items():
            if not isinstance(counties, list):
                continue
            if county in {normalize_county_name(x) for x in counties}:
                state = normalize_state_name(s)
                detection_log.append(f"State '{state}' inferred from county '{county}'.")
                logger.info({
                    "level": "INFO",
                    "type": "router",
                    "message": f"[Context Detection] State '{state}' inferred from county '{county}'.",
                    "session_id": session_id
                })
                break

    # --- 6. Try to extract state from URL ---
    if not state and url:
        url_lower = safe_lower(url)
        for s in known_states:
            if s in url_lower:
                state = s
                detection_log.append(f"State '{state}' detected from URL.")
                detection_log.append("[SOURCE] url:state_exact")
                logger.info({
                    "level": "INFO",
                    "type": "router",
                    "message": f"[Context Detection] State '{state}' detected from URL.",
                    "session_id": session_id
                })
                break
        # Fuzzy match state in URL
        if not state:
            url_tokens = re.split(r"[\W_]+", url_lower)
            matches = difflib.get_close_matches(" ".join(url_tokens), list(known_states), n=1, cutoff=0.7)
            if matches:
                state = safe_get_first(matches, "state_match", None, logger)
                detection_log.append(f"State '{state}' fuzzy-matched from URL tokens.")
                detection_log.append("[SOURCE] url:state_fuzzy")
                logger.info({
                    "level": "INFO",
                    "type": "router",
                    "message": f"[Context Detection] State '{state}' fuzzy-matched from URL tokens.",
                    "session_id": session_id
                })

    # --- 7. Try to extract state from contest titles ---
    if not state and contests:
        for contest in contests:
            if not isinstance(contest, dict):
                continue
            title = safe_get(contest, "title", "")
            title_lower = safe_lower(title)
            for s in known_states:
                if s in title_lower:
                    state = s
                    detection_log.append(f"State '{state}' detected from contest title: '{title}'")
                    detection_log.append("[SOURCE] title:state")
                    logger.info({
                        "level": "INFO",
                        "type": "router",
                        "message": f"[Context Detection] State '{state}' detected from contest title: '{title}'.",
                        "session_id": session_id
                    })
                    break
            if state:
                break

    # --- 8. Try to extract state from HTML using NLP entities ---
    if not state and html:
        entities = _get_entities()
        gpe_entities = [normalize_state_name(ent) for ent, label in entities if label in ("GPE", "LOC")]
        state_hits: Counter[str] = Counter(ent for ent in gpe_entities if ent in known_states)
        if state_hits:
            top_state, hits = state_hits.most_common(1)[0]
            min_hits = 1 if len(state_hits) == 1 else 2
            if hits >= min_hits:
                state = top_state
                detection_log.append(f"State '{state}' selected from HTML NLP entities (hits={hits}).")
                detection_log.append("[SOURCE] nlp:state_majority")
                logger.info({
                    "level": "INFO",
                    "type": "router",
                    "message": f"[Context Detection] State '{state}' selected from HTML NLP entities (hits={hits}).",
                    "session_id": session_id
                })

        if not state:
            for ent in gpe_entities:
                if ent in known_states:
                    state = ent
                    detection_log.append(f"State '{state}' detected from HTML NLP entity.")
                    detection_log.append("[SOURCE] nlp:state")
                    logger.info({
                        "level": "INFO",
                        "type": "router",
                        "message": f"[Context Detection] State '{state}' detected from HTML NLP entity.",
                        "session_id": session_id
                    })
                    break

    # --- 9. Special case: DC and other non-county states ---
    if state == "district_of_columbia":
        county = "district of columbia"
        detection_log.append("Special case: DC detected, setting county to 'district of columbia'.")
        logger.info({
            "level": "INFO",
            "type": "router",
            "message": "Special case: DC detected, setting county to 'district of columbia'.",
            "session_id": session_id
        })

    # --- 10. If state found but no county, check for available county handlers ---
    handler_path = None
    normalized_state = state  # already normalized
    normalized_county = county  # already normalized

    if normalized_state and not normalized_county:
        county_dir = os.path.join(
            PROJECT_ROOT, "webapp", "parser", "handlers", "states", normalized_state, "county"
        )
        available_counties = []
        if os.path.isdir(county_dir):
            for fname in os.listdir(county_dir):
                if fname.endswith(".py") and not fname.startswith("__"):
                    county_name = fname[:-3]
                    available_counties.append(county_name)
            detection_log.append(f"Available county handlers for state '{normalized_state}': {available_counties}")
            logger.info({
                "level": "INFO",
                "type": "router",
                "message": f"[Context Detection] Available county handlers for state '{normalized_state}': {available_counties}",
                "session_id": session_id
            })
            url_and_html = safe_lower((url or "") + " " + (html or ""))
            # Try exact match in URL/HTML
            for c in available_counties:
                if c in url_and_html:
                    normalized_county = c
                    detection_log.append(f"County '{normalized_county}' matched to available handler from URL/HTML context.")
                    logger.info({
                        "level": "INFO",
                        "type": "router",
                        "message": f"[Context Detection] County '{normalized_county}' matched to available handler from URL/HTML context.",
                        "session_id": session_id
                    })
                    break
            # Try fuzzy match in URL/HTML
            if not normalized_county and available_counties:
                tokens = re.split(r"[\W_]+", url_and_html)
                matches = difflib.get_close_matches(" ".join(tokens), available_counties, n=1, cutoff=0.7)
                if matches:
                    normalized_county = safe_get_first(matches, "county_handler_match", None, logger)
                    detection_log.append(f"County '{normalized_county}' fuzzy-matched to available handler from URL/HTML context.")
                    logger.info({
                        "level": "INFO",
                        "type": "router",
                        "message": f"[Context Detection] County '{normalized_county}' fuzzy-matched to available handler from URL/HTML context.",
                        "session_id": session_id
                    })
            # If only one county handler is available, use it as a fallback
            if not normalized_county and len(available_counties) == 1:
                normalized_county = safe_get_first(available_counties, "only_county_handler", None, logger)
                detection_log.append(f"Only one county handler available ('{normalized_county}'); using as fallback.")
                logger.info({
                    "level": "INFO",
                    "type": "router",
                    "message": f"[Context Detection] Only one county handler available ('{normalized_county}'); using as fallback.",
                    "session_id": session_id
                })
            elif not normalized_county:
                detection_log.append("No matching county handler found in URL/HTML; will use state handler.")
                logger.info({
                    "level": "INFO",
                    "type": "router",
                    "message": "No matching county handler found in URL/HTML; will use state handler.",
                    "session_id": session_id
                })
        else:
            detection_log.append(f"No county handler directory found for state '{normalized_state}'.")
            logger.info({
                "level": "INFO",
                "type": "router",
                "message": f"No county handler directory found for state '{normalized_state}'.",
                "session_id": session_id
            })

    # --- Set handler path based on what was found ---
    if normalized_state and normalized_county:
        handler_path = f"webapp.parser.handlers.states.{normalized_state}.county.{normalized_county}"
    elif normalized_state:
        state_handler_file = os.path.join(
            PROJECT_ROOT, "webapp", "parser", "handlers", "states", normalized_state, f"{normalized_state}.py"
        )
        if os.path.isfile(state_handler_file):
            handler_path = f"webapp.parser.handlers.states.{normalized_state}.{normalized_state}"
        else:
            handler_path = f"webapp.parser.handlers.states.{normalized_state}"

    # --- Final fallback ---
    if not normalized_county:
        detection_log.append("County could not be detected.")
        logger.info({
            "level": "INFO",
            "type": "router",
            "message": "County could not be detected.",
            "session_id": session_id
        })
    if not normalized_state:
        detection_log.append("State could not be detected.")
        logger.info({
            "level": "INFO",
            "type": "router",
            "message": "State could not be detected.",
            "session_id": session_id
        })
    # Summarize detection sources and compute a coarse confidence score
    try:
        srcs = [line.split('] ', 1)[1] for line in detection_log if isinstance(line, str) and line.startswith('[SOURCE]')]
        unique_sources = set(srcs)
        confidence = min(1.0, 0.2 * len(unique_sources))
        detection_log.append(f"[SUMMARY] sources={sorted(unique_sources)} confidence={confidence:.2f}")
    except Exception:
        pass
    if debug:
        for log in detection_log:
            logger.debug({
                "level": "DEBUG",
                "type": "router",
                "message": f"[Context Detection] {log}",
                "session_id": session_id
            })
    return normalized_county, normalized_state, handler_path, detection_log

# --- Core Coordinator Class ---

class ContextCoordinator(object):
    """
    Main interface for all context/NLP/ML operations.
    Use this class to access contests, buttons, panels, tables, candidates, precincts, etc.
    """
    _dom_parts_warning_count = 0
    def __init__(self, use_library=True, enable_ml=True, alert_monitor=True, debug=False) -> None:
        self.enable_ml = enable_ml
        self.alert_monitor = alert_monitor
        self.debug = debug
        self.data_service = ElectionDataService()
        self.organizer = ContextOrganizer(
            use_library=use_library,
            enable_ml=enable_ml,
            debug=debug
        )
        self.organized = None
        self.last_raw_context = None
        self.clicked_button_selectors = set()
        self.accepted_buttons_cache = {}
        self._semantic_model = None
        self.learning_mode = False  # Default to False for clarity
        self.alert_monitor_thread = None

        if enable_ml:
            self._semantic_model = ModelRegistry.get_sentence_transformer("all-MiniLM-L6-v2")

        if alert_monitor:
            self.start_alert_monitoring()
            
    def __del__(self) -> None:
        """
        Ensure alert monitoring thread is cleaned up on destruction.
        """
        try:
            if hasattr(self, "alert_monitor_thread") and self.alert_monitor_thread and self.alert_monitor_thread.is_alive():
                logger.info("[ALERT MONITOR] Stopping alert monitoring thread.")
                self.alert_monitor_thread.join(timeout=1)
                if self.alert_monitor_thread.is_alive():
                    logger.warning("[ALERT MONITOR] Thread did not stop cleanly.")
                else:
                    logger.info("[ALERT MONITOR] Thread stopped successfully.")
            else:
                logger.info("[ALERT MONITOR] No active thread to stop.")
        except Exception as e:
            logger.error(f"[ALERT MONITOR] Exception during cleanup: {e}", exc_info=True)
        finally:
            if hasattr(self, "alert_monitor_thread"):
                self.alert_monitor_thread = None

    @property
    def library(self):
        return getattr(self.organizer, "library", {})

    @property
    def pattern_kb(self) -> List[dict]:
        """
        Robust property to get the current pattern KB (feedback/ML/labeling knowledge base).
        Always returns a deduplicated list, never None.
        """
        try:
            # Prefer method if available
            if hasattr(self, "get_feedback_pattern_kb"):
                kb = self.get_feedback_pattern_kb()
                if kb:
                    seen = set()
                    deduped = []
                    for entry in kb:
                        key = safe_get(entry, "pattern_id") or safe_get(entry, "segment_hash")
                        if key and key not in seen:
                            seen.add(key)
                            deduped.append(entry)
                    return deduped
            # Fallback: try attribute
            pattern_kb = getattr(self, "_pattern_kb", None)
            if isinstance(pattern_kb, list):
                # Defensive deduplication
                seen = set()
                deduped = []
                for entry in pattern_kb:
                    if not isinstance(entry, dict):
                        continue
                    key = entry.get("pattern_id") or entry.get("segment_hash")
                    if key and key not in seen:
                        seen.add(key)
                        deduped.append(entry)
                return deduped
        except Exception as e:
            logger.error(f"[pattern_kb property] Error loading pattern KB: {e}")
        # Final fallback: load from disk
        try:
            kb = load_pattern_kb()
            return deduplicate_pattern_kb(kb)
        except Exception as e:
            logger.error(f"[pattern_kb property] Fallback load failed: {e}")
            return []

    # --- Batch orchestration ---

    def handle_batch(
        self,
        *,
        page: Any,
        context: Optional[dict],
        target_url: str,
        processed_info: Any,
        ai_analyze_results: Callable[[List[str], List[Dict[str, Any]], str, Dict[str, Any]], None],
        stream_results: Callable[[List[str], List[Dict[str, Any]], str, Dict[str, Any]], None],
        mark_url_processed: Callable[..., None],
        output_dir: str,
        session_id: Optional[str],
        handler: Any,
        initial_result: Optional[Tuple[List[str], List[Dict[str, Any]], str, Dict[str, Any]]] = None,
    ) -> None:
        """
        Entry point for batch execution when multiple contests are selected.

        Parameters mirror the HTML orchestrator so the coordinator can invoke
        downstream post-processing hooks (AI analysis, streaming, output tracking).
        """
        selected_races: List[Dict[str, Any]] = []
        if isinstance(context, dict):
            selected_races_raw = context.get("selected_races") or []
            if isinstance(selected_races_raw, list):
                selected_races = [race for race in selected_races_raw if race]

        if not selected_races and not initial_result:
            logger.warning({
                "level": "WARNING",
                "type": "batch",
                "message": "Batch mode requested but no selected races or initial result provided.",
                "session_id": session_id,
                "url": target_url,
            })
            return

        if handler is None or not hasattr(handler, "parse"):
            raise ValueError("Batch mode requires a handler with a callable 'parse' method.")

        try:
            processor = BatchProcessor(
                coordinator=self,
                handler=handler,
                page=page,
                base_context=context,
                selected_races=selected_races,
                initial_result=initial_result,
                session_id=session_id,
                target_url=target_url,
                output_dir=output_dir,
                processed_info=processed_info,
                ai_analyze_results=ai_analyze_results,
                stream_results=stream_results,
                mark_url_processed=mark_url_processed,
                max_workers=max(1, int(BATCH_MAX_WORKERS)),
            )
            processor.run()
        except Exception as exc:
            logger.error({
                "level": "ERROR",
                "type": "batch",
                "message": f"[Batch] Execution failed: {exc}",
                "session_id": session_id,
                "url": target_url,
            }, exc_info=True)
            raise

    def append_to_context_library(self, organized, path=None, merge_lists=True, deduplicate=True) -> bool:
        """
        Append or update the organized context into the context library JSON file.
        """
        try:
            return self.organizer.append_to_context_library(
                organized,
                path=path,
                merge_lists=merge_lists,
                deduplicate=deduplicate
            )
        except Exception as e:
            logger.error(f"[append_to_context_library] Failed: {e}", exc_info=True)
            return False

    def _build_enrichment_plan(self, raw_context, overrides=None) -> dict:
        """Derive a scoped enrichment plan so downstream work can run in targeted routes."""
        ctx = raw_context if isinstance(raw_context, dict) else {}

        def _has_any(keys: list[str]) -> bool:
            for key in keys:
                if key in ctx and ctx.get(key) not in (None, "", [], {}, ()):  # treat falsy containers as absent
                    return True
            return False

        raw_hint = (
            ctx.get("source_type")
            or ctx.get("source")
            or ctx.get("format")
        )
        normalized_hint = raw_hint.lower().strip() if isinstance(raw_hint, str) and raw_hint.strip() else ""

        detection_rules = [
            ("pdf", lambda: normalized_hint == "pdf" or _has_any(["pdf_context", "manual_file", "pdf_path", "pdf_metadata"])),
            ("ocr", lambda: normalized_hint in {"ocr", "image", "scan"} or _has_any(["ocr_blocks", "ocr_text", "image_path", "screenshot_path"])),
            ("csv", lambda: normalized_hint == "csv" or _has_any(["csv_rows", "csv_path", "csv_blob", "csv_context", "spreadsheet_rows", "spreadsheet_path"])),
            ("json", lambda: normalized_hint in {"json", "jsonl"} or _has_any(["json_blob", "json_rows", "json_context", "rawjson_enrichment", "api_json"])),
            ("api", lambda: normalized_hint == "api" or _has_any(["api_response", "api_payload", "webhook_event"])),
            ("xml", lambda: normalized_hint == "xml" or _has_any(["xml_payload", "xml_path", "xml_tree"])),
            ("html", lambda: normalized_hint == "html" or _has_any(["raw_html", "dom_parts", "tagged_segments_with_attrs"])),
        ]

        source_hint = ""
        for label, predicate in detection_rules:
            try:
                if predicate():
                    source_hint = label
                    break
            except Exception:
                continue
        if not source_hint:
            source_hint = normalized_hint or "html"

        format_profiles: dict[str, dict] = {
            "html": {
                "routes": {
                    "dom",
                    "sections",
                    "panels",
                    "buttons",
                    "headings",
                    "tables",
                    "candidate_panels",
                    "location_panels",
                    "ml",
                    "integrity",
                },
                "reason": "HTML/DOM source detected; enabling DOM scans and section grouping.",
                "tags": {"route:html_dom"},
            },
            "pdf": {
                "routes": {
                    "tables",
                    "candidate_panels",
                    "location_panels",
                    "ballot_types",
                    "results_timestamps",
                    "party_labels",
                    "vote_methods",
                    "ml",
                    "integrity",
                },
                "reason": "PDF ingestion detected; prioritize table reconstruction + party lookups.",
                "tags": {"route:pdf_tables"},
            },
            "ocr": {
                "routes": {
                    "dom",
                    "sections",
                    "tables",
                    "candidate_panels",
                    "location_panels",
                    "ml",
                },
                "reason": "OCR/image input detected; run DOM-style grouping plus ML cleanup only.",
                "tags": {"route:ocr_dom"},
            },
            "csv": {
                "routes": {
                    "tables",
                    "ballot_types",
                    "vote_methods",
                    "candidate_panels",
                    "ml",
                    "integrity",
                },
                "reason": "CSV/spreadsheet input detected; skip DOM and emphasize structured tables.",
                "tags": {"route:csv_tables"},
            },
            "json": {
                "routes": {
                    "contests",
                    "candidate_panels",
                    "location_panels",
                    "ballot_types",
                    "vote_methods",
                    "ml",
                    "integrity",
                },
                "reason": "JSON payload detected; treat as structured data with ML + integrity checks.",
                "tags": {"route:json_structured"},
            },
            "api": {
                "routes": {
                    "contests",
                    "candidate_panels",
                    "location_panels",
                    "ballot_types",
                    "vote_methods",
                    "ml",
                    "integrity",
                },
                "reason": "API feed detected; leverage structured enrichment routes only.",
                "tags": {"route:api_structured"},
            },
            "xml": {
                "routes": {
                    "tables",
                    "candidate_panels",
                    "location_panels",
                    "ballot_types",
                    "vote_methods",
                    "ml",
                },
                "reason": "XML payload detected; treat similar to structured tables without DOM work.",
                "tags": {"route:xml_structured"},
            },
        }

        routes: set[str] = set()
        reasoning: list[str] = []
        metadata_tags: set[str] = {f"source:{source_hint}"}
        dynamic_paths: list[dict[str, Any]] = []

        profile = format_profiles.get(source_hint) or format_profiles.get("html")
        if profile:
            routes.update(profile["routes"])
            reasoning.append(profile["reason"])
            metadata_tags.update(profile["tags"])
            dynamic_paths.append({
                "format": source_hint,
                "routes": sorted(profile["routes"]),
                "reason": profile["reason"],
                "trigger": "format_profile",
            })

        if ctx.get("raw_html") or ctx.get("dom_parts"):
            routes.update({"dom", "sections", "panels", "buttons", "headings"})
            reasoning.append("HTML context present; enabling panel/button grouping.")
            metadata_tags.add("route:dom_panels")
            dynamic_paths.append({
                "format": "html_dom_context",
                "routes": ["dom", "sections", "panels", "buttons", "headings"],
                "reason": "raw_html/dom_parts provided",
                "trigger": "raw_html",
            })

        if ctx.get("vote_methods") or ctx.get("ballot_types"):
            routes.update({"ballot_types", "vote_methods"})
            reasoning.append("Detected ballot/vote method clues in raw context.")
            dynamic_paths.append({
                "format": "ballot_meta",
                "routes": ["ballot_types", "vote_methods"],
                "reason": "vote method fields present",
                "trigger": "ballot_types"
            })

        if ctx.get("candidate_panels") or ctx.get("location_panels"):
            routes.update({"candidate_panels", "location_panels"})
            reasoning.append("Existing candidate/location panels provided; preserving enrichment path.")
            dynamic_paths.append({
                "format": "panel_inheritance",
                "routes": ["candidate_panels", "location_panels"],
                "reason": "panel payload detected",
                "trigger": "panels",
            })

        if ctx.get("tables") or ctx.get("line_records"):
            routes.add("tables")
            reasoning.append("Structured tables present; running table grouping route.")
            dynamic_paths.append({
                "format": "table_payload",
                "routes": ["tables"],
                "reason": "tables/line_records provided",
                "trigger": "tables",
            })

        if ctx.get("contests"):
            routes.update({"contests", "ml", "integrity"})
            reasoning.append("Contest payload found; keeping ML + integrity checks enabled.")

        if not routes:
            routes.update({"dom", "sections", "contests"})

        if not self.enable_ml and "ml" in routes:
            routes.discard("ml")
            reasoning.append("Coordinator ML disabled; dropped 'ml' route.")

        forced_routes = set()
        dropped_routes = set()
        if overrides:
            if isinstance(overrides, (list, set, tuple)):
                forced_routes = set(overrides)
            elif isinstance(overrides, dict):
                forced_routes = set(overrides.get("force_routes") or [])
                dropped_routes = set(overrides.get("drop_routes") or [])
                override_source = overrides.get("source")
                if override_source:
                    source_hint = override_source
            else:
                forced_routes = {overrides}
        if forced_routes:
            routes.update(forced_routes)
            metadata_tags.update({f"override_force:{route}" for route in forced_routes})
            reasoning.append(f"Forced routes applied: {sorted(forced_routes)}")
            dynamic_paths.append({
                "format": "override",
                "routes": sorted(forced_routes),
                "reason": "caller override",
                "trigger": "override_force",
            })
        if dropped_routes:
            routes.difference_update(dropped_routes)
            metadata_tags.update({f"override_drop:{route}" for route in dropped_routes})
            reasoning.append(f"Dropped routes per override: {sorted(dropped_routes)}")

        dependent_routes = {
            "panels",
            "buttons",
            "tables",
            "candidate_panels",
            "location_panels",
            "headings",
            "ballot_types",
            "results_timestamps",
            "party_labels",
            "vote_methods",
        }
        if routes & dependent_routes:
            routes.add("sections")

        plan = {
            "source": source_hint,
            "routes": sorted(routes),
            "reasoning": reasoning,
            "metadata_tags": sorted(metadata_tags),
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "dynamic_paths": dynamic_paths,
        }
        return plan

    def _log_enrichment_snapshot(self, plan: dict | None, organized_result: dict | None, summary: dict | None = None) -> None:
        if not plan:
            return
        organized = organized_result if isinstance(organized_result, dict) else {}
        contests = organized.get("contests") if isinstance(organized, dict) else []
        metadata = organized.get("metadata", {}) if isinstance(organized, dict) else {}
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "plan": plan,
            "contest_count": len(contests) if isinstance(contests, list) else 0,
            "routes_executed": plan.get("routes", []),
            "anomaly_count": len(organized.get("anomalies", [])) if isinstance(organized.get("anomalies", []), list) else 0,
            "integrity_issue_count": len(organized.get("integrity_issues", [])) if isinstance(organized.get("integrity_issues", []), list) else 0,
            "state": metadata.get("state") if isinstance(metadata, dict) else None,
            "county": metadata.get("county") if isinstance(metadata, dict) else None,
            "metadata_tags": plan.get("metadata_tags", []),
            "plan_reasoning": plan.get("reasoning", []),
            "dynamic_paths": plan.get("dynamic_paths", []),
        }
        if summary:
            entry["summary"] = summary
        log_path = os.path.join(LOG_DIR, "context_enrichment", "plan_snapshots.jsonl")
        self._log_jsonl(log_path, entry)
            
    def repair_contests_with_context(self, contests, context_library=None, db_service=None, parent_context=None, embedding_model=None, logs=None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Use ContextOrganizer.suggest_and_apply_fixes to robustly fill missing fields in contests.
        Returns (fixed_contests, fix_log).
        """
        context_library = context_library or (self.organizer.library if hasattr(self.organizer, "library") else {})
        embedding_model = embedding_model or self._semantic_model
        try:
            return ContextOrganizer.suggest_and_apply_fixes(
                contests,
                context_library,
                logs=logs,
                min_confidence=0.85,
                embedding_model=embedding_model,
                db_service=db_service or self.data_service,
                parent_context=parent_context
            )
        except Exception as e:
            logger.error(f"[repair_contests_with_context] Failed: {e}", exc_info=True)
            return contests, [{"error": str(e)}]
        
    # --- Monitoring, Reporting, and CLI ---            
    def start_alert_monitoring(self, background=True) -> Optional[threading.Thread]:
        """
        Start real-time alert monitoring, optionally in a background thread.
        Returns the thread object if background=True, otherwise None.
        """
        def run_monitor() -> None:
            try:
                monitor_db_for_alerts()
            except Exception as e:
                logger.error(f"[ALERT MONITOR] Exception: {e}", exc_info=True)

        if background:
            if self.alert_monitor_thread and self.alert_monitor_thread.is_alive():
                logger.info("[ALERT MONITOR] Already running.")
                return self.alert_monitor_thread
            t = threading.Thread(target=run_monitor, daemon=True)
            t.start()
            self.alert_monitor_thread = t
            logger.info("[ALERT MONITOR] Started in background thread.")
            return t
        else:
            logger.info("[ALERT MONITOR] Running in foreground (blocking).")
            run_monitor()
            return None

    def report_summary(self) -> None:
        contests = self.get_contests()
        logger.info(f"[bold cyan][COORDINATOR] {len(contests)} contests loaded[/bold cyan]")
        all_entities = set()
        all_labels = set()
        for c in contests:
            if not isinstance(c, dict) or "entities" not in c:
                continue
            for ent, label in c.get("entities", []):
                all_entities.add(ent)
                all_labels.add(label)
        logger.info(f"Unique entity labels: {sorted(all_labels)}")
        logger.info(f"Unique entities: {sorted(all_entities)}")
        logger.info(f"States: {sorted({c.get('state') for c in contests if c.get('state')})}")
        logger.info(f"Years: {sorted({c.get('year') for c in contests if c.get('year')})}")
        issues = self.validate_and_check_integrity()
        if issues["integrity_issues"]:
            logger.warning(f"[yellow]Integrity issues:[/yellow] {issues['integrity_issues']}")
        if issues["anomalies"]:
            logger.error(f"[red]Anomalies detected:[/red] {issues['anomalies']}")

    def _log_jsonl(self, log_path, log_entry):
        """Centralized JSONL logging utility."""
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "ab") as f:
            f.write(orjson.dumps(clean_for_json(log_entry)) + b"\n")

    def _safe_get(self, dct, key, default=None) -> Optional[Any]:
        """Safely get a key from a dict, returning default if not a dict or key missing."""
        return dct.get(key, default) if isinstance(dct, dict) else default

    def update_db_with_context(
            self,
            library: dict,
            db_path: str = None,
            enhanced: bool = True,
            update_tables: bool = True,
            update_contests: bool = True,
            update_panels: bool = True,
            update_buttons: bool = True,
            update_candidates: bool = True,
            update_parties: bool = True,
            update_offices: bool = True,
            update_districts: bool = True,
            update_results: bool = True,
            update_entities: bool = True,
            update_table_structures: bool = True,
            update_batch_metadata: bool = True,
            update_alerts: bool = True,
            update_embeddings: bool = True,
            log_success: bool = True
        ) -> None:
        """
        Robustly update the database with the provided context library.
        Supports batch updates for contests, table structures, panels, buttons, candidates, parties, offices, districts, results, entities, batch metadata, alerts, and embeddings.
        Uses ElectionDataService for all DB operations.
        """
        import os

        # Determine DB path if not provided
        if not db_path:
            db_path = CONTEXT_LIBRARY_PATH
        db_path = os.path.abspath(db_path)

        try:
            # --- Update contests ---
            if update_contests and "contests" in library:
                for contest in library["contests"]:
                    contest_dict = contest if isinstance(contest, dict) else {}
                    try:
                        self.data_service.upsert_contest(contest_dict)
                    except Exception as e:
                        logger.error(f"[update_db_with_context] Failed to upsert contest: {contest_dict.get('title', '')} - {e}")

            # --- Update table structures (legacy and ML-inferred) ---
            if update_tables and "tables" in library:
                tables_dict = library["tables"] if isinstance(library.get("tables"), dict) else {}
                for contest, tables in tables_dict.items():
                    for tbl in tables:
                        tbl_dict = tbl if isinstance(tbl, dict) else {}
                        headers = tbl_dict.get("headers") or tbl_dict.get("columns") or []
                        context = tbl_dict.get("context") or {}
                        ml_confidence = tbl_dict.get("ml_confidence")
                        confirmed_by_user = tbl_dict.get("confirmed_by_user", False)
                        try:
                            self.save_table_structure_to_db(
                                contest, headers, context, ml_confidence, confirmed_by_user
                            )
                        except Exception as e:
                            contest_title = contest.get("title", "") if isinstance(contest, dict) else str(contest)
                            logger.error(f"[update_db_with_context] Failed to save table structure for {contest_title}: {e}")

            # --- Update panels ---
            if update_panels and "panels" in library:
                panels_dict = library["panels"] if isinstance(library.get("panels"), dict) else {}
                for contest, panel in panels_dict.items():
                    contest_dict = contest if isinstance(contest, dict) else contest
                    panel_dict = panel if isinstance(panel, dict) else panel
                    try:
                        self.data_service.upsert_panel(contest_dict, panel_dict)
                    except Exception as e:
                        contest_title = contest_dict.get("title", "") if isinstance(contest_dict, dict) else str(contest_dict)
                        logger.error(f"[update_db_with_context] Failed to upsert panel for {contest_title}: {e}")

            # --- Update buttons ---
            if update_buttons and "buttons" in library:
                buttons_dict = library["buttons"] if isinstance(library.get("buttons"), dict) else {}
                for contest, buttons in buttons_dict.items():
                    contest_dict = contest if isinstance(contest, dict) else contest
                    for btn in buttons:
                        btn_dict = btn if isinstance(btn, dict) else btn
                        try:
                            self.data_service.upsert_button(contest_dict, btn_dict)
                        except Exception as e:
                            contest_title = contest_dict.get("title", "") if isinstance(contest_dict, dict) else str(contest_dict)
                            logger.error(f"[update_db_with_context] Failed to upsert button for {contest_title}: {e}")

            # --- Update candidates ---
            if update_candidates and "candidates" in library:
                for candidate in library["candidates"]:
                    candidate_dict = candidate if isinstance(candidate, dict) else {}
                    try:
                        self.data_service.upsert_candidate(candidate_dict)
                    except Exception as e:
                        logger.error(f"[update_db_with_context] Failed to upsert candidate: {candidate_dict.get('name', '')} - {e}")

            # --- Update parties ---
            if update_parties and "parties" in library:
                for party in library["parties"]:
                    party_dict = party if isinstance(party, dict) else {}
                    try:
                        self.data_service.upsert_party(party_dict)
                    except Exception as e:
                        logger.error(f"[update_db_with_context] Failed to upsert party: {party_dict.get('name', '')} - {e}")

            # --- Update offices ---
            if update_offices and "offices" in library:
                for office in library["offices"]:
                    office_dict = office if isinstance(office, dict) else {}
                    try:
                        self.data_service.upsert_office(office_dict)
                    except Exception as e:
                        logger.error(f"[update_db_with_context] Failed to upsert office: {office_dict.get('name', '')} - {e}")

            # --- Update districts ---
            if update_districts and "districts" in library:
                for district in library["districts"]:
                    district_dict = district if isinstance(district, dict) else {}
                    try:
                        self.data_service.upsert_district(district_dict)
                    except Exception as e:
                        logger.error(f"[update_db_with_context] Failed to upsert district: {district_dict.get('name', '')} - {e}")

            # --- Update results ---
            if update_results and "results" in library:
                for result in library["results"]:
                    result_dict = result if isinstance(result, dict) else {}
                    try:
                        self.data_service.upsert_result(result_dict)
                    except Exception as e:
                        logger.error(f"[update_db_with_context] Failed to upsert result: {result_dict.get('id', '')} - {e}")

            # --- Update entities (generic/misc entities) ---
            if update_entities and "entities" in library:
                for entity in library["entities"]:
                    entity_dict = entity if isinstance(entity, dict) else {}
                    try:
                        self.data_service.upsert_entity(entity_dict)
                    except Exception as e:
                        logger.error(f"[update_db_with_context] Failed to upsert entity: {entity_dict.get('value', '')} - {e}")

            # --- Update table_structures (ML-inferred/user-confirmed) ---
            if update_table_structures and "table_structures" in library:
                for ts in library["table_structures"]:
                    ts_dict = ts if isinstance(ts, dict) else {}
                    try:
                        self.data_service.upsert_table_structure(ts_dict)
                    except Exception as e:
                        logger.error(f"[update_db_with_context] Failed to upsert table_structure: {ts_dict.get('contest', '')} - {e}")

            # --- Update batch_metadata ---
            if update_batch_metadata and "batch_metadata" in library:
                for batch in library["batch_metadata"]:
                    batch_dict = batch if isinstance(batch, dict) else {}
                    try:
                        self.data_service.upsert_batch_metadata(batch_dict)
                    except Exception as e:
                        logger.error(f"[update_db_with_context] Failed to upsert batch_metadata: {batch_dict.get('batch_id', '')} - {e}")

            # --- Update alerts ---
            if update_alerts and "alerts" in library:
                for alert in library["alerts"]:
                    alert_dict = alert if isinstance(alert, dict) else {}
                    try:
                        self.data_service.upsert_alert(alert_dict)
                    except Exception as e:
                        logger.error(f"[update_db_with_context] Failed to upsert alert: {alert_dict.get('id', '')} - {e}")

            # --- Update embeddings (ML segment cache) ---
            if update_embeddings and "embeddings" in library:
                for emb in library["embeddings"]:
                    emb_dict = emb if isinstance(emb, dict) else {}
                    try:
                        self.data_service.upsert_embedding(emb_dict)
                    except Exception as e:
                        logger.error(f"[update_db_with_context] Failed to upsert embedding: {emb_dict.get('segment_hash', '')} - {e}")

            # --- Save the full library as a backup/atomic write ---
            if enhanced:
                try:
                    atomic_write_json(library, db_path)
                except Exception as e:
                    logger.error(f"[update_db_with_context] Failed to atomic write JSON: {e}")

            if log_success:
                logger.info(f"[update_db_with_context] Database updated at {db_path}")

        except Exception as e:
            logger.error(f"[update_db_with_context] Failed to update DB: {e}")

    def save_table_structure_to_db(
        self,
        contest: Dict[str, Any],
        headers: Dict[str, Any],
        context: Dict[str, Any],
        ml_confidence: Optional[float] = None,
        confirmed_by_user: bool = False
    ) -> dict:
        """
        Save or update a table structure for a contest in the database using ElectionDataService.
        Returns a dict with 'success' (bool), 'result' (any returned object), and 'error' (if any).
        """
        try:
            result = self.data_service.save_table_structure(
                contest, headers, context, ml_confidence, confirmed_by_user
            )
            logger.info(f"[ContextCoordinator] Saved table structure for contest: {contest} | Result: {result}")
            return {"success": True, "result": result, "error": None}
        except Exception as e:
            logger.error(f"[ContextCoordinator] Failed to save table structure: {e}", exc_info=True)
            return {"success": False, "result": None, "error": str(e)}

    def get_table_structure_from_db(
        self,
        contest: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve the best-matching table structure for a contest from the database using ElectionDataService.
        Returns a dict with headers, context, and ml_confidence, or None if not found.
        """
        try:
            result = self.data_service.get_table_structure(contest, context)
            if result:
                logger.info(f"[ContextCoordinator] Loaded table structure for contest: {contest}")
            else:
                logger.warning(f"[ContextCoordinator] No table structure found for contest: {contest}")
            return result
        except Exception as e:
            logger.error(f"[ContextCoordinator] Failed to load table structure: {e}", exc_info=True)
            return None

    def organize_and_enrich(self, raw_context, **kwargs) -> Dict[str, Any]:
        self.last_raw_context = raw_context
        overrides = kwargs.pop("route_overrides", None)
        provided_plan = kwargs.pop("enrichment_plan", None)
        enrichment_plan = provided_plan or self._build_enrichment_plan(raw_context, overrides=overrides)
        if enrichment_plan:
            kwargs["enrichment_plan"] = enrichment_plan
        result = self.organizer.organize_context(raw_context, **kwargs)
        # Defensive: handle error dict or None
        if result is None:
            self.organized = {}
            self._log_enrichment_snapshot(enrichment_plan, self.organized, summary=None)
            return self.organized
        if isinstance(result, dict) and "organized" in result:
            self.organized = result["organized"] if result["organized"] is not None else {}
        else:
            self.organized = result if isinstance(result, dict) else {}
        self._enrich_contests_with_nlp()
        summary = result.get("summary") if isinstance(result, dict) else None
        self._log_enrichment_snapshot(enrichment_plan, self.organized, summary=summary)
        return self.organized

    def organize_context_advanced(self, raw_context, **kwargs) -> Dict[str, Any]:
        """
        Call ContextOrganizer.organize_context with all advanced options.
        Returns the full result dict (including log, summary, error).
        """
        try:
            result = self.organizer.organize_context(raw_context, **kwargs)
            # Optionally update self.organized if desired:
            if isinstance(result, dict) and "organized" in result:
                self.organized = result["organized"]
            return result
        except Exception as e:
            logger.error(f"[organize_context_advanced] Failed: {e}", exc_info=True)
            return {"error": str(e)}

    def get_feedback_log(self, log_dir=None, min_count=2, deduplicate=True) -> dict:
        """
        Aggregate and analyze user feedback logs for advanced suggestions.
        Returns a dict with stats on removed/renamed columns, header corrections, and other feedback.
        - log_dir: Optionally override the log directory.
        - min_count: Minimum number of occurrences to consider a feedback significant.
        - deduplicate: Whether to deduplicate by normalized header names.
        """
        log_dir = log_dir or LOG_DIR
        feedback = {
            "removed_columns": defaultdict(int),
            "renamed_columns": defaultdict(str),
            "header_corrections": defaultdict(list),
            "structure_denials": defaultdict(int),
            "raw_entries": [],
        }

        # --- Scan removed_columns_cache.json ---
        removed_path = os.path.join(log_dir, "removed_columns_cache.json")
        if os.path.exists(removed_path):
            try:
                with open(removed_path, "rb") as f:
                    removed_data = orjson.loads(f.read())
                for contest, cols in safe_items(removed_data):
                    for col, count in safe_items(cols):
                        col_norm = safe_lower(safe_strip(safe_replace(col, " ", "")))
                        feedback["removed_columns"][col_norm] += count
            except Exception as e:
                logger.error(f"[get_feedback_log] Failed to load removed_columns_cache: {e}")

        # --- Scan denied_table_structures.json ---
        denied_path = os.path.join(log_dir, "denied_table_structures.json")
        if os.path.exists(denied_path):
            try:
                with open(denied_path, "rb") as f:
                    denied_data = orjson.loads(f.read())
                for sig, count in safe_items(denied_data):
                    feedback["structure_denials"][sig] += count
            except Exception as e:
                logger.error(f"[get_feedback_log] Failed to load denied_table_structures: {e}")

        # --- Scan table_structure_learning_log.jsonl for header corrections ---
        ts_log_path = os.path.join(log_dir, "table_structure_learning_log.jsonl")
        if os.path.exists(ts_log_path):
            try:
                with open(ts_log_path, "rb") as f:
                    for line in f:
                        try:
                            entry = orjson.loads(line)
                            safe_append(feedback["raw_entries"], entry, logger)
                            headers = safe_get(entry, "headers", [])
                            context = safe_get(entry, "context", {})
                            contest = safe_get(entry, "contest", "")
                            for h in headers:
                                h_norm = safe_lower(safe_strip(safe_replace(h, " ", "")))
                                safe_append(
                                    feedback["header_corrections"][h_norm],
                                    {"contest": contest, "context": context, "raw": h},
                                    logger
                                )
                        except Exception:
                            continue
            except Exception as e:
                logger.error(f"[get_feedback_log] Failed to load table_structure_learning_log: {e}")

        # --- Scan for renamed columns if possible (from feedback pattern KB or other logs) ---
        if hasattr(self, "get_feedback_pattern_kb"):
            pattern_kb = self.get_feedback_pattern_kb()
            for entry in pattern_kb:
                old = safe_get(entry, "old_header")
                new = safe_get(entry, "new_header")
                if old and new:
                    old_norm = safe_lower(safe_strip(safe_replace(old, " ", "")))
                    feedback["renamed_columns"][old_norm] = new

        # --- Filter by min_count and deduplicate ---
        def filter_dict(d, min_count):
            return {k: v for k, v in safe_items(d) if isinstance(v, int) and v >= min_count}

        feedback["removed_columns"] = filter_dict(feedback["removed_columns"], min_count)
        feedback["structure_denials"] = filter_dict(feedback["structure_denials"], min_count)

        # Optionally deduplicate header corrections and renamed columns
        if deduplicate:
            feedback["header_corrections"] = {
                k: {frozenset((frozenset(item.items()) if isinstance(item, dict) else item) for item in v)}
                for k, v in safe_items(feedback["header_corrections"])
            }
            feedback["renamed_columns"] = dict(feedback["renamed_columns"])

        return dict(feedback)
 
    def get_feedback_pattern_kb(self, log_path=None, deduplicate=True, min_fields=("pattern_id", "label", "html")) -> list:
        """
        Load and return feedback pattern KB entries from the feedback log.
        - log_path: Optional path override (defaults to segment_feedback_log.jsonl in LOG_DIR)
        - deduplicate: If True, deduplicate by pattern_id or segment_hash
        - min_fields: Tuple of required fields for a valid entry
        Returns a list of dicts, each representing a feedback KB entry.
        """
        if log_path is None:
            log_path = os.path.join(LOG_DIR, "segment_feedback_log.jsonl")
        entries = []
        seen = set()
        if not os.path.exists(log_path):
            logger.info(f"[get_feedback_pattern_kb] Feedback log not found: {log_path}")
            return []
        with open(log_path, "rb") as f:
            for line in f:
                try:
                    entry = orjson.loads(line)
                    # Defensive: must be a dict and have required fields
                    if not isinstance(entry, dict):
                        continue
                    if not all(field in entry and entry[field] for field in min_fields):
                        continue
                    # Defensive: deduplicate by pattern_id or segment_hash if requested
                    dedup_key = entry.get("pattern_id") or entry.get("segment_hash")
                    if deduplicate and dedup_key:
                        if dedup_key in seen:
                            continue
                        seen.add(dedup_key)
                    # Defensive: ensure embedding is a list (not np.ndarray or None)
                    emb = entry.get("embedding")
                    if emb is not None and not isinstance(emb, list):
                        try:
                            entry["embedding"] = list(emb)
                        except Exception:
                            entry["embedding"] = []
                    # Defensive: ensure html is a string
                    if not isinstance(entry.get("html", ""), str):
                        entry["html"] = str(entry.get("html", ""))
                    entries.append(entry)
                except Exception as e:
                    logger.warning(f"[get_feedback_pattern_kb] Skipping corrupt line: {e}")
                    continue
        logger.info(f"[get_feedback_pattern_kb] Loaded {len(entries)} feedback KB entries from log.")
        return entries
    
    def auto_label_segment(
        self,
        segment,
        context_library=None,
        context_cache=None,
        pattern_kb=None,
        model=None,
        ml_threshold=0.7
    ) -> str:
        """
        ML-driven DOM segment labeling using all available context, cache, DOM grouping, and heuristics.
        Uses context library, DOM parts, pattern KB, and semantic model for robust labeling.
        """
        context_library = context_library or getattr(self, "library", None)
        context_cache = context_cache or getattr(self, "context_cache", None)
        pattern_kb = pattern_kb or getattr(self, "pattern_kb", None)
        model = model or getattr(self, "_semantic_model", None)

        segment_dict = segment if isinstance(segment, dict) else {}
        segment_hash = segment_dict.get("segment_hash")
        if context_library and segment_hash:
            cached_segments = context_library.get("cached_segments", []) if isinstance(context_library, dict) else []
            for entry in cached_segments:
                entry_dict = entry if isinstance(entry, dict) else {}
                if entry_dict.get("segment_hash") == segment_hash and entry_dict.get("ml_label"):
                    return entry_dict["ml_label"]

        if pattern_kb and model and "html" in segment_dict:
            seg_emb = get_segment_embedding(model, segment_dict)
            if seg_emb is not None:
                best_score = 0
                best_label = None
                for pat in pattern_kb:
                    pat_dict = pat if isinstance(pat, dict) else {}
                    pat_emb = pat_dict.get("embedding")
                    if pat_emb is not None:
                        score = float(np.dot(seg_emb, pat_emb) / (np.linalg.norm(seg_emb) * np.linalg.norm(pat_emb)))
                        if score > best_score and score >= ml_threshold:
                            best_score = score
                            best_label = pat_dict.get("label")
                if best_label:
                    return best_label

        dom_parts = self.get_dom_parts()
        if dom_parts and "all_nodes" in dom_parts:
            all_nodes = dom_parts["all_nodes"]
            for node in all_nodes:
                node_dict = node if isinstance(node, dict) else {}
                if node_dict.get("html") == segment_dict.get("html") and node_dict.get("ml_label") and node_dict.get("ml_confidence", 0) >= ml_threshold:
                    return node_dict["ml_label"]
            grouped = self.group_dom_nodes_by_label(label_field="ml_label")
            for label, nodes in grouped.items():
                for node in nodes:
                    node_dict = node if isinstance(node, dict) else {}
                    if node_dict.get("html") == segment_dict.get("html"):
                        return label

        if "label" in segment_dict or "selector" in segment_dict:
            candidates = [segment_dict]
            ranked = merge_and_rank_candidates([], candidates, {}, [segment_dict.get("label", "")], model)
            if ranked:
                ranked_candidate = safe_get_first(ranked, "ranked_candidate", None, logger)
                if ranked_candidate and ranked_candidate.get("combined_score", 0) >= ml_threshold:
                    return ranked_candidate.get("label")

        if "html" in segment_dict:
            label = self.extract_field("panel", text=segment_dict.get("html"))
            if label:
                return label

        return "unknown"

    def segment_prompt(self, segment, session_id=None, reason=None):
        """
        Robust segment prompt for webapp GUI and CLI.
        - Prompts user for label/correction.
        - Optionally logs feedback for downstream learning.
        """
        logger.info(f"[SEGMENT_PROMPT] Segment needs review. Reason: {reason}. Session: {session_id}")
        html_preview = safe_get(segment, "html", "")
        # Interactive prompt (CLI or webapp)
        logger.info(f"[SEGMENT_PROMPT][interactive] Segment HTML: {html_preview[:200]}{'...' if len(html_preview) > 200 else ''}")
        label = None
        try:
            label = input("Please enter the semantic label for this segment: ").strip()
        except Exception:
            label = "unknown"
        self.log_field_selection(
            field_type="segment",
            field_name="segment_prompt",
            extracted_value=html_preview,
            method="interactive",
            score=1.0,
            result=reason,
            context={"session_id": session_id, "reason": reason},
            user_feedback=label
        )
        return label

    def group_dom_nodes_by_label(self, label_field="ml_label") -> Dict[str, List[Dict[str, Any]]]:
        """
        Group DOM nodes by a given label field using ContextOrganizer utility.
        Returns a dict mapping label values to lists of nodes.
        """
        if not self.organized or "dom_parts" not in self.organized:
            ContextCoordinator._dom_parts_warning_count += 1
            if ContextCoordinator._dom_parts_warning_count == 1:
                logger.warning("[group_dom_nodes_by_label] No organized DOM parts. (Further warnings suppressed)")
            elif ContextCoordinator._dom_parts_warning_count % 10 == 0:
                logger.warning(f"[group_dom_nodes_by_label] No organized DOM parts. (Occurred {ContextCoordinator._dom_parts_warning_count} times)")
            return {}
        dom_parts = self.organized.get("dom_parts", {}) if isinstance(self.organized, dict) else {}
        nodes = dom_parts.get("all_nodes", [])
        if not nodes:
            logger.warning("[group_dom_nodes_by_label] No DOM nodes found.")
            return {}
        try:
            return self.organizer.group_nodes_by_label(nodes, label_field=label_field)
        except Exception as e:
            logger.error(f"[group_dom_nodes_by_label] Failed: {e}", exc_info=True)
            return {}

    # --- Feedback, Learning, and Correction ---
    def submit_user_feedback(self, field_type, field_name, correct_value, context) -> Dict[str, Any]:
        """
        Submit user feedback for a field extraction/correction.
        Robust: checks for method existence, logs errors, and returns updated organized context.
        """
        try:
            if hasattr(self.organizer, "submit_user_feedback"):
                self.organizer.submit_user_feedback(field_type, field_name, correct_value, context)
            else:
                logger.warning("[submit_user_feedback] ContextOrganizer has no submit_user_feedback method.")
            self._enrich_contests_with_nlp()
            # Optionally log the feedback event
            self.log_field_selection(
                field_type=field_type,
                field_name=field_name,
                extracted_value=correct_value,
                method="user_feedback",
                score=1.0,
                result="user_feedback",
                context=context,
                user_feedback=correct_value
            )
        except Exception as e:
            logger.error(f"[submit_user_feedback] Failed to submit feedback: {e}", exc_info=True)
        return self.organized
    
    def correct_and_update_contest(self, contest_id, correction_data, validate_types=True, log_changes=True) -> None:
        """
        Advanced contest correction:
        - Syncs and validates types.
        - Logs changes and type corrections.
        - Optionally validates and reports type consistency.
        """
        contest = {"id": contest_id, **correction_data}
        sync_type_and_election_types(contest)
        if validate_types:
            if not contest.get("type_") or not contest.get("election_types"):
                logger.warning(f"[correct_and_update_contest] Contest {contest_id} missing type/election_types after sync.")
        try:
            self.data_service.update_contest_in_db(contest)
            self.organized = None
            self.organize_and_enrich(self.last_raw_context)
            if log_changes:
                self.organizer.log_field_selection(
                    field_type="contest",
                    field_name="correction",
                    extracted_value=correction_data,
                    method="manual",
                    score=1.0,
                    result="manual_pass",
                    context={"contest_id": contest_id, "type_": contest.get("type_"), "election_types": contest.get("election_types")},
                    user_feedback=None
                )
        except Exception as e:
            logger.error(f"[correct_and_update_contest] Failed to update contest: {e}", exc_info=True)

    def print_contest_summary(self) -> None:
        """
        Print a summary table of contests by state/county using ContextOrganizer.
        """
        if not self.organized or "contests" not in self.organized:
            logger.warning("[print_contest_summary] No organized contests to summarize.")
            return
        contests = self.organized["contests"]
        try:
            self.organizer.print_contest_summary(contests)
        except Exception as e:
            logger.error(f"[print_contest_summary] Failed: {e}", exc_info=True)

    def plot_contest_distribution(self) -> None:
        """
        Plot contest count by state/county using ContextOrganizer.
        """
        if not self.organized or "contests" not in self.organized:
            logger.warning("[plot_contest_distribution] No organized contests to plot.")
            return
        contests = self.organized["contests"]
        try:
            self.organizer.plot_contest_distribution(contests)
        except Exception as e:
            logger.error(f"[plot_contest_distribution] Failed: {e}", exc_info=True)

    def get_known_state_to_county_map(self) -> List[str]:
        """
        Return all known states (keys) from the canonical state-to-county mapping in librarian.py.
        """
        return list(KNOWN_STATE_TO_COUNTY_MAP.keys())

    def get_known_county_to_PRECINCTS_map(self) -> List[str]:
        """
        Return all known counties (keys) from the canonical county-to-precinct mapping in librarian.py.
        """
        return list(KNOWN_COUNTY_TO_PRECINCTS_MAP.keys())

    def get_known_states(self) -> List[str]:
        """
        Return all known states from the canonical mapping in librarian.py.
        """
        # STATE_MODULE_MAP keys are already normalized (snake_case)
        return list(STATE_MODULE_MAP.keys())

    def get_known_counties(self, state=None) -> List[str]:
        """
        Return all known counties from the canonical mapping in librarian.py.
        If a state is provided, return counties for that state only.
        """
        if state:
            state_norm = normalize_state_name(state)
            return KNOWN_STATE_TO_COUNTY_MAP.get(state_norm, [])
        # Flatten all counties if no state is specified
        counties = []
        for county_list in KNOWN_STATE_TO_COUNTY_MAP.values():
            counties.extend(county_list)
        return counties

    def get_dom_parts(self) -> Dict[str, Any]:
        """
        Return organized DOM parts (head, body, wrappers, tables, buttons, clickable, etc.).
        Suppress repeated warnings after a limit.
        """
        NO_DOM_WARNING_LIMIT = 5
        if not hasattr(self, "_no_dom_warning_count"):
            self._no_dom_warning_count = 0
        if not self.organized or "dom_parts" not in self.organized:
            if self._no_dom_warning_count < NO_DOM_WARNING_LIMIT:
                logger.warning("No organized DOM parts.")
                self._no_dom_warning_count += 1
            elif self._no_dom_warning_count == NO_DOM_WARNING_LIMIT:
                logger.warning("No organized DOM parts. (Further warnings suppressed)")
                self._no_dom_warning_count += 1
            # else: suppress further warnings
            return {}
        return self.organized["dom_parts"]

    def get_contest_groups(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Return contest groups by keyword (from ContextOrganizer).
        """
        if not self.organized or "contest_groups" not in self.organized:
            logger.warning("[get_contest_groups] No contest groups found.")
            return {}
        return self.organized["contest_groups"]

    def get_panel_groups(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Return panel groups by keyword (from ContextOrganizer).
        """
        if not self.organized or "panel_groups" not in self.organized:
            logger.warning("[get_panel_groups] No panel groups found.")
            return {}
        return self.organized["panel_groups"]

    def get_button_groups(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Return button groups by keyword (from ContextOrganizer).
        """
        if not self.organized or "button_groups" not in self.organized:
            logger.warning("[get_button_groups] No button groups found.")
            return {}
        return self.organized["button_groups"]

    def get_table_groups(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Return table groups by keyword (from ContextOrganizer).
        """
        if not self.organized or "table_groups" not in self.organized:
            logger.warning("[get_table_groups] No table groups found.")
            return {}
        return self.organized["table_groups"]

    def get_relationships(self) -> Dict[str, Any]:
        """
        Return party/candidate/district/state/county relationships.
        """
        if not self.organized:
            logger.warning("[get_relationships] No organized context.")
            return {}
        return {
            "party_to_candidates": self.organized.get("party_to_candidates", {}),
            "candidate_to_party": self.organized.get("candidate_to_party", {}),
            "candidate_to_district": self.organized.get("candidate_to_district", {}),
            "district_to_candidates": self.organized.get("district_to_candidates", {}),
            "state_to_counties": self.organized.get("state_to_counties", {}),
            "county_to_state": self.organized.get("county_to_state", {}),
        }

    def _enrich_contests_with_nlp(self, batch=True, sync_types=True, log_enrichment=True) -> None:
        """
        Advanced enrichment:
        - Batch processes contests for efficiency if batch=True.
        - Syncs types and logs enrichment.
        - Handles errors gracefully.
        """
        if not self.organized or "contests" not in self.organized:
            return
        contests = self.organized["contests"]
        if batch:
            # Batch process all contests at once for NLP extraction
            titles = [c.get("title", "") if isinstance(c, dict) else "" for c in contests]
            try:
                all_entities = [extract_entities(title) for title in titles]
                all_locations = [extract_locations(title) for title in titles]
                all_dates = [extract_dates(title) for title in titles]
                for idx, c in enumerate(contests):
                    if not isinstance(c, dict):
                        continue
                    c["entities"] = all_entities[idx]
                    c["locations"] = all_locations[idx]
                    c["dates"] = all_dates[idx]
                    if sync_types:
                        sync_type_and_election_types(c)
                    if log_enrichment and idx < 5:
                        logger.info(f"[enrich_contests_with_nlp] Enriched contest: {c.get('title', '')}")
            except Exception as e:
                logger.error(f"[enrich_contests_with_nlp] Batch error: {e}", exc_info=True)
        else:
            # Process contests one by one
            for idx, c in enumerate(contests):
                if not isinstance(c, dict):
                    continue
                try:
                    title = c.get("title", "")
                    c["entities"] = extract_entities(title)
                    c["locations"] = extract_locations(title)
                    c["dates"] = extract_dates(title)
                    if sync_types:
                        sync_type_and_election_types(c)
                    if log_enrichment and idx < 5:
                        logger.info(f"[enrich_contests_with_nlp] Enriched contest: {title}")
                except Exception as e:
                    logger.error(f"[enrich_contests_with_nlp] Error enriching contest '{c.get('title', '')}': {e}", exc_info=True)

    def fuzzy_score(self, a: str, b: str) -> float:
        """
        Compute a robust fuzzy string similarity score between two strings.
        Uses semantic model (if available), multiple rapidfuzz metrics, difflib, and aggregates for best result.
        Returns a float between 0.0 and 1.0.
        """

        def _is_numeric_sequence(seq) -> bool:
            # Robustly check if a sequence is numeric (list, tuple, np.ndarray, pandas.Series, etc.)
            if isinstance(seq, (list, tuple)):
                return all(isinstance(x, numbers.Number) for x in seq)
            # Check for numpy arrays or pandas Series/DataFrame columns
            if hasattr(seq, "dtype"):
                try:
                    # Defensive: only call .dtype if it's a type or np.dtype, not a method
                    dtype = getattr(seq, "dtype", None)
                    # dtype should be a type or np.dtype, not callable
                    if dtype is not None and not callable(dtype):
                        return np.issubdtype(dtype, np.number)
                except Exception:
                    return False
            return False

        def _to_numeric_array(seq) -> Optional[np.ndarray]:
            # Flattens and filters to only numeric values, robust to dtype errors
            if isinstance(seq, (list, tuple)):
                seq = [x for x in seq if isinstance(x, numbers.Number)]
            try:
                if _is_numeric_sequence(seq):
                    arr = np.array(seq, dtype=np.float32)
                elif hasattr(seq, "tolist"):
                    seq_list = safe_tolist(seq)
                    if _is_numeric_sequence(seq_list):
                        arr = np.array(seq_list, dtype=np.float32)
                    else:
                        return None
                else:
                    return None
                if arr.ndim != 1 or arr.size == 0 or np.any(np.isnan(arr)):
                    return None
                return arr
            except Exception as e:
                logger.error(f"[fuzzy_score] Could not convert embedding to numeric array: {e} | seq={seq}")
                return None

        try:
            def _normalize(s) -> str:
                return " ".join(str(s).strip().lower().split()) if s is not None else ""
            a_str = _normalize(a)
            b_str = _normalize(b)
            if not a_str or not b_str:
                logger.warning(f"[fuzzy_score] One or both inputs are empty: a='{a_str}', b='{b_str}'")
                return 0.0
            if a_str == b_str:
                logger.debug(f"[fuzzy_score] Exact match for '{a_str}' and '{b_str}'")
                return 1.0
            if len(a_str) < 2 or len(b_str) < 2:
                logger.warning(f"[fuzzy_score] One or both inputs are too short: a='{a_str}', b='{b_str}'")
                return 0.0

            scores = []
            model = getattr(self, "_semantic_model", None)
            # 1. Semantic model similarity (if available)
            if model is not None:
                try:
                    if hasattr(model, "similarity"):
                        sim = safe_similarity(model, a_str, b_str, logger)
                        if isinstance(sim, (float, int)):
                            logger.debug(f"[fuzzy_score] Used model.similarity: {sim}")
                            scores.append(float(sim))
                    elif hasattr(model, "encode"):
                        emb_a = safe_model_encode(model, [a_str])
                        emb_b = safe_model_encode(model, [b_str])
                        if emb_a is not None and emb_b is not None:
                            emb_a = emb_a[0] if isinstance(emb_a, (list, tuple)) else emb_a
                            emb_b = emb_b[0] if isinstance(emb_b, (list, tuple)) else emb_b
                            # Use robust numeric check before conversion
                            emb_a_np = _to_numeric_array(emb_a)
                            emb_b_np = _to_numeric_array(emb_b)
                            if emb_a_np is not None and emb_b_np is not None:
                                sim = float(np.dot(emb_a_np, emb_b_np) / (np.linalg.norm(emb_a_np) * np.linalg.norm(emb_b_np) + 1e-8))
                                logger.debug(f"[fuzzy_score] Used model.encode + cosine (np): {sim}")
                                scores.append(sim)
                            else:
                                logger.error(f"[fuzzy_score] Embeddings are not valid numeric arrays: emb_a={type(emb_a)}, emb_b={type(emb_b)}")
                except Exception as e:
                    logger.error(f"[fuzzy_score] Exception in semantic model: {e}", exc_info=True)

            # 2. Rapidfuzz metrics (ratio, partial_ratio, token_sort_ratio, token_set_ratio)
            try:
                rf_ratio = fuzz.ratio(a_str, b_str) / 100.0
                rf_partial = fuzz.partial_ratio(a_str, b_str) / 100.0
                rf_token_sort = fuzz.token_sort_ratio(a_str, b_str) / 100.0
                rf_token_set = fuzz.token_set_ratio(a_str, b_str) / 100.0
                logger.debug(f"[fuzzy_score] rapidfuzz.ratio: {rf_ratio}, partial: {rf_partial}, token_sort: {rf_token_sort}, token_set: {rf_token_set}")
                scores.extend([rf_ratio, rf_partial, rf_token_sort, rf_token_set])
            except Exception as e:
                logger.error(f"[fuzzy_score] Exception in rapidfuzz metrics: {e}", exc_info=True)

            # 3. difflib ratio
            try:
                difflib_score = difflib.SequenceMatcher(None, a_str, b_str).ratio()
                logger.debug(f"[fuzzy_score] Used difflib.SequenceMatcher: {difflib_score}")
                scores.append(difflib_score)
            except Exception as e:
                logger.error(f"[fuzzy_score] Exception in difflib.SequenceMatcher: {e}", exc_info=True)

            # 4. Aggregate: weighted mean, max, and penalize trivial matches
            if scores:
                # Remove any NaN or out-of-bounds scores
                scores = [s for s in scores if isinstance(s, (float, int)) and 0.0 <= s <= 1.0]
                if not scores:
                    logger.error(f"[fuzzy_score] All computed scores were invalid for a='{a_str}', b='{b_str}'")
                    return 0.0
                # Penalize trivial/short/numeric matches
                if len(a_str) <= 2 or len(b_str) <= 2 or a_str.isdigit() or b_str.isdigit():
                    logger.debug(f"[fuzzy_score] Penalizing trivial/short/numeric match: a='{a_str}', b='{b_str}'")
                    return min(max(max(scores) * 0.5, 0.0), 1.0)
                # Weighted mean: semantic (if present) gets higher weight, then rapidfuzz, then difflib
                semantic_scores = scores[:1] if model is not None else []
                rf_scores = scores[1:5] if model is not None else scores[:4]
                dl_score = scores[5] if len(scores) > 5 else (scores[4] if len(scores) > 4 else None)
                weights = []
                vals = []
                if semantic_scores:
                    weights.append(0.35)
                    vals.append(np.mean(semantic_scores))
                if rf_scores:
                    weights.append(0.55)
                    vals.append(np.mean(rf_scores))
                if dl_score is not None:
                    weights.append(0.1)
                    vals.append(dl_score)
                if vals and weights and len(vals) == len(weights):
                    agg = np.average(vals, weights=weights)
                else:
                    agg = np.mean(scores)
                logger.debug(f"[fuzzy_score] Aggregated weighted score: {agg}")
                return min(max(agg, 0.0), 1.0)

            logger.error(f"[fuzzy_score] All methods failed for a='{a_str}', b='{b_str}'")
            return 0.0
        except Exception as e:
            logger.error(f"[fuzzy_score] Unexpected error: {e}", exc_info=True)
            return 0.0

    def log_field_selection(
            self,
            field_type,
            field_name,
            extracted_value,
            method,
            score,
            result,
            context,
            user_feedback=None,
            log_path=None
        ) -> None:
        """
        Log field extraction/correction attempts for ML/feedback.
        Ensures log file is always inside the log/ directory and filename is sanitized.
        Robust to path injection, extension errors, and serialization issues.
        """
        # --- Sanitize and validate filename ---
        safe_field_type = safe_filename(field_type)
        default_filename = f"{safe_field_type}_selection_log.jsonl"
        log_dir = os.path.abspath(LOG_DIR)
        if log_path is None:
            log_path = os.path.join(log_dir, default_filename)
        else:
            # Only use the filename part, sanitize it, and force it into log/
            base = os.path.basename(log_path)
            # Ensure .jsonl extension
            if not safe_endswith(base, ".jsonl", logger):
                base = re.sub(r'(\.jsonl)?$', '', base) + ".jsonl"
            safe_base = safe_filename(base)
            log_path = os.path.join(log_dir, safe_base)
        # Defensive: prevent path traversal
        log_path = os.path.abspath(log_path)
        if not log_path.startswith(log_dir):
            logger.error(f"[log_field_selection] Unsafe log path detected: {log_path}")
            return

        # --- Prepare log entry ---
        try:
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "field_type": field_type,
                "field_name": field_name,
                "extracted_value": extracted_value,
                "method": method,
                "score": float(score) if score is not None else None,
                "result": result,
                "context": context,
                "user_feedback": user_feedback
            }
            # Defensive: clean for JSON, handle serialization errors
            entry_bytes = None
            try:
                entry_bytes = orjson.dumps(clean_for_json(log_entry)) + b"\n"
            except Exception as e:
                logger.error(f"[log_field_selection] Serialization error: {e}")
                # Fallback: try to convert problematic fields to string
                for k, v in log_entry.items():
                    if not isinstance(v, (str, int, float, bool, type(None), dict, list)):
                        log_entry[k] = str(v)
                entry_bytes = orjson.dumps(clean_for_json(log_entry)) + b"\n"
        except Exception as e:
            logger.error(f"[log_field_selection] Failed to prepare log entry: {e}")
            return

        # --- Write to file robustly ---
        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "ab") as f:
                f.write(entry_bytes)
        except Exception as e:
            logger.error(f"[log_field_selection] Failed to write log entry to {log_path}: {e}")

    def extract_entities(self, text, labels=None, first_only=False):
        """
        Unified NLP entity extraction using spaCy.
        - text: input string
        - labels: set or list of entity labels to filter (e.g., {"ORG", "PERSON"})
        - first_only: if True, return only the first match (else all)
        Returns: list of (entity, label) or a single (entity, label) if first_only
        """
        try:
            if not isinstance(text, str) or not text:
                if hasattr(self, "logger"):
                    self.logger.error("[extract_entities] Invalid text type: %s", type(text))
                return [] if not first_only else None
            entities = extract_entities(text)
            if not isinstance(entities, list):
                if hasattr(self, "logger"):
                    self.logger.error(
                        "[extract_entities] extract_entities did not return a list: %s",
                        type(entities),
                    )
                entities = []
            if labels:
                labels_set = set(labels)
                filtered = [(ent, label) for ent, label in entities if label in labels_set]
            else:
                filtered = entities
            if first_only:
                return safe_get_first(filtered, "entity_filtered", None, self.logger if hasattr(self, 'logger') else None) if filtered else None
            return filtered
        except Exception as e:
            if hasattr(self, "logger"):
                self.logger.error("[ContextCoordinator.extract_entities] Error: %s", e)
            return [] if not first_only else None

    def extract_locations(self, text, labels=None, first_only=False):
        """
        Unified location extraction using spaCy.
        - text: input string
        - labels: set or list of entity labels to filter (e.g., {"GPE", "LOC"})
        - first_only: if True, return only the first match (else all)
        Returns: list of (location, label) or a single (location, label) if first_only
        """
        try:
            if not text or not isinstance(text, str):
                return [] if not first_only else None
            locations = extract_locations(text)
            if labels:
                labels_set = set(labels)
                filtered = [(loc, label) for loc, label in locations if label in labels_set]
            else:
                filtered = locations
            if first_only:
                return safe_get_first(filtered, "locations_filtered", None, logger) if filtered else None
            return filtered
        except Exception as e:
            logger.error("[ContextCoordinator.extract_locations] Error: %s", e)
            return [] if not first_only else None

    def extract_dates(self, text, labels=None, first_only=False):
        """
        Unified date extraction using spaCy.
        - text: input string
        - labels: set or list of entity labels to filter (e.g., {"DATE"})
        - first_only: if True, return only the first match (else all)
        Returns: list of (date, label) or a single (date, label) if first_only
        """
        try:
            if not text or not isinstance(text, str):
                return [] if not first_only else None
            dates = extract_dates(text)
            if labels:
                labels_set = set(labels)
                filtered = [(date, label) for date, label in dates if label in labels_set]
            else:
                filtered = dates
            if first_only:
                return safe_get_first(filtered, "dates_filtered", None, logger) if filtered else None
            return filtered
        except Exception as e:
            logger.error("[ContextCoordinator.extract_dates] Error: %s", e)
            return [] if not first_only else None

    def extract_field(self, field_type, text=None, context=None, extra=None):
        """
        Unified extraction for all field types (party, panel, tables, precincts, states, election_types, years, buttons).
        - field_type: str, one of 'party', 'panel', 'tables', 'precincts', 'states', 'election_types', 'years', 'buttons'
        - text: main string to extract from (or list of strings for some types)
        - context: dict, optional, for lookups
        - extra: dict, optional, for additional params (e.g., state/county for precincts)
        Returns: extracted value(s)
        """
        context = context or {}
        extra = extra or {}

        # --- Patterns and lookups ---
        party_pattern = "|".join([re.escape(k) for k in PARTY_KEYWORDS])
        panel_pattern = "|".join([re.escape(k) for k in PANEL_TAGS])
        table_pattern = "|".join([re.escape(k) for k in TABLE_TAGS])
        location_pattern = "|".join([re.escape(k) for k in LOCATION_KEYWORDS])
        state_pattern = "|".join([re.escape(k) for k in STATE_TAGS])
        button_pattern = "|".join([re.escape(k) for k in BUTTON_TAGS])
        election_type_pattern = r"(primary|general|special|runoff|municipal|presidential|senate|mayoral|school board|" + location_pattern + ")"

        # --- Extraction strategies ---
        def regex_party(text):
            match = re.search(rf"({party_pattern})", text, re.IGNORECASE)
            if match:
                return (match.group(1), None, 0.9, "regex", "pass")
            return None

        def nlp_party(text):
            entities = self.extract_entities(text, labels={"ORG", "NORP"})
            known_parties = PARTY_KEYWORDS
            for ent, label in entities:
                best = process.extractOne(ent, known_parties)
                if best and best[1] > 80:
                    return (safe_get_first(best, "fuzzy_best_entity", None, logger), label, best[1] / 100.0, "spacy_ner_fuzzy", "pass")
            return None

        def fuzzy_party(text):
            known_parties = PARTY_KEYWORDS
            best = process.extractOne(text, known_parties)
            if best and best[1] > 80:
                return (safe_get_first(best, "fuzzy_best_entity", None, logger), None, best[1] / 100.0, "fuzzy", "pass")
            return None

        def regex_panel(text):
            match = re.search(rf"({panel_pattern})", text, re.IGNORECASE)
            if match:
                return (match.group(1), None, 0.9, "regex", "pass")
            return None

        def nlp_panel(text):
            entities = self.extract_entities(text, labels={"ORG", "NORP"})
            for ent, label in entities:
                return (ent, label, 0.85, "spacy_ner", "pass")
            return None

        def direct_panel(text):
            panel = self.get_panel(text)
            if panel:
                return (panel, None, 1.0, "direct_lookup", "pass")
            return None

        def regex_table(text):
            match = re.search(rf"({table_pattern})", text, re.IGNORECASE)
            if match:
                return ([match.group(1)], None, 0.9, "regex", "pass")
            return None

        def nlp_table(text):
            entities = self.extract_entities(text, labels={"ORG", "NORP"})
            for ent, label in entities:
                return ([ent], label, 0.85, "spacy_ner", "pass")
            return None

        def direct_table(text):
            tables = self.get_tables(text)
            if tables:
                return (tables, None, 1.0, "direct_lookup", "pass")
            return None

        def regex_precinct(text):
            match = re.search(rf"({location_pattern})", text, re.IGNORECASE)
            if match:
                return ([match.group(1)], None, 0.9, "regex", "pass")
            return None

        def nlp_precinct(text):
            entities = self.extract_entities(text, labels={"ORG", "NORP"})
            for ent, label in entities:
                return ([ent], label, 0.85, "spacy_ner", "pass")
            return None

        def direct_precinct(_):
            state = extra.get("state")
            county = extra.get("county")
            precincts = self.get_precincts(state=state, county=county)
            if precincts:
                return (precincts, None, 1.0, "direct_lookup", "pass")
            return None

        def regex_state(text):
            match = re.search(rf"({state_pattern})", text, re.IGNORECASE)
            if match:
                return (text, None, 0.9, "regex", "pass")
            return None

        def nlp_state(text):
            entities = self.extract_entities(text, labels={"ORG", "NORP"})
            for ent, label in entities:
                return (ent, label, 0.85, "spacy_ner", "pass")
            return None

        def direct_state(_):
            states = self.get_states()
            if states:
                return (states, None, 1.0, "direct_lookup", "pass")
            return None

        def regex_election_type(text):
            match = re.search(election_type_pattern, text, re.IGNORECASE)
            if match:
                return (match.group(1), None, 0.9, "regex", "pass")
            return None

        def nlp_election_type(text):
            entities = self.extract_entities(text, labels={"ORG", "NORP"})
            for ent, label in entities:
                return (ent, label, 0.85, "spacy_ner", "pass")
            return None

        def direct_election_type(_):
            types = self.get_election_types()
            if types:
                return (types, None, 1.0, "direct_lookup", "pass")
            return None

        def regex_year(text):
            match = re.search(r"\b(19|20)\d{2}\b", text)
            if match:
                return (match.group(0), None, 0.9, "regex", "pass")
            return None

        def nlp_year(text):
            entities = self.extract_entities(text, labels={"DATE"})
            for ent, label in entities:
                if re.match(r"\b(19|20)\d{2}\b", ent):
                    return (ent, label, 0.85, "spacy_ner", "pass")
            return None

        def direct_year(_):
            years = self.get_years()
            if years:
                return (years, None, 1.0, "direct_lookup", "pass")
            return None

        def regex_button(text):
            match = re.search(rf"({button_pattern})", text, re.IGNORECASE)
            if match:
                return (match.group(1), None, 0.9, "regex", "pass")
            return None

        def nlp_button(text):
            entities = self.extract_entities(text, labels={"ORG", "NORP"})
            for ent, label in entities:
                return (ent, label, 0.85, "spacy_ner", "pass")
            return None

        def direct_button(_):
            contest = extra.get("contest")
            keyword = extra.get("keyword")
            url = extra.get("url")
            candidates = []
            buttons = self.get_buttons(contest=contest, keyword=keyword, url=url)
            for btn in buttons:
                if not isinstance(btn, dict):
                    continue
                label = btn.get("label")
                if label:
                    candidates.append(label)
            if candidates:
                return (list(dict.fromkeys(candidates)), None, 1.0, "direct_lookup", "pass")
            return None

        # --- Strategy selection ---
        strategies = {
            "party": [("regex", regex_party), ("nlp", nlp_party), ("fuzzy", fuzzy_party)],
            "panel": [("regex", regex_panel), ("nlp", nlp_panel), ("direct_lookup", direct_panel)],
            "tables": [("regex", regex_table), ("nlp", nlp_table), ("direct_lookup", direct_table)],
            "precincts": [("regex", regex_precinct), ("nlp", nlp_precinct), ("direct_lookup", direct_precinct)],
            "states": [("regex", regex_state), ("nlp", nlp_state), ("direct_lookup", direct_state)],
            "election_types": [("regex", regex_election_type), ("nlp", nlp_election_type), ("direct_lookup", direct_election_type)],
            "years": [("regex", regex_year), ("nlp", nlp_year), ("direct_lookup", direct_year)],
            "buttons": [("regex", regex_button), ("nlp", nlp_button), ("direct_lookup", direct_button)],
            # Vote_methods (uses BALLOT_TYPES as the base vocabulary)
            "vote_methods": [("regex", lambda t: (
                [kw for kw in BALLOT_TYPES if isinstance(t, str) and kw.lower() in t.lower()] or None,
                None, 0.85, "regex", "pass"
            ))],
        }

        # --- Extraction ---
        if field_type not in strategies:
            logger.warning(f"[extract_field] Unknown field_type: {field_type}")
            return None

        # For types that may use multiple sources (like buttons), try all
        if field_type == "buttons":
            sources = [extra.get("contest") or "", extra.get("keyword") or "", extra.get("url") or ""]
            found = []
            found_methods = []
            found_labels = []
            for src in sources:
                value, label, score, method, result = self._extract_with_strategies(
                    src,
                    strategies[field_type][:2]  # regex, nlp
                )
                if value:
                    found.append(value)
                    found_methods.append(method)
                    found_labels.append(label if label is not None else src)
            if not found:
                value, label, score, method, result = self._extract_with_strategies(
                    "", [strategies[field_type][2]]
                )
                if value:
                    found = value if isinstance(value, list) else [value]
                    found_methods = [method] * len(found)
                    found_labels = [label if label is not None else "direct_lookup"] * len(found)
                else:
                    found = []
                    found_methods = []
                    found_labels = []
            else:
                score = 0.9
                method = "regex"
                result = "pass"
            # Deduplicate
            if isinstance(found, list):
                found = list(dict.fromkeys(found))
            # Log each found value with its method and label - Get cooking Mr. White - No real blue used
            for val, meth, lab in zip(found, found_methods, found_labels):
                self.log_field_selection(
                    field_type="buttons",
                    field_name="buttons",
                    extracted_value=val,
                    method=meth,
                    score=score,
                    result=result,
                    context={**extra, "source_label": lab},
                    user_feedback=None,
                    log_path="field_selection_log.jsonl"
                )
            return found

        # For other types
        value, label, score, method, result = self._extract_with_strategies(
            text or "",
            strategies[field_type]
        )
        # Try to extract a label for logging (if possible)
        label_val = label if label is not None else (text if isinstance(text, str) else str(text))
        self.log_field_selection(
            field_type=field_type,
            field_name=field_type,
            extracted_value=value,
            method=method,
            score=score,
            result=result,
            context={**extra, "source_label": label_val},
            user_feedback=None,
            log_path="field_selection_log.jsonl"
        )
        return value

    def _extract_with_strategies(self, text, strategies) -> tuple:
        """
        Try a list of (method, function) strategies on text, returning the first successful result.
        Each function should return (value, label, score, method, result) or None.
        """
        for method, func in strategies:
            result = func(text)
            if result and isinstance(result, (tuple, list)) and len(result) > 0:
                # Unpack by position, fill missing fields with defaults
                value = result[0] if len(result) > 0 else None
                label = result[1] if len(result) > 1 else None
                score = result[2] if len(result) > 2 else 1.0
                result_val = result[4] if len(result) > 4 else (result[3] if len(result) > 3 else "pass")
                return (
                    value,
                    label,
                    score,
                    method,
                    result_val
                )
        return (None, None, 0.0, "fail", "none")

    def score_header_ml(self, title: str, context: dict = None) -> float:
        """
        ML-driven scoring for table headers.
        Uses semantic similarity, keyword matching, entity detection, and context features.
        Returns a float score between 0.0 and 1.0.
        """
        try:
            context_dict = context if isinstance(context, dict) else {}
            model = getattr(self, "_semantic_model", None)
            known_headers = set(context_dict.get("known_headers", []))
            known_labels = set(context_dict.get("known_labels", []))
            contest_obj = context_dict.get("contest", {})
            contest_title = contest_obj.get("title", "") if isinstance(contest_obj, dict) else str(contest_obj)
            # 1. Semantic similarity to known headers
            sim_scores = []
            if model and known_headers and isinstance(title, str):
                for h in known_headers:
                    sim = get_semantic_score(model, title, h, logger)
                    if isinstance(sim, (float, int)):
                        sim_scores.append(sim)
                max_sim = max(sim_scores) if sim_scores else 0.0
            else:
                max_sim = 0.0
            # 2. Fuzzy match to known headers
            fuzzy_scores = [difflib.SequenceMatcher(None, safe_lower(h), safe_lower(title)).ratio() for h in known_headers if isinstance(h, str) and isinstance(title, str)]
            max_fuzzy = max(fuzzy_scores) if fuzzy_scores else 0.0
            # 3. Entity detection
            ents = self.extract_entities(title)
            entity_boost = 0.0
            best_entity = None
            for ent, label in ents:
                if label in {"PERSON", "CANDIDATE", "ORG", "NORP", "GPE", "LOC"}:
                    entity_boost = 0.2
                    best_entity = ent
                    # Optionally: use semantic similarity to contest title or known labels
                    if model and contest_title and isinstance(ent, str):
                        sim = get_semantic_score(model, ent, contest_title, logger)
                        if isinstance(sim, (float, int)):
                            entity_boost += 0.1 * sim
                    # Optionally: use fuzzy match to known labels
                    if known_labels and isinstance(ent, str):
                        fuzzy_scores = [difflib.SequenceMatcher(None, safe_lower(lbl), safe_lower(ent)).ratio() for lbl in known_labels if isinstance(lbl, str)]
                        entity_boost += 0.05 * max(fuzzy_scores) if fuzzy_scores else 0.0
                    break
            # 4. Contextual match to contest title
            context_sim = get_semantic_score(model, title, contest_title, logger) if model and contest_title and isinstance(title, str) else 0.0
            # 5. Length and capitalization heuristic
            length_score = min(len(title) / 20.0, 0.2) if isinstance(title, str) else 0.0
            title_chars = list(title) if isinstance(title, str) else title
            first_char = safe_get_first(title_chars, "title_first_char", "", logger)
            cap_score = 0.1 if isinstance(title, str) and len(title) > 2 and safe_isupper(first_char, logger) else 0.0
            # 6. Aggregate score
            score = (
                0.4 * max_sim +
                0.2 * max_fuzzy +
                0.1 * context_sim +
                entity_boost +
                length_score +
                cap_score
            )
            # Clamp between 0.0 and 1.0
            return max(0.0, min(score, 1.0)), best_entity
        except Exception as e:
            logger.error(f"[score_header_ml] Error scoring header '{title}': {e}")
            return 0.5, None

    def score_entry(self, title: str, context: dict = None) -> float:
        """
        ML-driven scoring for any entry (header, label, etc.).
        Uses semantic similarity, fuzzy matching, entity detection, and context features.
        Returns a float score between 0.0 and 1.0.
        """
        try:
            context_dict = context if isinstance(context, dict) else {}
            model = getattr(self, "_semantic_model", None)
            contest_obj = context_dict.get("contest", {})
            contest_title = contest_obj.get("title", "") if isinstance(contest_obj, dict) else str(contest_obj)
            known_labels = set(context_dict.get("known_labels", []))
            sim_scores = []
            if model and known_labels:
                for lbl in known_labels:
                    sim = get_semantic_score(model, title, lbl)
                    sim_scores.append(sim)
                max_sim = max(sim_scores) if sim_scores else 0.0
            else:
                max_sim = 0.0
            fuzzy_scores = [difflib.SequenceMatcher(None, safe_lower(lbl), safe_lower(title)).ratio() for lbl in known_labels]
            max_fuzzy = max(fuzzy_scores) if fuzzy_scores else 0.0
            ents = self.extract_entities(title)
            entity_boost = 0.0
            for ent, label in ents:
                if label in {"PERSON", "CANDIDATE", "ORG", "NORP", "GPE", "LOC"}:
                    entity_boost = 0.2
                    break
            context_sim = get_semantic_score(model, title, contest_title) if model and contest_title else 0.0
            length_score = min(len(title) / 20.0, 0.2) if isinstance(title, str) else 0.0
            title_chars = list(title) if isinstance(title, str) else title
            first_char = safe_get_first(title_chars, "title_first_char", "", logger)
            cap_score = 0.1 if isinstance(title, str) and len(title) > 2 and safe_isupper(first_char, logger) else 0.0
            score = (
                0.4 * max_sim +
                0.2 * max_fuzzy +
                0.1 * context_sim +
                entity_boost +
                length_score +
                cap_score
            )
            return max(0.0, min(score, 1.0))
        except Exception as e:
            logger.error(f"[score_entry] Error scoring entry '{title}': {e}")
            return 0.5

    def score_header(self, title, context=None) -> float:
        """
        Score a table header using ML, NLP, or fallback heuristics.
        - title: header string
        - context: optional dict with additional info (e.g., contest, known headers)
        Returns: float score between 0.0 and 1.0
        """
        try:
            # Use ML model if available
            if hasattr(self, "score_entry"):
                return float(self.score_entry(title, context if isinstance(context, dict) else {}))
            if hasattr(self, "score_header_ml"):
                return float(self.score_header_ml(title, context if isinstance(context, dict) else {}))
            # Use NLP entity type as a weak signal
            if hasattr(self, "extract_entities"):
                ents = self.extract_entities(title)
                if ents:
                    for ent, label in ents:
                        if label in {"PERSON", "CANDIDATE", "ORG", "NORP", "GPE", "LOC"}:
                            return 0.8
            known_headers = set()
            if context and isinstance(context, dict):
                known_headers = set(context.get("known_headers", []))
            if known_headers and safe_lower(title) in (safe_lower(h) for h in known_headers):
                return 0.9
            # Defensive conversion for capitalization
            title_chars = list(title) if isinstance(title, str) else title
            first_char = safe_get_first(title_chars, "title_first_char", "", logger)
            if isinstance(title, str) and len(title) > 2 and safe_isupper(first_char, logger):
                return 0.6
            return 0.5
        except Exception as e:
            logger.error(f"[score_header] Error scoring header '{title}': {e}")
            return 0.5
    
    # --- DB/Service Delegation ---
    def get_full_contest(self, contest_id, enrich=True, validate_types=True, log_access=True) -> Optional[Dict[str, Any]]:
        """
        Advanced full contest accessor:
        - Enriches with NLP and type sync.
        - Validates type consistency.
        - Logs access and issues.
        """
        contest = self.data_service.get_full_contest(contest_id)
        if isinstance(contest, dict):
            try:
                sync_type_and_election_types(contest)
                if enrich:
                    contest["entities"] = self.extract_entities(contest.get("title", ""))
                    contest["locations"] = self.extract_locations(contest.get("title", ""))
                    contest["dates"] = self.extract_dates(contest.get("title", ""))
                if validate_types and (not contest.get("type_") or not contest.get("election_types")):
                    logger.warning(f"[get_full_contest] Contest {contest_id} missing type/election_types after sync.")
                if log_access:
                    logger.info(f"[get_full_contest] Accessed contest {contest_id}: {contest.get('title', '')}")
            except Exception as e:
                logger.error(f"[get_full_contest] Error enriching contest: {e}", exc_info=True)
        return contest

    def get_all_full_contests(
        self,
        filters: Optional[dict] = None,
        limit: int = 100,
        enrich: bool = True,
        deduplicate: bool = True,
        fuzzy: bool = False,
        semantic: bool = False,
        return_summary: bool = False
    ) -> List[dict]:
        """
        Advanced accessor for all contests:
        - Enriches with NLP and type sync.
        - Deduplicates and validates.
        - Supports flexible filtering (exact, fuzzy, semantic).
        - Optionally returns summary of type issues.
        """
        try:
            # Defensive: Only pass valid filters
            valid_filters = {k: v for k, v in filters.items() if v is not None and v != ""} if isinstance(filters, dict) else {}
            contests = self.data_service.get_contests_by_advanced_filter(valid_filters, limit=limit)
            seen_ids = set()
            enriched = []
            type_issues = []
            for c in contests:
                if not isinstance(c, dict):
                    continue
                sync_type_and_election_types(c)
                if enrich:
                    c["entities"] = self.extract_entities(c.get("title", ""))
                    c["locations"] = self.extract_locations(c.get("title", ""))
                    c["dates"] = self.extract_dates(c.get("title", ""))
                dedup_key = c.get("id") or c.get("title")
                if deduplicate and dedup_key in seen_ids:
                    continue
                seen_ids.add(dedup_key)
                if not c.get("type_") or not c.get("election_types"):
                    type_issues.append({"contest": c.get("title"), "issue": "Missing type or election_types"})
                enriched.append(c)
            def match(c):
                c_dict = c if isinstance(c, dict) else {}
                if not valid_filters:
                    return True
                for k, v in valid_filters.items():
                    val = str(c_dict.get(k, "")).lower()
                    tgt = str(v).lower()
                    if fuzzy:
                        if difflib.SequenceMatcher(None, val, tgt).ratio() < 0.7:
                            return False
                    elif semantic and hasattr(self, "_semantic_model"):
                        score = get_semantic_score(self._semantic_model, val, tgt)
                        if score < 0.7:
                            return False
                    else:
                        if tgt not in val:
                            return False
                return True
            filtered = [c for c in enriched if match(c)]
            if return_summary:
                return {"contests": clean_for_json(filtered), "type_issues": type_issues}
            return clean_for_json(filtered)
        except Exception as e:
            logger.error(f"[get_all_full_contests] Error: {e}", exc_info=True)
            return []

    def list_tables(self, validate=True) -> List[str]:
        """
        List all tables, optionally validating existence and schema.
        """
        try:
            tables = self.data_service.list_tables()
            if validate:
                valid_tables = []
                for tbl in tables:
                    meta = self.get_table_metadata(tbl)
                    if meta and meta.get("columns"):
                        valid_tables.append(tbl)
                    else:
                        logger.warning(f"[list_tables] Table '{tbl}' missing metadata or columns.")
                return valid_tables
            return tables
        except Exception as e:
            logger.error(f"[list_tables] Error: {e}", exc_info=True)
            return []

    def describe_table(self, table_name, enrich=True) -> Optional[Dict[str, Any]]:
        """
        Describe a table, optionally enriching with schema and sample data.
        """
        try:
            desc = self.data_service.describe_table(table_name)
            if enrich and desc:
                meta = self.get_table_metadata(table_name)
                if meta:
                    desc["metadata"] = meta
                # Optionally add sample rows
                if hasattr(self.data_service, "get_sample_rows"):
                    desc["sample_rows"] = self.data_service.get_sample_rows(table_name, limit=5)
            return desc
        except Exception as e:
            logger.error(f"[describe_table] Error: {e}", exc_info=True)
            return None

    def get_table_metadata(self, table_name, validate=True) -> Optional[Dict[str, Any]]:
        """
        Get table metadata, optionally validating schema.
        """
        try:
            meta = self.data_service.get_table_metadata(table_name)
            if validate and meta and not meta.get("columns"):
                logger.warning(f"[get_table_metadata] Table '{table_name}' missing columns.")
            return meta
        except Exception as e:
            logger.error(f"[get_table_metadata] Error: {e}", exc_info=True)
            return None

    def check_missing_tables(self, expected_tables=None) -> List[str]:
        """
        Check for missing tables against an expected list.
        """
        try:
            existing = set(self.data_service.list_tables())
            missing = []
            if expected_tables:
                missing = [tbl for tbl in expected_tables if tbl not in existing]
            else:
                missing = self.data_service.check_missing_tables()
            if missing:
                logger.warning(f"[check_missing_tables] Missing tables: {missing}")
            return missing
        except Exception as e:
            logger.error(f"[check_missing_tables] Error: {e}", exc_info=True)
            return []

    def get_table_structures(self, filters=None, limit=100, confirmed_only=False, enrich=True, deduplicate=True) -> List[Dict[str, Any]]:
        """
        Advanced accessor for table structures:
        - Enriches with NLP and validation.
        - Deduplicates and supports flexible filtering.
        Returns a list of enriched table structure dicts, each including score and best_entity.
        """
        try:
            structures = self.data_service.fetch_table_structures(filters=filters, limit=limit, confirmed_only=confirmed_only)
            seen = set()
            enriched = []
            for ts in structures:
                ts_dict = ts if isinstance(ts, dict) else {}
                if enrich:
                    score, best_entity = self.score_header_ml(ts_dict.get("title", ""), ts_dict.get("context", {}))
                    ts["score"] = score
                    ts["best_entity"] = best_entity
                dedup_key = ts_dict.get("id") or ts_dict.get("title")
                if deduplicate and dedup_key in seen:
                    continue
                seen.add(dedup_key)
                enriched.append(ts)
            return clean_for_json(enriched)
        except Exception as e:
            logger.error(f"[get_table_structures] Error: {e}", exc_info=True)
            return []

    def get_table_structure(self, contest, context=None, enrich=True) -> Optional[Dict[str, Any]]:
        """
        Get table structure for a contest, optionally enriching with ML/NLP.
        Returns the enriched table structure dict, including score and best_entity.
        """
        try:
            ts = self.data_service.get_table_structure(contest, context)
            if enrich and ts and isinstance(ts, dict):
                score, best_entity = self.score_header_ml(ts.get("title", ""), ts.get("context", {}))
                ts["score"] = score
                ts["best_entity"] = best_entity
            return clean_for_json(ts)
        except Exception as e:
            logger.error(f"[get_table_structure] Error: {e}", exc_info=True)
            return None

    def save_table_structure(self, contest, headers, context, ml_confidence=None, confirmed_by_user=False, validate=True) -> bool:
        """
        Save table structure with validation and logging.
        """
        try:
            if validate and not headers:
                logger.error("[save_table_structure] No headers provided.")
                return False
            result = self.data_service.save_table_structure(contest, headers, context, ml_confidence, confirmed_by_user)
            if result:
                logger.info(f"[save_table_structure] Saved structure for contest: {contest}")
            else:
                logger.warning(f"[save_table_structure] Failed to save structure for contest: {contest}")
            return result
        except Exception as e:
            logger.error(f"[save_table_structure] Error: {e}", exc_info=True)
            return False

    # --- Context/Contest Accessors ---
    def get_contests(self, filters=None, enrich=True, deduplicate=True, fuzzy=False, semantic=False, return_summary=False) -> List[Dict[str, Any]]:
        """
        Advanced contest accessor:
        - Enriches contests with NLP, type sync, and deduplication.
        - Supports flexible filtering: exact, partial, fuzzy, semantic.
        - Optionally returns a summary of type consistency issues.
        """
        contests = safe_get(self.organized, "contests", [])
        seen_ids = set()
        enriched = []
        type_issues = []
        for c in contests:
            if not isinstance(c, dict):
                continue
            try:
                sync_type_and_election_types(c)
                if enrich:
                    c["entities"] = self.extract_entities(c.get("title", ""))
                    c["locations"] = self.extract_locations(c.get("title", ""))
                    c["dates"] = self.extract_dates(c.get("title", ""))
                dedup_key = c.get("id") or c.get("title")
                if deduplicate and dedup_key in seen_ids:
                    continue
                seen_ids.add(dedup_key)
                if not c.get("type_") or not c.get("election_types"):
                    type_issues.append({"contest": c.get("title"), "issue": "Missing type or election_types"})
                enriched.append(c)
            except Exception as e:
                logger.error(f"[get_contests] Error enriching contest: {e}", exc_info=True)
                continue
        def match(c):
            if not filters:
                return True
            for k, v in safe_items(filters):
                val = safe_lower(safe_get(c, k, ""))
                tgt = safe_lower(v)
                if fuzzy:
                    if difflib.SequenceMatcher(None, val, tgt).ratio() < 0.7:
                        return False
                elif semantic and hasattr(self, "_semantic_model"):
                    score = get_semantic_score(self._semantic_model, val, tgt)
                    if score < 0.7:
                        return False
                else:
                    if tgt not in val:
                        return False
            return True
        filtered = [c for c in enriched if match(c)]
        if return_summary:
            return {"contests": clean_for_json(filtered), "type_issues": type_issues}
        return clean_for_json(filtered)

    def get_buttons(self, contest: Dict[str, Any], keyword: str = None, url: str = None) -> List[Dict[str, Any]]:
        """
        Return all buttons, or those for a specific contest, or matching a keyword/URL.
        First, check the button selection log for a successful match.
        """
        # Sanitize log path to prevent path-injection
        safe_log_filename = "button_selection_log.jsonl"
        log_dir = os.path.abspath(LOG_DIR)
        log_path = os.path.join(log_dir, safe_log_filename)
        if not log_path.startswith(log_dir):
            logger.error(f"[get_buttons] Unsafe log path detected: {log_path}")
            return []
        os.makedirs(log_dir, exist_ok=True)
        if os.path.exists(log_path):
            with open(log_path, "rb") as f:
                for line in f:
                    try:
                        entry = orjson.loads(line)
                    except Exception:
                        continue
                    if not isinstance(entry, dict):
                        continue
                    # Contest match (safe_lower for robust comparison)
                    if contest and safe_lower(entry.get("contest", "")) == safe_lower(safe_get(contest, "title", "")) and safe_startswith(entry.get("result", ""), "pass", logger):
                        button = {
                            "label": entry.get("button_label"),
                            "selector": entry.get("selector"),
                        }
                        return clean_for_json([button])
                    # Keyword match (safe_lower for both sides)
                    if keyword and safe_lower(keyword) in safe_lower(entry.get("button_label", "")) and safe_startswith(entry.get("result", ""), "pass", logger):
                        button = {
                            "label": entry.get("button_label"),
                            "selector": entry.get("selector"),
                        }
                        return clean_for_json([button])
                    # URL match (safe_get for selector)
                    if url and url in safe_get(entry, "selector", "") and safe_startswith(entry.get("result", ""), "pass", logger):
                        button = {
                            "label": entry.get("button_label"),
                            "selector": entry.get("selector"),
                        }
                        return clean_for_json([button])

        # 2. Fallback to existing logic
        if not self.organized:
            return []
        buttons_dict = safe_get(self.organized, "buttons", {})
        results = []
        if contest:
            results = safe_get(buttons_dict, safe_get(contest, "title", ""), [])
            if results:
                return clean_for_json(results)
        if keyword:
            keyword = safe_lower(keyword)
            btn_lists = buttons_dict.values() if isinstance(buttons_dict, dict) else buttons_dict
            for btn_list in btn_lists:
                for btn in btn_list:
                    if not isinstance(btn, dict):
                        continue
                    if keyword in safe_lower(safe_get(btn, "label", "")) or keyword in safe_lower(safe_get(btn, "selector", "")):
                        results.append(btn)
            if results:
                return clean_for_json(results)
        if url:
            for btn_list in buttons_dict.values():
                for btn in btn_list:
                    if not isinstance(btn, dict):
                        continue
                    if url in safe_get(btn, "selector", ""):
                        results.append(btn)
            if results:
                return clean_for_json(results)
        all_buttons = []
        btn_lists = buttons_dict.values() if isinstance(buttons_dict, dict) else buttons_dict
        for btns in btn_lists:
            all_buttons.extend(btns)
        return clean_for_json(all_buttons)

    def matches_html_label_pattern(label, patterns) -> bool:
        """Check if label matches any HTML-specific regex pattern."""
        for pat in patterns:
            if re.search(pat, label, re.IGNORECASE):
                return True
        return False

    def log_pattern_attempt(self, label, pattern, result, context=None) -> None:
        """Log each pattern attempt for self-learning."""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "label": label,
            "pattern": pattern,
            "result": result,  # e.g., "match", "no_match", "clicked", "skipped"
            "context": context or {}
        }
        os.makedirs(LOG_DIR, exist_ok=True)
        log_path = os.path.join(LOG_DIR, "pattern_attempts_log.jsonl")
        with open(log_path, "ab") as f:
            f.write(orjson.dumps(clean_for_json(log_entry)) + b"\n")

    def get_best_button_advanced(
        self,
        page,
        contest: dict = None,
        keywords: list[str] = None,
        context: dict = None,
        fuzzy_thresholds: list[float] = None,
        prompt_user_for_button: bool = None,
        confirm_button_callback: Callable[[Dict[str, Any]], None] = None,
        learning_mode: bool = True
    ) -> Tuple[Optional[Dict[str, Any]], int]:
        """
        Advanced button selection: combines memory, DOM, semantic similarity, adaptive threshold, and feedback.
        Now supports confirmation, exclusion of rejected buttons, and learning mode (auto-apply corrections from log/DB).
        """
        # --- Protection: ensure contest is a dict ---
        if contest is not None and not isinstance(contest, dict):
            contest = {"title": str(contest)}
            logger.warning(f"[get_best_button_advanced] Contest argument was not a dict. Converted to: {contest}")
        # --- Protection: ensure keywords is a list ---
        if keywords is not None and not isinstance(keywords, list):
            keywords = [str(keywords)]
            logger.warning(f"[get_best_button_advanced] Keywords argument was not a list. Converted to: {keywords}")
        # --- Protection: ensure context is a dict ---
        if context is not None and not isinstance(context, dict):
            context = {"url": str(context)}
            logger.warning(f"[get_best_button_advanced] Context argument was not a dict. Converted to: {context}")
        # --- Initialize defaults ---
        if not isinstance(self.clicked_button_selectors, set):
            self.clicked_button_selectors = set()
        if not hasattr(self, "_semantic_model"):
            self._semantic_model = None
        if not isinstance(self._semantic_model, object):
            logger.warning("[get_best_button_advanced] _semantic_model is not set or is not an object. Using None.")
            self._semantic_model = None
        if fuzzy_thresholds is None:
            fuzzy_thresholds = [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5]
        model = self._semantic_model
        context = context or {}
        context.update({
            "contest": contest,
            "year": context.get("year", ""),
            "type_": contest.get("type_") if isinstance(contest, dict) else "",
            "election_types": contest.get("election_types") if isinstance(contest, dict) else [],
            "county": context.get("county", ""),
            "state": context.get("state", "")
        })

        # --- 1. Learning mode: check log/DB for confirmed button ---
        if learning_mode:
            learned_btn = self._get_confirmed_button_from_log(contest, keywords, context)
            if isinstance(learned_btn, dict) and learned_btn.get("selector") not in self.clicked_button_selectors:
                selector_html = learned_btn.get("selector", "")
                dom_candidates = []
                # Use a broad selector for all clickable elements
                BUTTON_SELECTORS = (
                    "button, a[href], [role='button'], input[type='button'], input[type='submit'], "
                    "[tabindex]:not([tabindex='-1'])"
                )
                button_features = safe_locator(page, BUTTON_SELECTORS, logger)
                for i in range(safe_count(button_features, logger)):
                    btn = safe_nth(button_features, i, logger)
                    try:
                        btn_html = safe_evaluate(btn, "el => el.outerHTML", logger)
                        if btn_html == selector_html:
                            learned_btn["element_handle"] = btn
                            learned_btn["is_visible"] = safe_is_visible(btn, logger)
                            learned_btn["is_clickable"] = safe_is_enabled(btn, logger)
                            break
                    except Exception as e:
                        if logger:
                            logger.error("[get_best_button_advanced] Error scanning learned button: %s", e)
                        continue
                if (
                    isinstance(learned_btn, dict)
                    and learned_btn.get("element_handle")
                    and learned_btn.get("is_visible")
                    and learned_btn.get("is_clickable")
                ):
                    logger.info(f"[green][LEARNING] Auto-applying learned button: {learned_btn.get('label')}[/green]")
                    try:
                        safe_click(learned_btn.get("element_handle"), logger)
                        safe_wait_for_timeout(page, 1500, logger)
                        self.clicked_button_selectors.add(learned_btn.get("selector"))
                        return learned_btn, 0
                    except Exception:
                        logger.error("[LEARNING] Failed to click learned button element.", exc_info=True)
                else:
                    logger.error("[red][ERROR] No element_handle found for the learned button candidate.[/red]")

        # --- 2. Gather candidates from memory/log ---
        memory_candidates = []
        logged_buttons = self.get_buttons(contest=contest, keyword=keywords, url=context.get("url", ""))
        if logged_buttons:
            for btn in logged_buttons:
                btn = btn.copy()
                btn["source"] = "memory"
                memory_candidates.append(btn)

        # --- 3. Gather live candidates from DOM ---
        dom_candidates = []
        # Use a broad selector for all clickable elements
        BUTTON_SELECTORS = (
            "button, a[href], [role='button'], input[type='button'], input[type='submit'], "
            "[tabindex]:not([tabindex='-1'])"
        )
        button_features = safe_locator(page, BUTTON_SELECTORS, logger)

        def scan_btn(btn, i) -> None:
            try:
                # Robust label extraction
                label = safe_inner_text(btn, logger) or ""
                if not label:
                    # Try aria-label or value attribute
                    label = safe_get_attribute(btn, "aria-label", logger) or safe_get_attribute(btn, "value", logger) or ""
                class_name = safe_get_attribute(btn, "class", logger) or ""
                role = safe_get_attribute(btn, "role", logger) or ""
                tag = safe_evaluate(btn, "el => el.tagName", logger)
                tag = tag.lower() if isinstance(tag, str) else ""
                is_visible = safe_is_visible(btn, logger)
                is_enabled = safe_is_enabled(btn, logger)
                selector = safe_evaluate(btn, "el => el.outerHTML", logger) if btn else ""
                candidate = {
                    "label": label.strip(),
                    "class": class_name,
                    "role": role,
                    "tag": tag,
                    "selector": selector or "",
                    "is_visible": is_visible,
                    "is_clickable": is_enabled,
                    "source": "dom",
                    "element_handle": btn,
                }
                dom_candidates.append(candidate)
                self._log_button_memory(candidate, contest, "scanned")
            except Exception as e:
                logger.error(f"[scan_btn] Error: {e}")

        scan_buttons_with_progress(
            [safe_nth(button_features, i, logger) for i in range(safe_count(button_features, logger))],
            scan_callback=scan_btn
        )

        # --- 4. Merge, deduplicate, and rank all candidates ---
        all_candidates = merge_and_rank_candidates(memory_candidates, dom_candidates, context, keywords, model)
        all_candidates = [c for c in all_candidates if isinstance(c, dict) and c.get("selector") not in self.clicked_button_selectors]

        # --- 5. Adaptive threshold: try high, then lower if no match ---
        excluded_labels = set()
        while True:
            found = False
            for threshold in fuzzy_thresholds:
                for idx, cand in enumerate(all_candidates):
                    if not isinstance(cand, dict):
                        continue
                    if cand.get("combined_score", 0) >= threshold and cand.get("is_visible") and cand.get("is_clickable"):
                        if cand.get("label") in excluded_labels:
                            continue
                        confirmed = True
                        if confirm_button_callback:
                            confirmed = confirm_button_callback(cand)
                        if confirmed:
                            logger.info(f"[bold green][Coordinator] Confirmed button: '{cand.get('label')}' (score={cand.get('combined_score', 0):.2f})[/bold green]")
                            self._log_button_memory(cand, contest, f"confirmed_pass_{cand.get('combined_score', 0):.2f}")
                            if not isinstance(cand, dict):
                                logger.error(f"[red][ERROR] Candidate is not a dict: {cand}[/red]")
                                continue
                            if learning_mode:
                                self._log_confirmed_button_for_learning(cand, contest, context)
                            self.clicked_button_selectors.add(cand.get("selector"))
                            try:
                                safe_click(cand.get("element_handle"), logger)
                                safe_wait_for_timeout(page, 1500, logger)
                            except Exception as e:
                                logger.error(f"[get_best_button_advanced] Click/wait error: {e}")
                            return cand, idx
                        else:
                            excluded_labels.add(cand.get("label"))
                            logger.warning(f"[yellow][Coordinator] Button '{cand.get('label')}' rejected, retrying...[/yellow]")
                            found = True
                            break
                if found:
                    break
            else:
                break

        # --- 6. Feedback UI: Prompt user for manual correction ---
        if prompt_user_for_button:
            if not isinstance(context, dict):
                context = {}
            chosen_btn, chosen_idx = prompt_user_for_button(page, all_candidates, context.get("toggle_name", ""))
            if chosen_btn and chosen_idx is not None:
                chosen_btn["context"] = context
                self._log_button_memory(chosen_btn, contest, "manual_correction")
                if learning_mode:
                    self._log_confirmed_button_for_learning(chosen_btn, contest, context)
                return chosen_btn, chosen_idx

        logger.error(f"[red][ERROR] No suitable button could be clicked for '{context.get('toggle_name', '')}'.[/red]")
        return None, None

    def _log_confirmed_button_for_learning(self, button: dict = None, contest: dict = None, context: dict = None) -> None:
        """
        Log confirmed button for learning mode (auto-apply next time).
        """
        # Ensure button is a dict before using .get()
        if not isinstance(button, dict):
            return
        log_entry = {
            "contest": contest,
            "button_label": button.get("label"),
            "selector": button.get("selector"),
            "context": context,
            "result": "learning_confirmed"
        }
        os.makedirs(LOG_DIR, exist_ok=True)
        log_path = os.path.join(LOG_DIR, "button_learning_log.jsonl")
        with open(log_path, "ab") as f:
            f.write(orjson.dumps(clean_for_json(log_entry)) + b"\n")

    def _get_confirmed_button_from_log(self, contest: dict = None, keywords: list[str] = None, context: dict = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve a previously confirmed button from the learning log.
        """
        log_path = os.path.join(LOG_DIR, "button_learning_log.jsonl")
        if not os.path.exists(log_path):
            return None
        with open(log_path, "rb") as f:
            for line in f:
                try:
                    entry = orjson.loads(line)
                except Exception:
                    continue
                if not isinstance(entry, dict):
                    continue
                if entry.get("contest") == contest and entry.get("result") == "learning_confirmed":
                    return {
                        "label": entry.get("button_label"),
                        "selector": entry.get("selector"),
                        "context": entry.get("context"),
                        "source": "learning"
                    }
        return None

    def _log_button_memory(self, button: dict = None, contest: dict = None, result: str = None) -> None:
        """
        Log button selection attempts for future ML or rule improvements.
        """
        # Ensure button is a dict before using .get()
        if not isinstance(button, dict):
            return
        log_entry = {
            "contest": contest,
            "button_label": button.get("label"),
            "selector": button.get("selector"),
            "result": result
        }
        os.makedirs(LOG_DIR, exist_ok=True)
        log_path = os.path.join(LOG_DIR, "button_selection_log.jsonl")
        with open(log_path, "ab") as f:
            f.write(orjson.dumps(clean_for_json(log_entry)) + b"\n")

    def record_navigation_feedback(
        self,
        *,
        script_id: str | None,
        success: bool,
        context_before: dict | None,
        context_after: dict | None,
        telemetry: list[dict] | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Persist navigation runner outcomes for ML feedback loops."""

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "script_id": script_id,
            "success": bool(success),
            "context_before": context_before or {},
            "context_after": context_after or {},
            "telemetry": telemetry or [],
            "metadata": metadata or {},
        }
        os.makedirs(LOG_DIR, exist_ok=True)
        log_path = os.path.join(LOG_DIR, "navigation_learning_log.jsonl")
        with open(log_path, "ab") as handle:
            handle.write(orjson.dumps(clean_for_json(log_entry)) + b"\n")

    # --- Table structure learning/lookup ---
    def get_table_structure_from_log(
        self,
        contest: dict | None = None,
        context: dict | None = None,
        learning_mode: bool = True,
    ) -> Optional[list[str]]:
        """Retrieve a previously learned table structure from the learning log."""
        log_path = os.path.join(LOG_DIR, "table_structure_learning_log.jsonl")
        # 1. Learning mode: check log for confirmed structure
        if learning_mode and os.path.exists(log_path):
            with open(log_path, "rb") as f:
                for line in f:
                    try:
                        entry = orjson.loads(line)
                    except Exception:
                        continue
                    if not isinstance(entry, dict):
                        continue
                    if entry.get("contest") == contest and entry.get("result") == "learning_confirmed":
                        return clean_for_json(entry.get("headers"), [])
        # 2. Fallback: return None (caller should trigger extraction and confirmation)
        return None

    def log_table_structure(self, contest: dict = None, headers: list[str] = None, context: dict = None) -> None:
        """
        Log confirmed table structure for learning mode.
        """
        log_entry = {
            "contest": contest,
            "headers": headers,
            "context": context,
            "result": "learning_confirmed"
        }
        os.makedirs(LOG_DIR, exist_ok=True)
        log_path = os.path.join(LOG_DIR, "table_structure_learning_log.jsonl")
        with open(log_path, "ab") as f:
            f.write(orjson.dumps(clean_for_json(log_entry)) + b"\n")

    # --- CLI for reviewing/editing corrections and feedback ---
    def review_and_edit_corrections(self, field_type="buttons") -> None:
        """
        Launch the manual_correction CLI for reviewing/editing corrections and feedback.
        """
        script_path = os.path.join(os.path.dirname(__file__), "..", "health", "manual_correction.py")
        subprocess.run(["python", script_path, "--fields", field_type, "--feedback", "--enhanced"], check=True, cwd=PROJECT_ROOT)

    # --- Learning mode: auto-apply corrections from log/database ---
    def enable_learning_mode(self) -> None:
        """
        Enable learning mode for auto-applying corrections from logs/database.
        """
        self.learning_mode = True

    def disable_learning_mode(self) -> None:
        """
        Disable learning mode.
        """
        self.learning_mode = False

    def get_panel(self, contest: dict = None) -> dict:
        if not self.organized:
            return None
        panels = safe_get(self.organized, "panels", {})
        if not isinstance(panels, dict):
            return None
        return clean_for_json(safe_get(panels, contest, None))

    def get_tables(self, contest: dict = None) -> list[dict]:
        if not self.organized:
            return []
        tables = safe_get(self.organized, "tables", {})
        if not isinstance(tables, dict):
            return []
        contest_title = safe_get(contest, "title", "") if contest else ""
        return clean_for_json(safe_get(tables, contest_title, []))

    def get_candidates(self, contest: dict = None) -> list[str]:
        """
        Extract candidate names from contest entities or table headers.
        """
        candidates = set()
        if contest is None:
            contests = self.get_contests()
        else:
            contests = [contest]
        for c in contests:
            if not isinstance(c, dict):
                continue
            for ent, label in c.get("entities", []):
                if label in {"PERSON", "CANDIDATE"}:
                    candidates.add(ent)
            # Optionally: parse table headers for candidate names
            for tbl in self.get_tables(c):
                if not isinstance(tbl, dict):
                    continue
                headers = tbl.get("headers", [])
                for h in headers:
                    if isinstance(h, str) and "candidate" in h.lower():
                        candidates.add(h)
        return clean_for_json(list(candidates))

    def get_precincts(self, state: str = None, county: str = None) -> list[str]:
        """
        Return known precincts for a state/county from the static mapping in librarian.py.
        Returns an empty list if neither is provided.
        """       
        if county:
            precincts_map = KNOWN_COUNTY_TO_PRECINCTS_MAP
            if isinstance(precincts_map, dict):
                return clean_for_json(precincts_map.get(county, []))
            return []
        if state:
            state_map = KNOWN_STATE_TO_COUNTY_MAP
            if isinstance(state_map, dict):
                return clean_for_json(state_map.get(state, []))
            return []
        return []

    def get_states(self) -> list[str]:
        """
        Return all known states from the static mapping in librarian.py.
        """
        return list(STATE_MODULE_MAP.keys())

    def get_election_types(self) -> list[str]:
        """
        Return all known election types from the static mapping in librarian.py.
        """
        return list(ELECTION_TYPES)


    def get_years(self) -> list[int]:
        """
        Return all years found in contests.
        """      
        contests = self.get_contests()
        return clean_for_json(sorted({c.get("year") for c in contests if isinstance(c, dict) and c.get("year")}))

    # --- Integrity & Anomaly Checks ---

    def _log_get_contests_access(self, filters) -> None:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "method": "get_contests",
            "filters": filters,
        }
        os.makedirs(LOG_DIR, exist_ok=True)
        log_path = os.path.join(LOG_DIR, "get_contests_access_log.jsonl")
        with open(log_path, "ab") as f:
            f.write(orjson.dumps(clean_for_json(log_entry)) + b"\n")

    def _log_get_buttons_access(self, contest: Dict[str, Any], keyword: str, url: str) -> None:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "method": "get_buttons",
            "contest": contest,
            "keyword": keyword,
            "url": url,
        }

        os.makedirs(LOG_DIR, exist_ok=True)
        log_path = os.path.join(LOG_DIR, "get_buttons_access_log.jsonl")
        with open(log_path, "ab") as f:
            f.write(orjson.dumps(clean_for_json(log_entry)) + b"\n")

    def _log_get_best_button_access(self, contest: Dict[str, Any], keywords: List[str], class_hint: str, url: str) -> None:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "method": "get_best_button",
            "contest": contest,
            "keywords": keywords,
            "class_hint": class_hint,
            "url": url,
        }
        os.makedirs(LOG_DIR, exist_ok=True)
        log_path = os.path.join(LOG_DIR, "get_best_button_access_log.jsonl")
        with open(log_path, "ab") as f:
            f.write(orjson.dumps(clean_for_json(log_entry)) + b"\n")

    def _log_get_panel_access(self, contest: Dict[str, Any]) -> None:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "method": "get_panel",
            "contest": contest,
        }
        os.makedirs(LOG_DIR, exist_ok=True)
        log_path = os.path.join(LOG_DIR, "get_panel_access_log.jsonl")
        with open(log_path, "ab") as f:
            f.write(orjson.dumps(clean_for_json(log_entry)) + b"\n")

    def _log_get_tables_access(self, contest: Dict[str, Any]) -> None:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "method": "get_tables",
            "contest": contest,
        }
        os.makedirs(LOG_DIR, exist_ok=True)
        log_path = os.path.join(LOG_DIR, "get_tables_access_log.jsonl")
        with open(log_path, "ab") as f:
            f.write(orjson.dumps(clean_for_json(log_entry)) + b"\n")

    def _log_get_candidates_access(self, contest: Dict[str, Any]) -> None:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "method": "get_candidates",
            "contest": contest,
        }
        os.makedirs(LOG_DIR, exist_ok=True)
        log_path = os.path.join(LOG_DIR, "get_candidates_access_log.jsonl")
        with open(log_path, "ab") as f:
            f.write(orjson.dumps(clean_for_json(log_entry)) + b"\n")

    def _log_get_precincts_access(self, state, county) -> None:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "method": "get_precincts",
            "state": state,
            "county": county,
        }
        os.makedirs(LOG_DIR, exist_ok=True)
        log_path = os.path.join(LOG_DIR, "get_precincts_access_log.jsonl")
        with open(log_path, "ab") as f:
            f.write(orjson.dumps(clean_for_json(log_entry)) + b"\n")

    def _log_get_states_access(self) -> None:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "method": "get_states",
        }
        os.makedirs(LOG_DIR, exist_ok=True)
        log_path = os.path.join(LOG_DIR, "get_states_access_log.jsonl")
        with open(log_path, "ab") as f:
            f.write(orjson.dumps(clean_for_json(log_entry)) + b"\n")

    def _log_get_election_types_access(self) -> None:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "method": "get_election_types",
        }
        os.makedirs(LOG_DIR, exist_ok=True)
        log_path = os.path.join(LOG_DIR, "get_election_types_access_log.jsonl")
        with open(log_path, "ab") as f:
            f.write(orjson.dumps(clean_for_json(log_entry)) + b"\n")

    def _log_get_years_access(self) -> None:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "method": "get_years",
        }
        os.makedirs(LOG_DIR, exist_ok=True)
        log_path = os.path.join(LOG_DIR, "get_years_access_log.jsonl")
        with open(log_path, "ab") as f:
            f.write(orjson.dumps(clean_for_json(log_entry)) + b"\n")
            
    # --- Dynamic Data for Downstream Consumers ---
    def get_for_selector(self) -> dict:
        return self.organizer.get_for_selector()

    def get_for_table_builder(self) -> dict:
        return self.organizer.get_for_table_builder()

    def get_for_html_handler(self) -> dict:
        return self.organizer.get_for_html_handler()

    def get_for_state_router(self) -> dict:
        return self.organizer.get_for_state_router()

    # --- Integrity & Anomaly Checks ---
    def validate_and_check_integrity(self, expected_year=None) -> dict:
        contests = self.get_contests()
        integrity_issues = election_integrity_checks(contests)
        advanced_issues = advanced_cross_field_validation(contests)
        anomalies, clusters = detect_anomalies_with_ml(contests)
        features = []
        le_state = LabelEncoder()
        le_county = LabelEncoder()
        states = [c.get("state", "unknown") for c in contests if isinstance(c, dict)]
        counties = [c.get("county", "unknown") for c in contests if isinstance(c, dict)]
        le_state.fit(states)
        le_county.fit(counties)
        for c in contests:
            if not isinstance(c, dict):
                continue
            features.append([
                safe_get_first(le_state.transform([c.get("state", "unknown")]), "le_state_transform", None, logger, default=0),
                safe_get_first(le_county.transform([c.get("county", "unknown")]), "le_county_transform", None, logger, default=0),
                int(c.get("year", 0)) if str(c.get("year", "0")).isdigit() else 0,
                len(str(c.get("title", ""))),
            ])
        X = np.array(features)
        print_integrity_summary(contests, expected_year, X=X)
        date_anomalies = []
        if expected_year:
            for c in contests:
                if not isinstance(c, dict):
                    continue
                dates = c.get("dates", [])
                if not any(str(expected_year) in d for d in dates):
                    date_anomalies.append(c)
        return {
            "integrity_issues": integrity_issues,
            "advanced_issues": advanced_issues,
            "anomalies": anomalies,
            "clusters": clusters.tolist() if hasattr(clusters, "tolist") else clusters,
            "date_anomalies": date_anomalies
        }
        
# --- CLI Entrypoint ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ContextCoordinator CLI")
    parser.add_argument("--monitor", action="store_true", help="Start alert monitoring")
    parser.add_argument("--no-background", action="store_true", help="Run alert monitoring in foreground")
    args = parser.parse_args()

    if not args.sample and not args.monitor:
        parser.print_help()
