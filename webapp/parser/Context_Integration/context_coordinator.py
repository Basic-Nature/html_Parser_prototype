"""
context_coordinator.py

Production-grade Context Coordinator for Election Data Pipeline

- Orchestrates advanced context analysis, NLP, and ML integrity checks.
- Bridges between spaCy (NLP), context_organizer (DOM/ML), and downstream consumers (selectors, handlers, routers).
- Provides robust, dynamic, and cache-aware access to contests, buttons, panels, tables, candidates, precincts, etc.
- Ensures all data is validated, deduplicated, and anomaly-checked before output.
"""
import re
import os
import numpy as np
import orjson
from datetime import datetime, timezone
import types
from fuzzywuzzy import fuzz, process
from ..utils.shared_logger import rprint, logging
import difflib
from ..utils.shared_logic import (
    scan_buttons_with_progress, keyphrase_match,
    normalize_state_name, normalize_county_name
)
from ..bots.librarian import ( 
    load_context_library, 
    update_context_library,
    STATE_ABBR,
    PARTY_KEYWORDS,
    LOCATION_KEYWORDS,
    STATE_MODULE_MAP,
    KNOWN_STATE_TO_COUNTY_MAP,
    KNOWN_COUNTY_TO_PRECINCTS_MAP
)
from sklearn.preprocessing import LabelEncoder
import subprocess
from rich.console import Console
from ..config import PROJECT_ROOT, CONTEXT_LIBRARY_PATH, LOG_DIR
import threading
console = Console()

from ..utils.spacy_utils import (
    extract_entities, extract_locations, extract_dates
)
from .Integrity_check import (
    detect_anomalies_with_ml,
    election_integrity_checks,
    monitor_db_for_alerts,
    advanced_cross_field_validation,
    print_integrity_summary
)
from .context_organizer import ContextOrganizer, clean_for_json
import inspect

def _sanitize_log_filename(name):
    # Only allow alphanumeric, underscore, and dash
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', name) 

def get_semantic_score(model, text1, text2):
    """
    Compute semantic similarity between two strings using SentenceTransformer.
    """
    if not text1 or not text2:
        return 0.0
    emb1 = model.encode(text1, convert_to_tensor=False, show_progress_bar=False)
    emb2 = model.encode(text2, convert_to_tensor=False, show_progress_bar=False)
    from sentence_transformers import util
    return float(util.pytorch_cos_sim(emb1, emb2)[0][0])

def merge_and_rank_candidates(
    memory_candidates, dom_candidates, context, keywords, model,
    fuzzy_weight=0.3, semantic_weight=0.3, context_weight=0.2, hierarchy_weight=0.2
):
    """
    Merge memory and DOM candidates, deduplicate, and rank by combined fuzzy and semantic score.
    """
    seen = set()
    all_candidates = []
    for cand in memory_candidates + dom_candidates:
        if not isinstance(cand, dict):
            continue
        key = (cand.get("label", ""), cand.get("selector", ""))
        if key not in seen:
            seen.add(key)
            all_candidates.append(cand)

    context_str = " ".join([
        str(context.get("contest_title", "")),
        str(context.get("year", "")),
        str(context.get("election_types", "")),
        str(context.get("county", "")),
        str(context.get("state", "")),
    ]).strip()

    expected_class = context.get("expected_class", "")
    expected_tag = context.get("expected_tag", "")
    contest_title = context.get("contest_title", "")

    for cand in all_candidates:
        if not isinstance(cand, dict):
            continue
        label = cand.get("label", "") or ""
        # Strong full-string match
        full_match = int(label.strip().lower() == contest_title.strip().lower())
        # Keyphrase-aware match
        keyphrase_score = 0.0
        for kw in (keywords or []):
            if keyphrase_match(label, kw, min_words=2, fuzzy_cutoff=0.85):
                keyphrase_score = 1.0
                break
        # Fuzzy/semantic as fallback
        fuzzy_scores = [
            difflib.SequenceMatcher(None, kw.lower(), label.lower()).ratio()
            for kw in (keywords or [])
        ]
        fuzzy_score = max(fuzzy_scores) if fuzzy_scores else 0.0
        semantic_score = get_semantic_score(model, context_str, label)
        # Context proximity
        context_heading = cand.get("context_heading", "")
        context_proximity = 0.0
        if context_heading and contest_title:
            context_proximity = get_semantic_score(model, contest_title, context_heading)
        # Hierarchy/class/tag bonus
        hierarchy_score = 0.0
        if expected_class and expected_class in cand.get("class", ""):
            hierarchy_score += 0.5
        if expected_tag and expected_tag == cand.get("tag", ""):
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
            c.get("combined_score", 0),
            c.get("is_visible", False),
            c.get("is_clickable", False)
        ),
        reverse=True
    )
    return all_candidates

def call_handler_with_coordinator(handler, *args, coordinator=None, **kwargs):

    sig = inspect.signature(handler.parse)
    if 'coordinator' in sig.parameters:
        return handler.parse(*args, coordinator, **kwargs)
    else:
        return handler.parse(*args, **kwargs)

def dynamic_state_county_detection(context, html, debug=False):
    """
    Robustly detect county (first) and state (second) using all available clues and cross-referencing.
    Utilizes context fields, contest titles, URL, and canonical librarian mappings.
    Returns (county, state, handler_path, detection_log)
    """
    detection_log = []
    state_to_county = KNOWN_STATE_TO_COUNTY_MAP
    county_to_precinct = KNOWN_COUNTY_TO_PRECINCTS_MAP
    state_module_map = STATE_MODULE_MAP
    known_states = set(state_to_county.keys())
    all_counties = {normalize_county_name(c) for counties in state_to_county.values() for c in counties}
    all_precincts = {normalize_county_name(d) for precincts in county_to_precinct.values() for d in precincts}

    # --- 1. Try context fields directly (normalize and validate) ---
    if not isinstance(context, dict):
        context = {}
    raw_county = context.get("county")
    raw_state = context.get("state")
    county = normalize_county_name(raw_county) if raw_county else None
    state = normalize_state_name(raw_state) if raw_state else None

    # Validate county: is it a real county, or a precinct?
    if county:
        if county in all_counties:
            detection_log.append(f"County found in context: {county} (validated as county)")
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
                county = parent_county
            else:
                detection_log.append(f"County '{county}' found in context, but is a precinct with no parent mapping.")
        else:
            detection_log.append(f"County '{county}' found in context, but not recognized as county or precinct.")
            county = None

    # Validate state: is it a real state?
    if state:
        if state in known_states:
            detection_log.append(f"State found in context: {state} (validated as state)")
        else:
            # Try to map via state_module_map (handle abbreviations and fuzzy)
            mapped_state = state_module_map.get(state)
            if not mapped_state:
                # Try abbreviation
                abbr = state.lower()
                from ..bots.librarian import STATE_ABBR
                mapped_state = STATE_ABBR.get(abbr)
                if mapped_state:
                    detection_log.append(f"State '{state}' mapped from abbreviation to '{mapped_state}'.")
            if mapped_state:
                state = normalize_state_name(mapped_state)
                detection_log.append(f"State '{state}' found in context, mapped via state_module_map/abbr.")
            else:
                # Fuzzy match as last resort
                import difflib
                match = difflib.get_close_matches(state, known_states, n=1, cutoff=0.8)
                if match:
                    state = match[0]
                    detection_log.append(f"State '{state}' fuzzy-matched from context.")
                else:
                    detection_log.append(f"State '{state}' found in context, but not recognized.")
                    state = None

    # --- 2. Try to extract county from URL ---
    url = context.get("url", "") if isinstance(context, dict) else ""
    if not county and url:
        url_lower = url.lower()
        # Exact match
        for c in all_counties:
            if c in url_lower:
                county = c
                detection_log.append(f"County '{county}' detected from URL.")
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
                            break
                    if county:
                        break
        # Fuzzy match county in URL
        if not county:
            url_tokens = re.split(r"[\W_]+", url_lower)
            matches = difflib.get_close_matches(" ".join(url_tokens), all_counties, n=1, cutoff=0.7)
            if matches:
                county = matches[0]
                detection_log.append(f"County '{county}' fuzzy-matched from URL tokens.")
            else:
                matches = difflib.get_close_matches(" ".join(url_tokens), all_precincts, n=1, cutoff=0.7)
                if matches:
                    for c, precincts in county_to_precinct.items():
                        if not isinstance(precincts, list):
                            continue
                        if matches[0] in {normalize_county_name(x) for x in precincts}:
                            county = normalize_county_name(c)
                            detection_log.append(f"precinct '{matches[0]}' fuzzy-matched from URL tokens, mapped to county '{county}'")
                            break

    # --- 3. Try to extract county from contest titles ---
    contests = context.get("contests", []) if isinstance(context, dict) else []
    if not county and contests:
        for contest in contests:
            if not isinstance(contest, dict):
                continue
            title = contest.get("title", "")
            title_lower = title.lower()
            for c in all_counties:
                if re.search(rf"\b{re.escape(c)}\b", title_lower):
                    county = c
                    detection_log.append(f"County '{county}' detected from contest title: '{title}'")
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
                            break
                    if county:
                        break
            if county:
                break

    # --- 4. Try to extract county from HTML using NLP entities ---
    if not county and html:
        entities = extract_entities(html)
        gpe_entities = [normalize_county_name(ent) for ent, label in entities if label in ("GPE", "LOC")]
        for ent in gpe_entities:
            if ent in all_counties:
                county = ent
                detection_log.append(f"County '{county}' detected from HTML NLP entity.")
                break
            elif ent in all_precincts:
                for c, precincts in county_to_precinct.items():
                    if not isinstance(precincts, list):
                        continue
                    if ent in {normalize_county_name(x) for x in precincts}:
                        county = normalize_county_name(c)
                        detection_log.append(f"precinct '{ent}' detected from HTML NLP entity, mapped to county '{county}'")
                        break
                if county:
                    break

    # --- 5. Now try to detect state, using county if found ---
    if not state and county:
        for s, counties in state_to_county.items():
            if not isinstance(counties, list):
                continue
            if county in {normalize_county_name(x) for x in counties}:
                state = normalize_state_name(s)
                detection_log.append(f"State '{state}' inferred from county '{county}'.")
                break

    # --- 6. Try to extract state from URL ---
    if not state and url:
        url_lower = url.lower()
        for s in known_states:
            if s in url_lower:
                state = s
                detection_log.append(f"State '{state}' detected from URL.")
                break
        # Fuzzy match state in URL
        if not state:
            url_tokens = re.split(r"[\W_]+", url_lower)
            matches = difflib.get_close_matches(" ".join(url_tokens), list(known_states), n=1, cutoff=0.7)
            if matches:
                state = matches[0]
                detection_log.append(f"State '{state}' fuzzy-matched from URL tokens.")

    # --- 7. Try to extract state from contest titles ---
    if not state and contests:
        for contest in contests:
            if not isinstance(contest, dict):
                continue
            title = contest.get("title", "")
            title_lower = title.lower()
            for s in known_states:
                if s in title_lower:
                    state = s
                    detection_log.append(f"State '{state}' detected from contest title: '{title}'")
                    break
            if state:
                break

    # --- 8. Try to extract state from HTML using NLP entities ---
    if not state and html:
        entities = extract_entities(html)
        gpe_entities = [normalize_state_name(ent) for ent, label in entities if label in ("GPE", "LOC")]
        for ent in gpe_entities:
            if ent in known_states:
                state = ent
                detection_log.append(f"State '{state}' detected from HTML NLP entity.")
                break

    # --- 9. Special case: DC and other non-county states ---
    if state == "district_of_columbia":
        county = "district of columbia"
        detection_log.append("Special case: DC detected, setting county to 'district of columbia'.")

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
            url_and_html = ((url or "") + " " + (html or "")).lower()
            # Try exact match in URL/HTML
            for c in available_counties:
                if c in url_and_html:
                    normalized_county = c
                    detection_log.append(f"County '{normalized_county}' matched to available handler from URL/HTML context.")
                    break
            # Try fuzzy match in URL/HTML
            if not normalized_county and available_counties:
                tokens = re.split(r"[\W_]+", url_and_html)
                matches = difflib.get_close_matches(" ".join(tokens), available_counties, n=1, cutoff=0.7)
                if matches:
                    normalized_county = matches[0]
                    detection_log.append(f"County '{normalized_county}' fuzzy-matched to available handler from URL/HTML context.")
            # If only one county handler is available, use it as a fallback
            if not normalized_county and len(available_counties) == 1:
                normalized_county = available_counties[0]
                detection_log.append(f"Only one county handler available ('{normalized_county}'); using as fallback.")
            elif not normalized_county:
                detection_log.append("No matching county handler found in URL/HTML; will use state handler.")
        else:
            detection_log.append(f"No county handler directory found for state '{normalized_state}'.")

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
    if not normalized_state:
        detection_log.append("State could not be detected.")

    if debug:
        for log in detection_log:
            print("[dynamic_state_county_detection]", log)
    return normalized_county, normalized_state, handler_path, detection_log

# --- Core Coordinator Class ---

class ContextCoordinator:
    """
    Main interface for all context/NLP/ML operations.
    Use this class to access contests, buttons, panels, tables, candidates, precincts, etc.
    """
    def __init__(self, use_library=True, enable_ml=True, alert_monitor=True):
        self.library = load_context_library() if use_library else {}
        self.enable_ml = enable_ml
        self.alert_monitor = alert_monitor
        self.organized = None
        self.last_raw_context = None
        self.clicked_button_selectors = set()
        self.accepted_buttons_cache = {}
        self._semantic_model = None
        self.learning_mode = False  # Default to False for clarity
        self.alert_monitor_thread = None

        if enable_ml:
            from ..utils.model_registry import ModelRegistry
            self._semantic_model = ModelRegistry.get_sentence_transformer("all-MiniLM-L6-v2")

        if alert_monitor:
            self.start_alert_monitoring()
            
    def __del__(self):
        """
        Ensure alert monitoring thread is cleaned up on destruction.
        """
        try:
            if self.alert_monitor_thread and self.alert_monitor_thread.is_alive():
                logging.info("[ALERT MONITOR] Stopping alert monitoring thread.")
                self.alert_monitor_thread.join(timeout=1)
                if self.alert_monitor_thread.is_alive():
                    logging.warning("[ALERT MONITOR] Thread did not stop cleanly.")
                else:
                    logging.info("[ALERT MONITOR] Thread stopped successfully.")
            else:
                logging.info("[ALERT MONITOR] No active thread to stop.")
        except Exception as e:
            logging.error(f"[ALERT MONITOR] Exception during cleanup: {e}", exc_info=True)
        finally:
            self.alert_monitor_thread = None
            
    def start_alert_monitoring(self, background=True):
        """
        Start real-time alert monitoring, optionally in a background thread.
        """
        def run_monitor():
            try:
                monitor_db_for_alerts()
            except Exception as e:
                logging.error(f"[ALERT MONITOR] Exception: {e}", exc_info=True)

        if background:
            if self.alert_monitor_thread and self.alert_monitor_thread.is_alive():
                logging.info("[ALERT MONITOR] Already running.")
                return self.alert_monitor_thread
            t = threading.Thread(target=run_monitor, daemon=True)
            t.start()
            self.alert_monitor_thread = t
            logging.info("[ALERT MONITOR] Started in background thread.")
            return t
        else:
            run_monitor()
            return None
    def _log_jsonl(self, log_path, log_entry):
        """Centralized JSONL logging utility."""
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "ab") as f:
            f.write(orjson.dumps(clean_for_json(log_entry)) + b"\n")

    def _extract_with_strategies(self, text, strategies):
        """
        Try a list of (method, function) strategies on text, returning the first successful result.
        Each function should return (value, score, method, result) or None.
        """
        for method, func in strategies:
            result = func(text)
            if result and result[0]:
                return result + (method,)
        return (None, 0.0, "fail", "none")

    def _safe_get(self, dct, key, default=None):
        """Safely get a key from a dict, returning default if not a dict or key missing."""
        return dct.get(key, default) if isinstance(dct, dict) else default
            
    def save_table_structure_to_db(self, contest_title, headers, context, ml_confidence=None, confirmed_by_user=False):
        from .context_organizer import save_table_structure_to_db
        return save_table_structure_to_db(contest_title, headers, context, ml_confidence, confirmed_by_user)

    def get_table_structure_from_db(self, contest_title, context=None):
        from .context_organizer import get_table_structure_from_db
        return get_table_structure_from_db(contest_title, context)

    def organize_and_enrich(self, raw_context, contamination=None, n_estimators=100, random_state=42):
        """
        Organize raw context (from HTML/DOM or DB), deduplicate, cluster, and enrich with NLP.
        """
        self.last_raw_context = raw_context  # <-- Store the latest raw context
        organizer = ContextOrganizer(
            use_library=True,
            enable_ml=self.enable_ml,
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state
        )
        self.organized = organizer.organize_context(raw_context)
        self._enrich_contests_with_nlp()
        return self.organized

    def submit_user_feedback(self, field_type, field_name, correct_value, context):
        self.log_field_selection(
            field_type=field_type,
            field_name=field_name,
            extracted_value=correct_value,
            method="user_feedback",
            score=1.0,
            result="user_corrected",
            context=context,
            user_feedback=correct_value
        )  
        self._enrich_contests_with_nlp()         
        return self.organized

    def get_known_state_to_county_map(self):
        """
        Return all known states (keys) from the canonical state-to-county mapping in librarian.py.
        """
        return list(KNOWN_STATE_TO_COUNTY_MAP.keys())

    def get_known_county_to_PRECINCTS_map(self):
        """
        Return all known counties (keys) from the canonical county-to-precinct mapping in librarian.py.
        """
        return list(KNOWN_COUNTY_TO_PRECINCTS_MAP.keys())

    def get_known_states(self):
        """
        Return all known states from the canonical mapping in librarian.py.
        """
        # STATE_MODULE_MAP keys are already normalized (snake_case)
        return list(STATE_MODULE_MAP.keys())

    def get_known_counties(self, state=None):
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

    def _enrich_contests_with_nlp(self):
        """
        Add NLP-derived fields (entities, locations, dates) to each contest.
        """ 
        if not self.organized or "contests" not in self.organized:
            return
        for c in self.organized["contests"]:
            if not isinstance(c, dict):
                continue
            title = c.get("title", "")
            c["entities"] = extract_entities(title)
            c["locations"] = extract_locations(title)
            c["dates"] = extract_dates(title) 

    def fuzzy_score(self, a, b):
        """
        Compute a fuzzy string similarity score between two strings.
        """
        from fuzzywuzzy import fuzz
        model = self._semantic_model
        return fuzz.ratio(str(a), str(b))
               
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
    ):
        """
        Log field extraction/correction attempts for ML/feedback.
        Ensures log file is always inside the log/ directory and filename is sanitized.
        """
        # Always use log/ as the directory, and sanitize the filename
        safe_field_type = _sanitize_log_filename(field_type)
        if log_path is None:
            log_path = os.path.join(LOG_DIR, "log", f"{safe_field_type}_selection_log.jsonl")
        else:
            # Only use the filename part, sanitize it, and force it into log/
            base = os.path.basename(log_path)
            safe_base = _sanitize_log_filename(base)
            log_path = os.path.join(LOG_DIR, "log", safe_base)
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "field_type": field_type,
            "field_name": field_name,
            "extracted_value": extracted_value,
            "method": method,
            "score": score,
            "result": result,
            "context": context,
            "user_feedback": user_feedback
        }
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "ab") as f:
            f.write(orjson.dumps(clean_for_json(log_entry)) + b"\n")
    
    def extract_contest_title(self, contest):
        """
        Extract the contest title using ML/NLP/manual methods.
        Log the extraction attempt and result.
        """
        if not isinstance(contest, dict):
            return None
        extracted_value = contest.get("title")
        score = 1.0 if extracted_value else 0.0
        method = "manual" if extracted_value else "undefined"
        result = "pass" if extracted_value else "fail"
        user_feedback = None

        self.log_field_selection(
            field_type="contest",
            field_name="contest_title",
            extracted_value=extracted_value,
            method=method,
            score=score,
            result=result,
            context=contest,
            user_feedback=user_feedback,
            log_path="field_selection_log.jsonl"
        )
        return extracted_value

    def extract_candidate(self, contest):
        """
        Extract candidate names from contest using ML/NLP/manual methods.
        Log the extraction attempt and result.
        """
        if not isinstance(contest, dict):
            return []
        # Use entities if available
        candidates = []
        entities = contest.get("entities", [])
        for ent, label in entities:
            if label in {"PERSON", "CANDIDATE"}:
                candidates.append(ent)
        extracted_value = candidates
        score = 1.0 if candidates else 0.0
        method = "nlp"
        result = "pass" if candidates else "fail"
        user_feedback = None

        self.log_field_selection(
            field_type="candidate",
            field_name="candidate",
            extracted_value=extracted_value,
            method=method,
            score=score,
            result=result,
            context=contest,
            user_feedback=user_feedback,
            log_path="field_selection_log.jsonl"
        )
        return candidates

    def extract_party(self, contest):
        """
        Extract party using regex, spaCy NER, and fuzzy matching with PARTY_KEYWORDS.
        Log the extraction attempt and result.
        """
        if not isinstance(contest, dict):
            return None
        title = contest.get("title", "")

        party_pattern = "|".join([re.escape(k) for k in PARTY_KEYWORDS])

        def regex_party(text):
            match = re.search(rf"({party_pattern})", text, re.IGNORECASE)
            if match:
                return (match.group(1), 0.9, "regex", "pass")
            return None

        def nlp_party(text):
            entities = extract_entities(text)
            known_parties = PARTY_KEYWORDS
            for ent, label in entities:
                if label in {"ORG", "NORP"}:
                    best = process.extractOne(ent, known_parties)
                    if best and best[1] > 80:
                        return (best[0], best[1] / 100.0, "spacy_ner_fuzzy", "pass")
            return None

        def fuzzy_party(text):
            known_parties = PARTY_KEYWORDS
            best = process.extractOne(text, known_parties)
            if best and best[1] > 80:
                return (best[0], best[1] / 100.0, "fuzzy", "pass")
            return None

        value, score, method, result, used_method = self._extract_with_strategies(
            title,
            [("regex", regex_party), ("nlp", nlp_party), ("fuzzy", fuzzy_party)]
        )

        self.log_field_selection(
            field_type="party",
            field_name="party",
            extracted_value=value,
            method=used_method,
            score=score,
            result=result,
            context=contest,
            user_feedback=None,
            log_path="field_selection_log.jsonl"
        )
        return value
    
    def extract_panel(self, contest_title):
        """
        Extract the panel for a given contest title using regex, spaCy NER, and direct lookup.
        Log the extraction attempt and result.
        """
        panel_keywords = self.library.get("panel_tags", ["panel", "section", "container", "box", "area"])
        panel_pattern = "|".join([re.escape(k) for k in panel_keywords])

        def regex_panel(text):
            match = re.search(rf"({panel_pattern})", text, re.IGNORECASE)
            if match:
                return (match.group(1), 0.9, "regex", "pass")
            return None

        def nlp_panel(text):
            entities = extract_entities(text)
            for ent, label in entities:
                if label in {"ORG", "NORP"}:
                    return (ent, 0.85, "spacy_ner", "pass")
            return None

        def direct_lookup(text):
            panel = self.get_panel(text)
            if panel:
                return (panel, 1.0, "direct_lookup", "pass")
            return None

        value, score, method, result, used_method = self._extract_with_strategies(
            contest_title or "",
            [("regex", regex_panel), ("nlp", nlp_panel), ("direct_lookup", direct_lookup)]
        )

        self.log_field_selection(
            field_type="panel",
            field_name="panel",
            extracted_value=value,
            method=used_method,
            score=score,
            result=result,
            context={"contest_title": contest_title},
            user_feedback=None,
            log_path="field_selection_log.jsonl"
        )
        return value

    def extract_tables(self, contest_title):
        """
        Extract tables for a given contest title using regex, spaCy NER, and direct lookup.
        Log the extraction attempt and result.
        """
        table_keywords = self.library.get("table_tags", ["table", "results", "summary", "sheet", "spreadsheet", "grid"])
        table_pattern = "|".join([re.escape(k) for k in table_keywords])

        def regex_table(text):
            match = re.search(rf"({table_pattern})", text, re.IGNORECASE)
            if match:
                return ([match.group(1)], 0.9, "regex", "pass")
            return None

        def nlp_table(text):
            entities = extract_entities(text)
            for ent, label in entities:
                if label in {"ORG", "NORP"}:
                    return ([ent], 0.85, "spacy_ner", "pass")
            return None

        def direct_lookup(text):
            tables = self.get_tables(text)
            if tables:
                return (tables, 1.0, "direct_lookup", "pass")
            return None

        value, score, method, result, used_method = self._extract_with_strategies(
            contest_title or "",
            [("regex", regex_table), ("nlp", nlp_table), ("direct_lookup", direct_lookup)]
        )

        self.log_field_selection(
            field_type="tables",
            field_name="tables",
            extracted_value=value,
            method=used_method,
            score=score,
            result=result,
            context={"contest_title": contest_title},
            user_feedback=None,
            log_path="field_selection_log.jsonl"
        )
        return value

    def extract_precincts(self, state=None, county=None):
        """
        Extract known precincts for a state/county using regex, spaCy NER, and direct lookup.
        Log the extraction attempt and result.
        """
        location_pattern = "|".join([re.escape(k) for k in LOCATION_KEYWORDS])

        def regex_precinct(text):
            match = re.search(rf"({location_pattern})", text, re.IGNORECASE)
            if match:
                return ([match.group(1)], 0.9, "regex", "pass")
            return None

        def nlp_precinct(text):
            entities = extract_entities(text)
            for ent, label in entities:
                if label in {"ORG", "NORP"}:
                    return ([ent], 0.85, "spacy_ner", "pass")
            return None

        def direct_lookup(_):
            precincts = self.get_precincts(state=state, county=county)
            if precincts:
                return (precincts, 1.0, "direct_lookup", "pass")
            return None

        value, score, method, result, used_method = self._extract_with_strategies(
            state or county or "",
            [("regex", regex_precinct), ("nlp", nlp_precinct), ("direct_lookup", direct_lookup)]
        )

        self.log_field_selection(
            field_type="precincts",
            field_name="precincts",
            extracted_value=value,
            method=used_method,
            score=score,
            result=result,
            context={"state": state, "county": county},
            user_feedback=None,
            log_path="field_selection_log.jsonl"
        )
        return value

    def extract_states(self):
        """
        Extract all known states using regex, spaCy NER, and direct lookup.
        Log the extraction attempt and result.
        """
        state_keywords = self.library.get("state_tags", ["state", "province", "territory", "region"])
        state_pattern = "|".join([re.escape(k) for k in state_keywords])
        known_states = self.library.get("known_states", [])

        def regex_state(text):
            match = re.search(rf"({state_pattern})", text, re.IGNORECASE)
            if match:
                return (text, 0.9, "regex", "pass")
            return None

        def nlp_state(text):
            entities = extract_entities(text)
            for ent, label in entities:
                if label in {"ORG", "NORP"}:
                    return (ent, 0.85, "spacy_ner", "pass")
            return None

        def direct_lookup(_):
            states = self.get_states()
            if states:
                return (states, 1.0, "direct_lookup", "pass")
            return None

        # Try all known states
        found_states = []
        for s in known_states:
            value, score, method, result, used_method = self._extract_with_strategies(
                s,
                [("regex", regex_state), ("nlp", nlp_state)]
            )
            if value:
                found_states.append(value)
        if not found_states:
            value, score, method, result, used_method = self._extract_with_strategies(
                "", [("direct_lookup", direct_lookup)]
            )
            found_states = value if value else []
        else:
            score = 0.9
            method = "regex"
            result = "pass"
            used_method = "regex"

        self.log_field_selection(
            field_type="states",
            field_name="states",
            extracted_value=found_states,
            method=used_method,
            score=score,
            result=result,
            context={},
            user_feedback=None,
            log_path="field_selection_log.jsonl"
        )
        return found_states

    def extract_election_types(self):
        """
        Extract all known election types using regex, spaCy NER, and direct lookup.
        Log the extraction attempt and result.
        """
        known_types = self.library.get("election", [])
        location_pattern = "|".join([re.escape(k) for k in LOCATION_KEYWORDS])
        election_type_pattern = r"(primary|general|special|runoff|municipal|presidential|senate|mayoral|school board|" + location_pattern + ")"

        def regex_election_type(text):
            match = re.search(election_type_pattern, text, re.IGNORECASE)
            if match:
                return (match.group(1), 0.9, "regex", "pass")
            return None

        def nlp_election_type(text):
            entities = extract_entities(text)
            for ent, label in entities:
                if label in {"ORG", "NORP"}:
                    return (ent, 0.85, "spacy_ner", "pass")
            return None

        def direct_lookup(_):
            types = self.get_election_types()
            if types:
                return (types, 1.0, "direct_lookup", "pass")
            return None

        found_types = []
        for t in known_types:
            value, score, method, result, used_method = self._extract_with_strategies(
                t,
                [("regex", regex_election_type), ("nlp", nlp_election_type)]
            )
            if value:
                found_types.append(value)
        if not found_types:
            value, score, method, result, used_method = self._extract_with_strategies(
                "", [("direct_lookup", direct_lookup)]
            )
            found_types = value if value else []
        else:
            score = 0.9
            method = "regex"
            result = "pass"
            used_method = "regex"

        self.log_field_selection(
            field_type="election_types",
            field_name="election_types",
            extracted_value=found_types,
            method=used_method,
            score=score,
            result=result,
            context={},
            user_feedback=None,
            log_path="field_selection_log.jsonl"
        )
        return found_types

    def extract_years(self):
        """
        Extract all years found in contests using regex, spaCy NER, and direct lookup.
        Log the extraction attempt and result.
        """
        contests = self.get_contests()

        def regex_year(text):
            match = re.search(r"\b(19|20)\d{2}\b", text)
            if match:
                return (match.group(0), 0.9, "regex", "pass")
            return None

        def nlp_year(text):
            entities = extract_entities(text)
            for ent, label in entities:
                if label == "DATE" and re.match(r"\b(19|20)\d{2}\b", ent):
                    return (ent, 0.85, "spacy_ner", "pass")
            return None

        def direct_lookup(_):
            years = self.get_years()
            if years:
                return (years, 1.0, "direct_lookup", "pass")
            return None

        found_years = []
        for c in contests:
            title = str(c.get("title", ""))
            value, score, method, result, used_method = self._extract_with_strategies(
                title,
                [("regex", regex_year), ("nlp", nlp_year)]
            )
            if value:
                found_years.append(value)
        if not found_years:
            value, score, method, result, used_method = self._extract_with_strategies(
                "", [("direct_lookup", direct_lookup)]
            )
            found_years = value if value else []
        else:
            score = 0.9
            method = "regex"
            result = "pass"
            used_method = "regex"

        self.log_field_selection(
            field_type="years",
            field_name="years",
            extracted_value=found_years,
            method=used_method,
            score=score,
            result=result,
            context={},
            user_feedback=None,
            log_path="field_selection_log.jsonl"
        )
        return found_years

    def extract_buttons(self, contest_title=None, keyword=None, url=None):
        """
        Extract button labels using regex, spaCy NER (ORG/NORP), and direct lookup.
        Log the extraction attempt and result.
        """
        button_keywords = self.library.get("button_tags", [
            "Show Results", "Vote", "Submit", "Summary", "Next", "Continue", "Back",
            "Download", "Print", "Details", "Results", "Ballot", "Cast Vote"
        ])
        button_pattern = "|".join([re.escape(k) for k in button_keywords])

        sources = [contest_title or "", keyword or "", url or ""]

        def regex_button(text):
            match = re.search(rf"({button_pattern})", text, re.IGNORECASE)
            if match:
                return (match.group(1), 0.9, "regex", "pass")
            return None

        def nlp_button(text):
            entities = extract_entities(text)
            for ent, label in entities:
                if label in {"ORG", "NORP"}:
                    return (ent, 0.85, "spacy_ner", "pass")
            return None

        def direct_lookup(_):
            candidates = []
            buttons = self.get_buttons(contest_title=contest_title, keyword=keyword, url=url)
            for btn in buttons:
                if not isinstance(btn, dict):
                    continue
                label = btn.get("label")
                if label:
                    candidates.append(label)
            if candidates:
                return (list(dict.fromkeys(candidates)), 1.0, "direct_lookup", "pass")
            return None

        found_buttons = []
        for src in sources:
            value, score, method, result, used_method = self._extract_with_strategies(
                src,
                [("regex", regex_button), ("nlp", nlp_button)]
            )
            if value:
                found_buttons.append(value)
        if not found_buttons:
            value, score, method, result, used_method = self._extract_with_strategies(
                "", [("direct_lookup", direct_lookup)]
            )
            found_buttons = value if value else []
        else:
            score = 0.9
            method = "regex"
            result = "pass"
            used_method = "regex"

        # Deduplicate
        if isinstance(found_buttons, list):
            found_buttons = list(dict.fromkeys(found_buttons))

        self.log_field_selection(
            field_type="buttons",
            field_name="buttons",
            extracted_value=found_buttons,
            method=used_method,
            score=score,
            result=result,
            context={"contest_title": contest_title, "keyword": keyword, "url": url},
            user_feedback=None,
            log_path="field_selection_log.jsonl"
        )
        return found_buttons

    def score_header(self, title, context=None):
        # Simple fallback: just call score_entry or return a default score
        return self.score_entry(title) if hasattr(self, "score_entry") else 0.5

    # --- Data Accessors ---
    def get_contests(self, filters=None):
        """
        Return contests, optionally filtered by state, county, year, type, etc.
        """   
        if not isinstance(self.organized, dict):
            return []
        contests = self._safe_get(self.organized, "contests", [])
        if not filters:
            return clean_for_json(contests)
        def match(c):
            for k, v in filters.items():
                if not isinstance(c, dict):
                    return False
                if str(c.get(k, "")).lower() != str(v).lower():
                    return False
            return True
        return clean_for_json([c for c in contests if match(c)])

    def get_buttons(self, contest_title=None, keyword=None, url=None):
        """
        Return all buttons, or those for a specific contest, or matching a keyword/URL.
        First, check the button selection log for a successful match.
        """
        # 1. Check button selection log for a successful match
        os.makedirs(LOG_DIR, exist_ok=True)
        log_path = os.path.join(LOG_DIR, "button_selection_log.jsonl")
        if os.path.exists(log_path):
            with open(log_path, "rb") as f:
                for line in f:
                    try:
                        entry = orjson.loads(line)
                    except Exception:
                        continue
                    if not isinstance(entry, dict):
                        continue
                    # Check for a successful result for this contest_title/keyword/url
                    if contest_title and entry.get("contest_title") == contest_title and entry.get("result", "").startswith("pass"):
                        # Reconstruct a button dict from the log entry
                        button = {
                            "label": entry.get("button_label"),
                            "selector": entry.get("selector"),
                            # Optionally add more fields if you log them
                        }
                        return clean_for_json([button])
                    if keyword and keyword.lower() in (entry.get("button_label") or "").lower() and entry.get("result", "").startswith("pass"):
                        button = {
                            "label": entry.get("button_label"),
                            "selector": entry.get("selector"),
                        }
                        return clean_for_json([button])
                    if url and url in (entry.get("selector") or "") and entry.get("result", "").startswith("pass"):
                        button = {
                            "label": entry.get("button_label"),
                            "selector": entry.get("selector"),
                        }
                        return clean_for_json([button])

        # 2. Fallback to existing logic
        if not self.organized:
            return []
        buttons_dict = self.organized.get("buttons", {})
        results = []

        # By contest title (exact match)
        if contest_title and isinstance(contest_title, str):
            results = buttons_dict.get(contest_title, [])
            if results:
                return clean_for_json(results)

        # By keyword in label or selector
        if keyword:
            keyword = keyword.lower()
            for btn_list in buttons_dict.values():
                for btn in btn_list:
                    if not isinstance(btn, dict):
                        continue
                    if keyword in btn.get("label", "").lower() or keyword in btn.get("selector", "").lower():
                        results.append(btn)
            if results:
                return clean_for_json(results)

        # By URL (if you want to associate buttons with URLs)
        if url:
            for btn_list in buttons_dict.values():
                for btn in btn_list:
                    if not isinstance(btn, dict):  
                        continue
                    if url in btn.get("selector", ""):
                        results.append(btn)
            if results:
                return clean_for_json(results)

        # Fallback: return all buttons
        all_buttons = []
        for btns in buttons_dict.values():
            all_buttons.extend(btns)
        return clean_for_json(all_buttons)

    def matches_html_label_pattern(label, patterns):
        """Check if label matches any HTML-specific regex pattern."""
        for pat in patterns:
            if re.search(pat, label, re.IGNORECASE):
                return True
        return False

    def log_pattern_attempt(self, label, pattern, result, context=None):
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
        contest_title,
        keywords,
        context=None,
        fuzzy_thresholds=None,
        prompt_user_for_button=None,
        confirm_button_callback=None,
        learning_mode=True
    ):
        """
        Advanced button selection: combines memory, DOM, semantic similarity, adaptive threshold, and feedback.
        Now supports confirmation, exclusion of rejected buttons, and learning mode (auto-apply corrections from log/DB).
        """
        if fuzzy_thresholds is None:
            fuzzy_thresholds = [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5]
        model = self._semantic_model
        context = context or {}
        context.update({
            "contest_title": contest_title,
            "year": context.get("year", ""),
            "election_types": context.get("election_types", ""),
            "county": context.get("county", ""),
            "state": context.get("state", "")
        })

        # --- 1. Learning mode: check log/DB for confirmed button ---
        if learning_mode:
            learned_btn = self._get_confirmed_button_from_log(contest_title, keywords, context)
            if isinstance(learned_btn, dict) and learned_btn.get("selector") not in self.clicked_button_selectors:
                selector_html = learned_btn.get("selector", "")
                dom_candidates = []
                # Use a broad selector for all clickable elements
                BUTTON_SELECTORS = (
                    "button, a[href], [role='button'], input[type='button'], input[type='submit'], "
                    "[tabindex]:not([tabindex='-1'])"
                )
                button_features = page.locator(BUTTON_SELECTORS)
                for i in range(button_features.count()):
                    btn = button_features.nth(i)
                    try:
                        btn_html = btn.evaluate("el => el.outerHTML")
                        if btn_html == selector_html:
                            learned_btn["element_handle"] = btn
                            learned_btn["is_visible"] = btn.is_visible()
                            learned_btn["is_clickable"] = btn.is_enabled()
                            break
                    except Exception:
                        continue
                if (
                    isinstance(learned_btn, dict)
                    and learned_btn.get("element_handle")
                    and learned_btn.get("is_visible")
                    and learned_btn.get("is_clickable")
                ):
                    rprint(f"[green][LEARNING] Auto-applying learned button: {learned_btn.get('label')}[/green]")
                    try:
                        learned_btn["element_handle"].click()
                        page.wait_for_timeout(1500)
                        self.clicked_button_selectors.add(learned_btn.get("selector"))
                        return learned_btn, 0
                    except Exception:
                        rprint("[red][ERROR] Failed to click learned button element.[/red]")
                else:
                    rprint("[red][ERROR] No element_handle found for the learned button candidate.[/red]")

        # --- 2. Gather candidates from memory/log ---
        memory_candidates = []
        logged_buttons = self.get_buttons(contest_title=contest_title)
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
        button_features = page.locator(BUTTON_SELECTORS)

        def scan_btn(btn, i):
            try:
                # Robust label extraction
                label = btn.inner_text() or ""
                if not label:
                    # Try aria-label or value attribute
                    label = btn.get_attribute("aria-label") or btn.get_attribute("value") or ""
                class_name = btn.get_attribute("class") or ""
                role = btn.get_attribute("role") or ""
                tag = btn.evaluate("el => el.tagName").lower()
                is_visible = btn.is_visible()
                is_enabled = btn.is_enabled()
                selector = btn.evaluate("el => el.outerHTML") if btn else ""
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
                self._log_button_memory(candidate, contest_title, "scanned")
            except Exception:
                pass

        scan_buttons_with_progress([button_features.nth(i) for i in range(button_features.count())], scan_callback=scan_btn)

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
                            rprint(f"[bold green][Coordinator] Confirmed button: '{cand.get('label')}' (score={cand.get('combined_score', 0):.2f})[/bold green]")
                            self._log_button_memory(cand, contest_title, f"confirmed_pass_{cand.get('combined_score', 0):.2f}")
                            if not isinstance(cand, dict):
                                rprint(f"[red][ERROR] Candidate is not a dict: {cand}[/red]")
                                continue
                            if learning_mode:
                                self._log_confirmed_button_for_learning(cand, contest_title, context)
                            self.clicked_button_selectors.add(cand.get("selector"))
                            try:
                                cand["element_handle"].click()
                                page.wait_for_timeout(1500)
                            except Exception:
                                pass
                            return cand, idx
                        else:
                            excluded_labels.add(cand.get("label"))
                            rprint(f"[yellow][Coordinator] Button '{cand.get('label')}' rejected, retrying...[/yellow]")
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
                self._log_button_memory(chosen_btn, contest_title, "manual_correction")
                if learning_mode:
                    self._log_confirmed_button_for_learning(chosen_btn, contest_title, context)
                return chosen_btn, chosen_idx

        rprint(f"[red][ERROR] No suitable button could be clicked for '{context.get('toggle_name', '')}'.[/red]")
        return None, None

    def _log_confirmed_button_for_learning(self, button, contest_title, context):
        """
        Log confirmed button for learning mode (auto-apply next time).
        """
        # Ensure button is a dict before using .get()
        if not isinstance(button, dict):
            return
        log_entry = {
            "contest_title": contest_title,
            "button_label": button.get("label"),
            "selector": button.get("selector"),
            "context": context,
            "result": "learning_confirmed"
        }
        os.makedirs(LOG_DIR, exist_ok=True)
        log_path = os.path.join(LOG_DIR, "button_learning_log.jsonl")
        with open(log_path, "ab") as f:
            f.write(orjson.dumps(clean_for_json(log_entry)) + b"\n")

    def _get_confirmed_button_from_log(self, contest_title, keywords, context):
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
                if entry.get("contest_title") == contest_title and entry.get("result") == "learning_confirmed":
                    return {
                        "label": entry.get("button_label"),
                        "selector": entry.get("selector"),
                        "context": entry.get("context"),
                        "source": "learning"
                    }
        return None

    def _log_button_memory(self, button, contest_title, result):
        """
        Log button selection attempts for future ML or rule improvements.
        """
        # Ensure button is a dict before using .get()
        if not isinstance(button, dict):
            return
        log_entry = {
            "contest_title": contest_title,
            "button_label": button.get("label"),
            "selector": button.get("selector"),
            "result": result
        }
        os.makedirs(LOG_DIR, exist_ok=True)
        log_path = os.path.join(LOG_DIR, "button_selection_log.jsonl")
        with open(log_path, "ab") as f:
            f.write(orjson.dumps(clean_for_json(log_entry)) + b"\n")

    # --- Table structure learning/lookup ---
    def get_table_structure(self, contest_title, context=None, learning_mode=True):
        """
        Retrieve or learn the expected table structure for a contest.
        """
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
                    if entry.get("contest_title") == contest_title and entry.get("result") == "learning_confirmed":
                        return clean_for_json(entry.get("headers"), [])
        # 2. Fallback: return None (caller should trigger extraction and confirmation)
        return None

    def log_table_structure(self, contest_title, headers, context=None):
        """
        Log confirmed table structure for learning mode.
        """
        log_entry = {
            "contest_title": contest_title,
            "headers": headers,
            "context": context,
            "result": "learning_confirmed"
        }
        os.makedirs(LOG_DIR, exist_ok=True)
        log_path = os.path.join(LOG_DIR, "table_structure_learning_log.jsonl")
        with open(log_path, "ab") as f:
            f.write(orjson.dumps(clean_for_json(log_entry)) + b"\n")

    # --- CLI for reviewing/editing corrections and feedback ---
    def review_and_edit_corrections(self, field_type="buttons"):
        """
        Launch the manual_correction_bot CLI for reviewing/editing corrections and feedback.
        """
        script_path = os.path.join(os.path.dirname(__file__), "..", "bots", "manual_correction_bot.py")
        subprocess.run(["python", script_path, "--fields", field_type, "--feedback", "--enhanced"], check=True, cwd=PROJECT_ROOT)

    # --- Learning mode: auto-apply corrections from log/database ---
    def enable_learning_mode(self):
        """
        Enable learning mode for auto-applying corrections from logs/database.
        """
        self.learning_mode = True

    def disable_learning_mode(self):
        """
        Disable learning mode.
        """
        self.learning_mode = False
            
    def get_panel(self, contest_title):
        """
        Retrieve the panel for a given contest title.
        """
        if not isinstance(self.organized, dict):
            return None
        panels = self.organized.get("panels", {})
        if not isinstance(panels, dict):
            return None
        return clean_for_json(panels.get(contest_title))

    def get_tables(self, contest_title):
        """
        Retrieve tables for a given contest title.
        """
        if not isinstance(self.organized, dict):
            return []
        tables = self.organized.get("tables", {})
        if not isinstance(tables, dict):
            return []
        return clean_for_json(tables.get(contest_title, []))

    def get_candidates(self, contest_title=None):
        """
        Extract candidate names from contest entities or table headers.
        """
        candidates = set()
        contests = self.get_contests() if contest_title is None else [
            c for c in self.get_contests() if isinstance(c, dict) and c.get("title") == contest_title
        ]
        for c in contests:
            if not isinstance(c, dict):
                continue
            for ent, label in c.get("entities", []):
                if label in {"PERSON", "CANDIDATE"}:
                    candidates.add(ent)
            # Optionally: parse table headers for candidate names
            for tbl in self.get_tables(c.get("title", "")):
                if not isinstance(tbl, dict):
                    continue
                headers = tbl.get("headers", [])
                for h in headers:
                    if isinstance(h, str) and "candidate" in h.lower():
                        candidates.add(h)
        return clean_for_json(list(candidates))

    def get_precincts(self, state=None, county=None):
        """
        Return known precincts for a state/county from the library.
        """       
        if not isinstance(self.library, dict):
            return []
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
        return clean_for_json(self.library.get("known_precincts", []))

    def get_states(self):
        """
        Return all known states from the library.
        """
        if not isinstance(self.library, dict):
            return []
        return clean_for_json(self.library.get("known_states", []))

    def get_election_types(self):
        """
        Return all known election types from the library.
        """       
        if not isinstance(self.library, dict):
            return []
        return clean_for_json(self.library.get("election", []))

    def get_years(self):
        """
        Return all years found in contests.
        """      
        contests = self.get_contests()
        return clean_for_json(sorted({c.get("year") for c in contests if isinstance(c, dict) and c.get("year")}))

    # --- Integrity & Anomaly Checks ---

    def _log_get_contests_access(self, filters):
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "method": "get_contests",
            "filters": filters,
        }
        os.makedirs(LOG_DIR, exist_ok=True)
        log_path = os.path.join(LOG_DIR, "get_contests_access_log.jsonl")
        with open(log_path, "ab") as f:
            f.write(orjson.dumps(clean_for_json(log_entry)) + b"\n")

    def _log_get_buttons_access(self, contest_title, keyword, url):
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "method": "get_buttons",
            "contest_title": contest_title,
            "keyword": keyword,
            "url": url,
        }

        os.makedirs(LOG_DIR, exist_ok=True)
        log_path = os.path.join(LOG_DIR, "get_buttons_access_log.jsonl")
        with open(log_path, "ab") as f:
            f.write(orjson.dumps(clean_for_json(log_entry)) + b"\n")

    def _log_get_best_button_access(self, contest_title, keywords, class_hint, url):
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "method": "get_best_button",
            "contest_title": contest_title,
            "keywords": keywords,
            "class_hint": class_hint,
            "url": url,
        }
        os.makedirs(LOG_DIR, exist_ok=True)
        log_path = os.path.join(LOG_DIR, "get_best_button_access_log.jsonl")
        with open(log_path, "ab") as f:
            f.write(orjson.dumps(clean_for_json(log_entry)) + b"\n")

    def _log_get_panel_access(self, contest_title):
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "method": "get_panel",
            "contest_title": contest_title,
        }
        os.makedirs(LOG_DIR, exist_ok=True)
        log_path = os.path.join(LOG_DIR, "get_panel_access_log.jsonl")
        with open(log_path, "ab") as f:
            f.write(orjson.dumps(clean_for_json(log_entry)) + b"\n")

    def _log_get_tables_access(self, contest_title):
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "method": "get_tables",
            "contest_title": contest_title,
        }
        os.makedirs(LOG_DIR, exist_ok=True)
        log_path = os.path.join(LOG_DIR, "get_tables_access_log.jsonl")
        with open(log_path, "ab") as f:
            f.write(orjson.dumps(clean_for_json(log_entry)) + b"\n")

    def _log_get_candidates_access(self, contest_title):
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "method": "get_candidates",
            "contest_title": contest_title,
        }
        os.makedirs(LOG_DIR, exist_ok=True)
        log_path = os.path.join(LOG_DIR, "get_candidates_access_log.jsonl")
        with open(log_path, "ab") as f:
            f.write(orjson.dumps(clean_for_json(log_entry)) + b"\n")

    def _log_get_precincts_access(self, state, county):
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

    def _log_get_states_access(self):
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "method": "get_states",
        }
        os.makedirs(LOG_DIR, exist_ok=True)
        log_path = os.path.join(LOG_DIR, "get_states_access_log.jsonl")
        with open(log_path, "ab") as f:
            f.write(orjson.dumps(clean_for_json(log_entry)) + b"\n")

    def _log_get_election_types_access(self):
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "method": "get_election_types",
        }
        os.makedirs(LOG_DIR, exist_ok=True)
        log_path = os.path.join(LOG_DIR, "get_election_types_access_log.jsonl")
        with open(log_path, "ab") as f:
            f.write(orjson.dumps(clean_for_json(log_entry)) + b"\n")

    def _log_get_years_access(self):
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "method": "get_years",
        }
        os.makedirs(LOG_DIR, exist_ok=True)
        log_path = os.path.join(LOG_DIR, "get_years_access_log.jsonl")
        with open(log_path, "ab") as f:
            f.write(orjson.dumps(clean_for_json(log_entry)) + b"\n")

    # --- Reporting ---

    def report_summary(self):
        """
        Print a summary of contests, entities, locations, and integrity issues.
        """  
        contests = self.get_contests()
        rprint(f"[bold cyan][COORDINATOR] {len(contests)} contests loaded[/bold cyan]")
        all_entities = set()
        all_labels = set()
        for c in contests:
            if not isinstance(c, dict):
                continue
            for ent, label in c.get("entities", []):
                all_entities.add(ent)
                all_labels.add(label)
        rprint(f"Unique entity labels: {sorted(all_labels)}")
        rprint(f"Unique entities: {sorted(all_entities)}")
        # Show states and years
        rprint(f"States: {sorted({c.get('state') for c in contests if c.get('state')})}")
        rprint(f"Years: {sorted({c.get('year') for c in contests if c.get('year')})}")
        # Integrity issues
        issues = self.validate_and_check_integrity()
        if issues["integrity_issues"]:
            rprint(f"[yellow]Integrity issues:[/yellow] {issues['integrity_issues']}")
        if issues["anomalies"]:
            rprint(f"[red]Anomalies detected:[/red] {issues['anomalies']}")
     
    # --- Dynamic Data for Downstream Consumers ---

    def get_for_selector(self):
        """
        Return contests, buttons, and patterns for contest_selector.
        """      
        if not isinstance(self.library, dict):
            noisy_patterns = []
        else:
            noisy_patterns = self.library.get("default_noisy_label_patterns", [])
        return clean_for_json({
            "contests": self.get_contests(),
            "buttons": self.get_buttons(),
            "noisy_patterns": noisy_patterns
        })

    def get_for_table_builder(self):
        """
        Return precinct headers and table tags for table_builder.
        """        
        if not isinstance(self.library, dict):
            precinct_headers = []
            table_tags = []
        else:
            precinct_headers = self.library.get("precinct_header_tags", [])
            table_tags = self.library.get("table_tags", [])
        return clean_for_json({
            "precinct_headers": precinct_headers,
            "table_tags": table_tags
        })

    def get_for_html_handler(self):
        """
        Return panel tags, contest panel tags, and selectors for html_handler.
        """       
        if not isinstance(self.library, dict):
            panel_tags = []
            contest_panel_tags = []
            all_selectors = []
        else:
            panel_tags = self.library.get("panel_tags", [])
            contest_panel_tags = self.library.get("contest_panel_tags", [])
            selectors = self.library.get("selectors", {})
            if not isinstance(selectors, dict):
                all_selectors = []
            else:
                all_selectors = selectors.get("all_selectors", [])
        return clean_for_json({
            "panel_tags": panel_tags,
            "contest_panel_tags": contest_panel_tags,
            "all_selectors": all_selectors
        })

    def get_for_state_router(self):
        """
        Return state_module_map for state_router.
        """       
        if not isinstance(self.library, dict):
            return clean_for_json({})
        return clean_for_json(self.library.get("state_module_map", {}))

    def correct_and_update_contest(self, contest_id, correction_data):
        """
        Update a contest in the DB and context library, then re-organize context.
        """
        from ..utils.db_utils import update_contest_in_db

        # 1. Update DB
        update_contest_in_db({"id": contest_id, **correction_data})

        # 2. Update context library if needed
        if not isinstance(self.library, dict):
            return
        for key, value in correction_data.items():
            if key == "county":
                known_counties = self.library.get("known_counties", [])
                if value not in known_counties:
                    self.library.setdefault("known_counties", []).append(value)
            if key == "state":
                known_states = self.library.get("known_states", [])
                if value not in known_states:
                    self.library.setdefault("known_states", []).append(value)
            # Add similar logic for other fields as needed

        # 3. Save updated context library (if you persist it)
        update_context_library(CONTEXT_LIBRARY_PATH, lambda lib: lib.update(self.library))

        # 4. Re-organize context
        self.organized = None
        # Optionally, re-run organize_and_enrich if you want to refresh immediately:
        self.organize_and_enrich(self.last_raw_context)

        # 5. Log correction
        self.log_field_selection(
            field_type="contest",
            field_name="correction",
            extracted_value=correction_data,
            method="manual",
            score=1.0,
            result="manual_pass",
            context={"contest_id": contest_id},
            user_feedback=None
        )       

    def validate_and_check_integrity(self, expected_year=None):
        """
        Run all integrity checks and anomaly detection on contest data.
        Returns a dict with issues, anomalies, clusters, and advanced validation.
        """  
        contests = self.get_contests()
        integrity_issues = election_integrity_checks(contests)
        advanced_issues = advanced_cross_field_validation(contests)
        anomalies, clusters = detect_anomalies_with_ml(contests)
        # Optionally plot clusters and anomalies
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
                le_state.transform([c.get("state", "unknown")])[0],
                le_county.transform([c.get("county", "unknown")])[0],
                int(c.get("year", 0)) if str(c.get("year", "0")).isdigit() else 0,
                len(str(c.get("title", ""))),
            ])
        X = np.array(features)
        print_integrity_summary(contests, expected_year, X=X)
        # Cross-check with expected year
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
