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
from fuzzywuzzy import process
from ..utils.shared_logger import SharedLogger
import difflib
from ..utils.shared_logic import (
    scan_buttons_with_progress, keyphrase_match,
    normalize_state_name, normalize_county_name
)
from ..bots.librarian import ( 
    PARTY_KEYWORDS,
    LOCATION_KEYWORDS,
    STATE_MODULE_MAP,
    KNOWN_STATE_TO_COUNTY_MAP,
    KNOWN_COUNTY_TO_PRECINCTS_MAP,
    ELECTION_TYPES,
    TABLE_TAGS,
    PANEL_TAGS,
    STATE_TAGS,
    BUTTON_TAGS, atomic_write_json 
)
from sklearn.preprocessing import LabelEncoder
import subprocess
from ..config import PROJECT_ROOT, LOG_DIR, CONTEXT_LIBRARY_PATH
import threading

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
from ..services.election_data_services import ElectionDataService
import inspect
from typing import Optional, Any, List, Dict, Tuple, Callable
logger = SharedLogger()
def _sanitize_log_filename(name: str) -> str:
    # Only allow alphanumeric, underscore, and dash
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', name) 

def get_semantic_score(model, text1, text2) -> float:
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
) -> List[Dict[str, Any]]:
    """
    Merge memory and DOM candidates, deduplicate, and rank by combined fuzzy and semantic score.
    """
    seen = set()
    all_candidates = []
    for cand in memory_candidates + dom_candidates:
        if not isinstance(cand, dict) or not cand.get("label"):
            continue
        key = (cand.get("label", ""), cand.get("selector", ""))
        if key not in seen:
            seen.add(key)
            all_candidates.append(cand)

    contest = context.get("contest", {})
    contest = contest.get("title", "") if contest else ""
    context_str = " ".join([
        contest,
        str(context.get("year", "")),
        str(context.get("type_", "")),
        str(context.get("county", "")),
        str(context.get("state", "")),
    ]).strip()

    expected_class = context.get("expected_class", "")
    expected_tag = context.get("expected_tag", "")

    for cand in all_candidates:
        if not isinstance(cand, dict):
            continue
        label = cand.get("label", "") or ""
        # Strong full-string match
        full_match = int(label.strip().lower() == contest.strip().lower())
        # Keyphrase-aware match
        keyphrase_score = 0.0
        for kw in (keywords or []):
            if keyphrase_match(label, kw, min_words=2, fuzzy_cutoff=0.85) or keyphrase_match(label, kw, min_words=2, fuzzy_cutoff=0.85):
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
        if context_heading and contest:
            context_proximity = get_semantic_score(model, contest, context_heading)
        # Hierarchy/class/tag bonus
        hierarchy_score = 0.0
        if expected_class and expected_class in cand.get("class", "") or expected_class in cand.get("class", "").lower():
            hierarchy_score += 0.5
        if expected_tag and expected_tag == cand.get("tag", "") or expected_tag in cand.get("tag", "").lower():
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

def call_handler_with_coordinator(handler, *args, coordinator=None, **kwargs) -> Any:

    sig = inspect.signature(handler.parse)
    if 'coordinator' in sig.parameters:
        return handler.parse(*args, coordinator, **kwargs)
    else:
        return handler.parse(*args, **kwargs)

def dynamic_state_county_detection(context, html, debug=False) -> tuple:
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
    state_to_county_values = state_to_county.values() if isinstance(state_to_county, dict) else state_to_county
    all_counties = {normalize_county_name(c) for counties in state_to_county_values for c in counties}

    county_to_precinct_values = county_to_precinct.values() if isinstance(county_to_precinct, dict) else county_to_precinct
    all_precincts = {normalize_county_name(d) for precincts in county_to_precinct_values for d in precincts}

    # --- 1. Try context fields directly (normalize and validate) ---
    if not isinstance(context, dict) or not context:
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
            logger.info("[dynamic_state_county_detection]", log)
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
            from ..utils.model_registry import ModelRegistry
            self._semantic_model = ModelRegistry.get_sentence_transformer("all-MiniLM-L6-v2")

        if alert_monitor:
            self.start_alert_monitoring()
            
    def __del__(self) -> None:
        """
        Ensure alert monitoring thread is cleaned up on destruction.
        """
        try:
            if self.alert_monitor_thread and self.alert_monitor_thread.is_alive():
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
            self.alert_monitor_thread = None

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

    def _extract_with_strategies(self, text, strategies) -> tuple:
        """
        Try a list of (method, function) strategies on text, returning the first successful result.
        Each function should return (value, score, method, result) or None.
        """
        for method, func in strategies:
            result = func(text)
            if result and result[0]:
                return result + (method,)
        return (None, 0.0, "fail", "none")

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
                    try:
                        self.data_service.upsert_contest(contest)
                    except Exception as e:
                        logger.error(f"[update_db_with_context] Failed to upsert contest: {contest.get('title', '')} - {e}")

            # --- Update table structures (legacy and ML-inferred) ---
            if update_tables and "tables" in library:
                for contest, tables in library["tables"].items():
                    for tbl in tables:
                        headers = tbl.get("headers") or tbl.get("columns") or []
                        context = tbl.get("context") or {}
                        ml_confidence = tbl.get("ml_confidence")
                        confirmed_by_user = tbl.get("confirmed_by_user", False)
                        try:
                            self.save_table_structure_to_db(
                                contest, headers, context, ml_confidence, confirmed_by_user
                            )
                        except Exception as e:
                            logger.error(f"[update_db_with_context] Failed to save table structure for {contest.get('title', '')}: {e}")

            # --- Update panels ---
            if update_panels and "panels" in library:
                for contest, panel in library["panels"].items():
                    try:
                        self.data_service.upsert_panel(contest, panel)
                    except Exception as e:
                        logger.error(f"[update_db_with_context] Failed to upsert panel for {contest.get('title', '')}: {e}")

            # --- Update buttons ---
            if update_buttons and "buttons" in library:
                for contest, buttons in library["buttons"].items():
                    for btn in buttons:
                        try:
                            self.data_service.upsert_button(contest, btn)
                        except Exception as e:
                            logger.error(f"[update_db_with_context] Failed to upsert button for {contest.get('title', '')}: {e}")

            # --- Update candidates ---
            if update_candidates and "candidates" in library:
                for candidate in library["candidates"]:
                    try:
                        self.data_service.upsert_candidate(candidate)
                    except Exception as e:
                        logger.error(f"[update_db_with_context] Failed to upsert candidate: {candidate.get('name', '')} - {e}")

            # --- Update parties ---
            if update_parties and "parties" in library:
                for party in library["parties"]:
                    try:
                        self.data_service.upsert_party(party)
                    except Exception as e:
                        logger.error(f"[update_db_with_context] Failed to upsert party: {party.get('name', '')} - {e}")

            # --- Update offices ---
            if update_offices and "offices" in library:
                for office in library["offices"]:
                    try:
                        self.data_service.upsert_office(office)
                    except Exception as e:
                        logger.error(f"[update_db_with_context] Failed to upsert office: {office.get('name', '')} - {e}")

            # --- Update districts ---
            if update_districts and "districts" in library:
                for district in library["districts"]:
                    try:
                        self.data_service.upsert_district(district)
                    except Exception as e:
                        logger.error(f"[update_db_with_context] Failed to upsert district: {district.get('name', '')} - {e}")

            # --- Update results ---
            if update_results and "results" in library:
                for result in library["results"]:
                    try:
                        self.data_service.upsert_result(result)
                    except Exception as e:
                        logger.error(f"[update_db_with_context] Failed to upsert result: {result.get('id', '')} - {e}")

            # --- Update entities (generic/misc entities) ---
            if update_entities and "entities" in library:
                for entity in library["entities"]:
                    try:
                        self.data_service.upsert_entity(entity)
                    except Exception as e:
                        logger.error(f"[update_db_with_context] Failed to upsert entity: {entity.get('value', '')} - {e}")

            # --- Update table_structures (ML-inferred/user-confirmed) ---
            if update_table_structures and "table_structures" in library:
                for ts in library["table_structures"]:
                    try:
                        self.data_service.upsert_table_structure(ts)
                    except Exception as e:
                        logger.error(f"[update_db_with_context] Failed to upsert table_structure: {ts.get('contest', '')} - {e}")

            # --- Update batch_metadata ---
            if update_batch_metadata and "batch_metadata" in library:
                for batch in library["batch_metadata"]:
                    try:
                        self.data_service.upsert_batch_metadata(batch)
                    except Exception as e:
                        logger.error(f"[update_db_with_context] Failed to upsert batch_metadata: {batch.get('batch_id', '')} - {e}")

            # --- Update alerts ---
            if update_alerts and "alerts" in library:
                for alert in library["alerts"]:
                    try:
                        self.data_service.upsert_alert(alert)
                    except Exception as e:
                        logger.error(f"[update_db_with_context] Failed to upsert alert: {alert.get('id', '')} - {e}")

            # --- Update embeddings (ML segment cache) ---
            if update_embeddings and "embeddings" in library:
                for emb in library["embeddings"]:
                    try:
                        self.data_service.upsert_embedding(emb)
                    except Exception as e:
                        logger.error(f"[update_db_with_context] Failed to upsert embedding: {emb.get('segment_hash', '')} - {e}")

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

    def save_table_structure_to_db(self, contest: Dict[str, Any], headers: Dict[str, Any], context: Dict[str, Any], ml_confidence: Optional[float] = None, confirmed_by_user: bool = False) -> Dict[str, Any]:
        from .context_organizer import save_table_structure_to_db
        return save_table_structure_to_db(contest, headers, context, ml_confidence, confirmed_by_user)

    def get_table_structure_from_db(self, contest: Dict[str, Any], context: Dict[str, Any] = None) -> Optional[List[Dict[str, Any]]]:
        from .context_organizer import get_table_structure_from_db
        return get_table_structure_from_db(contest, context)

    def organize_and_enrich(self, raw_context, **kwargs) -> Dict[str, Any]:
        """
        Organize raw context (from HTML/DOM or DB), deduplicate, cluster, and enrich with NLP.
        """
        self.last_raw_context = raw_context
        result = self.organizer.organize_context(raw_context, **kwargs)
        self.organized = result["organized"]
        self._enrich_contests_with_nlp()
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
        
    def get_feedback_pattern_kb(self, log_path=None, deduplicate=True, min_fields=("pattern_id", "label", "html")) -> list:
        """
        Load and return feedback pattern KB entries from the feedback log.
        - log_path: Optional path override (defaults to segment_feedback_log.jsonl in LOG_DIR)
        - deduplicate: If True, deduplicate by pattern_id or segment_hash
        - min_fields: Tuple of required fields for a valid entry
        Returns a list of dicts, each representing a feedback KB entry.
        """
        import orjson
        import os

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
        # Use coordinator's own resources if not provided
        context_library = context_library or getattr(self, "library", None)
        context_cache = context_cache or getattr(self, "context_cache", None)
        pattern_kb = pattern_kb or getattr(self, "pattern_kb", None)
        model = model or getattr(self, "_semantic_model", None)

        # 1. Try cache/context library for a direct match
        segment_hash = segment.get("segment_hash")
        if context_library and segment_hash:
            cached_segments = context_library.get("cached_segments", [])
            for entry in cached_segments:
                if entry.get("segment_hash") == segment_hash and entry.get("ml_label"):
                    return entry["ml_label"]

        # 2. Try pattern KB for embedding similarity
        if pattern_kb and model and "html" in segment:
            from ..utils.html_scanner import get_segment_embedding
            seg_emb = get_segment_embedding(model, segment)
            if seg_emb is not None:
                best_score = 0
                best_label = None
                for pat in pattern_kb:
                    pat_emb = pat.get("embedding")
                    if pat_emb is not None:
                        # Cosine similarity
                        score = float(np.dot(seg_emb, pat_emb) / (np.linalg.norm(seg_emb) * np.linalg.norm(pat_emb)))
                        if score > best_score and score >= ml_threshold:
                            best_score = score
                            best_label = pat.get("label")
                if best_label:
                    return best_label

        # 3. Try DOM grouping by label
        dom_parts = self.get_dom_parts()
        if dom_parts and "all_nodes" in dom_parts:
            all_nodes = dom_parts["all_nodes"]
            # Use the same normalization/hash as above
            for node in all_nodes:
                if node.get("html") == segment.get("html") and node.get("ml_label") and node.get("ml_confidence", 0) >= ml_threshold:
                    return node["ml_label"]
            # Try grouping by label field
            grouped = self.group_dom_nodes_by_label(label_field="ml_label")
            for label, nodes in grouped.items():
                for node in nodes:
                    if node.get("html") == segment.get("html"):
                        return label

        # 4. Try merge_and_rank_candidates if segment is a candidate-like dict
        if "label" in segment or "selector" in segment:
            # Use merge_and_rank_candidates for robust scoring
            candidates = [segment]
            ranked = merge_and_rank_candidates([], candidates, {}, [segment.get("label", "")], model)
            if ranked and ranked[0].get("combined_score", 0) >= ml_threshold:
                return ranked[0]["label"]

        # 5. Fallback: use extract_field for heuristics
        if "html" in segment:
            label = self.extract_field("panel", text=segment["html"])
            if label:
                return label

        # 6. Final fallback: unknown
        return "unknown"

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
        nodes = self.organized["dom_parts"].get("all_nodes", [])
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
    
    def correct_and_update_contest(self, contest_id, correction_data) -> None:
        self.data_service.update_contest_in_db({"id": contest_id, **correction_data})
        self.organized = None
        self.organize_and_enrich(self.last_raw_context)
        self.organizer.log_field_selection(
            field_type="contest",
            field_name="correction",
            extracted_value=correction_data,
            method="manual",
            score=1.0,
            result="manual_pass",
            context={"contest_id": contest_id},
            user_feedback=None
        )

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

    def _enrich_contests_with_nlp(self) -> None:
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

    def fuzzy_score(self, a, b) -> float:
        """
        Compute a fuzzy string similarity score between two strings.
        """
        model = self._semantic_model
        return model.similarity(str(a), str(b))

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

    def extract_entities(self, text, labels=None, first_only=False):
        """
        Unified NLP entity extraction using spaCy.
        - text: input string
        - labels: set or list of entity labels to filter (e.g., {"ORG", "PERSON"})
        - first_only: if True, return only the first match (else all)
        Returns: list of (entity, label) or a single (entity, label) if first_only
        """
        try:
            if not text or not isinstance(text, str):
                return [] if not first_only else None
            from ..utils.spacy_utils import extract_entities
            entities = extract_entities(text)
            if labels:
                labels_set = set(labels)
                filtered = [(ent, label) for ent, label in entities if label in labels_set]
            else:
                filtered = entities
            if first_only:
                return filtered[0] if filtered else None
            return filtered
        except Exception as e:
            logger.error(f"[ContextCoordinator.extract_entities] Error: {e}")
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
            from ..utils.spacy_utils import extract_locations
            locations = extract_locations(text)
            if labels:
                labels_set = set(labels)
                filtered = [(loc, label) for loc, label in locations if label in labels_set]
            else:
                filtered = locations
            if first_only:
                return filtered[0] if filtered else None
            return filtered
        except Exception as e:
            logger.error(f"[ContextCoordinator.extract_locations] Error: {e}")
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
            from ..utils.spacy_utils import extract_dates
            dates = extract_dates(text)
            if labels:
                labels_set = set(labels)
                filtered = [(date, label) for date, label in dates if label in labels_set]
            else:
                filtered = dates
            if first_only:
                return filtered[0] if filtered else None
            return filtered
        except Exception as e:
            logger.error(f"[ContextCoordinator.extract_dates] Error: {e}")
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
                    return (best[0], label, best[1] / 100.0, "spacy_ner_fuzzy", "pass")
            return None

        def fuzzy_party(text):
            known_parties = PARTY_KEYWORDS
            best = process.extractOne(text, known_parties)
            if best and best[1] > 80:
                return (best[0], None, best[1] / 100.0, "fuzzy", "pass")
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
        }

        # --- Extraction ---
        if field_type not in strategies:
            logger.error(f"[extract_field] Unknown field_type: {field_type}")
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
            if result and result[0]:
                # Ensure result is a 5-tuple: (value, label, score, method, result)
                if len(result) == 5:
                    # Overwrite method to ensure consistency
                    return (result[0], result[1], result[2], method, result[4])
                elif len(result) == 4:
                    # Insert None as label, force method
                    return (result[0], None, result[1], method, result[3])
                elif len(result) == 3:
                    return (result[0], None, result[1], method, result[2])
                elif len(result) == 2:
                    return (result[0], None, result[1], method, "pass")
                elif len(result) == 1:
                    return (result[0], None, 1.0, method, "pass")
                else:
                    return (result[0], None, 1.0, method, "pass")
        return (None, None, 0.0, "fail", "none")

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
                return float(self.score_entry(title, context or {}))
            if hasattr(self, "score_header_ml"):
                return float(self.score_header_ml(title, context or {}))
            # Use NLP entity type as a weak signal
            if hasattr(self, "extract_entities"):
                ents = self.extract_entities(title)
                if ents:
                    # Boost if header is a known entity type
                    for ent, label in ents:
                        if label in {"PERSON", "CANDIDATE", "ORG", "NORP", "GPE", "LOC"}:
                            return 0.8
            # Use known header keywords if available in context
            known_headers = set()
            if context and isinstance(context, dict):
                known_headers = set(context.get("known_headers", []))
            if known_headers and title.lower() in (h.lower() for h in known_headers):
                return 0.9
            # Fallback: score by length and capitalization
            if isinstance(title, str) and len(title) > 2 and title[0].isupper():
                return 0.6
            # Default fallback
            return 0.5
        except Exception as e:
            logger.error(f"[score_header] Error scoring header '{title}': {e}")
            return 0.5
    
    # --- DB/Service Delegation ---
    def get_full_contest(self, contest_id) -> Optional[Dict[str, Any]]:
        return self.data_service.get_full_contest(contest_id)

    def get_all_full_contests(self, filters=None, limit=100) -> List[Dict[str, Any]]:
        return self.data_service.get_all_full_contests(filters=filters, limit=limit)

    def list_tables(self) -> List[str]:
        return self.data_service.list_tables()

    def describe_table(self, table_name) -> Optional[Dict[str, Any]]:
        return self.data_service.describe_table(table_name)

    def get_table_metadata(self, table_name) -> Optional[Dict[str, Any]]:
        return self.data_service.get_table_metadata(table_name)

    def check_missing_tables(self) -> List[str]:
        return self.data_service.check_missing_tables()

    def get_table_structures(self, filters=None, limit=100, confirmed_only=False) -> List[Dict[str, Any]]:
        return self.data_service.fetch_table_structures(filters=filters, limit=limit, confirmed_only=confirmed_only)

    def get_table_structure(self, contest, context=None) -> Optional[Dict[str, Any]]:
        return self.data_service.get_table_structure(contest, context)

    def save_table_structure(self, contest, headers, context, ml_confidence=None, confirmed_by_user=False) -> bool:
        return self.data_service.save_table_structure(contest, headers, context, ml_confidence, confirmed_by_user)

    # --- Context/Contest Accessors ---
    def get_contests(self, filters=None) -> List[Dict[str, Any]]:
        if not self.organized:
            return []
        contests = self.organized.get("contests", [])
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

    def get_buttons(self, contest: Dict[str, Any], keyword: str = None, url: str = None) -> List[Dict[str, Any]]:
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
                    # Check for a successful result for this contest/keyword/url
                    if contest and entry.get("contest") == contest.get("title") and entry.get("result", "").startswith("pass"):
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
        if contest:
            results = buttons_dict.get(contest.get("title"), [])
            if results:
                return clean_for_json(results)
        if keyword:
            keyword = keyword.lower()
            btn_lists = buttons_dict.values() if isinstance(buttons_dict, dict) else buttons_dict
            for btn_list in btn_lists:
                for btn in btn_list:
                    if not isinstance(btn, dict):
                        continue
                    if keyword in btn.get("label", "").lower() or keyword in btn.get("selector", "").lower():
                        results.append(btn)
            if results:
                return clean_for_json(results)
        if url:
            for btn_list in buttons_dict.values():
                for btn in btn_list:
                    if not isinstance(btn, dict):
                        continue
                    if url in btn.get("selector", ""):
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
        if fuzzy_thresholds is None:
            fuzzy_thresholds = [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5]
        model = self._semantic_model
        context = context or {}
        context.update({
            "contest": contest,
            "year": context.get("year", ""),
            "election_types": context.get("election_types", ""),
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
                    logger.info(f"[green][LEARNING] Auto-applying learned button: {learned_btn.get('label')}[/green]")
                    try:
                        learned_btn["element_handle"].click()
                        page.wait_for_timeout(1500)
                        self.clicked_button_selectors.add(learned_btn.get("selector"))
                        return learned_btn, 0
                    except Exception:
                        logger.error("[red][ERROR] Failed to click learned button element.[/red]")
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
        button_features = page.locator(BUTTON_SELECTORS)

        def scan_btn(btn, i) -> None:
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
                self._log_button_memory(candidate, contest, "scanned")
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
                            logger.info(f"[bold green][Coordinator] Confirmed button: '{cand.get('label')}' (score={cand.get('combined_score', 0):.2f})[/bold green]")
                            self._log_button_memory(cand, contest, f"confirmed_pass_{cand.get('combined_score', 0):.2f}")
                            if not isinstance(cand, dict):
                                logger.error(f"[red][ERROR] Candidate is not a dict: {cand}[/red]")
                                continue
                            if learning_mode:
                                self._log_confirmed_button_for_learning(cand, contest, context)
                            self.clicked_button_selectors.add(cand.get("selector"))
                            try:
                                cand["element_handle"].click()
                                page.wait_for_timeout(1500)
                            except Exception:
                                pass
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

    # --- Table structure learning/lookup ---
    def get_table_structure(self, contest: dict = None, context: dict = None, learning_mode: bool = True) -> Optional[list[str]]:
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
        Launch the manual_correction_bot CLI for reviewing/editing corrections and feedback.
        """
        script_path = os.path.join(os.path.dirname(__file__), "..", "bots", "manual_correction_bot.py")
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
        panels = self.organized.get("panels", {})
        if not isinstance(panels, dict):
            return None
        return clean_for_json(panels.get(contest))

    def get_tables(self, contest: dict = None) -> list[dict]:
        if not self.organized:
            return []
        tables = self.organized.get("tables", {})
        if not isinstance(tables, dict):
            return []
        return clean_for_json(tables.get(contest.get("title") if contest else "", []))

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
                le_state.transform([c.get("state", "unknown")])[0],
                le_county.transform([c.get("county", "unknown")])[0],
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
