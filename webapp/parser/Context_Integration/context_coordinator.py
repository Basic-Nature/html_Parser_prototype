"""
context_coordinator.py

Production-grade Context Coordinator for Election Data Pipeline

- Orchestrates advanced context analysis, NLP, and ML integrity checks.
- Bridges between spaCy (NLP), context_organizer (DOM/ML), and downstream consumers (selectors, handlers, routers).
- Provides robust, dynamic, and cache-aware access to contests, buttons, panels, tables, candidates, districts, etc.
- Ensures all data is validated, deduplicated, and anomaly-checked before output.
"""
import re
import os
import numpy as np
from ..utils.shared_logger import rprint
from sklearn.preprocessing import LabelEncoder

from ..utils.shared_logger import log_info, log_warning



from ..utils.spacy_utils import (
    extract_entities, extract_locations, extract_dates
)
from .Integrity_check import (
    detect_anomalies_with_ml, plot_clusters,
    election_integrity_checks, monitor_db_for_alerts, advanced_cross_field_validation
)
from .context_organizer import organize_context
import inspect
# --- Config ---
SAMPLE_JSON_PATH = os.path.join(os.path.dirname(__file__), "sample.json")

def get_Known_state_to_county_map(self):
    return [s.lower().replace(" ", "_") for s in self.library.get("Known_state_to_county_map", [])]

def get_Known_county_to_district_map(self):
    return [c.lower().replace(" ", "_") for c in self.library.get("Known_county_to_district_map", [])]

def get_known_states(self):
    return [s.lower().replace(" ", "_") for s in self.library.get("known_states", [])]

def get_known_counties(self):
    return [c.lower().replace(" ", "_") for c in self.library.get("known_counties", [])]

# --- Core Coordinator Class ---

class ContextCoordinator:
    """
    Main interface for all context/NLP/ML operations.
    Use this class to access contests, buttons, panels, tables, candidates, districts, etc.
    """

    def __init__(self, use_library=True, enable_ml=True, alert_monitor=True):
        from ..utils.shared_logic import load_context_library
        self.library = load_context_library() if use_library else {}
        self.enable_ml = enable_ml
        self.alert_monitor = alert_monitor
        self.organized = None
        if alert_monitor:
            self.start_alert_monitoring()

    def organize_and_enrich(self, raw_context, contamination=None, n_estimators=100, random_state=42):
        """
        Organize raw context (from HTML/DOM or DB), deduplicate, cluster, and enrich with NLP.
        """
        self.organized = organize_context(
            raw_context,
            use_library=True,
            enable_ml=self.enable_ml,
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state
        )
        self._enrich_contests_with_nlp()
        return self.organized

    def _enrich_contests_with_nlp(self):
        """
        Add NLP-derived fields (entities, locations, dates) to each contest.
        """
        if not self.organized or "contests" not in self.organized:
            return
        for c in self.organized["contests"]:
            title = c.get("title", "")
            c["entities"] = extract_entities(title)
            c["locations"] = extract_locations(title)
            c["dates"] = extract_dates(title)

    # --- Data Accessors ---

    def get_contests(self, filters=None):
        """
        Return contests, optionally filtered by state, county, year, type, etc.
        """
        contests = self.organized["contests"] if self.organized else []
        if not filters:
            return contests
        def match(c):
            for k, v in filters.items():
                if str(c.get(k, "")).lower() != str(v).lower():
                    return False
            return True
        return [c for c in contests if match(c)]

    def get_buttons(self, contest_title=None, keyword=None, url=None):
        """
        Return all buttons, or those for a specific contest, or matching a keyword/URL.
        """
        if not self.organized:
            return []
        buttons_dict = self.organized.get("buttons", {})
        results = []

        # 1. By contest title (exact match)
        if contest_title and isinstance(contest_title, str):
            results = buttons_dict.get(contest_title, [])
            if results:
                return results

        # 2. By keyword in label or selector
        if keyword:
            keyword = keyword.lower()
            for btn_list in buttons_dict.values():
                for btn in btn_list:
                    if keyword in btn.get("label", "").lower() or keyword in btn.get("selector", "").lower():
                        results.append(btn)
            if results:
                return results

        # 3. By URL (if you want to associate buttons with URLs)
        if url:
            for btn_list in buttons_dict.values():
                for btn in btn_list:
                    if url in btn.get("selector", ""):
                        results.append(btn)
            if results:
                return results

        # 4. Fallback: return all buttons
        all_buttons = []
        for btns in buttons_dict.values():
            all_buttons.extend(btns)
        return all_buttons

    def get_best_button(
        self,
        contest_title=None,
        keywords=None,
        class_hint=None,
        url=None,
        prefer_clickable=True,
        prefer_visible=True,
        log_memory=True,
        page=None,  # Pass Playwright page if you want to scan live DOM
        max_attempts=3,
        fuzzy_threshold=0.7
    ):
        """
        Return the best button by contest, keyword, selector, or URL.
        If not found, scan page for candidates using regex/fuzzy match and retry.
        Logs all attempts for ML/rule improvement.
        """
        # 1. Try by contest title first
        buttons = self.get_buttons(contest_title)
        # 2. Try by keywords
        if not buttons and keywords:
            for kw in keywords:
                buttons = self.get_buttons(keyword=kw)
                if buttons:
                    break
        # 3. Try by URL
        if not buttons and url:
            buttons = self.get_buttons(url=url)

        # 4. Filter by clickable/visible
        if prefer_clickable:
            buttons = [b for b in buttons if b.get("is_clickable", False)] or buttons
        if prefer_visible:
            buttons = [b for b in buttons if b.get("is_visible", True)] or buttons

        # 5. Try to match by keywords/class_hint
        if keywords:
            for btn in buttons:
                if any(kw.lower() in btn.get("label", "").lower() for kw in keywords):
                    if not class_hint or class_hint in btn.get("class", ""):
                        if log_memory:
                            self._log_button_memory(btn, contest_title, "pass")
                        return btn
        if class_hint:
            for btn in buttons:
                if class_hint in btn.get("class", ""):
                    if log_memory:
                        self._log_button_memory(btn, contest_title, "pass")
                    return btn

        # 6. If still not found, scan live DOM for candidates using regex/fuzzy match
        if page is not None:
            # Scan all visible/clickable buttons
            BUTTON_SELECTORS = "button, a, [role='button'], input[type='button'], input[type='submit']"
            button_features = page.locator(BUTTON_SELECTORS)
            candidates = []
            for i in range(button_features.count()):
                btn = button_features.nth(i)
                label = btn.inner_text() or ""
                class_name = btn.get_attribute("class") or ""
                is_visible = btn.is_visible()
                is_enabled = btn.is_enabled()
                selector = None
                try:
                    selector = btn.evaluate("el => el.outerHTML")
                except Exception:
                    pass
                candidates.append({
                    "label": label,
                    "class": class_name,
                    "selector": selector,
                    "is_visible": is_visible,
                    "is_clickable": is_enabled
                })
                if log_memory:
                    self._log_button_memory(
                        {"label": label, "class": class_name, "selector": selector, "is_visible": is_visible, "is_clickable": is_enabled},
                        contest_title,
                        "scanned"
                    )

            # Score candidates by fuzzy match to keywords
            best_score = 0
            best_candidate = None
            for cand in candidates:
                for kw in (keywords or []):
                    # Fuzzy match on label
                    score = difflib.SequenceMatcher(None, kw.lower(), (cand["label"] or "").lower()).ratio()
                    if score > best_score:
                        best_score = score
                        best_candidate = cand
                    # Regex partial match
                    if re.search(re.escape(kw), cand["label"], re.IGNORECASE):
                        best_score = 1.0
                        best_candidate = cand
                        break
            if best_candidate and best_score >= fuzzy_threshold:
                if log_memory:
                    self._log_button_memory(best_candidate, contest_title, f"fuzzy_pass_{best_score:.2f}")
                return best_candidate

        # 7. Fallback: return first button and log as "fail"
        if buttons and isinstance(buttons[0], dict):
            if log_memory:
                self._log_button_memory(buttons[0], contest_title, "fail")
            return buttons[0]
        return None
    
    def _log_button_memory(self, button, contest_title, result):
        """
        Log button selection attempts for future ML or rule improvements.
        """
        # You could append to a file, DB, or in-memory structure
        log_entry = {
            "contest_title": contest_title,
            "button_label": button.get("label"),
            "selector": button.get("selector"),
            "result": result
        }
        # Example: append to a file for later analysis
        with open("button_selection_log.jsonl", "a", encoding="utf-8") as f:
            import json
            f.write(json.dumps(log_entry) + "\n")
    def get_panel(self, contest_title):
        """
        Retrieve the panel for a given contest title.
        """
        return self.organized.get("panels", {}).get(contest_title) if self.organized else None

    def get_tables(self, contest_title):
        """
        Retrieve tables for a given contest title.
        """
        return self.organized.get("tables", {}).get(contest_title, []) if self.organized else []

    def get_candidates(self, contest_title=None):
        """
        Extract candidate names from contest entities or table headers.
        """
        candidates = set()
        contests = self.get_contests() if contest_title is None else [c for c in self.get_contests() if c.get("title") == contest_title]
        for c in contests:
            for ent, label in c.get("entities", []):
                if label in {"PERSON", "CANDIDATE"}:
                    candidates.add(ent)
            # Optionally: parse table headers for candidate names
            for tbl in self.get_tables(c.get("title", "")):
                headers = tbl.get("headers", [])
                for h in headers:
                    if "candidate" in h.lower():
                        candidates.add(h)
        return list(candidates)

    def get_districts(self, state=None, county=None):
        """
        Return known districts for a state/county from the library.
        """
        if not self.library:
            return []
        if county:
            return self.library.get("Known_county_to_district_map", {}).get(county, [])
        if state:
            return self.library.get("Known_state_to_county_map", {}).get(state, [])
        return self.library.get("known_districts", [])

    def get_states(self):
        """
        Return all known states from the library.
        """
        return self.library.get("known_states", [])

    def get_election_types(self):
        """
        Return all known election types from the library.
        """
        return self.library.get("election", [])

    def get_years(self):
        """
        Return all years found in contests.
        """
        contests = self.get_contests()
        return sorted({c.get("year") for c in contests if c.get("year")})

    # --- Integrity & Anomaly Checks ---

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
        states = [c.get("state", "unknown") for c in contests]
        counties = [c.get("county", "unknown") for c in contests]
        le_state.fit(states)
        le_county.fit(counties)
        for c in contests:
            features.append([
                le_state.transform([c.get("state", "unknown")])[0],
                le_county.transform([c.get("county", "unknown")])[0],
                int(c.get("year", 0)) if str(c.get("year", "0")).isdigit() else 0,
                len(str(c.get("title", ""))),
            ])
        X = np.array(features)
        plot_clusters(X, clusters, anomalies=anomalies)
        # Cross-check with expected year
        date_anomalies = []
        if expected_year:
            for c in contests:
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

    def start_alert_monitoring(self, db_path=None, poll_interval=10):
        """
        Start real-time alert monitoring in a background thread.
        """
        from ..utils.db_utils import DB_PATH
        monitor_db_for_alerts(db_path=db_path, poll_interval=poll_interval)

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
        return {
            "contests": self.get_contests(),
            "buttons": self.get_buttons(),
            "noisy_patterns": self.library.get("default_noisy_label_patterns", [])
        }

    def get_for_table_builder(self):
        """
        Return precinct headers and table tags for table_builder.
        """
        return {
            "precinct_headers": self.library.get("precinct_header_tags", []),
            "table_tags": self.library.get("table_tags", [])
        }

    def get_for_html_handler(self):
        """
        Return panel tags, contest panel tags, and selectors for html_handler.
        """
        return {
            "panel_tags": self.library.get("panel_tags", []),
            "contest_panel_tags": self.library.get("contest_panel_tags", []),
            "all_selectors": self.library.get("selectors", {}).get("all_selectors", [])
        }

    def get_for_state_router(self):
        """
        Return state_module_map for state_router.
        """
        return self.library.get("state_module_map", {})

    # --- Correction/Update ---

    def correct_and_update_contest(self, contest_id, correction_data):
        """
        Update a contest in the DB and refresh the organized context.
        """
        from ..utils.db_utils import update_contest_in_db, DB_PATH
        update_contest_in_db({"id": contest_id, **correction_data})
        # Optionally re-fetch and re-organize
        self.organized = None

def call_handler_with_coordinator(handler, *args, coordinator=None, **kwargs):
    sig = inspect.signature(handler.parse)
    if 'coordinator' in sig.parameters:
        return handler.parse(*args, coordinator, **kwargs)
    else:
        return handler.parse(*args, **kwargs)

    # --- Sample Usage ---

def sample_usage():
    """
    Example: Run the coordinator on a sample context and print a summary.
    """
    rprint("[bold green]=== Sample Usage: ContextCoordinator ===[/bold green]")
    # 1. Load a sample context (simulate HTML/DOM extraction)
    sample_context = {
        "contests": [
            {"title": "2024 Presidential Election - New York", "year": 2024, "type": "Presidential", "state": "New York"},
            {"title": "2022 Senate Race - California", "year": 2022, "type": "Senate", "state": "California"},
            {"title": "2024 Mayoral Election - Houston, TX", "year": 2024, "type": "Mayoral", "state": "Texas"},
            {"title": "2023 School Board - Miami", "year": 2023, "type": "School Board", "state": "Florida"},
        ],
        "buttons": [
            {"label": "Show Results", "is_clickable": True, "is_visible": True},
            {"label": "Vote Method", "is_clickable": True, "is_visible": True},
            {"label": "Summary", "is_clickable": True, "is_visible": True}
        ]
    }
    coordinator = ContextCoordinator()
    coordinator.organize_and_enrich(sample_context)
    coordinator.report_summary()
    # Example: Get best button for a contest
    btn = coordinator.get_best_button("2024 Presidential Election - New York", keywords=["Show Results"])
    rprint(f"[bold green]Best button for NY Presidential:[/bold green] {btn}")
    # Example: Get candidates for a contest
    candidates = coordinator.get_candidates("2024 Presidential Election - New York")
    rprint(f"[bold green]Candidates for NY Presidential:[/bold green] {candidates}")
    # Example: Get districts for New York
    districts = coordinator.get_districts(state="new_york")
    rprint(f"[bold green]Districts for New York:[/bold green] {districts}")
    # Example: Validate and check integrity
    issues = coordinator.validate_and_check_integrity(expected_year=2024)
    rprint(f"[bold green]Integrity/Anomaly Issues:[/bold green] {issues}")

    # Example: Send data to output_utils (pseudo-code)
    # from ..utils.output_utils import finalize_election_output
    # headers, data = ... # Extracted from tables
    # contest_title = "2024 Presidential Election - New York"
    # metadata = coordinator.organized.get("metadata", {})
    # finalize_election_output(headers, data, contest_title, metadata)

    rprint("[bold green]=== End Sample Usage ===[/bold green]")

import difflib

def dynamic_state_county_detection(context, html, context_library, debug=False):
    """
    Dynamically detect county (first) and state (second) using all available clues.
    If only state is found, checks for available county handler modules.
    Returns (county, state, handler_path, detection_log)
    """
    from ..utils.spacy_utils import extract_entities

    detection_log = []
    county = context.get("county")
    state = context.get("state")
    url = context.get("url", "")
    contests = context.get("contests", [])
    known_states = context_library.get("known_states", [])
    state_to_county = context_library.get("Known_state_to_county_map", {})
    all_counties = set()
    for counties in state_to_county.values():
        all_counties.update(counties)
    all_counties = list(all_counties)

    # --- 1. Try context fields directly ---
    if county:
        detection_log.append(f"County found in context: {county}")
    if state:
        detection_log.append(f"State found in context: {state}")

    # --- 2. Try to extract county from URL ---
    if not county and url:
        for c in all_counties:
            c_norm = c.lower().replace(" ", "_")
            if c_norm in url.lower():
                county = c
                detection_log.append(f"County '{county}' detected from URL.")
                break

    # --- 3. Try to extract county from contest titles ---
    if not county and contests:
        for contest in contests:
            title = contest.get("title", "")
            for c in all_counties:
                if c.lower() in title.lower():
                    county = c
                    detection_log.append(f"County '{county}' detected from contest title: '{title}'")
                    break
            if county:
                break

    # --- 4. Try to extract county from HTML using NLP entities ---
    if not county and html:
        entities = extract_entities(html)
        gpe_entities = [ent for ent, label in entities if label in ("GPE", "LOC")]
        for ent in gpe_entities:
            matches = difflib.get_close_matches(ent.lower(), [c.lower() for c in all_counties], n=1, cutoff=0.7)
            if matches:
                county = all_counties[[c.lower() for c in all_counties].index(matches[0])]
                detection_log.append(f"County '{county}' detected from HTML NLP entity: '{ent}'")
                break

    # --- 5. Fuzzy match county if still not found ---
    if not county and url:
        url_tokens = re.split(r"[\W_]+", url.lower())
        matches = difflib.get_close_matches(" ".join(url_tokens), [c.lower() for c in all_counties], n=1, cutoff=0.6)
        if matches:
            county = all_counties[[c.lower() for c in all_counties].index(matches[0])]
            detection_log.append(f"County '{county}' fuzzy-matched from URL tokens.")

    # --- 6. Now try to detect state, using county if found ---
    if not state and county:
        for s, counties in state_to_county.items():
            if county in counties:
                state = s
                detection_log.append(f"State '{state}' inferred from county '{county}'.")
                break

    # --- 7. Try to extract state from URL ---
    if not state and url:
        for s in known_states:
            s_norm = s.lower().replace(" ", "_")
            if s_norm in url.lower():
                state = s
                detection_log.append(f"State '{state}' detected from URL.")
                break

    # --- 8. Try to extract state from contest titles ---
    if not state and contests:
        for contest in contests:
            title = contest.get("title", "")
            for s in known_states:
                if s.lower() in title.lower():
                    state = s
                    detection_log.append(f"State '{state}' detected from contest title: '{title}'")
                    break
            if state:
                break

    # --- 9. Try to extract state from HTML using NLP entities ---
    if not state and html:
        entities = extract_entities(html)
        gpe_entities = [ent for ent, label in entities if label in ("GPE", "LOC")]
        for ent in gpe_entities:
            matches = difflib.get_close_matches(ent.lower(), [s.lower() for s in known_states], n=1, cutoff=0.7)
            if matches:
                state = known_states[[s.lower() for s in known_states].index(matches[0])]
                detection_log.append(f"State '{state}' detected from HTML NLP entity: '{ent}'")
                break

    # --- 10. Fuzzy match state if still not found ---
    if not state and url:
        url_tokens = re.split(r"[\W_]+", url.lower())
        matches = difflib.get_close_matches(" ".join(url_tokens), [s.lower() for s in known_states], n=1, cutoff=0.6)
        if matches:
            state = known_states[[s.lower() for s in known_states].index(matches[0])]
            detection_log.append(f"State '{state}' fuzzy-matched from URL tokens.")

    # --- 11. If state found but no county, check for available county handlers ---
    handler_path = None
    if state and not county:
        # Check for county handler modules in webapp/parser/handlers/states/{state}/county/
        state_key = state.lower().replace(" ", "_")
        county_dir = os.path.join(
            os.path.dirname(__file__), "..", "handlers", "states", state_key, "county"
        )
        county_dir = os.path.abspath(county_dir)
        available_counties = []
        if os.path.isdir(county_dir):
            for fname in os.listdir(county_dir):
                if fname.endswith(".py") and not fname.startswith("__"):
                    county_name = fname[:-3].replace("_", " ").title()
                    available_counties.append(county_name)
            detection_log.append(f"Available county handlers for state '{state}': {available_counties}")
            # Try to match county from URL or HTML context to available counties
            url_and_html = (url + " " + html).lower()
            for c in available_counties:
                c_norm = c.lower().replace(" ", "_")
                if c_norm in url_and_html:
                    county = c
                    detection_log.append(f"County '{county}' matched to available handler from URL/HTML context.")
                    break
            if not county and available_counties:
                detection_log.append("No matching county handler found in URL/HTML; will use state handler.")
        else:
            detection_log.append(f"No county handler directory found for state '{state}'.")

        # Set handler path
        if county:
            handler_path = f"webapp.parser.handlers.states.{state_key}.county.{county.lower().replace(' ', '_')}"
        else:
            handler_path = f"webapp.parser.handlers.states.{state_key}"

    # --- 12. If both found, set handler path ---
    if state and county:
        state_key = state.lower().replace(" ", "_")
        county_key = county.lower().replace(" ", "_")
        handler_path = f"webapp.parser.handlers.states.{state_key}.county.{county_key}"

    # --- 13. If only state found, fallback to state handler ---
    if state and not county and not handler_path:
        state_key = state.lower().replace(" ", "_")
        handler_path = f"webapp.parser.handlers.states.{state_key}"

    # --- 14. Final fallback ---
    if not county:
        detection_log.append("County could not be detected.")
    if not state:
        detection_log.append("State could not be detected.")

    if debug:
        for log in detection_log:
            print("[dynamic_state_county_detection]", log)

    return county, state, handler_path, detection_log

# --- Alert Monitoring (run in production) ---
def start_alert_monitoring():
    from ..utils.db_utils import DB_PATH
    monitor_db_for_alerts(db_path=DB_PATH, poll_interval=10)

# --- CLI Entrypoint ---
if __name__ == "__main__":
    sample_usage()
    # To enable alert monitoring in production, uncomment:
    # start_alert_monitoring()

    # To add more sample cases, copy the sample_context and modify as needed.
    # For production, instantiate ContextCoordinator and call organize_and_enrich with real context.
