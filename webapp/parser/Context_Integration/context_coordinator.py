"""
context_coordinator.py

Production-grade Context Coordinator for Election Data Pipeline

- Orchestrates advanced context analysis, NLP, and ML integrity checks.
- Bridges between spaCy (NLP), context_organizer (DOM/ML), and downstream consumers (selectors, handlers, routers).
- Provides robust, dynamic, and cache-aware access to contests, buttons, panels, tables, candidates, districts, etc.
- Ensures all data is validated, deduplicated, and anomaly-checked before output.
"""

import os
import json
import glob
import datetime
import numpy as np
from ..utils.shared_logger import rprint
from sklearn.preprocessing import LabelEncoder

from ..utils.shared_logger import log_info, log_warning
from ..utils.shared_logic import load_context_library

from ..utils.spacy_utils import (
    extract_entities, extract_locations, extract_dates
)
from .Integrity_check import (
    detect_anomalies_with_ml, plot_clusters,
    election_integrity_checks, monitor_db_for_alerts, advanced_cross_field_validation
)
from .context_organizer import organize_context

# --- Config ---
SAMPLE_JSON_PATH = os.path.join(os.path.dirname(__file__), "sample.json")

# --- Core Coordinator Class ---

class ContextCoordinator:
    """
    Main interface for all context/NLP/ML operations.
    Use this class to access contests, buttons, panels, tables, candidates, districts, etc.
    """

    def __init__(self, use_library=True, enable_ml=True, alert_monitor=True):
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

    def get_buttons(self, contest_title=None):
        """
        Return all buttons, or those for a specific contest.
        """
        if not self.organized:
            return []
        if contest_title:
            return self.organized.get("buttons", {}).get(contest_title, []) \
                or self.organized.get("buttons", {}).get("__unmatched__", [])
        # Flatten all buttons
        all_buttons = []
        for btns in self.organized.get("buttons", {}).values():
            all_buttons.extend(btns)
        return all_buttons

    def get_best_button(self, contest_title, keywords=None, class_hint=None, prefer_clickable=True, prefer_visible=True):
        """
        Retrieve the best button for a contest, optionally filtering by keywords and class hints.
        """
        buttons = self.get_buttons(contest_title)
        if prefer_clickable:
            buttons = [b for b in buttons if b.get("is_clickable", False)] or buttons
        if prefer_visible:
            buttons = [b for b in buttons if b.get("is_visible", True)] or buttons
        if keywords:
            for btn in buttons:
                if any(kw.lower() in btn.get("label", "").lower() for kw in keywords):
                    if not class_hint or class_hint in btn.get("class", ""):
                        return btn
        if class_hint:
            for btn in buttons:
                if class_hint in btn.get("class", ""):
                    return btn
        return buttons[0] if buttons else None

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
