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
import json
import datetime
import types
from fuzzywuzzy import fuzz, process
from ..utils.shared_logger import rprint
from ..utils.shared_logic import (
    scan_buttons_with_progress, keyphrase_match,
    normalize_state_name, normalize_county_name
)
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import LabelEncoder
import subprocess
from rich.console import Console

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
from .context_organizer import organize_context
import inspect
# --- Config ---
SAMPLE_JSON_PATH = os.path.join(os.path.dirname(__file__), "sample.json")

def safe_for_json(obj):
    """Recursively remove non-serializable items from dicts/lists."""
    if isinstance(obj, dict):
        return {k: safe_for_json(v) for k, v in obj.items() if not isinstance(v, types.FunctionType)}
    elif isinstance(obj, list):
        return [safe_for_json(i) for i in obj if not isinstance(i, types.FunctionType)]
    elif isinstance(obj, types.FunctionType):
        return None
    else:
        return obj

def get_Known_state_to_county_map(self):
    return [s.lower().replace(" ", "_") for s in self.library.get("Known_state_to_county_map", [])]

def get_Known_county_to_district_map(self):
    return [c.lower().replace(" ", "_") for c in self.library.get("Known_county_to_district_map", [])]

def get_known_states(self):
    return [s.lower().replace(" ", "_") for s in self.library.get("known_states", [])]

def get_known_counties(self):
    return [c.lower().replace(" ", "_") for c in self.library.get("known_counties", [])]

def _sanitize_log_filename(name):
    # Only allow alphanumeric, underscore, and dash
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', name) 

def get_semantic_score(model, text1, text2):
    """
    Compute semantic similarity between two strings using SentenceTransformer.
    """
    if not text1 or not text2:
        return 0.0
    emb1 = model.encode(text1, convert_to_tensor=False)
    emb2 = model.encode(text2, convert_to_tensor=False)
    return float(util.pytorch_cos_sim(emb1, emb2)[0][0])

def merge_and_rank_candidates(memory_candidates, dom_candidates, context, keywords, model,
                             fuzzy_weight=0.3, semantic_weight=0.3, context_weight=0.2, hierarchy_weight=0.2):
    """
    Merge memory and DOM candidates, deduplicate, and rank by combined fuzzy and semantic score.
    """
    seen = set()
    all_candidates = []
    for cand in memory_candidates + dom_candidates:
        key = (cand.get("label", ""), cand.get("selector", ""))
        if key not in seen:
            seen.add(key)
            all_candidates.append(cand)

    context_str = " ".join([
        str(context.get("contest_title", "")),
        str(context.get("year", "")),
        str(context.get("election_type", "")),
        str(context.get("county", "")),
        str(context.get("state", "")),
    ]).strip()

    expected_class = context.get("expected_class", "")
    expected_tag = context.get("expected_tag", "")
    contest_title = context.get("contest_title", "")

    for cand in all_candidates:
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

    all_candidates.sort(
        key=lambda c: (
            c["combined_score"],
            c.get("is_visible", False),
            c.get("is_clickable", False)
        ),
        reverse=True
    )
    return all_candidates

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
        self.last_raw_context = None 
        self.clicked_button_selectors = set()
        if alert_monitor:
            self.start_alert_monitoring()

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

    def correct_and_update_contest(self, contest_id, correction_data):
        """
        Update a contest in the DB and context library, then re-organize context.
        """
        from ..utils.db_utils import update_contest_in_db
        from ..utils.shared_logic import save_context_library  # If you have a save function

        # 1. Update DB
        update_contest_in_db({"id": contest_id, **correction_data})

        # 2. Update context library if needed
        for key, value in correction_data.items():
            if key == "county" and value not in self.library.get("known_counties", []):
                self.library.setdefault("known_counties", []).append(value)
            if key == "state" and value not in self.library.get("known_states", []):
                self.library.setdefault("known_states", []).append(value)
            # Add similar logic for other fields as needed

        # 3. Save updated context library (if you persist it)
        save_context_library(self.library)  # Uncomment if you have this function

        # 4. Re-organize context using the last raw context
        self.organized = None
        if self.last_raw_context is not None:
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

    def fuzzy_score(self, a, b):
        """
        Compute a fuzzy string similarity score between two strings.
        """
        from fuzzywuzzy import fuzz
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
            log_path = os.path.join("log", f"{safe_field_type}_selection_log.jsonl")
        else:
            # Only use the filename part, sanitize it, and force it into log/
            base = os.path.basename(log_path)
            safe_base = _sanitize_log_filename(base)
            log_path = os.path.join("log", safe_base)
        log_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
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
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(safe_for_json(log_entry), ensure_ascii=False) + "\n")

    def extract_entities(self, text):
        """
        Extract entities from text using spaCy or your preferred NER model.
        Returns a list of (entity_text, entity_label) tuples.
        """
        # Example using spaCy (make sure spacy and a model are installed)
        import spacy
        nlp = getattr(self, "_spacy_nlp", None)
        if nlp is None:
            nlp = spacy.load("en_core_web_sm")
            self._spacy_nlp = nlp
        doc = nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]
    
    def extract_district(self, contest):
        """
        Extract the district from a contest using regex, spaCy NER, and fuzzy matching.
        Log the extraction attempt and result.
        """
        title = contest.get("title", "")
        extracted_value = None
        score = 0.0
        method = "regex"
        result = "fail"
        user_feedback = None

        # 1. Regex for "District N"
        match = re.search(r"District\s+(\d+)", title, re.IGNORECASE)
        if match:
            extracted_value = match.group(1)
            score = 0.95
            method = "regex"
            result = "pass"
        else:
            # 2. spaCy NER for ORDINAL or CARDINAL
            entities = extract_entities(title)
            for ent, label in entities:
                if label in {"ORDINAL", "CARDINAL"} and ent.isdigit():
                    extracted_value = ent
                    score = 0.85
                    method = "spacy_ner"
                    result = "pass"
                    break
            # 3. Fuzzy match for "district" context
            if not extracted_value:
                tokens = title.split()
                for i, token in enumerate(tokens):
                    if fuzz.ratio(token.lower(), "district") > 80 and i + 1 < len(tokens):
                        next_token = tokens[i + 1]
                        if next_token.isdigit():
                            extracted_value = next_token
                            score = 0.75
                            method = "fuzzy"
                            result = "pass"
                            break
        self.log_field_selection(
            field_type="district",
            field_name="district",
            extracted_value=extracted_value,
            method=method,
            score=score,
            result=result,
            context=contest,
            user_feedback=user_feedback,
            log_path="field_selection_log.jsonl"
        )
        return extracted_value

    def extract_contest_title(self, contest):
        """
        Extract the contest title using ML/NLP/manual methods.
        Log the extraction attempt and result.
        """
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
        # Example: Use entities if available
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
        Extract party from contest using regex, spaCy NER, and fuzzy matching.
        Log the extraction attempt and result.
        """
        title = contest.get("title", "")
        extracted_value = None
        score = 0.0
        method = "regex"
        result = "fail"
        user_feedback = None

        # 1. Regex for common parties
        match = re.search(r"(Democrat|Republican|Green|Libertarian|Independent)", title, re.IGNORECASE)
        if match:
            extracted_value = match.group(1)
            score = 0.9
            method = "regex"
            result = "pass"
        else:
            # 2. spaCy NER for ORG or NORP
            entities = extract_entities(title)
            known_parties = ["Democrat", "Republican", "Green", "Libertarian", "Independent"]
            for ent, label in entities:
                if label in {"ORG", "NORP"}:
                    best_match, best_score = process.extractOne(ent, known_parties)
                    if best_score > 80:
                        extracted_value = best_match
                        score = best_score / 100.0
                        method = "spacy_ner_fuzzy"
                        result = "pass"
                        break
            # 3. Fuzzy match directly in title
            if not extracted_value:
                best_match, best_score = process.extractOne(title, known_parties)
                if best_score > 80:
                    extracted_value = best_match
                    score = best_score / 100.0
                    method = "fuzzy"
                    result = "pass"

        self.log_field_selection(
            field_type="party",
            field_name="party",
            extracted_value=extracted_value,
            method=method,
            score=score,
            result=result,
            context=contest,
            user_feedback=user_feedback,
            log_path="field_selection_log.jsonl"
        )
        return extracted_value

    def extract_panel(self, contest_title):
        """
        Extract the panel for a given contest title using regex, spaCy NER, and direct lookup.
        Log the extraction attempt and result.
        """
        panel = None
        method = "direct_lookup"
        score = 0.0
        result = "fail"
        user_feedback = None

        # 1. Regex for common panel words
        if contest_title:
            match = re.search(r"(panel|section|container|box|area)", contest_title, re.IGNORECASE)
            if match:
                panel = match.group(1)
                method = "regex"
                score = 0.9
                result = "pass"

        # 2. spaCy NER for ORG/NORP
        if not panel and contest_title:
            entities = extract_entities(contest_title)
            for ent, label in entities:
                if label in {"ORG", "NORP"}:
                    panel = ent
                    method = "spacy_ner"
                    score = 0.85
                    result = "pass"
                    break

        # 3. Fallback to direct lookup
        if not panel:
            panel = self.get_panel(contest_title)
            if panel:
                method = "direct_lookup"
                score = 1.0
                result = "pass"

        self.log_field_selection(
            field_type="panel",
            field_name="panel",
            extracted_value=panel,
            method=method,
            score=score,
            result=result,
            context={"contest_title": contest_title},
            user_feedback=user_feedback,
            log_path="field_selection_log.jsonl"
        )
        return panel

    def extract_tables(self, contest_title):
        """
        Extract tables for a given contest title using regex, spaCy NER, and direct lookup.
        Log the extraction attempt and result.
        """
        tables = []
        method = "direct_lookup"
        score = 0.0
        result = "fail"
        user_feedback = None

        # 1. Regex for table-like words
        if contest_title:
            match = re.search(r"(table|results|summary|sheet|spreadsheet|grid)", contest_title, re.IGNORECASE)
            if match:
                tables.append(match.group(1))
                method = "regex"
                score = 0.9
                result = "pass"

        # 2. spaCy NER for ORG/NORP
        if not tables and contest_title:
            entities = extract_entities(contest_title)
            for ent, label in entities:
                if label in {"ORG", "NORP"}:
                    tables.append(ent)
                    method = "spacy_ner"
                    score = 0.85
                    result = "pass"
                    break

        # 3. Fallback to direct lookup
        if not tables:
            tables = self.get_tables(contest_title)
            if tables:
                method = "direct_lookup"
                score = 1.0
                result = "pass"

        self.log_field_selection(
            field_type="tables",
            field_name="tables",
            extracted_value=tables,
            method=method,
            score=score,
            result=result,
            context={"contest_title": contest_title},
            user_feedback=user_feedback,
            log_path="field_selection_log.jsonl"
        )
        return tables

    def extract_districts(self, state=None, county=None):
        """
        Extract known districts for a state/county using regex, spaCy NER, and direct lookup.
        Log the extraction attempt and result.
        """
        districts = []
        method = "direct_lookup"
        score = 0.0
        result = "fail"
        user_feedback = None

        # 1. Regex for district-like words
        if state:
            match = re.search(r"(district|ward|precinct|zone|division)", state, re.IGNORECASE)
            if match:
                districts.append(match.group(1))
                method = "regex"
                score = 0.9
                result = "pass"
        if county and not districts:
            match = re.search(r"(district|ward|precinct|zone|division)", county, re.IGNORECASE)
            if match:
                districts.append(match.group(1))
                method = "regex"
                score = 0.9
                result = "pass"

        # 2. spaCy NER for ORG/NORP
        if not districts and state:
            entities = extract_entities(state)
            for ent, label in entities:
                if label in {"ORG", "NORP"}:
                    districts.append(ent)
                    method = "spacy_ner"
                    score = 0.85
                    result = "pass"
                    break
        if not districts and county:
            entities = extract_entities(county)
            for ent, label in entities:
                if label in {"ORG", "NORP"}:
                    districts.append(ent)
                    method = "spacy_ner"
                    score = 0.85
                    result = "pass"
                    break

        # 3. Fallback to direct lookup
        if not districts:
            districts = self.get_districts(state=state, county=county)
            if districts:
                method = "direct_lookup"
                score = 1.0
                result = "pass"

        self.log_field_selection(
            field_type="districts",
            field_name="districts",
            extracted_value=districts,
            method=method,
            score=score,
            result=result,
            context={"state": state, "county": county},
            user_feedback=user_feedback,
            log_path="field_selection_log.jsonl"
        )
        return districts

    def extract_states(self):
        """
        Extract all known states using regex, spaCy NER, and direct lookup.
        Log the extraction attempt and result.
        """
        states = []
        method = "direct_lookup"
        score = 0.0
        result = "fail"
        user_feedback = None

        # 1. Regex for state-like words
        known_states = self.library.get("known_states", [])
        for s in known_states:
            match = re.search(r"(state|province|territory|region)", s, re.IGNORECASE)
            if match:
                states.append(s)
        if states:
            method = "regex"
            score = 0.9
            result = "pass"

        # 2. spaCy NER for ORG/NORP
        if not states:
            for s in known_states:
                entities = extract_entities(s)
                for ent, label in entities:
                    if label in {"ORG", "NORP"}:
                        states.append(ent)
                        method = "spacy_ner"
                        score = 0.85
                        result = "pass"
                        break
                if states:
                    break

        # 3. Fallback to direct lookup
        if not states:
            states = self.get_states()
            if states:
                method = "direct_lookup"
                score = 1.0
                result = "pass"

        self.log_field_selection(
            field_type="states",
            field_name="states",
            extracted_value=states,
            method=method,
            score=score,
            result=result,
            context={},
            user_feedback=user_feedback,
            log_path="field_selection_log.jsonl"
        )
        return states

    def extract_election_types(self):
        """
        Extract all known election types using regex, spaCy NER, and direct lookup.
        Log the extraction attempt and result.
        """
        election_types = []
        method = "direct_lookup"
        score = 0.0
        result = "fail"
        user_feedback = None

        # 1. Regex for election type words
        known_types = self.library.get("election", [])
        for t in known_types:
            match = re.search(r"(primary|general|special|runoff|municipal|presidential|senate|mayoral|school board)", t, re.IGNORECASE)
            if match:
                election_types.append(match.group(1))
        if election_types:
            method = "regex"
            score = 0.9
            result = "pass"

        # 2. spaCy NER for ORG/NORP
        if not election_types:
            for t in known_types:
                entities = extract_entities(t)
                for ent, label in entities:
                    if label in {"ORG", "NORP"}:
                        election_types.append(ent)
                        method = "spacy_ner"
                        score = 0.85
                        result = "pass"
                        break
                if election_types:
                    break

        # 3. Fallback to direct lookup
        if not election_types:
            election_types = self.get_election_types()
            if election_types:
                method = "direct_lookup"
                score = 1.0
                result = "pass"

        self.log_field_selection(
            field_type="election_types",
            field_name="election_types",
            extracted_value=election_types,
            method=method,
            score=score,
            result=result,
            context={},
            user_feedback=user_feedback,
            log_path="field_selection_log.jsonl"
        )
        return election_types

    def extract_years(self):
        """
        Extract all years found in contests using regex, spaCy NER, and direct lookup.
        Log the extraction attempt and result.
        """
        years = []
        method = "direct_lookup"
        score = 0.0
        result = "fail"
        user_feedback = None

        # 1. Regex for 4-digit years
        contests = self.get_contests()
        for c in contests:
            match = re.search(r"\b(19|20)\d{2}\b", str(c.get("title", "")))
            if match:
                years.append(match.group(0))
        if years:
            method = "regex"
            score = 0.9
            result = "pass"

        # 2. spaCy NER for DATE
        if not years:
            for c in contests:
                entities = extract_entities(c.get("title", ""))
                for ent, label in entities:
                    if label == "DATE" and re.match(r"\b(19|20)\d{2}\b", ent):
                        years.append(ent)
                        method = "spacy_ner"
                        score = 0.85
                        result = "pass"
                        break
                if years:
                    break

        # 3. Fallback to direct lookup
        if not years:
            years = self.get_years()
            if years:
                method = "direct_lookup"
                score = 1.0
                result = "pass"

        self.log_field_selection(
            field_type="years",
            field_name="years",
            extracted_value=years,
            method=method,
            score=score,
            result=result,
            context={},
            user_feedback=user_feedback,
            log_path="field_selection_log.jsonl"
        )
        return years

    def extract_buttons(self, contest_title=None, keyword=None, url=None):
        """
        Extract button labels using regex, spaCy NER (ORG/NORP), and direct lookup.
        Log the extraction attempt and result.
        """
        candidates = []
        method = "direct_lookup"
        score = 0.0
        result = "fail"
        user_feedback = None

        # 1. Regex for button-like words
        sources = [contest_title or "", keyword or "", url or ""]
        for src in sources:
            match = re.search(r"(Show Results|Vote|Submit|Summary|Next|Continue|Back|Download|Print|Details|Results|Ballot|Cast Vote)", src, re.IGNORECASE)
            if match:
                candidates.append(match.group(1))
        if candidates:
            method = "regex"
            score = 0.9
            result = "pass"

        # 2. spaCy NER for ORG/NORP
        if not candidates:
            for src in sources:
                entities = extract_entities(src)
                for ent, label in entities:
                    if label in {"ORG", "NORP"}:
                        candidates.append(ent)
                        method = "spacy_ner"
                        score = 0.85
                        result = "pass"
                        break
                if candidates:
                    break

        # 3. Fallback to get_buttons accessor
        if not candidates:
            buttons = self.get_buttons(contest_title=contest_title, keyword=keyword, url=url)
            for btn in buttons:
                label = btn.get("label")
                if label:
                    candidates.append(label)
            if candidates:
                method = "direct_lookup"
                score = 1.0
                result = "pass"

        # Deduplicate
        candidates = list(dict.fromkeys(candidates))

        self.log_field_selection(
            field_type="buttons",
            field_name="buttons",
            extracted_value=candidates,
            method=method,
            score=score,
            result=result,
            context={"contest_title": contest_title, "keyword": keyword, "url": url},
            user_feedback=user_feedback,
            log_path="field_selection_log.jsonl"
        )
        return candidates

    def score_header(self, title, context=None):
        # Simple fallback: just call score_entry or return a default score
        return self.score_entry(title) if hasattr(self, "score_entry") else 0.5

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
        First, check the button selection log for a successful match.
        """
        # 1. Check button selection log for a successful match
        log_path = "button_selection_log.jsonl"
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                    except Exception:
                        continue
                    # Check for a successful result for this contest_title/keyword/url
                    if contest_title and entry.get("contest_title") == contest_title and entry.get("result", "").startswith("pass"):
                        # Reconstruct a button dict from the log entry
                        button = {
                            "label": entry.get("button_label"),
                            "selector": entry.get("selector"),
                            # Optionally add more fields if you log them
                        }
                        return [button]
                    if keyword and keyword.lower() in (entry.get("button_label") or "").lower() and entry.get("result", "").startswith("pass"):
                        button = {
                            "label": entry.get("button_label"),
                            "selector": entry.get("selector"),
                        }
                        return [button]
                    if url and url in (entry.get("selector") or "") and entry.get("result", "").startswith("pass"):
                        button = {
                            "label": entry.get("button_label"),
                            "selector": entry.get("selector"),
                        }
                        return [button]

        # 2. Fallback to existing logic
        if not self.organized:
            return []
        buttons_dict = self.organized.get("buttons", {})
        results = []

        # By contest title (exact match)
        if contest_title and isinstance(contest_title, str):
            results = buttons_dict.get(contest_title, [])
            if results:
                return results

        # By keyword in label or selector
        if keyword:
            keyword = keyword.lower()
            for btn_list in buttons_dict.values():
                for btn in btn_list:
                    if keyword in btn.get("label", "").lower() or keyword in btn.get("selector", "").lower():
                        results.append(btn)
            if results:
                return results

        # By URL (if you want to associate buttons with URLs)
        if url:
            for btn_list in buttons_dict.values():
                for btn in btn_list:
                    if url in btn.get("selector", ""):
                        results.append(btn)
            if results:
                return results

        # Fallback: return all buttons
        all_buttons = []
        for btns in buttons_dict.values():
            all_buttons.extend(btns)
        return all_buttons

    def matches_html_label_pattern(label, patterns):
        """Check if label matches any HTML-specific regex pattern."""
        for pat in patterns:
            if re.search(pat, label, re.IGNORECASE):
                return True
        return False

    def log_pattern_attempt(self, label, pattern, result, context=None):
        """Log each pattern attempt for self-learning."""
        log_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "label": label,
            "pattern": pattern,
            "result": result,  # e.g., "match", "no_match", "clicked", "skipped"
            "context": context or {}
        }
        log_dir = "log"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "pattern_attempts_log.jsonl")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(safe_for_json(log_entry)) + "\n")

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
        if not hasattr(self, "_semantic_model"):
            self._semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
        model = self._semantic_model
        context = context or {}
        context.update({
            "contest_title": contest_title,
            "year": context.get("year", ""),
            "election_type": context.get("election_type", ""),
            "county": context.get("county", ""),
            "state": context.get("state", "")
        })

        # --- 1. Learning mode: check log/DB for confirmed button ---
        if learning_mode:
            learned_btn = self._get_confirmed_button_from_log(contest_title, keywords, context)
            if learned_btn and learned_btn.get("selector") not in self.clicked_button_selectors:
                selector_html = learned_btn.get("selector", "")
                dom_candidates = []
                # PATCH: Use a broad selector for all clickable elements
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
                if learned_btn.get("element_handle") and learned_btn.get("is_visible") and learned_btn.get("is_clickable"):
                    rprint(f"[green][LEARNING] Auto-applying learned button: {learned_btn['label']}[/green]")
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
        # PATCH: Use a broad selector for all clickable elements
        BUTTON_SELECTORS = (
            "button, a[href], [role='button'], input[type='button'], input[type='submit'], "
            "[tabindex]:not([tabindex='-1'])"
        )
        button_features = page.locator(BUTTON_SELECTORS)

        def scan_btn(btn, i):
            try:
                # PATCH: Robust label extraction
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
        all_candidates = [c for c in all_candidates if c.get("selector") not in self.clicked_button_selectors]

        # --- 5. Adaptive threshold: try high, then lower if no match ---
        excluded_labels = set()
        while True:
            found = False
            for threshold in fuzzy_thresholds:
                for idx, cand in enumerate(all_candidates):
                    if cand["combined_score"] >= threshold and cand.get("is_visible") and cand.get("is_clickable"):
                        if cand["label"] in excluded_labels:
                            continue
                        confirmed = True
                        if confirm_button_callback:
                            confirmed = confirm_button_callback(cand)
                        if confirmed:
                            rprint(f"[bold green][Coordinator] Confirmed button: '{cand['label']}' (score={cand['combined_score']:.2f})[/bold green]")
                            self._log_button_memory(cand, contest_title, f"confirmed_pass_{cand['combined_score']:.2f}")
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
                            excluded_labels.add(cand["label"])
                            rprint(f"[yellow][Coordinator] Button '{cand['label']}' rejected, retrying...[/yellow]")
                            found = True
                            break
                if found:
                    break
            else:
                break

        # --- 6. Feedback UI: Prompt user for manual correction ---
        if prompt_user_for_button:
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
        log_entry = {
            "contest_title": contest_title,
            "button_label": button.get("label"),
            "selector": button.get("selector"),
            "context": context,
            "result": "learning_confirmed"
        }
        log_dir = "log"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "button_learning_log.jsonl")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(safe_for_json(log_entry)) + "\n")

    def _get_confirmed_button_from_log(self, contest_title, keywords, context):
        """
        Retrieve a previously confirmed button from the learning log.
        """
        log_path = os.path.join("log", "button_learning_log.jsonl")
        if not os.path.exists(log_path):
            return None
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except Exception:
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
        log_entry = {
            "contest_title": contest_title,
            "button_label": button.get("label"),
            "selector": button.get("selector"),
            "result": result
        }
        log_dir = "log"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "button_selection_log.jsonl")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(safe_for_json(log_entry)) + "\n")

    # --- Table structure learning/lookup ---
    def get_table_structure(self, contest_title, context=None, learning_mode=True):
        """
        Retrieve or learn the expected table structure for a contest.
        """
        log_path = os.path.join("log", "table_structure_learning_log.jsonl")
        # 1. Learning mode: check log for confirmed structure
        if learning_mode and os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                    except Exception:
                        continue
                    if entry.get("contest_title") == contest_title and entry.get("result") == "learning_confirmed":
                        return entry.get("headers")
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
        log_dir = "log"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "table_structure_learning_log.jsonl")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(safe_for_json(log_entry)) + "\n")

    # --- CLI for reviewing/editing corrections and feedback ---
    def review_and_edit_corrections(self, field_type="buttons"):
        """
        Launch the manual_correction_bot CLI for reviewing/editing corrections and feedback.
        """
        script_path = os.path.join(os.path.dirname(__file__), "..", "bots", "manual_correction_bot.py")
        subprocess.run(["python", script_path, "--fields", field_type, "--feedback", "--enhanced"])

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

    def _log_get_contests_access(self, filters):
        log_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "method": "get_contests",
            "filters": filters,
        }
        log_dir = "log"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "get_contests_access_log.jsonl")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(safe_for_json(log_entry)) + "\n")

    def _log_get_buttons_access(self, contest_title, keyword, url):
        log_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "method": "get_buttons",
            "contest_title": contest_title,
            "keyword": keyword,
            "url": url,
        }

        log_dir = "log"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "get_buttons_access_log.jsonl")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(safe_for_json(log_entry)) + "\n")

    def _log_get_best_button_access(self, contest_title, keywords, class_hint, url):
        log_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "method": "get_best_button",
            "contest_title": contest_title,
            "keywords": keywords,
            "class_hint": class_hint,
            "url": url,
        }
        log_dir = "log"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "get_best_button_access_log.jsonl")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(safe_for_json(log_entry)) + "\n")

    def _log_get_panel_access(self, contest_title):
        log_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "method": "get_panel",
            "contest_title": contest_title,
        }
        log_dir = "log"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "get_panel_access_log.jsonl")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(safe_for_json(log_entry)) + "\n")

    def _log_get_tables_access(self, contest_title):
        log_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "method": "get_tables",
            "contest_title": contest_title,
        }
        log_dir = "log"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "get_tables_access_log.jsonl")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(safe_for_json(log_entry)) + "\n")

    def _log_get_candidates_access(self, contest_title):
        log_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "method": "get_candidates",
            "contest_title": contest_title,
        }
        log_dir = "log"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "get_candidates_access_log.jsonl")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(safe_for_json(log_entry)) + "\n")

    def _log_get_districts_access(self, state, county):
        log_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "method": "get_districts",
            "state": state,
            "county": county,
        }
        log_dir = "log"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "get_districts_access_log.jsonl")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(safe_for_json(log_entry)) + "\n")

    def _log_get_states_access(self):
        log_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "method": "get_states",
        }
        log_dir = "log"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "get_states_access_log.jsonl")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(safe_for_json(log_entry)) + "\n")

    def _log_get_election_types_access(self):
        log_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "method": "get_election_types",
        }
        log_dir = "log"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "get_election_types_access_log.jsonl")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(safe_for_json(log_entry)) + "\n")

    def _log_get_years_access(self):
        log_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "method": "get_years",
        }
        log_dir = "log"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "get_years_access_log.jsonl")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(safe_for_json(log_entry)) + "\n")

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
        Update a contest in the DB and context library, then re-organize context.
        """
        from ..utils.db_utils import update_contest_in_db
        from ..utils.shared_logic import save_context_library  # If you have a save function

        # 1. Update DB
        update_contest_in_db({"id": contest_id, **correction_data})

        # 2. Update context library if needed
        for key, value in correction_data.items():
            if key == "county" and value not in self.library.get("known_counties", []):
                self.library.setdefault("known_counties", []).append(value)
            if key == "state" and value not in self.library.get("known_states", []):
                self.library.setdefault("known_states", []).append(value)
            # Add similar logic for other fields as needed

        # 3. Save updated context library (if you persist it)
        save_context_library(self.library)  # Uncomment if you have this function

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
        print_integrity_summary(contests, expected_year, X=X)
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
    Robustly detect county (first) and state (second) using all available clues and cross-referencing.
    Utilizes context fields, contest titles, URL, and context_library mappings.
    Returns (county, state, handler_path, detection_log)
    """
    detection_log = []
    # --- Load and normalize all mappings ---
    state_module_map = context_library.get("state_module_map", {})
    state_to_county = context_library.get("Known_state_to_county_map", {})
    county_to_district = context_library.get("Known_county_to_district_map", {})
    known_states = [normalize_state_name(s) for s in context_library.get("known_states", [])]
    all_counties = []
    for counties in state_to_county.values():
        all_counties.extend([normalize_county_name(c) for c in counties])
    all_counties = list(set(all_counties))
    all_districts = []
    for districts in county_to_district.values():
        all_districts.extend([normalize_county_name(d) for d in districts])
    all_districts = list(set(all_districts))

    # --- 1. Try context fields directly (normalize and validate) ---
    raw_county = context.get("county")
    raw_state = context.get("state")
    county = normalize_county_name(raw_county) if raw_county else None
    state = normalize_state_name(raw_state) if raw_state else None

    # Validate county: is it a real county, or a district?
    if county:
        if county in all_counties:
            detection_log.append(f"County found in context: {county} (validated as county)")
        elif county in all_districts:
            # Map up to parent county
            parent_county = None
            for c, districts in county_to_district.items():
                if county in [normalize_county_name(d) for d in districts]:
                    parent_county = normalize_county_name(c)
                    break
            if parent_county:
                detection_log.append(f"County '{county}' found in context, but is a district. Mapped to parent county '{parent_county}'.")
                county = parent_county
            else:
                detection_log.append(f"County '{county}' found in context, but is a district with no parent mapping.")
        else:
            detection_log.append(f"County '{county}' found in context, but not recognized as county or district.")
            county = None

    # Validate state: is it a real state?
    if state:
        if state in known_states:
            detection_log.append(f"State found in context: {state} (validated as state)")
        else:
            # Try to map via state_module_map
            mapped_state = state_module_map.get(state, None)
            if mapped_state:
                detection_log.append(f"State '{state}' found in context, mapped to '{mapped_state}' via state_module_map.")
                state = normalize_state_name(mapped_state)
            else:
                detection_log.append(f"State '{state}' found in context, but not recognized.")
                state = None

    # --- 2. Try to extract county from contest titles (if not found or not valid) ---
    if not county and context.get("contests"):
        for contest in context["contests"]:
            title = contest.get("title", "")
            # Try to match any county as a whole word in the title
            for c in all_counties:
                if re.search(rf"\b{re.escape(c)}\b", title.lower()):
                    county = c
                    detection_log.append(f"County '{county}' detected from contest title: '{title}'")
                    break
            if not county:
                # Try to match any district as a whole word in the title
                for d in all_districts:
                    if re.search(rf"\b{re.escape(d)}\b", title.lower()):
                        # Map up to parent county
                        for c, districts in county_to_district.items():
                            if d in [normalize_county_name(x) for x in districts]:
                                county = normalize_county_name(c)
                                detection_log.append(f"District '{d}' detected from contest title: '{title}', mapped to county '{county}'")
                                break
                        if county:
                            break
            if county:
                break

    # --- 3. Try to extract county from URL (if still not found) ---
    url = context.get("url", "")
    if not county and url:
        for c in all_counties:
            if c in url.lower():
                county = c
                detection_log.append(f"County '{county}' detected from URL.")
                break
        if not county:
            for d in all_districts:
                if d in url.lower():
                    for c, districts in county_to_district.items():
                        if d in [normalize_county_name(x) for x in districts]:
                            county = normalize_county_name(c)
                            detection_log.append(f"District '{d}' detected from URL, mapped to county '{county}'")
                            break
                    if county:
                        break

    # --- 4. Try to extract county from HTML using NLP entities ---
    if not county and html:
        from ..utils.spacy_utils import extract_entities
        entities = extract_entities(html)
        gpe_entities = [normalize_county_name(ent) for ent, label in entities if label in ("GPE", "LOC")]
        for ent in gpe_entities:
            if ent in all_counties:
                county = ent
                detection_log.append(f"County '{county}' detected from HTML NLP entity.")
                break
            elif ent in all_districts:
                for c, districts in county_to_district.items():
                    if ent in [normalize_county_name(x) for x in districts]:
                        county = normalize_county_name(c)
                        detection_log.append(f"District '{ent}' detected from HTML NLP entity, mapped to county '{county}'")
                        break
                if county:
                    break

    # --- 5. Fuzzy match county if still not found ---
    if not county and url:
        url_tokens = re.split(r"[\W_]+", url.lower())
        matches = difflib.get_close_matches(" ".join(url_tokens), all_counties, n=1, cutoff=0.7)
        if matches:
            county = matches[0]
            detection_log.append(f"County '{county}' fuzzy-matched from URL tokens.")
        else:
            matches = difflib.get_close_matches(" ".join(url_tokens), all_districts, n=1, cutoff=0.7)
            if matches:
                # Map up to parent county
                for c, districts in county_to_district.items():
                    if matches[0] in [normalize_county_name(x) for x in districts]:
                        county = normalize_county_name(c)
                        detection_log.append(f"District '{matches[0]}' fuzzy-matched from URL tokens, mapped to county '{county}'")
                        break

    # --- 6. Now try to detect state, using county if found ---
    if not state and county:
        for s, counties in state_to_county.items():
            if county in [normalize_county_name(x) for x in counties]:
                state = normalize_state_name(s)
                detection_log.append(f"State '{state}' inferred from county '{county}'.")
                break

    # --- 7. Try to extract state from URL (if still not found) ---
    if not state and url:
        for s in known_states:
            if s in url.lower():
                state = s
                detection_log.append(f"State '{state}' detected from URL.")
                break

    # --- 8. Try to extract state from contest titles (if still not found) ---
    if not state and context.get("contests"):
        for contest in context["contests"]:
            title = contest.get("title", "")
            for s in known_states:
                if s in title.lower():
                    state = s
                    detection_log.append(f"State '{state}' detected from contest title: '{title}'")
                    break
            if state:
                break

    # --- 9. Try to extract state from HTML using NLP entities ---
    if not state and html:
        from ..utils.spacy_utils import extract_entities
        entities = extract_entities(html)
        gpe_entities = [normalize_state_name(ent) for ent, label in entities if label in ("GPE", "LOC")]
        for ent in gpe_entities:
            if ent in known_states:
                state = ent
                detection_log.append(f"State '{state}' detected from HTML NLP entity.")
                break

    # --- 10. Fuzzy match state if still not found ---
    if not state and url:
        url_tokens = re.split(r"[\W_]+", url.lower())
        matches = difflib.get_close_matches(" ".join(url_tokens), known_states, n=1, cutoff=0.7)
        if matches:
            state = matches[0]
            detection_log.append(f"State '{state}' fuzzy-matched from URL tokens.")

    # --- 11. If state found but no county, check for available county handlers ---
    handler_path = None
    if state and not county:
        state_key = state
        county_dir = os.path.join(
            os.path.dirname(__file__), "..", "handlers", "states", state_key, "county"
        )
        county_dir = os.path.abspath(county_dir)
        available_counties = []
        if os.path.isdir(county_dir):
            for fname in os.listdir(county_dir):
                if fname.endswith(".py") and not fname.startswith("__"):
                    county_name = fname[:-3]
                    available_counties.append(county_name)
            detection_log.append(f"Available county handlers for state '{state}': {available_counties}")
            # Try to match county from URL or HTML context to available counties
            url_and_html = (url + " " + html).lower()
            for c in available_counties:
                if c in url_and_html:
                    county = c
                    detection_log.append(f"County '{county}' matched to available handler from URL/HTML context.")
                    break
            if not county and available_counties:
                detection_log.append("No matching county handler found in URL/HTML; will use state handler.")
        else:
            detection_log.append(f"No county handler directory found for state '{state}'.")

        # Set handler path
        if county:
            handler_path = f"webapp.parser.handlers.states.{state_key}.county.{county}"
        else:
            handler_path = f"webapp.parser.handlers.states.{state_key}"

    # --- 12. If both found, set handler path ---
    if state and county:
        handler_path = f"webapp.parser.handlers.states.{state}.county.{county}"

    # --- 13. If only state found, fallback to state handler ---
    if state and not county and not handler_path:
        handler_path = f"webapp.parser.handlers.states.{state}"

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
