"""
context_organizer.py

Advanced context organizer for election HTML parsing and data integrity.
Handles data formatting, ML anomaly detection, cache-aware learning, clustering, and robust DB.
Delegates NLP/semantic logic to the context_coordinator and spacy_utils modules.
"""
from __future__ import annotations

import itertools
import os
import re
import types
from collections import Counter, defaultdict
from collections.abc import Hashable
from datetime import datetime, timezone
from difflib import get_close_matches

import matplotlib.pyplot as plt
import numpy as np
import orjson
from rich.table import Table
from sqlalchemy.exc import SQLAlchemyError

from ..config import CONTEXT_DB_PATH, CONTEXT_LIBRARY_PATH, LOG_DIR
from ..services.election_data_services import ElectionDataService
from ..utils.html_scanner import load_context_cache_from_disk
from ..utils.logger_singleton import console, logger
from ..utils.misc_utils import load_output_cache, load_processed_urls
from ..utils.model_registry import ModelRegistry
from ..utils.shared_logic import (
    _sync_type_and_election_types,
    flatten_raw_field,
    infer_contest_fields,
    normalize_label,
    safe_add,
    safe_db_call,
    safe_filename,
    safe_get_first,
    safe_items,
    safe_model_encode,
    safe_update,
    scan_environment,
)
from .Context_Library.constants import (
    BALLOT_TYPES,
    CANDIDATE_KEYWORDS,
    CONTEST_KEYWORDS,
    LOCATION_KEYWORDS,
    MISC_FOOTER_KEYWORDS,
    PARTY_KEYWORDS,
    PERCENT_KEYWORDS,
    TOTAL_KEYWORDS,
)
from .Integrity_check import detect_anomalies_with_ml, election_integrity_checks, print_ml_anomalies
from .librarian import clean_for_json, load_context_library, update_context_library

processed_urls = load_processed_urls()
output_cache = load_output_cache()
_spinner = itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])

def get_loading_indicator() -> str:
    return next(_spinner)

def ensure_dict(obj):
    if isinstance(obj, dict):
        return obj
    elif isinstance(obj, list):
        # Use 'label' or 'title' as key if possible
        result = {}
        for i, v in enumerate(obj):
            key = v.get("label") or v.get("title") or str(i) if isinstance(v, dict) else str(i)
            result[key] = v
        return result
    else:
        return {}

def remove_functions(obj) -> dict:
    if isinstance(obj, dict):
        return {k: remove_functions(v) for k, v in obj.items() if not isinstance(v, types.FunctionType)}
    elif isinstance(obj, list):
        return [remove_functions(v) for v in obj]
    else:
        return obj

def contest_hash(c) -> int:
    # Defensive: convert election_types to tuple for hashing, handle None and non-dict
    c_dict = c if isinstance(c, dict) else {}
    election_types = tuple(c_dict.get("election_types") or [])
    return hash((
        c_dict.get("title"),
        c_dict.get("year"),
        c_dict.get("county"),
        c_dict.get("type_"),
        election_types
    ))

def repair_dom_segments(segments) -> list:
    """
    Efficiently repairs parent/child relationships in a list of DOM segments.
    Ensures all children point to the correct parent and all indices are valid.
    Returns the repaired segments.
    """
    idx_map = {}
    for i, seg in enumerate(segments):
        if isinstance(seg, dict):
            idx = seg.get("_idx", i)
            idx_map[idx] = seg

    # Pass 1: Normalize children and parent_idx
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        children = seg.get("children", [])
        normalized_children = []
        for c in children:
            if isinstance(c, int) and c in idx_map:
                normalized_children.append(c)
            elif hasattr(c, "_idx"):
                c_idx = getattr(c, "_idx", None)
                if isinstance(c_idx, int) and c_idx in idx_map:
                    normalized_children.append(c_idx)
            # else skip
        seg["children"] = [c for c in normalized_children if c is not None]

        parent_idx = seg.get("parent_idx")
        if isinstance(parent_idx, dict):
            seg["parent_idx"] = parent_idx.get("_idx")
        elif not (isinstance(parent_idx, int) or parent_idx is None):
            seg["parent_idx"] = None
        parent_idx = seg.get("parent_idx")
        if parent_idx not in idx_map and parent_idx is not None:
            seg["parent_idx"] = None

    # Pass 2: Enforce bidirectional consistency
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        for child_idx in list(seg.get("children", [])):
            child = idx_map.get(child_idx)
            if isinstance(child, dict) and child.get("parent_idx") != seg.get("_idx"):
                child["parent_idx"] = seg.get("_idx")

    # Pass 3: Remove children that do not point back to parent
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        seg_idx = seg.get("_idx") if isinstance(seg, dict) else None
        valid_children = []
        for c in seg.get("children", []):
            child_node = idx_map.get(c)
            if not isinstance(child_node, dict):
                continue
            parent_idx_val = child_node.get("parent_idx") if isinstance(child_node, dict) else None
            if parent_idx_val == seg_idx:
                valid_children.append(c)
        seg["children"] = valid_children
    return segments

def _defensive_dom_check(dom_parts, url, logger=logger, log_errors=True) -> dict:
    """
    Efficiently checks for missing/empty lists in dom_parts for all major field types.
    Returns a dict of errors found.
    If log_errors is False, does not log errors (just returns the dict).
    """
    errors = {}
    field_types = [
        'contests', 'panels', 'tables', 'buttons', 'candidate_panels',
        'location_panels', 'headings', 'ballot_types', 'results_timestamps',
        'party_labels', 'vote_methods'
    ]
    dom_parts_dict = dom_parts if isinstance(dom_parts, dict) else {}
    for field in field_types:
        items = dom_parts_dict.get(field, [])
        if not isinstance(items, list) or not items or safe_get_first(items, field, url, logger) is None:
            errors[field] = f"No {field} found for {url}"
            if log_errors:
                logger.error(errors[field])
    return errors

class ContextOrganizer(object):
    def __init__(
        self,
        use_library=True,
        enable_ml=True,
        contamination=None,
        n_estimators=100,
        random_state=42,
        embedding_model="all-MiniLM-L6-v2",
        plot_anomalies=True,
        debug=False,
        fuzzy_cutoff=0.6
    ) -> None:
        self.use_library = use_library
        self.enable_ml = enable_ml
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.embedding_model = embedding_model  # can be string or model object
        self.plot_anomalies = plot_anomalies
        self.db_path = CONTEXT_DB_PATH
        self.context_library_path = CONTEXT_LIBRARY_PATH
        self.library = load_context_library() if use_library else self._default_library()
        if not isinstance(self.library, dict):
            logger.error(f"ERROR: self.library is not a dict! It is: {type(self.library)}")
            raise ValueError("Loaded context library is not a dict!")
        logger.debug(f"DEBUG: type(self.library) = {type(self.library)}")
        if not isinstance(self.library, dict):
            logger.error(f"ERROR: self.library is not a dict! It is: {type(self.library)}")
            raise ValueError("Loaded context library is not a dict!")
        self.organized = None
        self.processed_urls = load_processed_urls()
        self.output_cache = load_output_cache()
        self.debug = debug
        self.fuzzy_cutoff = fuzzy_cutoff
        self._context_cache = load_context_cache_from_disk()
        # --- Embedding model validation/loading ---
        self.embedding_model_obj = None
        self.data_service = ElectionDataService()
        try:
            if isinstance(self.embedding_model, str):
                self.embedding_model_obj = ModelRegistry.get_sentence_transformer(self.embedding_model)
                logger.info(f"[CONTEXT ORGANIZER] Loaded embedding model: {self.embedding_model}")
            elif hasattr(self.embedding_model, "encode"):
                # Looks like a SentenceTransformer or compatible model
                self.embedding_model_obj = self.embedding_model
                logger.info("[CONTEXT ORGANIZER] Using provided embedding model object.")
            else:
                # If it's a method, class, or something else, warn and set to None
                logger.warning(f"[CONTEXT ORGANIZER] Provided embedding_model is not a recognized model instance or string. Type: {type(self.embedding_model)}. Setting to None.")
                self.embedding_model_obj = None
        except Exception as e:
            logger.error(f"[CONTEXT ORGANIZER] Failed to load embedding model: {e}")
            self.embedding_model_obj = None

    @staticmethod
    def _default_library() -> dict:
        return {
            "contests": [],
            "buttons": [],
            "panels": [],
            "tables": [],
            "alerts": [],
            "labels": [],
            "election": [],
            "regex": [],
            "HTML_TAGS": [],
            "common_output_headers": [],
            "common_error_patterns": [],
            "domain_selectors": {},
            "domain_scrolls": {},
            "button_keywords": [],
            "contest_type_patterns": [],
            "vote_method_patterns": [],
            "location_patterns": [],
            "percent_patterns": [],
            "anomaly_log": [],
            "user_feedback": [],
            "download_link_patterns": [],
            "table_tags": [],
            "section_keywords": [],
            "output_file_patterns": [],
            "active_domains": [],
            "inactive_domains": [],
            "captcha_patterns": [],
            "captcha_solutions": {},
            "last_updated": None,
            "version": "1.2.0",
            "selectors": {},
            "precinct_header_tags": [],
            "default_noisy_labels": [],
            "download_links": []
        }
    @staticmethod
    def print_contest_summary(contests) -> None:
        table = Table(title="Contest Summary by State/County")
        table.add_column("Title")
        table.add_column("State")
        table.add_column("County")
        table.add_column("Year")
        for c in contests:
            if not isinstance(c, dict):
                title = state = county = year = ""
            else:
                title = str(c.get("title", ""))
                state = str(c.get("state", ""))
                county = str(c.get("county", ""))
                year = str(c.get("year", ""))
            table.add_row(title, state, county, year)
        console.print(table)

    @staticmethod
    def plot_contest_distribution(contests) -> None:
        state_county = [
            (
                c.get("state", "Unknown") if isinstance(c, dict) else "Unknown",
                c.get("county", "Unknown") if isinstance(c, dict) else "Unknown"
            )
            for c in contests
        ]
        counter = Counter(state_county)
        if counter:
            labels, values = zip(*counter.items())
            label_strs = [f"{s}\n{c}" for s, c in labels]
            plt.figure(figsize=(10, 5))
            plt.bar(label_strs, values)
            plt.xticks(rotation=90)
            plt.title("Contest Count by State/County")
            plt.tight_layout()
            plt.show()
        
    @staticmethod
    def suggest_and_apply_fixes(
        contests,
        context_library,
        logs=None,
        min_confidence=0.85,
        embedding_model=None,
        db_service: 'ElectionDataService' = None,
        parent_context=None,
    ) -> tuple:
        """
        Fix missing state/county/year/type_/election_types using all available context:
        Returns: (fixed_contests, fix_log)
        Ensures _fixed_fields is always a set internally, but converts to list before serialization.
        Handles edge cases for empty lists and index errors.
        """
        from ..utils.html_scanner import extract_year_and_type

        fix_log = []
        logs = logs if isinstance(logs, list) else []

        # Ensure _fixed_fields is a set for all contests and flatten raw
        for c in contests:
            if not isinstance(c, dict):
                continue
            if "_fixed_fields" not in c or not isinstance(c["_fixed_fields"], set):
                c["_fixed_fields"] = set(c.get("_fixed_fields", []))
            # Always flatten raw to avoid infinite nesting
            if "raw" in c and isinstance(c["raw"], dict):
                c["raw"] = flatten_raw_field(c["raw"])
            else:
                c["raw"] = flatten_raw_field(c)

        # Build lookup tables from context_library
        title_to_state = {}
        title_to_county = {}
        title_to_year = {}
        title_to_type_ = {}
        title_to_election_types = {}
        contests_lib = context_library.get("contests", []) if isinstance(context_library, dict) else []
        for lib_c in contests_lib:
            if not isinstance(lib_c, dict):
                continue
            title = lib_c.get("title") or lib_c.get("label")
            key = title.lower() if isinstance(title, str) else ""
            if key:
                if lib_c.get("state"):
                    title_to_state[key] = lib_c["state"]
                if lib_c.get("county"):
                    title_to_county[key] = lib_c["county"]
                if lib_c.get("year"):
                    title_to_year[key] = lib_c["year"]
                if lib_c.get("type_"):
                    title_to_type_[key] = lib_c["type_"]
                if lib_c.get("election_types"):
                    title_to_election_types[key] = lib_c["election_types"]

        # --- ML Embedding Preparation ---
        lib_titles, lib_states, lib_counties, lib_years, lib_types = [], [], [], [], []
        for lib_c in contests_lib:
            if not isinstance(lib_c, dict):
                continue
            title = lib_c.get("title") or lib_c.get("label")
            if isinstance(title, str) and (lib_c.get("state") or lib_c.get("county")):
                lib_titles.append(title)
                lib_states.append(lib_c.get("state"))
                lib_counties.append(lib_c.get("county"))
                lib_years.append(lib_c.get("year"))
                lib_types.append(lib_c.get("type_"))

        lib_embeddings = None
        if embedding_model and lib_titles:
            try:
                lib_embeddings = safe_model_encode(embedding_model, lib_titles)
            except Exception:
                lib_embeddings = None

        # Helper: update both contest and its raw field
        def update_field(c, field, value, reason):
            if not isinstance(c, dict):
                return
            c[field] = value
            raw = c.get("raw")
            if isinstance(raw, dict):
                raw[field] = value
            if "_fixed_fields" not in c or not isinstance(c["_fixed_fields"], set):
                c["_fixed_fields"] = set(c.get("_fixed_fields", []))
            safe_add(c["_fixed_fields"], field)
            logs.append(f"[FIX] {c.get('title','?')} - {field}: {value} ({reason})")

        # Helper: get from parent context
        def get_from_parent(field):
            if parent_context and isinstance(parent_context, dict):
                return parent_context.get(field)
            return None

        # Helper: get from DB
        def get_from_db(title, field, url=None):
            if db_service and isinstance(title, str):
                db_contests = db_service.get_contests_by_advanced_filter(filters={"title": title}, limit=1)
                if db_contests and isinstance(db_contests, list) and len(db_contests) > 0:
                    db_contest = safe_get_first(db_contests, "db_contests", url, logger)
                    if isinstance(db_contest, dict):
                        db_val = db_contest.get(field)
                        if db_val:
                            return db_val
            return None

        # Try to fix each contest
        fixed_hashes = set()
        for c in contests:
            if not isinstance(c, dict):
                continue
            c_hash = contest_hash(c)
            if c_hash in fixed_hashes:
                continue
            fixed = False
            reasons = []
            title_val = c.get("title")
            title = title_val.strip().lower() if isinstance(title_val, str) else ""
            raw_flat = flatten_raw_field(c.get("raw", {})) if isinstance(c.get("raw", {}), dict) else {}

            # --- Fix each field in robust order ---
            for field, lookup, lib_list, lib_vals in [
                ("state", title_to_state, lib_states, lib_states),
                ("county", title_to_county, lib_counties, lib_counties),
                ("year", title_to_year, lib_years, lib_years),
                ("type_", title_to_type_, lib_types, lib_types),
            ]:
                if c.get(field) or field in c.get("_fixed_fields", []):
                    continue

                # 1. Flattened raw field
                if isinstance(raw_flat, dict) and raw_flat.get(field):
                    update_field(c, field, raw_flat[field], "flattened raw")
                    reasons.append("flattened raw")
                    fixed = True
                    continue

                # 2. Context library
                if title in lookup:
                    update_field(c, field, lookup[title], "context_library")
                    reasons.append("context_library")
                    fixed = True
                    continue

                # 3. Database
                db_val = get_from_db(c.get("title"), field, url=c.get("source_url"))
                if db_val:
                    update_field(c, field, db_val, "database")
                    reasons.append("database")
                    fixed = True
                    continue

                # 4. Majority vote
                vals = [x.get(field) for x in contests if isinstance(x, dict) and x.get(field)]
                if vals:
                    try:
                        most_common = max(set(vals), key=vals.count)
                        update_field(c, field, most_common, "majority vote")
                        reasons.append("majority vote")
                        fixed = True
                        continue
                    except Exception:
                        pass

                # 5. Fuzzy match
                lookup_keys = list(lookup.keys())
                matches = get_close_matches(title, lookup_keys, n=1, cutoff=0.8) if isinstance(title, str) else []
                if matches:
                    url = c.get("source_url", "")
                    match_key = safe_get_first(matches, "matches", url, logger)
                    if match_key in lookup:
                        update_field(c, field, lookup[match_key], f"fuzzy match: {match_key}")
                        reasons.append(f"fuzzy match: {match_key}")
                        fixed = True
                        continue

                # 6. ML similarity
                if (
                    embedding_model and lib_embeddings is not None
                    and lib_list and isinstance(lib_list, list)
                    and len(lib_list) == len(lib_titles)
                ):
                    try:
                        query_title = c.get("title")
                        query_embs = safe_model_encode(embedding_model, [query_title if isinstance(query_title, str) else ""])
                        # Ensure query_embs is always a list
                        if isinstance(query_embs, np.ndarray):
                            query_embs = query_embs.tolist()
                        if query_embs is not None and len(query_embs) > 0:
                            query_emb = safe_get_first(query_embs, "query_embs", c.get("source_url", ""), logger)
                            if query_emb is not None:
                                sims = np.dot(lib_embeddings, query_emb) / (
                                    np.linalg.norm(lib_embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-8
                                )
                                if len(sims) == 0:
                                    continue
                                best_idx = int(np.argmax(sims))
                                best_score = sims[best_idx]
                                if (
                                    best_idx < len(lib_vals)
                                    and best_score > min_confidence
                                    and lib_vals[best_idx]
                                ):
                                    update_field(
                                        c, field, lib_vals[best_idx],
                                        f"ML similarity: {lib_titles[best_idx]} (sim={best_score:.2f})"
                                    )
                                    reasons.append(
                                        f"ML similarity: {lib_titles[best_idx]} (sim={best_score:.2f})"
                                    )
                                    fixed = True
                                    continue
                                else:
                                    reasons.append(
                                        f"ML similarity for {field} below threshold ({best_score:.2f} < {min_confidence})"
                                    )
                    except Exception as e:
                        reasons.append(f"ML similarity failed: {e}")

                # 7. Parent context
                parent_val = get_from_parent(field)
                if parent_val:
                    update_field(c, field, parent_val, "parent context")
                    reasons.append("parent context")
                    fixed = True
                    continue

                # 8. Fallback: extract_year_and_type (only for year/type_)
                if field in ("year", "type_"):
                    title_for_extract = c.get("title", "")
                    url_for_extract = c.get("source_url", "")
                    y, t, _, _ = extract_year_and_type(title_for_extract if isinstance(title_for_extract, str) else "", url=url_for_extract if isinstance(url_for_extract, str) else "")
                    if field == "year" and y:
                        update_field(c, "year", y, "extract_year_and_type")
                        reasons.append("extract_year_and_type")
                        fixed = True
                        continue
                    if field == "type_" and t:
                        update_field(c, "type_", t, "extract_year_and_type")
                        reasons.append("extract_year_and_type")
                        fixed = True
                        continue

            # --- Fix election_types as a list ---
            election_types = c.get("election_types") if isinstance(c, dict) else []
            if not isinstance(election_types, list):
                election_types = [election_types] if election_types else []
            # 1. Flattened raw field
            raw_election_types = raw_flat.get("election_types") if isinstance(raw_flat, dict) else None
            if raw_election_types:
                if isinstance(raw_election_types, list):
                    election_types = raw_election_types
                else:
                    election_types = [raw_election_types]
                update_field(c, "election_types", election_types, "flattened raw")
                reasons.append("flattened raw")
                fixed = True
            # 2. Context library
            if not election_types and title in title_to_election_types:
                lib_etypes = title_to_election_types[title]
                if isinstance(lib_etypes, list):
                    election_types = lib_etypes
                else:
                    election_types = [lib_etypes]
                update_field(c, "election_types", election_types, "context_library")
                reasons.append("context_library")
                fixed = True
            # 3. Database
            if not election_types:
                db_val = get_from_db(c.get("title"), "election_types", url=c.get("source_url"))
                if db_val:
                    if isinstance(db_val, list):
                        election_types = db_val
                    else:
                        election_types = [db_val]
                    update_field(c, "election_types", election_types, "database")
                    reasons.append("database")
                    fixed = True
            # 4. Majority vote
            if not election_types:
                vals = []
                for x in contests:
                    if not isinstance(x, dict):
                        continue
                    et = x.get("election_types")
                    if isinstance(et, list):
                        vals.extend(et)
                    elif et:
                        vals.append(et)
                if vals:
                    counter = Counter(vals)
                    most_common = [etype for etype, count in counter.items() if count == max(counter.values())]
                    election_types = most_common
                    update_field(c, "election_types", election_types, "majority vote")
                    reasons.append("majority vote")
                    fixed = True
            # 5. Parent context
            if not election_types:
                parent_val = get_from_parent("election_types")
                if parent_val:
                    if isinstance(parent_val, list):
                        election_types = parent_val
                    else:
                        election_types = [parent_val]
                    update_field(c, "election_types", election_types, "parent context")
                    reasons.append("parent context")
                    fixed = True
            # Defensive: always set as list
            if not isinstance(election_types, list):
                election_types = [election_types] if election_types else []
            c["election_types"] = election_types

            if fixed and reasons:
                fix_log.append({"title": c.get("title") if isinstance(c, dict) else None, "fixes": reasons})
                fixed_hashes.add(c_hash)

        # Convert _fixed_fields sets to lists for downstream serialization safety
        for c in contests:
            if not isinstance(c, dict):
                continue
            if "_fixed_fields" in c and isinstance(c["_fixed_fields"], set):
                c["_fixed_fields"] = list(c["_fixed_fields"])
            # Defensive: ensure election_types is always a list
            if "election_types" not in c or not isinstance(c["election_types"], list):
                c["election_types"] = []

        return contests, fix_log

    def get_table_structures(self, filters=None, limit=100, confirmed_only=False) -> list:
        """
        Fetch table structures with optional filters and confirmation status.
        """
        return self.data_service.fetch_table_structures(filters=filters, limit=limit, confirmed_only=confirmed_only)

    def get_full_contest(self, contest_id) -> dict:
        """
        Return a contest with all related data (state, county, office, candidates, results).
        """
        contest = self.data_service.get_full_contest(contest_id)
        _sync_type_and_election_types(contest)
        return contest

    def get_all_full_contests(self, filters=None, limit=100) -> list[dict]:
        """
        Return all contests with related data, optionally filtered.
        """
        return self.data_service.get_all_full_contests(filters=filters, limit=limit)

    def list_tables(self) -> list[str]:
        """
        Return a list of all table names in the current DB schema.
        """
        return self.data_service.list_tables()

    def describe_table(self, table_name) -> dict:
        """
        Return columns and relationships for a given table.
        """
        return self.data_service.describe_table(table_name)

    def get_contests_by_advanced_filter(self, filters: dict, columns: list = None, limit=100) -> list:
        """
        Fetch contests with advanced filters and optional column selection.
        """
        # You may want to add this method to ElectionDataService for full encapsulation.
        return self.data_service.get_contests_by_advanced_filter(filters, columns, limit)

    def get_table_metadata(self, table_name) -> dict:
        """
        Return column names and types for a given table.
        """
        return self.data_service.get_table_metadata(table_name)

    def check_missing_tables(self) -> list:
        """
        Return a list of expected tables that are missing in the DB.
        """
        return self.data_service.check_missing_tables()

    def _describe_embedding_model(self, model) -> str:
        """
        Return a human-friendly description of the embedding model.
        Uses ModelRegistry.get_model_name if available, else falls back to class name or str.
        """
        try:
            if model is None:
                return "None"
            if callable(model) and not hasattr(model, "model_name_or_path"):
                return f"{type(model).__name__} (not loaded)"
            # Use ModelRegistry utility if available

            if hasattr(ModelRegistry, "get_model_name"):
                name = ModelRegistry.get_model_name(model)
                if name and isinstance(name, str):
                    return name
            if hasattr(model, "model_name_or_path"):
                return str(getattr(model, "model_name_or_path"))
            if hasattr(model, "__class__"):
                return model.__class__.__name__
            return str(model)[:80]
        except Exception as e:
            return f"Unknown model ({e})"

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

        safe_field_type = safe_filename(field_type)
        if log_path is None:
            log_path = os.path.join(LOG_DIR, f"{safe_field_type}_selection_log.jsonl")
        else:
            base = os.path.basename(log_path)
            safe_base = safe_filename(base)
            log_path = os.path.join(LOG_DIR, safe_base)
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
            f.write(orjson.dumps(log_entry) + b"\n")

    def get_for_state_router(self) -> dict:
        """
        Return a summary of contests and metadata for state routing logic.
        """
        if not isinstance(self.organized, dict) or "contests" not in self.organized or not isinstance(self.organized.get("contests"), list):
            return {}
        contests = self.organized.get("contests", [])
        states = list({c.get("state") for c in contests if isinstance(c, dict) and c.get("state")})
        counties = list({c.get("county") for c in contests if isinstance(c, dict) and c.get("county")})
        return {
            "states": states,
            "counties": counties,
            "contests": contests,
            "metadata": self.organized.get("metadata", {}) if isinstance(self.organized, dict) else {},
            "election_types": self.organized.get("election_types", []) if isinstance(self.organized, dict) else [],
        }

    def get_for_html_handler(self) -> dict:
        """
        Return all organized context needed for HTML handler logic.
        """
        return self.organized if self.organized else {}

    def get_for_table_builder(self) -> dict:
        """
        Return all table structures and related context for table builder logic.
        """
        if not isinstance(self.organized, dict):
            return {}
        return {
            "tables": self.organized.get("tables", {}) if isinstance(self.organized, dict) else {},
            "contests": self.organized.get("contests", []) if isinstance(self.organized, dict) else [],
            "metadata": self.organized.get("metadata", {}) if isinstance(self.organized, dict) else {},
        }

    def get_for_selector(self) -> dict:
        """
        Return all selectors and related context for selector logic.
        """
        if not isinstance(self.organized, dict):
            return {}
        return {
            "selectors": self.organized.get("selectors", {}) if isinstance(self.organized, dict) else {},
            "contests": self.organized.get("contests", []) if isinstance(self.organized, dict) else [],
            "election_types": self.organized.get("election_types", []) if isinstance(self.organized, dict) else [],
            "noisy_patterns": self.organized.get("noisy_patterns", []) if isinstance(self.organized, dict) else [],
            "buttons": self.organized.get("buttons", {}) if isinstance(self.organized, dict) else {},
            "panels": self.organized.get("panels", {}) if isinstance(self.organized, dict) else {},
            "tables": self.organized.get("tables", {}) if isinstance(self.organized, dict) else {},
            "candidate_panels": self.organized.get("candidate_panels", {}) if isinstance(self.organized, dict) else {},
            "location_panels": self.organized.get("location_panels", {}) if isinstance(self.organized, dict) else {},
            "headings": self.organized.get("headings", {}) if isinstance(self.organized, dict) else {},
            "ballot_types": self.organized.get("ballot_types", {}) if isinstance(self.organized, dict) else {},
            "results_timestamps": self.organized.get("results_timestamps", {}) if isinstance(self.organized, dict) else {},
            "party_labels": self.organized.get("party_labels", {}) if isinstance(self.organized, dict) else {},
            "vote_methods": self.organized.get("vote_methods", {}) if isinstance(self.organized, dict) else {},
            "metadata": self.organized.get("metadata", {}) if isinstance(self.organized, dict) else {},
        }  
        
    def organize_context(
        self,
        raw_context,
        button_features=None,
        panel_features=None,
        use_library=None,
        cache=None,
        enable_ml=None,
        contamination=None,
        n_estimators=None,
        random_state=None,
        embedding_model=None,
        plot_anomalies=None,
        plot_clusters_flag=True,
        debug=None,
        fuzzy_cutoff=None,
        suppress_dom_errors=False,
    ) -> dict:
        """
        Organizes the context for a parsed HTML page, including DOM structure, contests, panels, buttons, tables, and ML features.
        Ensures all contests have title, year, type_, state, county using all available context, title parsing, and fallback logic.
        Enhanced: robust keyword-based grouping, use_library/cache integration, and diagnostics.
        Handles edge cases to avoid list index out of range errors.
        """
        
        # Defensive logging of raw_context keys
        if isinstance(raw_context, dict):
            logger.debug("DEBUG: raw_context keys: %s", list(raw_context.keys()))
        else:
            logger.debug("DEBUG: raw_context is not a dict, type: %s", type(raw_context))
        debug = self.debug if debug is None else debug
        fuzzy_cutoff = self.fuzzy_cutoff if fuzzy_cutoff is None else fuzzy_cutoff
        embedding_model = embedding_model if embedding_model is not None else self.embedding_model_obj
        plot_anomalies = plot_anomalies if plot_anomalies is not None else self.plot_anomalies
        log = []
        summary = {"attempts": [], "final": None, "error": None}

        logger.info(
            "\n[CONTEXT ORGANIZER] Pipeline configuration:\n"
            f"  • Embedding model: {self._describe_embedding_model(embedding_model)}\n"
            f"  • Plot anomalies:  {plot_anomalies}\n"
        )
        organized = None
        try:
            # --- Use/merge context library if provided ---
            context_library = self.library.copy() if hasattr(self, 'library') else {}
            if use_library:
                context_library.update(use_library)
                log.append("[LIBRARY] Merged use_library into context_library.")
            # --- Use cache if provided ---
            if cache is not None:
                if hasattr(self, '_context_cache'):
                    self._context_cache.update(cache)
                else:
                    self._context_cache = cache.copy() if isinstance(cache, dict) else dict(cache) if isinstance(cache, list) else {}
                log.append(f"[CACHE] Using provided cache with {len(cache) if hasattr(cache, '__len__') else 0} entries.")

            tagged_segments = raw_context.get("tagged_segments_with_attrs", []) if isinstance(raw_context, dict) else []
            url_value = raw_context.get("url", "") if isinstance(raw_context, dict) else ""

            # --- Virtual root handling: _idx=0, all real roots point to it as parent ---
            virtual_root = {
                "tag": "url_root",
                "attrs": {},
                "classes": [],
                "id": "url_root",
                "html": url_value,
                "is_button": False,
                "is_clickable": False,
                "children": [],
                "parent_idx": None,
                "start": None,
                "end": None,
                "_idx": 0
            }
            for seg in tagged_segments:
                if not isinstance(seg, dict):
                    continue
                seg["_idx"] = seg.get("_idx", 0) + 1
                if seg.get("parent_idx") is None:
                    seg["parent_idx"] = 0
                elif isinstance(seg.get("parent_idx"), int):
                    seg["parent_idx"] += 1
                seg["children"] = [c + 1 if isinstance(c, int) else c for c in seg.get("children", [])]
            tagged_segments = [virtual_root] + tagged_segments
            tagged_segments = repair_dom_segments(tagged_segments)
            dom_tree = self.build_dom_tree(tagged_segments)
            dom_tree["source_url"] = url_value
            dom_parts = self.expose_dom_parts(dom_tree)

            # --- Defensive: check for missing/empty lists in dom_parts ---
            url = raw_context.get("url", "") if isinstance(raw_context, dict) else ""
            context = {}
            dom_errors = _defensive_dom_check(dom_parts, url, logger, log_errors=not suppress_dom_errors)
            if dom_errors:
                context['dom_errors'] = dom_errors

            # --- Logging dom_errors for diagnostics ---
            if dom_errors:
                summary['dom_errors'] = dom_errors
                log.append(f"[DOM_PARTS] Errors: {dom_errors}")

            # --- Merge and deduplicate all context types from DB ---
            def dedupe(items, key_fields):
                seen = set()
                deduped = []
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    key = tuple(item.get(f) for f in key_fields)
                    if key not in seen and any(item.get(f) for f in key_fields):
                        deduped.append(item)
                        seen.add(key)
                return deduped

            def safe_list(val):
                return val if isinstance(val, list) else []

            # --- Organize contests, panels, tables, etc. (original logic unchanged) ---
            db_contests = safe_db_call(self.data_service.get_all_full_contests, limit=500, default=[], logger=logger)
            contests = safe_list((raw_context if isinstance(raw_context, dict) else {}).get("contests", [])) + db_contests
            contests = dedupe(contests, ["title", "year", "type_"])

            db_panels = safe_db_call(self.data_service.get_all_panels, limit=500, default=[], logger=logger)
            panels = safe_list((raw_context if isinstance(raw_context, dict) else {}).get("panels", [])) + db_panels
            panels = dedupe(panels, ["panel_text", "segment_hash"])

            db_tables = safe_db_call(self.data_service.get_all_tables, limit=500, default=[], logger=logger)
            tables = safe_list((raw_context if isinstance(raw_context, dict) else {}).get("tables", [])) + db_tables
            tables = dedupe(tables, ["table_text", "segment_hash"])

            db_candidate_panels = safe_db_call(self.data_service.get_all_candidate_panels, limit=500, default=[], logger=logger)
            candidate_panels = safe_list((raw_context if isinstance(raw_context, dict) else {}).get("candidate_panels", [])) + db_candidate_panels
            candidate_panels = dedupe(candidate_panels, ["candidate_panel_text", "segment_hash"])

            db_location_panels = safe_db_call(self.data_service.get_all_location_panels, limit=500, default=[], logger=logger)
            location_panels = safe_list((raw_context if isinstance(raw_context, dict) else {}).get("location_panels", [])) + db_location_panels
            location_panels = dedupe(location_panels, ["location_panel_text", "segment_hash"])

            db_headings = safe_db_call(self.data_service.get_all_headings, limit=500, default=[], logger=logger)
            headings = safe_list((raw_context if isinstance(raw_context, dict) else {}).get("headings", [])) + db_headings
            headings = dedupe(headings, ["heading_text", "segment_hash"])

            db_ballot_types = safe_db_call(self.data_service.get_all_ballot_types, limit=500, default=[], logger=logger)
            ballot_types = safe_list((raw_context if isinstance(raw_context, dict) else {}).get("ballot_types", [])) + db_ballot_types
            ballot_types = dedupe(ballot_types, ["ballot_types_text", "segment_hash"])

            db_results_timestamps = safe_db_call(self.data_service.get_all_results_timestamps, limit=500, default=[], logger=logger)
            results_timestamps = safe_list((raw_context if isinstance(raw_context, dict) else {}).get("results_timestamps", [])) + db_results_timestamps
            results_timestamps = dedupe(results_timestamps, ["timestamp_text", "segment_hash"])

            db_party_labels = safe_db_call(self.data_service.get_all_party_labels, limit=500, default=[], logger=logger)
            party_labels = safe_list((raw_context if isinstance(raw_context, dict) else {}).get("party_labels", [])) + db_party_labels
            party_labels = dedupe(party_labels, ["party_label_text", "segment_hash"])

            db_vote_methods = safe_db_call(self.data_service.get_all_vote_methods, limit=500, default=[], logger=logger)
            vote_methods = safe_list((raw_context if isinstance(raw_context, dict) else {}).get("vote_methods", [])) + db_vote_methods
            vote_methods = dedupe(vote_methods, ["vote_method_text", "segment_hash"])

            # --- Robust contest organization using all available keywords ---
            contests_out = []
            contests_seen = set()
            raw_contests = safe_list((raw_context if isinstance(raw_context, dict) else {}).get("contests", []))
            for c in raw_contests:
                if not isinstance(c, dict) or not c.get("title"):
                    continue
                title = c.get("title") or c.get("label") or ""
                if not title or len(str(title)) > 500:
                    logger.warning(f"[CONTEST] Skipping contest with suspiciously large or missing title: {str(title)[:100]}...")
                    continue
                norm_title = normalize_label(title)
                if norm_title not in contests_seen:
                    contests_seen.add(norm_title)
                    year, type_, state, county = infer_contest_fields(
                        c,
                        context_library,
                        db_service=self.data_service,
                        embedding_model=embedding_model,
                        log=log
                    )
                    contests_out.append({
                        "title": title,
                        "year": year,
                        "type_": type_,
                        "state": state or (c.get("state") if isinstance(c, dict) else None) or (raw_context.get("state") if isinstance(raw_context, dict) else None),
                        "county": county or (c.get("county") if isinstance(c, dict) else None) or (raw_context.get("county") if isinstance(raw_context, dict) else None),
                        "raw": flatten_raw_field(c)
                    })
            for c in safe_list(context_library.get("contests", [])):
                if not isinstance(c, dict) or not c.get("title"):
                    continue
                norm_title = normalize_label(c.get("title", c.get("label", str(c))))
                if norm_title not in contests_seen and all(
                    norm_title != normalize_label(c2.get("title", "")) if isinstance(c2, dict) else True
                    for c2 in contests_out
                ):
                    contests_out.append(c)
                    contests_seen.add(norm_title)
            contests = contests_out

            from ..utils.html_scanner import extract_year_and_type
            if isinstance(raw_context, dict):
                state_context = raw_context.get("state")
                county_context = raw_context.get("county")
            else:
                state_context = None
                county_context = None
            years = [c.get("year") for c in contests if isinstance(c, dict) and c.get("year")]
            types = [c.get("type_") for c in contests if isinstance(c, dict) and c.get("type_")]
            unique_years = set(y for y in years if y)
            unique_types = set(t for t in types if t)

            for c in contests:
                if not isinstance(c, dict):
                    continue

                # Fill state/county from context if missing
                if not c.get("state") and state_context:
                    c["state"] = state_context
                if not c.get("county") and county_context:
                    c["county"] = county_context

                # Extract year/type once per contest if needed
                title_val = c.get("title", "")
                url_val = raw_context.get("url", "") if isinstance(raw_context, dict) else ""
                y, t, _, _ = extract_year_and_type(title_val if isinstance(title_val, str) else "", url=url_val if isinstance(url_val, str) else "")

                # Fill year
                if not c.get("year"):
                    if len(unique_years) == 1:
                        c["year"] = safe_get_first(list(unique_years), "unique_years", url, logger)
                    elif y:
                        c["year"] = y

                # Fill type_
                if not c.get("type_"):
                    if len(unique_types) == 1:
                        c["type_"] = safe_get_first(list(unique_types), "unique_types", url, logger)
                    elif t:
                        c["type_"] = t

                # Fallback: try to infer from raw field
                raw_field = c.get("raw")
                if isinstance(raw_field, dict):
                    if not c.get("state") and raw_field.get("state"):
                        c["state"] = raw_field.get("state")
                    if not c.get("county") and raw_field.get("county"):
                        c["county"] = raw_field.get("county")
                    if not c.get("year") and raw_field.get("year"):
                        c["year"] = raw_field.get("year")
                    if not c.get("type_") and raw_field.get("type_"):
                        c["type_"] = raw_field.get("type_")
            # --- Filter out contests missing title (required) ---
            filtered_out = []
            filtered_contests = []
            for c in contests:
                if not (isinstance(c, dict) and c.get("title")):
                    filtered_out.append((c, "missing title"))
                    continue
                filtered_contests.append(c)
            if filtered_out:
                logger.warning(f"[CONTEST] Filtered out {len(filtered_out)} contests due to missing required fields.")
                for c, reason in filtered_out[:5]:
                    logger.warning(f"  [Filtered] {reason}: {str(c)[:100]}...")
            contests = filtered_contests
            if not contests:
                logger.warning("[CONTEST] No contests with required fields for downstream output.")

            # --- Panels: relaxed filtering, only require panel_text ---
            panels_dict = {}
            lib_panels = safe_list(context_library.get("panels", []))
            def find_panel_by_title(title):
                norm_title = normalize_label(title)
                # 1. panel_features
                if panel_features and isinstance(panel_features, list):
                    for p in panel_features:
                        if not isinstance(p, dict):
                            continue
                        if normalize_label(p.get("label", p.get("panel_text", ""))) == norm_title:
                            return p
                # Defensive extraction of raw_panels
                raw_panels = raw_context.get("panels", {}) if isinstance(raw_context, dict) else {}
                if isinstance(raw_panels, dict):
                    for k, p in raw_panels.items():
                        if not isinstance(p, dict):
                            continue
                        if normalize_label(p.get("label", p.get("panel_text", ""))) == norm_title:
                            return p
                elif isinstance(raw_panels, list):
                    for p in raw_panels:
                        if not isinstance(p, dict):
                            continue
                        if normalize_label(p.get("label", p.get("panel_text", ""))) == norm_title:
                            return p
                # 3. context_library panels
                for p in lib_panels:
                    if not isinstance(p, dict):
                        continue
                    if normalize_label(p.get("label", p.get("panel_text", ""))) == norm_title:
                        return p
                return None

            for c in contests:
                panel = find_panel_by_title(c["title"])
                if panel and panel.get("panel_text"):
                    panels_dict[c["title"]] = panel

            # Defensive extraction of raw_panels
            raw_panels = raw_context.get("panels", {}) if isinstance(raw_context, dict) else {}

            # Defensive conversion to list for iteration
            if isinstance(raw_panels, dict):
                raw_panels_list = list(raw_panels.values())
            elif isinstance(raw_panels, list):
                raw_panels_list = raw_panels
            else:
                raw_panels_list = []

            # Defensive iteration over panels
            for p in (panel_features if isinstance(panel_features, list) else []) + raw_panels_list + (lib_panels if isinstance(lib_panels, list) else []):
                if not isinstance(p, dict):
                    continue
                if not p.get("panel_text"):
                    continue
                label = p.get("label", p.get("panel_text", ""))
                if label not in panels_dict:
                    panels_dict[label] = p

            # --- Tables: relaxed filtering, only require table_text ---
            tables_by_contest = defaultdict(list)
            raw_tables = safe_list(raw_context.get("tables", []) if isinstance(raw_context, dict) else [])
            lib_tables = safe_list(context_library.get("tables", []))
            all_tables = raw_tables + lib_tables
            for tbl in all_tables:
                if not isinstance(tbl, dict):
                    continue
                if not tbl.get("table_text"):
                    continue
                for c in contests:
                    if (
                        isinstance(c, dict) and isinstance(tbl, dict)
                        and isinstance(c.get("title"), str) and isinstance(tbl.get("label"), str)
                    ):
                        c_title = c.get("title")
                        tbl_label = tbl.get("label")
                        # Only proceed if both are non-empty strings and .lower() is safe
                        if (
                            isinstance(c_title, str) and isinstance(tbl_label, str)
                            and c_title and tbl_label
                            and c_title.lower() in tbl_label.lower()
                        ):
                            tables_by_contest[c["title"]].append(tbl)
                    if not any(tbl in v for v in tables_by_contest.values()):
                        tables_by_contest["__unmatched__"].append(tbl)

            # --- Buttons: relaxed filtering, only require label ---
            buttons_by_contest = defaultdict(list)
            raw_buttons = safe_list(button_features) if isinstance(button_features, list) else safe_list(raw_context.get("buttons", []) if isinstance(raw_context, dict) else [])
            lib_buttons = safe_list(context_library.get("buttons", []))
            all_buttons = raw_buttons + lib_buttons
            unmatched_buttons = []
            for btn in all_buttons:
                if not isinstance(btn, dict):
                    continue
                label = btn.get("label")
                if not isinstance(label, str) or not label:
                    continue
                matched = False
                for c in contests:
                    if not isinstance(c, dict):
                        continue
                    title = c.get("title")
                    if not isinstance(title, str) or not title:
                        continue
                    # Safeguard .lower() calls
                    label_lower = label.lower() if isinstance(label, str) else ""
                    title_lower = title.lower() if isinstance(title, str) else ""
                    if title_lower and label_lower and title_lower in label_lower:
                        buttons_by_contest[title].append(btn)
                        matched = True
                    elif "election" in label_lower and "election" in title_lower:
                        buttons_by_contest[title].append(btn)
                        matched = True
                if not matched:
                    unmatched_buttons.append(btn)
            for btn in unmatched_buttons:
                buttons_by_contest["__unmatched__"].append(btn)

            # --- Grouping by keywords with specific label fields for each type ---
            keyword_sets = {
                "location": LOCATION_KEYWORDS,
                "candidate": CANDIDATE_KEYWORDS,
                "party": PARTY_KEYWORDS,
                "ballot_types": set(BALLOT_TYPES),
                "contest": CONTEST_KEYWORDS,
                "percent": PERCENT_KEYWORDS,
                "total": TOTAL_KEYWORDS,
                "footer": MISC_FOOTER_KEYWORDS
            }
            def group_by_keywords(items, label_fields=None, keyword_sets=None, fuzzy_cutoff=0.85) -> dict:
                if label_fields is None:
                    label_fields = ["label", "button_text", "panel_text", "table_text", "candidate_panel_text",
                                    "location_panel_text", "heading_text", "ballot_types_text", "timestamp_text",
                                    "party_label_text", "vote_method_text"]
                if keyword_sets is None:
                    raise ValueError("keyword_sets must be provided")
                groups = {k: [] for k in keyword_sets}
                seen = {k: set() for k in keyword_sets}
                def normalize(text):
                    return re.sub(r"[^\w\s]", "", (text or "").lower()).strip()
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    label = ""
                    for field in label_fields:
                        label = item.get(field, "")
                        if label:
                            break
                    label_norm = normalize(label)
                    tokens = set(label_norm.split())
                    for group, keywords in safe_items(keyword_sets):
                        matched = False
                        for kw in keywords:
                            kw_norm = normalize(kw)
                            if kw_norm in label_norm or any(kw_norm in t for t in tokens):
                                matched = True
                                break
                            if get_close_matches(kw_norm, [label_norm], n=1, cutoff=fuzzy_cutoff):
                                matched = True
                                break
                        if matched:
                            dedup_key = item.get("segment_hash", label_norm)
                            if dedup_key not in seen[group]:
                                groups[group].append(item)
                                seen[group].add(dedup_key)
                return groups

            # --- Use specific label fields for each group ---
            panels = ensure_dict(group_by_keywords(
                [p for p in panels_dict.values() if p],
                label_fields=["panel_text"],
                keyword_sets=keyword_sets
            ))
            buttons = ensure_dict(group_by_keywords(
                all_buttons,
                label_fields=["button_text", "label"],
                keyword_sets=keyword_sets
            ))
            tables = ensure_dict(group_by_keywords(
                all_tables,
                label_fields=["table_text"],
                keyword_sets=keyword_sets
            ))
            candidate_panels = ensure_dict(group_by_keywords(
                candidate_panels,
                label_fields=["candidate_panel_text"],
                keyword_sets=keyword_sets
            ))
            location_panels = ensure_dict(group_by_keywords(
                location_panels,
                label_fields=["location_panel_text"],
                keyword_sets=keyword_sets
            ))
            headings = ensure_dict(group_by_keywords(
                headings,
                label_fields=["heading_text"],
                keyword_sets=keyword_sets
            ))
            ballot_types = ensure_dict(group_by_keywords(
                ballot_types,
                label_fields=["ballot_types_text"],
                keyword_sets=keyword_sets
            ))
            results_timestamps = ensure_dict(group_by_keywords(
                results_timestamps,
                label_fields=["timestamp_text"],
                keyword_sets=keyword_sets
            ))
            party_labels = ensure_dict(group_by_keywords(
                party_labels,
                label_fields=["party_label_text"],
                keyword_sets=keyword_sets
            ))
            vote_methods = ensure_dict(group_by_keywords(
                vote_methods,
                label_fields=["vote_method_text"],
                keyword_sets=keyword_sets
            ))

            # --- Logging group sizes for diagnostics ---
            log.append(f"[KEYWORDS] Panel groups: { {k: len(v) for k,v in panels.items()} }")
            log.append(f"[KEYWORDS] Button groups (by button_text): { {k: len(v) for k,v in buttons.items()} }")
            log.append(f"[KEYWORDS] Table groups: { {k: len(v) for k,v in tables.items()} }")
            log.append(f"[KEYWORDS] Candidate panel groups: { {k: len(v) for k,v in candidate_panels.items()} }")
            log.append(f"[KEYWORDS] Location panel groups: { {k: len(v) for k,v in location_panels.items()} }")
            log.append(f"[KEYWORDS] Heading groups: { {k: len(v) for k,v in headings.items()} }")
            log.append(f"[KEYWORDS] Ballot type groups: { {k: len(v) for k,v in ballot_types.items()} }")
            log.append(f"[KEYWORDS] Results timestamp groups: { {k: len(v) for k,v in results_timestamps.items()} }")
            log.append(f"[KEYWORDS] Party label groups: { {k: len(v) for k,v in party_labels.items()} }")
            log.append(f"[KEYWORDS] Vote method groups: { {k: len(v) for k,v in vote_methods.items()} }")

            # --- ML anomaly detection and integrity checks ---
            anomalies, clusters = [], []
            try:
                if enable_ml and len(contests) > 0:
                    anomalies, clusters = detect_anomalies_with_ml(
                        contests,
                        contamination=contamination,
                        n_estimators=n_estimators,
                        random_state=random_state,
                        embedding_model=embedding_model
                    )
                    if anomalies:
                        for idx in anomalies:
                            if idx < len(contests):
                                contest = contests[idx]
                                title = contest.get('title', str(contest)) if isinstance(contest, dict) else str(contest)
                                logger.info(f"[bold magenta][ML][/bold magenta] Context anomaly detected: [bold yellow]{title}[/bold yellow]\n  [dim]Context:[/dim] {contest}")
                            else:
                                logger.warning(f"[ML] Anomaly index {idx} out of range for contests list of length {len(contests)}")
                    if plot_clusters_flag:
                        plot_clusters_flag = print_ml_anomalies(anomalies, contests)
            except Exception as e:
                logger.error(f"[bold red][ML] Anomaly detection failed:[/bold red] {e}")

            integrity_issues = election_integrity_checks(contests)
            contests, fix_log = self.suggest_and_apply_fixes(
                contests,
                context_library,
                logs=log,
                min_confidence=0.8,
                embedding_model=embedding_model if embedding_model is not None else self.embedding_model_obj
            )
            # --- Sync type_ and election_types for all contests ---
            for contest in contests:
                _sync_type_and_election_types(contest)

            # Get best contest type/election_types for fallback
            best_contest = contests[0] if contests else {}
            best_type = best_contest.get("type_")
            best_election_types = best_contest.get("election_types", [])

            # Sync other sections
            for section in [tables, candidate_panels, location_panels, ballot_types]:
                for item in section:
                    _sync_type_and_election_types(item, fallback_types=best_election_types, fallback_type=best_type)

            # Sync top-level organized dict
            _sync_type_and_election_types(organized, fallback_types=best_election_types, fallback_type=best_type)
            if fix_log:
                logger.info("[bold green]Auto-fixes applied:[/bold green]")
                for entry in fix_log:
                    logger.warning(f"  [yellow]{entry['title']}[/yellow]: {', '.join(entry['fixes'])}")
            integrity_issues = election_integrity_checks(contests)
            for issue, contest in integrity_issues:
                if issue == "duplicate":
                    logger.warning(f"[bold yellow][INTEGRITY][/bold yellow] Duplicate contest detected.\n  [dim]Context:[/dim] {contest}")
                elif issue == "missing_location":
                    logger.warning(f"[bold yellow][INTEGRITY][/bold yellow] Contest missing location info.\n  [dim]Context:[/dim] {contest}")
                elif issue == "missing_year":
                    logger.warning(f"[bold yellow][INTEGRITY][/bold yellow] Contest missing year.\n  [dim]Context:[/dim] {contest}")

            if len(contests) > 50:
                logger.error(f"[bold red][CONTEXT ORGANIZER][/bold red] High contest count detected — possible congestion.\n  [dim]Context:[/dim] contest_count={len(contests)}")

            organized = {
                "contests": contests,
                "panels": panels,
                "buttons": buttons,
                "tables": tables,
                "candidate_panels": candidate_panels,
                "location_panels": location_panels,
                "headings": headings,
                "ballot_types": ballot_types,
                "results_timestamps": results_timestamps,
                "party_labels": party_labels,
                "vote_methods": vote_methods,
                "election_types": raw_context.get("election_types", []) if isinstance(raw_context, dict) else [],
                "noisy_patterns": raw_context.get("noisy_patterns", []) if isinstance(raw_context, dict) else [],
                "metadata": {
                    "state": raw_context.get("state") if isinstance(raw_context, dict) else None,
                    "county": raw_context.get("county") if isinstance(raw_context, dict) else None,
                    "source_url": raw_context.get("url") if isinstance(raw_context, dict) else None,
                    "election_types": raw_context.get("election_types") if isinstance(raw_context, dict) else [],
                    "scrape_time": raw_context.get("scrape_time") if isinstance(raw_context, dict) else None,
                    "year": None,
                    "race": raw_context.get("race") if isinstance(raw_context, dict) else None,
                    "environment": scan_environment(),
                },
                "dom_tree": dom_tree,
                "dom_parts": dom_parts,
                "anomalies": [contests[i] for i in anomalies if isinstance(i, int) and 0 <= i < len(contests)] if anomalies else [],
                "clusters": clusters.tolist() if hasattr(clusters, "tolist") else clusters,
                "integrity_issues": integrity_issues,
            }
            valid_years = [
                c.get("year") for c in contests
                if isinstance(c, dict) and c.get("year") and c.get("type_") and str(c.get("year")).isdigit()
            ]
            metadata = organized["metadata"]
            _sync_type_and_election_types(metadata, fallback_types=best_election_types, fallback_type=best_type)
            if valid_years:
                metadata["year"] = safe_get_first(valid_years, "valid_years", url, logger, default="Unknown")
            else:
                metadata["year"] = "Unknown"
            self.append_to_context_library(organized, path=self.context_library_path)
            # If pivot added RawJSON enrichment to context, surface it in metadata (if present)
            if isinstance(self.last_raw_context, dict):
                pass  # placeholder if needed
            # Safer: allow any upstream context to place enrichment into raw_context["rawjson_enrichment"]
            rje = raw_context.get("rawjson_enrichment") if isinstance(raw_context, dict) else None
            if rje and isinstance(organized, dict) and "metadata" in organized:
                organized["metadata"]["rawjson_enrichment"] = rje
            logger.info(
                f"[CONTEXT ORGANIZER] Organized context for {len(contests)} contests. "
                f"Anomalies: {len(anomalies)}  Integrity issues: {len(integrity_issues)}"
            )
            logger.info(
                f"[bold green][CONTEXT ORGANIZER][/bold green] Organized context for [bold]{len(contests)}[/bold] contests.\n"
                f"  [magenta]Anomalies:[/magenta] {len(anomalies)}  [yellow]Integrity issues:[/yellow] {len(integrity_issues)}"
            )

            # Insert or update contests robustly using SQLAlchemy ORM
            try:
                for c in contests:
                    self.data_service.upsert_contest(c)
            except SQLAlchemyError as e:
                logger.error(f"[DB][Contest] Error upserting contests: {e}")

            # --- Dynamic state/county detection if missing ---
            missing_location = any(
                not (c.get("state") if isinstance(c, dict) else None) or not (c.get("county") if isinstance(c, dict) else None)
                for c in contests
            )
            if missing_location:
                from .context_coordinator import dynamic_state_county_detection
                html = raw_context.get("raw_html", "") if isinstance(raw_context, dict) else ""
                county, state, handler_path, detection_log = dynamic_state_county_detection(
                    raw_context, html, debug=True
                )
                for log_entry in detection_log:
                    log.append(f"[Dynamic Detection] {log_entry}")
                    if debug:
                        logger.info(f"[ContextOrganizer][Dynamic Detection] {log_entry}")
                if state:
                    raw_context["state"] = state
                if county:
                    raw_context["county"] = county
                # Update organized metadata so downstream has accurate location + diagnostics
                if isinstance(organized, dict) and "metadata" in organized:
                    md = organized["metadata"]
                    # Only fill if missing; do not write "Unknown"
                    if not md.get("state") and state:
                        md["state"] = state
                    if not md.get("county") and county:
                        md["county"] = county
                    md["location_detection"] = {
                        "handler_path": handler_path,
                        "log": detection_log
                    }
                summary["final"] = {"state": state, "county": county, "handler_path": handler_path}
                log.append(f"Final detected state: {state}, county: {county}, handler_path: {handler_path}")

            self.append_to_context_library(organized, path=self.context_library_path)
            logger.info(f"[CONTEXT ORGANIZER] Organized context for {len(contests)} contests.")
            self.organized = organized
        except IndexError as e:
            logger.error(f"[ERROR] Exception while processing {url}: {e}")
            context['dom_parts_keys'] = list(dom_parts.keys()) if isinstance(dom_parts, dict) else []
            context['dom_parts_lengths'] = {
                k: len(v) if isinstance(v, list) else None
                for k, v in dom_parts.items()
            } if isinstance(dom_parts, dict) else {}
            summary['error'] = context['error']
            log.append(f"[EXCEPTION] {context['error']}")
            return {
                "organized": None,
                "summary": summary,
                "log": log,
                "error": context['error'],
            }
    
    def build_dom_tree(self, segments) -> dict:
        """
        Build a robust DOM tree from a list of segments.
        Ensures parent/child consistency, valid indices, logs inconsistencies, and handles edge cases.
        Returns root indices and all nodes.
        """
        try:
            segments = repair_dom_segments(segments)
            nodes = []
            idx_map = {}
            for seg in segments:
                node = dict(seg)
                node.setdefault("children", [])
                node.setdefault("parent_idx", None)
                node.setdefault("_idx", seg.get("_idx") if isinstance(seg, dict) else None)
                node.setdefault("start", None)
                node.setdefault("end", None)
                nodes.append(node)
                idx_map[node["_idx"]] = node

            # Defensive parent/child consistency check and tree building
            for node in nodes:
                children = node.get("children", []) if isinstance(node, dict) else []
                for child_idx in children:
                    parent_idx_val = None
                    child_node = idx_map.get(child_idx)
                    if isinstance(child_node, dict):
                        parent_idx_val = child_node.get("parent_idx")
                    node_idx_val = node.get("_idx") if isinstance(node, dict) else None
                    if (
                        child_idx not in idx_map
                        or parent_idx_val != node_idx_val
                    ):
                        indicator = get_loading_indicator()
                        console.print(
                            f"{indicator} Inconsistent parent/child: node {node.get('_idx') if isinstance(node, dict) else None} child {child_idx}",
                            highlight=False,
                            end="\r"
                        )

            # Defensive: ensure no cycles and all indices valid
            valid_indices = set(idx_map.keys())
            for node in nodes:
                children = node.get("children", []) if isinstance(node, dict) else []
                node["children"] = [c for c in children if c in valid_indices and c != (node.get("_idx") if isinstance(node, dict) else None)]
                parent_idx = node.get("parent_idx") if isinstance(node, dict) else None
                if parent_idx not in valid_indices and parent_idx is not None:
                    node["parent_idx"] = None

            # Defensive: ensure roots are truly root nodes
            roots = [node for node in nodes if (node.get("parent_idx") if isinstance(node, dict) else None) is None]
            if not roots:
                root_node = next((n for n in nodes if (n.get("_idx") if isinstance(n, dict) else None) == 0), None)
                if root_node:
                    roots = [root_node]
                else:
                    roots = [nodes[0]] if nodes else []

            dom_tree = {
                "roots": [node.get("_idx") if isinstance(node, dict) else None for node in roots if (node.get("_idx") if isinstance(node, dict) else None) is not None],
                "nodes": nodes
            }
            return dom_tree
        except Exception as e:
            logger.error(f"[build_dom_tree] Error: {e}")
            return {"roots": [], "nodes": []}

    def expose_dom_parts(self, dom_tree) -> dict:
        """
        Efficiently expose organized DOM parts for downstream use.
        Returns dicts for head, body, wrappers, tables, buttons, clickable, and all nodes.
        Adds 'direct_parent' field for each node.
        """
        nodes = dom_tree.get("nodes", []) if isinstance(dom_tree, dict) else []
        idx_map = {n.get("_idx"): n for n in nodes if isinstance(n, dict) and n.get("_idx") is not None}
        for n in nodes:
            if not isinstance(n, dict):
                continue
            parent_idx = n.get("parent_idx")
            direct_parent_tag = None
            if parent_idx is not None and parent_idx in idx_map:
                parent_node = idx_map[parent_idx]
                if isinstance(parent_node, dict):
                    direct_parent_tag = parent_node.get("tag")
            n["direct_parent"] = direct_parent_tag

        def safe_tag(node):
            tag = node.get("tag") if isinstance(node, dict) else None
            return tag.lower() if isinstance(tag, str) else ""

        return {
            "head_nodes": [n for n in nodes if safe_tag(n) == "head"],
            "body_nodes": [n for n in nodes if safe_tag(n) == "body"],
            "wrappers": [n for n in nodes if safe_tag(n) in ("div", "section", "form")],
            "tables": [n for n in nodes if safe_tag(n) == "table"],
            "buttons": [n for n in nodes if isinstance(n, dict) and n.get("is_button")],
            "clickable": [n for n in nodes if isinstance(n, dict) and n.get("is_clickable")],
            "all_nodes": nodes,
            "roots": dom_tree.get("roots", []) if isinstance(dom_tree, dict) else []
        }

    def extract_html_by_idx(self, nodes, idx, full_html) -> str:
        """
        Extract the exact HTML for a node using its start/end indices.
        Robust to missing indices, out-of-bounds, and malformed nodes.
        """
        try:
            if not isinstance(nodes, list) or not isinstance(idx, int) or idx < 0 or idx >= len(nodes):
                return ""
            node = nodes[idx]
            if not isinstance(node, dict):
                return ""
            start = node.get("start")
            end = node.get("end")
            if (
                start is not None and end is not None
                and isinstance(start, int) and isinstance(end, int)
                and 0 <= start < end <= len(full_html)
            ):
                return full_html[start:end]
            html = node.get("html", "")
            return html if isinstance(html, str) else str(html)
        except Exception as e:
            logger.error(f"[extract_html_by_idx] Error: {e}")
            return ""

    def extract_subtree_html(self, nodes, idx, full_html) -> str:
        """
        Recursively extract the HTML for a node and all its descendants.
        Robust to cycles, missing children, and index errors.
        """
        try:
            if not isinstance(nodes, list) or not isinstance(idx, int) or idx < 0 or idx >= len(nodes):
                return ""
            node = nodes[idx]
            if not isinstance(node, dict):
                return ""
            visited = set()
            indices = [idx]
            stack = list(node.get("children", []))
            while stack:
                child_idx = stack.pop()
                if not isinstance(child_idx, int) or child_idx < 0 or child_idx >= len(nodes) or child_idx in visited:
                    continue
                visited.add(child_idx)
                indices.append(child_idx)
                child_node = nodes[child_idx]
                if isinstance(child_node, dict):
                    stack.extend(child_node.get("children", []))
            starts = [
                node.get("start")
                for i in indices
                for node in [nodes[i]]  # single-item list for scoping
                if isinstance(node, dict) and "start" in node and node.get("start") is not None
            ]
            ends = [
                node.get("end")
                for i in indices
                for node in [nodes[i]]
                if isinstance(node, dict) and "end" in node and node.get("end") is not None
            ]
            if starts and ends:
                min_start = min(starts)
                max_end = max(ends)
                if (
                    isinstance(min_start, int) and isinstance(max_end, int)
                    and 0 <= min_start < max_end <= len(full_html)
                ):
                    return full_html[min_start:max_end]
            html = node.get("html", "")
            return html if isinstance(html, str) else str(html)
        except Exception as e:
            logger.error(f"[extract_subtree_html] Error: {e}")
            return ""

    def group_nodes_by_label(self, nodes, label_field="ml_label") -> dict:
        """
        Group nodes by a given label field (e.g., ml_label or semantic_tags).
        Robust to missing fields, lists, and non-string labels.
        Returns a dict: label -> list of nodes.
        """
        groups = defaultdict(list)
        try:
            for node in nodes:
                if not isinstance(node, dict):
                    continue
                label = node.get(label_field)
                if isinstance(label, list):
                    for label_value in label:
                        if label_value is not None:
                            groups[str(label_value)].append(node)
                elif label is not None:
                    groups[str(label)].append(node)
        except Exception as e:
            logger.error(f"[group_nodes_by_label] Error: {e}")
        return dict(groups)

    def get_panels_and_tables(self, dom_tree) -> list:
        """
        Returns a list of (panel_heading, [tables]) for each panel/section in the DOM.
        """
        panels = []
        nodes = dom_tree.get("nodes", []) if isinstance(dom_tree, dict) else []
        for node in nodes:
            if not isinstance(node, dict):
                continue
            tag = node.get("tag")
            if not isinstance(tag, str) or tag not in ("section", "div", "fieldset", "panel"):
                continue

            # Find heading among children
            heading = None
            children = node.get("children", [])
            for child_idx in children:
                if not isinstance(child_idx, int) or child_idx < 0 or child_idx >= len(nodes):
                    continue
                child = nodes[child_idx]
                if not isinstance(child, dict):
                    continue
                child_tag = child.get("tag")
                if isinstance(child_tag, str) and child_tag.lower() in ("h1", "h2", "h3", "h4", "h5", "h6"):
                    heading = child.get("html")
                    break

            # Find tables among children with advanced type and value checks
            tables = []
            for cidx in children:
                if not isinstance(cidx, int) or cidx < 0 or cidx >= len(nodes):
                    continue
                table_node = nodes[cidx]
                if not isinstance(table_node, dict):
                    continue
                tag_val = table_node.get("tag")
                # Advanced tag checks
                if (
                    isinstance(tag_val, str)
                    and tag_val is not None
                    and tag_val.strip() != ""
                ):
                    tag_stripped = tag_val.strip()
                    tag_lower = tag_stripped.lower() if isinstance(tag_stripped, str) else ""
                    if tag_lower == "table":
                        tables.append(table_node)

            if heading and tables:
                panels.append((heading, tables))
        return panels

    def submit_user_feedback(self, field_type, field_name, correct_value, context) -> None:
        """
        Store or process user feedback for a field extraction/correction.
        logs, deduplicates, persists to disk, and can update context library.
        """

        # Defensive: ensure feedback list exists
        if not hasattr(self, "user_feedback") or not isinstance(self.user_feedback, list):
            self.user_feedback = []

        # Defensive: sanitize and flatten context
        def safe_context(ctx) -> dict:
            try:
                if isinstance(ctx, dict):
                    return {k: str(v)[:500] for k, v in ctx.items()}
                return str(ctx)[:500]
            except Exception:
                return str(ctx)[:500]

        feedback_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "field_type": str(field_type)[:100],
            "field_name": str(field_name)[:100],
            "correct_value": str(correct_value)[:1000],
            "context": safe_context(context),
        }

        # Deduplicate: avoid duplicate feedback for same field/context in this session
        dedup_key = (
            feedback_entry["field_type"],
            feedback_entry["field_name"],
            feedback_entry["correct_value"],
            str(feedback_entry["context"])
        )
        if not hasattr(self, "_feedback_seen"):
            self._feedback_seen = set()
        if dedup_key in self._feedback_seen:
            logger.info(f"[ContextOrganizer] Duplicate feedback skipped: {feedback_entry}")
            return
        self._feedback_seen.add(dedup_key)
        self.user_feedback.append(feedback_entry)

        # Persist to disk robustly
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            log_path = os.path.join(LOG_DIR, "user_feedback_log.jsonl")
            with open(log_path, "ab") as f:
                f.write(orjson.dumps(feedback_entry) + b"\n")
        except Exception as e:
            logger.error(f"[ContextOrganizer] Failed to persist user feedback: {e}")

        # Optionally: update context library (extend as needed)
        try:
            if hasattr(self, "library") and isinstance(self.library, dict):
                if "user_feedback" not in self.library or not isinstance(self.library["user_feedback"], list):
                    self.library["user_feedback"] = []
                self.library["user_feedback"].append(feedback_entry)
        except Exception as e:
            logger.warning(f"[ContextOrganizer] Could not update context library with feedback: {e}")

        logger.info(f"[ContextOrganizer] User feedback submitted: {feedback_entry}")

    def append_to_context_library(self, organized, path=None, merge_lists=True, deduplicate=True) -> None:
        """
        Robustly append or update the organized context into the context library JSON file.
        - merge_lists: If True, lists are merged (with deduplication if deduplicate=True).
        - deduplicate: If True, removes duplicates from merged lists based on dict content or value.
        """
        def merge_dicts(a, b) -> dict:
            """Recursively merge dict b into dict a."""
            for k, v in safe_items(b):
                if k in a:
                    if isinstance(a[k], dict) and isinstance(v, dict):
                        merge_dicts(a[k], v)
                    elif isinstance(a[k], list) and isinstance(v, list) and merge_lists:
                        combined = a[k] + v
                        if deduplicate:
                            # Remove duplicates (works for dicts and primitives)
                            seen = set()
                            deduped = []
                            for item in combined:
                                try:
                                    key = orjson.dumps(item) if isinstance(item, Hashable) else str(item)
                                except Exception:
                                    key = str(item)
                                if key not in seen:
                                    seen.add(key)
                                    deduped.append(item)
                            a[k] = deduped
                        else:
                            a[k] = combined
                    else:
                        a[k] = v
                else:
                    a[k] = v
            return a

        path = path or self.context_library_path
        try:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    library = orjson.loads(f.read())
            else:
                library = {}
            organized_clean = clean_for_json(remove_functions(organized))
            library = merge_dicts(library, organized_clean)
            library["last_updated"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            library = clean_for_json(library)
            safe_update(library, organized_clean, logger)
            update_context_library(path, library)
            logger.info(f"[CONTEXT ORGANIZER] Appended/merged context to library at {path}")
        except Exception as e:
            logger.error(f"[CONTEXT ORGANIZER] Failed to append to context library: {e}")

    def save_table_structure_to_db(self, contest, headers, context, ml_confidence=None, confirmed_by_user=False) -> None:
        """
        Save or update a table structure for a contest in the database.
        """
        from ..utils.db_utils import save_table_structure_to_db
        try:
            save_table_structure_to_db(contest, headers, context, ml_confidence, confirmed_by_user)
            logger.info(f"[CONTEXT ORGANIZER] Saved table structure for contest: {contest}")
        except Exception as e:
            logger.error(f"[CONTEXT ORGANIZER] Failed to save table structure: {e}")

    def get_table_structure_from_db(self, contest, context=None) -> dict:
        """
        Retrieve a table structure for a contest from the database.
        """
        from ..utils.db_utils import get_table_structure_from_db
        try:
            result = get_table_structure_from_db(contest, context)
            if result:
                logger.info(f"[CONTEXT ORGANIZER] Loaded table structure for contest: {contest}")
            else:
                logger.warning(f"[CONTEXT ORGANIZER] No table structure found for contest: {contest}")
            return result
        except Exception as e:
            logger.error(f"[CONTEXT ORGANIZER] Failed to load table structure: {e}")
            return None
