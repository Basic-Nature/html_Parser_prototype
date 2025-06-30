"""
context_organizer.py

Advanced context organizer for election HTML parsing and data integrity.
Handles data formatting, ML anomaly detection, cache-aware learning, clustering, and robust DB.
Delegates NLP/semantic logic to the context_coordinator and spacy_utils modules.
"""

from datetime import datetime, timezone
import os
import orjson
from collections import defaultdict
import types
import collections.abc
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from ..utils.db_utils import (
    load_processed_urls,
    load_output_cache,
    normalize_label,
    get_session,
)
from ..utils.model_registry import ModelRegistry
from ..utils.models import Contest, TableStructure
from ..utils.shared_logic import scan_environment
from ..bots.librarian import load_context_library, update_context_library
from .Integrity_check import (
    detect_anomalies_with_ml, print_ml_anomalies, election_integrity_checks
)
from ..utils.shared_logger import log_info, log_warning, log_error
from rich.table import Table
from rich.console import Console
import matplotlib.pyplot as plt
from collections import Counter

console = Console()

from ..config import BASE_DIR, CONTEXT_LIBRARY_PATH, CONTEXT_DB_PATH

PROCESSED_URLS_CACHE = os.path.join(BASE_DIR, ".processed_urls")
OUTPUT_CACHE = os.path.join(BASE_DIR, ".output_cache.jsonl")
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# --- DB Schema Setup (now handled by Alembic migrations) ---
def ensure_db_schema():
    # Schema is managed by Alembic migrations; nothing to do here
    pass

ensure_db_schema()

processed_urls = load_processed_urls()
output_cache = load_output_cache()

def _to_json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_safe(v) for v in obj]
    return obj

def remove_functions(obj):
    if isinstance(obj, dict):
        return {k: remove_functions(v) for k, v in obj.items() if not isinstance(v, types.FunctionType)}
    elif isinstance(obj, list):
        return [remove_functions(v) for v in obj]
    else:
        return obj

def save_table_structure_to_db(contest_title, headers, context, ml_confidence=None, confirmed_by_user=False):
    """
    Upsert a table structure using SQLAlchemy ORM. Updates if contest_title exists, else inserts.
    """
    try:
        with get_session() as session:
            obj = session.execute(
                select(TableStructure).where(TableStructure.contest_title == contest_title)
            ).scalar_one_or_none()
            if obj:
                obj.headers = orjson.dumps(headers)
                obj.context = orjson.dumps(context)
                obj.ml_confidence = ml_confidence
                obj.confirmed_by_user = confirmed_by_user
            else:
                obj = TableStructure(
                    contest_title=contest_title,
                    headers=orjson.dumps(headers),
                    context=orjson.dumps(context),
                    ml_confidence=ml_confidence,
                    confirmed_by_user=confirmed_by_user
                )
                session.add(obj)
            session.commit()
    except SQLAlchemyError as e:
        log_error(f"[DB][TableStructure] Error saving: {e}")
        raise

def get_table_structure_from_db(contest_title, context=None):
    """
    Retrieve the best-matching table structure for a contest_title using SQLAlchemy ORM.
    """
    try:
        with get_session() as session:
            row = session.execute(
                select(TableStructure).where(TableStructure.contest_title == contest_title)
                .order_by(TableStructure.confirmed_by_user.desc(), TableStructure.ml_confidence.desc())
                .limit(1)
            ).scalar_one_or_none()
        if row:
            headers = robust_orjson_loads(row.headers)
            context = robust_orjson_loads(row.context)
            ml_confidence = row.ml_confidence
            return {"headers": headers, "context": context, "ml_confidence": ml_confidence}
        return None
    except SQLAlchemyError as e:
        log_error(f"[DB][TableStructure] Error loading: {e}")
        return None

def upsert_contest(session, contest_dict):
    """
    Upsert a contest using SQLAlchemy ORM. Updates if exists, else inserts.
    """
    obj = session.execute(
        select(Contest).where(
            Contest.title == contest_dict.get("title"),
            Contest.year == contest_dict.get("year"),
            Contest.type == contest_dict.get("type"),
            Contest.state == contest_dict.get("state"),
            Contest.county == contest_dict.get("county")
        )
    ).scalar_one_or_none()
    if obj:
        obj.metadata = orjson.dumps(contest_dict)
    else:
        obj = Contest(
            title=contest_dict.get("title"),
            year=contest_dict.get("year"),
            type=contest_dict.get("type"),
            state=contest_dict.get("state"),
            county=contest_dict.get("county"),
            metadata=orjson.dumps(contest_dict)
        )
        session.add(obj)

def robust_orjson_loads(val):
    """Load JSON robustly from either bytes or str."""
    if isinstance(val, bytes):
        return orjson.loads(val)
    elif isinstance(val, str):
        return orjson.loads(val.encode("utf-8"))
    else:
        raise TypeError(f"Cannot decode type {type(val)} with orjson")

def repair_dom_segments(segments):
    """
    Repairs parent/child relationships in a list of DOM segments.
    Ensures that for every node, all children point back to the parent.
    Returns the repaired segments.
    """
    # Build index map
    idx_map = {seg.get("_idx", i): seg for i, seg in enumerate(segments)}
    # Pass 1: Fix children to be int indices, remove None
    for seg in segments:
        seg["children"] = [c if isinstance(c, int) else getattr(c, "_idx", None) for c in seg.get("children", [])]
        seg["children"] = [c for c in seg["children"] if c is not None]
    # Pass 2: Fix parent_idx to be int or None
    for seg in segments:
        if isinstance(seg.get("parent_idx"), dict):
            seg["parent_idx"] = seg["parent_idx"].get("_idx")
        elif not (isinstance(seg.get("parent_idx"), int) or seg.get("parent_idx") is None):
            seg["parent_idx"] = None
    # Pass 3: Enforce bidirectional consistency
    for seg in segments:
        for child_idx in list(seg["children"]):
            child = idx_map.get(child_idx)
            if child is not None and child.get("parent_idx") != seg["_idx"]:
                # Fix the child's parent_idx
                child["parent_idx"] = seg["_idx"]
    # Pass 4: Remove children that do not point back to parent
    for seg in segments:
        seg["children"] = [c for c in seg["children"] if idx_map.get(c, {}).get("parent_idx") == seg["_idx"]]
    return segments

class ContextOrganizer:
    def __init__(
        self,
        use_library=True,
        enable_ml=True,
        contamination=None,
        n_estimators=100,
        random_state=42,
        embedding_model="all-MiniLM-L6-v2",
        plot_anomalies=True,
        logger=None,
        debug=False,
        fuzzy_cutoff=0.6
    ):
        from ..utils.shared_logger import logger as shared_logger
        self.use_library = use_library
        self.enable_ml = enable_ml
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.embedding_model = embedding_model  # can be string or model object
        self.plot_anomalies = plot_anomalies
        self.logger = logger or shared_logger
        self.db_path = CONTEXT_DB_PATH
        self.context_library_path = CONTEXT_LIBRARY_PATH
        self.library = load_context_library() if use_library else self._default_library()
        self.organized = None
        self.processed_urls = load_processed_urls()
        self.output_cache = load_output_cache()
        self.debug = debug
        self.fuzzy_cutoff = fuzzy_cutoff
        ensure_db_schema()
        # --- Embedding model validation/loading ---
        self.embedding_model_obj = None
        try:
            from ..utils.model_registry import ModelRegistry
            if isinstance(self.embedding_model, str):
                self.embedding_model_obj = ModelRegistry.get_sentence_transformer(self.embedding_model)
                self.log_info(f"[CONTEXT ORGANIZER] Loaded embedding model: {self.embedding_model}")
            elif hasattr(self.embedding_model, "encode"):
                # Looks like a SentenceTransformer or compatible model
                self.embedding_model_obj = self.embedding_model
                self.log_info(f"[CONTEXT ORGANIZER] Using provided embedding model object.")
            else:
                # If it's a method, class, or something else, warn and set to None
                self.log_warning(f"[CONTEXT ORGANIZER] Provided embedding_model is not a recognized model instance or string. Type: {type(self.embedding_model)}. Setting to None.")
                self.embedding_model_obj = None
        except Exception as e:
            self.log_error(f"[CONTEXT ORGANIZER] Failed to load embedding model: {e}")
            self.embedding_model_obj = None

    @staticmethod
    def _default_library():
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
            "panel_tags": [],
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
    def print_contest_summary(contests):
        table = Table(title="Contest Summary by State/County")
        table.add_column("Title")
        table.add_column("State")
        table.add_column("County")
        table.add_column("Year")
        for c in contests:
            table.add_row(
                str(c.get("title", "")),
                str(c.get("state", "")),
                str(c.get("county", "")),
                str(c.get("year", ""))
            )
        console = Console()
        console.print(table)

    @staticmethod
    def plot_contest_distribution(contests):
        state_county = [ (c.get("state", "Unknown"), c.get("county", "Unknown")) for c in contests ]
        counter = Counter(state_county)
        labels, values = zip(*counter.items())
        label_strs = [f"{s}\n{c}" for s,c in labels]
        plt.figure(figsize=(10,5))
        plt.bar(label_strs, values)
        plt.xticks(rotation=90)
        plt.title("Contest Count by State/County")
        plt.tight_layout()
        plt.show()
    @staticmethod
    def suggest_and_apply_fixes(contests, context_library, logs=None, min_confidence=0.8, embedding_model=None):
        """
        Try to fix missing state/county/year/type using context_library, logs, and ML similarity.
        Returns: (fixed_contests, fix_log)
        """
        from difflib import get_close_matches
        import numpy as np

        fix_log = []
        # Build lookup tables from context_library
        title_to_state = {}
        title_to_county = {}
        title_to_year = {}
        title_to_type = {}
        for c in context_library.get("contests", []):
            if not isinstance(c, dict):
                continue
            title = c.get("title") or c.get("label")
            if title:
                key = title.lower()
                if c.get("state"):
                    title_to_state[key] = c["state"]
                if c.get("county"):
                    title_to_county[key] = c["county"]
                if c.get("year"):
                    title_to_year[key] = c["year"]
                if c.get("type"):
                    title_to_type[key] = c["type"]

        # --- ML Embedding Preparation ---
        # Build embeddings for context_library contests with known state/county
        lib_titles = []
        lib_states = []
        lib_counties = []
        lib_years = []
        lib_types = []
        for c in context_library.get("contests", []):
            if not isinstance(c, dict):
                continue
            title = c.get("title") or c.get("label")
            if title and (c.get("state") or c.get("county")):
                lib_titles.append(title)
                lib_states.append(c.get("state"))
                lib_counties.append(c.get("county"))
                lib_years.append(c.get("year"))
                lib_types.append(c.get("type"))
        lib_embeddings = None
        if embedding_model and lib_titles:
            try:
                lib_embeddings = embedding_model.encode(lib_titles)
            except Exception:
                lib_embeddings = None
        min_confidence = 0.85
        # Try to fix each contest
        for idx, c in enumerate(contests):
            fixed = False
            reasons = []
            title = (c.get("title") or "").lower()
            # Fix state
            if not c.get("state"):
                # 1. Try context_library
                if title in title_to_state:
                    c["state"] = title_to_state[title]
                    reasons.append("filled state from context_library")
                    fixed = True
                # 2. Try majority vote from other contests
                elif contests:
                    states = [x.get("state") for x in contests if x.get("state")]
                    if states:
                        most_common = max(set(states), key=states.count)
                        c["state"] = most_common
                        reasons.append("filled state from majority vote")
                        fixed = True
                # 3. Try fuzzy match
                else:
                    matches = get_close_matches(title, list(title_to_state.keys()), n=1, cutoff=0.8)
                    if matches:
                        c["state"] = title_to_state[matches[0]]
                        reasons.append(f"filled state from fuzzy match: {matches[0]}")
                        fixed = True
                # 4. ML similarity
                if not c.get("state") and embedding_model and lib_embeddings is not None:
                    try:
                        query_emb = embedding_model.encode([c.get("title") or ""])[0]
                        sims = np.dot(lib_embeddings, query_emb) / (
                            np.linalg.norm(lib_embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-8
                        )
                        best_idx = int(np.argmax(sims))
                        best_score = sims[best_idx]
                        if best_score > min_confidence and lib_states[best_idx]:
                            c["state"] = lib_states[best_idx]
                            reasons.append(
                                f"filled state from ML similarity: {lib_titles[best_idx]} (sim={best_score:.2f})"
                            )
                            fixed = True
                        else:
                            reasons.append(
                                f"ML similarity for state below threshold ({best_score:.2f} < {min_confidence})"
                            )
                    except Exception as e:
                        reasons.append(f"ML similarity failed: {e}")
            # Fix county
            if not c.get("county"):
                if title in title_to_county:
                    c["county"] = title_to_county[title]
                    reasons.append("filled county from context_library")
                    fixed = True
                elif contests:
                    counties = [x.get("county") for x in contests if x.get("county")]
                    if counties:
                        most_common = max(set(counties), key=counties.count)
                        c["county"] = most_common
                        reasons.append("filled county from majority vote")
                        fixed = True
                else:
                    matches = get_close_matches(title, list(title_to_county.keys()), n=1, cutoff=0.8)
                    if matches:
                        c["county"] = title_to_county[matches[0]]
                        reasons.append(f"filled county from fuzzy match: {matches[0]}")
                        fixed = True
                # ML similarity
                if not c.get("county") and embedding_model and lib_embeddings is not None:
                    try:
                        query_emb = embedding_model.encode([c.get("title") or ""])[0]
                        sims = np.dot(lib_embeddings, query_emb) / (
                            np.linalg.norm(lib_embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-8
                        )
                        best_idx = int(np.argmax(sims))
                        best_score = sims[best_idx]
                        if best_score > min_confidence and lib_counties[best_idx]:
                            c["county"] = lib_counties[best_idx]
                            reasons.append(
                                f"filled county from ML similarity: {lib_titles[best_idx]} (sim={best_score:.2f})"
                            )
                            fixed = True
                        else:
                            reasons.append(
                                f"ML similarity for county below threshold ({best_score:.2f} < {min_confidence})"
                            )
                    except Exception as e:
                        reasons.append(f"ML similarity failed: {e}")
            # Fix year
            if not c.get("year"):
                if title in title_to_year:
                    c["year"] = title_to_year[title]
                    reasons.append("filled year from context_library")
                    fixed = True
                elif contests:
                    years = [x.get("year") for x in contests if x.get("year")]
                    if years:
                        most_common = max(set(years), key=years.count)
                        c["year"] = most_common
                        reasons.append("filled year from majority vote")
                        fixed = True
                else:
                    matches = get_close_matches(title, list(title_to_year.keys()), n=1, cutoff=0.8)
                    if matches:
                        c["year"] = title_to_year[matches[0]]
                        reasons.append(f"filled year from fuzzy match: {matches[0]}")
                        fixed = True
                # ML similarity
                if not c.get("year") and embedding_model and lib_embeddings is not None:
                    try:
                        query_emb = embedding_model.encode([c.get("title") or ""])[0]
                        sims = np.dot(lib_embeddings, query_emb) / (
                            np.linalg.norm(lib_embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-8
                        )
                        best_idx = int(np.argmax(sims))
                        best_score = sims[best_idx]
                        if best_score > min_confidence and lib_years[best_idx]:
                            c["year"] = lib_years[best_idx]
                            reasons.append(
                                f"filled year from ML similarity: {lib_titles[best_idx]} (sim={best_score:.2f})"
                            )
                            fixed = True
                        else:
                            reasons.append(
                                f"ML similarity for state below threshold ({best_score:.2f} < {min_confidence})"
                            )
                    except Exception as e:
                        reasons.append(f"ML similarity failed: {e}")
            # Fix type
            if not c.get("type"):
                if title in title_to_type:
                    c["type"] = title_to_type[title]
                    reasons.append("filled type from context_library")
                    fixed = True
                elif contests:
                    types = [x.get("type") for x in contests if x.get("type")]
                    if types:
                        most_common = max(set(types), key=types.count)
                        c["type"] = most_common
                        reasons.append("filled type from majority vote")
                        fixed = True
                else:
                    matches = get_close_matches(title, list(title_to_type.keys()), n=1, cutoff=0.8)
                    if matches:
                        c["type"] = title_to_type[matches[0]]
                        reasons.append(f"filled type from fuzzy match: {matches[0]}")
                        fixed = True
                # ML similarity
                if not c.get("type") and embedding_model and lib_embeddings is not None:
                    try:
                        query_emb = embedding_model.encode([c.get("title") or ""])[0]
                        sims = np.dot(lib_embeddings, query_emb) / (
                            np.linalg.norm(lib_embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-8
                        )
                        best_idx = int(np.argmax(sims))
                        best_score = sims[best_idx]
                        if best_score > min_confidence and lib_types[best_idx]:
                            c["type"] = lib_types[best_idx]
                            reasons.append(
                                f"filled type from ML similarity: {lib_titles[best_idx]} (sim={best_score:.2f})"
                            )
                            fixed = True
                        else:
                            reasons.append(
                                f"ML similarity for type below threshold ({best_score:.2f} < {min_confidence})"
                            )
                    except Exception as e:
                        reasons.append(f"ML similarity failed: {e}")
            if fixed:
                fix_log.append({"title": c.get("title"), "fixes": reasons})
        return contests, fix_log
    @staticmethod
    def _describe_embedding_model(model):
        """
        Return a human-friendly description of the embedding model.
        Uses ModelRegistry.get_model_name if available, else falls back to class name or str.
        """
        try:
            # If model is a method or not an instance, return its type
            if model is None:
                return "None"
            # Avoid passing a method or class to get_model_name
            if callable(model) and not hasattr(model, "model_name_or_path"):
                return f"{type(model).__name__} (not loaded)"
            # Use ModelRegistry utility if available
            if hasattr(ModelRegistry, "get_model_name"):
                name = ModelRegistry.get_model_name(model)
                if name and isinstance(name, str):
                    return name
            # Common attribute for SentenceTransformer
            if hasattr(model, "model_name_or_path"):
                return str(getattr(model, "model_name_or_path"))
            # Fallback to class name
            if hasattr(model, "__class__"):
                return model.__class__.__name__
            # Fallback to string representation
            return str(model)[:80]
        except Exception as e:
            return f"Unknown model ({e})"
    
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
        fuzzy_cutoff=None
    ):
        """
        Organizes the context for a parsed HTML page, including DOM structure, contests, panels, buttons, tables, and ML features.
        Now includes dynamic state/county detection, verbose logging, and returns a detailed result object.
        Enhanced: robust keyword-based grouping, use_library/cache integration, and diagnostics.
        """
        from ..bots.librarian import (
            LOCATION_KEYWORDS, CANDIDATE_KEYWORDS, PARTY_KEYWORDS, BALLOT_TYPES, CONTEST_KEYWORDS, PERCENT_KEYWORDS, TOTAL_KEYWORDS, MISC_FOOTER_KEYWORDS
        )
        debug = self.debug if debug is None else debug
        fuzzy_cutoff = self.fuzzy_cutoff if fuzzy_cutoff is None else fuzzy_cutoff
        # --- Use class-level embedding_model and plot_anomalies unless overridden ---
        embedding_model = embedding_model if embedding_model is not None else self.embedding_model_obj
        plot_anomalies = plot_anomalies if plot_anomalies is not None else self.plot_anomalies
        self.log_info(
            "\n[CONTEXT ORGANIZER] Pipeline configuration:\n"
            f"  • Embedding model: { self._describe_embedding_model(embedding_model) }\n"
            f"  • Plot anomalies:  { plot_anomalies }\n"
        )
        log = []
        summary = {"attempts": [], "final": None, "error": None}

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
                self._context_cache = cache.copy()
            log.append(f"[CACHE] Using provided cache with {len(cache)} entries.")

        if "panels" in raw_context and isinstance(raw_context["panels"], list):
            raw_context["panels"] = {}

        tagged_segments = raw_context.get("tagged_segments_with_attrs", [])
        url_value = raw_context.get("url", "")

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
            seg["_idx"] = seg.get("_idx", 0) + 1
            if seg.get("parent_idx") is None:
                seg["parent_idx"] = 0
            elif isinstance(seg["parent_idx"], int):
                seg["parent_idx"] += 1
            seg["children"] = [c + 1 if isinstance(c, int) else c for c in seg.get("children", [])]
        tagged_segments = [virtual_root] + tagged_segments
        tagged_segments = repair_dom_segments(tagged_segments)
        dom_tree = self.build_dom_tree(tagged_segments)
        dom_tree["source_url"] = url_value
        dom_parts = self.expose_dom_parts(dom_tree)

        # --- Robust contest organization using all available keywords ---
        contests = []
        contest_titles = set()
        for c in raw_context.get("contests", []):
            title = c.get("title") or c.get("label") or c
            norm_title = normalize_label(title)
            if norm_title not in contest_titles:
                contest_titles.add(norm_title)
                contests.append({
                    "title": title,
                    "year": c.get("year"),
                    "type": c.get("type"),
                    "state": raw_context.get("state"),
                    "county": raw_context.get("county"),
                    "raw": c
                })
        for c in context_library.get("contests", []):
            if not isinstance(c, dict):
                continue
            norm_title = normalize_label(c.get("title", c.get("label", str(c))))
            if norm_title not in contest_titles:
                contests.append(c)
                contest_titles.add(norm_title)
        if debug:
            self.print_contest_summary(contests)
            self.plot_contest_distribution(contests)
        # --- Keyword-based grouping for contests ---
        keyword_sets = {
            "location": LOCATION_KEYWORDS,
            "candidate": CANDIDATE_KEYWORDS,
            "party": PARTY_KEYWORDS,
            "ballot_type": set(BALLOT_TYPES),
            "contest": CONTEST_KEYWORDS,
            "percent": PERCENT_KEYWORDS,
            "total": TOTAL_KEYWORDS,
            "footer": MISC_FOOTER_KEYWORDS
        }
        contest_groups = {k: [] for k in keyword_sets}
        for c in contests:
            title = c.get("title", "").lower()
            for group, keywords in keyword_sets.items():
                if any(kw in title for kw in keywords):
                    contest_groups[group].append(c)
        log.append(f"[KEYWORDS] Contest groups: {{k: len(v) for k,v in contest_groups.items()}}")

        # --- Robust panel/table/button grouping using keywords and library ---
        def group_by_keywords(items, label_field="label"):
            groups = {k: [] for k in keyword_sets}
            for item in items:
                if not isinstance(item, dict):
                    continue
                label = item.get(label_field, "").lower()
                for group, keywords in keyword_sets.items():
                    if any(kw in label for kw in keywords):
                        groups[group].append(item)
            return groups
        # Panels
        panels = {}
        for c in contests:
            panel = None
            if panel_features:
                panel = next((p for p in panel_features if normalize_label(p.get("label", "")) == normalize_label(c["title"])), None)
            if not panel:
                panel = raw_context.get("panels", {}).get(c["title"])
            if not panel and "panels" in context_library:
                # Only consider dicts, skip strings or other types
                panel = next(
                    (p for p in context_library["panels"]
                    if isinstance(p, dict) and normalize_label(p.get("label", "")) == normalize_label(c["title"])),
                    None
                )
            panels[c["title"]] = panel
        panel_groups = group_by_keywords([p for p in panels.values() if p], label_field="label")
        log.append(f"[KEYWORDS] Panel groups: {{k: len(v) for k,v in panel_groups.items()}}")
        # Buttons
        buttons_by_contest = defaultdict(list)
        if not isinstance(raw_context, dict):
            raw_context = {"buttons": raw_context}
        raw_buttons = button_features or raw_context.get("buttons", [])
        if not isinstance(raw_buttons, list):
            raw_buttons = []
        if not isinstance(context_library, dict):
            context_library = {"buttons": []}
        lib_buttons = context_library.get("buttons", [])
        if not isinstance(lib_buttons, list):
            lib_buttons = []
        all_buttons = raw_buttons + lib_buttons
        unmatched_buttons = []
        for btn in all_buttons:
            if not isinstance(btn, dict):
                continue
            matched = False
            for c in contests:
                if c["title"].lower() in btn.get("label", "").lower():
                    buttons_by_contest[c["title"]].append(btn)
                    matched = True
                elif "election" in btn.get("label", "").lower() and "election" in c["title"].lower():
                    buttons_by_contest[c["title"]].append(btn)
                    matched = True
            if not matched:
                unmatched_buttons.append(btn)
        for btn in unmatched_buttons:
            buttons_by_contest["__unmatched__"].append(btn)
        button_groups = group_by_keywords(all_buttons, label_field="label")
        log.append(f"[KEYWORDS] Button groups: {{k: len(v) for k,v in button_groups.items()}}")
        # Tables
        tables_by_contest = defaultdict(list)
        
        raw_tables = raw_context.get("tables", [])
        if not isinstance(raw_tables, list):
            raw_tables = []
        lib_tables = context_library.get("tables", [])
        if not isinstance(lib_tables, list):
            lib_tables = []
        all_tables = raw_tables + lib_tables
        for tbl in all_tables:
            if not isinstance(tbl, dict):
                continue
            for c in contests:
                if c["title"].lower() in tbl.get("label", "").lower():
                    tables_by_contest[c["title"]].append(tbl)
            if not any(tbl in v for v in tables_by_contest.values()):
                tables_by_contest["__unmatched__"].append(tbl)
        table_groups = group_by_keywords(all_tables, label_field="label")
        log.append(f"[KEYWORDS] Table groups: {{k: len(v) for k,v in table_groups.items()}}")

        metadata = {
            "state": raw_context.get("state"),
            "county": raw_context.get("county"),
            "source_url": raw_context.get("url"),
            "election_type": raw_context.get("election_type"),
            "scrape_time": raw_context.get("scrape_time"),
            "year": None,
            "race": raw_context.get("race"),
            "environment": scan_environment(),
        }

        anomalies, clusters = [], []
        if enable_ml and len(contests) > 0:
            try:
                anomalies, clusters = detect_anomalies_with_ml(
                    contests,
                    contamination=contamination,
                    n_estimators=n_estimators,
                    random_state=random_state,
                    embedding_model=embedding_model
                )
                if anomalies:
                    for idx in anomalies:
                        contest = contests[idx]
                        title = contest.get('title', str(contest))
                        log_info(f"[bold magenta][ML][/bold magenta] Context anomaly detected: [bold yellow]{title}[/bold yellow]\n  [dim]Context:[/dim] {contest}")
                if plot_clusters_flag:
                    plot_clusters_flag =print_ml_anomalies(anomalies, contests)
            except Exception as e:
                log_error(f"[bold red][ML] Anomaly detection failed:[/bold red] {e}")

        integrity_issues = election_integrity_checks(contests)
        contests, fix_log = self.suggest_and_apply_fixes(
            contests,
            context_library,
            logs=log,
            min_confidence=0.8,
            embedding_model=embedding_model if embedding_model is not None else self.embedding_model_obj
        )
        if fix_log:
            log_info("[bold green]Auto-fixes applied:[/bold green]")
            for entry in fix_log:
                log_warning(f"  [yellow]{entry['title']}[/yellow]: {', '.join(entry['fixes'])}")
        # Optionally, re-run integrity checks to see if issues remain
        integrity_issues = election_integrity_checks(contests)
        for issue, contest in integrity_issues:
            if issue == "duplicate":
                log_warning(f"[bold yellow][INTEGRITY][/bold yellow] Duplicate contest detected.\n  [dim]Context:[/dim] {contest}")
            elif issue == "missing_location":
                log_warning(f"[bold yellow][INTEGRITY][/bold yellow] Contest missing location info.\n  [dim]Context:[/dim] {contest}")
            elif issue == "missing_year":
                log_warning(f"[bold yellow][INTEGRITY][/bold yellow] Contest missing year.\n  [dim]Context:[/dim] {contest}")

        if len(contests) > 50:
            log_error(f"[bold red][CONTEXT ORGANIZER][/bold red] High contest count detected — possible congestion.\n  [dim]Context:[/dim] contest_count={len(contests)}")

        # --- Advanced relationship extraction: party/candidate/district/state/county mappings ---
        party_to_candidates = defaultdict(set)
        candidate_to_party = defaultdict(set)
        candidate_to_district = defaultdict(set)
        district_to_candidates = defaultdict(set)
        state_to_counties = defaultdict(set)
        county_to_state = dict()
        # Use contests, panels, tables, and DOM segments for mapping
        for c in contests:
            party = c.get("party") or c.get("party_label") or c.get("affiliation")
            candidate = c.get("candidate") or c.get("candidates")
            district = c.get("district") or c.get("district_name")
            state = c.get("state")
            county = c.get("county")
            # Handle lists and strings
            if isinstance(candidate, str):
                candidate = [candidate]
            if isinstance(party, str):
                party = [party]
            if isinstance(district, str):
                district = [district]
            # Party <-> Candidate
            if party and candidate:
                for p in party:
                    for cand in candidate:
                        if p and cand:
                            party_to_candidates[p.strip()].add(cand.strip())
                            candidate_to_party[cand.strip()].add(p.strip())
            # Candidate <-> District
            if candidate and district:
                for cand in candidate:
                    for d in district:
                        if cand and d:
                            candidate_to_district[cand.strip()].add(d.strip())
                            district_to_candidates[d.strip()].add(cand.strip())
            # State <-> County
            if state and county:
                state_to_counties[state.strip()].add(county.strip())
                county_to_state[county.strip()] = state.strip()
        # Also scan panels/tables for party/candidate/district/state/county
        for group in [panels.values(), tables_by_contest.values()]:
            for items in group:
                if items is None:
                    continue
                if isinstance(items, dict):
                    items = [items]
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    label = (item.get("label") or "").lower()
                    for p in PARTY_KEYWORDS:
                        if p in label:
                            party_to_candidates[p].add(label)
                    for cand in CANDIDATE_KEYWORDS:
                        if cand in label:
                            candidate_to_party[label].add(cand)
                    for d in CONTEST_KEYWORDS:
                        if d in label:
                            candidate_to_district[label].add(d)
        # Convert sets to sorted lists for output
        party_to_candidates = {k: sorted(v) for k, v in party_to_candidates.items()}
        candidate_to_party = {k: sorted(v) for k, v in candidate_to_party.items()}
        candidate_to_district = {k: sorted(v) for k, v in candidate_to_district.items()}
        district_to_candidates = {k: sorted(v) for k, v in district_to_candidates.items()}
        state_to_counties = {k: sorted(v) for k, v in state_to_counties.items()}
        # Log mappings for diagnostics
        log.append(f"[RELATIONSHIPS] party_to_candidates: { {k: len(v) for k,v in party_to_candidates.items()} }")
        log.append(f"[RELATIONSHIPS] candidate_to_party: { {k: len(v) for k,v in candidate_to_party.items()} }")
        log.append(f"[RELATIONSHIPS] candidate_to_district: { {k: len(v) for k,v in candidate_to_district.items()} }")
        log.append(f"[RELATIONSHIPS] district_to_candidates: { {k: len(v) for k,v in district_to_candidates.items()} }")
        log.append(f"[RELATIONSHIPS] state_to_counties: { {k: len(v) for k,v in state_to_counties.items()} }")
        log.append(f"[RELATIONSHIPS] county_to_state: {county_to_state}")

        organized = {
            "contests": contests,
            "contest_groups": contest_groups,
            "panel_groups": panel_groups,
            "button_groups": button_groups,
            "table_groups": table_groups,
            "buttons": dict(buttons_by_contest),
            "panels": panels,
            "tables": dict(tables_by_contest),
            "metadata": metadata,
            "anomalies": [contests[i] for i in anomalies] if anomalies else [],
            "clusters": clusters.tolist() if hasattr(clusters, "tolist") else clusters,
            "integrity_issues": integrity_issues,
            "dom_tree": dom_tree,
            "dom_parts": dom_parts,
            # --- Advanced mappings for downstream use ---
            "party_to_candidates": party_to_candidates,
            "candidate_to_party": candidate_to_party,
            "candidate_to_district": candidate_to_district,
            "district_to_candidates": district_to_candidates,
            "state_to_counties": state_to_counties,
            "county_to_state": county_to_state,
        }
        valid_years = [
            c.get("year")
            for c in contests
            if c.get("year") and c.get("type") and str(c.get("year")).isdigit()
        ]
        if valid_years:
            metadata["year"] = valid_years[0]
        else:
            metadata["year"] = "Unknown"
        self.append_to_context_library(organized, path=self.context_library_path)
        self.log_info(
            f"[CONTEXT ORGANIZER] Organized context for {len(contests)} contests. "
            f"Anomalies: {len(anomalies)}  Integrity issues: {len(integrity_issues)}"
        )
        log_info(
            f"[bold green][CONTEXT ORGANIZER][/bold green] Organized context for [bold]{len(contests)}[/bold] contests.\n"
            f"  [magenta]Anomalies:[/magenta] {len(anomalies)}  [yellow]Integrity issues:[/yellow] {len(integrity_issues)}"
        )

        # Insert or update contests robustly using SQLAlchemy ORM
        try:
            with get_session() as session:
                for c in contests:
                    upsert_contest(session, c)
                session.commit()
        except SQLAlchemyError as e:
            self.log_error(f"[DB][Contest] Error upserting contests: {e}")

        missing_location = any(
            not c.get("state") or not c.get("county")
            for c in contests
        )

        if missing_location:
            from .context_coordinator import dynamic_state_county_detection
            html = raw_context.get("raw_html", "")
            county, state, handler_path, detection_log = dynamic_state_county_detection(
                raw_context, html, debug=True
            )
            for log_entry in detection_log:
                log.append(f"[Dynamic Detection] {log_entry}")
                if debug:
                    self.log_info(f"[ContextOrganizer][Dynamic Detection] {log_entry}")
            if state:
                raw_context["state"] = state
            if county:
                raw_context["county"] = county
            summary["final"] = {"state": state, "county": county, "handler_path": handler_path}
            log.append(f"Final detected state: {state}, county: {county}, handler_path: {handler_path}")

        result = {
            "organized": organized,
            "summary": summary,
            "log": log,
            "error": None,
        }
        self.organized = organized
        return result

    def build_dom_tree(self, segments):
        """
        Build a DOM tree from a list of segments that already include parent/child relationships and indices.
        Each node contains: tag, attrs, classes, id, html, children, parent_idx, start, end, _idx.
        Returns the root nodes (usually head/body/url_root) and a flat list with parent/child relationships.
        Ensures parent/child consistency and logs any inconsistencies.
        """
        nodes = []
        idx_map = {}
        for i, seg in enumerate(segments):
            node = dict(seg)
            node.setdefault("children", [])
            node.setdefault("parent_idx", None)
            node.setdefault("_idx", i)
            node.setdefault("start", None)
            node.setdefault("end", None)
            nodes.append(node)
            idx_map[node["_idx"]] = node

        # Fix up children and parent_idx in case they're not integers
        for node in nodes:
            node["children"] = [c if isinstance(c, int) else getattr(c, "_idx", None) for c in node.get("children", [])]
            node["children"] = [c for c in node["children"] if c is not None]
            if isinstance(node.get("parent_idx"), dict):
                node["parent_idx"] = node["parent_idx"].get("_idx")
            elif not (isinstance(node.get("parent_idx"), int) or node.get("parent_idx") is None):
                node["parent_idx"] = None

        # Parent/child consistency check
        for node in nodes:
            for child_idx in node["children"]:
                if child_idx >= len(nodes) or nodes[child_idx].get("parent_idx") != node["_idx"]:
                    self.log_warning(f"Inconsistent parent/child: node {node['_idx']} child {child_idx}")

        roots = [node for node in nodes if node["parent_idx"] is None]
        dom_tree = {
            "roots": [node["_idx"] for node in roots],
            "nodes": nodes
        }
        return dom_tree

    def get_panels_and_tables(self, dom_tree):
        """
        Returns a list of (panel_heading, [tables]) for each panel/section in the DOM.
        """
        panels = []
        nodes = dom_tree["nodes"]
        for node in nodes:
            if node["tag"] in ("section", "div", "fieldset", "panel"):
                # Find heading among children
                heading = None
                for child_idx in node["children"]:
                    child = nodes[child_idx]
                    if child["tag"] in ("h1", "h2", "h3", "h4", "h5", "h6"):
                        heading = child["html"]
                        break
                # Find tables among children
                tables = [nodes[cidx] for cidx in node["children"] if nodes[cidx]["tag"] == "table"]
                if heading and tables:
                    panels.append((heading, tables))
        return panels

    def extract_html_by_idx(self, nodes, idx, full_html):
        """
        Extract the exact HTML for a node using its start/end indices.
        """
        node = nodes[idx]
        if node.get("start") is not None and node.get("end") is not None:
            return full_html[node["start"]:node["end"]]
        return node.get("html", "")

    def extract_subtree_html(self, nodes, idx, full_html):
        """
        Recursively extract the HTML for a node and all its descendants.
        """
        node = nodes[idx]
        if not node["children"]:
            return self.extract_html_by_idx(nodes, idx, full_html)
        # Get the minimal start and maximal end among all descendants
        indices = [idx]
        stack = list(node["children"])
        while stack:
            child_idx = stack.pop()
            indices.append(child_idx)
            stack.extend(nodes[child_idx]["children"])
        starts = [nodes[i]["start"] for i in indices if nodes[i].get("start") is not None]
        ends = [nodes[i]["end"] for i in indices if nodes[i].get("end") is not None]
        if starts and ends:
            return full_html[min(starts):max(ends)]
        return node.get("html", "")

    def group_nodes_by_label(self, nodes, label_field="ml_label"):
        """
        Group nodes by a given label field (e.g., ml_label or semantic_tags).
        Returns a dict: label -> list of nodes.
        """
        from collections import defaultdict
        groups = defaultdict(list)
        for node in nodes:
            label = node.get(label_field)
            if isinstance(label, list):
                for l in label:
                    groups[l].append(node)
            elif label:
                groups[label].append(node)
        return dict(groups)

    def expose_dom_parts(self, dom_tree):
        """
        Expose organized DOM parts for downstream use.
        Returns dicts for head, body, wrappers, tables, buttons, etc.
        Adds 'direct_parent' field for each node.
        """
        nodes = dom_tree["nodes"]
        head_nodes = [n for n in nodes if n["tag"].lower() == "head"]
        body_nodes = [n for n in nodes if n["tag"].lower() == "body"]
        wrappers = [n for n in nodes if n["tag"].lower() in ("div", "section", "form")]
        tables = [n for n in nodes if n["tag"].lower() == "table"]
        buttons = [n for n in nodes if n.get("is_button")]
        clickable = [n for n in nodes if n.get("is_clickable")]
        for n in nodes:
            n["direct_parent"] = nodes[n["parent_idx"]]["tag"] if n["parent_idx"] is not None else None
        return {
            "head_nodes": head_nodes,
            "body_nodes": body_nodes,
            "wrappers": wrappers,
            "tables": tables,
            "buttons": buttons,
            "clickable": clickable,
            "all_nodes": nodes,
            "roots": dom_tree["roots"]
        }

    def append_to_context_library(self, organized, path=None, merge_lists=True, deduplicate=True):
        """
        Robustly append or update the organized context into the context library JSON file.
        - merge_lists: If True, lists are merged (with deduplication if deduplicate=True).
        - deduplicate: If True, removes duplicates from merged lists based on dict content or value.
        """

        def merge_dicts(a, b):
            """Recursively merge dict b into dict a."""
            for k, v in b.items():
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
                                    key = orjson.dumps(item) if isinstance(item, collections.abc.Hashable) else str(item)
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
            library = merge_dicts(library, remove_functions(organized))
            library["last_updated"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            update_context_library(path, lambda lib: lib.update(_to_json_safe(library)))
            self.log_info(f"[CONTEXT ORGANIZER] Appended/merged context to library at {path}")
        except Exception as e:
            self.log_error(f"[CONTEXT ORGANIZER] Failed to append to context library: {e}")

    def save_table_structure_to_db(self, contest_title, headers, context, ml_confidence=None, confirmed_by_user=False):
        """
        Save or update a table structure for a contest in the database.
        """
        try:
            save_table_structure_to_db(contest_title, headers, context, ml_confidence, confirmed_by_user)
            self.log_info(f"[CONTEXT ORGANIZER] Saved table structure for contest: {contest_title}")
        except Exception as e:
            self.log_error(f"[CONTEXT ORGANIZER] Failed to save table structure: {e}")

    def get_table_structure_from_db(self, contest_title, context=None):
        """
        Retrieve a table structure for a contest from the database.
        """
        try:
            result = get_table_structure_from_db(contest_title, context)
            if result:
                self.log_info(f"[CONTEXT ORGANIZER] Loaded table structure for contest: {contest_title}")
            else:
                self.log_warning(f"[CONTEXT ORGANIZER] No table structure found for contest: {contest_title}")
            return result
        except Exception as e:
            self.log_error(f"[CONTEXT ORGANIZER] Failed to load table structure: {e}")
            return None

# --- Backward-compatible function for legacy imports ---
def organize_context(*args, **kwargs):
    return ContextOrganizer().organize_context(*args, **kwargs)
