import json, os, re
from typing import Set, List
from ..config import CONTEXT_LIBRARY_PATH, PROJECT_ROOT, LOG_DIR, BASE_DIR
import orjson
import subprocess
import sys
import time
from tempfile import NamedTemporaryFile
import shutil
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import time
import threading
import shutil
import tempfile
from ..utils.shared_logger import logger

_CONTEXT_LOCK = threading.Lock()
SCHEMA_VERSION = "1.0"

DEFAULT_STRUCTURE = {
    "schema_version": SCHEMA_VERSION,
    "contests": [],
    "panels": [],
    "tables": [],
    "buttons": [],
    "metadata": {},
}
_context_library_cache = None

# --- Central Dynamic Sets (used everywhere) ---
HTML_TAGS: Set[str] = set([
    "html", "head", "title", "body", "h1", "h2", "h3", "h4", "h5", "h6",
    "b", "i", "center", "ul", "li", "br", "p", "hr", "img", "a", "span", "div", "button", "input", "form", "table"
])
PANEL_TAGS: Set[str] = set([
    "section", "fieldset", "panel", "div", "p-panel", "app-ballot-item-wrapper", "article"
])
HEADING_TAGS: Set[str] = set([
    "h1", "h2", "h3", "h4", "h5", "h6", "span", "b", "strong"
])
CUSTOM_ATTR_PATTERNS: List[re.Pattern] = [
    re.compile(r"^data-"),
    re.compile(r"^aria-"),
    re.compile(r"^role$"),
]

DISTRICT_REGEX = re.compile(
    r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)*\s*\d{1,3}|District\s*\d{1,3}|Ward\s*\d{1,3}|Precinct\s*\d{1,3}|ED\s*\d{1,3})\b"
)

# --- Table/Entity Keywords (from table_core, dynamic_table_extractor, etc.) ---
BALLOT_TYPES = [
    "Election Day", "Early Voting", "Absentee", "Mail", "Provisional", "Affidavit", "Other", "Void"
]
BALLOT_TYPE_SORT_ORDER = [
    "Election Day", "Early Voting", "Absentee", "Mail", "Absentee Mail"
]
LOCATION_KEYWORDS = {
    "precinct", "ward", "district", "location", "area", "city", "municipal", "town",
    "borough", "village", "county", "division", "subdistrict", "polling place", "ed", "municipality",
    "section", "region", "zone", "subdivision", "community", "neighborhood", "block", "site",
    "station", "place", "locale", "sector", "unit", "assembly district", "senate district",
    "school district", "congressional district", "judicial district", "supervisorial district",
    "council district", "precinct number", "precinct name", "district number", "district name",
    "polling location", "poll site", "polling station", "precinct id", "district id"
}
PERCENT_KEYWORDS = {
    "% precincts reporting", "% reported", "percent reported", "fully reported", "precincts reporting"
}
TOTAL_KEYWORDS = {"total", "sum", "votes", "overall", "all", "Percent Reported", "Reporting Status"}
MISC_FOOTER_KEYWORDS = {"undervote", "overvote", "scattering", "write-in", "blank", "void", "spoiled"}
CANDIDATE_KEYWORDS = {
    "candidate", "candidates", "name", "nominee", "person", "individual", "contestant",
    "office", "incumbent", "challenger", "write-in", "write in", "writein", "option", "choice",
    "party", "affiliation", "designation", "slate", "ticket", "representative", "member", "appointee"
}
PARTY_KEYWORDS = {
    "democratic", "republican", "working families", "conservative", "green", "libertarian",
    "independent", "write-in", "write in", "writein", "other", "constitution", "socialist",
    "progressive", "labor", "peace and freedom", "american independent", "no party", "nonpartisan",
    "unaffiliated", "unknown", "blank", "void", "spoiled", "scattering", "undeclared", "unaffiliated",
    "party", "affiliation", "designation"
}
LOCATION_ABBREVIATIONS = {
    "ed", "ward", "wd", "dist", "district", "pct", "prec", "precinct", "muni", "mun", "area", "city", "cty",
    "munic", "borough", "boro", "vill", "vlg", "village", "cnty", "county", "div", "division", "subdist", "subdistrict",
    "pollpl", "poll pl", "polling place", "pl", "section", "sec", "region", "reg", "zone", "zn", "subdivision", "sd",
    "comm", "community", "neigh", "neighborhood", "blk", "block", "site", "station", "stn", "locale", "sector", "unit",
    "ad", "assembly district", "sd", "senate district", "cd", "congressional district", "jd", "judicial district",
    "sup dist", "supervisorial district", "council dist", "council district", "precinct no", "precinct num", "precinct number",
    "precinct name", "district no", "district num", "district number", "district name", "poll loc", "poll location",
    "poll site", "polling station", "precinct id", "district id"
}
VALID_TYPES = {"general", "primary", "presidential preference", "special", "runoff", "municipal", "local"}
CONTEST_KEYWORDS = {
        "president", "senate", "congress", "governor", "mayor", "school board", "proposition", "referendum",
        "assembly", "council", "trustee", "justice", "clerk", "judge", "district", "proposal", "village", "town"
    }
ALWAYS_IGNORE_TAGS = {
        "script", "style", "svg", "path", "defs", "g", "canvas", "noscript", "meta", "link", "base", "title"
    }
ALWAYS_IGNORE_CLASSES = {
        "visually-hidden", "sr-only", "skip-link", "screen-reader", "aria-hidden", "d-none", "hidden", "offscreen"
    }
ALWAYS_IGNORE_IDS = {
        "skip-link", "hidden", "aria-hidden"
    }
ROOT_CONTAINER_TAGS = {"body", "html", "app-root"}

ICON_CLASSES = {
        "pi", "bi", "fa", "fas", "far", "fal", "fad", "fab", "glyphicon", "icon", "material-icons",
        "mdi", "octicon", "feather", "ion", "ionicon", "anticon", "euiicon", "p-button-icon", "p-icon",
        "fa-solid", "fa-regular", "fa-light", "fa-duotone", "fa-brands", "fa-stack", "fa-stack-1x", "fa-stack-2x",
        "fa-fw", "fa-li", "fa-border", "fa-spin", "fa-pulse", "fa-inverse", "fa-layers", "fa-layers-text", "fa-layers-counter",
        "oi", "eva", "eva-icon", "remixicon", "ri", "icofont", "icn", "flaticon", "glyph", "iconify", "iconfont",
        "uicon", "uik", "uik-icon", "uik-button-icon", "octicon", "octicon-alert", "octicon-info", "octicon-check",
        "octicon-x", "octicon-star", "octicon-stop", "octicon-download", "octicon-upload", "octicon-arrow", "octicon-chevron",
        "octicon-dot", "octicon-dot-fill", "octicon-dot-outline", "octicon-dot-circle", "octicon-dot-square",
        "icon-label", "icon-btn", "icon-button", "icon-container", "icon-wrapper", "icon-box", "icon-bg", "icon-bg-light",
        "icon-bg-dark", "icon-bg-primary", "icon-bg-secondary", "icon-bg-success", "icon-bg-danger", "icon-bg-warning",
        "icon-bg-info", "icon-bg-white", "icon-bg-black", "icon-bg-gray", "icon-bg-grey", "icon-bg-transparent",
        "icon-bg-gradient", "icon-bg-image", "icon-bg-pattern", "icon-bg-shape", "icon-bg-circle", "icon-bg-square",
        "icon-bg-rectangle", "icon-bg-oval", "icon-bg-round", "icon-bg-pill", "icon-bg-dot", "icon-bg-line",
        "icon-bg-arrow", "icon-bg-chevron", "icon-bg-star", "icon-bg-heart", "icon-bg-check", "icon-bg-x", "icon-bg-plus",
        "icon-bg-minus", "icon-bg-close", "icon-bg-open", "icon-bg-expand", "icon-bg-collapse", "icon-bg-menu", "icon-bg-more",
        "icon-bg-less", "icon-bg-up", "icon-bg-down", "icon-bg-left", "icon-bg-right", "icon-bg-top", "icon-bg-bottom",
        "icon-bg-center", "icon-bg-middle", "icon-bg-end", "icon-bg-start", "icon-bg-first", "icon-bg-last", "icon-bg-prev",
        "icon-bg-next"
    }
ICON_TAGS = {"i", "svg", "path", "g", "span"}

# --- Canonical Segment Labeling & Normalization ---
CANONICAL_SEGMENT_LABELS = {
    # Add common canonical mappings here
    "election results": "results_table",
    "results by precinct": "location_panel",
    "summary": "summary",
    "total votes": "total_votes",
    "precincts reporting": "reporting_status",
    "candidate": "candidate_panel",
    "ballot type": "ballot_type",
    "download": "download_link",
    # Add more as needed
}

BUTTON_CLASSES = {"btn", "button", "toggle", "switch", "p-button", "mat-button", "v-btn", "ant-btn", "el-button"}

HEADING_CLASSES = {"heading", "header", "title", "h1", "h2", "h3", "h4", "h5", "h6", "section-title", "panel-title"}

PANEL_CLASSES = {"panel", "card", "container", "box", "section-panel", "mat-card", "el-card", "ant-card", "v-card"}

TIMESTAMP_CLASSES = {
        "time-ago", "timestamp", "last-updated", "results-timestamp", "update-time", "posted", "modified", "date", "datetime"
    }
TIMESTAMP_ID_PATTERNS = [
        r"timestamp", r"time[-_]?ago", r"last[-_]?updated", r"update[-_]?time", r"posted", r"modified", r"date", r"datetime"
    ]
TIMESTAMP_ATTRS = [
        "timeago", "datetime", "data-timestamp", "data-updated", "data-date", "data-time", "data-last-updated"
    ]

STRUCTURAL_TAGS = {"br", "hr", "wbr", "col", "colgroup", "thead", "tbody", "tfoot", "tr", "th", "td"}

def atomic_write_json(obj, path):
    path = Path(path)
    backup_path = path.with_suffix(path.suffix + ".bak")
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with NamedTemporaryFile("wb", delete=False, dir=path.parent) as tf:
        tf.write(orjson.dumps(obj, option=orjson.OPT_INDENT_2))
        temp_name = tf.name
    if path.exists():
        shutil.copy2(path, backup_path)
    shutil.move(temp_name, path)

# --- Extend/Modify Functions ---
def extend_panel_tags(new_tags: List[str]):
    global PANEL_TAGS
    PANEL_TAGS |= set(t.lower() for t in new_tags)

def extend_heading_tags(new_tags: List[str]):
    global HEADING_TAGS
    HEADING_TAGS |= set(t.lower() for t in new_tags)

def extend_html_tags(new_tags: List[str]):
    global HTML_TAGS
    HTML_TAGS |= set(t.lower() for t in new_tags)

def extend_custom_attr_patterns(new_patterns: List[str]):
    global CUSTOM_ATTR_PATTERNS
    for pat in new_patterns:
        if isinstance(pat, str):
            CUSTOM_ATTR_PATTERNS.append(re.compile(pat))
        else:
            CUSTOM_ATTR_PATTERNS.append(pat)

def extend_location_keywords(new_keywords: List[str]):
    global LOCATION_KEYWORDS
    LOCATION_KEYWORDS |= set(k.lower() for k in new_keywords)

def extend_candidate_keywords(new_keywords: List[str]):
    global CANDIDATE_KEYWORDS
    CANDIDATE_KEYWORDS |= set(k.lower() for k in new_keywords)

def extend_ballot_types(new_types: List[str]):
    global BALLOT_TYPES
    BALLOT_TYPES.extend([t for t in new_types if t not in BALLOT_TYPES])

def safe_join(base, *paths):
    final_path = os.path.abspath(os.path.join(base, *paths))
    if not final_path.startswith(os.path.abspath(base)):
        print(f"DEBUG: Attempted to join {paths} to base {base} -> {final_path}")
        raise ValueError("Attempted Path Traversal Detected!")
    return final_path

# --- Context Library Integration ---
def robust_orjson_loads(val):
    if isinstance(val, bytes):
        return orjson.loads(val)
    elif isinstance(val, str):
        return orjson.loads(val.encode("utf-8"))
    else:
        raise TypeError(f"Cannot decode type {type(val)} with orjson")

def load_context_library(path=CONTEXT_LIBRARY_PATH):
    """
    Loads the context library robustly:
    - If missing, creates with default structure.
    - If empty or corrupt, backs up and re-initializes.
    - If missing keys, adds them (preserving existing data).
    - Extends dynamic sets with loaded values.
    """
    safe_path = path
    os.makedirs(os.path.dirname(safe_path), exist_ok=True)

    def merge_defaults(existing, defaults):
        changed = False
        for k, v in defaults.items():
            if k not in existing:
                existing[k] = v
                changed = True
            elif isinstance(v, dict) and isinstance(existing[k], dict):
                if merge_defaults(existing[k], v):
                    changed = True
        return changed

    # If file does not exist or is empty, create with defaults
    if not os.path.exists(safe_path) or os.path.getsize(safe_path) == 0:
        context_lib = {
            "panel_tags": list(PANEL_TAGS),
            "heading_tags": list(HEADING_TAGS),
            "custom_attr_patterns": [pat.pattern for pat in CUSTOM_ATTR_PATTERNS],
            "location_keywords": list(LOCATION_KEYWORDS),
            "candidate_keywords": list(CANDIDATE_KEYWORDS),
            "ballot_types": list(BALLOT_TYPES),
            **DEFAULT_STRUCTURE
        }
        with open(safe_path, "wb") as f:
            f.write(orjson.dumps(context_lib, option=orjson.OPT_INDENT_2))
        return context_lib

    # Try to load, back up and re-init if corrupt
    try:
        with open(safe_path, "rb") as f:
            data = f.read()
            if not data:
                # Empty file, treat as missing
                context_lib = {
                    "panel_tags": list(PANEL_TAGS),
                    "heading_tags": list(HEADING_TAGS),
                    "custom_attr_patterns": [pat.pattern for pat in CUSTOM_ATTR_PATTERNS],
                    "location_keywords": list(LOCATION_KEYWORDS),
                    "candidate_keywords": list(CANDIDATE_KEYWORDS),
                    "ballot_types": list(BALLOT_TYPES),
                    **DEFAULT_STRUCTURE
                }
                with open(safe_path, "wb") as fw:
                    fw.write(orjson.dumps(context_lib, option=orjson.OPT_INDENT_2))
                return context_lib
            context_lib = robust_orjson_loads(data)
    except Exception as e:
        # Backup corrupt file before overwriting
        backup_path = safe_path + ".corrupt"
        try:
            os.rename(safe_path, backup_path)
        except Exception:
            pass
        context_lib = {
            "panel_tags": list(PANEL_TAGS),
            "heading_tags": list(HEADING_TAGS),
            "custom_attr_patterns": [pat.pattern for pat in CUSTOM_ATTR_PATTERNS],
            "location_keywords": list(LOCATION_KEYWORDS),
            "candidate_keywords": list(CANDIDATE_KEYWORDS),
            "ballot_types": list(BALLOT_TYPES),
            **DEFAULT_STRUCTURE
        }
        with open(safe_path, "wb") as f:
            f.write(orjson.dumps(context_lib, option=orjson.OPT_INDENT_2))
        return context_lib

    # Merge in any missing keys from default (preserve existing data)
    if merge_defaults(context_lib, DEFAULT_STRUCTURE):
        save_context_library(context_lib, safe_path)

    # Extend dynamic sets with loaded values
    if "panel_tags" in context_lib:
        extend_panel_tags(context_lib["panel_tags"])
    if "heading_tags" in context_lib:
        extend_heading_tags(context_lib["heading_tags"])
    if "custom_attr_patterns" in context_lib:
        extend_custom_attr_patterns(context_lib["custom_attr_patterns"])
    if "location_keywords" in context_lib:
        extend_location_keywords(context_lib["location_keywords"])
    if "candidate_keywords" in context_lib:
        extend_candidate_keywords(context_lib["candidate_keywords"])
    if "ballot_types" in context_lib:
        extend_ballot_types(context_lib["ballot_types"])

    return context_lib

    # Merge in any missing keys from default (preserve existing data)
    if merge_defaults(context_lib, DEFAULT_STRUCTURE):
        save_context_library(context_lib, safe_path)

    # Extend dynamic sets with loaded values
    if "panel_tags" in context_lib:
        extend_panel_tags(context_lib["panel_tags"])
    if "heading_tags" in context_lib:
        extend_heading_tags(context_lib["heading_tags"])
    if "custom_attr_patterns" in context_lib:
        extend_custom_attr_patterns(context_lib["custom_attr_patterns"])
    if "location_keywords" in context_lib:
        extend_location_keywords(context_lib["location_keywords"])
    if "candidate_keywords" in context_lib:
        extend_candidate_keywords(context_lib["candidate_keywords"])
    if "ballot_types" in context_lib:
        extend_ballot_types(context_lib["ballot_types"])

    return context_lib
    
def update_context_library(path, update_fn):
    """
    Safely update the context library at `path` by applying `update_fn(library)`.
    Loads, mutates, and saves the full dict atomically.
    """
    with _CONTEXT_LOCK:
        lib = load_context_library(path)
        update_fn(lib)
        save_context_library(lib, path)

def file_hash(path):
    """Return SHA256 hash of file contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()
       
def backup_context_library(path=CONTEXT_LIBRARY_PATH, max_backups=5):
    """
    Make a timestamped backup of the context library before overwriting,
    but only if the content has changed. Keep only the most recent `max_backups` backups.
    """
    if not os.path.exists(path):
        return

    # Check if last backup is identical; if so, skip backup
    dir_ = os.path.dirname(path)
    base = os.path.basename(path)
    backups = sorted(
        [f for f in os.listdir(dir_) if f.startswith(base) and f.endswith(".bak")],
        reverse=True
    )
    current_hash = file_hash(path)
    if backups:
        last_backup_path = os.path.join(dir_, backups[0])
        try:
            if file_hash(last_backup_path) == current_hash:
                # No change since last backup
                return
        except Exception:
            pass

    # Make new backup
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = f"{path}.{timestamp}.bak"
    shutil.copy2(path, backup_path)

    # Prune old backups
    backups = sorted(
        [f for f in os.listdir(dir_) if f.startswith(base) and f.endswith(".bak")],
        reverse=True
    )
    for old in backups[max_backups:]:
        try:
            os.remove(os.path.join(dir_, old))
        except Exception:
            pass

def save_context_library(lib, path=None):
    """
    Robustly save the context library:
    - Always makes a timestamped backup before writing.
    - Writes atomically (temp file, then replace).
    - Never truncates or loses data on failure.
    """
    if path is None:
        path = CONTEXT_LIBRARY_PATH
    safe_path = safe_join(BASE_DIR, os.path.relpath(path, BASE_DIR))
    backup_context_library(safe_path)
    data = orjson.dumps(lib, option=orjson.OPT_INDENT_2)
    # Write to a temp file first
    dir_name = os.path.dirname(safe_path)
    with tempfile.NamedTemporaryFile("wb", dir=dir_name, delete=False) as tf:
        tf.write(data)
        temp_path = tf.name
    # Atomically replace the original file
    os.replace(temp_path, safe_path)

def merge_and_save_context_library(partial_dict, path=CONTEXT_LIBRARY_PATH):
    """
    Safely merge a partial dict into the context library and save atomically.
    """
    lib = load_context_library(path)
    lib.update(partial_dict)
    save_context_library(lib, path)

def update_context_library_field(key, value, path=CONTEXT_LIBRARY_PATH):
    """
    Safely update a top-level key in the context library.
    """
    lib = load_context_library(path)
    old_value = lib.get(key, None)
    lib[key] = value
    save_context_library(lib, path)
    # Optionally log the change
    logger.info(f"Updated context_library field '{key}': {old_value} -> {value}")

def update_domain_selector_cache(domain, selector, label, success=True):
    lib = load_context_library()
    domain_selectors = lib.setdefault("domain_selectors", {})
    entry = {
        "selector": selector,
        "label": label,
        "success_count": 1 if success else 0,
        "last_used": datetime.now(timezone.utc).isoformat()
    }
    found = False
    for e in domain_selectors.get(domain, []):
        if e["selector"] == selector:
            e["success_count"] += 1 if success else 0
            e["last_used"] = entry["last_used"]
            found = True
            break
    if not found:
        domain_selectors.setdefault(domain, []).append(entry)
    # Only update the domain_selectors field in the context library
    update_context_library_field("domain_selectors", domain_selectors)

def get_domain_selectors(domain):
    lib = load_context_library()
    return lib.get("domain_selectors", {}).get(domain, [])

def log_selector_attempt(domain, selector, label, success):
    lib = load_context_library()
    attempts = lib.setdefault("selector_attempts", [])
    attempts.append({
        "domain": domain,
        "selector": selector,
        "label": label,
        "success": success,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })
    update_context_library_field("selector_attempts", attempts)

# --- Unknown Tag/Attr Logging for ML/LLM Feedback ---
UNKNOWN_TAGS_LOG = set()
UNKNOWN_ATTRS_LOG = set()

def _get_log_path(filename: str) -> str:
    # Use the centralized LOG_DIR for all logs
    os.makedirs(LOG_DIR, exist_ok=True)
    return os.path.join(LOG_DIR, filename)

def log_unknown_tag(tag: str):
    if tag not in HTML_TAGS:
        UNKNOWN_TAGS_LOG.add(tag)
        try:
            log_path = _get_log_path("unknown_tags_log.jsonl")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"tag": tag}) + "\n")
        except Exception:
            pass

def log_unknown_attr(attr: str):
    if not any(pat.match(attr) for pat in CUSTOM_ATTR_PATTERNS):
        UNKNOWN_ATTRS_LOG.add(attr)
        try:
            log_path = _get_log_path("unknown_attrs_log.jsonl")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"attr": attr}) + "\n")
        except Exception:
            pass

# --- ML/LLM Feedback Integration Example ---
def integrate_llm_feedback(new_panel_tags=None, new_heading_tags=None, new_attr_patterns=None, new_location_keywords=None, new_candidate_keywords=None, new_ballot_types=None):
    if new_panel_tags:
        extend_panel_tags(new_panel_tags)
    if new_heading_tags:
        extend_heading_tags(new_heading_tags)
    if new_attr_patterns:
        extend_custom_attr_patterns(new_attr_patterns)
    if new_location_keywords:
        extend_location_keywords(new_location_keywords)
    if new_candidate_keywords:
        extend_candidate_keywords(new_candidate_keywords)
    if new_ballot_types:
        extend_ballot_types(new_ballot_types)
    save_context_library()

# --- Load context library at import time ---
load_context_library()



_segment_label_cache = {}

def normalize_segment_text(text: str) -> str:
    """Normalize segment text for canonical lookup (lowercase, strip, collapse spaces)."""
    if not text:
        return ""
    return " ".join(text.lower().strip().split())

def get_canonical_segment_label(text: str) -> str:
    """Return canonical label for normalized segment text, or None if not found."""
    norm = normalize_segment_text(text)
    return CANONICAL_SEGMENT_LABELS.get(norm)

def cache_segment_label(text: str, label: str):
    norm = normalize_segment_text(text)
    _segment_label_cache[norm] = label

def get_cached_segment_label(text: str) -> str:
    norm = normalize_segment_text(text)
    return _segment_label_cache.get(norm)

# --- Self-Heal Mode ---
def self_heal_context_library(max_retries=3, cooldown=2):
    """Self-heal: scan for misaligned NER, run correction bot, reload context library, repeat until clean or max_retries."""
    scan_script = os.path.join(os.path.dirname(__file__), "scan_misaligned_ner.py")
    for attempt in range(1, max_retries + 1):
        print(f"\n[LIBRARIAN SELF-HEAL] Attempt {attempt}...")
        scan_cmd = [sys.executable, scan_script, "--jsonl", "log/spacy_ner_train_data.jsonl"]
        scan_result = subprocess.run(scan_cmd, check=True, cwd=PROJECT_ROOT)
        if scan_result.returncode == 0:
            print("[LIBRARIAN SELF-HEAL] Data is clean. Exiting self-heal mode.")
            return 0
        print("[LIBRARIAN SELF-HEAL] Misalignments found. Launching manual_correction_bot...")
        bot_cmd = [sys.executable, "-m", "webapp.parser.bots.manual_correction_bot", "--enhanced"]
        subprocess.run(bot_cmd, check=True, cwd=PROJECT_ROOT)
        print(f"[LIBRARIAN SELF-HEAL] Sleeping {cooldown}s before rescanning...")
        time.sleep(cooldown)
    print("[LIBRARIAN SELF-HEAL] Max retries reached. Some misalignments may remain.")
    return 2

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Librarian utility for context library management.")
    parser.add_argument("--self-heal", action="store_true", help="Loop: scan -> correct -> rescan until clean or max retries")
    parser.add_argument("--max-retries", type=int, default=3, help="Max self-heal attempts")
    parser.add_argument("--cooldown", type=int, default=2, help="Seconds to wait between self-heal attempts")
    args = parser.parse_args()
    if args.self_heal:
        sys.exit(self_heal_context_library(args.max_retries, args.cooldown))
# --- Export all sets for use in other modules ---
__all__ = [
    "HTML_TAGS", "PANEL_TAGS", "HEADING_TAGS", "CUSTOM_ATTR_PATTERNS", "DISTRICT_REGEX",
    "BALLOT_TYPES", "BALLOT_TYPE_SORT_ORDER", "LOCATION_KEYWORDS", "PERCENT_KEYWORDS", "TOTAL_KEYWORDS",
    "MISC_FOOTER_KEYWORDS", "CANDIDATE_KEYWORDS", "PARTY_KEYWORDS", "LOCATION_ABBREVIATIONS", "VALID_TYPES", "CONTEST_KEYWORDS",
    "extend_panel_tags", "extend_heading_tags", "extend_html_tags", "extend_custom_attr_patterns",
    "extend_location_keywords", "extend_candidate_keywords", "extend_ballot_types",
    "log_unknown_tag", "log_unknown_attr", "integrate_llm_feedback","CANONICAL_SEGMENT_LABELS", 
    "normalize_segment_text", "get_canonical_segment_label", "cache_segment_label", "get_cached_segment_label",
    "ROOT_CONTAINER_TAGS", "ALWAYS_IGNORE_TAGS", "ALWAYS_IGNORE_CLASSES", "ALWAYS_IGNORE_IDS", "ICON_CLASSES", "ICON_TAGS", "BUTTON_CLASSES",
    "HEADING_CLASSES", "PANEL_CLASSES", "TIMESTAMP_CLASSES", "STRUCTURAL_TAGS", "TIMESTAMP_ID_PATTERNS", "TIMESTAMP_ATTRS",
    "STRUCTURAL_TAGS"
]
