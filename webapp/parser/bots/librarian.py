import json, os, re
from typing import Set, List, Dict, Any
from ..config import CONTEXT_LIBRARY_PATH, PROJECT_ROOT, LOG_DIR
import orjson
import subprocess
import sys
import time
from ..utils.shared_logic import update_context_library_field

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

# --- Context Library Integration ---
def robust_orjson_loads(val):
    if isinstance(val, bytes):
        return orjson.loads(val)
    elif isinstance(val, str):
        return orjson.loads(val.encode("utf-8"))
    else:
        raise TypeError(f"Cannot decode type {type(val)} with orjson")

def load_context_library(path=CONTEXT_LIBRARY_PATH):
    # If file is missing or empty, initialize with default structure
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        context_lib = {
            "panel_tags": list(PANEL_TAGS),
            "heading_tags": list(HEADING_TAGS),
            "custom_attr_patterns": [pat.pattern for pat in CUSTOM_ATTR_PATTERNS],
            "location_keywords": list(LOCATION_KEYWORDS),
            "candidate_keywords": list(CANDIDATE_KEYWORDS),
            "ballot_types": list(BALLOT_TYPES),
        }
        with open(path, "wb") as f:
            f.write(orjson.dumps(context_lib, option=orjson.OPT_INDENT_2))
        return context_lib
    with open(path, "rb") as f:
        context_lib = robust_orjson_loads(f.read())
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

def save_context_library():
    update_context_library_field("panel_tags", list(PANEL_TAGS))
    update_context_library_field("heading_tags", list(HEADING_TAGS))
    update_context_library_field("custom_attr_patterns", [pat.pattern for pat in CUSTOM_ATTR_PATTERNS])
    update_context_library_field("location_keywords", list(LOCATION_KEYWORDS))
    update_context_library_field("candidate_keywords", list(CANDIDATE_KEYWORDS))
    update_context_library_field("ballot_types", list(BALLOT_TYPES))

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
    "log_unknown_tag", "log_unknown_attr", "integrate_llm_feedback", "save_context_library",
    "CANONICAL_SEGMENT_LABELS", "normalize_segment_text", "get_canonical_segment_label", "cache_segment_label", "get_cached_segment_label",
    "ROOT_CONTAINER_TAGS", "ALWAYS_IGNORE_TAGS", "ALWAYS_IGNORE_CLASSES", "ALWAYS_IGNORE_IDS", "ICON_CLASSES", "ICON_TAGS", "BUTTON_CLASSES",
    "HEADING_CLASSES", "PANEL_CLASSES", "TIMESTAMP_CLASSES", "STRUCTURAL_TAGS", "TIMESTAMP_ID_PATTERNS", "TIMESTAMP_ATTRS",
    "STRUCTURAL_TAGS"
]
