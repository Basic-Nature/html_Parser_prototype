import json, os, re
from typing import Set, List, Dict, Any
from ..config import CONTEXT_LIBRARY_PATH, BASE_DIR
LOG_PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
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
def load_context_library():
    if os.path.exists(CONTEXT_LIBRARY_PATH):
        with open(CONTEXT_LIBRARY_PATH, "r", encoding="utf-8") as f:
            context_lib = json.load(f)
            extend_panel_tags(context_lib.get("panel_tags", []))
            extend_heading_tags(context_lib.get("heading_tags", []))
            extend_custom_attr_patterns(context_lib.get("custom_attr_patterns", []))
            extend_location_keywords(context_lib.get("location_keywords", []))
            extend_candidate_keywords(context_lib.get("candidate_keywords", []))
            extend_ballot_types(context_lib.get("ballot_types", []))

def save_context_library():
    context_lib = {
        "panel_tags": list(PANEL_TAGS),
        "heading_tags": list(HEADING_TAGS),
        "custom_attr_patterns": [pat.pattern for pat in CUSTOM_ATTR_PATTERNS],
        "location_keywords": list(LOCATION_KEYWORDS),
        "candidate_keywords": list(CANDIDATE_KEYWORDS),
        "ballot_types": list(BALLOT_TYPES),
    }
    with open(CONTEXT_LIBRARY_PATH, "w", encoding="utf-8") as f:
        json.dump(context_lib, f, indent=2)

# --- Unknown Tag/Attr Logging for ML/LLM Feedback ---
UNKNOWN_TAGS_LOG = set()
UNKNOWN_ATTRS_LOG = set()

def _get_log_path(filename: str) -> str:
    # Get the parent directory of webapp (i.e., project root)
    log_dir = os.path.join(LOG_PARENT_DIR, "log")
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, filename)

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

# --- Export all sets for use in other modules ---
__all__ = [
    "HTML_TAGS", "PANEL_TAGS", "HEADING_TAGS", "CUSTOM_ATTR_PATTERNS", "DISTRICT_REGEX",
    "BALLOT_TYPES", "BALLOT_TYPE_SORT_ORDER", "LOCATION_KEYWORDS", "PERCENT_KEYWORDS", "TOTAL_KEYWORDS",
    "MISC_FOOTER_KEYWORDS", "CANDIDATE_KEYWORDS", "PARTY_KEYWORDS", "LOCATION_ABBREVIATIONS", "VALID_TYPES", "CONTEST_KEYWORDS",
    "extend_panel_tags", "extend_heading_tags", "extend_html_tags", "extend_custom_attr_patterns",
    "extend_location_keywords", "extend_candidate_keywords", "extend_ballot_types",
    "log_unknown_tag", "log_unknown_attr", "integrate_llm_feedback", "save_context_library",
    "CANONICAL_SEGMENT_LABELS", "normalize_segment_text", "get_canonical_segment_label", "cache_segment_label", "get_cached_segment_label",
]
