import json, os, re
from typing import Set, List, Dict, Any

# --- Path to your context library ---
CONTEXT_LIBRARY_PATH = os.path.join(
    os.path.dirname(__file__), "..", "Context_Integration", "Context_Library", "context_library.json"
)

# --- Central Dynamic Sets (used everywhere) ---
HTML_TAGS: Set[str] = set([
    "html", "head", "title", "body", "h1", "h2", "h3", "h4", "h5", "h6",
    "b", "i", "center", "ul", "li", "br", "p", "hr", "img", "a", "span", "div", "button", "input", "form", "table"
])
PANEL_TAGS: Set[str] = set([
    "section", "fieldset", "panel", "div", "p-panel", "app-ballot-item-wrapper", "article"
])
HEADING_TAGS: Set[str] = set([
    "h1", "h2", "h3", "h4", "h5", "h6"
])
CUSTOM_ATTR_PATTERNS: List[re.Pattern] = [
    re.compile(r"^data-"),
    re.compile(r"^aria-"),
    re.compile(r"^role$"),
]

# --- Table/Entity Keywords (from table_core, dynamic_table_extractor, etc.) ---
BALLOT_TYPES = [
    "Election Day", "Early Voting", "Absentee", "Mail", "Provisional", "Affidavit", "Other", "Void"
]
LOCATION_KEYWORDS = {
    "precinct", "ward", "district", "location", "area", "city", "municipal", "town",
    "borough", "village", "county", "division", "subdistrict", "polling place", "ed", "municipality"
}
PERCENT_KEYWORDS = {
    "% precincts reporting", "% reported", "percent reported", "fully reported", "precincts reporting"
}
TOTAL_KEYWORDS = {"total", "sum", "votes", "overall", "all", "Percent Reported", "Reporting Status"}
MISC_FOOTER_KEYWORDS = {"undervote", "overvote", "scattering", "write-in", "blank", "void", "spoiled"}
CANDIDATE_KEYWORDS = {
    "candidate", "candidates", "name", "nominee", "person", "individual", "contestant"
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

def log_unknown_tag(tag: str):
    if tag not in HTML_TAGS:
        UNKNOWN_TAGS_LOG.add(tag)
        # Optionally: write to a file for LLM/human review

def log_unknown_attr(attr: str):
    if not any(pat.match(attr) for pat in CUSTOM_ATTR_PATTERNS):
        UNKNOWN_ATTRS_LOG.add(attr)
        # Optionally: write to a file for LLM/human review

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

# --- Export all sets for use in other modules ---
__all__ = [
    "HTML_TAGS", "PANEL_TAGS", "HEADING_TAGS", "CUSTOM_ATTR_PATTERNS",
    "BALLOT_TYPES", "LOCATION_KEYWORDS", "PERCENT_KEYWORDS", "TOTAL_KEYWORDS",
    "MISC_FOOTER_KEYWORDS", "CANDIDATE_KEYWORDS",
    "extend_panel_tags", "extend_heading_tags", "extend_html_tags", "extend_custom_attr_patterns",
    "extend_location_keywords", "extend_candidate_keywords", "extend_ballot_types",
    "log_unknown_tag", "log_unknown_attr", "integrate_llm_feedback", "save_context_library"
]