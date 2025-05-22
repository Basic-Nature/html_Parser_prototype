# utils/html_scanner.py
# ==============================================================
# Extracts general HTML context from pages: contest names, year, state/county hints.
# Dynamically builds a context dictionary for use by downstream handlers.
# This module is responsible for scanning and analyzing HTML pages to extract contextual metadata
# such as race names, election years, inferred state/county, and election types. This information is
# used for routing and contest selection in the broader HTML election parsing workflow.
# ==============================================================
import re
import time
import os
from collections import defaultdict
from typing import Dict


from utils.shared_logger import logger
#
CONTEST_PANEL_TAGS = [
    "p-panel", "div.panel", "section.contest-panel"
]

PRECINCT_ELEMENT_TAGS = [
    "h3", "strong", "b", "span", "table"
]

CONTEST_HEADER_SELECTORS = [
    "h1", "h2", "h3", "h4", "h5", "h6"
]
# List of known election-related race keywords used to identify contests
RACE_KEYWORDS = [
    # Federal
    "President", "Vice President", "Presidential", "Senate", "Senator", "Congress", "Representative", "Electors",
    # State-level
    "Governor", "Lieutenant Governor", "Attorney General", "Comptroller", "Treasurer", "Secretary of State",
    "State Senator", "State Assembly", "State Representative", "Assembly Member", "Member of Assembly",
    # Local
    "Mayor", "City Council", "Councilmember", "County Clerk", "Sheriff", "Assessor", "District Attorney",
    "County Commissioner", "City Auditor", "Board of Supervisors", "Town Council", "Clerk",
    # Judicial
    "Judge", "Justice", "Supreme Court Justice", "District Court", "Appellate Court", "Trustee",
    # Education
    "School Board", "Board of Education", "Superintendent of Schools",
    # Ballot Items
    "Ballot Measure", "Proposition", "Initiative", "Referendum", "Recall", "Amendment",
    # Election Types
    "Primary", "General Election", "Special Election", "Runoff"
]
IGNORE_SUBSTRINGS = ["turnout", "ballots cast", "voter registration", "unofficial", "total votes"]

# Regex for matching years
YEAR_REGEX = re.compile(r"\b((?:19|20)\d{2})\b")

# Delay before scanning content to allow dynamic content to load
SCAN_WAIT_SECONDS = int(os.getenv("SCAN_WAIT_SECONDS", "7"))

# State/county keywords to assist in inference if present on the page
STATE_HINTS = [
    "new york", "ny", "rockland",
    "arizona", "az",
    "pennsylvania", "pa",
    "florida", "fl",
]


def scan_html_for_context(page) -> Dict:
    """
    Scans the page HTML for:
    - Structured elements (h1–h6, td, div.card-title, etc.)
    - Election years and race keywords
    - Inferred state, county, election type

    Returns a context dictionary with all structured insights.
    """
    context_result = {
        "available_races": defaultdict(list),
        "_source_map": defaultdict(list),
        "raw_text": "",
        "indicators": [],
        "state": "Unknown",
        "county": None,
        "election_type": None,
        "error": None
    }

    try:
        logger.info(f"[SCAN] Waiting {SCAN_WAIT_SECONDS} seconds for page content to load...")
        time.sleep(SCAN_WAIT_SECONDS)

        # Extract full inner text of the body
        text = page.inner_text("body")
        context_result["raw_text"] = text

        # Target elements expected to hold contest/race information (expanded)
        elements = page.query_selector_all("""
            h1, h2, h3, h4, h5, h6,
            td, th,
            .card-title, .panel-header, .section-title,
            span[ng-bind='levels.$key'],
            li, label, option, button, a,
            .race-title, .election-name, .contest-name, .header-label,
            div[data-title], *[role="heading"],
            section h3, article h4,
            #contest-list *, div.tile, div.card, div.label-box
        """)
        last_detected_year = None
        last_detected_type = None

        # Search for race keywords and year context
        for el in elements:
            try:
                segment = el.inner_text().strip()
                try:
                    tag_name = el.evaluate("e => e.tagName").lower()
                    context_result["_source_map"][tag_name].append(segment)
                except Exception:
                    pass
            except Exception:
                continue

            if not segment:
                continue

            line_lower = segment.lower()
            # Flag if "County Breakdown" appears (used by state handlers like PA)
            if "county breakdown" in line_lower:
                context_result["requires_county_click"] = True
            if any(ignore in line_lower for ignore in IGNORE_SUBSTRINGS):
                continue
            year_matches = YEAR_REGEX.findall(segment)
            if year_matches:
                last_detected_year = max(year_matches)
                logger.debug(f"[SCAN] Detected year: {last_detected_year} from line: '{segment[:50]}...")

            for keyword in RACE_KEYWORDS:
                if keyword.lower() in line_lower:
                    tag_year = last_detected_year if last_detected_year else "Unknown"

                    # Attempt to detect election type dynamically
                    if "general" in line_lower:
                        detected_type = "General"
                    elif "primary" in line_lower:
                        detected_type = "Primary"
                    elif "special" in line_lower:
                        detected_type = "Special"
                    elif "runoff" in line_lower:
                        detected_type = "Runoff"
                    else:
                        detected_type = context_result.get("election_type") or last_detected_type or "Unknown"

                    context_result["election_type"] = last_detected_type = detected_type

                    if isinstance(context_result["available_races"].get(tag_year), list):
                        # Promote old flat list structure to dict
                        context_result["available_races"][tag_year] = {"Uncategorized": context_result["available_races"][tag_year]}
                    context_result["available_races"].setdefault(tag_year, {}).setdefault(detected_type, []).append(segment)
                    break  # Stop after first match

            for hint in STATE_HINTS:
                if hint in line_lower:
                    context_result["state"] = hint.upper() if len(hint) == 2 else hint.title()

        # Flatten all detected races
        all_races = []
        for year_group in context_result["available_races"].values():
            if isinstance(year_group, dict):
                for races in year_group.values():
                    all_races.extend(races)
            elif isinstance(year_group, list):
                all_races.extend(year_group)
        context_result["indicators"] = sorted(set(all_races))

        # Infer county from enhanced voting style URL
        url = page.url.lower()
        match = re.search(r"public/([a-z\-]+)-county", url)
        if match:
            county_slug = match.group(1).replace("-", " ")
            context_result["county"] = county_slug.title()

            # Try to guess state using router
            if context_result["state"] == "Unknown":
                from state_router import get_handler
                handler = get_handler(state_abbreviation=None, county_name=county_slug.title())
                if handler:
                    inferred = handler.__name__.split(".")[-3].replace("_", " ").title()
                    context_result["state"] = inferred.upper() if len(inferred) == 2 else inferred

        # Attempt to identify election type
        for election_type in ["general", "primary", "special", "runoff"]:
            if election_type in text.lower():
                normalized_type = election_type.strip().title()
                context_result["election_type"] = last_detected_type = normalized_type
                break

        # Display what was found if any
        if context_result["available_races"]:
            logger.debug("[SCAN] Detected races and years:")
            for year, races in context_result["available_races"].items():
                logger.debug(f"  {year}: {', '.join(sorted(set(races)))}")
        else:
            logger.warning("[SCAN] No known race keywords detected in page.")

    except Exception as e:
        err_msg = f"[SCAN ERROR] HTML parsing failed: {e}"
        logger.error(err_msg)
        context_result["error"] = err_msg
    try:
        candidate_tags = page.query_selector_all("span[ng-bind='item.CandidateName']")
        candidates = [c.inner_text().strip() for c in candidate_tags if c.inner_text().strip()]
        if candidates:
            context_result["candidates"] = sorted(set(candidates))
    except Exception:  
        pass
    return context_result

def get_detected_races_from_context(context_result):
    """
    Returns a flat, sorted list of (year, election_type, race) tuples.
    """
    flat = []
    for year, year_group in context_result.get("available_races", {}).items():
        if isinstance(year_group, dict):
            for etype, races in year_group.items():
                for race in races:
                    flat.append((year, etype, race))
        elif isinstance(year_group, list):  # fallback structure
            for race in year_group:
                flat.append((year, "Unknown", race))
    return sorted(flat)

