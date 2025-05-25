##shared_logic.py - Common parsing utilities across states and formats.

##This module is designed to centralize shared patterns that repeat across many state or format handlers.
##While election website structures vary significantly by vendor or county/state implementation,
##many elements are consistent such as contest labeling, vote method naming, and tabular breakdowns.
##These helpers promote DRY principles and consistent behavior across handlers.
import difflib
import os
import re
from ..utils.shared_logger import logger, rprint
# Dynamically generate all header tags up to h24

HEADER_TAGS = [f"h{i}" for i in range(1, 25)]
# Common element tags to scan
ELEMENT_TAGS = HEADER_TAGS + [
    "strong", "b", "span", "table", "div", "li", "label", "option", "button", "a", "td", "th"
]
# Add common ARIA and role-based selectors
ROLE_TAGS = [
    '[role]', '[aria-label]', '[data-title]', '[class*="header"]', '[class*="title"]', '[class*="name"]'
]
# Combine all selectors for a broad query
ALL_SELECTORS = ", ".join(ELEMENT_TAGS + ROLE_TAGS)

# Regex for matching years and symbols
YEAR_REGEX = re.compile(r"\b((?:19|20)\d{2})\b")
SYMBOL_REGEX = re.compile(r"[\s\n\r\t\$0]+")  # Remove whitespace, newlines, tabs, $0, etc.

# Max wait for dynamic content
SCAN_WAIT_SECONDS = int(os.getenv("SCAN_WAIT_SECONDS", "7"))

IGNORE_SUBSTRINGS = [
    "turnout", "ballots cast", "voter registration", "unofficial", "total votes"
]

# Core parts for contest/office detection
CONTEST_PARTS = [
    "President", "Vice President", "Senate", "Senator", "Congress", "Representative", "Governor",
    "Board", "Supervisor", "Elections", "Registration", "District", "County", "City", "Town", "Ward",
    "School", "Education", "Court", "Justice", "Judge", "Clerk", "Commissioner", "Trustee", "Auditor", "state",
    "Assembly", "House", "Delegate", "Elector", "Electors", "Attorney General", "Comptroller", "Treasurer",
    "Secretary of State", "District Attorney", "Public Utility", "Soil and Water Conservation",
    "Conservation District", "Conservation Board", "Conservation Supervisor", "Conservation Director",
    "Conservation Commissioner", "Conservation Board of Supervisors", "Conservation Board of Directors",
    "Conservation Board of Commissioners", "Conservation Board of Assessors", "Conservation Board of Auditors",
]

# Full contest labels (for direct matching)
COMMON_CONTEST_LABELS = [
    "President", "Vice President", "Presidential", "Senate", "Senator", "Congress", "Representative", "Electors",
    "House of Representatives", "House", "District Representative", "District Delegate",
    "Governor", "Lieutenant Governor", "Attorney General", "Comptroller", "Treasurer", "Secretary of State",
    "State Senator", "State Assembly", "State Representative", "Assembly Member", "Member of Assembly",
    "State House", "State Senate", "State House of Representatives", "State Delegate", "State Board of Education",
    "State Board of Elections", "State Board of Equalization", "State Board of Supervisors",
    "State Board of Trustees", "State Board of Directors", "State Board of Commissioners",
    "State Board of Assessors", "State Board of Auditors", "State Board of Registrars",
    "State Board of Supervisors of Elections", "State Board of Supervisors of Registration",
    "State Board of Supervisors of Voter Registration", "State Board of Supervisors of Elections and Registration",
    "State Board of Supervisors of Elections and Voter Registration", "State Board of Supervisors of Elections and Registrars", 
    "State Board of Supervisors of Elections and Registrars of Voters", "State Board of Supervisors of Elections and Registrars of Voter Registration",
    "State Board of Supervisors of Elections and Registrars of Voter Registration and Elections",
    "Mayor", "City Council", "Councilmember", "County Clerk", "Sheriff", "Assessor", "District Attorney",
    "County Commissioner", "City Auditor", "Board of Supervisors", "Town Council", "Clerk",
    "Judge", "Justice", "Supreme Court Justice", "District Court", "Appellate Court", "Trustee",
    "Circuit Court", "Magistrate", "Municipal Court", "Family Court", "Probate Court",
    "School Board", "Board of Education", "Superintendent of Schools",
    "School Committee", "School Trustee", "Board of Trustees", "Board of School Directors",
    "Public Utility Commissioner", "Soil and Water Conservation District Supervisor",
    "Soil and Water Conservation Board", "Soil and Water Conservation District Director",
    "Soil and Water Conservation District Board", "Soil and Water Conservation District Commissioner"
]

# Map parts to full labels for reverse lookup
PART_TO_LABELS = {}
for label in COMMON_CONTEST_LABELS:
    for part in CONTEST_PARTS:
        if part.lower() in label.lower():
            PART_TO_LABELS.setdefault(part, set()).add(label)

# For state board detection
STATE_BOARD_PARTS = [
    "State Board", "Supervisors", "Elections", "Registration", "Voter Registration", "Registrars", "Voters"
]

COMMON_PRECINCT_HEADERS = [
    "Precinct", "Ward", "District", "Voting District", "County", "City", "Township", "Neighborhood",
    "Polling Place", "Election District", "Voting Area", "Electoral District", "Community District",
    "Voting Center", "Voting Location", "Polling Station", "Polling Area", "Voting Precinct",
    "Voting Place", "Polling District", "Electoral Area", "Electoral Division", "Electoral Zone",
    "Electoral Region", "Electoral Section", "Electoral Ward", "Electoral Unit", "Electoral Subdivision",
    "Electoral Precinct", "Electoral Districts", "Electoral Wards", "Electoral Units", "Electoral Subdivisions"
]

COMMON_REPORTING_STATUS = [
    "Reported", "Reporting", "Counted", "Counting", "Tabulated", "Tabulation",
    "Final", "Complete", "Partial", "In Progress", "Unofficial"
]

COMMON_VOTE_METHODS = [
    "Absentee", "Early Voting", "Election Day", "Vote By Mail", "Vote By Mail Ballots",
    "Vote By Mail Votes", "Vote By Mail Voting", "Vote By Mail Ballot", "Vote By Mail Vote",
    "In-Person", "In Person", "In-Person Voting"
]

def normalize_text(text):
    """Strips and lowers text for fuzzy matching purposes."""
    return re.sub(r"\s+", " ", text.strip().lower())

def match_any(label, keywords):
    """Case-insensitive substring match against a list of keywords."""
    label = normalize_text(label)
    return any(k.lower() in label for k in keywords)

def is_state_board_label(label):
    """Detects if a label is a 'State Board' type using component parts."""
    label = normalize_text(label)
    if "state board" in label:
        for part in STATE_BOARD_PARTS[1:]:
            if part.lower() in label:
                return True
    return False

def is_contest_label(label):
    """Detects if a label is a contest/race label by direct or part-based match."""
    label_norm = normalize_text(label)
    # Direct match
    if match_any(label_norm, COMMON_CONTEST_LABELS):
        return True
    # Part-based match: at least 2 unique parts present
    parts_found = [part for part in CONTEST_PARTS if part.lower() in label_norm]
    return len(parts_found) >= 2

def get_contest_parts(label):
    """Returns the set of contest parts found in a label."""
    label_norm = normalize_text(label)
    return {part for part in CONTEST_PARTS if part.lower() in label_norm}

def is_precinct_header(label):
    return match_any(label, COMMON_PRECINCT_HEADERS)

def is_vote_method(label):
    return match_any(label, COMMON_VOTE_METHODS)

def is_reporting_status(label):
    return match_any(label, COMMON_REPORTING_STATUS)

def build_csv_headers(rows):
    """Union of all keys in all rows, sorted for consistency."""
    headers = set()
    for row in rows:
        headers.update(row.keys())
    return sorted(headers)

def canonicalize_column(label, method=None):
    """Returns a canonical column name for CSV output."""
    parts = get_contest_parts(label)
    if not parts:
        return label
    base = " / ".join(sorted(parts))
    if method:
        return f"{base} - {method}"
    return base

# Example: Use in your pipeline
# for header in table_headers:
#     if is_precinct_header(header):
#         ...
#     elif is_vote_method(header):
#         ...
#     elif is_reporting_status(header):
#         ...
#     elif is_contest_label(header):
#         col_name = canonicalize_column(header)
#         ...

def find_best_toggle(
    search_root,
    selectors,
    keywords,
    logger=None,
    verbose=False,
    heading_match=None,
    heading_tags=None,
    max_heading_level=20,
    page=None,
    interactive=False
):
    """
    Attempts to find and click a toggle/button/link matching handler-provided keywords.
    If no match is found, interactively prompts the user to select from best candidates.
    Returns True if clicked, False otherwise.
    """
    def matches_keywords(text, keywords):
        text_norm = re.sub(r'\s+', ' ', text or '').strip().lower()
        for keyword in keywords:
            keyword_norm = re.sub(r'\s+', ' ', keyword or '').strip().lower()
            if keyword_norm in text_norm:
                return True
            # Fuzzy match
            if difflib.get_close_matches(keyword_norm, [text_norm], n=1, cutoff=0.8):
                return True
        return False

    candidates = []
    for selector in selectors:
        elements = search_root.locator(selector)
        for i in range(elements.count()):
            el = elements.nth(i)
            try:
                if not el.is_visible() or not el.is_enabled():
                    continue
                label = el.inner_text().strip()
                aria_label = el.get_attribute("aria-label") or ""
                all_labels = [label, aria_label]
                # If heading_match is required, check parent headings
                if heading_match:
                    panel = el.locator(f"xpath=ancestor::*[self::{' or self::'.join(['div','section','p-panel'])}][1]")
                    found_heading = False
                    tags_to_check = heading_tags or [f"h{n}" for n in range(1, max_heading_level+1)]
                    for tag in tags_to_check:
                        heading = panel.locator(tag)
                        if heading.count() > 0:
                            heading_text = heading.inner_text().strip().lower()
                            if heading_match.lower() in heading_text:
                                found_heading = True
                                break
                    if not found_heading:
                        continue
                # Keyword match
                if any(matches_keywords(l, keywords) for l in all_labels if l):
                    el.scroll_into_view_if_needed()
                    el.click()
                    if logger and verbose:
                        logger.info(f"[TOGGLE] Clicked: '{label or aria_label}' (selector: {selector})")
                    return True
                # Collect for fallback
                candidates.append((el, label, aria_label, selector))
            except Exception as e:
                if logger:
                    logger.debug(f"[TOGGLE] Error checking toggle: {e}")
                continue

    # If no match, offer interactive prompt if enabled
    if interactive and candidates:
        print("\n[INTERACTIVE] No automatic toggle match found.")
        print("Available clickable elements (best fuzzy matches first):")
        scored = []
        for el, label, aria_label, selector in candidates:
            best_score = 0
            best_kw = ""
            for kw in keywords:
                score = difflib.SequenceMatcher(None, kw.lower(), (label or aria_label).lower()).ratio()
                if score > best_score:
                    best_score = score
                    best_kw = kw
            scored.append((best_score, label, aria_label, selector, el))
        scored.sort(reverse=True)
        for idx, (score, label, aria_label, selector, el) in enumerate(scored[:10]):
            print(f"{idx+1}. [{score:.2f}] '{label or aria_label}' (selector: {selector})")
        choice = input("Enter the number of the element to click (or press Enter to skip): ")
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(scored):
                el = scored[idx][4]
                el.scroll_into_view_if_needed()
                el.click()
                print(f"[INTERACTIVE] Clicked: '{scored[idx][1] or scored[idx][2]}'")
                return True

    if logger:
        logger.warning(f"[TOGGLE] No toggle found for selectors={selectors}, keywords={keywords}, heading_match={heading_match}")
    return False

def click_dynamic_toggle(
    page,
    container,
    handler_keywords,
    logger=None,
    verbose=False,
    heading_match=None,
    heading_tags=None,
    max_heading_level=20,
    interactive=False
):
    """
    Handler supplies handler_keywords (list of phrases).
    Attempts to click a matching toggle, or interactively prompts if not found.
    """
    selectors = ["button", "a", "[role='button']", "div[tabindex]", "span[tabindex]"]
    search_root = container if container else page
    return find_best_toggle(
        search_root=search_root,
        selectors=selectors,
        keywords=handler_keywords,
        logger=logger,
        verbose=verbose,
        heading_match=heading_match,
        heading_tags=heading_tags,
        max_heading_level=max_heading_level,
        page=page,
        interactive=interactive
    )

def get_custom_buttons(container):
    """
    Returns all custom clickable elements in the container.
    """
    return container.query_selector_all('[role="button"], div[tabindex], span[tabindex]')

def autoscroll_until_stable(page, max_stable_frames=5, step=8000, delay_ms=200):
    """
    Continuously scrolls a Playwright page until its scroll height stabilizes.
    Useful for dynamic election websites where all precinct data is only visible after scrolling.
    """
    logger.info("[SCROLL] Starting auto-scroll until page height stabilizes...")
    # Scroll to the top of the page first
    page.evaluate("window.scrollTo(0, 0)")
    # Wait for a moment to allow the page to settle
    page.wait_for_timeout(delay_ms)
    stable = 0
    last_height = 0
    while stable < max_stable_frames:
        current_height = page.evaluate("() => document.body.scrollHeight")
        if current_height == last_height:
            stable += 1
        else:
            stable = 0
        last_height = current_height
        page.evaluate(f"window.scrollBy(0, {step})")
        page.wait_for_timeout(delay_ms)
    logger.info("[SCROLL] Completed scrolling until page height stabilized.")

