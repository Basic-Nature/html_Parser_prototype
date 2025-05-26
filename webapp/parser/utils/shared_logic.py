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
    "Award Program", "Vice President", "Presidential", "Senate", "Senator", "Congress", "Representative", "Electors",
    "House of Representatives", "Proposition", "Amendment","House", "District Representative", "District Delegate",
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
    contest_heading=None,
    heading_tags=None,
    max_heading_level=6,
    logger=None,
    verbose=False,
    interactive=False
):
    import difflib

    def normalize(text):
        return (text or "").strip().lower()

    def get_nearest_heading(el):
        tags = heading_tags or [f"h{i}" for i in range(1, max_heading_level+1)]
        for tag in tags:
            try:
                heading = el.locator(f"xpath=ancestor::{tag}[1]")
                if heading.count() > 0:
                    return heading.nth(0).inner_text().strip()
            except Exception:
                continue
        return ""

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
                class_attr = el.get_attribute("class") or ""
                nearest_heading = get_nearest_heading(el)
                heading_score = 0
                if contest_heading and nearest_heading:
                    if normalize(contest_heading) in normalize(nearest_heading):
                        heading_score = 2
                    elif normalize(contest_heading).split()[0] in normalize(nearest_heading):
                        heading_score = 1
                # --- PATCH: Use substring and fuzzy matching, case-insensitive ---
                best_score = 0
                for kw in keywords:
                    for txt in [label, aria_label]:
                        txt_norm = normalize(txt)
                        kw_norm = normalize(kw)
                        if kw_norm in txt_norm:
                            score = 2.0  # Strong substring match
                        else:
                            score = difflib.SequenceMatcher(None, kw_norm, txt_norm).ratio()
                        if score > best_score:
                            best_score = score
                class_score = 1 if "btn-outline-dark" in class_attr else 0
                total_score = best_score + heading_score + class_score
                candidates.append((total_score, el, label, aria_label, class_attr, nearest_heading, selector))
            except Exception as e:
                if logger:
                    logger.debug(f"[TOGGLE] Error: {e}")
                continue

    candidates.sort(reverse=True, key=lambda x: x[0])

    for score, el, label, aria_label, class_attr, nearest_heading, selector in candidates:
        if score >= 1.5:  # Tune this threshold as needed
            el.scroll_into_view_if_needed()
            el.click()
            if logger and verbose:
                logger.info(f"[TOGGLE] Clicked: '{label or aria_label}' (class: {class_attr}) [heading: {nearest_heading}]")
            return True

    # Diagnostics: print all candidates if nothing matched
    if verbose or interactive:
        print("\n[DIAGNOSTIC] No automatic toggle match found.")
        print("Available clickable elements (sorted by score):")
        seen = set()
        filtered_candidates = []
        for score, el, label, aria_label, class_attr, nearest_heading, selector in candidates:
            # Filter out low-score options
            if score < 0.3:
                continue
            # Remove duplicates based on label, class, selector
            key = (label, class_attr, selector)
            if key in seen:
                continue
            seen.add(key)
            filtered_candidates.append((score, label, class_attr, nearest_heading, selector))
        for idx, (score, label, class_attr, nearest_heading, selector) in enumerate(filtered_candidates):
            print(f"{idx+1}. [score={score:.2f}] '{label}' (class: {class_attr}) [heading: {nearest_heading}] (selector: {selector})")
        if interactive and filtered_candidates:
            choice = input("Enter the number of the element to click (or press Enter to skip): ")
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(filtered_candidates):
                    # Find the original element to click
                    orig_idx = [i for i, cand in enumerate(candidates)
                                if (cand[2], cand[4], cand[6]) == (filtered_candidates[idx][1], filtered_candidates[idx][2], filtered_candidates[idx][4])][0]
                    el = candidates[orig_idx][1]
                    el.scroll_into_view_if_needed()
                    el.click()
                    print(f"[INTERACTIVE] Clicked: '{filtered_candidates[idx][1]}'")
                    return True
    if logger:
        logger.warning(f"[TOGGLE] No toggle found for selectors={selectors}, keywords={keywords}, contest_heading={contest_heading}")
    return False

def parse_single_contest(page, html_context, state, county, find_contest_panel):
    contest_title = html_context.get("selected_race")
    rprint(f"[cyan][INFO] Processing contest: {contest_title}[/cyan]")

    contest_panel = find_contest_panel(page, contest_title)
    if not contest_panel:
        rprint("[red][ERROR] Contest panel not found. Skipping.[/red]")
        return None, None, None, {"skipped": True}

    # Pass contest_title as contest_heading!
    click_dynamic_toggle(
        page,
        container=contest_panel,
        handler_keywords=[
            "View results by election district"
        ],
        logger=logger,
        verbose=True,
        interactive=True,
        heading_match=contest_title  # <-- Passes contest_heading here to shared_logic!
    )

    click_dynamic_toggle(
        page,
        container=contest_panel,
        handler_keywords=[
            "Vote Method", "Voting Method", "Ballot Method"
        ],
        logger=logger,
        verbose=True,
        interactive=True,
    )


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
    selectors = [
        "button",
        "a[role='button']",
        "input[type='button']",
        "input[type='submit']",
        "input[type='reset']",
        "[role='button']",
        ".btn", ".btn-outline-dark", ".btn-primary", ".btn-success", ".btn-info",
        ".btn-secondary", ".btn-link", ".btn-default", ".btn-danger", ".btn-warning", ".btn-light",
        ".btn-outline-primary", ".btn-outline-success", ".btn-outline-info",
        ".btn-outline-secondary", ".btn-outline-danger", ".btn-outline-warning", ".btn-outline-light",
        "div[class*='button']", "span[class*='button']",
        "div[class*='btn']", "span[class*='btn']",
        "app-vote-method button",
        "a",  # All <a> tags
        "span",  # All <span> tags (for clickable spans)
        "div[tabindex]", "span[tabindex]",
        "div[onclick]", "span[onclick]",
        "[onclick]", "[tabindex]",
    ]
    # Add Playwright :has-text() selectors for each keyword
    for kw in handler_keywords:
        selectors.append(f"a:has-text('{kw}')")
        selectors.append(f"button:has-text('{kw}')")
        selectors.append(f"span:has-text('{kw}')")
        selectors.append(f"div:has-text('{kw}')")
        selectors.append(f"label:has-text('{kw}')")
    search_root = container if container else page
    return find_best_toggle(
        search_root=search_root,
        selectors=selectors,
        keywords=handler_keywords,
        contest_heading=heading_match,
        logger=logger,
        verbose=verbose,
        heading_tags=heading_tags,
        max_heading_level=max_heading_level,
        interactive=interactive
    )

def find_best_toggle(
    search_root,
    selectors,
    keywords,
    contest_heading=None,
    heading_tags=None,
    max_heading_level=6,
    logger=None,
    verbose=False,
    interactive=False,
    custom_button_fallback=True,
    user_custom_selectors=None
):
    import difflib

    def normalize(text):
        return (text or "").strip().lower()

    def get_nearest_heading(el):
        tags = heading_tags or [f"h{i}" for i in range(1, max_heading_level+1)]
        for tag in tags:
            try:
                heading = el.locator(f"xpath=ancestor::{tag}[1]")
                if heading.count() > 0:
                    return heading.nth(0).inner_text().strip()
            except Exception:
                continue
        return ""

    candidates = []
    # 1. Try all selectors (including user-supplied)
    all_selectors = selectors[:]
    if user_custom_selectors:
        all_selectors.extend(user_custom_selectors)
    for selector in all_selectors:
        try:
            elements = search_root.locator(selector)
            for i in range(elements.count()):
                el = elements.nth(i)
                try:
                    if not el.is_visible() or not el.is_enabled():
                        continue
                    label = el.inner_text().strip()
                    aria_label = el.get_attribute("aria-label") or ""
                    class_attr = el.get_attribute("class") or ""
                    nearest_heading = get_nearest_heading(el)
                    heading_score = 0
                    if contest_heading and nearest_heading:
                        if normalize(contest_heading) in normalize(nearest_heading):
                            heading_score = 2
                        elif normalize(contest_heading).split()[0] in normalize(nearest_heading):
                            heading_score = 1
                    best_score = 0
                    for kw in keywords:
                        for txt in [label, aria_label]:
                            txt_norm = normalize(txt)
                            kw_norm = normalize(kw)
                            if kw_norm in txt_norm:
                                score = 2.0
                            else:
                                score = difflib.SequenceMatcher(None, kw_norm, txt_norm).ratio()
                            if score > best_score:
                                best_score = score
                    class_score = 1 if "btn-outline-dark" in class_attr else 0
                    total_score = best_score + heading_score + class_score
                    candidates.append((total_score, el, label, aria_label, class_attr, nearest_heading, selector))
                except Exception:
                    continue
        except Exception:
            continue

    # 2. Fallback: search for any visible element containing the keyword text
    for kw in keywords:
        try:
            elements = search_root.locator(f"*:has-text('{kw}')")
            for i in range(elements.count()):
                el = elements.nth(i)
                try:
                    if not el.is_visible() or not el.is_enabled():
                        continue
                    label = el.inner_text().strip()
                    aria_label = el.get_attribute("aria-label") or ""
                    class_attr = el.get_attribute("class") or ""
                    nearest_heading = get_nearest_heading(el)
                    total_score = 2.5
                    candidates.append((total_score, el, label, aria_label, class_attr, nearest_heading, f"*:has-text('{kw}')"))
                except Exception:
                    continue
        except Exception:
            continue
    """
    user_custom_selectors = []
    find_best_toggle(..., user_custom_selectors=user_custom_selectors)
    # Save user_custom_selectors to disk if changed
    """
    # 3. Fallback: get custom clickable elements if enabled
    if custom_button_fallback:
        try:
            custom_buttons = get_custom_buttons(search_root)
            for el in custom_buttons:
                try:
                    if not el.is_visible() or not el.is_enabled():
                        continue
                    label = el.inner_text().strip()
                    aria_label = el.get_attribute("aria-label") or ""
                    class_attr = el.get_attribute("class") or ""
                    nearest_heading = get_nearest_heading(el)
                    best_score = 0
                    for kw in keywords:
                        for txt in [label, aria_label]:
                            txt_norm = normalize(txt)
                            kw_norm = normalize(kw)
                            if kw_norm in txt_norm:
                                score = 2.0
                            else:
                                score = difflib.SequenceMatcher(None, kw_norm, txt_norm).ratio()
                            if score > best_score:
                                best_score = score
                    total_score = best_score + 0.5  # Slightly lower than direct match
                    candidates.append((total_score, el, label, aria_label, class_attr, nearest_heading, "custom_button"))
                except Exception:
                    continue
        except Exception:
            pass

    # Deduplicate by (label, class, selector)
    seen = set()
    unique_candidates = []
    for cand in candidates:
        key = (cand[2], cand[4], cand[6])
        if key not in seen:
            seen.add(key)
            unique_candidates.append(cand)

    unique_candidates.sort(reverse=True, key=lambda x: x[0])

    for score, el, label, aria_label, class_attr, nearest_heading, selector in unique_candidates:
        if score >= 1.5:
            el.scroll_into_view_if_needed()
            el.click()
            if logger and verbose:
                logger.info(f"[TOGGLE] Clicked: '{label or aria_label}' (class: {class_attr}) [heading: {nearest_heading}] (selector: {selector})")
            return True

    # Diagnostics: print all candidates if nothing matched
    if verbose or interactive:
        print("\n[DIAGNOSTIC] No automatic toggle match found.")
        print("Available clickable elements (sorted by score):")
        for idx, (score, label, class_attr, nearest_heading, selector) in enumerate(
            [(c[0], c[2], c[4], c[5], c[6]) for c in unique_candidates if c[0] >= 0.3]
        ):
            print(f"{idx+1}. [score={score:.2f}] '{label}' (class: {class_attr}) [heading: {nearest_heading}] (selector: {selector})")
        if interactive and unique_candidates:
            choice = input("Enter the number of the element to click (or press Enter to skip): ")
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(unique_candidates):
                    el = unique_candidates[idx][1]
                    el.scroll_into_view_if_needed()
                    el.click()
                    print(f"[INTERACTIVE] Clicked: '{unique_candidates[idx][2]}'")
                    # Optionally, ask user if this should be saved as a new selector
                    if user_custom_selectors is not None:
                        add = input("Save this selector for future runs? (y/n): ").strip().lower()
                        if add == "y":
                            user_custom_selectors.append(selector)
                    return True
    if logger:
        logger.warning(f"[TOGGLE] No toggle found for selectors={selectors}, keywords={keywords}, contest_heading={contest_heading}")
    return False



def get_custom_buttons(container):
    """
    Returns all custom clickable elements in the container.
    This is robust for many HTML patterns seen in election and admin UIs.
    """
    selectors = [
        '[role="button"]',
        'div[tabindex]', 'span[tabindex]', 'li[tabindex]',
        'div[onclick]', 'span[onclick]', 'li[onclick]',
        '[tabindex]:not(input):not(textarea):not(select)',  # Any focusable non-input
        '[onclick]:not(a):not(button)',  # Any clickable non-link/button
        '[class*="button"]', '[class*="btn"]', '[class*="click"]', '[class*="toggle"]',
        '[data-action]', '[data-toggle]', '[data-role*="button"]', '[data-role*="toggle"]',
        '[aria-pressed]', '[aria-expanded]', '[aria-controls]',
        'a:not([href^="http"]):not([href^="#"])',  # Local anchor links (often used as buttons)
        'label[for]',  # Labels that may act as toggles
        'input[type="checkbox"]', 'input[type="radio"]',  # Sometimes styled as buttons
        'summary',  # <summary> tag in <details>
        'mat-button', 'mat-icon-button', 'mat-raised-button',  # Angular Material
        'button', 'a[role="button"]',  # Redundant but safe
    ]
    selector_str = ", ".join(selectors)
    return container.query_selector_all(selector_str)

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

