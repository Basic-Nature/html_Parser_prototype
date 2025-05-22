##shared_logic.py - Common parsing utilities across states and formats.

##This module is designed to centralize shared patterns that repeat across many state or format handlers.
##While election website structures vary significantly by vendor or county/state implementation,
##many elements are consistent such as contest labeling, vote method naming, and tabular breakdowns.
##These helpers promote DRY principles and consistent behavior across handlers.
import difflib
from numpy import e

from typing import List, Dict
from utils.shared_logger import logger, rprint
import re


# Common keyword mappings
COMMON_CONTEST_LABELS = [
    "President", "U.S. Senate", "U.S. House", "Governor", "Lieutenant Governor",
    "Attorney General", "State Senate", "State House", "Supreme Court",
    "Ballot Measure", "Constitutional Amendment", "Referendum", "Proposition"
]
# Common vote method labels
# These are the most common vote method labels used in election reporting

COMMON_VOTE_METHODS = [
    "Election Day", "Early Voting", "Absentee", "Mail", "Provisional", "Total"
]

COMMON_PRECINCT_HEADERS = [
    "Precinct", "Ward", "District", "Voting District"
]
def normalize_label(label) -> str:
    """
    Ensures a contest/race label is a string.
    - Joins tuple elements with a space.
    - Converts other types to string.
    """
    if isinstance(label, tuple):
        return " ".join(str(x) for x in label)
    return str(label)

def normalize_text(text):
    """
    Strips and lowers text for fuzzy matching purposes.
    """
    return re.sub(r"\s+", " ", text.strip().lower())

def extract_common_contests(text_blocks):
    """
    Returns a set of detected contests based on shared labels.
    """
    found = set()
    for block in text_blocks:
        norm = normalize_text(block)
        for label in COMMON_CONTEST_LABELS:
            if normalize_text(label) in norm:
                found.add(label)
    return sorted(found)

def extract_vote_methods(columns):
    """
    Identifies which vote method types are represented in table headers.
    """
    methods_found = set()
    for col in columns:
        for method in COMMON_VOTE_METHODS:
            if method.lower() in col.lower():
                methods_found.add(method)
    return sorted(methods_found)

def detect_precinct_column(headers):
    """
    Detects the column index that likely corresponds to precinct names.
    """
    for i, col in enumerate(headers):
        for keyword in COMMON_PRECINCT_HEADERS:
            if keyword.lower() in col.lower():
                return i
    return None

def clean_candidate_name(name):
    """
    Standardizes spacing and casing for candidate names.
    """
    return re.sub(r"\s+", " ", name.strip()).title()

def is_contest_row(text, min_length=15):
    """
    Heuristic: Contest rows are typically longer, capitalized, and have few numbers.
    """
    if len(text) < min_length:
        return False
    if re.search(r"\d", text):
        return False
    if text.isupper() or text.istitle():
        return True
    return False

def parse_numeric(val):
    """
    Cleans commas and parses numbers safely.
    """
    try:
        return int(val.replace(",", ""))
    except Exception:
        return None
# (Core utility, not for direct use in handlers)
def _find_and_click_toggle(
    search_root,
    selectors,
    keywords,
    logger=None,
    verbose=False,
    wait_selector=None,
    heading_match=None,
    heading_tags=None,
    max_heading_level=20,
    screenshot_on_fail=False,
    screenshot_prefix="toggle_fail",
    page=None
):
    """
    Core utility to find and click a toggle/button/link.
    - selectors: list of CSS selectors to try
    - keywords: list of keywords to match (in text or aria-label)
    - heading_match: if provided, only click if inside a parent with a heading matching this string
    - heading_tags: list of heading tags to check (e.g. ["h1", "h2", "p-span"])
    - max_heading_level: max hN to check if heading_tags not provided
    - screenshot_on_fail: if True, saves screenshot on failure
    - page: playwright page object (for screenshot)
    Returns True if clicked, False otherwise.
    """
    import difflib
    import re

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
                    heading_text = ""
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
                    if wait_selector and page:
                        try:
                            page.wait_for_selector(wait_selector, timeout=3000)
                        except Exception as e:
                            if logger:
                                logger.warning(f"[TOGGLE] Wait for selector '{wait_selector}' failed: {e}")
                    return True
            except Exception as e:
                if logger:
                    logger.debug(f"[TOGGLE] Error checking toggle: {e}")
                continue
    if logger:
        logger.warning(f"[TOGGLE] No toggle found for selectors={selectors}, keywords={keywords}, heading_match={heading_match}")
    if screenshot_on_fail and page:
        fname = f"{screenshot_prefix}_{'_'.join(keywords)}.png"
        page.screenshot(path=fname)
        if logger:
            logger.warning(f"[TOGGLE] Screenshot saved: {fname}")
    return False
def click_toggles_with_url_check(
    page,
    container,
    keyword_sets,
    logger=None,
    verbose=False,
    screenshot_on_fail=True,
    wait_selector=None,
    max_retries=2
):
    """
    Tries all keyword sets on all common selectors in the given container.
    Returns a list of (clicked, url_before, url_after) for each keyword set.
    """
    selectors = ["button", "p-togglebutton", "a", "[role='button']", "div[tabindex]", "span[tabindex]"]
    results = []
    for keywords in keyword_sets:
        url_before = page.url
        toggled = False
        for attempt in range(max_retries):
            toggled = _find_and_click_toggle(
                search_root=container,
                selectors=selectors,
                keywords=keywords,
                logger=logger,
                verbose=verbose,
                wait_selector=wait_selector,
                screenshot_on_fail=screenshot_on_fail,
                screenshot_prefix="toggle_fail",
                page=page
            )
            if toggled:
                break
        url_after = page.url
        results.append((toggled, url_before, url_after))
    return results

def click_contest_toggle_dynamic_heading(
    page,
    link_text,
    contest_title,
    panel_selector="p-panel",
    heading_tags=None,
    max_heading_level=20,
    extra_heading_tags=None,
    logger=None,
    verbose=False,
    wait_selector=None
):
    """
    Clicks a link/button with given text, but only within a panel whose heading matches contest_title.
    """
    selectors = [
        f"{panel_selector} a:has-text('{link_text}')",
        f"{panel_selector} button:has-text('{link_text}')"
    ]
    # Compose heading tags list
    tags = heading_tags or [f"h{n}" for n in range(1, max_heading_level+1)]
    if extra_heading_tags:
        tags += extra_heading_tags
    return _find_and_click_toggle(
        search_root=page,
        selectors=selectors,
        keywords=[link_text],
        logger=logger,
        verbose=verbose,
        wait_selector=wait_selector,
        heading_match=contest_title,
        heading_tags=tags,
        page=page
    )
    
def click_vote_method_toggle(
    page,
    keywords=None,
    logger=None,
    verbose=False,
    wait_selector=None,
    container=None
):
    """
    Finds and clicks a 'Vote Method' toggle/button/link.
    """
    if keywords is None:
        keywords = ["Vote Method", "Voting Method", "Show Vote Methods", "Display Vote Methods"]
    selectors = ["button", "a", "[role='button']", "div[tabindex]", "span[tabindex]"]
    search_root = container if container else page
    return _find_and_click_toggle(
        search_root=search_root,
        selectors=selectors,
        keywords=keywords,
        logger=logger,
        verbose=verbose,
        wait_selector=wait_selector,
        page=page
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
def extract_precincts(page, precinct_col_index):
    """
    Extracts precinct names from a table based on the detected precinct column index.
    """
    precincts = []
    try:
        rows = page.query_selector_all("tbody tr")
        for row in rows:
            cells = row.query_selector_all("td")
            if len(cells) > precinct_col_index:
                precinct_name = cells[precinct_col_index].inner_text().strip()
                if precinct_name and not any(char.isdigit() for char in precinct_name):
                    precincts.append(precinct_name)
    except Exception as e:
        logger.debug(f"[PRECINCT] Error extracting precincts: {e}")
    return precincts

def build_precinct_reporting_lookup(page, indicators=None):
    """
    Extracts a dictionary mapping precinct titles to reporting percentages.
    """
    if indicators is None:
        indicators = ["fully reported", "complete", "reported 100%"]

    lookup = {}
    panels = page.query_selector_all("div.p-panel-footer")
    if not panels:
        logger.debug("[LOOKUP] No reporting panels found.")
        return lookup
    logger.info(f"[LOOKUP] Found {len(panels)} reporting panels.")
    # Iterate through each panel to extract title and reporting percentage
    for panel in panels:
        # Attempt to find the closest parent panel element
        # and extract the title and reporting percentage
        try:
            parent = panel.evaluate_handle("el => el.closest('p-panel')")
            if not parent:
                logger.debug("[LOOKUP] No parent panel found.")
                continue
            # Extract the title from the closest parent panel
            # and the reporting percentage from the current panel
            title_element = None
            for i in range(1, 25):  # Checks h1 through h24
                if not title_element:
                    title_element = parent.query_selector(f"h{i}")
            title = title_element.inner_text().strip() if title_element else None
            # Extract the reporting percentage from the panel
            # This assumes the reporting percentage is in a <span> element
            span = panel.query_selector("span")
            if not span:
                logger.debug("[LOOKUP] No span found in reporting panel.")
                continue
            # Extract the text from the span and check for indicators
            # If title is found, check for indicators in the text
            # If title is not found, skip this panel
            # Normalize the text for comparison
            # This assumes the text is in lowercase and stripped of whitespace
            text = span.inner_text().strip().lower() if span else ""
            if not text:
                logger.debug("[LOOKUP] No text found in reporting panel.")
                continue
            if title:
                title = title.lower()
                # Check for indicators in the text
                # If any indicator is found, set the percentage to 100%
                # Otherwise, extract the percentage using regex
                if any(indicator in text for indicator in indicators):
                    pct = "100.00%"
                else:
                    match = re.search(r"([\d.]+%)", text)
                    if match:
                        pct = match.group(1)
                    else:
                        logger.debug("[LOOKUP] No percentage found in reporting panel.")
                        continue
            else:
                # If title is not found, extract the percentage using regex
                match = re.search(r"([\d.]+%)", text)
                if not match:
                    logger.debug("[LOOKUP] No percentage found in reporting panel.")
                    continue
                # If no title is found, set the percentage to 0.00%
                # This assumes the text is in lowercase and stripped of whitespace
                # This is a fallback in case the title is not found
                # or if the title is not descriptive enough
                pct = match.group(1) if match else "0.00%"
                lookup[title.lower()] = pct
            # Normalize the title for comparison
            title = re.sub(r"\s+", " ", title).strip()
            # Check if the title is already in the lookup
            if title in lookup:
                # If the title is already in the lookup, skip this panel
                logger.debug(f"[LOOKUP] Title '{title}' already exists in lookup.")
                continue
            # Add the title and percentage to the lookup
            lookup[title] = pct
        except Exception as e:
            logger.debug(f"[LOOKUP] Error parsing reporting panel: {e}")
            continue
    return lookup

def parse_candidate_vote_table(table_element, current_precinct, method_names, reporting_pct="0.00%"):
    """
    Converts a DOM table element into a Smart Elections-style row for a single precinct.

    Args:
        table_element: Playwright DOM element (the <table>)
        current_precinct (str): Name of the precinct currently being processed.
        method_names (List[str]): List of vote methods detected.
        reporting_pct (str): The percentage reporting for this precinct.

    Returns:
        Dict[str, str]: Row with standardized candidate-method vote fields and metadata.

    Note:
        This assumes a structure where the first column is the candidate name and last column is total votes.
        Handlers are expected to adapt the parsing if structure differs.
    """
    row = {"Precinct": current_precinct, "% Precincts Reporting": reporting_pct}
    try:
        headers = table_element.query_selector_all('thead tr th')
        rows = table_element.query_selector_all('tbody tr')
        colnames = [h.inner_text().strip() for h in headers]

        for r in rows:
            cells = r.query_selector_all('td')
            if len(cells) < 2:
                continue
            full_name = cells[0].inner_text().strip()
            name_parts = full_name.split()
            if len(name_parts) >= 3:
                # Assuming the last part is the party and the rest is the candidate name
                # This is a heuristic and may need adjustment based on actual data
                candidate_name = " ".join(name_parts[1:-1])
                party = name_parts[-1]
                # Normalize candidate name and party
                candidate_name = clean_candidate_name(candidate_name)
                party = party.strip().upper()
                # Create a canonical name for the candidate
                # This is a heuristic and may need adjustment based on actual data
                if party in ["DEM", "REP", "IND", "LIB", "GRE"]:
                    # Assuming party is a 3-letter abbreviation
                    # This is a heuristic and may need adjustment based on actual data
                    party = party.title()
                canonical = f"{candidate_name} ({party})"
            elif len(name_parts) == 2:
                # Assuming the last part is the party and the first part is the candidate name
                # This is a heuristic and may need adjustment based on actual data
                candidate_name = name_parts[0]
                party = name_parts[1]
                # Normalize candidate name and party
                candidate_name = clean_candidate_name(candidate_name)
                party = party.strip().upper()
                # Create a canonical name for the candidate
                # This is a heuristic and may need adjustment based on actual data
                if party in ["DEM", "REP", "IND", "LIB", "GRE"]:
                    # Assuming party is a 3-letter abbreviation
                    # This is a heuristic and may need adjustment based on actual data
                    party = party.title()
                canonical = f"{candidate_name} ({party})"
            elif len(name_parts) == 1:
                # Assuming the candidate name is a single word
                # This is a heuristic and may need adjustment based on actual data
                candidate_name = name_parts[0]
                party = ""
                # Normalize candidate name and party
                candidate_name = clean_candidate_name(candidate_name)
                party = party.strip().upper()
                # Create a canonical name for the candidate
                # This is a heuristic and may need adjustment based on actual data
                if party in ["DEM", "REP", "IND", "LIB", "GRE"]:
                    # Assuming party is a 3-letter abbreviation
                    # This is a heuristic and may need adjustment based on actual data
                    party = party.title()
                canonical = f"{candidate_name} ({party})"
            elif len(name_parts) == 0:
                # If no name parts are found, skip this row
                logger.debug("[TABLE] No name parts found for candidate.")
                continue
            elif len(name_parts) == 1 and not name_parts[0].isdigit():
                # If only one name part is found and it's not a digit, use it as the candidate name
                candidate_name = name_parts[0]
                party = ""
                # Normalize candidate name and party
                candidate_name = clean_candidate_name(candidate_name)
                party = party.strip().upper()
                # Create a canonical name for the candidate
                # This is a heuristic and may need adjustment based on actual data
                if party in ["DEM", "REP", "IND", "LIB", "GRE"]:
                    # Assuming party is a 3-letter abbreviation
                    # This is a heuristic and may need adjustment based on actual data
                    party = party.title()
                canonical = f"{candidate_name} ({party})"
            elif len(name_parts) == 1 and name_parts[0].isdigit():
                # If only one name part is found and it's a digit, use it as the precinct name
                # This is a heuristic and may need adjustment based on actual data
                candidate_name = name_parts[0]
                party = ""
                # Normalize candidate name and party
                candidate_name = clean_candidate_name(candidate_name)
                party = party.strip().upper()
                # Create a canonical name for the candidate
                # This is a heuristic and may need adjustment based on actual data
                if party in ["DEM", "REP", "IND", "LIB", "GRE"]:
                    # Assuming party is a 3-letter abbreviation
                    # This is a heuristic and may need adjustment based on actual data
                    party = party.title()
                canonical = f"{candidate_name} ({party})"
            elif len(name_parts) == 0 and cells[0].inner_text().strip() == "Total":
                # If the first cell is "Total", skip this row
                logger.debug("[TABLE] Skipping 'Total' row.")
                continue
            elif len(name_parts) == 0 and cells[0].inner_text().strip() == "Total Votes":
                # If the first cell is "Total Votes", skip this row
                logger.debug("[TABLE] Skipping 'Total Votes' row.")
                continue
            elif len(name_parts) == 0 and cells[0].inner_text().strip() == "Total Ballots":
                # If the first cell is "Total Ballots", skip this row
                logger.debug("[TABLE] Skipping 'Total Ballots' row.")
                continue
            elif len(name_parts) == 0 and cells[0].inner_text().strip() == "Total Votes Cast":
                # If the first cell is "Total Votes Cast", skip this row
                logger.debug("[TABLE] Skipping 'Total Votes Cast' row.")
                continue
            elif len(name_parts) == 0 and cells[0].inner_text().strip() == "Total Ballots Cast":
                # If the first cell is "Total Ballots Cast", skip this row
                logger.debug("[TABLE] Skipping 'Total Ballots Cast' row.")
                continue
            elif len(name_parts) == 0 and cells[0].inner_text().strip() == "Total Votes Counted":
                # If the first cell is "Total Votes Counted", skip this row
                logger.debug("[TABLE] Skipping 'Total Votes Counted' row.")
                continue    
            elif len(name_parts) == 0 and cells[0].inner_text().strip() == "Total Ballots Counted":
                # If the first cell is "Total Ballots Counted", skip this row
                logger.debug("[TABLE] Skipping 'Total Ballots Counted' row.")
                continue
            elif len(name_parts) == 0 and cells[0].inner_text().strip() == "Total Votes Remaining":
                # If the first cell is "Total Votes Remaining", skip this row
                logger.debug("[TABLE] Skipping 'Total Votes Remaining' row.")
                continue
            elif len(name_parts) == 0 and cells[0].inner_text().strip() == "Total Ballots Remaining":
                # If the first cell is "Total Ballots Remaining", skip this row
                logger.debug("[TABLE] Skipping 'Total Ballots Remaining' row.")
                continue    
            elif len(name_parts) == 0 and cells[0].inner_text().strip() == "Total Votes Outstanding":
                # If the first cell is "Total Votes Outstanding", skip this row
                logger.debug("[TABLE] Skipping 'Total Votes Outstanding' row.")
                continue
            elif len(name_parts) == 0 and cells[0].inner_text().strip() == "Total Ballots Outstanding":
                # If the first cell is "Total Ballots Outstanding", skip this row
                logger.debug("[TABLE] Skipping 'Total Ballots Outstanding' row.")
                continue
            elif len(name_parts) == 0 and cells[0].inner_text().strip() == "Total Votes Uncounted":
                # If the first cell is "Total Votes Uncounted", skip this row
                logger.debug("[TABLE] Skipping 'Total Votes Uncounted' row.")
                continue
            elif len(name_parts) == 0 and cells[0].inner_text().strip() == "Total Ballots Uncounted":
                # If the first cell is "Total Ballots Uncounted", skip this row
                logger.debug("[TABLE] Skipping 'Total Ballots Uncounted' row.")
                continue
            elif len(name_parts) == 0 and cells[0].inner_text().strip() == "Total Votes Disputed":
                # If the first cell is "Total Votes Disputed", skip this row
                logger.debug("[TABLE] Skipping 'Total Votes Disputed' row.")
                continue
            elif len(name_parts) == 0 and cells[0].inner_text().strip() == "Total Ballots Disputed":
                # If the first cell is "Total Ballots Disputed", skip this row
                logger.debug("[TABLE] Skipping 'Total Ballots Disputed' row.")
                continue
            elif len(name_parts) == 0 and cells[0].inner_text().strip() == "Total Votes Invalid":
                # If the first cell is "Total Votes Invalid", skip this row
                logger.debug("[TABLE] Skipping 'Total Votes Invalid' row.")
                continue
            elif len(name_parts) == 0 and cells[0].inner_text().strip() == "Total Ballots Invalid":
                # If the first cell is "Total Ballots Invalid", skip this row
                logger.debug("[TABLE] Skipping 'Total Ballots Invalid' row.")
                continue
            elif len(name_parts) == 0 and cells[0].inner_text().strip() == "Total Votes Spoiled":
                # If the first cell is "Total Votes Spoiled", skip this row
                logger.debug("[TABLE] Skipping 'Total Votes Spoiled' row.")
                continue
            elif len(name_parts) == 0 and cells[0].inner_text().strip() == "Total Ballots Spoiled":
                # If the first cell is "Total Ballots Spoiled", skip this row
                logger.debug("[TABLE] Skipping 'Total Ballots Spoiled' row.")
                continue
            elif len(name_parts) == 0 and cells[0].inner_text().strip() == "Total Votes Rejected":
                # If the first cell is "Total Votes Rejected", skip this row
                logger.debug("[TABLE] Skipping 'Total Votes Rejected' row.")
                continue
            elif len(name_parts) == 0 and cells[0].inner_text().strip() == "Total Ballots Rejected":
                # If the first cell is "Total Ballots Rejected", skip this row
                logger.debug("[TABLE] Skipping 'Total Ballots Rejected' row.")
                continue
            elif len(name_parts) == 0 and cells[0].inner_text().strip() == "Total Votes Canceled":
                # If the first cell is "Total Votes Canceled", skip this row
                logger.debug("[TABLE] Skipping 'Total Votes Canceled' row.")
                continue
            elif len(name_parts) == 0 and cells[0].inner_text().strip() == "Total Ballots Canceled":
                # If the first cell is "Total Ballots Canceled", skip this row
                logger.debug("[TABLE] Skipping 'Total Ballots Canceled' row.")
                continue
            elif len(name_parts) == 0 and cells[0].inner_text().strip() == "Total Votes Disqualified":
                # If the first cell is "Total Votes Disqualified", skip this row
                logger.debug("[TABLE] Skipping 'Total Votes Disqualified' row.")
                continue
            elif len(name_parts) == 0 and cells[0].inner_text().strip() == "Total Ballots Disqualified":
                # If the first cell is "Total Ballots Disqualified", skip this row
                logger.debug("[TABLE] Skipping 'Total Ballots Disqualified' row.")
                continue
            elif len(name_parts) == 0 and cells[0].inner_text().strip() == "Total Votes Nullified":
                # If the first cell is "Total Votes Nullified", skip this row
                logger.debug("[TABLE] Skipping 'Total Votes Nullified' row.")
                continue
            elif len(name_parts) == 0 and cells[0].inner_text().strip() == "Total Ballots Nullified":
                # If the first cell is "Total Ballots Nullified", skip this row
                logger.debug("[TABLE] Skipping 'Total Ballots Nullified' row.")
                continue
            elif len(name_parts) == 0 and cells[0].inner_text().strip() == "Total Votes Voided":
                # If the first cell is "Total Votes Voided", skip this row
                logger.debug("[TABLE] Skipping 'Total Votes Voided' row.")
                continue
            elif len(name_parts) == 0 and cells[0].inner_text().strip() == "Total Ballots Voided":
                # If the first cell is "Total Ballots Voided", skip this row
                logger.debug("[TABLE] Skipping 'Total Ballots Voided' row.")
                continue
            else:
                canonical = full_name
            # Normalize the canonical name
            canonical = re.sub(r"\s+", " ", canonical).strip()
            # Check if the canonical name is already in the row
            if canonical in row:
                # If the canonical name is already in the row, skip this candidate
                logger.debug(f"[TABLE] Candidate '{canonical}' already exists in row.")
                continue
            # Add the canonical name to the row
            row[canonical] = full_name  
            # Extract the vote counts for each method
            # This assumes the last cell is the total votes
            # and the rest are the method votes
            # This is a heuristic and may need adjustment based on actual data
            # Use inner_text() to get the text from each cell
            # and strip any leading/trailing whitespace
            

            method_votes = [c.inner_text().strip() for c in cells[1:-1]]
            # Check if the number of method votes matches the number of method names
            if len(method_votes) != len(method_names):
                logger.debug(f"[TABLE] Number of method votes ({len(method_votes)}) does not match number of method names ({len(method_names)}).")
                continue
            # Extract the total votes from the last cell
            # This assumes the last cell is the total votes
            total = cells[-1].inner_text().strip()
            # Check if the total is a valid number
            if not re.match(r"^\d+(\.\d+)?$", total.replace(",", "").replace("-", "")):
                logger.debug(f"[TABLE] Total '{total}' is not a valid number.")
                continue
            # Normalize the total votes
            total = total.replace(",", "").replace("-", "0")
            # Add the method votes and total to the row
            # This assumes the method votes are in the same order as the method names
            # This is a heuristic and may need adjustment based on actual data
            for i, method in enumerate(method_names):   
                # Check if the method is already in the row
                if method in row:
                    # If the method is already in the row, skip this candidate
                    logger.debug(f"[TABLE] Method '{method}' already exists in row.")
                    continue
                # Add the method vote to the row
                # This assumes the method votes are in the same order as the method names
                # This is a heuristic and may need adjustment based on actual data
                vote = method_votes[i].replace(",", "").replace("-", "0")
                # Check if the vote is a valid number
                if not re.match(r"^\d+(\.\d+)?$", vote):
                    logger.debug(f"[TABLE] Vote '{vote}' is not a valid number.")
                    continue
                # Normalize the vote
                vote = vote.replace(",", "").replace("-", "0")
                # Add the method vote to the row
                row[f"{canonical} - {method}"] = vote
            # Add the total to the row
            # This assumes the total is a valid number  
            # This is a heuristic and may need adjustment based on actual data
            # Check if the total is a valid number
            if not re.match(r"^\d+(\.\d+)?$", total.replace(",", "").replace("-", "")):
                logger.debug(f"[TABLE] Total '{total}' is not a valid number.")
                continue
            # Normalize the total votes
            total = total.replace(",", "").replace("-", "0")
            # Add the total to the row
            # This assumes the total is a valid number
            # This is a heuristic and may need adjustment based on actual data
            # Check if the total is already in the row  
            if f"{canonical} - Total" in row:
                # If the total is already in the row, skip this candidate
                logger.debug(f"[TABLE] Total '{total}' already exists in row.")
                continue

            for method, vote in zip(method_names, method_votes):
                row[f"{canonical} - {method}"] = vote
            row[f"{canonical} - Total"] = total
    except Exception as e:
        logger.error(f"[TABLE] Failed to parse candidate vote table: {e}")
    return row

# Future additions:
# - Regex patterns for year/election name detection
# - State abbreviation helpers (NY, AZ, PA, etc.)
# - Vote total reconciliation checks
# - Ballot method synonyms or normalization
