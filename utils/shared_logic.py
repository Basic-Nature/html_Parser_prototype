##shared_logic.py - Common parsing utilities across states and formats.

##This module is designed to centralize shared patterns that repeat across many state or format handlers.
##While election website structures vary significantly by vendor or county/state implementation,
##many elements are consistent such as contest labeling, vote method naming, and tabular breakdowns.
##These helpers promote DRY principles and consistent behavior across handlers.

from utils.shared_logger import log_info, log_debug, log_warning, log_error
import re


# Common keyword mappings
COMMON_CONTEST_LABELS = [
    "President", "U.S. Senate", "U.S. House", "Governor", "Lieutenant Governor",
    "Attorney General", "State Senate", "State House", "Supreme Court",
    "Ballot Measure", "Constitutional Amendment", "Referendum", "Proposition"
]

COMMON_VOTE_METHODS = [
    "Election Day", "Early Voting", "Absentee", "Mail", "Provisional", "Total"
]

COMMON_PRECINCT_HEADERS = [
    "Precinct", "Ward", "District", "Voting District"
]

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


# === Precinct Parsing Utilities ===
def click_vote_method_toggle(page, keywords=None):
    """
    Attempts to locate and click a toggle button for showing vote method breakdowns.
    Supports common <button> elements as well as <p-togglebutton> with onlabel/offlabel.
    """
    if keywords is None:
        keywords = ["Vote Method", "Voting Details", "Show Breakdown", "Voting Method", "Ballot Method"]

    toggled = False

    # First: check standard button elements
    buttons = page.locator("button")
    for i in range(buttons.count()):
        btn = buttons.nth(i)
        try:
            label = btn.inner_text().strip()
            if any(k.lower() in label.lower() for k in keywords):
                btn.scroll_into_view_if_needed()
                btn.click()
                page.wait_for_timeout(1000)
                log_info(f"[TOGGLE] Button clicked: '{label}'")
                toggled = True
                break
        except Exception as e:
            log_debug(f"[TOGGLE] Button check failed: {e}")
            continue

    # If not found, fallback to p-togglebutton detection
    if not toggled:
        toggles = page.query_selector_all("p-togglebutton")
        for toggle in toggles:
            try:
                label = toggle.get_attribute("onlabel") or toggle.get_attribute("aria-label") or ""
                if any(k.lower() in label.lower() for k in keywords):
                    toggle.scroll_into_view_if_needed()
                    toggle.click(force=True)
                    page.wait_for_timeout(1000)
                    log_info(f"[TOGGLE] Custom toggle clicked via onlabel: '{label}'")
                    toggled = True
                    break
            except Exception as e:
                log_debug(f"[TOGGLE] Fallback toggle failed: {e}")
                continue

    if not toggled:
        log_warning("[TOGGLE] No matching toggle button found.")
    return toggled

def autoscroll_until_stable(page, max_stable_frames=5, step=3000, delay_ms=500):
    """
    Continuously scrolls a Playwright page until its scroll height stabilizes.
    Useful for dynamic election websites where all precinct data is only visible after scrolling.
    """
    page.evaluate("window.scrollTo(0, 0)")
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
    log_info("[SCROLL] Completed scrolling until page height stabilized.")

def build_precinct_reporting_lookup(page, indicators=None):
    """
    Extracts a dictionary mapping precinct titles to reporting percentages.
    """
    if indicators is None:
        indicators = ["fully reported", "complete", "reported 100%"]

    lookup = {}
    panels = page.query_selector_all("div.p-panel-footer")
    for panel in panels:
        try:
            parent = panel.evaluate_handle("el => el.closest('p-panel')")
            title_element = parent.query_selector("h1 span")
            title = title_element.inner_text().strip() if title_element else None
            span = panel.query_selector("span")
            text = span.inner_text().strip().lower() if span else ""
            if title:
                if any(indicator in text for indicator in indicators):
                    pct = "100.00%"
                else:
                    match = re.search(r"([\d.]+%)", text)
                    pct = match.group(1) if match else "0.00%"
                lookup[title.lower()] = pct
        except Exception as e:
            log_debug(f"[LOOKUP] Error parsing reporting panel: {e}")
            continue
    return lookup
  
def detect_precinct_headers(elements):
    """
    Scans a list of DOM elements and identifies potential precinct section headers.
    This includes H3, STRONG, SPAN, and B tags that contain keywords like 'Ward', 'District', etc.

    Args:
        elements (List): List of Playwright DOM nodes.

    Returns:
        List[str]: List of detected precinct names.

    Note:
        This function is only for identification. It does not attempt to infer reporting percentages.
    """
    precincts = []
    for el in elements:
        try:
            tag = el.evaluate("e => e.tagName").strip().upper()
            if tag in ["H3", "STRONG", "B", "SPAN"]:
                label = el.inner_text().strip()
                if any(k in label for k in COMMON_PRECINCT_HEADERS) or any(char.isdigit() for char in label):
                    precincts.append(label)
        except Exception as e:
            log_debug(f"[HEADER] Failed to evaluate tag or read label: {e}")
            continue
    return precincts


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
                candidate_name = " ".join(name_parts[1:-1])
                party = name_parts[-1]
                canonical = f"{candidate_name} ({party})"
            else:
                canonical = full_name

            method_votes = [c.inner_text().strip() for c in cells[1:-1]]
            total = cells[-1].inner_text().strip()

            for method, vote in zip(method_names, method_votes):
                row[f"{canonical} - {method}"] = vote
            row[f"{canonical} - Total"] = total
    except Exception as e:
        log_error(f"[TABLE] Failed to parse candidate vote table: {e}")
    return row


def calculate_grand_totals(rows):
    """
    Sums all numeric columns across a list of parsed precinct rows.

    Args:
        rows (List[Dict[str, str]]): List of Smart Elections-style rows.

    Returns:
        Dict[str, str]: A 'Grand Totals' row.

    Note:
        Skips fields like 'Precinct' and '% Precincts Reporting'.
    """
    totals = {}
    for row in rows:
        for k, v in row.items():
            if k in ["Precinct", "% Precincts Reporting"]:
                continue
            try:
                totals[k] = totals.get(k, 0) + int(v.replace(",", "").replace("-", "0"))
            except:
                continue
    totals["Precinct"] = "Grand Total"
    totals["% Precincts Reporting"] = ""
    return totals


# Future additions:
# - Regex patterns for year/election name detection
# - State abbreviation helpers (NY, AZ, PA, etc.)
# - Vote total reconciliation checks
# - Ballot method synonyms or normalization
