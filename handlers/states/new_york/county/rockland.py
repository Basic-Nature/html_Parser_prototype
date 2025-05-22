# rockland.py
# ==============================================================================
# Rockland County (NY) parser for extracting election results from enhanced voting portals.
# Supports both HTML-based and JSON-based flows, contest-level selection,
# precinct-level parsing, method-wise vote breakdown, and Smart Elections output format.
# Outputs to a standardized folder with timestamped CSV and detailed metadata.
# ==============================================================================

from playwright.sync_api import Page
from handlers.formats import html_handler as fallback_html_handler
from handlers.formats import json_handler
from handlers.formats.html_handler import extract_contest_panel, extract_precinct_tables
from utils.html_scanner import get_detected_races_from_context
from utils.output_utils import finalize_election_output
from utils.shared_logger import logging
from utils.shared_logic import (
    autoscroll_until_stable,
    click_toggles_with_url_check,
    build_precinct_reporting_lookup,
    detect_precinct_headers,
    parse_candidate_vote_table,
    calculate_grand_totals
)
import os
import re
from dotenv import load_dotenv
from rich import print as rprint
from tqdm import tqdm

load_dotenv()
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
VERBOSE = LOG_LEVEL == "DEBUG"
logging.basicConfig(level=LOG_LEVEL)

from typing import Optional

NOISY_LABELS = [
    "view results by election district",
    "summary by method",
    "download",
    "vote method",
    "voting method",
]
NOISY_LABEL_PATTERNS = [
    r"view results? by election district\s*[:\n]?$",
    r"summary by method\s*[:\n]?$",
    r"download\s*[:\n]?$",
    r"vote method\s*[:\n]?$",
    r"voting method\s*[:\n]?$",
    r"^vote for \d+$"
]
NOISY_LABELS = [label.lower() for label in NOISY_LABELS]

def is_noisy_label(label: str) -> bool:
    """
    Check if a label is considered noisy based on predefined patterns.
    """
    label = label.lower()
    for pattern in NOISY_LABEL_PATTERNS:
        if re.search(pattern, label):
            return True
    # Check for exact matches with noisy labels
    if label in NOISY_LABELS:
        return True
    # Check for any noisy label in the string
    # This is a more lenient check, but it can be useful for certain cases
        
        
    return any(noisy in label for noisy in NOISY_LABELS)

def parse(page: Page, html_context: Optional[dict] = None):
    if html_context is None:
        html_context = {}
    rprint("[bold cyan][Rockland Handler] Parsing Rockland County Enhanced Voting page...[/bold cyan]")

    if html_context.get("json_source"):
        if VERBOSE:
            rprint("[blue][INFO] JSON source detected. Routing to JSON parser.[/blue]")
        result = json_handler.parse(page, html_context)
        if not result or (isinstance(result, tuple) and all(v is None for v in result)):
            rprint("[red][ERROR] JSON parse failed or returned no data. Skipping further processing.[/red]")
            return None, None, None, {"skipped": True}
        if VERBOSE:
            rprint("[green][INFO] JSON parse successful. Bypassing HTML and returning results.[/green]")
        return result
    detected = get_detected_races_from_context(html_context)
    filtered_races = [race for race in detected if not is_noisy_label(race)]
    if not filtered_races:
        rprint("[yellow]No valid contests detected after filtering. Skipping.[/yellow]")
        return None, None, None, {"skipped": True}
    contest_title = html_context.get("selected_race")
    race_core = contest_title.split("2024")[-1].strip().lower() if contest_title else ""
    if "view results by election district" in race_core:
        rprint("[red]Selected entry is a navigation link, not a real contest. Skipping.[/red]")
        return None, None, None, {"skipped": True}
    if not race_core:
        rprint("[yellow]No contest race selected. Re-launching contest list.[/yellow]")
        return fallback_html_handler.parse(page, html_context)
    if VERBOSE:
        rprint(f"[blue][DEBUG] Rockland handler received selected race: '{contest_title}'[/blue]")

    page.wait_for_timeout(500)
    rprint("[cyan][INFO] Waiting for contest-specific page content...[/cyan]")
    for idx, race in enumerate(filtered_races):
        rprint(f"  [{idx}] {race}")
    # --- Dynamic toggles: View results by election district, then Vote Method ---
    toggle_results = click_toggles_with_url_check(
        page,
        [
            ["View results by election district"],  # First toggle
            ["Vote Method", "Voting Method", "Ballot Method"],  # Second toggle
        ],
        logger=logging,
        verbose=VERBOSE
    )

    for idx, (clicked, url_before, url_after) in enumerate(toggle_results):
        if not clicked:
            rprint(f"[yellow][WARN] Toggle {idx+1} not found.[/yellow]")
        else:
            rprint(f"[cyan][INFO] Toggle {idx+1} clicked. URL before: {url_before} | after: {url_after}[/cyan]")

    # Check for "No Results" message
    no_results = page.locator("text=No results").count()
    if no_results > 0:
        rprint("[red][ERROR] No results found on the page. Skipping further processing.[/red]")
        return None, None, None, {"skipped": True}

    # Scroll page to load dynamic precincts
    rprint("[cyan][INFO] Scrolling to load precincts...[/cyan]")
    autoscroll_until_stable(page)

    # Build reporting percentage lookup
    # Build reporting percentage lookup
    precinct_reporting_lookup = build_precinct_reporting_lookup(page)
    contest_panel = extract_contest_panel(page, contest_title)
    if not contest_panel:
        rprint("[red][ERROR] Contest panel not found. Skipping further processing.[/red]")
        return None, None, None, {"skipped": True}

    precinct_tables = extract_precinct_tables(contest_panel)
    data = []
    method_names = None

    for precinct_name, table in precinct_tables:
        if not table:
            rprint(f"[red][ERROR] No table found for precinct '{precinct_name}'. Skipping.[/red]")
            continue
        if is_noisy_label(precinct_name):
            rprint(f"[yellow][WARN] Noisy label detected in precinct name: '{precinct_name}'. Skipping.[/yellow]")
            continue
        if VERBOSE:
            rprint(f"[blue][DEBUG] Parsing precinct: {precinct_name}[/blue]")

        # Extract method_names from table headers if not already set
        if method_names is None:
            headers = table.query_selector_all('thead tr th')
            method_names = [h.inner_text().strip() for h in headers][1:-1]  # skip first (candidate), last (total)

        reporting_pct = precinct_reporting_lookup.get(precinct_name.lower(), "0.00%")
        row = parse_candidate_vote_table(table, precinct_name, method_names, reporting_pct)
        if not row:
            rprint(f"[red][ERROR] No data parsed for precinct '{precinct_name}'. Skipping.[/red]")
            continue
        data.append(row)

    if not data:
        rprint("[red][ERROR] No precinct data was parsed.[/red]")
        return None, None, None, {"skipped": True}

    # Compute and append grand totals row
    grand_total = calculate_grand_totals(data)
    data.append(grand_total)

    # Assemble headers from union of all rows
    headers = sorted(set().union(*(row.keys() for row in data)))

    metadata = {
        "state": "NY",
        "county": "Rockland",
        "race": contest_title,
        "source": page.url,
        "handler": "rockland"
    }

    metadata["output_file"] = finalize_election_output(headers, data, contest_title, metadata).get("csv_path")
    return headers, data, contest_title, metadata
# End of file