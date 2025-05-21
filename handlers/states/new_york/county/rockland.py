# rockland.py
# ==============================================================================
# Rockland County (NY) parser for extracting election results from enhanced voting portals.
# Supports both HTML-based and JSON-based flows, contest-level selection,
# precinct-level parsing, method-wise vote breakdown, and Smart Elections output format.
# Outputs to a standardized folder with timestamped CSV and detailed metadata.
# ==============================================================================

from playwright.sync_api import Page
from handlers.formats import json_handler
from utils.output_utils import finalize_election_output
from utils.shared_logger import logging
from utils.shared_logic import (
    autoscroll_until_stable,
    click_vote_method_toggle,
    build_precinct_reporting_lookup,
    detect_precinct_headers,
    parse_candidate_vote_table,
    calculate_grand_totals
)
import os
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
    r"view results? by.*district",
    r"proposal number",
    r"proposition",
    r"^vote for \d+$"
]

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
    race_title = html_context.get("selected_race")
    race_core = race_title.split("2024:")[-1].strip().lower() if race_title else ""
    if "view results by election district" in race_core:
       rprint("[red]Selected entry is a navigation link, not a real contest. Skipping.[/red]")
       return None, None, None, {"skipped": True} 
    if not race_core:
        rprint("[yellow]No contest race selected. Re-launching contest list.[/yellow]")
        from handlers.formats import html_handler
        return html_handler.parse(page, html_context)
    if VERBOSE:
        rprint(f"[blue][DEBUG] Rockland handler received selected race: '{race_title}'[/blue]")
    
    page.wait_for_timeout(500)
    rprint("[cyan][INFO] Waiting for contest-specific page content...[/cyan]")
    links = page.locator("a:has-text('View results by election district')")
    if links.count() == 0:
        rprint("[red]No contest links found on the page.[/red]")
        return None, None, None, {"skipped": True}

    for i in range(links.count()):
        link = links.nth(i)
        try:
            section = link.locator("xpath=ancestor::p-panel[1]//h1")
            heading_text = section.inner_text().strip().lower() if section.is_visible() else ""
            link_text = (link.inner_text() or "").strip().lower()
            if race_core and (race_core in heading_text or race_core in link_text):
                rprint(f"[cyan][INFO] Matched contest link for: {race_title}[/cyan]")
                link.scroll_into_view_if_needed()
                page.wait_for_timeout(500)
                rprint(f"[cyan][INFO] URL before click: {page.url}[/cyan]")
                link.click(force=True, timeout=5000)
                page.wait_for_timeout(2500)
                rprint(f"[cyan][INFO] URL after click: {page.url}[/cyan]")
                break
        except Exception as e:
            if VERBOSE:
                rprint(f"[yellow][WARN] Skipped a link due to error: {e}[/yellow]")

    # Toggle Vote Method if available — AFTER confirming we're on the precinct detail page
    rprint("[cyan][INFO] Waiting for toggle to appear after contest view loads...[/cyan]")
    try:
        page.wait_for_selector("p-togglebutton", timeout=5000)
    except Exception:
        rprint("[yellow][WARN] Toggle button did not render within timeout. Proceeding anyway.[/yellow]")
    toggled = click_vote_method_toggle(page, keywords=["Vote Method", "Voting Method", "Ballot Method"])
    if not toggled:
        rprint("[yellow][WARN] Vote method toggle not found. Some columns may be missing.[/yellow]")
    page.wait_for_timeout(1000) # Allow time for toggle to settle
    # Check for "No Results" message
    no_results = page.locator("text=No results").count()
    if no_results > 0:
        rprint("[red][ERROR] No results found on the page. Skipping further processing.[/red]")
        return None, None, None, {"skipped": True}  
    no_results = page.locator("text=No results").count()
           
    # Scroll page to load dynamic precincts
    rprint("[cyan][INFO] Scrolling to load precincts...[/cyan]")
    autoscroll_until_stable(page)

    # Build reporting percentage lookup
    precinct_reporting_lookup = build_precinct_reporting_lookup(page)

    # Parse all precinct tables and build Smart Elections rows
    elements = page.query_selector_all('h3, strong, b, span, table')
    precinct_headers = detect_precinct_headers(elements)
    precinct_headers = [h for h in precinct_headers if "vote for" not in h.lower()]
    precinct_data = []
    method_names = []

    estimated = len(precinct_headers)
    progress = tqdm(total=estimated or 1, desc="Loading precinct results", unit="precinct", colour="#45818E", dynamic_ncols=True, leave=True, disable=not VERBOSE)

    current_precinct = None
    for el in elements:
        tag = el.evaluate("e => e.tagName").strip().upper()

        if tag in ["H3", "STRONG", "B", "SPAN"]:
            label = el.inner_text().strip()
            if label and "vote for" not in label.lower() and label in precinct_headers:
                current_precinct = label
                lookup_key = label.lower()
                reporting_pct = precinct_reporting_lookup.get(lookup_key, "0.00%")
                progress.update(1)

        elif tag == "TABLE" and current_precinct:
            reporting_pct = locals().get("reporting_pct", "0.00%")
            row = parse_candidate_vote_table(el, current_precinct, method_names, reporting_pct)
            precinct_data.append(row)
            current_precinct = None

    if not precinct_data:
        rprint("[red][ERROR] No precinct data was parsed.[/red]")
        return None, None, None, {"skipped": True}

    # Compute and append grand totals row
    grand_total = calculate_grand_totals(precinct_data)
    precinct_data.append(grand_total)

    # Assemble headers from union of all rows
    all_headers = sorted(set().union(*(row.keys() for row in precinct_data)))
    progress.n = progress.total
    progress.refresh()
    progress.close()

    metadata = {
        "state": "NY",
        "county": "Rockland",
        "race": race_title,
        "source": page.url,
        "handler": "rockland"
    }

    metadata["output_file"] = finalize_election_output(all_headers, precinct_data, metadata).get("csv_path")
    return  all_headers, precinct_data, race_title, metadata
# End of file

