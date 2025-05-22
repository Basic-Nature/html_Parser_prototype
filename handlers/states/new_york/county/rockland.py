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
from utils.contest_selector import select_contest
from utils.html_scanner import scan_html_for_context, get_detected_races_from_context
from utils.output_utils import calculate_grand_totals, finalize_election_output
from utils.shared_logger import logging, logger
from utils.shared_logic import (
    autoscroll_until_stable,
    click_contest_toggle_dynamic_heading,
    click_vote_method_toggle,
    build_precinct_reporting_lookup,
    parse_candidate_vote_table,
    
)
import os

from dotenv import load_dotenv
from rich import print as rprint


load_dotenv()
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
VERBOSE = LOG_LEVEL == "DEBUG"
logging.basicConfig(level=LOG_LEVEL)

from typing import Optional


def parse(page: Page, html_context: Optional[dict] = None):
    if html_context is None:
        html_context = {}
    if "__root" not in html_context:
        html_context["__root"] = True  # Mark this as the root call
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
    contest_title = html_context.get("selected_race")
    if not contest_title:
        if html_context.get("__root", False):  # Only show UI on root call
            rprint("[cyan]Preparing contest selections. Please wait...[/cyan]")
            context = scan_html_for_context(page)
            detected = get_detected_races_from_context(context)
            logger.debug(f"[DEBUG] All detected races: {detected}")
            selected = select_contest(detected)
            if not selected:
                rprint("[red]No contest selected. Skipping.[/red]")
                return None, None, None, {"skipped": True}
            if isinstance(selected, list):
                # If multiple contests selected, process each one by title
                results = []
                for contest_tuple in selected:
                    contest_title = contest_tuple[2]  # third element is the contest title
                    html_context_copy = dict(html_context)
                    html_context_copy["selected_race"] = contest_title
                    html_context_copy.pop("__root", None)  # Remove root flag for recursion
                    result = parse(page, html_context_copy)
                    results.append(result)
                return results[0] if results else (None, None, None, {"skipped": True})
            else:
                # Single contest selected
                contest_title = selected[2]  # third element is the contest title
                html_context["selected_race"] = contest_title
                html_context.pop("__root", None)
                return parse(page, html_context)
        else:
            # If not root, just skip (should not happen)
            return None, None, None, {"skipped": True}
    race_core = contest_title.split("2024")[-1].strip().lower()
    if "view results by election district" in race_core:
        rprint("[red]Selected entry is a navigation link, not a real contest. Skipping.[/red]")
        return None, None, None, {"skipped": True}
    if VERBOSE:
        rprint(f"[blue][DEBUG] Rockland handler received selected race: '{contest_title}'[/blue]")

    page.wait_for_timeout(500)
    rprint("[cyan][INFO] Waiting for contest-specific page content...[/cyan]")
    # Wait for the contest panel to load
    contest_panel = extract_contest_panel(page, contest_title)
    if not contest_panel:
        rprint("[red][ERROR] Contest panel not found. Skipping further processing.[/red]")
        return None, None, None, {"skipped": True}
        
    # --- Robust contest-specific toggle, then dynamic vote method toggle ---

    # 1. Click "View results by election district" ONLY for the selected contest
    contest_toggle_clicked = click_contest_toggle_dynamic_heading(
        page,
        link_text="View results by election district",
        contest_title=contest_title,
        panel_selector="p-panel",  # adjust if your contest panels use a different tag
        extra_heading_tags=["p-span", "span"],  # add custom heading tags if needed
        logger=logger,
        verbose=VERBOSE
    )
    if contest_toggle_clicked:
        rprint("[cyan][INFO] Contest-specific toggle clicked.[/cyan]")
    else:
        rprint("[yellow][WARN] Contest-specific toggle not found for this contest.[/yellow]")

    # 2. Click "Vote Method" toggle (if present) in the contest panel
    vote_method_clicked = click_vote_method_toggle(
        page,
        keywords=["Vote Method", "Voting Method", "Ballot Method"],
        logger=logger,
        verbose=VERBOSE,
        container=contest_panel  # restrict search to the contest panel
    )
    if vote_method_clicked:
        rprint("[cyan][INFO] Vote method toggle clicked.[/cyan]")
    else:
        rprint("[yellow][WARN] Vote method toggle not found in contest panel.[/yellow]")
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
        if not precinct_name:
            rprint(f"[red][ERROR] No precinct name found for table. Skipping.[/red]")
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