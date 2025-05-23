from playwright.sync_api import Page
from .....handlers.formats import json_handler
from .....handlers.formats.html_handler import extract_contest_panel, extract_precinct_tables
from .....utils.contest_selector import select_contest
from .....utils.html_scanner import scan_html_for_context, get_detected_races_from_context
from .....utils.table_builder import (
    extract_table_data,
    parse_candidate_vote_table,
    calculate_grand_totals,
)
from .....utils.output_utils import finalize_election_output
from .....utils.shared_logger import logging, logger
from .....utils.shared_logic import (
    autoscroll_until_stable,
    click_dynamic_toggle,
)
import os
from dotenv import load_dotenv
from rich import print as rprint

load_dotenv()
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
VERBOSE = LOG_LEVEL == "DEBUG"
logging.basicConfig(level=LOG_LEVEL)

from typing import Optional, Tuple, List, Dict, Any

def clean_and_finalize(headers: List[str], data: List[dict], contest_title: str, metadata: dict) -> Tuple[List[str], List[dict], str, dict]:
    """
    Cleans headers, appends grand totals, writes output, and returns all.
    """
    # Remove duplicate headers, sort for consistency
    headers = sorted(set(headers))
    # Remove empty or all-NA rows
    data = [row for row in data if any(str(v).strip() for v in row.values())]
    # Append grand totals row
    grand_total = calculate_grand_totals(data)
    data.append(grand_total)
    # Recompute headers in case grand_total added new fields
    headers = sorted(set().union(*(row.keys() for row in data)))
    # Write output and metadata
    metadata["output_file"] = finalize_election_output(headers, data, contest_title, metadata).get("csv_path")
    return headers, data, contest_title, metadata

def parse(page: Page, html_context: Optional[dict] = None) -> Tuple[List[str], List[dict], str, dict]:
    """
    Main entry point for Rockland County handler.
    Always returns cleaned, normalized data and metadata.
    """
    if html_context is None:
        html_context = {}
    if "__root" not in html_context:
        html_context["__root"] = True  # Mark this as the root call

    rprint("[bold cyan][Rockland Handler] Parsing Rockland County Enhanced Voting page...[/bold cyan]")

    # --- JSON FLOW ---
    if html_context.get("json_source"):
        if VERBOSE:
            rprint("[blue][INFO] JSON source detected. Routing to JSON parser.[/blue]")
        result = json_handler.parse(page, html_context)
        if not result or (isinstance(result, tuple) and all(v is None for v in result)):
            rprint("[red][ERROR] JSON parse failed or returned no data. Skipping further processing.[/red]")
            return None, None, None, {"skipped": True}
        if VERBOSE:
            rprint("[green][INFO] JSON parse successful. Cleaning and finalizing results.[/green]")
        headers, data, contest_title, metadata = result
        return clean_and_finalize(headers, data, contest_title, metadata)

    # --- HTML FLOW ---
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
                # Return the first result (or aggregate as needed)
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
    handler_keywords = [
        "View results by election district",
        "View Results",
        "Results by District"
    ]
    contest_toggle_clicked = click_dynamic_toggle(
        page,
        container=contest_panel,
        handler_keywords=handler_keywords,
        logger=logger,
        verbose=VERBOSE,
        interactive=True
    )

    if not contest_toggle_clicked:
        rprint("[yellow][WARN] Contest-specific toggle not found for this contest.[/yellow]")
        # Diagnostic: List all clickable elements in the contest panel
        if contest_panel and hasattr(contest_panel, "locator"):
            clickable = contest_panel.locator("button, a, [role='button'], div[tabindex], span[tabindex]")
            rprint("[yellow][DEBUG] Clickable elements in contest panel:")
            for i in range(clickable.count()):
                try:
                    text = clickable.nth(i).inner_text().strip()
                    rprint(f"  - {text}")
                except Exception:
                    continue

    # 2. Click "Vote Method" toggle (if present) in the contest panel
    vote_method_clicked = click_dynamic_toggle(
        page,
        container=contest_panel,
        handler_keywords=["Vote Method", "Voting Method", "Ballot Method"],
        logger=logger,
        verbose=VERBOSE,
        interactive=True
    )
    if vote_method_clicked:
        rprint("[cyan][INFO] Vote method toggle clicked.[/cyan]")
    else:
        rprint("[yellow][WARN] Vote method toggle not found in contest panel.[/yellow]")
        # Diagnostic: List all clickable elements in the contest panel
        if contest_panel and hasattr(contest_panel, "locator"):
            clickable = contest_panel.locator("button, a, [role='button'], div[tabindex], span[tabindex]")
            rprint("[yellow][DEBUG] Clickable elements in contest panel:")
            for i in range(clickable.count()):
                try:
                    text = clickable.nth(i).inner_text().strip()
                    rprint(f"  - {text}")
                except Exception:
                    continue
        # Check for "No Results" message
        no_results = page.locator("text=No results").count()
        if no_results > 0:
            rprint("[red][ERROR] No results found on the page. Skipping further processing.[/red]")
            return None, None, None, {"skipped": True}

    # Scroll page to load dynamic precincts
    rprint("[cyan][INFO] Scrolling to load precincts...[/cyan]")
    autoscroll_until_stable(page)

    # Extract contest panel and precinct tables
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
            headers, _ = extract_table_data(table)
            if len(headers) > 2:
                method_names = headers[1:-1]
            else:
                method_names = []

        # Use table_builder to parse and clean the row
        row = parse_candidate_vote_table(table, precinct_name, method_names)
        if not row:
            rprint(f"[red][ERROR] No data parsed for precinct '{precinct_name}'. Skipping.[/red]")
            continue
        data.append(row)

    if not data:
        rprint("[red][ERROR] No precinct data was parsed.[/red]")
        return None, None, None, {"skipped": True}

    # Assemble headers from union of all rows
    headers = sorted(set().union(*(row.keys() for row in data)))

    metadata = {
        "state": "NY",
        "county": "Rockland",
        "race": contest_title,
        "source": page.url,
        "handler": "rockland"
    }

    # Clean, finalize, and write output
    return clean_and_finalize(headers, data, contest_title, metadata)