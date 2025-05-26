from playwright.sync_api import Page
from .....utils.shared_logger import logger, rprint
from .....Context_Integration.context_organizer import organize_context
from .....utils.output_utils import finalize_election_output
from .....utils.table_builder import extract_table_data, calculate_grand_totals
from .....utils.html_scanner import scan_html_for_context, get_detected_races_from_context
from .....utils.contest_selector import select_contest

def parse(page: Page, html_context: dict = None):
    """
    Example county handler.
    Handles all contests/races for this county, using shared utilities.
    Returns (headers, data, contest_title, metadata).
    """
    if html_context is None:
        html_context = {}

    rprint("[bold cyan][Example County Handler] Parsing county results page...[/bold cyan]")

    # --- 1. Scan HTML for context and contests ---
    context = scan_html_for_context(page)
    html_context.update(context)
    state = html_context.get("state", "EX")
    county = html_context.get("county", "Example County")

    # --- 2. Contest selection ---
    detected_races = get_detected_races_from_context(html_context)
    selected = select_contest(detected_races)
    if not selected:
        rprint("[red]No contest selected. Skipping.[/red]")
        return None, None, None, {"skipped": True}

    # If multiple contests, process each (aggregate or return first)
    if isinstance(selected, list):
        results = []
        for contest_tuple in selected:
            contest_title = contest_tuple[2]  # (year, etype, race)
            html_context_copy = dict(html_context)
            html_context_copy["selected_race"] = contest_title
            result = parse_single_contest(page, html_context_copy, state, county)
            results.append(result)
        return results[0] if results else (None, None, None, {"skipped": True})
    else:
        contest_title = selected[2]
        html_context["selected_race"] = contest_title
        return parse_single_contest(page, html_context, state, county)

def parse_single_contest(page, html_context, state, county):
    """
    Parses a single contest (race) from the county page.
    """
    contest_title = html_context.get("selected_race")
    rprint(f"[cyan][INFO] Processing contest: {contest_title}[/cyan]")

    # --- Example: Find the contest panel and extract tables ---
    from .....handlers.formats.html_handler import extract_contest_panel, extract_precinct_tables

    contest_panel = extract_contest_panel(page, contest_title)
    if not contest_panel:
        rprint("[red][ERROR] Contest panel not found. Skipping.[/red]")
        return None, None, None, {"skipped": True}

    precinct_tables = extract_precinct_tables(contest_panel)
    data = []
    for precinct_name, table in precinct_tables:
        if not table or not precinct_name:
            continue
        headers, rows = extract_table_data(table)
        for row in rows:
            row["Precinct"] = precinct_name
            data.append(row)

    if not data:
        rprint("[red][ERROR] No precinct data was parsed.[/red]")
        return None, None, None, {"skipped": True}

    # Assemble headers and finalize output
    headers = sorted(set().union(*(row.keys() for row in data)))
    metadata = {
        "state": state,
        "county": county,
        "race": contest_title,
        "source": page.url,
        "handler": "example_county"
    }
    return finalize_and_output(headers, data, contest_title, metadata)

def finalize_and_output(headers, data, contest_title, metadata):
    """
    Cleans, finalizes, and writes output using shared output utilities.
    """
    # Remove empty or all-NA rows
    data = [row for row in data if any(str(v).strip() for v in row.values())]
    # Append grand totals row
    grand_total = calculate_grand_totals(data)
    data.append(grand_total)
    # Recompute headers in case grand_total added new fields
    headers = sorted(set().union(*(row.keys() for row in data)))
    # --- Enrich metadata and context ---
    organized = organize_context_with_cache(metadata)
    metadata = organized.get("metadata", metadata)
    # Write output and metadata
    result = finalize_election_output(headers, data, contest_title, metadata)
    contest_title = result.get("contest_title", contest_title)
    metadata = result.get("metadata", metadata)
    return headers, data, contest_title, metadata