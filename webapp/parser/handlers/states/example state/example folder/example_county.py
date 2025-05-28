from playwright.sync_api import Page
from .....utils.logger_instance import logger
from .....utils.shared_logger import rprint
from .....utils.output_utils import finalize_election_output
from .....utils.table_builder import extract_table_data, calculate_grand_totals
from .....utils.html_scanner import scan_html_for_context
from .....utils.contest_selector import select_contest
from .....handlers.formats.html_handler import extract_contest_panel, extract_precinct_tables
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .....Context_Integration.context_coordinator import ContextCoordinator

def parse(
    page: Page,
    coordinator: "ContextCoordinator",
    html_context: dict = None,
    non_interactive: bool = False
):
    """
    Main entry point for Example County handler.
    - Scans HTML for context and contests
    - Lets user select contest(s)
    - Parses each contest and outputs results
    Returns (headers, data, contest_title, metadata) or a list of such tuples.
    """
    if html_context is None:
        html_context = {}

    rprint("[bold cyan][Example County Handler] Parsing county results page...[/bold cyan]")

    # 1. Scan HTML for context and update html_context
    context = scan_html_for_context(page)
    html_context.update(context)
    state = html_context.get("state", "EX")
    county = html_context.get("county", "Example County")

    # 2. Organize and enrich context with coordinator
    if coordinator is None:
        from .....Context_Integration.context_coordinator import ContextCoordinator
        coordinator = ContextCoordinator()
    if not getattr(coordinator, "organized", None):
        coordinator.organize_and_enrich(html_context)

    # 3. Contest selection using coordinator
    selected = select_contest(
        coordinator,
        state=state,
        county=county,
        year=html_context.get("year"),
        non_interactive=non_interactive
    )
    if not selected:
        rprint("[red]No contest selected. Skipping.[/red]")
        return None, None, None, {"skipped": True}

    # 4. If multiple contests, process each (aggregate or return first)
    if isinstance(selected, list):
        results = []
        for contest in selected:
            contest_title = contest.get("title") if isinstance(contest, dict) else contest
            html_context_copy = dict(html_context)
            html_context_copy["selected_race"] = contest_title
            result = parse_single_contest(page, html_context_copy, state, county, coordinator)
            results.append(result)
        return results[0] if results else (None, None, None, {"skipped": True})
    else:
        contest_title = selected.get("title") if isinstance(selected, dict) else selected
        html_context["selected_race"] = contest_title
        return parse_single_contest(page, html_context, state, county, coordinator)

def parse_single_contest(page, html_context, state, county, coordinator):
    """
    Parses a single contest (race) from the county page.
    If no contest panel is found, prompts user for manual correction.
    """
    contest_title = html_context.get("selected_race")
    rprint(f"[cyan][INFO] Processing contest: {contest_title}[/cyan]")

    contest_panel = extract_contest_panel(page, contest_title)
    if not contest_panel:
        rprint("[red][ERROR] Contest panel not found. Skipping.[/red]")
        # --- FEEDBACK UI: Prompt user for manual correction ---
        print("\n[FEEDBACK] Could not find contest panel. Please enter the correct contest title or leave blank to skip:")
        user_input = input("Contest title: ").strip()
        if user_input:
            html_context["selected_race"] = user_input
            rprint(f"[bold green][FEEDBACK] You entered: '{user_input}'[/bold green]")
            # Try again with user input
            contest_panel = extract_contest_panel(page, user_input)
            if not contest_panel:
                rprint("[red][ERROR] Still could not find contest panel. Skipping.[/red]")
                return None, None, None, {"skipped": True}
        else:
            rprint("[yellow][FEEDBACK] Skipped manual correction.[/yellow]")
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
        "state": state or "Unknown",
        "county": county or "Unknown",
        "race": contest_title or "Unknown",
        "source": getattr(page, "url", "Unknown"),
        "handler": "example_county"
    }
    return finalize_and_output(headers, data, contest_title, metadata)

def finalize_and_output(headers, data, contest_title, metadata):
    """
    Cleans, finalizes, and writes output using shared output utilities.
    - Removes empty/all-NA rows
    - Appends grand totals row
    - Recomputes headers
    - Writes output and metadata
    """
    # Remove empty or all-NA rows
    data = [row for row in data if any(str(v).strip() for v in row.values())]
    # Append grand totals row
    grand_total = calculate_grand_totals(data)
    data.append(grand_total)
    # Recompute headers in case grand_total added new fields
    headers = sorted(set().union(*(row.keys() for row in data)))
    # Write output and metadata
    result = finalize_election_output(headers, data, contest_title, metadata)
    contest_title = result.get("contest_title", contest_title)
    metadata = result.get("metadata", metadata)
    return headers, data, contest_title, metadata