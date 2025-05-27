import importlib
from playwright.sync_api import Page
from typing import Optional, Tuple, Any, List, Dict

from ....handlers.formats.html_handler import extract_contest_panel, extract_precinct_tables
from ....utils.shared_logger import logger
from ....utils.shared_logger import rprint
from ....utils.output_utils import finalize_election_output
from ....utils.contest_selector import select_contest
from ....utils.table_builder import extract_table_data, calculate_grand_totals
from ....utils.html_scanner import scan_html_for_context

def parse(
    page: Page,
    html_context: Optional[dict] = None,
    coordinator=None,
    non_interactive: bool = False
) -> Tuple[Any, Any, Any, dict]:
    """
    Example state handler, fully integrated with context coordinator and shared utilities.
    - If the state has county-specific handlers, delegates to them.
    - Otherwise, handles all counties within a single webpage.
    Returns (headers, data, contest_title, metadata).
    """
    from ....Context_Integration.context_coordinator import ContextCoordinator
    if html_context is None:
        html_context = {}

    # --- 1. Try to delegate to a county handler if county is specified ---
    county = (html_context.get("county") or "").strip().lower().replace(" ", "_")
    if county:
        module_path = f"webapp.parser.handlers.states.example.county.{county}"
        try:
            county_module = importlib.import_module(module_path)
            logger.info(f"[Example Handler] Routing to county parser: {module_path}")
            return county_module.parse(page, html_context, coordinator=coordinator, non_interactive=non_interactive)
        except ModuleNotFoundError:
            logger.warning(f"[Example Handler] No specific parser implemented for county: '{county}'. Continuing with state-level logic.")
        except Exception as e:
            logger.error(f"[Example Handler] Error in county parser: {e}")
            return None, None, None, {"error": str(e)}

    # --- 2. Otherwise, handle all counties within this page ---
    logger.info("[Example Handler] No county-specific handler found. Attempting state-level parsing.")

    # Scan for context and contests
    context = scan_html_for_context(page)
    html_context.update(context)
    state = html_context.get("state", "Example")
    county = html_context.get("county", None)

    # --- 3. Organize and enrich context with coordinator ---
    if coordinator is None:
        coordinator = ContextCoordinator()
        coordinator.organize_and_enrich(html_context)
    else:
        if not coordinator.organized:
            coordinator.organize_and_enrich(html_context)

    # --- 4. Contest selection using coordinator ---
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

    # If multiple contests, process each (aggregate or return first)
    if isinstance(selected, list):
        results = []
        for contest in selected:
            contest_title = contest.get("title") if isinstance(contest, dict) else contest
            html_context_copy = dict(html_context)
            html_context_copy["selected_race"] = contest_title
            result = parse_single_contest(page, html_context_copy, state, county, coordinator)
            results.append(result)
        # Return the first result, or aggregate as needed
        return results[0] if results else (None, None, None, {"skipped": True})
    else:
        contest_title = selected.get("title") if isinstance(selected, dict) else selected
        html_context["selected_race"] = contest_title
        return parse_single_contest(page, html_context, state, county, coordinator)

def parse_single_contest(page, html_context, state, county, coordinator):
    """
    Parses a single contest (race) from the page.
    """
    contest_title = html_context.get("selected_race")
    rprint(f"[cyan][INFO] Processing contest: {contest_title}[/cyan]")

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
        "state": state or "Unknown",
        "county": county or "Unknown",
        "race": contest_title or "Unknown",
        "source": getattr(page, "url", "Unknown"),
        "handler": "example"
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
    # Write output and metadata
    result = finalize_election_output(headers, data, contest_title, metadata)
    contest_title = result.get("contest_title", contest_title)
    metadata = result.get("metadata", metadata)
    return headers, data, contest_title, metadata