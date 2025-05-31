import importlib
from playwright.sync_api import Page
from typing import Optional, Tuple, Any, List, Dict

from ....utils.shared_logger import logger
from ....utils.shared_logger import rprint
from ....utils.output_utils import finalize_election_output
from ....utils.contest_selector import select_contest
from ....utils.table_builder import extract_table_data, build_dynamic_table, find_tables_with_headings
from ....utils.html_scanner import scan_html_for_context
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ....Context_Integration.context_coordinator import ContextCoordinator

def parse(
    page: Page,
    coordinator: "ContextCoordinator",
    html_context: Optional[dict] = None,
    non_interactive: bool = False
) -> Tuple[Any, Any, Any, dict]:
    """
    Example state handler, fully integrated with context coordinator and shared utilities.
    - If the state has county-specific handlers, delegates to them.
    - Otherwise, handles all counties within a single webpage.
    Returns (headers, data, contest_title, metadata).
    """
    if html_context is None:
        html_context = {}

    # --- 1. Try to delegate to a county handler if county is specified ---
    county = (html_context.get("county") or "").strip().lower().replace(" ", "_")
    if county:
        module_path = f"webapp.parser.handlers.states.example.county.{county}"
        try:
            county_module = importlib.import_module(module_path)
            logger.info(f"[Example Handler] Routing to county parser: {module_path}")
            return county_module.parse(page, coordinator=coordinator, html_context=html_context, non_interactive=non_interactive)
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
        from ....Context_Integration.context_coordinator import ContextCoordinator
        coordinator = ContextCoordinator()
        coordinator.organize_and_enrich(html_context)
    else:
        if not getattr(coordinator, "organized", False):
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
        return results[0] if results else (None, None, None, {"skipped": True})
    else:
        contest_title = selected.get("title") if isinstance(selected, dict) else selected
        html_context["selected_race"] = contest_title
        return parse_single_contest(page, html_context, state, county, coordinator)

def parse_single_contest(page, html_context, state, county, coordinator):
    """
    Parses a single contest (race) from the page using robust table/heading association.
    """
    contest_title = html_context.get("selected_race")
    rprint(f"[cyan][INFO] Processing contest: {contest_title}[/cyan]")

    # --- Robust table extraction using heading association ---
    segments = html_context.get("tagged_segments_with_attrs", [])
    precinct_tables = find_tables_with_headings(page, dom_segments=segments)

    all_data_rows = []
    headers = None
    last_table_with_no_headers = None

    for precinct_name, table in precinct_tables:
        if not table or not precinct_name:
            continue
        headers, data_rows = extract_table_data(table)
        if not headers:
            last_table_with_no_headers = table
            continue
        for row in data_rows:
            row["Precinct"] = precinct_name
        all_data_rows.extend(data_rows)

    if not headers:
        if last_table_with_no_headers:
            try:
                table_html = last_table_with_no_headers.inner_html()
            except Exception:
                table_html = "[unavailable]"
            rprint(f"[red][ERROR] No headers found in any table. Example table HTML:\n{table_html}[/red]")
        else:
            rprint(f"[red][ERROR] No headers found and no table available for debugging.[/red]")
        return None, None, None, {"skipped": True}

    # --- Build dynamic table for all precincts ---
    headers, data = build_dynamic_table(headers, all_data_rows, coordinator, html_context)

    if not data:
        rprint("[red][ERROR] No precinct data was parsed.[/red]")
        return None, None, None, {"skipped": True}

    # --- Assemble headers and finalize output ---
    headers = sorted(set().union(*(row.keys() for row in data)))
    metadata = {
        "state": state or "Unknown",
        "county": county or "Unknown",
        "race": contest_title or "Unknown",
        "source": getattr(page, "url", "Unknown"),
        "handler": "example"
    }
    result = finalize_election_output(headers, data, coordinator, contest_title, state, county)
    if isinstance(result, dict):
        if "csv_path" in result:
            metadata["output_file"] = result["csv_path"]
        if "metadata_path" in result:
            metadata["metadata_path"] = result["metadata_path"]
        metadata.update(result)
    return headers, data, contest_title, metadata