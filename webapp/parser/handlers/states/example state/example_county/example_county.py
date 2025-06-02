from playwright.sync_api import Page
from .....utils.logger_instance import logger
from .....utils.shared_logger import rprint
from .....utils.output_utils import finalize_election_output
from .....utils.table_builder import build_dynamic_table
from .....utils.html_scanner import scan_html_for_context
from .....utils.contest_selector import select_contest
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
            result = parse_single_contest_dynamic(page, html_context_copy, state, county, coordinator)
            results.append(result)
        return results[0] if results else (None, None, None, {"skipped": True})
    else:
        contest_title = selected.get("title") if isinstance(selected, dict) else selected
        html_context["selected_race"] = contest_title
        return parse_single_contest_dynamic(page, html_context, state, county, coordinator)

def parse_single_contest_dynamic(page, html_context, state, county, coordinator):
    """
    Parses a single contest (race) from the county page using dynamic, context/NLP-driven extraction.
    """
    contest_title = html_context.get("selected_race")
    rprint(f"[cyan][INFO] Processing contest: {contest_title}[/cyan]")

    # --- Use context/NLP to guide extraction ---
    entities = coordinator.extract_entities(contest_title)
    locations = [ent for ent, label in entities if label in ("GPE", "LOC", "FAC", "ORG") or "district" in ent.lower()]
    expected_location = locations[0] if locations else None

    # --- Try extracting ballot items from div-based containers first ---
    ballot_items = []
    selectors = [
        ".ballot-option", ".candidate-info", ".contest-row", ".result-row", ".header", ".race-row", ".proposition-row"
    ]
    for selector in selectors:
        items = page.locator(selector)
        for i in range(items.count()):
            item = items.nth(i)
            cells = item.locator("> *")
            row = [cells.nth(j).inner_text().strip() for j in range(cells.count())]
            if any(row):
                ballot_items.append(row)

    if ballot_items:
        first_row = ballot_items[0]
        known_keywords = ["candidate", "votes", "party", "precinct", "choice", "option", "response", "total"]
        if sum(1 for cell in first_row if any(kw in cell.lower() for kw in known_keywords)) >= 2:
            headers = first_row
            data_rows = [dict(zip(headers, row)) for row in ballot_items[1:]]
        else:
            headers = []
            for idx in range(len(first_row)):
                if expected_location and idx == 0:
                    headers.append(expected_location)
                elif idx == 0:
                    headers.append("Candidate")
                elif idx == 1:
                    headers.append("Party")
                elif idx == 2:
                    headers.append("Votes")
                else:
                    headers.append(f"Column {idx+1}")
            data_rows = [dict(zip(headers, row)) for row in ballot_items]
    else:
        # Fallback: try table-based extraction as a last resort
        rprint(f"[yellow][WARNING] No ballot items found by div selectors. Trying table-based extraction...[/yellow]")
        from .....utils.table_builder import robust_table_extraction
        headers, data_rows = robust_table_extraction(page, html_context)
        if not headers or not data_rows:
            rprint(f"[red][ERROR] No headers found and no table available for debugging.[/red]")
            return None, None, contest_title, {"skipped": True}

    # --- Build dynamic table ---
    headers, data = build_dynamic_table(headers, data_rows, coordinator, html_context)

    if not data:
        rprint("[red][ERROR] No contest data was parsed.[/red]")
        return None, None, contest_title, {"skipped": True}

    # --- Assemble headers and finalize output ---
    headers = sorted(set().union(*(row.keys() for row in data)))
    metadata = {
        "state": state or "Unknown",
        "county": county or "Unknown",
        "race": contest_title or "Unknown",
        "source": getattr(page, "url", "Unknown"),
        "handler": "example_county"
    }
    result = finalize_election_output(headers, data, coordinator, contest_title, state, county)
    if isinstance(result, dict):
        if "csv_path" in result:
            metadata["output_file"] = result["csv_path"]
        if "metadata_path" in result:
            metadata["metadata_path"] = result["metadata_path"]
        metadata.update(result)
    return headers, data, contest_title, metadata