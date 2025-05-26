from playwright.sync_api import Page
from .....Context_Integration.context_organizer import organize_context
from .....handlers.formats.html_handler import extract_contest_panel, extract_precinct_tables
import os
from .....utils.html_scanner import scan_html_for_context, get_detected_races_from_context
from .....utils.format_router import detect_format_from_links, prompt_user_for_format, route_format_handler
from .....utils.download_utils import download_confirmed_file
from .....utils.contest_selector import select_contest
from .....utils.table_builder import extract_table_data, calculate_grand_totals
from .....utils.output_utils import finalize_election_output
from .....utils.shared_logger import logger, rprint
from .....utils.shared_logic import autoscroll_until_stable, find_and_click_toggle, ALL_SELECTORS



def parse(page: Page, html_context: dict = None):
    """
    Rockland County handler: delegates all generic logic to shared utilities.
    Only customizes steps unique to Rockland's site.
    """
    if html_context is None:
        html_context = {}

    rprint("[bold cyan][Rockland Handler] Parsing Rockland County Enhanced Voting page...[/bold cyan]")

    # --- 1. Scan for downloadable formats (JSON/CSV/PDF) ---
    found_files = detect_format_from_links(page)
    if found_files:
        for fmt, url in found_files:
            filename = os.path.basename(url)
            rprint(f"[bold green]Discovered {fmt.upper()} file:[/bold green] {filename}")
            # Prompt user to confirm download/parse
            user_confirm = prompt_user_for_format([(fmt, url)], logger=logger)
            if user_confirm and user_confirm[0]:
                handler = route_format_handler(fmt)
                if handler:
                    file_path = download_confirmed_file(url, page.url)
                    if not file_path:
                        rprint(f"[red]Failed to download {filename}. Continuing with HTML parsing.[/red]")
                        continue
                    # Parse with the appropriate handler
                    result = handler.parse(None, {"filename": file_path, **html_context})
                    if result and isinstance(result, tuple) and len(result) == 4:
                        headers, data, contest_title, metadata = result
                        return finalize_and_output(headers, data, contest_title, metadata)
                    else:
                        rprint(f"[red]Handler for {fmt} did not return expected structure. Skipping.[/red]")
                        continue
            rprint(f"[yellow]Skipping {filename}, continuing with HTML parsing.[/yellow]")

    # --- 2. Scan HTML for context and contests ---
    context = scan_html_for_context(page)
    html_context.update(context)
    state = html_context.get("state", "NY")
    county = html_context.get("county", "Rockland")

    # --- 3. Contest selection ---
    detected_races = get_detected_races_from_context(html_context)
    selected = select_contest(detected_races)
    if not selected:
        rprint("[red]No contest selected. Skipping.[/red]")
        return None, None, None, {"skipped": True}
    # If multiple contests, process each
    if isinstance(selected, list):
        results = []
        for contest_tuple in selected:
            contest_title = contest_tuple[2]  # (year, etype, race)
            html_context_copy = dict(html_context)
            html_context_copy["selected_race"] = contest_title
            result = parse_single_contest(page, html_context_copy, state, county, find_contest_panel=extract_contest_panel)
            results.append(result)
        # Return the first result (or aggregate as needed)
        return results[0] if results else (None, None, None, {"skipped": True})
    else:
        contest_title = selected[2]
        html_context["selected_race"] = contest_title
        return parse_single_contest(page, html_context, state, county, find_contest_panel=extract_contest_panel)

def parse_single_contest(page, html_context, state, county, find_contest_panel):
    contest_title = html_context.get("selected_race")
    rprint(f"[cyan][INFO] Processing contest: {contest_title}[/cyan]")

    contest_panel = find_contest_panel(page, contest_title)
    if not contest_panel:
        rprint("[red][ERROR] Contest panel not found. Skipping.[/red]")
        return None, None, None, {"skipped": True}

    # --- 1. Toggle "View results by election district" ---
    button_features = ALL_SELECTORS(page, container=contest_panel)
    toggle_button = next(
        (btn for btn in button_features if "election-district" in btn["label"].lower() or "your-class-name" in btn["class"]), 
        None
    )
    handler_selectors = []
    handler_keywords = []
    if toggle_button:
        if toggle_button["selector"]:
            handler_selectors.append(toggle_button["selector"])
        if toggle_button["label"]:
            handler_keywords.append(toggle_button["label"])

    # Click the first toggle
    find_and_click_toggle(
        page,
        container=contest_panel,
        handler_selectors=handler_selectors if handler_selectors else None,
        handler_keywords=handler_keywords if handler_keywords else ["View results by election district"],
        logger=logger,
        verbose=True,
        interactive=True
    )

    # --- 2. Wait for precincts to load (table or new panel) ---
    autoscroll_until_stable(page, wait_for_selector="table")

    # --- 3. Now look for and toggle vote method, if present ---
    # (This must be after the first toggle, as the vote method toggle may not exist before)
    find_and_click_toggle(
        page, 
        container=contest_panel, 
        handler_keywords=[
            "Vote Method", "Voting Method", "Ballot Method"
        ], 
        logger=logger, 
        verbose=True, 
        interactive=True
    )

    # --- 4. Scroll again if needed ---
    autoscroll_until_stable(page)

    # --- 5. Extract precinct tables ---
    precinct_tables = extract_precinct_tables(contest_panel)
    data = []
    method_names = None

    for precinct_name, table in precinct_tables:
        if not table or not precinct_name:
            continue
        # Extract method names from headers if not set
        if method_names is None:
            headers, _ = extract_table_data(table)
            if len(headers) > 2:
                method_names = headers[1:-1]
            else:
                method_names = []
        headers, rows = extract_table_data(table)
        for row in rows:
            # Optionally, add precinct_name to each row if needed
            row["Precinct"] = precinct_name
            data.append(row)

    if not data:
        rprint(f"[yellow][DEBUG] precinct_tables: {precinct_tables}[/yellow]")
        rprint(f"[yellow][DEBUG] method_names: {method_names}[/yellow]")
        rprint(f"[yellow][DEBUG] data: {data}[/yellow]")
        rprint("[red][ERROR] No precinct data was parsed.[/red]")
        return None, None, None, {"skipped": True}

    # Assemble headers and finalize output
    headers = sorted(set().union(*(row.keys() for row in data)))
    metadata = {
        "state": state,
        "county": county,
        "race": contest_title,
        "source": page.url,
        "handler": "rockland"
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
    organized = organize_context(metadata)
    metadata = organized.get("metadata", metadata)
    # Write output and metadata
    result = finalize_election_output(headers, data, contest_title, metadata)
    contest_title = result.get("contest_title", contest_title)
    metadata = result.get("metadata", metadata)
    return headers, data, contest_title, metadata