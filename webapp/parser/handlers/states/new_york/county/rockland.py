from playwright.sync_api import Page

from .....utils.html_scanner import scan_html_for_context
from .....utils.contest_selector import select_contest
from .....utils.table_builder import build_dynamic_table, extract_table_data, find_tables_with_headings
from .....utils.output_utils import finalize_election_output
from .....utils.logger_instance import logger
from .....utils.shared_logger import rprint
from .....utils.shared_logic import autoscroll_until_stable
from .....utils.user_prompt import prompt_user_for_button, confirm_button_callback
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .....Context_Integration.context_coordinator import ContextCoordinator
import numpy as e
BUTTON_SELECTORS = "button, a, [role='button'], input[type='button'], input[type='submit']"



def find_and_click_best_button(
    page,
    coordinator,
    contest_title,
    keywords,
    toggle_name,
    context=None
):
    """
    Minimal handler: delegates all button logic to ContextCoordinator.
    Returns True if a button was clicked, else False.
    """
    btn, idx = coordinator.get_best_button_advanced(
        page=page,
        contest_title=contest_title,
        keywords=keywords,
        context=context or {"toggle_name": toggle_name},
        confirm_button_callback=confirm_button_callback, # Uncomment if you auto button click is giving wrong button
        prompt_user_for_button=prompt_user_for_button
    )
    if btn:
        # Debug print: show candidate info before clicking
        rprint(f"[bold yellow][DEBUG] About to click button:[/bold yellow]")
        rprint(f"  Label: {btn.get('label')}")
        rprint(f"  Selector: {btn.get('selector')}")
        rprint(f"  Class: {btn.get('class')}")
        rprint(f"  Tag: {btn.get('tag')}")
        rprint(f"  Context Heading: {btn.get('context_heading')}")
        rprint(f"  Context Anchor: {btn.get('context_anchor')}")
        # Print element handle debug info
        if "element_handle" in btn:
            rprint(f"  Element Handle ID: {id(btn['element_handle'])}")
            try:
                outer_html = btn["element_handle"].evaluate("el => el.outerHTML")
                rprint(f"  Outer HTML: {outer_html}")
            except Exception:
                rprint("  Outer HTML: [unavailable]")
            btn["element_handle"].click()
            return True
        else:
            rprint("[red][ERROR] No element_handle found for the selected button candidate.[/red]")
    rprint(f"[red][ERROR] No suitable '{toggle_name}' button could be clicked.[/red]")
    return False

def parse(page: Page, coordinator: "ContextCoordinator", html_context: dict = None, non_interactive=False):
    """
    Rockland County handler: all logic in one place.
    - Scans HTML for context and contests
    - Lets user select contest
    - Toggles "View results by election district" and "Vote Method"
    - Autoscrolls as needed
    - Extracts tables and outputs results
    """
    if html_context is None:
        html_context = {}

    rprint("[bold cyan][Rockland Handler] Parsing Rockland County Enhanced Voting page...[/bold cyan]")

    # --- 1. Scan HTML for context and contests ---
    state = html_context.get("state", "NY")
    county = html_context.get("county", "Rockland")

    # --- 3. Contest selection ---
    selected = select_contest(
        coordinator,
        state=state,
        county=county,
        year=html_context.get("year"),
        non_interactive=False
    )
    if not selected:
        rprint("[red]No contest selected. Skipping.[/red]")
        return None, None, None, {"skipped": True}
    # If multiple contests, process each (return first result or aggregate as needed)
    # --- 5. Toggle "View results by election district" ---
    if isinstance(selected, list):
        results = []
        for contest in selected:
            contest_title = contest.get("title") if isinstance(contest, dict) else contest
            html_context["selected_race"] = contest_title
            rprint(f"[cyan][INFO] Processing contest: {contest_title}[/cyan]")

            # --- Button toggles for this contest ---
            contest_title_for_button = contest_title if contest_title else None
            election_district_keywords = [
                r"view results? by election district[\s:]*$", "View results by election district", 
                "results by election district",  "election district", 
                "View results by"
            ]
            find_and_click_best_button(
                page,
                coordinator,
                contest_title_for_button,
                election_district_keywords,
                toggle_name="View results by election district",
                context={**html_context, "toggle_name": "View results by election district"}
            )
            autoscroll_until_stable(page, wait_for_selector="table")

            vote_method_keywords = [
                "vote method", "Vote Method", "Vote method", "Method"
            ]
            find_and_click_best_button(
                page,
                coordinator,
                contest_title_for_button,
                vote_method_keywords,
                toggle_name="Vote Method",
                context={**html_context, "toggle_name": "Vote Method"}
            )
            autoscroll_until_stable(page)

    # --- 9. Extract precinct tables robustly using DOM scan and heading association ---
    segments = html_context.get("tagged_segments_with_attrs", [])
    precinct_tables = find_tables_with_headings(page, dom_segments=segments)

    all_data_rows = []
    headers = None
    last_table_with_no_headers = None

    for precinct_name, table in precinct_tables:
        if not table or not precinct_name:
            continue
        headers, data_rows = extract_table_data(table)
        print("DEBUG: Extracted headers:", headers)
        if not headers:
            last_table_with_no_headers = table
            continue
        # Patch: If the first header is "Candidate" and the values look like precincts, rename to "Precinct"
        if headers and headers[0].lower() == "candidate":
            candidate_col = [row.get("Candidate", "") for row in data_rows]
            if sum(1 for v in candidate_col if "precinct" in v.lower() or "ed" in v.lower() or v.strip().isdigit()) > len(candidate_col) // 2:
                headers[0] = "Precinct"
                for row in data_rows:
                    row["Precinct"] = row.pop("Candidate")
        for row in data_rows:
            row["Precinct"] = precinct_name  # Always set the correct precinct name
        all_data_rows.extend(data_rows)

    # --- Now call build_dynamic_table ONCE for all precincts ---
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
    headers, rows = build_dynamic_table(headers, all_data_rows, coordinator, html_context)
    data = rows

    if not data:
        rprint(f"[yellow][DEBUG] precinct_tables: {precinct_tables}[/yellow]")
        rprint(f"[yellow][DEBUG] data: {data}[/yellow]")
        rprint("[red][ERROR] No precinct data was parsed.[/red]")
        return None, None, None, {"skipped": True}

    # --- 10. Assemble headers and finalize output ---
    headers = sorted(set().union(*(row.keys() for row in data)))
    print("DEBUG: Finalized headers:", headers)
    metadata = {
        "state": state or "Unknown",
        "county": county or "Unknown",
        "race": contest_title or "Unknown",
        "source": getattr(page, "url", "Unknown"),
        "handler": "rockland"
    }
    if "year" in html_context:
        metadata["year"] = html_context["year"]
    if "election_type" in html_context:
        metadata["election_type"] = html_context["election_type"]
        print("DEBUG: headers before finalize:", headers)
        print("DEBUG: first row before finalize:", data[0] if data else None)
        print("DEBUG: contest_title before finalize:", contest_title)
    result = finalize_election_output(headers, data, coordinator, contest_title, state, county)
    # Merge metadata for output
    if isinstance(result, dict):
        # Ensure output_file is present for the main pipeline
        if "csv_path" in result:
            metadata["output_file"] = result["csv_path"]
        # Optionally include metadata_path or other keys if needed
        if "metadata_path" in result:
            metadata["metadata_path"] = result["metadata_path"]
        metadata.update(result)
    return headers, data, contest_title, metadata