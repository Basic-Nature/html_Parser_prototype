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
context_cache = {}

def score_table(table, context, coordinator):
    """
    Score a table based on known district/precinct names, result keywords, and table size.
    """
    headers, data = extract_table_data(table)
    score = 0
    # 1. Known district/precinct names in first column
    known_districts = set()
    if hasattr(coordinator, "get_districts"):
        known_districts = set(coordinator.get_districts(
            state=context.get("state"), county=context.get("county")
        ) or [])
    if headers and data:
        first_col = [row.get(headers[0], "") for row in data]
        matches = sum(1 for val in first_col if any(str(d).lower() in str(val).lower() for d in known_districts))
        score += matches * 3  # weight matches higher
    # 2. Known result keywords in headers
    result_keywords = {"votes", "candidate", "precinct", "total"}
    score += sum(2 for h in headers if any(kw in h.lower() for kw in result_keywords))
    # 3. Table size
    score += len(data)
    # 4. Prefer tables with more numeric columns (likely results)
    if headers and data:
        numeric_cols = sum(
            1 for h in headers
            if all(str(row.get(h, "")).replace(",", "").replace(".", "").isdigit() or row.get(h, "") == "" for row in data)
        )
        score += numeric_cols
    return score, headers, data

def handle_toggle_and_rescan(
    page,
    coordinator,
    context_cache,
    contest_title,
    keywords,
    toggle_name,
    html_context,
    extra_context=None
):
    """
    Clicks the best button for the toggle, waits for page change, and re-scans context if changed.
    Always uses the coordinator's learned button logic.
    """
    from .....html_election_parser import get_page_hash, get_or_scan_context

    prev_hash = get_page_hash(page)
    # Always use the advanced button logic (learned button first)
    btn, idx = coordinator.get_best_button_advanced(
        page=page,
        contest_title=contest_title,
        keywords=keywords,
        context={**(extra_context or {}), **(html_context or {}), "toggle_name": toggle_name},
        confirm_button_callback=confirm_button_callback,
        prompt_user_for_button=prompt_user_for_button,
        learning_mode=True
    )
    if btn and "element_handle" in btn:
        element = btn["element_handle"]
        if element.is_visible() and element.is_enabled():
            try:
                element.click(timeout=5000)  # Shorter timeout, fail fast if not clickable
            except Exception as e:
                rprint(f"[red][ERROR] Failed to click button '{btn.get('label', '')}': {e}[/red]")
                return html_context  # Return unchanged context if click fails
        else:
            rprint(f"[yellow][WARNING] Button '{btn.get('label', '')}' is not clickable (visible={element.is_visible()}, enabled={element.is_enabled()})[/yellow]")
            return html_context  # Return unchanged context if not clickable
    else:
        rprint(f"[red][ERROR] No suitable '{toggle_name}' button could be clicked.[/red]")
        return html_context  # Return unchanged context if no button

    autoscroll_until_stable(page)
    new_hash = get_page_hash(page)
    if new_hash != prev_hash:
        # Use the orchestrator's context cache and loader
        html_context = get_or_scan_context(page, coordinator)
        context_cache[new_hash] = html_context
        rprint(f"[green][TOGGLE] Page changed after '{toggle_name}' toggle. Context re-scanned.[/green]")
    else:
        rprint(f"[yellow][TOGGLE] Page did not change after '{toggle_name}' toggle. Skipping re-scan.[/yellow]")
    return html_context

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
    # --- 5. Toggle "View results by election district" ---
            html_context = handle_toggle_and_rescan(
                page,
                coordinator,
                context_cache,
                contest_title_for_button,
                election_district_keywords,
                "View results by election district",
                html_context,
                extra_context={"toggle_name": "View results by election district"}
            )
            autoscroll_until_stable(page)

            vote_method_keywords = [
                "vote method", "Vote Method", "Vote method", "Method"
            ]
            html_context = handle_toggle_and_rescan(
                page,
                coordinator,
                context_cache,
                contest_title_for_button,
                vote_method_keywords,
                "Vote Method",
                html_context,
                extra_context={"toggle_name": "Vote Method"}
            )
            autoscroll_until_stable(page)

    # --- 9. Extract precinct tables robustly using DOM scan and heading association ---
    segments = html_context.get("tagged_segments_with_attrs", [])
    precinct_tables = find_tables_with_headings(page, dom_segments=segments)

    # Score all tables and pick the best one(s)
    scored_tables = []
    for precinct_name, table in precinct_tables:
        if not table:
            continue
        score, headers, data_rows = score_table(table, html_context, coordinator)
        if not headers:
            continue
        # Heuristic: If the first header is "Candidate" and values look like precincts, rename to "Precinct"
        if headers and headers[0].lower() == "candidate":
            candidate_col = [row.get("Candidate", "") for row in data_rows]
            if sum(1 for v in candidate_col if "precinct" in v.lower() or "ed" in v.lower() or v.strip().isdigit()) > len(candidate_col) // 2:
                headers[0] = "Precinct"
                for row in data_rows:
                    row["Precinct"] = row.pop("Candidate")
        # Always set the correct precinct name if available
        if precinct_name:
            for row in data_rows:
                row["Precinct"] = precinct_name
        scored_tables.append((score, headers, data_rows, precinct_name, table))

    # Pick the highest scoring table(s)
    scored_tables.sort(reverse=True, key=lambda x: x[0])
    if not scored_tables:
        logger.warning(f"[TABLE EXTRACTION] No suitable table found for contest: {html_context.get('selected_race')}")
        # Try to extract data from first table even if headers are missing
        if precinct_tables:
            first_table = precinct_tables[0][1]
            headers, data_rows = extract_table_data(first_table)
            if data_rows:
                headers = headers or [f"Column {i+1}" for i in range(len(data_rows[0]))]
                rprint(f"[yellow][WARNING] No headers found, using generic headers: {headers}[/yellow]")
                # Proceed with build_dynamic_table
                headers, rows = build_dynamic_table(headers, data_rows, coordinator, html_context)
                if rows:
                    data = rows
                    # ...finalize output as usual...
                    # (copy the rest of your output logic here)
                    # return headers, data, contest_title, metadata
                else:
                    rprint(f"[red][ERROR] No data could be parsed from fallback table.[/red]")
            else:
                rprint(f"[red][ERROR] No data found in fallback table.[/red]")
        else:
            rprint(f"[red][ERROR] No headers found and no table available for debugging.[/red]")
        return None, None, None, {"skipped": True}

    if not scored_tables and precinct_tables:
        rprint("[yellow][PROMPT] No headers found. Please select a table to inspect:")
        for idx, (heading, table) in enumerate(precinct_tables):
            preview = table.inner_text()[:120].replace("\n", " ")
            rprint(f"[{idx}] Heading: {heading} | Preview: {preview}...")
        sel = input("Enter table index to inspect (or blank to skip): ").strip()
        if sel.isdigit():
            idx = int(sel)
            if 0 <= idx < len(precinct_tables):
                table = precinct_tables[idx][1]
                headers, data_rows = extract_table_data(table)
                rprint(f"[yellow][DEBUG] Table headers: {headers}")
                rprint(f"[yellow][DEBUG] First row: {data_rows[0] if data_rows else None}")
        return None, None, None, {"skipped": True}

    if not precinct_tables:
        # Fallback: try all tables on the page
        all_tables = page.locator("table")
        for i in range(all_tables.count()):
            table = all_tables.nth(i)
            headers, data_rows = extract_table_data(table)
            if headers or data_rows:
                rprint(f"[yellow][DEBUG] Fallback table #{i}: headers={headers}, first row={data_rows[0] if data_rows else None}")

    # Optionally, you can aggregate data from all high-scoring tables, or just use the top one
    top_score = scored_tables[0][0]
    top_tables = [t for t in scored_tables if t[0] == top_score]
    all_data_rows = []
    headers = None
    for score, hdrs, data_rows, precinct_name, table in top_tables:
        headers = hdrs
        all_data_rows.extend(data_rows)

    if len(top_tables) > 1:
        rprint(f"[yellow][DEBUG] Multiple tables tied for top score. Prompting user for selection.[/yellow]")
        # Show a preview of each table (headers, first row) and prompt user
        for idx, (score, hdrs, data_rows, precinct_name, table) in enumerate(top_tables):
            rprint(f"[{idx}] Precinct: {precinct_name}, Headers: {hdrs}, First row: {data_rows[0] if data_rows else None}")
        sel = input("Select table index (or press Enter for first): ").strip()
        try:
            sel_idx = int(sel)
            headers, all_data_rows = top_tables[sel_idx][1], top_tables[sel_idx][2]
        except Exception:
            headers, all_data_rows = top_tables[0][1], top_tables[0][2]

    logger.info(f"[LEARNING] Confirmed table pattern: headers={headers}, num_rows={len(all_data_rows)}")
       
    # --- Now call build_dynamic_table ONCE for all precincts ---
    if not headers:
        rprint(f"[red][ERROR] No headers found in any table. No suitable table available for debugging.[/red]")
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