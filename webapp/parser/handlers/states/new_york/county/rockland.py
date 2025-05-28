from playwright.sync_api import Page
import os
import re
import difflib
from .....utils.html_scanner import scan_html_for_context
from .....utils.contest_selector import select_contest
from .....utils.table_builder import extract_table_data
from .....utils.output_utils import finalize_election_output
from .....utils.logger_instance import logger
from .....utils.shared_logger import rprint
from .....utils.shared_logic import autoscroll_until_stable
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .....Context_Integration.context_coordinator import ContextCoordinator

BUTTON_SELECTORS = "button, a, [role='button'], input[type='button'], input[type='submit']"

def prompt_user_for_button(page, candidates, toggle_name):
    """
    Feedback UI: Prompt user to select the correct button from candidates.
    """
    print(f"\n[FEEDBACK] Please select the correct button for '{toggle_name}':")
    for idx, btn in enumerate(candidates):
        print(f"{idx}: label='{btn['label']}' | class='{btn['class']}' | visible={btn['is_visible']} | enabled={btn['is_clickable']}")
    try:
        choice = int(input("Enter the number of the correct button (or -1 to skip): "))
        if 0 <= choice < len(candidates):
            chosen_btn = candidates[choice]
            rprint(f"[bold green][FEEDBACK] You selected: '{chosen_btn['label']}'[/bold green]")
            return chosen_btn, choice
        else:
            rprint("[yellow][FEEDBACK] Skipped manual correction.[/yellow]")
            return None, None
    except Exception as e:
        rprint(f"[red][FEEDBACK ERROR] {e}[/red]")
        return None, None

def find_and_click_best_button(
    page,
    coordinator,
    contest_title,
    keywords,
    toggle_name,
    fuzzy_threshold=0.7
):
    """
    Try to find and click the best button using context, ML, fuzzy, and feedback UI.
    Returns True if a button was clicked, else False.
    """
    best_button = coordinator.get_best_button(
        contest_title=contest_title,
        keywords=keywords,
        url=getattr(page, "url", None),
        prefer_clickable=True,
        prefer_visible=True,
        log_memory=True,
        page=None  # We'll do our own DOM scan if needed
    )
    rprint(f"[DEBUG] get_best_button ({toggle_name}) returned: {best_button}")

    if best_button and best_button.get("label"):
        rprint(f"[bold green][Rockland Handler] Clicking best button: '{best_button.get('label')}'[/bold green]")
        btn_label = best_button.get("label")
        btn_selector = best_button.get("selector", None)
        # Try selector first if available, else fallback to label
        if btn_selector:
            button_locator = page.locator(btn_selector)
        else:
            button_locator = page.locator(f"button:has-text('{btn_label}')")
            if button_locator.count() == 0:
                button_locator = page.locator(f"a:has-text('{btn_label}')")
        if button_locator.count() > 0:
            button_locator.first.click()
            return True
        else:
            rprint(f"[yellow][WARN] Could not find button with label '{btn_label}' or selector '{btn_selector}' on page.[/yellow]")
            coordinator._log_button_memory({"label": btn_label, "selector": btn_selector}, contest_title, "fail")

    # --- If not found, scan DOM for candidates and score them ---
    rprint(f"[yellow][WARN] No suitable '{toggle_name}' button found by coordinator. Scanning page for new candidates...[/yellow]")
    button_features = page.locator(BUTTON_SELECTORS)
    candidates = []
    for i in range(button_features.count()):
        btn = button_features.nth(i)
        label = btn.inner_text() or ""
        class_name = btn.get_attribute("class") or ""
        is_visible = btn.is_visible()
        is_enabled = btn.is_enabled()
        selector = None
        try:
            selector = btn.evaluate("el => el.outerHTML")
        except Exception:
            pass
        rprint(f"[ML-FEEDBACK][Button {i}] label='{label}', class='{class_name}', visible={is_visible}, enabled={is_enabled}")
        candidate = {
            "label": label,
            "class": class_name,
            "selector": selector,
            "is_visible": is_visible,
            "is_clickable": is_enabled
        }
        candidates.append(candidate)
        coordinator._log_button_memory(candidate, contest_title, "scanned")

    # --- Score candidates by fuzzy/regex match ---
    best_score = 0
    best_candidate = None
    best_idx = None
    for idx, cand in enumerate(candidates):
        for kw in (keywords or []):
            score = difflib.SequenceMatcher(None, kw.lower(), (cand["label"] or "").lower()).ratio()
            if score > best_score:
                best_score = score
                best_candidate = cand
                best_idx = idx
            if kw.lower() in (cand["label"] or "").lower():
                best_score = 1.0
                best_candidate = cand
                best_idx = idx
                break
            if cand["label"] and re.search(re.escape(kw), cand["label"], re.IGNORECASE):
                best_score = 1.0
                best_candidate = cand
                best_idx = idx
                break
    if best_candidate and best_score >= fuzzy_threshold:
        rprint(f"[bold green][Rockland Handler] Fuzzy/regex match found: '{best_candidate['label']}' (score={best_score:.2f})[/bold green]")
        coordinator._log_button_memory(best_candidate, contest_title, f"fuzzy_pass_{best_score:.2f}")
        button_locator = page.locator(f"{BUTTON_SELECTORS} >> nth={best_idx}")
        if button_locator.count() > 0:
            button_locator.first.click()
            return True

    # --- Feedback UI: Prompt user for manual correction ---
    chosen_btn, chosen_idx = prompt_user_for_button(page, candidates, toggle_name)
    if chosen_btn and chosen_idx is not None:
        coordinator._log_button_memory(chosen_btn, contest_title, "manual_correction")
        button_locator = page.locator(f"{BUTTON_SELECTORS} >> nth={chosen_idx}")
        if button_locator.count() > 0:
            button_locator.first.click()
            return True

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
    handler_options = {}
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
        non_interactive=non_interactive
    )
    if not selected:
        rprint("[red]No contest selected. Skipping.[/red]")
        return None, None, None, {"skipped": True}

    # If multiple contests, process each (return first result or aggregate as needed)
    if isinstance(selected, list):
        selected = selected[0]

    contest_title = selected.get("title") if isinstance(selected, dict) else selected
    html_context["selected_race"] = contest_title
    rprint(f"[cyan][INFO] Processing contest: {contest_title}[/cyan]")

    # --- 5. Toggle "View results by election district" ---
    contest_title_for_button = contest_title if contest_title else None
    election_district_keywords = [
        "election district", "view results by election district", "status for each election district",
        "View results by district", "Show results by election district", "District results"
    ]
    find_and_click_best_button(
        page,
        coordinator,
        contest_title_for_button,
        election_district_keywords,
        toggle_name="View results by election district"
    )

    # --- 6. Autoscroll to ensure table loads ---
    autoscroll_until_stable(page, wait_for_selector="table")

    # --- 7. Toggle "Vote Method" if present ---
    vote_method_keywords = [
        "vote method", "Voting Method", "Show vote method", "Summary by method", "View by method", "Method"
    ]
    find_and_click_best_button(
        page,
        coordinator,
        contest_title_for_button,
        vote_method_keywords,
        toggle_name="Vote Method"
    )

    # --- 8. Autoscroll again if needed ---
    autoscroll_until_stable(page)

    # --- 9. Extract precinct tables ---
    tables = page.locator("table")
    precinct_tables = []
    for i in range(tables.count()):
        table = tables.nth(i)
        precinct_name = None
        try:
            header_locator = table.locator("xpath=preceding-sibling::*[self::h1 or self::h2 or self::h3 or self::h4 or self::h5 or self::h6][1]")
            if header_locator.count() > 0:
                precinct_name = header_locator.nth(0).inner_text().strip()
        except Exception:
            pass
        if not precinct_name:
            precinct_name = f"Precinct {i+1}"
        precinct_tables.append((precinct_name, table))

    data = []
    method_names = None

    for precinct_name, table in precinct_tables:
        if not table or not precinct_name:
            continue
        if method_names is None:
            headers, _ = extract_table_data(table)
            if len(headers) > 2:
                method_names = headers[1:-1]
            else:
                method_names = []
        headers, rows = extract_table_data(table)
        for row in rows:
            row["Precinct"] = precinct_name
            data.append(row)

    if not data:
        rprint(f"[yellow][DEBUG] precinct_tables: {precinct_tables}[/yellow]")
        rprint(f"[yellow][DEBUG] method_names: {method_names}[/yellow]")
        rprint(f"[yellow][DEBUG] data: {data}[/yellow]")
        rprint("[red][ERROR] No precinct data was parsed.[/red]")
        return None, None, None, {"skipped": True}

    # --- 10. Assemble headers and finalize output ---
    headers = sorted(set().union(*(row.keys() for row in data)))
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

    result = finalize_election_output(headers, data, coordinator, contest_title, handler_options, state, county)
    # Merge handler_options and metadata for output
    if isinstance(result, dict):
        # Ensure output_file is present for the main pipeline
        if "csv_path" in result:
            metadata["output_file"] = result["csv_path"]
        # Optionally include metadata_path or other keys if needed
        if "metadata_path" in result:
            metadata["metadata_path"] = result["metadata_path"]
        metadata.update(result)
    return headers, data, contest_title, metadata