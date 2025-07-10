from playwright.sync_api import Page

from .....utils.contest_selector import select_contest
from .....utils.table_builder import build_dynamic_table
from .....utils.table_core import harmonize_headers_and_data
from .....utils.output_utils import finalize_election_output
from .....utils.shared_logger import log_error, log_info, log_warning, log_debug
from .....utils.shared_logic import autoscroll_until_stable
from .....utils.user_prompt import UserPrompt
from .....utils.html_scanner import scan_html_for_context

prompt = UserPrompt()

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .....Context_Integration.context_coordinator import ContextCoordinator
import numpy as e
BUTTON_SELECTORS = "button, a, [role='button'], input[type='button'], input[type='submit']"
context_cache = {}
accepted_buttons_cache = {}

def parse(page: Page, coordinator: "ContextCoordinator", html_context: dict = None, non_interactive=False, **kwargs):
    """
    Rockland County handler: all logic in one place.
    - Scans HTML for context and contests
    - Lets user select contest
    - Toggles "View results by election district" and "Vote Method"
    - Autoscrolls as needed (only once, after all toggles)
    - Extracts tables and outputs results
    """
    from .....Context_Integration.context_organizer import clean_for_json
    if html_context is None:
        html_context = {}

    log_info("[bold cyan][Rockland Handler] Parsing Rockland County Enhanced Voting page...[/bold cyan]")

    # --- 1. Scan HTML and organize context before contest selection ---
    context_result = scan_html_for_context(
        target_url=getattr(page, "url", None),
        page=page,
        coordinator=coordinator,
        debug=False,
    )
    state = context_result.get("state") or "NY"
    county = context_result.get("county") or "Rockland"
    year = context_result.get("year")
    for contest in context_result.get("contests", []):
        if contest.get("state") is None:
            contest["state"] = state
        if contest.get("county") is None:
            contest["county"] = county
        if contest.get("year") is None and year is not None:
            contest["year"] = year
    context_result = clean_for_json(context_result)
    coordinator.organize_and_enrich(context_result)
    selector_data = coordinator.get_for_selector()
    log_debug("DEBUG: selector_data['contests']:", selector_data.get("contests", []))
    # --- 3. Contest selection ---
    context_for_selector = {
        "state": state,
        "county": county,
        "year": year,
        "contests": context_result.get("contests", []),
        **{k: v for k, v in html_context.items() if k not in ("state", "county", "year", "contests")}
    }
    selected = select_contest(
        coordinator,
        state=state,
        county=county,
        year=year,
        non_interactive=non_interactive,
        context=context_for_selector
    )
    if not selected:
        log_warning("[red]No contest selected. Skipping.[/red]")
        return None, None, None, {"skipped": True}
    # If multiple contests, process each (return first result or aggregate as needed)
    if isinstance(selected, list):
        results = []
        for contest in selected:
            user_selected_title = contest.get("title") if isinstance(contest, dict) else contest
            html_context["selected_race"] = user_selected_title
            log_info(f"[cyan][INFO] Processing contest: {user_selected_title}[/cyan]")

            # --- Button toggles for this contest ---
            contest_title_for_button = user_selected_title if user_selected_title else None

            # --- Toggle "View results by election district" ---
            election_district_keywords = [
                r"view results? by election district[\s:]*$", "View results by election district", 
                "results by election district",  "election district", 
                "View results by"
            ]
            toggle_name = "View results by election district"
            log_debug(f"[DEBUG] About to toggle first button: {toggle_name}")
            btn, idx = coordinator.get_best_button_advanced(
                page=page,
                contest_title=contest_title_for_button,
                keywords=election_district_keywords,
                context={**html_context, "toggle_name": toggle_name},
                confirm_button_callback=prompt.confirm_button_callback,
                prompt_user_for_button=prompt.prompt_user_for_button,
                learning_mode=True,
            )
            if btn and "element_handle" in btn:
                element = btn["element_handle"]
                # Only click if not already clicked by coordinator (learning mode)
                if btn.get("selector") not in coordinator.clicked_button_selectors:
                    if element.is_visible() and element.is_enabled():
                        try:
                            log_debug(f"[blue][DEBUG] Clicking button: '{btn.get('label', '')}' for toggle '{toggle_name}'")
                            element.click(timeout=5000)
                            page.wait_for_timeout(3000)
                            log_debug(f"[green][DEBUG] Button click for '{toggle_name}' completed.[/green]")
                            coordinator.clicked_button_selectors.add(btn.get("selector"))
                        except Exception as e:
                            log_error(f"[red][ERROR] Failed to click button '{btn.get('label', '')}': {e}[/red]")
                    else:
                        log_warning(f"[yellow][WARNING] Button '{btn.get('label', '')}' is not clickable (visible={element.is_visible()}, enabled={element.is_enabled()})[/yellow]")
                else:
                    log_debug(f"[yellow][DEBUG] Button '{btn.get('label', '')}' was already clicked by learning mode.[/yellow]")
            else:
                log_error(f"[red][ERROR] No suitable '{toggle_name}' button could be clicked.[/red]")

            log_debug(f"[DEBUG] Finished toggle first button: {toggle_name}")

            # --- Toggle "Vote Method" ---
            vote_method_keywords = [
                "vote method", "Vote Method", "Vote method", "Method"
            ]
            toggle_name = "Vote Method"
            log_debug(f"[DEBUG] About to toggle second button: {toggle_name}")
            btn, idx = coordinator.get_best_button_advanced(
                page=page,
                contest_title=contest_title_for_button,
                keywords=vote_method_keywords,
                context={**html_context, "toggle_name": toggle_name},
                confirm_button_callback=prompt.confirm_button_callback,
                prompt_user_for_button=prompt.prompt_user_for_button,
                learning_mode=True,
            )
            if btn and "element_handle" in btn:
                element = btn["element_handle"]
                # Only click if not already clicked by coordinator (learning mode)
                if btn.get("selector") not in coordinator.clicked_button_selectors:
                    if element.is_visible() and element.is_enabled():
                        try:
                            log_debug(f"[blue][DEBUG] Clicking button: '{btn.get('label', '')}' for toggle '{toggle_name}'")
                            element.click(timeout=5000)
                            page.wait_for_timeout(3000)
                            log_debug(f"[green][DEBUG] Button click for '{toggle_name}' completed.[/green]")
                            coordinator.clicked_button_selectors.add(btn.get("selector"))
                        except Exception as e:
                            log_error(f"[red][ERROR] Failed to click button '{btn.get('label', '')}': {e}[/red]")
                    else:
                        log_warning(f"[yellow][WARNING] Button '{btn.get('label', '')}' is not clickable (visible={element.is_visible()}, enabled={element.is_enabled()})[/yellow]")
                else:
                    log_debug(f"[yellow][DEBUG] Button '{btn.get('label', '')}' was already clicked by learning mode.[/yellow]")
            else:
                log_error(f"[red][ERROR] No suitable '{toggle_name}' button could be clicked.[/red]")
            log_debug(f"[DEBUG] Finished toggle second button: {toggle_name}")

            # --- Only autoscroll once, after all toggles ---
            autoscroll_until_stable(page)
            page.wait_for_timeout(3000)

            # --- 9. Extract ballot items using DOM scan and context/NLP ---
            html = page.content()
            with open("rockland_debug.html", "w", encoding="utf-8") as f:
                f.write(html)

            # Use the context coordinator and scan_html_for_context to extract everything
            context_result = scan_html_for_context(
                target_url=getattr(page, "url", None),
                page=page,
                coordinator=coordinator,
                debug=False,
            )

            segments = context_result.get("tagged_segments_with_attrs", [])
            panels = context_result.get("panels", [])

            # Fallback: group segments by panel label if panels missing
            if not panels and "tagged_segments_with_attrs" in context_result:
                from collections import defaultdict
                panels_by_heading = defaultdict(list)
                for seg in context_result["tagged_segments_with_attrs"]:
                    if seg.get("ml_label") == "panel":
                        panels_by_heading[seg.get("panel_heading", "Unknown")].append(seg)
                panels = [{"panel_heading": k, "tables": v} for k, v in panels_by_heading.items()]

            log_debug(f"[DEBUG] Found {len(panels)} panels after context/NLP pipeline.")
            if not panels:
                log_debug("[yellow][DEBUG] No panels found, falling back to direct table scan.[/yellow]")
                tables = page.locator("table")
                contest_title = html_context.get("selected_race") or html_context.get("contest_title") or "Unknown Contest"
                all_panel_rows = []
                all_panel_headers = set()
                for i in range(tables.count()):
                    table_html = tables.nth(i).evaluate("el => el.outerHTML")
                    extraction_context = {
                        "panel_heading": f"Table {i+1}",
                        "coordinator": coordinator,
                        "page": page,
                        "html_context": html_context,
                        "table_html": table_html,
                        "segments": segments,
                        "panels": [],
                    }
                    headers, data, entity_info = build_dynamic_table(
                        contest_title, None, None, coordinator, extraction_context
                    )
                    if headers and data:
                        all_panel_rows.extend(data)
                        all_panel_headers.update(headers)
            else:
                all_panel_rows = []
                all_panel_headers = set()
                for panel in panels:
                    panel_fields = {
                        "panel_heading": panel.get("panel_heading"),
                        "Precinct": panel.get("Precinct"),
                        "district": panel.get("district"),
                        "panel_tag": panel.get("panel_tag"),
                        "fully_reported": panel.get("fully_reported", ""),
                        "ml_confidence": panel.get("ml_confidence"),
                        "association_log": panel.get("association_log"),
                        "panel_ml_label": panel.get("panel_tag"),
                    }
                    for table in panel.get("tables", []):
                        table_fields = {
                            "table_idx": table.get("table_idx"),
                            "table_html": table.get("table_html"),
                            "ml_panel_score": table.get("ml_panel_score"),
                        }
                        table_html = table.get("table_html")
                        if not table_html:
                            continue
                        extraction_context = {
                            **panel_fields,
                            **table_fields,
                            "coordinator": coordinator,
                            "page": page,
                            "html_context": html_context,
                            "segments": segments,
                            "panels": panels,
                        }
                        # Propagate contest and location fields
                        for field in ("selected_race", "state", "county", "year", "election_types"):
                            if field in html_context:
                                extraction_context[field if field != "selected_race" else "contest_title"] = html_context[field]
                        headers, data, entity_info = build_dynamic_table(
                            extraction_context.get("contest_title", "Unknown Contest"),
                            None,
                            None,
                            coordinator,
                            extraction_context
                        )
                        for row in data:
                            for k, v in extraction_context.items():
                                if k not in row and v is not None and k not in ("coordinator", "page", "html_context", "segments", "panels"):
                                    row[k] = v
                        precinct = extraction_context.get("Precinct") or extraction_context.get("panel_heading") or extraction_context.get("district")
                        if "Precinct" not in headers and precinct:
                            headers = ["Precinct"] + headers
                            for row in data:
                                row["Precinct"] = precinct
                        for row in data:
                            if "Precinct" not in row and precinct:
                                row["Precinct"] = precinct
                        all_panel_rows.extend(data)
                        all_panel_headers.update(headers)

            all_panel_headers = list(all_panel_headers)
            merged_headers, merged_data = harmonize_headers_and_data(all_panel_headers, all_panel_rows)

            # --- 10. Assemble headers and finalize output ---
            if not merged_data:
                log_error(f"[red][ERROR] No data could be parsed from ballot items or robust extraction.[/red]")
                return None, None, contest_title, {"skipped": True}

            metadata = {
                "state": html_context.get("state", "NY"),
                "county": html_context.get("county", "Rockland"),
                "race": contest_title,
                "source": getattr(page, "url", "Unknown"),
                "handler": "rockland",
            }
            if "year" in html_context:
                metadata["year"] = html_context["year"]
            if "election_types" in html_context:
                metadata["election_types"] = html_context["election_types"]

            result = finalize_election_output(merged_headers, merged_data, coordinator, contest_title, metadata["state"], metadata["county"], context=metadata)
            if isinstance(result, dict):
                metadata.update(result)
            return merged_headers, merged_data, contest_title, metadata