from playwright.sync_api import Page

from .....utils.contest_selector import select_contest
from .....utils.table_builder import build_dynamic_table
from .....utils.output_utils import finalize_election_output
from .....utils.shared_logger import rprint
from .....utils.shared_logic import autoscroll_until_stable
from .....utils.user_prompt import prompt_user_for_button, confirm_button_callback
from .....utils.html_scanner import extract_tagged_segments_with_attrs, extract_panel_table_hierarchy
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .....Context_Integration.context_coordinator import ContextCoordinator
import numpy as e
BUTTON_SELECTORS = "button, a, [role='button'], input[type='button'], input[type='submit']"
context_cache = {}
accepted_buttons_cache = {}

def parse(page: Page, coordinator: "ContextCoordinator", html_context: dict = None, non_interactive=False):
    """
    Rockland County handler: all logic in one place.
    - Scans HTML for context and contests
    - Lets user select contest
    - Toggles "View results by election district" and "Vote Method"
    - Autoscrolls as needed (only once, after all toggles)
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
    if isinstance(selected, list):
        results = []
        for contest in selected:
            user_selected_title = contest.get("title") if isinstance(contest, dict) else contest
            html_context["selected_race"] = user_selected_title
            rprint(f"[cyan][INFO] Processing contest: {user_selected_title}[/cyan]")

            # --- Button toggles for this contest ---
            contest_title_for_button = user_selected_title if user_selected_title else None

            # --- Toggle "View results by election district" ---
            election_district_keywords = [
                r"view results? by election district[\s:]*$", "View results by election district", 
                "results by election district",  "election district", 
                "View results by"
            ]
            toggle_name = "View results by election district"
            rprint(f"[DEBUG] About to toggle first button: {toggle_name}")
            btn, idx = coordinator.get_best_button_advanced(
                page=page,
                contest_title=contest_title_for_button,
                keywords=election_district_keywords,
                context={**html_context, "toggle_name": toggle_name},
                confirm_button_callback=confirm_button_callback,
                prompt_user_for_button=prompt_user_for_button,
                learning_mode=True,
            )
            if btn and "element_handle" in btn:
                element = btn["element_handle"]
                # Only click if not already clicked by coordinator (learning mode)
                if btn.get("selector") not in coordinator.clicked_button_selectors:
                    if element.is_visible() and element.is_enabled():
                        try:
                            rprint(f"[blue][DEBUG] Clicking button: '{btn.get('label', '')}' for toggle '{toggle_name}'")
                            element.click(timeout=5000)
                            page.wait_for_timeout(3000)
                            rprint(f"[green][DEBUG] Button click for '{toggle_name}' completed.[/green]")
                            coordinator.clicked_button_selectors.add(btn.get("selector"))
                        except Exception as e:
                            rprint(f"[red][ERROR] Failed to click button '{btn.get('label', '')}': {e}[/red]")
                    else:
                        rprint(f"[yellow][WARNING] Button '{btn.get('label', '')}' is not clickable (visible={element.is_visible()}, enabled={element.is_enabled()})[/yellow]")
                else:
                    rprint(f"[yellow][DEBUG] Button '{btn.get('label', '')}' was already clicked by learning mode.[/yellow]")
            else:
                rprint(f"[red][ERROR] No suitable '{toggle_name}' button could be clicked.[/red]")

            rprint(f"[DEBUG] Finished toggle first button: {toggle_name}")

            # --- Toggle "Vote Method" ---
            vote_method_keywords = [
                "vote method", "Vote Method", "Vote method", "Method"
            ]
            toggle_name = "Vote Method"
            rprint(f"[DEBUG] About to toggle second button: {toggle_name}")
            btn, idx = coordinator.get_best_button_advanced(
                page=page,
                contest_title=contest_title_for_button,
                keywords=vote_method_keywords,
                context={**html_context, "toggle_name": toggle_name},
                confirm_button_callback=confirm_button_callback,
                prompt_user_for_button=prompt_user_for_button,
                learning_mode=True,
            )
            if btn and "element_handle" in btn:
                element = btn["element_handle"]
                # Only click if not already clicked by coordinator (learning mode)
                if btn.get("selector") not in coordinator.clicked_button_selectors:
                    if element.is_visible() and element.is_enabled():
                        try:
                            rprint(f"[blue][DEBUG] Clicking button: '{btn.get('label', '')}' for toggle '{toggle_name}'")
                            element.click(timeout=5000)
                            page.wait_for_timeout(3000)
                            rprint(f"[green][DEBUG] Button click for '{toggle_name}' completed.[/green]")
                            coordinator.clicked_button_selectors.add(btn.get("selector"))
                        except Exception as e:
                            rprint(f"[red][ERROR] Failed to click button '{btn.get('label', '')}': {e}[/red]")
                    else:
                        rprint(f"[yellow][WARNING] Button '{btn.get('label', '')}' is not clickable (visible={element.is_visible()}, enabled={element.is_enabled()})[/yellow]")
                else:
                    rprint(f"[yellow][DEBUG] Button '{btn.get('label', '')}' was already clicked by learning mode.[/yellow]")
            else:
                rprint(f"[red][ERROR] No suitable '{toggle_name}' button could be clicked.[/red]")
            rprint(f"[DEBUG] Finished toggle second button: {toggle_name}")

            # --- Only autoscroll once, after all toggles ---
            autoscroll_until_stable(page)
            page.wait_for_timeout(3000)

            # --- 9. Extract ballot items using DOM scan and context/NLP ---
            html = page.content()
            with open("rockland_debug.html", "w", encoding="utf-8") as f:
                f.write(html)
            rprint(f"[DEBUG] HTML length: {len(html)}")
            rprint(f"[DEBUG] HTML after toggles (first 1000 chars):\n{html[:1000]}")
            segments = extract_tagged_segments_with_attrs(html)
            for seg in segments:
                if seg["tag"] == "table":
                    parent_idx = seg.get("parent_idx")
                    parent = segments[parent_idx] if parent_idx is not None else None
                    print("TABLE SEGMENT:")
                    print("  Table classes:", seg.get("classes"))
                    if parent:
                        print("  Parent tag/classes:", parent["tag"], parent.get("classes"))            
            rprint(f"[DEBUG] All segment tags: {[seg['tag'] for seg in segments]}")
            rprint(f"[DEBUG] Extracted {len(segments)} segments. Tags: {[seg['tag'] for seg in segments[:20]]}")
            panels = extract_panel_table_hierarchy(segments)
            
            rprint(f"[DEBUG] Found {len(panels)} panels after extract_panel_table_hierarchy.")
            for i, panel in enumerate(panels):
                rprint(f"[DEBUG] Panel {i}: heading={panel.get('panel_heading')}, tables={len(panel.get('tables', []))}")

            if not panels:
                rprint("[red][DEBUG] No panels found in HTML. Check extract_panel_table_hierarchy logic or input HTML.")
                
            all_results = []
            contest_title = html_context.get("selected_race") or html_context.get("contest_title") or "Unknown Contest"
            state = html_context.get("state", "NY")
            county = html_context.get("county", "Rockland")

            for panel in panels:
                district = panel["panel_heading"] or "Unknown District"
                for table in panel["tables"]:
                    table_html = table["table_html"]
                    extraction_context = {
                        "district": district,
                        "panel_heading": panel["panel_heading"],
                        "panel_tag": panel["panel_tag"],
                        "coordinator": coordinator,
                        "page": page,
                        "html_context": html_context,
                        "table_html": table_html,
                        "fully_reported": panel.get("fully_reported", ""),
                    }
                    # Optionally, you can pass table_html to table_builder if you adapt it to accept raw HTML
                    headers, data, entity_info = build_dynamic_table(
                        contest_title, [], [], coordinator, extraction_context
                    )
                    # Add district to each row for context
                    for row in data:
                        row["District"] = district
                    if "District" not in headers:
                        headers = ["District"] + headers
                    all_results.append((headers, data, contest_title, entity_info))

            # --- 10. Assemble headers and finalize output ---
            if not all_results:
                rprint(f"[red][ERROR] No data could be parsed from ballot items or robust extraction.[/red]")
                return None, None, contest_title, {"skipped": True}

            # Merge all results
            merged_headers = set()
            merged_data = []
            for headers, data, _, _ in all_results:
                merged_headers.update(headers)
                merged_data.extend(data)
            merged_headers = sorted(merged_headers)

            metadata = {
                "state": state,
                "county": county,
                "race": contest_title,
                "source": getattr(page, "url", "Unknown"),
                "handler": "rockland",
            }
            if "year" in html_context:
                metadata["year"] = html_context["year"]
            if "election_type" in html_context:
                metadata["election_type"] = html_context["election_type"]

            result = finalize_election_output(merged_headers, merged_data, coordinator, contest_title, state, county, context=metadata)
            if isinstance(result, dict):
                metadata.update(result)
            return merged_headers, merged_data, contest_title, metadata