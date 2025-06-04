from playwright.sync_api import Page

from .....utils.contest_selector import select_contest
from .....utils.table_builder import build_dynamic_table
from .....utils.table_core import robust_table_extraction
from .....utils.output_utils import finalize_election_output
from .....utils.shared_logger import rprint
from .....utils.shared_logic import autoscroll_until_stable
from .....utils.user_prompt import prompt_user_for_button, confirm_button_callback
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

            # --- 9. Extract ballot items using DOM scan and context/NLP ---
            contest_title = user_selected_title
            entities = coordinator.extract_entities(contest_title)
            locations = [ent for ent, label in entities if label in ("GPE", "LOC", "FAC", "ORG") or "district" in ent.lower()]
            expected_location = locations[0] if locations else None

            extraction_context = {
                "contest_title": contest_title,
                "entities": entities,
                "expected_location": expected_location,
                "html_context": html_context,
                "coordinator": coordinator,
                "page": page,
                # Add more as needed
            }
            html_context["coordinator"] = coordinator
            
            headers, data = build_dynamic_table(
                contest_title, [], [], coordinator, html_context
            )

            if not data:
                rprint(f"[red][ERROR] No data could be parsed from ballot items or robust extraction.[/red]")
                results.append((None, None, contest_title, {"skipped": True}))
                continue

            # --- 10. Assemble headers and finalize output ---
            headers = sorted(
                set(
                    k for row in data for k in row.keys() if k is not None
                )
            )
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
            if isinstance(result, dict):
                if "csv_path" in result:
                    metadata["output_file"] = result["csv_path"]
                if "metadata_path" in result:
                    metadata["metadata_path"] = result["metadata_path"]
                metadata.update(result)
            results.append((headers, data, contest_title, metadata))
        return results[0] if results else (None, None, None, {"skipped": True})