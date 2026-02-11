from typing import TYPE_CHECKING

from playwright.sync_api import Page

from .....Context_Integration.librarian import clean_for_json
from .....utils.browser_utils import (
    autoscroll_until_stable,
    safe_click,
    safe_is_enabled,
    safe_is_visible,
)
from .....utils.contest_selector import select_contest_auto_first
from .....utils.html_scanner import scan_html_for_context
from .....utils.logger_singleton import logger, prompt
from .....utils.output_utils import finalize_election_output
from .....utils.shared_logic import safe_get
from .....utils.table_builder import build_dynamic_table
from .....utils.table_core import harmonize_headers_and_data

if TYPE_CHECKING:
    from .....Context_Integration.context_coordinator import ContextCoordinator

BUTTON_SELECTORS = "button, a, [role='button'], input[type='button'], input[type='submit']"
context_cache = {}
accepted_buttons_cache = {}

def parse(page: Page = None, html_context: dict = None, coordinator: "ContextCoordinator" = None, context=None, session_id=None, logger=logger, **kwargs) -> tuple:
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
    from .....Context_Integration.context_coordinator import ContextCoordinator
    coordinator = ContextCoordinator()
    logger.info("[bold cyan][Rockland Handler] Parsing Rockland County Enhanced Voting page...[/bold cyan]")

    # --- 1. Scan HTML and organize context before contest selection ---
    context_result = scan_html_for_context(
        target_url=getattr(page, "url", None),
        page=page,
        coordinator=coordinator,
        session_id=session_id if session_id is not None else getattr(coordinator, "session_id", None),
        allow_duplicates=getattr(coordinator, "allow_duplicates", False),
        context_cache=context_cache,
        debug=False,  
        **kwargs      
    )
    
    state = context_result.get("state") or "NY"
    county = context_result.get("county") or "Rockland"
    year = context_result.get("year")
    for contest in safe_get(context_result, "contests", []):
        if safe_get(contest, "state", None) is None:
            contest["state"] = state
        if safe_get(contest, "county", None) is None:
            contest["county"] = county
        if safe_get(contest, "year", None) is None and year is not None:
            contest["year"] = year
        if session_id is not None:
            contest["session_id"] = session_id
            
    context_result = clean_for_json(context_result)
    result = coordinator.organize_and_enrich(context_result)
    if "organized" in result and "dom_parts" in result["organized"]:
        logger.debug("[DEBUG] dom_parts successfully organized.")
    else:
        logger.warning("[WARNING] dom_parts missing after organize_and_enrich.")
    selector_data = coordinator.get_for_selector()
    logger.debug("DEBUG: selector_data['contests']:", selector_data.get("contests", []))
    # --- 3. Contest selection ---
    context_for_selector = {
        "state": state,
        "county": county,
        "year": year,
        "contests": context_result.get("contests", []),
        **{k: v for k, v in html_context.items() if k not in ("state", "county", "year", "contests")}
    }
    if session_id is not None:
        context_for_selector["session_id"] = session_id
        
    selected = select_contest_auto_first(
        coordinator=coordinator,
        context=context_for_selector,
        session_id=session_id,
        allow_multiple=False,
        force_interactive=False
    )
    
    if not selected:
        logger.warning("[red]No contest selected. Skipping.[/red]")
        return None, None, None, {"skipped": True}
    # If multiple contests, process each (return first result or aggregate as needed)
    if isinstance(selected, list):
        results = []
        for contest in selected:
            user_selected_contest = contest if isinstance(contest, dict) else {"title": contest}
            html_context["selected_race"] = user_selected_contest.get("title")
            logger.info(f"[cyan][INFO] Processing contest: {user_selected_contest.get('title')}[/cyan]")

            # --- Button toggles for this contest ---
            contest_for_button = user_selected_contest

            # --- Toggle "View results by election district" ---
            election_district_keywords = [
                r"view results? by election district[\s:]*$", "View results by election district", 
                "results by election district",  "election district", 
                "View results by"
            ]
            toggle_name = "View results by election district"
            logger.debug(f"[DEBUG] About to toggle first button: {toggle_name}")
            btn1, idx1 = coordinator.get_best_button_advanced(
                page=page,
                contest=contest_for_button,
                keywords=election_district_keywords,
                context={**html_context, "toggle_name": toggle_name},
                confirm_button_callback=prompt.confirm_button_callback,
                prompt_user_for_button=prompt.prompt_user_for_button,
                learning_mode=True,
            )
            if btn1 and "element_handle" in btn1:
                element = btn1["element_handle"]
                # Only click if not already clicked by coordinator (learning mode)
                if btn1.get("selector") not in coordinator.clicked_button_selectors:
                    if safe_is_visible(element, logger) and safe_is_enabled(element, logger):
                        try:
                            logger.debug(f"[blue][DEBUG] Clicking button: '{btn1.get('label', '')}' for toggle '{toggle_name}'")
                            safe_click(element, logger)
                            page.wait_for_timeout(3000)
                            logger.debug(f"[green][DEBUG] Button click for '{toggle_name}' completed.[/green]")
                            coordinator.clicked_button_selectors.add(btn1.get("selector"))
                        except Exception as e:
                            logger.error(f"[red][ERROR] Failed to click button '{btn1.get('label', '')}': {e}[/red]")
                    else:
                        logger.warning(f"[yellow][WARNING] Button '{btn1.get('label', '')}' is not clickable (visible={safe_is_visible(element, logger)}, enabled={safe_is_enabled(element, logger)})[/yellow]")
                else:
                    logger.debug(f"[yellow][DEBUG] Button '{btn1.get('label', '')}' was already clicked by learning mode.[/yellow]")
            else:
                logger.error(f"[red][ERROR] No suitable '{toggle_name}' button could be clicked.[/red]")

            logger.debug(f"[DEBUG] Finished toggle first button: {toggle_name}")

            # --- Toggle "Vote Method" ---
            vote_method_keywords = [
                "vote method", "Vote Method", "Vote method", "Method"
            ]
            toggle_name2 = "Vote Method"
            logger.debug(f"[DEBUG] About to toggle second button: {toggle_name}")
            btn2, idx2 = coordinator.get_best_button_advanced(
                page=page,
                contest=contest_for_button,
                keywords=vote_method_keywords,
                context={**html_context, "toggle_name": toggle_name2},
                confirm_button_callback=prompt.confirm_button_callback,
                prompt_user_for_button=prompt.prompt_user_for_button,
                learning_mode=True,
            )
            if btn2 and "element_handle" in btn2:
                element = btn2["element_handle"]
                # Only click if not already clicked by coordinator (learning mode)
                if btn2.get("selector") not in coordinator.clicked_button_selectors:
                    if safe_is_visible(element, logger) and safe_is_enabled(element, logger):
                        try:
                            logger.debug(f"[blue][DEBUG] Clicking button: '{btn2.get('label', '')}' for toggle '{toggle_name2}'")
                            safe_click(element, logger)
                            page.wait_for_timeout(3000)
                            logger.debug(f"[green][DEBUG] Button click for '{toggle_name2}' completed.[/green]")
                            coordinator.clicked_button_selectors.add(btn2.get("selector"))
                        except Exception as e:
                            logger.error(f"[red][ERROR] Failed to click button '{btn2.get('label', '')}': {e}[/red]")
                    else:
                        logger.warning(f"[yellow][WARNING] Button '{btn2.get('label', '')}' is not clickable (visible={safe_is_visible(element, logger)}, enabled={safe_is_enabled(element, logger)})[/yellow]")
                else:
                    logger.debug(f"[yellow][DEBUG] Button '{btn2.get('label', '')}' was already clicked by learning mode.[/yellow]")
            else:
                logger.error(f"[red][ERROR] No suitable '{toggle_name2}' button could be clicked.[/red]")
            logger.debug(f"[DEBUG] Finished toggle second button: {toggle_name2}")

            # --- Only autoscroll once, after all toggles ---
            autoscroll_until_stable(page, session_id=session_id)
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
                session_id=session_id if session_id is not None else getattr(coordinator, "session_id", None),
                allow_duplicates=getattr(coordinator, "allow_duplicates", False),
                context_cache=context_cache,
                debug=False,
                **kwargs
            )

            segments = context_result.get("tagged_segments_with_attrs", [])
            panels = context_result.get("panels", [])

            # Fallback: group segments by panel label if panels missing
            if not panels and "tagged_segments_with_attrs" in context_result:
                from collections import defaultdict
                panels_by_heading = defaultdict(list)
                for seg in safe_get(context_result, "tagged_segments_with_attrs", []):
                    if safe_get(seg, "ml_label", None) == "panel":
                        panels_by_heading[safe_get(seg, "panel_heading", "Unknown")].append(seg)
                panels = [{"panel_heading": k, "tables": v} for k, v in panels_by_heading.items()]

            logger.debug(f"[DEBUG] Found {len(panels)} panels after context/NLP pipeline.")
            if not panels:
                logger.debug("[yellow][DEBUG] No panels found, falling back to direct table scan.[/yellow]")
                tables = page.locator("table")
                contest = html_context.get("selected_race") or html_context.get("contest") or "Unknown Contest"
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
                    headers, data, _ = build_dynamic_table(
                        contest, None, None, coordinator, extraction_context
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
                    for table in safe_get(panel, "tables", []):
                        table_fields = {
                            "table_idx": safe_get(table, "table_idx"),
                            "table_html": safe_get(table, "table_html"),
                            "ml_panel_score": safe_get(table, "ml_panel_score"),
                        }
                        table_html = safe_get(table, "table_html")
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
                                extraction_context[field if field != "selected_race" else "contest"] = html_context[field]
                        headers, data, _ = build_dynamic_table(
                            extraction_context.get("contest", "Unknown Contest"),
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
                logger.error("[red][ERROR] No data could be parsed from ballot items or robust extraction.[/red]")
                return None, None, contest, {"skipped": True}

            metadata = {
                "state": html_context.get("state", "NY"),
                "county": html_context.get("county", "Rockland"),
                "race": contest,
                "source": getattr(page, "url", "Unknown"),
                "handler": "rockland",
                "session_id": session_id
            }
            if "year" in html_context:
                metadata["year"] = html_context["year"]
            if "election_types" in html_context:
                metadata["election_types"] = html_context["election_types"]

            result = finalize_election_output(merged_headers, merged_data, coordinator, contest, metadata["state"], metadata["county"], context=metadata)
            if isinstance(result, dict):
                metadata.update(result)
            results.append({
                "contest": user_selected_contest,
                "button_election_district": {
                    "button": btn1,
                    "index": idx1
                },
                "button_vote_method": {
                    "button": btn2,
                    "index": idx2
                },
                "headers": merged_headers,
                "data": merged_data,
                "metadata": metadata
            })
            return merged_headers, merged_data, contest, metadata