# handlers/formats/html_handler.py
# ==============================================================
# Fallback handler for generic HTML parsing.
# Used when structured formats (JSON, CSV, PDF) are not present.
# This routes through the state_router using html_scanner context.
# ==============================================================

from playwright.sync_api import Page
from utils.shared_logic import normalize_text
from state_router import get_handler as get_state_handler, resolve_state_handler
from utils.shared_logger import log_info, log_debug, log_warning, log_error
from rich import print as rprint
from typing import Optional
from utils.html_scanner import get_detected_races_from_context
import re

# This fallback HTML handler is invoked when no state or county-specific handler is matched.
# It handles race prompting, HTML table fallback extraction, and potential re-routing.

def parse(page: Page, html_context: Optional[dict] = None):
    log_info("[HTML Handler] Executing fallback HTML handler...")

    # Early exit if format_router already completed the parse
    if html_context and html_context.get("format_handler_complete"):
        log_info("[HTML Handler] Format handler previously completed parsing. Skipping HTML handler.")
        return None, None, None, {"skipped": True}

    # STEP 1 — Extract or prompt contest race title centrally here
    if html_context is None:
        html_context = {}

    # Check if race already selected
    race_title = html_context.get("selected_race")

    # If not selected, and available_races exist, prompt user
    if not race_title and "available_races" in html_context:
        rprint("[bold #eb4f43]Contest Races Detected:[/bold #eb4f43]")

        if "filter_race_year" in html_context:
            rprint(f"[dim]Active year filter: {html_context['filter_race_year']}[/dim]")
        if "filter_race_type" in html_context:
            rprint(f"[dim]Active type filter(s): {', '.join(html_context['filter_race_type'])}[/dim]")
        detected = get_detected_races_from_context(html_context)
        all_years = sorted(set(year for year, _, _ in detected))
        all_types = sorted(set(etype for _, etype, _ in detected))

        if len(all_years) > 1:
            filter_year = input("[PROMPT] Filter to a specific year? (e.g., 2024) Leave blank for all: ").strip()
            if filter_year:
                html_context["filter_race_year"] = filter_year

        if len(all_types) > 1:
            formatted_options = ", ".join(all_types)
            filter_type = input(f"[PROMPT] Filter to a specific election type? [Options: {formatted_options}] Leave blank for all: ").strip()
            if filter_type:
                html_context["filter_race_type"] = [ftype.strip() for ftype in filter_type.split(",") if ftype.strip()]      
        raw_detected = get_detected_races_from_context(html_context)
        state_filter_phrases = []
        resolved_handler = resolve_state_handler(html_context.get("url", html_context.get("source", "")))
        if resolved_handler and hasattr(resolved_handler, "NOISY_LABELS"):
            state_filter_phrases += [phrase.lower() for phrase in resolved_handler.NOISY_LABELS]
        if html_context.get("state"):
            handler = get_state_handler(state_abbreviation=html_context["state"], county_name=html_context.get("county"))
            if handler and hasattr(handler, "NOISY_LABELS"):
                state_filter_phrases += [phrase.lower() for phrase in handler.NOISY_LABELS]
        log_debug(f"[DEBUG] NOISY_LABELS active: {state_filter_phrases}")      
        log_info(f"[INFO] Filtering {len(raw_detected)} races using state noise filters: {state_filter_phrases}")
        filter_types = html_context.get("filter_race_type", [])
        if isinstance(filter_types, str):
            filter_types = [filter_types]
        filtered = [
            tup for tup in raw_detected
            if not any(phrase in normalize_text(tup[2]).lower() for phrase in state_filter_phrases)
            and (not filter_types or any(ftype.lower() in tup[1].lower() for ftype in filter_types))
        ] 
        filter_year = html_context.get("filter_race_year")
        if filter_year:
            filtered = [tup for tup in filtered if str(tup[0]) == str(filter_year)]    
        grouped = {}     
        for year, etype, race in filtered:
            grouped.setdefault(year, {}).setdefault(etype, []).append(race)
        flat_list = []
        total_detected = len(filtered)
        total_shown = 0
        for year in sorted(grouped):
            shown_this_year = 0
            rprint(f"[bold #87cefa]{year}[/bold #87cefa]")  # Added line break for clarity
            for etype in sorted(grouped[year]):
                rprint(f"  [bold cyan]{etype}[/bold cyan]")
                for race in grouped[year][etype]:
                    label = f"{year} • {etype} — {race}"
                    flat_list.append(label)
                                        # Skip redundant general labels when real races exist
                    lower_race = race.lower()
                    if any(keyword in lower_race for keyword in ["general election", "primary election", "special election"]):
                        continue
                    label = f"{year} • {etype} — {race}"
                    flat_list.append(label)
                    rprint(f"    [#eb4f43][{len(flat_list)-1}] {race}[/#eb4f43]")
                    total_shown += 1
                    shown_this_year += 1
                    total_shown += 1
        if total_shown == 0:
            rprint("[yellow]All contests filtered out. Try removing year/type filters.[/yellow]")
        rprint(f"[dim]Displayed {total_shown} of {total_detected} detected contests after filters.[/dim]")
        rprint("[dim]Type 'show all' to ignore filters and view full race list.[/dim]")
        choice = input("[PROMPT] Enter contest index, 'all', 'show all', or leave blank to skip: ").strip().lower()
        if choice == "all":
            html_context["selected_races"] = flat_list
            rprint(f"[green]All {len(flat_list)} contests selected for batch processing.[/green]")
            return None, None, None, {"batch_mode": True, "selected_races": flat_list}
        elif choice == "show all":
            html_context.pop("filter_race_type", None)
            html_context.pop("filter_race_year", None)
            html_context["ignore_filters"] = True
            return parse(page, html_context)
        elif choice.isdigit() and int(choice) < len(flat_list):
            race_title = flat_list[int(choice)]
            html_context["selected_race"] = race_title
        else:
            rprint("[yellow]No contest selected. Skipping HTML parsing.[/yellow]")
            return None, None, None, {"skipped": True}

    # STEP 2 — Infer state dynamically if not present
    if not html_context or 'state' not in html_context or html_context['state'] == 'Unknown':
        if html_context is None:
            html_context = {}
        handler_module = resolve_state_handler(page.url)
        if not handler_module and "raw_text" in html_context:
            combined = page.url + " " + html_context["raw_text"]
            handler_module = resolve_state_handler(combined)
        if handler_module:
            inferred_state = handler_module.__name__.split(".")[-1].upper()
            html_context['state'] = inferred_state
            log_info(f"[INFO] Inferred state via resolve_state_handler: {inferred_state}")
        else:  
            log_warning("[WARN] Could not infer state from URL or page text.")
    log_info(f"[INFO] Inferred state: {html_context.get('state', 'Unknown')}")

    # STEP 3 — Redirect to state/county-specific handler if matched
    log_debug(f"[DEBUG] Attempting to route using get_handler(state='{html_context.get('state')}', county='{html_context.get('county')}')")
    state_handler = get_state_handler(
        state_abbreviation=html_context.get("state"),
        county_name=html_context.get("county")
    )
    if state_handler:
        log_info(f"[HTML Handler] Redirecting to handler for {html_context.get('state')} / {html_context.get('county') or 'state-default'}...")
        return state_handler.parse(page, html_context)

    # STEP 4 — Final fallback: Basic HTML table extraction if no route matched
    try:
        table = page.query_selector("table")
        if not table:
            raise RuntimeError("No table found on the page.")
        headers = [th.inner_text().strip() for th in table.query_selector_all("thead tr th")]
        rows = []
        for row in table.query_selector_all("tbody tr"):
            cells = row.query_selector_all("td")
            row_data = {headers[i]: cells[i].inner_text().strip() for i in range(min(len(headers), len(cells)))}
            rows.append(row_data)
    except Exception as e:
        log_error(f"[ERROR] Failed to extract table from page: {e}")
        return None, None, None, {"error": str(e)}

    # STEP 5 — Output metadata for this HTML-based parse session
    clean_race = normalize_text(race_title).strip().lower() if race_title else "unknown"
    clean_race = re.sub(r'[\\/:*?"<>|]', '_', clean_race)

    metadata = {
        "race": clean_race,
        "source": page.url,
        "handler": "html_handler",
        "state": html_context.get("state", "Unknown"),
        "county": html_context.get("county", None),
        "year": html_context.get("year", "Unknown")
    }

    return race_title, headers, rows, metadata
# End of file