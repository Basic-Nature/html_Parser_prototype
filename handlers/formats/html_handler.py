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
    # Early exit if JSON source detected
    if html_context and html_context.get("json_source"):
        log_info("[HTML Handler] JSON source detected. Skipping HTML handler.")
        return None, None, None, {"skipped": True}
    # STEP 1 — Extract or prompt contest race title centrally here
    if html_context is None:
        html_context = {}
    if "raw_text" in html_context:
        # Attempt to extract    
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
        state_regex_filters = []

        resolved_handler = resolve_state_handler(html_context.get("url", html_context.get("source", "")))
        if resolved_handler:
            state_filter_phrases += getattr(resolved_handler, "NOISY_LABELS", [])
            state_regex_filters += getattr(resolved_handler, "NOISY_LABEL_PATTERNS", [])

        if html_context.get("state"):
            handler = get_state_handler(state_abbreviation=html_context["state"], county_name=html_context.get("county"))
            if handler:
                state_filter_phrases += getattr(handler, "NOISY_LABELS", [])
                state_regex_filters += getattr(handler, "NOISY_LABEL_PATTERNS", [])

        state_filter_phrases = [phrase.lower() for phrase in state_filter_phrases]

        # precompile regex patterns for efficiency
        compiled_patterns = [re.compile(pat, re.IGNORECASE) for pat in state_regex_filters]

        def is_noisy_race(race_clean):
            for phrase in state_filter_phrases:
                if phrase in race_clean:
                    return True
            for pattern in compiled_patterns:
                if pattern.search(race_clean):
                    return True
            return False

        log_debug(f"[DEBUG] NOISY_LABELS active: {state_filter_phrases}")
        log_debug(f"[DEBUG] NOISY_LABEL_PATTERNS active: {state_regex_filters}")
        log_info(f"[INFO] Filtering {len(raw_detected)} races using state noise filters")

        filter_types = html_context.get("filter_race_type", [])
        if isinstance(filter_types, str):
            filter_types = [filter_types]
        if isinstance(filter_types, list):
            filter_types = [ftype.strip() for ftype in filter_types if ftype.strip()]
        if not filter_types:
            filter_types = [ftype for ftype in all_types if ftype not in ["General Election", "Primary Election"]]

        filtered = []
        for tup in raw_detected:
            if not tup or len(tup) < 3:
                continue
            race_clean = re.sub(r'[\s:]+', ' ', normalize_text(tup[2]).lower()).strip()
            

            # Skip year/type titles like "2024 General Election" that match year + type
            context_year = html_context.get("filter_race_year") or html_context.get("year")
            if isinstance(context_year, list):
                context_year = context_year[0] if context_year else None
            if isinstance(context_year, int):
                context_year = str(context_year)
            if isinstance(context_year, str):
                context_year = re.sub(r'[\s:]+', ' ', normalize_text(context_year).lower()).strip()
            if context_year and context_year.isdigit():
                context_year = context_year.split(" ")[0]
            if context_year and len(context_year) == 4:
                context_year = context_year.split(" ")[0]
            context_type = html_context.get("filter_race_type")
            if isinstance(context_type, list):
                context_type = context_type[0] if context_type else None
            if isinstance(context_type, str):
                context_type = re.sub(r'[\s:]+', ' ', normalize_text(context_type).lower()).strip()
            if context_year and context_type:
                if race_clean == f"{context_year} {context_type}".lower():
                    log_debug(f"[SKIP] Skipping self-referential header: {race_clean}")
                    continue    

            if is_noisy_race(race_clean):
                
                if "View results" in race_clean:
                    log_warning(f"[WARN] Possibly unfiltered navigation phrase: {race_clean}")
                    
                continue
            

            if filter_types and not any(ftype.lower() in tup[1].lower() for ftype in filter_types):
                continue

            filtered.append(tup)
            

        filter_year = html_context.get("filter_race_year")
        if isinstance(filter_year, list):
            filter_year = filter_year[0] if filter_year else None
        if isinstance(filter_year, str):
            filter_year = re.sub(r'[\s:]+', ' ', normalize_text(filter_year).lower()).strip()
        if isinstance(filter_year, int):
            filter_year = str(filter_year)
        if filter_year:
            filtered = [tup for tup in filtered if str(tup[0]) == str(filter_year)]
        if not filtered:
            rprint("[yellow]No contests match the selected filters. Try removing them.[/yellow]")
            return None, None, None, {"skipped": True}
        rprint(f"[bold #eb4f43]Filtered contests: {len(filtered)}[/bold #eb4f43]")
        rprint(f"[dim]Filtered by year: {filter_year}[/dim]")
        rprint(f"[dim]Filtered by type: {', '.join(filter_types)}[/dim]")
        rprint(f"[dim]Filtered by noisy labels: {', '.join(state_filter_phrases)}[/dim]")
        rprint(f"[dim]Filtered by noisy patterns: {', '.join(state_regex_filters)}[/dim]")

        grouped = {}
        # Group by year and type
        # This is a nested dictionary where the first key is the year and the second key is the type
        for year, etype, race in filtered:
            grouped.setdefault(year, {}).setdefault(etype, []).append(race)
        # Sort
        for year in grouped:
            for etype in grouped[year]:
                grouped[year][etype] = sorted(grouped[year][etype], key=lambda x: normalize_text(x).lower())
        # Sort by year and type
        grouped = {year: {etype: grouped[year][etype] for etype in sorted(grouped[year])} for year in sorted(grouped)}
        # Print the grouped
        rprint("[bold #87cefa]Grouped contests:[/bold #87cefa]")
        for year in sorted(grouped):
            rprint(f"[bold #87cefa]{year}[/bold #87cefa]")
            for etype in sorted(grouped[year]):
                rprint(f"  [bold cyan]{etype}[/bold cyan]")

        flat_list = []
        # Flatten the grouped list
        for year in sorted(grouped):
            for etype in sorted(grouped[year]):
                total_detected = len(filtered)
        total_shown = 0
        # Show
        rprint("[bold #87cefa]Available contests:[/bold #87cefa]")
        rprint(f"[bold #87cefa]Detected contests: {total_detected}[/bold #87cefa]") 
        rprint(f"[bold #87cefa]Filtered contests: {len(filtered)}[/bold #87cefa]")
        rprint(f"[bold #87cefa]Available contests: {len(flat_list)}[/bold #87cefa]")
        rprint(f"[dim]Detected contests: {total_detected}[/dim]")
        rprint(f"[dim]Filtered contests: {len(filtered)}[/dim]")
        rprint(f"[dim]Available contests: {len(flat_list)}[/dim]")
        rprint(f"[dim]Filtered by year: {filter_year}[/dim]")
        rprint(f"[dim]Filtered by type: {', '.join(filter_types)}[/dim]")
        rprint(f"[dim]Filtered by noisy labels: {', '.join(state_filter_phrases)}[/dim]")
        rprint(f"[dim]Filtered by noisy patterns: {', '.join(state_regex_filters)}[/dim]")
        seen = set()
        

        for year in sorted(grouped):
            rprint(f"[bold #87cefa]{year}[/bold #87cefa]")
            if year not in grouped:
                continue
            for etype in sorted(grouped[year]):
                if etype not in grouped[year]:
                    continue
                rprint(f"  [bold cyan]{etype}[/bold cyan]")
                shown_this_year = 0
                if shown_this_year > 0:
                    rprint(f"[dim]Displayed {shown_this_year} contests for {year} {etype}.[/dim]")
                if etype not in grouped[year]:
                    continue
                if not grouped[year][etype]:
                    rprint(f"[dim]No contests for {year} {etype}.[/dim]")
                    continue
                if len(grouped[year][etype]) > 1:
                    rprint(f"[dim]Multiple contests for {year} {etype}.[/dim]")
                for race in grouped[year][etype]:
                    label = f"{year} • {etype} — {race}"
                    key = normalize_text(label).strip().lower()
                    if key in seen:
                        rprint(f"    [dim]Duplicate contest: {label}[/dim]")
                        continue
                    seen.add(key)
                    flat_list.append(label)
                    if shown_this_year == 0:
                        rprint(f"    [#eb4f43][{len(flat_list)-1}] {label}[/#eb4f43]")
                    else:
                        
                        rprint(f"    [#eb4f43][{len(flat_list)-1}] {race}[/#eb4f43]")
                    total_shown += 1
                    if len(flat_list) > 1:
                        rprint(f"[dim]Displayed {shown_this_year} contests for {year} {etype}.[/dim]")
                    shown_this_year += 1
                if shown_this_year == 0:
                    rprint(f"[dim]No contests for {year} {etype}.[/dim]")
                if shown_this_year > 0:
                    rprint(f"[dim]Displayed {shown_this_year} contests for {year} {etype}.[/dim]")
        if len(flat_list) == 0:
            rprint("[yellow]No contests detected. Try removing year/type filters.[/yellow]")
        if len(flat_list) == 1:
            rprint("[yellow]Only one contest detected. No need to select.[/yellow]")
        if len(flat_list) > 1:
            rprint("[yellow]Multiple contests detected. Please select one.[/yellow]")
        if total_shown == 0:
            
            rprint("[yellow]All contests filtered out. Try removing year/type filters.[/yellow]")
        if total_shown == 1:
            rprint("[yellow]Only one contest detected. No need to select.[/yellow]")
        if total_shown > 1:
            rprint("[yellow]Multiple contests detected. Please select one.[/yellow]")
        if total_shown == 0:
            rprint("[yellow]No contests detected. Try removing year/type filters.[/yellow]")
        rprint(f"[dim]Displayed {total_shown} of {total_detected} detected contests after filters.[/dim]")
        rprint("[dim]Type 'show all' to ignore filters and view full race list.[/dim]")

        choice = input("[PROMPT] Enter contest index, 'all', 'show all', or leave blank to skip: ").strip().lower()
        if choice == "":
            rprint("[yellow]No contest selected. Skipping HTML parsing.[/yellow]")
            return None, None, None, {"skipped": True}
        if choice == "exit":
            rprint("[yellow]Exiting contest selection.[/yellow]")
            return None, None, None, {"exit": True}
        if choice == "skip":
            rprint("[yellow]Skipping contest selection.[/yellow]")
            return None, None, None, {"skipped": True}
        if choice == "none":
            rprint("[yellow]No contest selected. Skipping HTML parsing.[/yellow]")
            return None, None, None, {"skipped": True}
        if choice == "all":
            rprint("[yellow]All contests selected for batch processing.[/yellow]")
            html_context["selected_races"] = flat_list
            html_context["ignore_filters"] = True
            rprint(f"[green]All {len(flat_list)} contests selected for batch processing.[/green]")
            return None, None, None, {"batch_mode": True, "selected_races": flat_list}
        elif choice == "show all":
            rprint("[yellow]Ignoring filters and showing all contests.[/yellow]")
            html_context["selected_races"] = flat_list
            html_context["ignore_filters"] = True
            html_context.pop("filter_race_type", None)
            html_context.pop("filter_race_year", None)
            return parse(page, html_context)
        if choice.isdigit() and int(choice) >= len(flat_list):
            rprint(f"[yellow]Invalid contest index: {choice}. Skipping HTML parsing.[/yellow]")
            return None, None, None, {"skipped": True}
        if choice.isdigit() and int(choice) >= 0 and int(choice) < len(flat_list):
            rprint(f"[green]Contest selected: {flat_list[int(choice)]}[/green]")
            html_context["selected_race"] = flat_list[int(choice)]
            html_context["ignore_filters"] = True
        
        elif choice.isdigit() and int(choice) < len(flat_list): 
            rprint(f"[green]Contest selected: {flat_list[int(choice)]}[/green]")
            html_context["selected_race"] = flat_list[int(choice)]
            html_context["ignore_filters"] = True
            race_title = flat_list[int(choice)]
            html_context["selected_race"] = race_title
        else:
            rprint("[yellow]No contest selected. Skipping HTML parsing.[/yellow]")
            return None, None, None, {"skipped": True}

    # STEP 2 — Infer state dynamically if not present
    if not html_context or 'state' not in html_context or html_context['state'] == 'Unknown':
        log_info("[INFO] Inferring state from URL or page text...")
        if html_context is None:
            html_context = {}
        handler_module = resolve_state_handler(page.url)
        if not handler_module and "raw_text" in html_context:
            # Attempt to resolve state from raw text if available
            # This is a fallback if the URL doesn't provide enough context
            # Note: This is a basic heuristic and may not be accurate
            # In a real-world scenario, you might want to use a more sophisticated method
            # to extract the state from the text
            # For example, using regex to find state names or abbreviations
            # or using a library that can parse and understand the text better
            # This is a placeholder for the actual logic
            # that would be used to extract the state from the text
            # For now, we'll just log the raw text for debugging
            log_debug(f"[DEBUG] Raw text for state inference: {html_context['raw_text']}")
            # Attempt to resolve state from raw text
            combined = page.url + " " + html_context["raw_text"]
            handler_module = resolve_state_handler(combined)
        if handler_module and hasattr(handler_module, "get_state_from_text"):
            # Use the handler's method to extract state from text
            inferred_state = handler_module.get_state_from_text(html_context["raw_text"])
            if inferred_state:
                html_context['state'] = inferred_state
                log_info(f"[INFO] Inferred state via get_state_from_text: {inferred_state}")
        elif handler_module and hasattr(handler_module, "get_state_from_url"):
            # Use the handler's method to extract state from URL
            inferred_state = handler_module.get_state_from_url(page.url)
            if inferred_state:
                html_context['state'] = inferred_state
                log_info(f"[INFO] Inferred state via get_state_from_url: {inferred_state}")
        elif handler_module and hasattr(handler_module, "get_state_from_page"):
            # Use the handler's method to extract state from page
            inferred_state = handler_module.get_state_from_page(page)
            if inferred_state:
                html_context['state'] = inferred_state
                log_info(f"[INFO] Inferred state via get_state_from_page: {inferred_state}")
        elif handler_module and hasattr(handler_module, "get_state_from_context"):
            # Use the handler's method to extract state from context
            inferred_state = handler_module.get_state_from_context(html_context)
            if inferred_state:
                html_context['state'] = inferred_state
                log_info(f"[INFO] Inferred state via get_state_from_context: {inferred_state}")
        elif handler_module and hasattr(handler_module, "get_state_from_text"):
            # Use the handler's method to extract state from text
            inferred_state = handler_module.get_state_from_text(html_context["raw_text"])
            if inferred_state:
                html_context['state'] = inferred_state
                log_info(f"[INFO] Inferred state via get_state_from_text: {inferred_state}")
        elif handler_module and hasattr(handler_module, "get_state_from_url"):
            # Use the handler's method to extract state from URL
            inferred_state = handler_module.get_state_from_url(page.url)
            if inferred_state:
                html_context['state'] = inferred_state
                log_info(f"[INFO] Inferred state via get_state_from_url: {inferred_state}")
        if handler_module:
            # Extract state abbreviation from the module name
            # e.g., "handlers.states.new_york" -> "NY"
            inferred_state = handler_module.__name__.split(".")[-1].upper()
            if inferred_state:
                # Check if the inferred state is a valid abbreviation
                if len(inferred_state) == 2 and inferred_state.isalpha():
                    html_context['state'] = inferred_state
                    log_info(f"[INFO] Inferred state via module name: {inferred_state}")
                else:
                    log_warning(f"[WARN] Inferred state from module name is not a valid abbreviation: {inferred_state}")
        elif "state" in html_context:
            # Attempt to resolve state from the page text
            # This is a fallback if the URL doesn't provide enough context
            # Note: This is a basic heuristic and may not be accurate
            html_context['state'] = inferred_state
            log_info(f"[INFO] Inferred state via resolve_state_handler: {inferred_state}")
        else:
            log_warning("[WARN] Could not infer state from URL or page text.")
    log_info(f"[INFO] Inferred state: {html_context.get('state', 'Unknown')}")

    # STEP 3 — Redirect to state/county-specific handler if matched
    # Check if the state handler is already resolved
    if html_context.get("state") and html_context.get("county"):
        # If state and county are already set, use them directly
        log_debug(f"[DEBUG] State and county already set in context: {html_context['state']}, {html_context['county']}")
    log_debug(f"[DEBUG] Attempting to route using get_handler(state='{html_context.get('state')}', county='{html_context.get('county')}')")
    state_handler = get_state_handler(
        state_abbreviation=html_context.get("state"),
        county_name=html_context.get("county")
    )
    if state_handler:
        # Check if the handler has a parse method
        if hasattr(state_handler, "parse"):
            log_info(f"[HTML Handler] Redirecting to state handler: {state_handler.__name__}...")
            return state_handler.parse(page, html_context)
        else:
            log_warning(f"[WARN] State handler {state_handler.__name__} does not have a parse method.")
    elif html_context.get("state") and not html_context.get("county"):
        # If only the state is set, use the state handler directly
        log_info(f"[HTML Handler] Redirecting to state handler: {html_context['state']}...")
        state_handler = get_state_handler(state_abbreviation=html_context["state"])
        log_info(f"[HTML Handler] Redirecting to handler for {html_context.get('state')} / {html_context.get('county') or 'state-default'}...")
        return state_handler.parse(page, html_context)

    # STEP 4 — Final fallback: Basic HTML table extraction if no route matched
    try:
        table = page.query_selector("table")
        if not table:
            # Attempt to find a table with a specific class or ID
            table = page.query_selector("table#resultsTable, table.results-table")
        if not table:
            raise RuntimeError("No table found on the page.")
        # Extract
        headers = [th.inner_text().strip() for th in table.query_selector_all("thead tr th")]
        if not headers:
            # Attempt to find headers in the first row of the table
            headers = [th.inner_text().strip() for th in table.query_selector_all("tbody tr:first-child th")]
        if not headers:
            raise RuntimeError("No headers found in the table.")
        # Extract rows
        # Use a list comprehension to extract data from each row
        rows = []
        for row in table.query_selector_all("tbody tr"):
            # Check if the row is empty or contains only headers
            if not row.query_selector("td"):
                continue
            # Extract data from each cell in the row
            # Use a dictionary comprehension to create a dictionary for each row
            # Use the headers as keys and the cell values as values
            # Use the min function to avoid index errors if the number of headers and cells differ
            cells = row.query_selector_all("td")
            row_data = {headers[i]: cells[i].inner_text().strip() for i in range(min(len(headers), len(cells)))}
            rows.append(row_data)
        if not rows:
            raise RuntimeError("No rows found in the table.")
        log_info(f"[HTML Handler] Extracted {len(rows)} rows from the table.")
    except RuntimeError as e:
        log_error(f"[ERROR] Failed to extract table from page: {e}")
        return None, None, None, {"error": str(e)}
    except AttributeError as e:
        log_error(f"[ERROR] Failed to extract table from page: {e}")
        return None, None, None, {"error": str(e)}
    except TypeError as e:
        log_error(f"[ERROR] Failed to extract table from page: {e}")
        return None, None, None, {"error": str(e)}
    except ValueError as e:
        log_error(f"[ERROR] Failed to extract table from page: {e}")
        return None, None, None, {"error": str(e)}
    except KeyError as e:
        log_error(f"[ERROR] Failed to extract table from page: {e}")
        return None, None, None, {"error": str(e)}
    except IndexError as e:
        log_error(f"[ERROR] Failed to extract table from page: {e}")
        return None, None, None, {"error": str(e)}
    except Exception as e:
        # Catch-all for any other exceptions
        # This is a generic error handler for any unexpected errors
        log_error(f"[ERROR] Failed to extract table from page: {e}")
        return None, None, None, {"error": str(e)}

    # STEP 5 — Output metadata for this HTML-based parse session
    clean_race = normalize_text(race_title).strip().lower() if race_title else "unknown"
    # Normalize
    clean_race = re.sub(r'[\s:]+', ' ', clean_race).strip()
    # Remove special characters
    # This regex replaces any character that is not a word character (alphanumeric or underscore)
    # or whitespace with an underscore
    # This is useful for sanitizing
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
