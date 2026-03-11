import csv
import os
from pathlib import Path

from ....config import BASE_DIR
from ....utils.browser_utils import (
    safe_click,
    safe_inner_text,
    safe_query_selector_all,
    safe_wait_for_timeout,
)
from ....utils.logger_singleton import logger
from ....utils.output_utils import finalize_election_output
from ....utils.shared_logic import (
    safe_get,
    safe_isdigit,
    safe_lower,
    safe_replace,
    safe_strip,
)

INPUT_DIR = Path(os.path.join(BASE_DIR, "input"))
OUTPUT_DIR = Path(os.path.join(BASE_DIR, "output"))

def apply_navigation_steps(page, config):
    steps = safe_get(config, "nav_actions", [])
    for step in steps:
        try:
            step_type = safe_get(step, "type_", "")
            selector = safe_get(step, "selector", "")
            delay = safe_get(step, "delay", 1000)
            seconds = safe_get(step, "seconds", 1)
            if step_type == "click":
                el = safe_query_selector_all(page, selector)
                el = el[0] if el else None
                if el:
                    logger.info(f"[NAV] Clicking {selector}")
                    safe_click(el, logger)
                    safe_wait_for_timeout(page, delay, logger)
            elif step_type == "wait":
                logger.info(f"[NAV] Waiting {seconds}s")
                safe_wait_for_timeout(page, seconds * 1000, logger)
        except Exception as e:
            logger.warning(f"[NAV] Step failed: {step} — {e}")

def parse(page=None, html_context=None, coordinator=None, context=None, session_id=None, **kwargs):
    html_context = html_context if isinstance(html_context, dict) else {}
    config = safe_get(html_context, "config", {})
    logger.info("[PA Handler] Contest routing active — using shared contest context with state-level extraction.")

    # STEP 1: Navigation (if needed)
    apply_navigation_steps(page, config)

    header_text = safe_get(html_context, "selected_race", "Unknown")
    logger.warning(f"[bold yellow]Detected election:[/bold yellow] {header_text}")
    resp = safe_lower(safe_strip(input("Do you want to continue parsing this election's contests? (y/n): ")))
    if resp != "y":
        logger.info("[cyan]Election skipped. Exploring other available elections...[/cyan]")
        try:
            # Update to use safe_query_selector_all for elections_toggle
            elections_toggle = safe_query_selector_all(page, "a[aria-label='Elections']")
            elections_toggle = elections_toggle[0] if elections_toggle else None
            if elections_toggle:
                safe_click(elections_toggle, logger)
                safe_wait_for_timeout(page, 1000, logger)
                race_links = safe_query_selector_all(page, "ul.dropdown-menu li a")
                for i, link in enumerate(race_links):
                    label = safe_strip(safe_inner_text(link, logger))
                    logger.info(f"[{i}] {label}")
                choice = safe_strip(input("Select an election to load by index: "))
                if safe_isdigit(choice):
                    idx = int(choice)
                    safe_click(race_links[idx], logger)
                    safe_wait_for_timeout(page, 3000, logger)
                else:
                    logger.warning("[PA] Invalid index input for election selection.")
            else:
                logger.warning("[PA] Elections dropdown not found.")
        except Exception as e:
            logger.warning(f"[PA] Failed to expand Elections menu or load selection: {e}")

    logger.info("[INFO] Pennsylvania handler activated. Waiting for CSV download logic.")
    apply_navigation_steps(page, config)

    # Click into County Breakdown view if flagged by scanner
    if safe_get(config, "requires_county_click", False):
        try:
            logger.info("[PA] Clicking County Breakdown link based on scanner signal...")
            county_link = safe_query_selector_all(page, "a:has-text('County Breakdown')")
            county_link = county_link[0] if county_link else None
            if county_link:
                safe_click(county_link, logger)
                safe_wait_for_timeout(page, 4000, logger)
                logger.info("[PA] County-level view loaded.")
            else:
                logger.warning("[PA] County Breakdown link not found.")
        except Exception as e:
            logger.warning(f"[PA] Failed to click County Breakdown link: {e}")

    # Look for a CSV file in the input directory
    try:
        csv_files = [f for f in os.listdir(INPUT_DIR) if safe_lower(f).endswith(".csv")]
    except Exception as e:
        logger.error(f"[ERROR] Could not list input directory: {e}")
        return [], [], "Pennsylvania (CSV not found)", {}

    if not csv_files:
        logger.error(f"[ERROR] No CSV files found in input directory: {INPUT_DIR}")
        return [], [], "Pennsylvania (CSV not found)", {}

    # If multiple CSVs, prompt user to select
    if len(csv_files) > 1:
        logger.warning("[yellow]Multiple CSV files found in input. Please select one:[/yellow]")
        for i, fname in enumerate(csv_files):
            logger.info(f"  [bold cyan][{i}][/bold cyan] {fname}")
        try:
            idx_input = safe_strip(input("Select CSV file index: "))
            if not safe_isdigit(idx_input):
                logger.error("[ERROR] Invalid selection (not a digit).")
                return [], [], "Pennsylvania (CSV selection error)", {}
            idx = int(idx_input)
            if idx < 0 or idx >= len(csv_files):
                logger.error("[ERROR] Index out of range.")
                return [], [], "Pennsylvania (CSV selection error)", {}
            csv_path = INPUT_DIR / csv_files[idx]
        except Exception:
            logger.error("[ERROR] Invalid selection.")
            return [], [], "Pennsylvania (CSV selection error)", {}
    else:
        csv_path = INPUT_DIR / csv_files[0]

    if not csv_path.exists():
        logger.error(f"[ERROR] CSV file not found: {csv_path}")
        return [], [], "Pennsylvania (CSV not found)", {}

    data = []
    headers = []

    try:
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            if reader.fieldnames is None:
                logger.error("[ERROR] CSV file appears to be empty or missing headers.") 
                return [], [], "Pennsylvania CSV Missing Headers", {}
            headers = reader.fieldnames
            for row in reader:
                data.append(row)

            # Compute a grand total row for numeric columns
            numeric_columns = [
                h for h in headers
                if all(
                    safe_isdigit(
                        safe_strip(
                            safe_lower(
                                safe_replace(
                                    safe_replace(
                                        safe_get(row, h, ''), ',', ''
                                    ), '.', ''
                                )
                            )
                        )
                    ) for row in data
                )
            ]
            totals = {h: 0 for h in numeric_columns}
            for row in data:
                for h in numeric_columns:
                    try:
                        val = safe_strip(
                            safe_lower(
                                safe_replace(safe_get(row, h, "0"), ",", "")
                            )
                        )
                        totals[h] += int(val) if safe_isdigit(val) else 0
                    except Exception:
                        continue

            totals_row = {key: "" for key in headers}
            for h in numeric_columns:
                totals_row[h] = str(totals[h])
            if headers:
                totals_row[headers[0]] = "Grand Total"
            data.append(totals_row)

        contest = header_text if header_text else "Pennsylvania County Results"
        metadata = {
            "state": "PA",
            "county": safe_get(html_context, "county", "Unknown"),
            "handler": "pennsylvania",
            "race": contest if contest else "Unknown"
        }

        result = finalize_election_output(headers, data, contest, metadata)
        contest = safe_get(result, "contest", contest)
        metadata = safe_get(result, "metadata", metadata)
        return headers, data, contest, metadata

    except Exception as e:
        logger.error(f"[ERROR] Failed to read or write CSV: {e}")
        return [], [], "Pennsylvania CSV Parse Error", {}