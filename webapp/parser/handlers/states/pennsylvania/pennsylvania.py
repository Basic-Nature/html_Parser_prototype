# handlers/pennsylvania.py
# ==============================================================
# Handler for Pennsylvania election result pages that provide
# downloadable CSV files (e.g., county-level reporting portals).
# ==============================================================

import csv
import os
from datetime import datetime
from utils.shared_logger import logger
from rich import print as rprint
from utils.output_utils import get_output_path, format_timestamp, sanitize_filename, finalize_election_output

def apply_navigation_steps(page, config):
    steps = config.get("nav_actions", [])
    for step in steps:
        try:
            if step["type"] == "click":
                el = page.query_selector(step["selector"])
                if el:
                    logger.info(f"[NAV] Clicking {step['selector']}")
                    el.click()
                    page.wait_for_timeout(step.get("delay", 1000))
            elif step["type"] == "wait":
                logger.info(f"[NAV] Waiting {step['seconds']}s")
                page.wait_for_timeout(step["seconds"] * 1000)
        except Exception as e:
            logger.warning(f"[NAV] Step failed: {step} — {e}")


def handle(page, config, html_context=None):
    html_context = html_context or {}
    logger.info("[PA Handler] Contest routing active — using shared contest context with state-level extraction.")
    # STEP 1: Check election header after navigation is complete
    apply_navigation_steps(page, config)

    header_text = html_context.get("selected_race", "Unknown")
    rprint(f"[bold yellow]Detected election:[/bold yellow] {header_text}")
    resp = input("Do you want to continue parsing this election's contests? (y/n): ").strip().lower()
    if resp != "y":
        rprint("[cyan]Election skipped. Exploring other available elections...[/cyan]")
        try:
            elections_toggle = page.query_selector("a[aria-label='Elections']")
            if elections_toggle:
                elections_toggle.click()
                page.wait_for_timeout(1000)
                race_links = page.query_selector_all("ul.dropdown-menu li a")
                for i, link in enumerate(race_links):
                    label = link.inner_text().strip()
                    print(f"[{i}] {label}")
                choice = input("Select an election to load by index: ").strip()
                race_links[int(choice)].click()
                page.wait_for_timeout(3000)
            else:
                logger.warning("[PA] Elections dropdown not found.")
        except Exception as e:
            logger.warning(f"[PA] Failed to expand Elections menu or load selection: {e}")

        # Open the Elections dropdown
        try:
            elections_toggle = page.query_selector("a[aria-label='Elections']")
            if elections_toggle:
                elections_toggle.click()
                page.wait_for_timeout(1000)

                # List available race-year combinations
                race_links = page.query_selector_all("ul.dropdown-menu li a")
                for i, link in enumerate(race_links):
                    label = link.inner_text().strip()
                    print(f"[{i}] {label}")

                choice = input("Select an election to load by index: ").strip()
                race_links[int(choice)].click()
                page.wait_for_timeout(3000)
            else:
                logger.warning("[PA] Elections dropdown not found.")
        except Exception as e:
            logger.warning(f"[PA] Failed to expand Elections menu or load selection: {e}")
    logger.info("[INFO] Pennsylvania handler activated. Waiting for CSV download logic.")
    apply_navigation_steps(page, config)

    # Click into County Breakdown view if flagged by scanner
    if config.get("requires_county_click"):
        try:
            logger.info("[PA] Clicking County Breakdown link based on scanner signal...")
            county_link = page.query_selector("a:has-text('County Breakdown')")
            if county_link:
                county_link.click()
                page.wait_for_timeout(4000)
                logger.info("[PA] County-level view loaded.")
            else:
                logger.warning("[PA] County Breakdown link not found.")
        except Exception as e:
            logger.warning(f"[PA] Failed to click County Breakdown link: {e}")

    # Placeholder: assumes a CSV is manually downloaded
    csv_path = os.path.join("input", "pennsylvania_example.csv")

    if not os.path.exists(csv_path):
        logger.error(f"[ERROR] CSV file not found: {csv_path}")
        return "Pennsylvania (CSV not found)", [], [], {}

    data = []
    headers = []

    try:
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            if reader.fieldnames is None:
                logger.error("[ERROR] CSV file appears to be empty or missing headers.") 
                return "Pennsylvania CSV Missing Headers", [], [], {}
            headers = reader.fieldnames[2:] 
            for row in reader:
                data.append(row)

            # Compute a grand total row
            numeric_columns = [h for h in headers if all(row.get(h, '').replace(',', '').replace('.', '').isdigit() for row in data)]
            totals = {h: 0 for h in numeric_columns}
            for row in data:
                for h in numeric_columns:
                    try:
                        totals[h] += int(row.get(h, "0").replace(",", "").strip())
                    except:
                        continue

            totals_row = {key: "" for key in headers}
            for h in numeric_columns:
                totals_row[h] = str(totals[h])
            totals_row[headers[0]] = "Grand Total"
            data.append(totals_row)

        contest_title = header_text or "Pennsylvania County Results"

        metadata = {
            "state": "PA",
            "county": "",
            "handler": "pennsylvania",
            "race": contest_title
        }

        from utils.output_utils import finalize_election_output
        result = finalize_election_output(headers, data, metadata)
        return result.get("contest_title", contest_title), headers, data, result.get("metadata", metadata)

    except Exception as e:
        logger.error(f"[ERROR] Failed to read or write CSV: {e}")
        return "Pennsylvania CSV Parse Error", [], [], {}
