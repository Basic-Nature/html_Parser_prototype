# handlers/states/pennsylvania/pennsylvania.py
# ==============================================================
# Handler for Pennsylvania election result pages that provide
# downloadable CSV files (e.g., county-level reporting portals).
# ==============================================================

import os
from pathlib import Path
import csv
from ....utils.shared_logger import logger
from rich import print as rprint
from ....utils.output_utils import finalize_election_output

# Use BASE_DIR and INPUT_DIR for robust path handling
BASE_DIR = Path(__file__).parents[3]
INPUT_DIR = BASE_DIR / "input"

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

def parse(page, html_context=None):
    html_context = html_context or {}
    config = html_context.get("config", {})
    logger.info("[PA Handler] Contest routing active — using shared contest context with state-level extraction.")

    # STEP 1: Navigation (if needed)
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

    # Look for a CSV file in the input directory
    csv_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".csv")]
    if not csv_files:
        logger.error(f"[ERROR] No CSV files found in input directory: {INPUT_DIR}")
        return [], [], "Pennsylvania (CSV not found)", {}

    # If multiple CSVs, prompt user to select
    if len(csv_files) > 1:
        rprint("[yellow]Multiple CSV files found in input. Please select one:[/yellow]")
        for i, fname in enumerate(csv_files):
            rprint(f"  [bold cyan][{i}][/bold cyan] {fname}")
        try:
            idx = int(input("Select CSV file index: ").strip())
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
            numeric_columns = [h for h in headers if all(row.get(h, '').replace(',', '').replace('.', '').isdigit() for row in data)]
            totals = {h: 0 for h in numeric_columns}
            for row in data:
                for h in numeric_columns:
                    try:
                        totals[h] += int(row.get(h, "0").replace(",", "").strip())
                    except Exception:
                        continue

            totals_row = {key: "" for key in headers}
            for h in numeric_columns:
                totals_row[h] = str(totals[h])
            if headers:
                totals_row[headers[0]] = "Grand Total"
            data.append(totals_row)

        contest_title = header_text or "Pennsylvania County Results"
        metadata = {
            "state": "PA",
            "county": html_context.get("county", ""),
            "handler": "pennsylvania",
            "race": contest_title
        }

        result = finalize_election_output(headers, data, contest_title, metadata)
        return headers, data, contest_title, result.get("metadata", metadata)

    except Exception as e:
        logger.error(f"[ERROR] Failed to read or write CSV: {e}")
        return [], [], "Pennsylvania CSV Parse Error", {}