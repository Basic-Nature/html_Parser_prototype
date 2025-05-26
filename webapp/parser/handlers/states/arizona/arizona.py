# handlers/arizona.py
# ==============================================================
# Handler for Arizona election result sites with expandable cards
# and toggles between 'Vote Type' and 'By County' views.
# ==============================================================
import os
import json
from tqdm import tqdm
from ....utils.shared_logger import log_info, log_debug, log_warning, log_error
from rich import print as rprint
from ....Context_Integration.context_organizer import organize_context
from ....utils.output_utils import finalize_election_output

# Load config from context library if available
CONTEXT_LIBRARY_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "Context_Integration", "context_library.json"
)
if os.path.exists(CONTEXT_LIBRARY_PATH):
    with open(CONTEXT_LIBRARY_PATH, "r", encoding="utf-8") as f:
        CONTEXT_LIBRARY = json.load(f)
    STATE_CONFIGS = CONTEXT_LIBRARY.get("state_configs", {})
    config = STATE_CONFIGS.get("arizona", {})
else:
    config = {}

# Fallback defaults if not set in context library
config.setdefault("view_more_selector", "button:has-text('View More')")
config.setdefault("vote_type_toggle_selector", "button:has-text('Vote Type')")
config.setdefault("county_toggle_selector", "button:has-text('By County')")

def parse(page, html_context=None):
    if html_context is None:
        html_context = {}

    print("[INFO] Arizona handler activated. Expanding race-level cards...")

    # Step 1: Click all 'View More' buttons if present
    view_more_selector = config.get("view_more_selector")
    if view_more_selector:
        buttons = page.locator(view_more_selector)
        for i in range(buttons.count()):
            try:
                btn = buttons.nth(i)
                btn.scroll_into_view_if_needed()
                btn.click()
                page.wait_for_timeout(500)
                log_info(f"[INFO] Expanded card {i+1}/{buttons.count()}")
            except Exception as e:
                log_warning(f"[WARN] Could not expand card {i+1}: {e}")

    # Step 2: Toggle to 'Vote Type' view
    vote_type_selector = config.get("vote_type_toggle_selector")
    if vote_type_selector:
        try:
            vote_toggle = page.locator(vote_type_selector)
            if vote_toggle.count() > 0:
                vote_toggle.first.scroll_into_view_if_needed()
                vote_toggle.first.click()
                log_info("[INFO] Toggled to 'Vote Type' view")
                page.wait_for_timeout(1500)
        except Exception as e:
            log_warning(f"[WARN] Vote Type toggle failed: {e}")

    # Step 3: Toggle to 'By County' view if present
    county_selector = config.get("county_toggle_selector")
    if county_selector:
        try:
            county_toggle = page.locator(county_selector)
            if county_toggle.count() > 0:
                county_toggle.first.scroll_into_view_if_needed()
                county_toggle.first.click()
                log_info("[INFO] Toggled to 'By County' view")
                page.wait_for_timeout(1500)
        except Exception as e:
            log_warning(f"[WARN] County toggle failed: {e}")

    # Step 4: Attempt to find all visible tables and modal/overlay containers
    log_info("[INFO] Attempting to extract precinct-level data...")
    log_info("[INFO] Also collecting county-wide totals from overlay where available.")
    all_candidates = set()
    precinct_data = []
        # Capture everything including modal content (vote types)
    all_elements = page.query_selector_all('h1, h2, h3, strong, b, span, table, div, section, article, a, p')

    current_precinct = None
    for el in all_elements:
        tag = el.evaluate("e => e.tagName").strip().upper()
        text_preview = el.inner_text().strip().replace("\n", " ")[:100]
        log_debug(f"[DEBUG] Element tag: {tag} | Text: {text_preview}")
        try:
            if tag in ["H3", "STRONG", "B", "SPAN"]:
                label = el.inner_text().strip()
                if any(w in label for w in ["County", "Precinct", "District"]):
                    current_precinct = label
                    print(f"[DEBUG] Found precinct header: {label}")
            elif tag == "TABLE" and current_precinct:
                headers = el.query_selector_all('thead tr th')
                rows = el.query_selector_all('tbody tr')
                if not headers or not rows:
                    continue
                column_names = [h.inner_text().strip() for h in headers]
                method_names = column_names[1:-1]
                row_blocks = []
                for row in tqdm(rows, desc=f"Parsing {current_precinct}"):
                    cells = [cell.inner_text().strip() for cell in row.query_selector_all('td')]
                    if len(cells) != len(column_names):
                        continue
                    full_name = cells[0].replace("\n", " ").strip()
                    name_parts = full_name.split()
                    if len(name_parts) >= 3:
                        candidate_name = " ".join(name_parts[1:-1])
                        party_label = name_parts[-1]
                        canonical = f"{candidate_name} ({party_label})"
                    else:
                        canonical = full_name.strip()
                    method_votes = cells[1:-1]
                    total_votes = cells[-1]
                    all_candidates.add(canonical)
                    row_blocks.append((canonical, method_votes, total_votes))
                full_row = {"Precinct Name": current_precinct}
                for candidate, method_votes, total in row_blocks:
                    for method, vote in zip(method_names, method_votes):
                        full_row[f"{candidate} - {method}"] = vote
                    full_row[f"{candidate} - Total Votes"] = total
                for candidate in all_candidates:
                    for method in method_names:
                        col = f"{candidate} - {method}"
                        if col not in full_row:
                            full_row[col] = "-"
                    total_col = f"{candidate} - Total Votes"
                    if total_col not in full_row:
                        full_row[total_col] = "-"
                precinct_data.append(full_row)
                current_precinct = None
        except Exception as e:
            log_error(f"[ERROR] Table parse error: {e}")

    # Extract county-level totals if available
    county_totals = {}
    for el in all_elements:
        try:
            text = el.inner_text().strip()
            if "Precincts Reporting:" in text:
                county_totals["Precincts Reporting"] = text.split(":")[-1].strip()
            elif "Registered Voters:" in text:
                county_totals["Registered Voters"] = text.split(":")[-1].strip()
            elif "Ballots Cast:" in text:
                county_totals["Ballots Cast"] = text.split(":")[-1].strip()
            elif "Voter Turnout:" in text:
                county_totals["Voter Turnout"] = text.split(":")[-1].strip()
        except:
            continue

    if county_totals:
        print("[SUMMARY] County-Level Totals Found:")
        for k, v in county_totals.items():
            print(f"  {k}: {v}")

    contest_title = "Arizona Statewide Results"
    headers_out = sorted([col for col in precinct_data[0] if col != "Precinct Name"] if precinct_data else [])
    if not precinct_data:
        print("[FALLBACK] No tables were parsed. Either no results are published yet or the structure has changed.")
        print("[FALLBACK] Please verify that the site has posted election data.")

        # Insert county-level totals as a dummy precinct row if any were found
    contest_title = "Arizona Statewide Results"
    headers_out = sorted([col for col in precinct_data[0] if col != "Precinct Name"] if precinct_data else [])
    metadata = {
        "state": "AZ",
        "race": contest_title or "Unknown",
        "handler": "arizona",
        "source": getattr(page, "url", "Unknown")
    }
    if county_totals:
        metadata.update(county_totals)

    # Enrich metadata and finalize output
    organized = organize_context(metadata)
    metadata = organized.get("metadata", metadata)
    result = finalize_election_output(headers_out, precinct_data, contest_title, metadata)
    contest_title = result.get("contest_title", contest_title)
    metadata = result.get("metadata", metadata)
    return headers_out, precinct_data, contest_title, metadata