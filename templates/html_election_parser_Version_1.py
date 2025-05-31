from playwright.sync_api import sync_playwright
import csv
import os
from datetime import datetime
from collections import defaultdict

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Launches the Playwright browser in a background window to avoid stealing focus
# and allows script automation to proceed without interruption.
def launch_browser():
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(headless=False, args=['--window-position=1000,1000', '--window-size=1280,720', '--disable-backgrounding-occluded-windows'])
    context = browser.new_context()
    page = context.new_page()
    return playwright, browser, page

# Scans visible page headers to list contest names likely to represent election races
# Filters headings using known race keywords.
def list_static_contests(page):
    headers = page.query_selector_all('h1, h2, h3, .header, .title, .contest-header')
    contests = []
    for h in headers:
        txt = h.inner_text().strip()
        if any(k in txt for k in ["Electors", "Senator", "Assembly", "Justice", "Proposition"]):
            contests.append(txt)
    if contests:
        print("[INFO] Contests available in static page content:")
        for idx, c in enumerate(contests):
            print(f"  [{idx}] {c}")
    return contests

# Attempts to match the user's contest input to available contest titles on the page.
# Supports exact, partial, or indexed selection (e.g. '[2] United States Senator').
def extract_visible_contest_title(page, target_static_contest=None):
    page.wait_for_timeout(1000)
    headings = page.query_selector_all('h1, h2, h3, .header, .title, .contest-header')
    all_titles = []
    seen = set()
    for h in headings:
        text = h.inner_text().strip()
        if text and any(k in text for k in ["Electors", "Senator", "Assembly", "Justice", "Proposition"]):
            if text not in seen:
                all_titles.append(text)
                seen.add(text)
    if target_static_contest:
        cleaned_input = target_static_contest.strip()
        if cleaned_input.startswith("[") and "]" in cleaned_input:
            idx_str = cleaned_input[1:cleaned_input.index("]")]
            if idx_str.isdigit():
                idx = int(idx_str)
                if 0 <= idx < len(all_titles):
                    return all_titles[idx]
        exact_matches = [t for t in all_titles if target_static_contest.lower() == t.lower()]
        partial_matches = [t for t in all_titles if target_static_contest.lower() in t.lower()]
        if exact_matches:
            return exact_matches[0]
        elif partial_matches:
            print("[INFO] Multiple contests matched your input. Please refine if needed.")
            for idx, name in enumerate(partial_matches):
                print(f"  [{idx}] {name}")
            return partial_matches[0]
    print("[WARNING] No matching contest title found. Defaulting to 'Unknown_Contest'")
    return "Unknown_Contest"

# Handles the interaction flow:
# 1. Clicking into the race's "View results" button.
# 2. Enabling 'Vote Method' display.
# 3. Autoscrolling until all precincts are visible.
def interact_with_enhancedvoting_ui(page, contest_title):
    initial_delay = 20000
    print("[INFO] Waiting for page content to stabilize before scrolling...")
    page.wait_for_timeout(initial_delay)
    try:
        page.wait_for_selector("text=View results by election district", timeout=10000)
        view_links = page.locator("a:has-text('View results by election district')")
        clicked = False
        for i in range(view_links.count()):
            candidate = view_links.nth(i)
            section_heading = candidate.locator("xpath=ancestor::p-panel[1]//h1")
            if section_heading.count() > 0 and contest_title.lower() in section_heading.inner_text().strip().lower():
                candidate.scroll_into_view_if_needed()
                candidate.click()
                clicked = True
                print(f"[INFO] Clicked contest-specific 'View results' for: {contest_title}")
                break
        if not clicked:
            print(f"[ERROR] Could not find matching 'View results' link for contest: {contest_title}. Please verify your selection.")
            return
        page.wait_for_timeout(1500)
        print("[INFO] Searching for global 'Vote Method' button")
        page.screenshot(path="vote_method_area_debug.png", full_page=True)
        vote_method_button = None
        buttons = page.locator("button")
        for i in range(buttons.count()):
            btn = buttons.nth(i)
            btn_text = btn.inner_text().strip()
            if "Vote Method" in btn_text:
                vote_method_button = btn
                break
        if vote_method_button:
            vote_method_button.scroll_into_view_if_needed()
            vote_method_button.click()
            print("[INFO] Successfully toggled global 'Vote Method' button")
        else:
            print("[WARNING] Could not locate 'Vote Method' button. See 'vote_method_area_debug.png' for clues.")
        print("[INFO] Starting auto-scroll. This may take several minutes...")
        page.evaluate("window.scrollTo(0, 0)")
        page.wait_for_timeout(1500)
        step = 1600
        total_wait = 0
        max_scroll_time = 300000
        last_height = 0
        same_count = 0
        while total_wait < max_scroll_time and same_count < 5:
            current_height = page.evaluate("() => document.body.scrollHeight")
            if current_height == last_height:
                same_count += 1
            else:
                same_count = 0
            last_height = current_height
            page.evaluate(f"window.scrollBy(0, {step})")
            total_wait += 1500
            print(f"[SCROLL] Elapsed: {total_wait // 1000}s | Height: {current_height} | Stable: {same_count} of 5")
            page.wait_for_timeout(2000)
        print("[INFO] Auto-scroll finished based on time or content stability.")
    except Exception as e:
        print(f"[WARNING] Could not complete enhanced voting UI steps: {e}")

# === Extract Table Content ===
def extract_table_data(page):
    # Attempt to extract 'Fully Reported' status
    report_status = ""
    try:
        report_element = page.locator("span.fw-bold.ng-star-inserted", has_text="Fully Reported")
        if report_element.count() > 0:
            report_status = "Fully Reported"
    except:
        report_status = ""

    """Parses precinct tables and returns structured candidate-method vote data."""
    all_candidates = set()
    method_names = []
    precinct_data = []
    all_elements = page.query_selector_all('h3, strong, b, span, table')
    current_precinct = None

    for el in all_elements:
        tag = el.evaluate("e => e.tagName").strip().upper()
        if tag in ["H3", "STRONG", "B", "SPAN"]:
            label = el.inner_text().strip()
            # Dynamically infer precincts by common structural patterns (e.g., 'Ward' or numeric suffix)
            if any(w in label for w in ["Ward", "District", "Precinct", "Point", "Town"]) or any(char.isdigit() for char in label):
                current_precinct = label
        elif tag == "TABLE" and current_precinct:
            headers = el.query_selector_all('thead tr th')
            rows = el.query_selector_all('tbody tr')
            if not headers or not rows:
                continue
            column_names = [h.inner_text().strip() for h in headers]
            print(f"[DEBUG] Table under precinct {current_precinct} with headers: {column_names}")
            method_names = column_names[1:-1]
            row_blocks = []
            for row in rows:
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
            full_row = {"% Precincts Reporting": report_status, "Precinct Name": current_precinct}
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
    headers_out = sorted([col for col in precinct_data[0] if col not in ("Precinct Name", "% Precincts Reporting")] if precinct_data else [])
    return headers_out, precinct_data

# === Validate Parsed Data ===
def validate_data(rows, expected_columns):
    for row in rows:
        for col in expected_columns:
            if col not in row:
                print(f"[WARNING] Missing expected column '{col}' in row: {row}")
                return False
    return True

# === Write to Output CSV ===
def write_to_csv(contest_name, headers, data):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = contest_name.replace(' ', '_').replace('\n', '').replace(':', '').replace('/', '_')
    filename = os.path.join(OUTPUT_DIR, f"{safe_name}_{timestamp}.csv")
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["% Precincts Reporting", "Precinct Name"] + headers)
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    print(f"[INFO] Saved structured contest results: {filename}")

# === Main Flow ===
def main(url):
    playwright, browser, page = launch_browser()
    page.goto(url)
    page.wait_for_timeout(2000)

    static_races = list_static_contests(page)
    print("[INPUT] Please type part of the contest name you'd like to extract:")
    target_static_contest = input("> ").strip()

    contest_title = extract_visible_contest_title(page, target_static_contest)
    print(f"[INFO] Extracting contest: {contest_title}")

    interact_with_enhancedvoting_ui(page, contest_title)

    headers, structured_data = extract_table_data(page)
    if structured_data:
        if validate_data(structured_data, headers):
            write_to_csv(contest_title, headers, structured_data)
        else:
            print("[WARNING] Data validation failed.")
    else:
        print("[WARNING] No structured table data extracted.")
    browser.close()
    playwright.stop()
    
# === Entry Point ===
if __name__ == "__main__":
    # main(f"file://{os.path.abspath('Results by Precinct Full.html')}")
    main("https://app.enhancedvoting.com/results/public/rockland-county-ny/elections/GE2024Results")