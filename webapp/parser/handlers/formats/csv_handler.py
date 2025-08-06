# ==============================================================
# 🗳️ Smart Elections: Universal CSV Election Results Parser
# ==============================================================
from __future__ import annotations
import csv
import os
import orjson
from ...config import BASE_DIR
from ...utils.logger_singleton import logger
from ...Context_Integration.Context_Library.constants import (
    LOCATION_KEYWORDS, CANDIDATE_KEYWORDS, BALLOT_TYPES, PARTY_KEYWORDS, TOTAL_KEYWORDS,
    MISC_FOOTER_KEYWORDS, CONTEST_KEYWORDS
)
from ...utils.table_core import harmonize_headers_and_data

def get_input_folder():
    # Parent of webapp, then 'input'
    return os.path.join(os.path.dirname(BASE_DIR), "input")

def get_output_folder():
    # Parent of webapp, then 'output'
    return os.path.join(os.path.dirname(BASE_DIR), "output")

def list_csv_files(input_folder):
    try:
        csv_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".csv")]
        csv_files.sort(key=lambda x: os.path.getmtime(os.path.join(input_folder, x)), reverse=True)
        return [os.path.join(input_folder, f) for f in csv_files]
    except Exception as e:
        logger.error(f"[ERROR] Failed to list CSV files: {e}")
        return []

def prompt_for_csv_file(input_folder):
    csv_files = list_csv_files(input_folder)
    if not csv_files:
        logger.error("[red][ERROR] No CSV files found in the input directory.[/red]")
        return None
    logger.warning("\n[yellow]Available CSV files in 'input' folder:[/yellow]")
    for i, f in enumerate(csv_files):
        logger.info(f"  [bold cyan][{i}][/bold cyan] {os.path.basename(f)}")
    idx = input("\n[PROMPT] Enter file index or press Enter to cancel: ").strip()
    if not idx:
        logger.warning("[yellow]No file selected. Skipping CSV parsing.[/yellow]")
        return None
    if idx.isdigit():
        try:
            return csv_files[int(idx)]
        except (IndexError, ValueError):
            logger.error("[red]Invalid index. Skipping CSV parsing.[/red]")
            return None
    logger.error("[red]Invalid selection. Skipping CSV parsing.[/red]")
    return None

def detect_headers_and_skip_metadata(f, handler_keywords):
    """Skip metadata lines and find the header row."""
    preview_lines = []
    try:
        for _ in range(10):
            preview_lines.append(next(f))
    except StopIteration:
        pass
    f.seek(0)
    detected = next((line for line in preview_lines if any(k in line.lower() for k in handler_keywords)), None)
    if detected:
        while True:
            line = f.readline()
            if not line:
                break
            if any(k in line.lower() for k in handler_keywords):
                break
        f.seek(f.tell())
    else:
        logger.warning("[yellow]No recognizable header found in preview. Proceed anyway? (y/n):[/yellow]")
        confirm = input().strip().lower()
        if confirm != 'y':
            logger.warning("[WARN] No header match found and user declined to proceed.")
            return False
        f.seek(f.tell())
    return True

def parse_csv_election_results(csv_path, output_dir=None):
    """
    Reads a CSV file, prompts for contest selection if needed, normalizes columns using librarian context,
    and writes harmonized output CSV and metadata to the output folder.
    """
    data = []
    headers = []
    contest_column = None

    # === Load CSV ===
    with open(csv_path, newline='', encoding='utf-8') as f:
        # Step: Handle embedded headers or skip metadata lines
        handler_keywords = list(LOCATION_KEYWORDS | CANDIDATE_KEYWORDS | BALLOT_TYPES | PARTY_KEYWORDS | TOTAL_KEYWORDS | MISC_FOOTER_KEYWORDS | CONTEST_KEYWORDS)
        if not detect_headers_and_skip_metadata(f, handler_keywords):
            return None, None, None, {"error": "Header match declined"}

        reader = csv.DictReader(f)
        headers = [h.strip() for h in reader.fieldnames or []]

        # Step: Detect contest/race column if present
        possible_contest_cols = [col for col in headers if any(k in col.lower() for k in ["contest", "race", "office"])]
        if possible_contest_cols:
            contest_column = possible_contest_cols[0]

        # Step: Read and clean data
        for row in reader:
            row = {k.strip(): v for k, v in row.items()}
            if any(val.strip() for val in row.values() if val):  # Skip empty/garbage rows
                data.append(row)

        # Step: If multiple contests, prompt user to select one
        contest = None
        if contest_column:
            contests = sorted({row[contest_column].strip() for row in data if row.get(contest_column)})
            if len(contests) > 1:
                logger.warning("\n[yellow]Multiple contests detected:[/yellow]")
                for i, name in enumerate(contests, 1):
                    logger.info(f" [bold cyan]{i:2d}[/bold cyan]. {name}")
                logger.info("\nEnter the contest name (exactly as shown), or type its number:")
                user_input = input("> ").strip()
                if user_input.isdigit():
                    idx = int(user_input)
                    try:
                        contest = contests[idx - 1]
                    except IndexError:
                        logger.error("[red]Invalid contest number.[/red]")
                        return None, None, None, {"error": "Invalid contest number"}
                else:
                    if user_input not in contests:
                        logger.error(f"[red][ERROR] Contest name '{user_input}' not found.[/red]")
                        return None, None, None, {"error": "Contest name not found"}
                    contest = user_input
                # Filter data to only selected contest
                data = [row for row in data if row.get(contest_column, "").strip() == contest]
            elif contests:
                contest = contests[0]
        else:
            contest = os.path.basename(csv_path).replace(".csv", "")

    # Step: Normalize candidate/precinct columns and harmonize using librarian context
    candidate_cols = [col for col in headers if any(k in col.lower() for k in CANDIDATE_KEYWORDS)]
    precinct_cols = [col for col in headers if any(k in col.lower() for k in LOCATION_KEYWORDS)]
    method_cols = [col for col in headers if any(m in col.lower() for m in BALLOT_TYPES | TOTAL_KEYWORDS | MISC_FOOTER_KEYWORDS)]

    # Build wide-format rows: one row per reporting unit, columns for each candidate-method
    wide_data = []
    reporting_unit_col = precinct_cols[0] if precinct_cols else headers[0]
    for row in data:
        wide_row = {reporting_unit_col: row.get(reporting_unit_col, "")}
        for cand_col in candidate_cols:
            candidate = row.get(cand_col, "")
            for method_col in method_cols:
                val = row.get(method_col, "")
                col_name = f"{candidate} - {method_col}"
                wide_row[col_name] = val
        if not candidate_cols:
            for method_col in method_cols:
                wide_row[method_col] = row.get(method_col, "")
        for col in headers:
            if col not in candidate_cols + method_cols + [reporting_unit_col]:
                wide_row[col] = row.get(col, "")
        wide_data.append(wide_row)

    # Build headers from all keys
    all_keys = set()
    for row in wide_data:
        all_keys.update(row.keys())
    headers = [reporting_unit_col] + sorted([k for k in all_keys if k != reporting_unit_col])

    # Harmonize and add grand total
    headers, wide_data = harmonize_headers_and_data(headers, wide_data)

    # === Setup Output Paths ===
    if output_dir is None:
        output_dir = get_output_folder()
    os.makedirs(output_dir, exist_ok=True)
    safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in contest).replace(" ", "_")
    output_csv = os.path.join(output_dir, f"{safe_title}_parsed.csv")
    output_meta = os.path.join(output_dir, f"{safe_title}_metadata.json")

    # === Write Output CSV ===
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in wide_data:
            writer.writerow(row)

    # === Write Metadata JSON ===
    metadata = {
        "race": contest,
        "input_file": os.path.basename(csv_path),
        "output_file": os.path.basename(output_csv),
        "headers": headers,
        "row_count": len(wide_data),
        "handler": "csv_handler"
    }
    with open(output_meta, "w") as jf:
        jf.write(orjson.dumps(metadata, option=orjson.OPT_INDENT_2).decode("utf-8"))

    logger.info(f"[bold green][OUTPUT][/bold green] Wrote [bold]{len(wide_data)}[/bold] rows to:\n  [cyan]{output_csv}[/cyan]")
    logger.info(f"[bold green][OUTPUT][/bold green] Metadata written to:\n  [cyan]{output_meta}[/cyan]")

    return headers, wide_data, contest, metadata

def parse(page=None, coordinator=None, html_context=None, manual_file=None, **kwargs):
    """
    Universal pipeline entry: Accepts a CSV file path (manual_file) from the format router,
    or prompts user to select a file from the input folder.
    Returns: headers, data, contest, metadata
    """
    html_context = html_context or {}
    if html_context.get("skip_format") or html_context.get("manual_skip"):
        logger.info("[SKIP] CSV parsing intentionally skipped via context flag.")
        return None, None, None, {"skipped": True}

    input_folder = get_input_folder()
    csv_path = None

    # 1. Use file handed over from format router if provided
    if manual_file and os.path.isfile(manual_file):
        csv_path = manual_file
    else:
        # 2. Otherwise, prompt user to select from input folder
        csv_path = prompt_for_csv_file(input_folder)
        if not csv_path:
            return None, None, None, {"skipped": True}

    try:
        logger.warning("[yellow]Available CSV file detected:[/yellow]")
        logger.warning(f"  [bold cyan]{os.path.basename(csv_path)}[/bold cyan]")
        user_input = input("[PROMPT] Parse this file? (y/n, or 'h' to fallback to HTML): ").strip().lower()
        if user_input == 'h':
            logger.info("[INFO] User opted to fallback to HTML scanning.")
            return None, None, None, {"fallback_to_html": True}
        elif user_input != 'y':
            logger.info("[INFO] User declined CSV parse. Skipping.")
            return None, None, None, {"skip_csv": True}
    except Exception as e:
        logger.warning(f"[WARN] Skipping user input prompt due to error: {e}")
        return None, None, None, {"error": str(e)}

    # --- Main CSV parsing logic ---
    return parse_csv_election_results(csv_path)

# If run as a script, allow standalone use
if __name__ == "__main__":
    input_folder = get_input_folder()
    csv_path = prompt_for_csv_file(input_folder)
    if csv_path:
        parse_csv_election_results(csv_path)
