# handlers/formats/csv_handler.py
# ==============================================================
# Handler for downloaded CSV files — NOT responsible for formatting output.
# Should only extract raw rows and return to the state/county handler for Smart Elections formatting.
# ==============================================================

import csv
import os
import re
from dotenv import load_dotenv
from ...state_router import get_handler_from_context
from ...utils.output_utils import finalize_election_output
from ...utils.shared_logger import logging, rprint, logger
from ...utils.table_builder import harmonize_rows, calculate_grand_totals, clean_candidate_name
from ...html_election_parser import organize_context_with_cache
load_dotenv()

def detect_csv_files(input_folder="input"):
    """Return a list of CSV files in the input folder, sorted by modified time (newest first)."""
    try:
        csv_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".csv")]
        csv_files.sort(key=lambda x: os.path.getmtime(os.path.join(input_folder, x)), reverse=True)
        return [os.path.join(input_folder, f) for f in csv_files]
    except Exception as e:
        logger.error(f"[ERROR] Failed to list CSV files: {e}")
        return []

def prompt_file_selection(csv_files):
    """Prompt user to select a CSV file from the list."""
    rprint("\n[yellow]Available CSV files in 'input' folder:[/yellow]")
    for i, f in enumerate(csv_files):
        rprint(f"  [bold cyan][{i}][/bold cyan] {os.path.basename(f)}")
    idx = input("\n[PROMPT] Enter file index or press Enter to cancel: ").strip()
    if not idx.isdigit():
        rprint("[yellow]No file selected. Skipping CSV parsing.[/yellow]")
        return None
    try:
        return csv_files[int(idx)]
    except (IndexError, ValueError):
        rprint("[red]Invalid index. Skipping CSV parsing.[/red]")
        return None

def detect_headers_and_skip_metadata(f, handler_keywords):
    """Skip metadata lines and find the header row."""
    preview_lines = [next(f) for _ in range(10)]
    f.seek(0)
    detected = next((line for line in preview_lines if any(k in line.lower() for k in handler_keywords)), None)
    if detected:
        while True:
            line = f.readline()
            if any(k in line.lower() for k in handler_keywords):
                break
        f.seek(f.tell())
    else:
        rprint("[yellow]No recognizable header found in preview. Proceed anyway? (y/n):[/yellow]")
        confirm = input().strip().lower()
        if confirm != 'y':
            logging.warning("[WARN] No header match found and user declined to proceed.")
            return False
        f.seek(f.tell())
    return True

def parse(page, html_context):
    """
    Parses the most recent CSV file in the input folder if available.
    Provides a prompt to continue or fallback. Does not use hardcoded filenames.
    Returns: headers, data, contest_title, metadata
    """
    # Respect early skip signal from calling context
    if html_context.get("skip_format") or html_context.get("manual_skip"):
        logger.info("[SKIP] CSV parsing intentionally skipped via context flag.")
        return None, None, None, {"skipped": True}

    csv_path = html_context.get("csv_source")
    if not csv_path:
        csv_files = detect_csv_files()
        if not csv_files:
            logger.error("[ERROR] No CSV files found in the input directory.")
            return None, None, None, {"error": "No CSV in input folder"}
        csv_path = prompt_file_selection(csv_files)
        if not csv_path:
            return None, None, None, {"skipped": True}

    try:
        rprint("[yellow]Available CSV file detected:[/yellow]")
        rprint(f"  [bold cyan]{os.path.basename(csv_path)}[/bold cyan]")
        user_input = input("[PROMPT] Parse this file? (y/n, or 'h' to fallback to HTML): ").strip().lower()
        if user_input == 'h':
            logging.info("[INFO] User opted to fallback to HTML scanning.")
            return None, None, None, {"fallback_to_html": True}
        elif user_input != 'y':
            logging.info("[INFO] User declined CSV parse. Skipping.")
            return None, None, None, {"skip_csv": True}
    except Exception as e:
        logging.warning(f"[WARN] Skipping user input prompt due to error: {e}")
        return None, None, None, {"error": str(e)}

    data = []
    headers = []
    contest_column = None
    try:
        with open(csv_path, newline='', encoding='utf-8') as f:
            # Step: Handle embedded headers or skip metadata lines
            default_keywords = os.getenv("CSV_HEADER_KEYWORDS", "precinct,votes,candidate").split(",")
            handler_keywords = default_keywords
            handler = get_handler_from_context(csv_path)
            if handler and hasattr(handler, "header_keywords"):
                handler_keywords = getattr(handler, "header_keywords")

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
            contest_title = None
            if contest_column:
                contests = sorted({row[contest_column].strip() for row in data if row.get(contest_column)})
                if len(contests) > 1:
                    rprint("\n[yellow]Multiple contests detected:[/yellow]")
                    for i, name in enumerate(contests, 1):
                        rprint(f" [bold cyan]{i:2d}[/bold cyan]. {name}")
                    rprint("\nEnter the contest name (exactly as shown), or type its number:")
                    user_input = input("> ").strip()
                    if user_input.isdigit():
                        idx = int(user_input)
                        try:
                            contest_title = contests[idx - 1]
                        except IndexError:
                            rprint("[red]Invalid contest number.[/red]")
                            return None, None, None, {"error": "Invalid contest number"}
                    else:
                        if user_input not in contests:
                            rprint(f"[red][ERROR] Contest name '{user_input}' not found.[/red]")
                            return None, None, None, {"error": "Contest name not found"}
                        contest_title = user_input
                    # Filter data to only selected contest
                    data = [row for row in data if row.get(contest_column, "").strip() == contest_title]
                elif contests:
                    contest_title = contests[0]
            else:
                contest_title = os.path.basename(csv_path).replace(".csv", "")

            # Step: Normalize candidate/precinct columns and harmonize
            candidate_cols = [col for col in headers if "candidate" in col.lower()]
            precinct_cols = [col for col in headers if "precinct" in col.lower() or "ward" in col.lower() or "district" in col.lower() or "county" in col.lower()]
            method_cols = [col for col in headers if any(m in col.lower() for m in ["election day", "early", "absentee", "mail", "provisional", "total"])]

            # Build wide-format rows: one row per reporting unit, columns for each candidate-method
            wide_data = []
            reporting_unit_col = precinct_cols[0] if precinct_cols else headers[0]
            for row in data:
                wide_row = {reporting_unit_col: row.get(reporting_unit_col, "")}
                for cand_col in candidate_cols:
                    candidate = clean_candidate_name(row.get(cand_col, ""))
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
            wide_data = harmonize_rows(headers, wide_data)
            wide_data.append(calculate_grand_totals(wide_data))

            # Step: Detect state and county from filename if not already present
            if "state" not in html_context or html_context["state"] == "Unknown":
                resolved = get_handler_from_context(csv_path)
                if resolved:
                    html_context["state"] = resolved.__name__.split(".")[-1].upper()

            if "county" not in html_context or html_context["county"] == "Unknown":
                fname = os.path.basename(csv_path).lower()
                for part in fname.replace(".csv", "").split("_"):
                    if "county" in part:
                        html_context["county"] = part.replace("county", "").strip().title() + " County"
                        break

            metadata = {
                "race": contest_title,
                "state": html_context.get("state", "Unknown"),
                "county": html_context.get("county", "Unknown"),
                "handler": "csv_handler"
            }

            # Enrich metadata and context using context_organizer
            from ...Context_Integration.context_organizer import organize_context
            organized = organize_context_with_cache(metadata)
            metadata = organized.get("metadata", metadata)

            # Output via finalize_election_output
            result = finalize_election_output(headers, wide_data, contest_title, metadata)
            contest_title = result.get("contest_title", contest_title)
            metadata = result.get("metadata", metadata)
            return headers, wide_data, contest_title, metadata

    except Exception as e:
        logging.error(f"[ERROR] Failed to parse CSV: {e}")
        return None, None, None, {"error": str(e)}