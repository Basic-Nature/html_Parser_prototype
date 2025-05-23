# handlers/formats/csv_handler.py
# ==============================================================
# Handler for downloaded CSV files — NOT responsible for formatting output.
# Should only extract raw rows and return to the state/county handler for Smart Elections formatting.
# ==============================================================

import csv
import os
import re
from dotenv import load_dotenv
from ...state_router import resolve_state_handler
from ...utils.output_utils import finalize_election_output
from ...utils.shared_logger import logging, rprint, logger

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
            handler = resolve_state_handler(csv_path)
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

            # Step: Null detection logging
            null_counts = {col: sum(1 for row in data if not row.get(col)) for col in headers}
            if any(count > 0 for count in null_counts.values()):
                if os.getenv("LOG_WARNINGS", "true").lower() == "true":
                    logging.warning("[Missing Data] Some values are missing in the CSV:")
                    for col, count in null_counts.items():
                        if count > 0:
                            logging.warning(f" - Column '{col}': {count} missing value(s)")

            # Step: Grand totals row calculation (numeric columns only)
            def is_number(val):
                try:
                    float(val.replace(',', '').strip())
                    return True
                except:
                    return False

            numeric_cols = [col for col in headers if all(is_number(row[col]) for row in data if row.get(col))]
            if numeric_cols:
                grand_total = {}
                for col in numeric_cols:
                    total = 0.0
                    for row in data:
                        val = row.get(col)
                        if val and is_number(val):
                            try:
                                total += float(val.replace(',', '').strip())
                            except Exception as e:
                                logging.warning(f"[WARN] Could not convert value '{val}' in column '{col}' to float: {e}")
                    grand_total[col] = total
                grand_total[headers[0]] = "Grand Totals"
                data.append(grand_total)

            # Step: Detect state and county from filename if not already present
            if "state" not in html_context or html_context["state"] == "Unknown":
                resolved = resolve_state_handler(csv_path)
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

            # Output via finalize_election_output
            result = finalize_election_output(headers, data, contest_title, metadata)
            contest_title = result.get("contest_title", contest_title)
            metadata = result.get("metadata", metadata)
            return headers, data, contest_title, metadata

    except Exception as e:
        logging.error(f"[ERROR] Failed to parse CSV: {e}")
        return None, None, None, {"error": str(e)}