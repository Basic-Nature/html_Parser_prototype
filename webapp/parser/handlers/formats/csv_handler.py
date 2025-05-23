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
from ...utils.output_utils import get_output_path, format_timestamp, finalize_election_output
from ...utils.shared_logger import logging, rprint, logger

load_dotenv()


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
        try:
            csv_files = [f for f in os.listdir("input") if f.lower().endswith(".csv")]
            if csv_files:
                csv_files.sort(key=lambda x: os.path.getmtime(os.path.join("input", x)), reverse=True)
                csv_path = os.path.join("input", csv_files[0])
                logging.info(f"[FALLBACK] Using most recent CSV file: {csv_path}")
            else:
                logging.error("[ERROR] No CSV files found in the input directory.")
                return None, None, None, {"error": "No CSV in input folder"}
        except Exception as e:
            logging.error(f"[ERROR] Failed to list CSV files: {e}")
            return None, None, None, {"error": str(e)}

    try:
        rprint("[yellow]Available CSV file detected:[/yellow]")
        rprint(f"  [bold cyan]{os.path.basename(csv_path)}[/bold cyan]")
        user_input = input("[PROMPT] Parse this file? (y/n, or 'h' to fallback to HTML): ").strip().lower()
        if user_input == 'h':
            logging.info("[INFO] User opted to fallback to HTML scanning.")
            return None, None, None, {"fallback_to_html": True}
        elif user_input != 'y':
            logging.info("[INFO] User declined CSV parse. Falling back to HTML-based scraping.")
            return None, None, None, {"skip_csv": True}
    except Exception as e:
        logging.warning(f"[WARN] Skipping user input prompt due to error: {e}")
        return None, None, None, {"error": str(e)}

    data = []
    contest_column = None
    headers = []
    try:
        with open(csv_path, newline='', encoding='utf-8') as f:
            # Step: Handle embedded headers or skip metadata lines
            default_keywords = os.getenv("CSV_HEADER_KEYWORDS", "precinct,votes,candidate").split(",")
            handler_keywords = default_keywords
            handler = resolve_state_handler(csv_path)
            if handler and hasattr(handler, "header_keywords"):
                handler_keywords = getattr(handler, "header_keywords")

            preview_lines = [next(f) for _ in range(10)]  # Read first 10 lines
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
                    return None, None, None, {"error": "Header match declined"}
                    
                f.seek(f.tell())  # Start DictReader here
            reader = csv.DictReader(f)
            headers = [h.strip() for h in reader.fieldnames or []]

            # Step: Determine index column label dynamically
            index_label = "Precinct"
            possible_labels = ["ward", "district", "town"]
            for label in possible_labels:
                if any(label.lower() in h.lower() for h in headers):
                    index_label = label.capitalize()
                    logging.info(f"[Label Detection] Using '{index_label}' as the row label based on column names.")
                    break

            for row in reader:
                row = {k.strip(): v for k, v in row.items()}
                if any(val.strip() for val in row.values() if val):  # Skip empty/garbage rows
                    data.append(row)

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
                "race": os.path.basename(csv_path).replace(".csv", "").replace("_", " ").title(),
                "state": html_context.get("state", "Unknown"),
                "county": html_context.get("county", "Unknown"),
                "handler": "csv_handler"
            }
            safe_title = re.sub(r"[\\/*?\"<>|]", "_", os.path.basename(csv_path).replace(".csv", ""))
            output_path = get_output_path(metadata["state"], metadata["county"], "parsed")
            timestamp = format_timestamp() if os.getenv("INCLUDE_TIMESTAMP_IN_FILENAME", "true").lower() == "true" else ""
            filename = f"{safe_title}_parsed_{timestamp}.csv" if timestamp else f"{safe_title}_parsed.csv"
            filepath = os.path.join(output_path, filename)
            
            result = finalize_election_output(headers, data, contest_title, metadata)
            contest_title = result.get("contest_title", os.path.basename(csv_path))
            metadata = result.get("metadata", metadata)
            return headers, data, contest_title, metadata
            

    except Exception as e:
        logging.error(f"[ERROR] Failed to parse CSV: {e}")
        return None, None, None, {"error": str(e)}
