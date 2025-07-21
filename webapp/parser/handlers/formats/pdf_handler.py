# ==============================================================
# 🗳️ Smart Elections: Universal PDF Election Results Parser
# ==============================================================

import os
import re
import csv
from concurrent.futures import ThreadPoolExecutor
from ...config import BASE_DIR
from ...utils.shared_logger import SharedLogger
from ...bots.librarian import (
    LOCATION_KEYWORDS, CANDIDATE_KEYWORDS, BALLOT_TYPES, PARTY_KEYWORDS, TOTAL_KEYWORDS,
    MISC_FOOTER_KEYWORDS, CONTEST_KEYWORDS
)
from ...utils.table_core import harmonize_headers_and_data
import orjson
try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError("You must install PyMuPDF to use the PDF handler: pip install pymupdf")

try:
    import pytesseract
    from PIL import Image
    import pdf2image
except ImportError:
    pytesseract = None
    pdf2image = None
logger = SharedLogger()
def get_input_folder():
    # Parent of webapp, then 'input'
    return os.path.join(os.path.dirname(BASE_DIR), "input")

def get_output_folder():
    # Parent of webapp, then 'output'
    return os.path.join(os.path.dirname(BASE_DIR), "output")

def list_pdf_files(input_folder):
    try:
        pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".pdf")]
        pdf_files.sort(key=lambda x: os.path.getmtime(os.path.join(input_folder, x)), reverse=True)
        return [os.path.join(input_folder, f) for f in pdf_files]
    except Exception as e:
        logger.error(f"[ERROR] Failed to list PDF files: {e}")
        return []

def prompt_for_pdf_file(input_folder):
    pdf_files = list_pdf_files(input_folder)
    if not pdf_files:
        logger.error("[red][ERROR] No PDF files found in the input directory.[/red]")
        return None
    logger.warning("\n[yellow]Available PDF files in 'input' folder:[/yellow]")
    for i, f in enumerate(pdf_files):
        logger.info(f"  [bold cyan][{i}][/bold cyan] {os.path.basename(f)}")
    idx = input("\n[PROMPT] Enter file index or press Enter to cancel: ").strip()
    if not idx:
        logger.warning("[yellow]No file selected. Skipping PDF parsing.[/yellow]")
        return None
    if idx.isdigit():
        try:
            return pdf_files[int(idx)]
        except (IndexError, ValueError):
            logger.error("[red]Invalid index. Skipping PDF parsing.[/red]")
            return None
    logger.error("[red]Invalid selection. Skipping PDF parsing.[/red]")
    return None

def ocr_multi_pass(images, passes=3, confidence_threshold=30):
    """Run OCR multiple times and aggregate results with confidence scoring."""
    ocr_runs = []
    pass_confidences = []

    def process_image_ocr(img):
        page_text = ""
        confidences = []
        if pytesseract:
            details = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT) if hasattr(pytesseract, "Output") else {}
            for j in range(len(details.get("text", []))):
                word = details["text"][j].strip()
                conf = details["conf"][j]
                if word:
                    try:
                        conf_val = float(conf)
                        confidences.append(conf_val)
                        if conf_val >= confidence_threshold:
                            page_text += word + " "
                    except ValueError:
                        continue
        return page_text, confidences

    for i in range(passes):
        logger.info(f"[INFO] OCR pass {i+1} of {passes}")
        ocr_text = ""
        confidences = []
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_image_ocr, images))
        for text, conf_list in results:
            ocr_text += text + "\n"
            confidences.extend(conf_list)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        pass_confidences.append(avg_conf)
        ocr_runs.append(ocr_text)

    # Merge and dedupe lines, cross-reference for best lines
    line_sets = [set(text.splitlines()) for text in ocr_runs]
    combined_lines = sorted(set.union(*line_sets))
    all_text = "\n".join(combined_lines)
    overall_avg = sum(pass_confidences) / len(pass_confidences) if pass_confidences else 0.0
    return all_text, overall_avg, ocr_runs

def infer_headers_and_methods(lines, table_hints):
    """Try to infer headers and method columns from lines."""
    header_candidates = [line for line in lines if sum(1 for hint in table_hints if hint in line.lower()) >= 2]
    headers = []
    if header_candidates:
        # Use the first candidate as header
        headers = re.split(r"\s{2,}|\t|,", header_candidates[0].strip())
        headers = [h.strip() for h in headers if h.strip()]
    return headers, header_candidates

def parse_pdf_election_results(pdf_path, output_dir=None):
    """
    Reads a PDF file, extracts tabular data (with OCR fallback), normalizes columns using librarian context,
    and writes harmonized output CSV and metadata to the output folder.
    """
    all_text = ""
    metadata = {}
    headers = []
    ocr_score = 0.0
    ocr_runs = []

    # === Extract text with PyMuPDF ===
    try:
        doc = fitz.open(pdf_path)
        for i in range(len(doc)):
            pdf_page = doc[i]
            all_text += pdf_page.get_text()
        doc.close()
    except Exception as e:
        logger.warning(f"[WARN] fitz text extraction failed: {e}")
        all_text = ""

    # === OCR fallback if needed ===
    if not all_text.strip() and pytesseract and pdf2image and os.getenv("ENABLE_OCR", "true").lower() == "true":
        logger.info("[INFO] Empty text result from PyMuPDF — attempting OCR fallback.")
        images = pdf2image.convert_from_path(pdf_path)
        all_text, ocr_score, ocr_runs = ocr_multi_pass(images, passes=3, confidence_threshold=30)
        metadata["ocr_confidence_avg"] = round(ocr_score, 2)
        metadata["ocr_passes"] = 3

    logger.debug("[DEBUG] PDF extracted text preview (first 500 chars):" + all_text[:500])

    # === Step: Basic check for tabular structure ===
    table_hints = list(LOCATION_KEYWORDS | CANDIDATE_KEYWORDS | BALLOT_TYPES | PARTY_KEYWORDS | TOTAL_KEYWORDS | MISC_FOOTER_KEYWORDS)
    lines = all_text.splitlines()
    headers, header_candidates = infer_headers_and_methods(lines, table_hints)

    # === Detect state and county from filename if not already present ===
    state = "Unknown"
    county = "Unknown"
    fname = os.path.basename(pdf_path).lower()
    for part in fname.replace(".pdf", "").split("_"):
        if "county" in part:
            county = part.replace("county", "").strip().title() + " County"
        if len(part) == 2 and part.isalpha():
            state = part.upper()
    metadata.update({
        "source_file": os.path.basename(pdf_path),
        "state": state,
        "county": county,
        "handler": "pdf_handler"
    })

    # === Attempt contest selection (if inferred columns contain contest-like fields) ===
    contest_column = None
    if headers:
        logger.warning("[yellow]Inferred Columns:[/yellow]")
        for i, col in enumerate(headers):
            logger.info(f"  [bold cyan]{i}[/bold cyan]: {col}")
        selection = input("[PROMPT] Select contest column index (or leave blank to skip): ").strip()
        if selection.isdigit():
            contest_column = headers[int(selection)]

    # === Attempt row splitting from lines if table detected ===
    data = []
    if headers:
        # Find the header line index
        header_line_idx = None
        for idx, line in enumerate(lines):
            if all(h.lower() in line.lower() for h in headers[:2]):  # crude match
                header_line_idx = idx
                break
        if header_line_idx is None and header_candidates:
            try:
                header_line_idx = lines.index(header_candidates[0])
            except ValueError:
                header_line_idx = 0
        if header_line_idx is None:
            header_line_idx = 0

        # Parse rows
        for line in lines[header_line_idx + 1:]:
            if not line.strip():
                continue
            # Try to split by multiple spaces, tabs, or commas
            row = re.split(r"\s{2,}|\t|,", line.strip())
            row = [cell.strip() for cell in row if cell.strip()]
            if len(row) == len(headers):
                row_dict = dict(zip(headers, row))
                data.append(row_dict)

        # Harmonize and format as wide CSV
        if data:
            candidate_cols = [col for col in headers if any(k in col.lower() for k in CANDIDATE_KEYWORDS)]
            precinct_cols = [col for col in headers if any(k in col.lower() for k in LOCATION_KEYWORDS)]
            method_cols = [col for col in headers if any(m in col.lower() for m in BALLOT_TYPES | TOTAL_KEYWORDS | MISC_FOOTER_KEYWORDS)]

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
            safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in os.path.basename(pdf_path).replace(".pdf", "")).replace(" ", "_")
            output_csv = os.path.join(output_dir, f"{safe_title}_parsed.csv")
            output_meta = os.path.join(output_dir, f"{safe_title}_metadata.json")

            # === Write Output CSV ===
            with open(output_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                for row in wide_data:
                    writer.writerow(row)

            # === Write Metadata JSON ===
            metadata.update({
                "output_file": os.path.basename(output_csv),
                "headers": headers,
                "row_count": len(wide_data)
            })
            with open(output_meta, "w") as jf:
                jf.write(orjson.dumps(metadata, option=orjson.OPT_INDENT_2).decode("utf-8"))

            logger.info(f"[bold green][OUTPUT][/bold green] Wrote [bold]{len(wide_data)}[/bold] rows to:\n  [cyan]{output_csv}[/cyan]")
            logger.info(f"[bold green][OUTPUT][/bold green] Metadata written to:\n  [cyan]{output_meta}[/cyan]")

            return headers, wide_data, safe_title, metadata

        else:
            unmatched_count = len(lines[header_line_idx + 1:])
            logger.warning(f"[WARN] No structured rows matched the inferred column count of {len(headers)}. Total lines scanned: {unmatched_count}")
            fallback_rows = [{"raw_line": line} for line in lines[header_line_idx + 1:]]
            # Write fallback output
            if output_dir is None:
                output_dir = get_output_folder()
            os.makedirs(output_dir, exist_ok=True)
            safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in os.path.basename(pdf_path).replace(".pdf", "")).replace(" ", "_")
            output_csv = os.path.join(output_dir, f"{safe_title}_parsed.csv")
            output_meta = os.path.join(output_dir, f"{safe_title}_metadata.json")
            with open(output_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["raw_line"])
                writer.writeheader()
                for row in fallback_rows:
                    writer.writerow(row)
            metadata.update({
                "output_file": os.path.basename(output_csv),
                "headers": ["raw_line"],
                "row_count": len(fallback_rows)
            })
            with open(output_meta, "w") as jf:
                jf.write(orjson.dumps(metadata, option=orjson.OPT_INDENT_2).decode("utf-8"))
            logger.warning(f"[bold yellow][OUTPUT][/bold yellow] Wrote fallback rows to:\n  [cyan]{output_csv}[/cyan]")
            return ["raw_line"], fallback_rows, safe_title, metadata

    # If no table, return plain text
    if output_dir is None:
        output_dir = get_output_folder()
    os.makedirs(output_dir, exist_ok=True)
    safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in os.path.basename(pdf_path).replace(".pdf", "")).replace(" ", "_")
    output_csv = os.path.join(output_dir, f"{safe_title}_parsed.csv")
    output_meta = os.path.join(output_dir, f"{safe_title}_metadata.json")
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text"])
        writer.writeheader()
        writer.writerow({"text": all_text})
    metadata.update({
        "output_file": os.path.basename(output_csv),
        "headers": ["text"],
        "row_count": 1
    })
    with open(output_meta, "w") as jf:
        jf.write(orjson.dumps(metadata, option=orjson.OPT_INDENT_2).decode("utf-8"))
    logger.warning(f"[bold yellow][OUTPUT][/bold yellow] Wrote plain text to:\n  [cyan]{output_csv}[/cyan]")
    return ["text"], [{"text": all_text}], safe_title, metadata

def parse(page=None, coordinator=None, html_context=None, non_interactive=False, manual_file=None, **kwargs):
    """
    Universal pipeline entry: Accepts a PDF file path (manual_file) from the format router,
    or prompts user to select a file from the input folder.
    Returns: headers, data, contest, metadata
    """
    html_context = html_context or {}
    if html_context.get("skip_format") or html_context.get("manual_skip"):
        logger.info("[SKIP] PDF parsing intentionally skipped via context flag.")
        return None, None, None, {"skipped": True}

    input_folder = get_input_folder()
    pdf_path = None

    # 1. Use file handed over from format router if provided
    if manual_file and os.path.isfile(manual_file):
        pdf_path = manual_file
    else:
        # 2. Otherwise, prompt user to select from input folder
        pdf_path = prompt_for_pdf_file(input_folder)
        if not pdf_path:
            return None, None, None, {"skipped": True}

    try:
        logger.warning("[yellow]Available PDF file detected:[/yellow]")
        logger.info(f"  [bold cyan]{os.path.basename(pdf_path)}[/bold cyan]")
        user_input = input("[PROMPT] Parse this file? (y/n): ").strip().lower()
        if user_input != 'y':
            logger.info("[INFO] User declined PDF parse. Skipping.")
            return None, None, None, {"skip_pdf": True}
    except Exception as e:
        logger.warning(f"[WARN] Skipping user input prompt due to error: {e}")
        return None, None, None, {"error": str(e)}

    # --- Main PDF parsing logic ---
    return parse_pdf_election_results(pdf_path)

# If run as a script, allow standalone use
if __name__ == "__main__":
    input_folder = get_input_folder()
    pdf_path = prompt_for_pdf_file(input_folder)
    if pdf_path:
        parse_pdf_election_results(pdf_path)
