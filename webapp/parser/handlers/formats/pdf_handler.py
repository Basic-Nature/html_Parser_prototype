# handlers/formats/pdf_handler.py
# ==============================================================
# Parses election results from PDF files.
# Includes dynamic scanning, OCR fallback, and optional user prompt.
# Enhanced for multi-pass OCR, accuracy scoring, and harmonized output.
# ==============================================================

import os
import re
import csv
from concurrent.futures import ThreadPoolExecutor
from rich import print as rprint
from ...state_router import get_handler_from_context
from ...utils.output_utils import finalize_election_output
from ...utils.shared_logger import logger
from ...utils.table_builder import harmonize_rows, calculate_grand_totals, clean_candidate_name

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

def detect_pdf_files(input_folder="input"):
    """Return a list of PDF files in the input folder, sorted by modified time (newest first)."""
    try:
        pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".pdf")]
        pdf_files.sort(key=lambda x: os.path.getmtime(os.path.join(input_folder, x)), reverse=True)
        return [os.path.join(input_folder, f) for f in pdf_files]
    except Exception as e:
        logger.error(f"[ERROR] Failed to list PDF files: {e}")
        return []

def prompt_file_selection(pdf_files):
    """Prompt user to select a PDF file from the list."""
    rprint("\n[yellow]Available PDF files in 'input' folder:[/yellow]")
    for i, f in enumerate(pdf_files):
        rprint(f"  [bold cyan][{i}][/bold cyan] {os.path.basename(f)}")
    idx = input("\n[PROMPT] Enter file index or press Enter to cancel: ").strip()
    if not idx.isdigit():
        rprint("[yellow]No file selected. Skipping PDF parsing.[/yellow]")
        return None
    try:
        return pdf_files[int(idx)]
    except (IndexError, ValueError):
        rprint("[red]Invalid index. Skipping PDF parsing.[/red]")
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

def parse(page, html_context):
    # Respect early skip signal from calling context
    if html_context.get("skip_format") or html_context.get("manual_skip"):
        logger.info("[SKIP] PDF parsing intentionally skipped via context flag.")
        return None, None, None, {"skipped": True}

    pdf_path = html_context.get("pdf_source")
    if not pdf_path:
        pdf_files = detect_pdf_files()
        if not pdf_files:
            logger.error("[ERROR] No PDF files found in the input directory.")
            return None, None, None, {"error": "No PDF in input folder"}
        pdf_path = prompt_file_selection(pdf_files)
        if not pdf_path:
            return None, None, None, {"skipped": True}

    try:
        user_input = input(f"[PROMPT] PDF file detected at {os.path.basename(pdf_path)}. Parse this file? (y/n, or 'h' to fallback to HTML): ").strip().lower()
        if user_input == 'h':
            logger.info("[INFO] User opted to fallback to HTML scanning.")
            return None, None, None, {"fallback_to_html": True}
        elif user_input != 'y':
            logger.info("[INFO] User declined PDF parse. Skipping.")
            return None, None, None, {"skip_pdf": True}
    except Exception as e:
        logger.warning(f"[WARN] Failed to capture input. Skipping PDF parse: {e}")
        return None, None, None, {"skip_pdf": True}

    all_text = ""
    metadata = {}
    headers = []
    ocr_passes = int(os.getenv("OCR_ATTEMPTS", "3"))
    confidence_threshold = float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "30"))
    ocr_score = 0.0
    ocr_runs = []

    try:
        # Extract text with PyMuPDF
        doc = fitz.open(pdf_path)
        for i in range(len(doc)):
            pdf_page = doc[i]
            all_text += pdf_page.get_text()
        doc.close()
    except Exception as e:
        logger.warning(f"[WARN] fitz text extraction failed: {e}")
        all_text = ""

    # OCR fallback if needed
    if not all_text.strip() and pytesseract and pdf2image and os.getenv("ENABLE_OCR", "true").lower() == "true":
        logger.info("[INFO] Empty text result from PyMuPDF — attempting OCR fallback.")
        images = pdf2image.convert_from_path(pdf_path)
        all_text, ocr_score, ocr_runs = ocr_multi_pass(images, passes=ocr_passes, confidence_threshold=confidence_threshold)
        metadata["ocr_confidence_avg"] = round(ocr_score, 2)
        metadata["ocr_passes"] = ocr_passes

    logger.debug("[DEBUG] PDF extracted text preview (first 500 chars):" + all_text[:500])

    # Step: Basic check for tabular structure
    table_hints = os.getenv("PDF_HEADER_KEYWORDS", "precinct,votes,candidate,early,absentee,provisional").split(",")
    lines = all_text.splitlines()
    headers, header_candidates = infer_headers_and_methods(lines, table_hints)

    # Detect state and county from filename if not already present
    if "state" not in html_context or html_context.get("state") == "Unknown":
        resolved = get_handler_from_context(pdf_path)
        if resolved:
            html_context["state"] = resolved.__name__.split(".")[-1].upper()

    if "county" not in html_context or html_context.get("county") == "Unknown":
        fname = os.path.basename(pdf_path).lower()
        for part in fname.replace(".pdf", "").split("_"):
            if "county" in part:
                html_context["county"] = part.replace("county", "").strip().title() + " County"
                break

    metadata.update({
        "source_file": os.path.basename(pdf_path),
        "state": html_context.get("state", "Unknown"),
        "county": html_context.get("county", "Unknown"),
        "handler": "pdf_handler"
    })

    # Attempt contest selection (if inferred columns contain contest-like fields)
    contest_column = None
    if headers:
        rprint("[yellow]Inferred Columns:[/yellow]")
        for i, col in enumerate(headers):
            rprint(f"  [bold cyan]{i}[/bold cyan]: {col}")
        selection = input("[PROMPT] Select contest column index (or leave blank to skip): ").strip()
        if selection.isdigit():
            contest_column = headers[int(selection)]

    # Attempt row splitting from lines if table detected
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
            # Try to detect candidate/method/reporting unit columns
            candidate_cols = [col for col in headers if "candidate" in col.lower()]
            precinct_cols = [col for col in headers if any(x in col.lower() for x in ["precinct", "ward", "district", "county"])]
            method_cols = [col for col in headers if any(m in col.lower() for m in ["election day", "early", "absentee", "mail", "provisional", "total"])]

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

            contest_title = os.path.basename(pdf_path).replace(".pdf", "")

            # --- Standardize: Enrich metadata and output ---
            from ...Context_Integration.context_organizer import organize_context
            organized = organize_context(metadata)
            metadata = organized.get("metadata", metadata)
            result = finalize_election_output(headers, wide_data, contest_title, metadata)
            contest_title = result.get("contest_title", contest_title)
            metadata = result.get("metadata", metadata)
            return headers, wide_data, contest_title, metadata

        else:
            unmatched_count = len(lines[header_line_idx + 1:])
            logger.warning(f"[WARN] No structured rows matched the inferred column count of {len(headers)}. Total lines scanned: {unmatched_count}")
            fallback_rows = [{"raw_line": line} for line in lines[header_line_idx + 1:]]
            # --- Standardize fallback ---
            from ...Context_Integration.context_organizer import organize_context
            organized = organize_context(metadata)
            metadata = organized.get("metadata", metadata)
            result = finalize_election_output(["raw_line"], fallback_rows, os.path.basename(pdf_path), metadata)
            contest_title = result.get("contest_title", os.path.basename(pdf_path))
            metadata = result.get("metadata", metadata)
            return ["raw_line"], fallback_rows, contest_title, metadata

    # If no table, return plain text
    from ...Context_Integration.context_organizer import organize_context
    organized = organize_context(metadata)
    metadata = organized.get("metadata", metadata)
    result = finalize_election_output(["text"], [{"text": all_text}], os.path.basename(pdf_path), metadata)
    contest_title = result.get("contest_title", os.path.basename(pdf_path))
    metadata = result.get("metadata", metadata)
    return ["text"], [{"text": all_text}], contest_title, metadata