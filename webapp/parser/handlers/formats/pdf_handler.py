# handlers/formats/pdf_handler.py
# ==============================================================
# Parses election results from PDF files.
# Includes dynamic scanning, OCR fallback, and optional user prompt.
# ==============================================================

import os
import re
import csv
from concurrent.futures import ThreadPoolExecutor
from rich import print as rprint
from ...state_router import resolve_state_handler
from ...utils.output_utils import finalize_election_output
from ...utils.shared_logger import logger

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
    ocr_passes = 0

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
        ocr_runs = []
        pass_confidences = []
        ocr_passes = int(os.getenv("OCR_ATTEMPTS", "3"))
        confidence_threshold = float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "30"))

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

        for i in range(ocr_passes):
            logger.info(f"[INFO] OCR pass {i+1} of {ocr_passes}")
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

        # Merge and dedupe lines
        line_sets = [set(text.splitlines()) for text in ocr_runs]
        combined_lines = sorted(set.union(*line_sets))
        all_text = "\n".join(combined_lines)
        overall_avg = sum(pass_confidences) / len(pass_confidences) if pass_confidences else 0.0
        metadata["ocr_confidence_avg"] = round(overall_avg, 2)

    logger.debug("[DEBUG] PDF extracted text preview (first 500 chars):" + all_text[:500])

    # Step: Basic check for tabular structure
    table_hints = os.getenv("PDF_HEADER_KEYWORDS", "precinct,votes,candidate,early,absentee,provisional").split(",")
    lines = all_text.splitlines()
    matches = [line for line in lines if sum(1 for hint in table_hints if hint in line.lower()) >= 2]
    multi_column_lines = [line for line in lines if line.count("  ") > 3 or line.count("\t") > 2]

    # Attempt basic column inference
    header_candidates = [line for line in lines if all(h in line.lower() for h in ["precinct", "votes"])]
    if header_candidates:
        headers = header_candidates[0].split()
        logger.info(f"[INFO] Inferred column headers: {headers}")
    else:
        logger.info("[INFO] No strong header line found. Will treat first non-empty row as fallback header.")

    # Detect state and county from filename if not already present
    if "state" not in html_context or html_context.get("state") == "Unknown":
        resolved = resolve_state_handler(pdf_path)
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
    if matches or multi_column_lines:
        content_lines = [line.strip() for line in lines if line.strip()]
        if header_candidates:
            try:
                start_index = content_lines.index(header_candidates[0])
            except ValueError:
                start_index = 0
        else:
            start_index = 0
        for line in content_lines[start_index + 1:]:
            row = line.split()
            if contest_column and headers and len(row) == len(headers):
                contest_value = row[headers.index(contest_column)]
                if contest_value:
                    data.append(dict(zip(headers, row)))
            elif headers and len(row) == len(headers):
                data.append(dict(zip(headers, row)))

        if data:
            contest_title = os.path.basename(pdf_path).replace(".pdf", "")
            # Output via finalize_election_output
            result = finalize_election_output(headers, data, contest_title, metadata)
            contest_title = result.get("contest_title", contest_title)
            metadata = result.get("metadata", metadata)
            return headers, data, contest_title, metadata
        else:
            unmatched_count = len(content_lines[start_index + 1:])
            logger.warning(f"[WARN] No structured rows matched the inferred column count of {len(headers)}. Total lines scanned: {unmatched_count}")
            fallback_rows = [{"raw_line": line} for line in content_lines[start_index + 1:]]
            return ["raw_line"], fallback_rows, os.path.basename(pdf_path), metadata

    # If no table, return plain text
    return ["text"], [{"text": all_text}], os.path.basename(pdf_path), metadata