# handlers/formats/pdf_handler.py
# ==============================================================
# Parses election results from PDF files.
# Includes dynamic scanning, OCR fallback, and optional user prompt.
# ==============================================================
import csv
import os
import re
from concurrent.futures import ThreadPoolExecutor
from rich import print as rprint
from webapp.parser.state_router import resolve_state_handler
from utils.output_utils import get_output_path, format_timestamp
from typing import cast
from fitz import Page

try:
    import fitz  # PyMuPDF
    from typing import cast
    from fitz import Page
except ImportError:
    raise ImportError("You must install PyMuPDF to use the PDF handler: pip install pymupdf")

try:
    import pytesseract
    from PIL import Image
    import pdf2image
except ImportError:
    pytesseract = None
    pdf2image = None

from utils.shared_logger import logger

def parse(page, html_context):
    # Respect early skip signal from calling context
    if html_context.get("skip_format") or html_context.get("manual_skip"):
        logger.info("[SKIP] PDF parsing intentionally skipped via context flag.")
        return None, None, None, {"skipped": True}    
    try:
        """
        Parses a PDF file from the /input/ folder (typically downloaded).
        Returns extracted plain text for further processing.
        User must approve before parsing.
        """
        pdf_path = html_context.get("pdf_source")
        if not pdf_path:
            try:
                pdf_files = [f for f in os.listdir("input") if f.lower().endswith(".pdf")]
                if pdf_files:
                    pdf_files.sort(key=lambda x: os.path.getmtime(os.path.join("input", x)), reverse=True)
                    pdf_path = os.path.join("input", pdf_files[0])
                    logger.info(f"[FALLBACK] Using most recent PDF file: {pdf_path}")
                else:
                    logger.error("[ERROR] No PDF files found in the input directory.")
                    return None, None, None, {"error": "No PDF in input folder"}
            except Exception as e:
                logger.error(f"[ERROR] Error during PDF file discovery: {e}")
                return None, None, None, {"error": str(e)}

        try:
            user_input = input(f"[PROMPT] PDF file detected at {os.path.basename(pdf_path)}. Parse this file? (y/n, or 'h' to fallback to HTML): ").strip().lower()
            if user_input == 'h':
                logger.info("[INFO] User opted to fallback to HTML scanning.")
                return None, None, None, {"fallback_to_html": True}
            elif user_input != 'y':
                logger.info("[INFO] User declined PDF parse. Falling back to HTML-based scraping.")
                return None, None, None, {"skip_pdf": True}
        except Exception as e:
            logger.warning(f"[WARN] Failed to capture input. Skipping PDF parse: {e}")
            return None, None, None, {"skip_pdf": True}      
        all_text = ""
        metadata = {}
        headers = []
        header_candidates = []
        ocr_passes = 0
      
        if os.getenv("USE_PYMUPDF", "true").lower() == "true":
            try:
                doc = fitz.open(pdf_path)
                for i in range(len(doc)):
                     pdf_page: fitz.Page = doc[i]  # type: ignore
                     all_text += pdf_page.get_text()  # type: ignore # ← SAFE: all_text predeclared
                doc.close()
            except Exception as e:
                logger.warning(f"[WARN] fitz text extraction failed: {e}")

            # Fix: ensure OCR only triggers when libraries are available
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
                                    else:
                                        logger.debug(f"[OCR-LOW] Word '{word}' skipped due to low confidence ({conf_val})")
                                except ValueError:
                                    logger.debug(f"[OCR-WARN] Invalid confidence '{conf}' for word '{word}'")
                    return page_text, confidences

                for i in range(ocr_passes):
                    logger.info(f"[INFO] OCR pass {i+1} of {ocr_passes}")
                    ocr_text = ""
                    confidences = []
                    with ThreadPoolExecutor() as executor:
                        results = list(executor.map(process_image_ocr, images))
                    page_scores = []
                    for page_num, (text, conf_list) in enumerate(results):
                        ocr_text += text + "\n"
                        confidences.extend(conf_list)
                        if conf_list:
                            page_avg = sum(conf_list) / len(conf_list)
                            page_scores.append((page_num + 1, page_avg))
                    if os.getenv("OCR_SHOW_HEATMAP", "true").lower() == "true": 
                        rprint("\n[yellow]OCR Confidence Heatmap (per page):[/yellow]")
                        for page, score in page_scores:
                            color = "green" if score >= 80 else "yellow" if score >= 50 else "red"
                            rprint(f"[{color}]Page {page:02}[/]: {score:.1f}")
                    low_pages = [f"Page {p}" for p, s in page_scores if s < confidence_threshold]
                    if low_pages:
                        logger.warning(f"[OCR] Low-confidence pages below threshold ({confidence_threshold}): {', '.join(low_pages)}")

                    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
                    logger.info(f"[OCR] Average confidence for pass {i+1}: {avg_conf:.2f}")
                    pass_confidences.append(avg_conf)
                    ocr_runs.append(ocr_text)

                # Merge and dedupe lines
                line_sets = [set(text.splitlines()) for text in ocr_runs]
                combined_lines = sorted(set.union(*line_sets))
                all_text = "".join(combined_lines)
                overall_avg = sum(pass_confidences) / len(pass_confidences) if pass_confidences else 0.0
                metadata["ocr_confidence_avg"] = round(overall_avg, 2)
                low_threshold = float(os.getenv("OCR_WARN_THRESHOLD", "35"))
                if overall_avg < low_threshold:
                    logger.warning(f"[WARN] Average OCR confidence ({overall_avg:.2f}) is below the warning threshold of {low_threshold}.")
                    retry_enabled = os.getenv("OCR_RETRY_IF_LOW", "true").lower() == "true"
                    if retry_enabled:
                        retry_count = int(html_context.get("ocr_retry_count", 0))
                        max_retries = int(os.getenv("OCR_MAX_RETRIES", "1"))
                        if retry_count < max_retries:
                            logger.warning(f"[RETRY] Confidence too low — retrying OCR (attempt {retry_count + 1} of {max_retries})")
                            html_context["ocr_retry_count"] = retry_count + 1
                            return parse(page, html_context)
                        else:
                            logger.warning("[RETRY] Max OCR retries reached. Skipping further attempts.")

        # Optional: export OCR output as .txt if enabled
        if os.getenv("EXPORT_PARSED_TEXT", "false").lower() == "true":
            parsed_dir = "parsed_output"
            os.makedirs(parsed_dir, exist_ok=True)
            csv_output_path = get_output_path(metadata.get("state"), metadata.get("county"), "parsed")
            out_path = os.path.join(csv_output_path, os.path.basename(pdf_path).replace(".pdf", ".txt"))
            with open(out_path, "w", encoding="utf-8") as f_out:
                f_out.write("# OCR Extracted Text (Post Deduplication)")
                timestamp = format_timestamp()
                f_out.write(f"# Timestamp: {timestamp}\n\n")
                f_out.write(all_text)
                f_out.write("# Inferred Metadata")
                for k, v in metadata.items():
                    f_out.write(f"{k}: {v}")
                f_out.write(f"OCR passes: {ocr_passes}")
            logger.info(f"[OUTPUT] Parsed OCR text saved to: {out_path}")

        logger.debug("[DEBUG] PDF extracted text preview (first 500 chars):" + all_text[:500])

        # Step: Basic check for tabular structure
        table_hints = os.getenv("PDF_HEADER_KEYWORDS", "precinct,votes,candidate,early,absentee,provisional").split(",")
        lines = all_text.splitlines()
        matches = [line for line in lines if sum(1 for hint in table_hints if hint in line.lower()) >= 2]
        multi_column_lines = [line for line in lines if line.count("  ") > 3 or line.count("	") > 2]

        if matches or multi_column_lines:
            logger.info("[INFO] Table-like structure detected in PDF — tabular extraction may be possible.")

            # Step: Attempt basic column inference
            header_candidates = [line for line in lines if all(h in line.lower() for h in ["precinct", "votes"])]
            if header_candidates:
                headers = header_candidates[0].split()
                logger.info(f"[INFO] Inferred column headers: {headers}")
            else:
                logger.info("[INFO] No strong header line found. Will treat first non-empty row as fallback header.")
        else:
            logger.info("[INFO] No table-like patterns detected in PDF — treating as plain text blob.")

        # Step: Detect state and county from filename if not already present
        if "state" not in html_context or html_context["state"] == "Unknown":
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
            "handler": "pdf_handler"
        })

        # Step: Attempt contest selection (if inferred columns contain contest-like fields)
        contest_column = None
        if headers:
            rprint("[yellow]Inferred Columns:[/yellow]")
            for i, col in enumerate(headers):
                rprint(f"  [bold cyan]{i}[/bold cyan]: {col}")
            selection = input("[PROMPT] Select contest column index (or leave blank to skip): ").strip()
            if selection.isdigit():
                contest_column = headers[int(selection)]

        # Step: Attempt row splitting from lines if table detected
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
                logger.info(f"[INFO] Extracted {len(data)} structured row(s) from table.")
            else:
                unmatched_count = len(content_lines[start_index + 1:])
                logger.warning(f"[WARN] No structured rows matched the inferred column count of {len(headers)}. Total lines scanned: {unmatched_count}")
                fallback_rows = [{"raw_line": line} for line in content_lines[start_index + 1:]]
                return os.path.basename(pdf_path), ["raw_line"], fallback_rows, metadata

        if data:
            safe_title = re.sub(r"[\\/*?\"<>|]", "_", os.path.basename(pdf_path).replace(".pdf", ""))
            output_path = get_output_path(metadata["state"], metadata.get("county", "Unknown"), "parsed")
            timestamp = format_timestamp() if os.getenv("INCLUDE_TIMESTAMP_IN_FILENAME", "true").lower() == "true" else ""
            filename = f"{safe_title}_results_{timestamp}.csv" if timestamp else f"{safe_title}_results.csv"
            filepath = os.path.join(output_path, filename)

            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(data)
                f.write(f"\n# Generated at: {format_timestamp()}")
            metadata["output_file"] = filepath
            logger.info(f"[OUTPUT] PDF Contest Results saved to: {filepath}")
            return headers, data, os.path.basename(pdf_path), metadata
        return os.path.basename(pdf_path), ["text"], [{"text": all_text}], metadata
    except Exception as e:
        logger.error(f"[ERROR] Failed to parse PDF: {e}")
        return None, None, None, {"error": str(e)}
