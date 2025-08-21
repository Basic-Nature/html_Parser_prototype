from __future__ import annotations
# ==============================================================
# 🗳️ Smart Elections: Universal PDF Election Results Parser
# ==============================================================
import os
import re
import csv
import time
from concurrent.futures import ThreadPoolExecutor
from ...config import (
    ENABLE_OCR, OUTPUT_DIR
)
from ...utils.logger_singleton import logger, prompt
from ...Context_Integration.Context_Library.constants import (
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

def ocr_multi_pass(images, passes=3, confidence_threshold=30, session_id=None):
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
        logger.info({
            "level": "INFO",
            "type": "handler",
            "message": f"[INFO] OCR pass {i+1} of {passes}",
            "session_id": session_id
        })
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

    line_sets = [set(text.splitlines()) for text in ocr_runs]
    combined_lines = sorted(set.union(*line_sets))
    all_text = "\n".join(combined_lines)
    overall_avg = sum(pass_confidences) / len(pass_confidences) if pass_confidences else 0.0
    return all_text, overall_avg, ocr_runs

def infer_headers_and_methods(lines, table_hints):
    header_candidates = [line for line in lines if sum(1 for hint in table_hints if hint in line.lower()) >= 2]
    headers = []
    if header_candidates:
        headers = re.split(r"\s{2,}|\t|,", header_candidates[0].strip())
        headers = [h.strip() for h in headers if h.strip()]
    return headers, header_candidates

def parse_pdf_election_results(pdf_path, session_id=None):
    all_text = ""
    metadata = {}
    headers = []
    ocr_score = 0.0
    ocr_runs = []

    try:
        doc = fitz.open(pdf_path)
        for i in range(len(doc)):
            pdf_page = doc[i]
            all_text += pdf_page.get_text()
        doc.close()
    except Exception as e:
        logger.warning({
            "level": "WARNING",
            "type": "handler",
            "message": f"[WARN] fitz text extraction failed: {e}",
            "session_id": session_id
        })
        all_text = ""

    if not all_text.strip() and pytesseract and pdf2image and ENABLE_OCR:
        logger.info({
            "level": "INFO",
            "type": "handler",
            "message": "[INFO] Empty text result from PyMuPDF — attempting OCR fallback.",
            "session_id": session_id
        })
        images = pdf2image.convert_from_path(pdf_path)
        all_text, ocr_score, ocr_runs = ocr_multi_pass(images, passes=3, confidence_threshold=30, session_id=session_id)
        metadata["ocr_confidence_avg"] = round(ocr_score, 2)
        metadata["ocr_passes"] = 3

    logger.debug({
        "level": "DEBUG",
        "type": "handler",
        "message": "[DEBUG] PDF extracted text preview (first 500 chars):" + all_text[:500],
        "session_id": session_id
    })

    table_hints = list(LOCATION_KEYWORDS | CANDIDATE_KEYWORDS | BALLOT_TYPES | PARTY_KEYWORDS | TOTAL_KEYWORDS | MISC_FOOTER_KEYWORDS | CONTEST_KEYWORDS)
    lines = all_text.splitlines()
    headers, header_candidates = infer_headers_and_methods(lines, table_hints)

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

    contest_column = None
    if headers:
        logger.info({
            "level": "INFO",
            "type": "input",
            "message": "[INFO] Inferred Columns:",
            "session_id": session_id
        })
        for i, col in enumerate(headers):
            logger.info({
                "level": "INFO",
                "type": "input",
                "message": f"  [{i}]: {col}",
                "session_id": session_id
            })
        prompt_message = "[PROMPT] Select contest column index (or leave blank to skip): "
        def validator(x):
            return x == "" or (x.isdigit() and 0 <= int(x) < len(headers))
        selection = prompt.prompt_input(
            prompt_message,
            validator=validator,
            session_id=session_id,
            context={"headers": headers}
        )
        if selection and selection.isdigit():
            contest_column = headers[int(selection)]

    data = []
    if headers:
        header_line_idx = None
        for idx, line in enumerate(lines):
            if all(h.lower() in line.lower() for h in headers[:2]):
                header_line_idx = idx
                break
        if header_line_idx is None and header_candidates:
            try:
                header_line_idx = lines.index(header_candidates[0])
            except ValueError:
                header_line_idx = 0
        if header_line_idx is None:
            header_line_idx = 0

        for line in lines[header_line_idx + 1:]:
            if not line.strip():
                continue
            row = re.split(r"\s{2,}|\t|,", line.strip())
            row = [cell.strip() for cell in row if cell.strip()]
            if len(row) == len(headers):
                row_dict = dict(zip(headers, row))
                data.append(row_dict)

        contest = None
        if contest_column:
            contests = sorted({row[contest_column].strip() for row in data if row.get(contest_column)})
            if len(contests) > 1:
                logger.info({
                    "level": "INFO",
                    "type": "input",
                    "message": "\nMultiple contests detected:",
                    "session_id": session_id
                })
                for i, name in enumerate(contests, 1):
                    logger.info({
                        "level": "INFO",
                        "type": "input",
                        "message": f" {i:2d}. {name}",
                        "session_id": session_id
                    })
                prompt_message = "\nEnter the contest name (exactly as shown), or type its number: "
                def validator(x):
                    x = str(x).strip()
                    if x.isdigit():
                        idx = int(x)
                        return 1 <= idx <= len(contests)
                    return x in contests
                user_input = prompt.prompt_input(
                    prompt_message,
                    validator=validator,
                    session_id=session_id,
                    context={"contests": contests}
                )
                if user_input is None:
                    logger.error({
                        "level": "ERROR",
                        "type": "input",
                        "message": "No contest selected.",
                        "session_id": session_id
                    })
                    return None, None, None, {"error": "No contest selected"}
                if str(user_input).isdigit():
                    idx = int(user_input)
                    try:
                        contest = contests[idx - 1]
                    except IndexError:
                        logger.error({
                            "level": "ERROR",
                            "type": "input",
                            "message": "Invalid contest number.",
                            "session_id": session_id
                        })
                        return None, None, None, {"error": "Invalid contest number"}
                else:
                    if user_input not in contests:
                        logger.error({
                            "level": "ERROR",
                            "type": "input",
                            "message": f"[ERROR] Contest name '{user_input}' not found.",
                            "session_id": session_id
                        })
                        return None, None, None, {"error": "Contest name not found"}
                    contest = user_input
                data = [row for row in data if row.get(contest_column, "").strip() == contest]
            elif contests:
                contest = contests[0]
        else:
            contest = os.path.basename(pdf_path).replace(".pdf", "")

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

            all_keys = set()
            for row in wide_data:
                all_keys.update(row.keys())
            headers = [reporting_unit_col] + sorted([k for k in all_keys if k != reporting_unit_col])
            headers, wide_data = harmonize_headers_and_data(headers, wide_data)

            safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in os.path.basename(pdf_path).replace(".pdf", "")).replace(" ", "_")
            output_csv = os.path.join(OUTPUT_DIR, f"{safe_title}_parsed.csv")
            output_meta = os.path.join(OUTPUT_DIR, f"{safe_title}_metadata.json")

            with open(output_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                for row in wide_data:
                    writer.writerow(row)

            metadata.update({
                "output_file": os.path.basename(output_csv),
                "headers": headers,
                "row_count": len(wide_data)
            })
            with open(output_meta, "w") as jf:
                jf.write(orjson.dumps(metadata, option=orjson.OPT_INDENT_2).decode("utf-8"))

            logger.info({
                "level": "INFO",
                "type": "output",
                "message": f"[OUTPUT] Wrote {len(wide_data)} rows to: {output_csv}",
                "session_id": session_id
            })
            logger.info({
                "level": "INFO",
                "type": "output",
                "message": f"[OUTPUT] Metadata written to: {output_meta}",
                "session_id": session_id
            })

            return headers, wide_data, safe_title, metadata

        else:
            unmatched_count = len(lines[header_line_idx + 1:])
            logger.warning({
                "level": "WARNING",
                "type": "output",
                "message": f"[WARN] No structured rows matched the inferred column count of {len(headers)}. Total lines scanned: {unmatched_count}",
                "session_id": session_id
            })
            fallback_rows = [{"raw_line": line} for line in lines[header_line_idx + 1:]]
            safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in os.path.basename(pdf_path).replace(".pdf", "")).replace(" ", "_")
            output_csv = os.path.join(OUTPUT_DIR, f"{safe_title}_parsed.csv")
            output_meta = os.path.join(OUTPUT_DIR, f"{safe_title}_metadata.json")
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
            logger.warning({
                "level": "WARNING",
                "type": "output",
                "message": f"[OUTPUT] Wrote fallback rows to: {output_csv}",
                "session_id": session_id
            })
            return ["raw_line"], fallback_rows, safe_title, metadata

    safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in os.path.basename(pdf_path).replace(".pdf", "")).replace(" ", "_")
    output_csv = os.path.join(OUTPUT_DIR, f"{safe_title}_parsed.csv")
    output_meta = os.path.join(OUTPUT_DIR, f"{safe_title}_metadata.json")
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
    logger.warning({
        "level": "WARNING",
        "type": "output",
        "message": f"[OUTPUT] Wrote plain text to: {output_csv}",
        "session_id": session_id
    })
    return ["text"], [{"text": all_text}], safe_title, metadata

def parse(page=None, coordinator=None, html_context=None, manual_file=None, session_id=None, **kwargs):
    """
    Universal pipeline entry: Accepts a PDF file path (manual_file) from the format router.
    Returns: headers, data, contest, metadata
    """
    html_context = html_context or {}
    if html_context.get("skip_format") or html_context.get("manual_skip"):
        logger.info({
            "level": "INFO",
            "type": "handler",
            "message": "[SKIP] PDF parsing intentionally skipped via context flag.",
            "session_id": session_id
        })
        return None, None, None, {"skipped": True}

    if not manual_file or not os.path.isfile(manual_file):
        logger.error({
            "level": "ERROR",
            "type": "handler",
            "message": "[ERROR] No PDF file provided to parse().",
            "session_id": session_id
        })
        return None, None, None, {"skipped": True}

    return parse_pdf_election_results(manual_file, session_id=session_id)