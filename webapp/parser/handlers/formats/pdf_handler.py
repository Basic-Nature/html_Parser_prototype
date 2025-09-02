from __future__ import annotations
# ==============================================================
# 🗳️ Smart Elections: Universal PDF Election Results Parser
# ==============================================================
import os
import re
import csv
import time
import platform
import shutil
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from ...config import (
    ENABLE_OCR, OUTPUT_DIR
)

# Optional flags/paths; provide safe defaults if missing
try:
    from ...config import ENABLE_OCR_FORCE, OCR_DEBUG_DIR
except Exception:
    ENABLE_OCR_FORCE = False
    OCR_DEBUG_DIR = os.path.join(OUTPUT_DIR, "ocr_debug")
    os.makedirs(OCR_DEBUG_DIR, exist_ok=True)

try:
    from ...config import POPPLER_PATH as CONFIG_POPPLER_PATH
except Exception:
    CONFIG_POPPLER_PATH = None
try:
    from ...config import TESSERACT_CMD as CONFIG_TESSERACT_CMD
except Exception:
    CONFIG_TESSERACT_CMD = None
    
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
    import pdf2image
    if CONFIG_TESSERACT_CMD:
        # Allow Windows to work without adding Tesseract to PATH
        try:
            pytesseract.pytesseract.tesseract_cmd = CONFIG_TESSERACT_CMD
        except Exception:
            pass
except ImportError:
    pytesseract = None
    pdf2image = None

def _log_ocr_environment(session_id=None):
    try:
        info = {
            "platform": platform.platform(),
            "pytesseract": bool(pytesseract),
            "pdf2image": bool(pdf2image),
            "poppler_path_env": bool(CONFIG_POPPLER_PATH),
            "pdftoppm_in_path": bool(shutil.which("pdftoppm")),
            "tesseract_cmd_set": bool(CONFIG_TESSERACT_CMD),
            "ENABLE_OCR": bool(ENABLE_OCR),
            "ENABLE_OCR_FORCE": bool(ENABLE_OCR_FORCE),
        }
        logger.info({
            "level": "INFO",
            "type": "handler",
            "message": "[ENV] PDF/OCR capabilities detected",
            "session_id": session_id,
            "env": info
        })
    except Exception:
        pass

def _detect_poppler_path() -> str | None:
    """
    Cross-platform Poppler locator.
    - Windows: try config POPPLER_PATH, env POPPLER_PATH, common install dirs.
    - Linux/macOS: if pdftoppm is in PATH, return None (pdf2image will use PATH).
    """
    # Config or env
    if CONFIG_POPPLER_PATH and os.path.isdir(CONFIG_POPPLER_PATH):
        return CONFIG_POPPLER_PATH
    env_path = os.environ.get("POPPLER_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path

    system = platform.system().lower()
    if system.startswith("win"):
        candidates = [
            r"C:\Program Files\poppler\Library\bin",
            r"C:\Program Files\poppler-24.08.0\Library\bin",
            r"C:\Program Files\poppler-24.07.0\Library\bin",
            r"C:\Program Files\poppler-24.06.0\Library\bin",
            r"C:\Program Files\poppler\bin",
            r"C:\poppler\bin",
        ]
        for p in candidates:
            if os.path.isdir(p):
                return p
        return None
    else:
        # On Linux/macOS we rely on PATH
        if shutil.which("pdftoppm") or shutil.which("pdftocairo"):
            return None
        return None

def _pdf_to_images(pdf_path: str, session_id=None, dpi: int = 200):
    """
    Try converting PDF pages to PIL Images:
    1) pdf2image with Poppler (Windows needs poppler_path)
    2) Fallback: render pages via PyMuPDF (fitz) to images
    """
    images = []
    # Try pdf2image first if available
    if pdf2image:
        try:
            poppler_path = _detect_poppler_path()
            kwargs = {"dpi": dpi}
            if poppler_path and platform.system().lower().startswith("win"):
                kwargs["poppler_path"] = poppler_path
            images = pdf2image.convert_from_path(pdf_path, **kwargs)
            if images:
                return images
        except Exception as e:
            logger.error({
                "level": "ERROR",
                "type": "handler",
                "message": f"[ERROR] pdf2image conversion failed (Poppler missing or error). Falling back to PyMuPDF render. {e}",
                "session_id": session_id
            })
            images = []

    # Fallback: render via PyMuPDF (no Poppler needed)
    try:
        doc = fitz.open(pdf_path)
        for i in range(len(doc)):
            page = doc[i]
            # Render to pixmap at requested DPI for OCR
            pix = page.get_pixmap(dpi=dpi)
            mode = "RGBA" if pix.alpha else "RGB"
            pil_img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
            images.append(pil_img)
        doc.close()
    except Exception as e:
        logger.error({
            "level": "ERROR",
            "type": "handler",
            "message": f"[ERROR] PyMuPDF render fallback failed: {e}",
            "session_id": session_id
        })
        images = []
    return images

def _prep_variants(images):
    """
    Yield (name, images_variant) for multiple preprocessing paths.
    """
    variants = []
    # identity
    variants.append(("none", images))
    # grayscale
    gray = [ImageOps.grayscale(img) for img in images]
    variants.append(("gray", gray))
    # adaptive threshold (simple)
    thresh = [ImageOps.autocontrast(ImageOps.grayscale(img)).point(lambda p: 255 if p > 180 else 0, mode='1') for img in images]
    variants.append(("thresh", thresh))
    # sharpen + contrast
    sharp = [ImageEnhance.Contrast(img.filter(ImageFilter.SHARPEN)).enhance(1.5) for img in gray]
    variants.append(("sharp_contrast", sharp))
    return variants

def _ocr_images(images, tesseract_config: str, confidence_threshold=30):
    """
    Run pytesseract on a list of PIL images and return combined text and avg confidence.
    """
    if not pytesseract:
        return "", 0.0, []

    page_texts = []
    confs_all = []
    # Per-page confidences for debugging
    per_page = []

    for img in images:
        text = ""
        details = {}
        if hasattr(pytesseract, "Output"):
            try:
                details = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=tesseract_config)
            except Exception:
                # Fallback to plain text if data API fails
                try:
                    text = pytesseract.image_to_string(img, config=tesseract_config)
                except Exception:
                    text = ""
            else:
                # Prefer confidences from the data API when available
                words = details.get("text", []) or []
                confs = details.get("conf", []) or []
                for j in range(len(words)):
                    word = (words[j] or "").strip()
                    conf_raw = confs[j] if j < len(confs) else "-1"
                    if word:
                        try:
                            conf_val = float(conf_raw)
                        except Exception:
                            conf_val = -1.0
                        confs_all.append(conf_val)
                        if conf_val >= confidence_threshold:
                            text += word + " "
        else:
            try:
                text = pytesseract.image_to_string(img, config=tesseract_config)
            except Exception:
                text = ""

        page_texts.append(text)
        if details:
            vals = []
            for c in details.get("conf", []):
                try:
                    vals.append(float(c))
                except Exception:
                    pass
            per_page.append(sum(vals) / len(vals) if vals else 0.0)
        else:
            per_page.append(0.0)

    avg_conf = sum(confs_all) / len(confs_all) if confs_all else 0.0
    return "\n".join(page_texts), avg_conf, per_page

def adaptive_ocr_pipeline(pdf_path, session_id=None, target_conf=60.0, max_seconds=120, max_runs=20):
    """
    Adaptive OCR loop:
    - Try different DPIs, preprocessors, and Tesseract configs (psm/oem)
    - Keep the best result by avg confidence
    - Early stop on reaching target_conf or exceeding budgets
    Returns: best_text, best_conf, runs_summary(list of dict)
    """
    start = time.time()
    runs_summary = []
    best = {"text": "", "conf": 0.0, "params": {}}

    dpi_list = [200, 250, 300]
    # Favor structured page mode first, then single-block variants
    psm_list = [6, 4, 3, 11, 12, 1, 13]
    oem_list = [3, 1]  # LSTM; legacy
    conf_threshold_word = 30

    for dpi in dpi_list:
        if time.time() - start > max_seconds or len(runs_summary) >= max_runs:
            break

        images = _pdf_to_images(pdf_path, session_id=session_id, dpi=dpi)
        if not images:
            continue

        for prep_name, prep_imgs in _prep_variants(images):
            if time.time() - start > max_seconds or len(runs_summary) >= max_runs:
                break

            for oem in oem_list:
                for psm in psm_list:
                    if time.time() - start > max_seconds or len(runs_summary) >= max_runs:
                        break
                    config = f"--oem {oem} --psm {psm}"
                    text, avg_conf, per_page = _ocr_images(prep_imgs, config, confidence_threshold=conf_threshold_word)

                    # Record run
                    run = {
                        "dpi": dpi,
                        "prep": prep_name,
                        "oem": oem,
                        "psm": psm,
                        "avg_conf": round(avg_conf, 2),
                        "per_page": [round(c, 2) for c in per_page]
                    }
                    runs_summary.append(run)

                    # Update best
                    if avg_conf > best["conf"]:
                        best = {"text": text, "conf": avg_conf, "params": {"dpi": dpi, "prep": prep_name, "oem": oem, "psm": psm}}

                    # Early stop when good enough
                    if avg_conf >= target_conf:
                        break
                else:
                    continue
                break
            else:
                continue
            break

    # Combine high-confidence lines across top runs to improve recall
    if runs_summary:
        # sort by confidence
        top = sorted(runs_summary, key=lambda r: r["avg_conf"], reverse=True)[:5]
        # Re-run OCR quickly for those top settings to collect lines
        line_sets = []
        for r in top:
            imgs = _pdf_to_images(pdf_path, session_id=session_id, dpi=r["dpi"]) or []
            if not imgs:
                continue
            # reapply preprocess
            prep_variants = dict(_prep_variants(imgs))
            imgs2 = prep_variants.get(r["prep"], imgs)
            txt, _, _ = _ocr_images(imgs2, f"--oem {r['oem']} --psm {r['psm']}", confidence_threshold=conf_threshold_word)
            line_sets.append(set((txt or "").splitlines()))
        if line_sets:
            combined = sorted(set.union(*line_sets))
            combined_text = "\n".join(combined)
            # Keep the better of combined vs. best raw
            if len(combined_text) > len(best["text"]):
                best["text"] = combined_text

    return best["text"], best["conf"], runs_summary

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

def _extract_text_multi(pdf_path, session_id=None):
    """
    Try multiple PyMuPDF extract modes and pick the longest.
    """
    try:
        doc = fitz.open(pdf_path)
        texts = {}
        modes = ["text", "blocks", "raw", "xhtml"]
        for m in modes:
            buf = []
            for i in range(len(doc)):
                try:
                    buf.append(doc[i].get_text(m))
                except Exception:
                    continue
            texts[m] = "\n".join(buf)
        doc.close()
        # pick the mode with most characters
        best_mode = max(texts, key=lambda k: len(texts.get(k) or ""))
        return texts.get(best_mode) or "", best_mode
    except Exception as e:
        logger.warning({
            "level": "WARNING",
            "type": "handler",
            "message": f"[WARN] Multi-mode text extraction failed: {e}",
            "session_id": session_id
        })
        return "", "error"

def _save_ocr_debug_images(pdf_path, session_id=None, dpi=300, limit=2):
    try:
        imgs = _pdf_to_images(pdf_path, session_id=session_id, dpi=dpi)[:limit]
        saved = []
        for idx, img in enumerate(imgs):
            out = os.path.join(
                OCR_DEBUG_DIR,
                f"{os.path.splitext(os.path.basename(pdf_path))[0]}_p{idx+1}_{dpi}dpi.png"
            )
            try:
                img.save(out)
                saved.append(str(out))
            except Exception:
                pass
        if saved:
            logger.info({
                "level": "INFO",
                "type": "handler",
                "message": "[DEBUG] Saved OCR debug raster(s)",
                "session_id": session_id,
                "files": saved
            })
    except Exception:
        pass

def infer_headers_and_methods(lines, table_hints):
    header_candidates = [line for line in lines if sum(1 for hint in table_hints if hint in line.lower()) >= 2]
    headers = []
    if header_candidates:
        headers = re.split(r"\s{2,}|\t|,", header_candidates[0].strip())
        headers = [h.strip() for h in headers if h.strip()]
    return headers, header_candidates

def parse_pdf_election_results(pdf_path, session_id=None):
    _log_ocr_environment(session_id=session_id)
    all_text = ""
    metadata = {}
    headers = []
    ocr_score = 0.0
    ocr_runs = []

    # Try standard text first
    try:
        doc = fitz.open(pdf_path)
        for i in range(len(doc)):
            all_text += doc[i].get_text()
        doc.close()
    except Exception as e:
        logger.warning({
            "level": "WARNING",
            "type": "handler",
            "message": f"[WARN] fitz text extraction failed: {e}",
            "session_id": session_id
        })
        all_text = ""

    # If empty or forced, try alternative extract modes
    if (not all_text.strip()) or ENABLE_OCR_FORCE:
        alt_text, mode_used = _extract_text_multi(pdf_path, session_id=session_id)
        if len(alt_text) > len(all_text):
            all_text = alt_text
            metadata["fitz_mode_used"] = mode_used

    # OCR fallback (adaptive, cross‑platform)
    need_ocr = (not all_text.strip()) and bool(pytesseract) and ENABLE_OCR
    if need_ocr or ENABLE_OCR_FORCE:
        if not all_text.strip():
            logger.info({
                "level": "INFO",
                "type": "handler",
                "message": "[INFO] Empty/forced OCR — attempting adaptive OCR fallback.",
                "session_id": session_id
            })
        _save_ocr_debug_images(pdf_path, session_id=session_id, dpi=300, limit=2)
        best_text, best_conf, runs_summary = adaptive_ocr_pipeline(
            pdf_path,
            session_id=session_id,
            target_conf=65.0,
            max_seconds=150,
            max_runs=28
        )
        # Prefer OCR text if we had none, or if significantly better
        if (not all_text.strip()) or (len(best_text) > len(all_text) * 1.25):
            all_text = best_text or all_text
            ocr_score = best_conf or 0.0
            ocr_runs = runs_summary or []
            metadata["ocr_confidence_avg"] = round(ocr_score, 2)
            metadata["ocr_runs"] = ocr_runs
            metadata["ocr_used"] = True
        else:
            metadata["ocr_used"] = False

    logger.debug({
        "level": "DEBUG",
        "type": "handler",
        "message": "[DEBUG] PDF extracted text preview (first 500 chars):" + (all_text[:500] if isinstance(all_text, str) else str(all_text)[:500]),
        "session_id": session_id
    })

    # Fix set|list union TypeError by casting to set(...)
    table_hints = list(
        set(LOCATION_KEYWORDS) | set(CANDIDATE_KEYWORDS) | set(BALLOT_TYPES) |
        set(PARTY_KEYWORDS) | set(TOTAL_KEYWORDS) | set(MISC_FOOTER_KEYWORDS) | set(CONTEST_KEYWORDS)
    )
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
            method_keys = set(BALLOT_TYPES) | set(TOTAL_KEYWORDS) | set(MISC_FOOTER_KEYWORDS)
            method_cols = [col for col in headers if any(m in col.lower() for m in method_keys)]

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