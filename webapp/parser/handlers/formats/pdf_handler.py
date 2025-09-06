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
import html
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
    MISC_FOOTER_KEYWORDS, CONTEST_KEYWORDS, CONTEST_TITLE_SKIP_PHRASES,
    CONTEST_HEADER_KEYWORDS, CONTEST_HEADER_PREFERENCE
)
from ...utils.table_core import harmonize_headers_and_data
import orjson
from ...utils.contest_selector import select_contest
from ...utils.table_builder import build_table_noninteractive
from ...utils.output_utils import finalize_election_output
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

def _detect_contest_titles_from_text(lines):
    """
    Heuristic detection of contest titles from plain PDF text using constants.
    - Keep lines containing contest/office keywords
    - Drop known skip phrases and very short/noisy lines
    """
    titles = []
    skip_set = {s.lower() for s in (CONTEST_TITLE_SKIP_PHRASES or set())}
    for line in lines:
        raw = (line or "").strip()
        low = raw.lower()
        if not raw or len(raw) < 6:
            continue
        if any(s in low for s in skip_set):
            continue
        if any(kw in low for kw in CONTEST_KEYWORDS):
            titles.append(raw)
    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for t in titles:
        k = t.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(t)
    return uniq[:50]

def _is_mostly_markup(text: str) -> bool:
    """
    Return True if the extracted 'text' is actually markup-wrappers (e.g., <img> tags) with little real text.
    """
    if not isinstance(text, str):
        return False
    s = text.strip().lower()
    if not s:
        return False
    # Heuristics: presence of HTML tags + low alphabetic character count
    has_tags = any(tok in s for tok in ("<img", "<div", "<span", "<html", "<svg", "<p", "<table", "data:image/"))
    if not has_tags:
        return False
    alpha = sum(1 for ch in s[:8000] if ch.isalpha())
    return alpha < 200

def _sanitize_extracted_text(text: str) -> str:
    """
    Convert raw extracted content (which may contain XHTML, <img src="data:image..."> etc.)
    into neat, readable lines for downstream steps.
    - Remove data:image/base64 payloads and HTML tags
    - Unescape entities
    - Collapse whitespace
    - Drop extremely noisy/empty lines
    """
    if not isinstance(text, str):
        return ""
    # Remove data:image base64 attributes entirely
    text = re.sub(r'src\s*=\s*"data:image/[^"]+"', 'src="[image]"', text, flags=re.IGNORECASE)
    # Remove long base64-like runs that may appear outside attributes
    text = re.sub(r'[A-Za-z0-9+/=]{200,}', ' ', text)
    # Strip all HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Unescape HTML entities
    try:
        text = html.unescape(text)
    except Exception:
        pass
    # Normalize whitespace but keep line structure
    lines = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        # Collapse internal whitespace
        s = re.sub(r'\s+', ' ', s)
        # Heuristics: keep lines that have some alphanum signal
        alnum = sum(ch.isalnum() for ch in s)
        if alnum < 2:
            continue
        # Drop bracket-only image placeholders
        if s in {"[image]", "[data]"}:
            continue
        # Avoid lines that are mostly punctuation
        punct = sum(not ch.isalnum() and not ch.isspace() for ch in s)
        if alnum and punct / max(1, len(s)) > 0.6:
            continue
        lines.append(s)
    # Deduplicate consecutive duplicates
    neat = []
    last = None
    for l in lines:
        if l != last:
            neat.append(l)
            last = l
    return "\n".join(neat)

def _safe_slug(text: str, max_len: int = 100) -> str:
    """
    Make a filesystem-friendly slug:
    - Keep alnum, space, underscore, hyphen; replace others with '_'
    - Collapse repeated underscores/spaces; convert spaces to underscores
    - Trim length to max_len
    """
    if not isinstance(text, str):
        return ""
    stem = os.path.splitext(text)[0]
    s = "".join(c if c.isalnum() or c in " _-" else "_" for c in stem)
    s = re.sub(r"[ _]+", " ", s).strip()
    s = s.replace(" ", "_")
    s = re.sub(r"_+", "_", s)
    return s[:max_len] or "untitled"

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
    Only use modes that return strings.
    """
    try:
        doc = fitz.open(pdf_path)
        texts = {}
        # use string-returning modes only
        modes = ["text", "raw", "html", "xhtml"]
        for m in modes:
            buf = []
            for i in range(len(doc)):
                try:
                    t = doc[i].get_text(m)
                    if not isinstance(t, str):
                        t = ""
                    buf.append(t)
                except Exception:
                    continue
            texts[m] = "\n".join(buf)
        doc.close()
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

def parse_pdf_election_results(pdf_path, session_id=None, coordinator=None) -> tuple[list[str], list[dict], str, dict]:
    """ Main PDF handler function."""
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

    # If the "text" is markup-only, treat as empty to force OCR
    if _is_mostly_markup(all_text):
        logger.info({
            "level": "INFO",
            "type": "handler",
            "message": "[INFO] Detected markup-only PDF text — switching to OCR.",
            "session_id": session_id
        })
        all_text = ""

    # OCR fallback (adaptive, cross‑platform)
    has_text = bool((all_text or "").strip())
    need_ocr = (not has_text) and bool(pytesseract) and ENABLE_OCR

    # If forcing OCR but pytesseract is unavailable, log and skip the loop
    if (not pytesseract) and ENABLE_OCR_FORCE:
        logger.warning({
            "level": "WARNING",
            "type": "handler",
            "message": "[WARN] ENABLE_OCR_FORCE is set but Tesseract is unavailable; skipping OCR fallback.",
            "session_id": session_id
        })

    if need_ocr or (ENABLE_OCR_FORCE and pytesseract):
        if not has_text:
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
        if (not has_text) or (len(best_text) > len(all_text) * 1.25):
            all_text = best_text or all_text
            ocr_score = best_conf or 0.0
            ocr_runs = runs_summary or []
            metadata["ocr_confidence_avg"] = round(ocr_score, 2)
            metadata["ocr_runs"] = ocr_runs
            metadata["ocr_used"] = True
        else:
            metadata["ocr_used"] = False

    clean_text = _sanitize_extracted_text(all_text)
    if not clean_text and all_text:
        # If sanitization nuked everything (e.g., fully-tagged), keep minimal fallback
        clean_text = os.path.splitext(os.path.basename(pdf_path))[0]

    logger.debug({
        "level": "DEBUG",
        "type": "handler",
        "message": "[DEBUG] PDF extracted text preview (first 500 chars):" + (clean_text[:500] if isinstance(clean_text, str) else str(clean_text)[:500]),
        "session_id": session_id
    })

    table_hints = list(
        set(LOCATION_KEYWORDS) | set(CANDIDATE_KEYWORDS) | set(BALLOT_TYPES) |
        set(PARTY_KEYWORDS) | set(TOTAL_KEYWORDS) | set(MISC_FOOTER_KEYWORDS) | set(CONTEST_KEYWORDS)
    )
    # Use sanitized text from here on
    lines = clean_text.splitlines()
    headers, header_candidates = infer_headers_and_methods(lines, table_hints)

    # Detect potential contests from text as hints
    detected_titles = _detect_contest_titles_from_text(lines)
    if not detected_titles:
        detected_titles = [os.path.basename(pdf_path).replace(".pdf", "")]

    # Derive light context from filename for better selection (before prompting)
    fname = os.path.basename(pdf_path).lower()
    state = "Unknown"
    county = "Unknown"
    year = None
    for part in fname.replace(".pdf", "").split("_"):
        if "county" in part:
            county = part.replace("county", "").strip().title() + " County"
        if len(part) == 2 and part.isalpha():
            state = part.upper()
    m = re.search(r"(19|20)\d{2}", fname)
    if m:
        try:
            year = int(m.group(0))
        except Exception:
            year = None

    # Single contest fast-path or unified selector pass (no duplicate prompts)
    if len(detected_titles) == 1:
        selected_contest_title = detected_titles[0]
    else:
        selector_data = {
            "contests": [{"title": t} for t in detected_titles],
            "noisy_patterns": [s.lower() for s in (CONTEST_TITLE_SKIP_PHRASES or set())]
        }
        selected = select_contest(
            coordinator=coordinator,
            state=state, county=county, year=year,
            session_id=session_id,
            context={"selector_data": selector_data},
            allow_multiple=False,
            prompt_message="[PROMPT] Select contest (index, text, or 'cancel'): ",
            force_interactive=True,
            disable_ml_verify=False
        )
        if not selected:
            logger.warning({
                "level": "WARNING",
                "type": "handler",
                "message": "[WARN] No contest selected. Using filename as fallback.",
                "session_id": session_id
            })
            selected_contest_title = os.path.basename(pdf_path).replace(".pdf", "")
        else:
            selected_contest_title = (selected[0] or {}).get("title") or detected_titles[0]

    # Update metadata using derived context
    metadata.update({
        "source_file": os.path.basename(pdf_path),
        "state": state,
        "county": county,
        "handler": "pdf_handler",
        "contest": selected_contest_title
    })

    contest_column = None
    if headers:
        # Auto-detect a contest-like column; avoid interactive re-prompt
        contest_header_keywords = CONTEST_HEADER_KEYWORDS
        candidates = [h for h in headers if any(kw in h.lower() for kw in contest_header_keywords)]

        if len(candidates) == 1:
            contest_column = candidates[0]
            logger.info({
                "level": "INFO",
                "type": "handler",
                "message": f"[INFO] Auto-selected contest column: {contest_column}",
                "session_id": session_id
            })
        elif len(candidates) > 1:
            # Rank by preference order from constants
            pref = CONTEST_HEADER_PREFERENCE
            def rank(h):
                low = h.lower()
                for i, kw in enumerate(pref):
                    if kw in low:
                        return i
                return len(pref)
            candidates.sort(key=rank)
            if rank(candidates[0]) < rank(candidates[1]):
                contest_column = candidates[0]
                logger.info({
                    "level": "INFO",
                    "type": "handler",
                    "message": f"[INFO] Auto-selected contest column (ranked): {contest_column}",
                    "session_id": session_id
                })
            else:
                logger.info({
                    "level": "INFO",
                    "type": "handler",
                    "message": "[INFO] Multiple possible contest columns detected; skipping auto-selection to avoid extra prompts.",
                    "session_id": session_id
                })

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

        # If we have a contest column, filter to the selected contest
        contest = selected_contest_title
        if contest_column:
            def _norm_title(s: str) -> str:
                s = (s or "").lower().strip()
                s = re.sub(r'[\s\-_/]+', ' ', s)
                s = re.sub(r'[^a-z0-9 ]+', '', s)
                return re.sub(r'\s+', ' ', s).strip()

            def _tokens(s: str) -> set[str]:
                return set(re.findall(r'[a-z0-9]+', (s or "").lower()))

            norm_selected = _norm_title(contest)
            present_values = sorted({(r.get(contest_column, "") or "").strip() for r in data if r.get(contest_column)})
            norm_map = {v: _norm_title(v) for v in present_values}

            # 1) exact normalized match
            exact = [v for v, nv in norm_map.items() if nv == norm_selected]
            chosen_value = exact[0] if exact else None

            # 2) token-overlap fallback if no exact
            if not chosen_value:
                sel_tok = _tokens(contest)
                scored = []
                for v in present_values:
                    vt = _tokens(v)
                    inter = len(sel_tok & vt)
                    union = len(sel_tok | vt) or 1
                    jacc = inter / union
                    # small boost for prefix/substring matches
                    if norm_selected and norm_map[v].startswith(norm_selected):
                        jacc += 0.15
                    if norm_selected and norm_selected in norm_map[v]:
                        jacc += 0.10
                    scored.append((jacc, v))
                scored.sort(reverse=True)
                if scored and scored[0][0] >= 0.45:
                    chosen_value = scored[0][1]

            if chosen_value:
                data = [r for r in data if _norm_title(r.get(contest_column, "")) == _norm_title(chosen_value)]
                logger.info({
                    "level": "INFO",
                    "type": "handler",
                    "message": f"[INFO] Filtered rows to contest '{chosen_value}' via column '{contest_column}'.",
                    "session_id": session_id
                })
            else:
                logger.warning({
                    "level": "WARNING",
                    "type": "handler",
                    "message": f"[WARN] Selected contest '{contest}' not found in column '{contest_column}'. Skipping row filter.",
                    "session_id": session_id,
                    "present": present_values[:25]
                })

        if data:
            # Harmonize headers/data and build table
            headers, data = harmonize_headers_and_data(
                headers,
                data,
                context={
                    "contest": selected_contest_title,
                    "state": state,
                    "county": county,
                }
            )
            domain = os.path.basename(pdf_path)
            context = {
                "contest": selected_contest_title,
                "state": state,
                "county": county,
                "year": year,
                "session_id": session_id,
                "handler": "pdf_handler",
                "ocr_confidence_avg": metadata.get("ocr_confidence_avg"),
                "ocr_used": metadata.get("ocr_used"),
            }
            headers_final, data_final, _entity_info = build_table_noninteractive(
                domain=domain,
                headers=headers,
                data=data,
                coordinator=coordinator,
                context=context,
                pivot_to_wide=True,
                debug=False
            )

            result = finalize_election_output(
                headers=headers_final,
                data=data_final,
                coordinator=coordinator,
                contest=selected_contest_title,
                state=state,
                county=county,
                context={
                    "handler": "pdf_handler",
                    "input_file": os.path.basename(pdf_path),
                    "session_id": session_id,
                    "ocr_confidence_avg": metadata.get("ocr_confidence_avg"),
                    "ocr_used": metadata.get("ocr_used")
                },
                enable_user_feedback=False,
                session_id=session_id
            )

            metadata.update({
                "output_file": os.path.basename(result.get("csv_path", "")),
                "headers": headers_final,
                "row_count": len(data_final),
                "csv_path": result.get("csv_path"),
                "metadata_path": result.get("metadata_path"),
            })

            logger.info({
                "level": "INFO",
                "type": "output",
                "message": f"[OUTPUT] Wrote {len(data_final)} rows to: {result.get('csv_path')}",
                "session_id": session_id
            })
            logger.info({
                "level": "INFO",
                "type": "output",
                "message": f"[OUTPUT] Metadata written to: {result.get('metadata_path')}",
                "session_id": session_id
            })

            return headers_final, data_final, selected_contest_title, metadata

        else:
            unmatched_count = len(lines[header_line_idx + 1:])
            logger.warning({
                "level": "WARNING",
                "type": "output",
                "message": f"[WARN] No structured rows matched the inferred column count of {len(headers)}. Total lines scanned: {unmatched_count}",
                "session_id": session_id
            })
            fallback_rows = [{"raw_line": line} for line in lines[header_line_idx + 1:]]
            result = finalize_election_output(
                headers=["raw_line"],
                data=fallback_rows,
                coordinator=coordinator,
                contest=selected_contest_title,
                state=state,
                county=county,
                context={
                    "handler": "pdf_handler",
                    "input_file": os.path.basename(pdf_path),
                    "session_id": session_id,
                    "fallback": True
                },
                enable_user_feedback=False,
                session_id=session_id
            )
            metadata.update({
                "output_file": os.path.basename(result.get("csv_path", "")),
                "headers": ["raw_line"],
                "row_count": len(fallback_rows),
                "csv_path": result.get("csv_path"),
                "metadata_path": result.get("metadata_path"),
            })
            logger.warning({
                "level": "WARNING",
                "type": "output",
                "message": f"[OUTPUT] Wrote fallback rows to: {result.get('csv_path')}",
                "session_id": session_id
            })
            return ["raw_line"], fallback_rows, selected_contest_title, metadata

    # Plain text fallback
    result = finalize_election_output(
        headers=["text"],
        data=[{"text": clean_text}],
        coordinator=coordinator,
        contest=selected_contest_title,
        state=state,
        county=county,
        context={
            "handler": "pdf_handler",
            "input_file": os.path.basename(pdf_path),
            "session_id": session_id,
            "text_sanitized": True,
            "raw_text_len": len(all_text or ""),
            "clean_text_len": len(clean_text or "")
        },
        enable_user_feedback=False,
        session_id=session_id
    )
    metadata.update({
        "output_file": os.path.basename(result.get("csv_path", "")),
        "headers": ["text"],
        "row_count": 1,
        "text_sanitized": True,
        "raw_text_len": len(all_text or ""),
        "clean_text_len": len(clean_text or ""),
        "csv_path": result.get("csv_path"),
        "metadata_path": result.get("metadata_path")
    })
    logger.warning({
        "level": "WARNING",
        "type": "output",
        "message": f"[OUTPUT] Wrote plain text to: {result.get('csv_path')}",
        "session_id": session_id
    })
    return ["text"], [{"text": clean_text}], selected_contest_title, metadata

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

    result = parse_pdf_election_results(manual_file, session_id=session_id, coordinator=coordinator)

    # Defensive: always return a 4-tuple, never a bool
    if not (isinstance(result, tuple) and len(result) == 4):
        logger.error({
            "level": "ERROR",
            "type": "handler",
            "message": "[ERROR] Invalid result from parse_pdf_election_results (expected 4-tuple).",
            "session_id": session_id,
            "got_type": type(result).__name__
        })
        return None, None, None, {"error": "Invalid parse result"}
    return result