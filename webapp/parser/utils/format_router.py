import os
import time
import re
from typing import Optional, Tuple
from difflib import get_close_matches
from ..handlers.formats import json_handler, pdf_handler, csv_handler
from .logger_singleton import logger, prompt
from .shared_logic import (
    safe_lower, safe_get, safe_isdigit, safe_parse
)
from .browser_utils import (
    safe_content, safe_query_selector_all, safe_context_library, safe_context_result,
    safe_get_attribute, safe_url
)
from urllib.parse import urljoin
from ..config import SUPPORTED_FORMATS, DISABLE_HTML_FALLBACK
from .download_utils import download_file
from .html_scanner import load_pattern_kb, append_pattern_kb
from ..Context_Integration.Context_Library.constants import CONTEST_KEYWORDS

def extract_contest_from_filename(filename: str) -> str:
    """
    Extracts contest/race/type from a filename using canonical keywords, regex, and fuzzy matching.
    Returns the best match or "Other".
    """
    name = filename.lower().replace("_", " ").replace("-", " ")
    # 1. Exact/substring match (prefer longest keyword)
    best_kw = ""
    for kw in CONTEST_KEYWORDS:
        if kw.lower() in name and len(kw) > len(best_kw):
            best_kw = kw
    if best_kw:
        return best_kw.title()
    # 2. Regex: match any keyword as a whole word (prefer longest)
    pattern = r"\b(" + "|".join(sorted((re.escape(kw) for kw in CONTEST_KEYWORDS), key=len, reverse=True)) + r")\b"
    m = re.search(pattern, name)
    if m:
        return m.group(1).title()
    # 3. Fuzzy match: allow for typos, abbreviations, etc.
    words = set(name.split())
    candidates = []
    for word in words:
        matches = get_close_matches(word, CONTEST_KEYWORDS, n=1, cutoff=0.85)
        if matches:
            candidates.append(matches[0])
    if candidates:
        # Prefer the longest candidate (most specific)
        return max(candidates, key=len).title()
    # 4. Fallback: common contest/race patterns
    m2 = re.search(
        r"(mayor|council(?: member|man|woman|men|women)?|president|comptroller|public advocate|district attorney|borough(?: president)?|senator|judge|delegate|leader|committee|recap|edlevel)",
        name
    )
    if m2:
        return m2.group(1).title()
    return "Other"

def summarize_downloads(options):
    """
    Summarizes the available downloads by format.
    """
    summary = {}
    for fmt, url in options:
        ext = fmt.upper()
        summary[ext] = summary.get(ext, 0) + 1
    return ", ".join(f"{k}: {v}" for k, v in summary.items())

def detect_format_from_links(page, base_url=None, auto_confirm=False) -> list[tuple[str, str]]:
    """
    Scans a webpage for file links matching supported extensions.
    Returns a flat list in discovery order: [("json", url1), ("csv", url2), ...]
    """
    links = safe_query_selector_all(page, "a")
    found = {ext: [] for ext in SUPPORTED_FORMATS}
    logger.info({
        "level": "INFO",
        "type": "status",
        "message": "[INFO] Scanning for available download links...",
    })
    for link in links:
        try:
            href = safe_get_attribute(link, "href", logger) or ""
            for ext in SUPPORTED_FORMATS:
                if safe_lower(ext) in safe_lower(href):
                    abs_url = urljoin(base_url or safe_url(page), href)
                    found[ext].append(abs_url)
                    logger.debug({
                        "level": "DEBUG",
                        "type": "download",
                        "message": f"[DEBUG] Found {ext} link: {abs_url}",
                    })
        except Exception as e:
            logger.debug({
                "level": "DEBUG",
                "type": "download",
                "message": f"[DEBUG] Failed to evaluate a link: {e}",
            })

    flat_results = []
    for ext in SUPPORTED_FORMATS:
        for url in found[ext]:
            flat_results.append((safe_lower(ext).strip("."), url))
    if not flat_results:
        logger.warning({
            "level": "WARNING",
            "type": "download",
            "message": "[WARN] No supported file formats found on the page.",
        })
    # Auto-confirm logic: return only the first found format if enabled
    if auto_confirm and flat_results:
        logger.info({
            "level": "INFO",
            "type": "download",
            "message": f"[INFO] Auto-confirm enabled. Automatically selecting: {flat_results[0]}",
        })
        return [flat_results[0]]
    return flat_results

def route_format_handler(format_str: str) -> Optional[object]:
    fmt = format_str.lower().strip().lstrip('.')
    try:
        if fmt == "json":
            return json_handler
        if fmt == "pdf":
            return pdf_handler
        if fmt == "csv":
            return csv_handler
        logger.warning({
            "level": "WARNING",
            "type": "router",
            "message": f"[WARN] Unsupported format requested: {format_str}",
        })
        return None
    except ImportError as e:
        logger.warning({
            "level": "WARNING",
            "type": "router",
            "message": f"[Router] Failed to load handler for format {format_str}: {e}",
        })
        return None

def extract_download_links_from_html(html, exts=None) -> list[dict]:
    """
    Extract download links from raw HTML using regex for common file extensions.
    Returns a list of dicts: {"href": ..., "format": ..., "source": "html"}
    """
    if exts is None:
        exts = SUPPORTED_FORMATS
    pattern = re.compile(r'href=[\'"]([^\'"]+\.(?:' + '|'.join(ext[1:] for ext in exts) + r'))[\'"]', re.IGNORECASE)
    matches = pattern.findall(html)
    links = []
    for href in matches:
        for ext in exts:
            if safe_lower(href).endswith(safe_lower(ext)):
                links.append({
                    "href": href,
                    "format": safe_lower(ext).strip("."),
                    "source": "html"
                })
    return links

def prompt_and_handle_download(
    page,
    target_url,
    rejected_downloads=None,
    session_id=None,
    manual_upload_mode=False,
    uploads_dir=None  
) -> Tuple[Optional[dict], bool]:
    """
    Extracts download links (from context library, DOM, and HTML), prompts user for format,
    downloads file, and routes to handler.
    Returns (result, handled) where handled=True if a format was selected and processed.
    """
    if rejected_downloads is None:
        rejected_downloads = set()

    # --- Manual Upload Mode ---
    if manual_upload_mode and uploads_dir:
        logger.info({
            "level": "INFO",
            "type": "manual_override",
            "message": f"[ManualOverride] Entering manual upload mode with uploads_dir={uploads_dir}",
            "session_id": session_id
        })
        files = [f for f in os.listdir(uploads_dir) if os.path.isfile(os.path.join(uploads_dir, f))]
        if not files:
            logger.error({
                "level": "ERROR",
                "type": "manual_override",
                "message": "[ManualOverride] No files found in uploads folder.",
                "session_id": session_id
            })
            return None, False

        # Prompt user to select file
        prompt_message = "[PROMPT] Select a file to parse from uploads:"
        def validator(x):
            return safe_isdigit(x) and 0 <= int(x) < len(files)
        try:
            selection = prompt.prompt_input(
                prompt_message,
                validator=validator,
                session_id=session_id,
                context={"files": files}
            )
            if not isinstance(selection, str):
                raise ValueError("Non-string selection")
            selected_index = int(selection.strip())
        except (ValueError, EOFError, KeyboardInterrupt):
            logger.error({
                "level": "ERROR",
                "type": "manual_override",
                "message": "[ManualOverride] Invalid selection. Aborting manual parse.",
                "session_id": session_id
            })
            return None, False

        target_file = files[selected_index]
        full_path = os.path.join(uploads_dir, target_file)
        fmt = os.path.splitext(target_file)[1].lower().lstrip('.')
        handler = route_format_handler(fmt)
        if not handler:
            logger.error({
                "level": "ERROR",
                "type": "manual_override",
                "message": f"[ManualOverride] No handler for format: {fmt}",
                "session_id": session_id
            })
            return None, False

        logger.info({
            "level": "INFO",
            "type": "manual_override",
            "message": f"[ManualOverride] Parsing file: {target_file}",
            "session_id": session_id,
            "file_path": full_path
        })
        result = safe_parse(
            handler,
            page=None,
            manual_file=full_path,
            source_url=target_url,
            logger=logger,
            session_id=session_id
        )
        valid = isinstance(result, tuple) and len(result) == 4
        if not valid:
            logger.error({
                "level": "ERROR",
                "type": "manual_override",
                "message": "[ManualOverride] Invalid result format.",
                "session_id": session_id
            })
            return None, False
        return result, True

    # --- Not in manual upload mode: fallback to DOM/HTML logic ---
    logger.info({
        "level": "INFO",
        "type": "manual_override",
        "message": "[ManualOverride] Manual upload mode not engaged or failed. Falling back to DOM/HTML scan.",
        "session_id": session_id
    })

    html = safe_content(page, session_id=session_id)

    # 1. Extract links from context library
    supported_links = []
    context_lib = safe_context_library(page, session_id=session_id)
    download_links = context_lib.get("download_links", [])
    if isinstance(download_links, list):
        supported_links = [link for link in download_links if isinstance(link, dict)]

    # 2. Extract links from DOM (anchor tags) for supported formats
    dom_links = []
    try:
        anchors = safe_query_selector_all(page, "a")
        for a in anchors:
            href = safe_get_attribute(a, "href", logger)
            if not href:
                continue
            for ext in [".json", ".csv", ".pdf"]:
                if safe_lower(ext) in safe_lower(href):
                    dom_links.append({
                        "href": href,
                        "format": safe_lower(ext).strip("."),
                        "source": "dom"
                    })
    except Exception as e:
        logger.warning({
            "level": "WARNING",
            "type": "download",
            "message": f"[format_router] DOM scan failed: {e}",
            "session_id": session_id
        })

    # 3. Extract links dynamically from HTML (regex or pattern-based)
    dynamic_links = extract_download_links_from_html(html, exts=[".json", ".csv", ".pdf"])

    # 4. Merge and deduplicate all links by (href, format)
    all_links = {}
    for link in supported_links + dom_links + dynamic_links:
        href = safe_get(link, "href", None)
        fmt = safe_get(link, "format", None)
        if href and fmt:
            all_links[(href, fmt)] = link
    merged_links = list(all_links.values())

    # 5. Remove rejected
    new_links = [
        link for link in merged_links
        if isinstance(link, dict) and safe_get(link, "href") not in rejected_downloads
    ]
    if not new_links:
        logger.info({
            "level": "INFO",
            "type": "download",
            "message": "[format_router] No new downloadable links found.",
            "session_id": session_id
        })
        return None, False

    # 6. Update context metadata with discovered links (for downstream use, analytics, or UI)
    context_result = safe_context_result(page, session_id=session_id)
    if isinstance(context_result, dict):
        context_result.setdefault("metadata", {})["download_links"] = merged_links
        logger.debug({
            "level": "DEBUG",
            "type": "download",
            "message": f"[format_router][Session:{session_id}] Context metadata updated with download_links.",
            "session_id": session_id
        })

    # 7. Add to pattern KB for ML-driven format clustering
    format_kb = load_pattern_kb(session_id=session_id) if 'session_id' in load_pattern_kb.__code__.co_varnames else load_pattern_kb()
    kb_entries = []
    for link in merged_links:
        fmt = safe_get(link, "format", "")
        kb_entry = {
            "pattern_id": f"format_{fmt}_{os.path.basename(safe_get(link, 'href', ''))}",
            "label": "download_format",
            "format": fmt,
            "href": safe_get(link, "href", ""),
            "source_url": getattr(page, "url", target_url),
            "timestamp": time.time(),
            "embedding": [],
            "session_id": session_id
        }
        kb_entries.append(kb_entry)
        append_pattern_kb(kb_entry)
    logger.debug({
        "level": "DEBUG",
        "type": "download",
        "message": f"[format_router][Session:{session_id}] Pattern KB entries added: {len(kb_entries)}",
        "session_id": session_id
    })
    logger.debug({
        "level": "DEBUG",
        "type": "download",
        "message": f"[format_router][Session:{session_id}] KB snapshot: {format_kb}",
        "session_id": session_id
    })

    # 8. Prompt user for format (with contest context)
    confirmed = [
        (safe_lower(safe_get(link, "format", "")), safe_get(link, "href", ""))
        for link in new_links if isinstance(link, dict)
    ]
    # Attach contest context for each file
    context_options = []
    for fmt, url in confirmed:
        fname = os.path.basename(url)
        contest = extract_contest_from_filename(fname)
        context_options.append((fmt, url, contest, fname))

    summary = summarize_downloads([(fmt, url) for fmt, url, _, _ in context_options])
    logger.info({
        "level": "INFO",
        "type": "prompt",
        "message": f"Summary of detected downloads: {summary}",
        "session_id": session_id
    })

    # Build prompt message with contest context
    format_options = [
        f"{str(fmt).upper()} ({fname}) [{contest}]"
        for fmt, url, contest, fname in context_options
    ]
    options_lines = [f"  [{i}] {opt}" for i, opt in enumerate(format_options)]
    if not DISABLE_HTML_FALLBACK:
        options_lines.append("  [n or Enter] Skip download")
    options_str = "\n".join(options_lines)

    prompt_message = (
        f"[PROMPT][Session:{session_id}] Select a format to download:\n"
        f"{options_str}\n"
        f"Enter the number of your choice (0-{len(format_options)-1})"
        + ("" if DISABLE_HTML_FALLBACK else " or 'n' to skip")
        + " (type 'cancel' to abort): "
    )

    def validator(x) -> bool:
        x = str(x).strip().lower()
        valid_indices = [str(i) for i in range(len(format_options))]
        valid_options = [opt.lower() for opt in format_options]
        if DISABLE_HTML_FALLBACK:
            return x in valid_indices or x in valid_options
        else:
            return x in ("", "n") or x in valid_indices or x in valid_options

    try:
        selection = prompt.prompt_input(
            prompt_message,
            default=None if DISABLE_HTML_FALLBACK else "n",
            validator=validator,
            session_id=session_id,
            context={"options": format_options, "confirmed": context_options, "summary": summary}
        )
    except Exception as e:
        logger.error({
            "level": "ERROR",
            "type": "prompt",
            "message": f"Exception during prompt: {e}",
            "session_id": session_id
        })
        return None, None

    # Handle skip/none
    if selection is None or (not DISABLE_HTML_FALLBACK and str(selection).strip().lower() in ("", "n")):
        msg = (
            "No selection made. Aborting as HTML fallback is disabled."
            if DISABLE_HTML_FALLBACK else
            "User chose to skip format download."
        )
        logger.info({
            "level": "INFO",
            "type": "prompt",
            "message": msg,
            "session_id": session_id
        })
        return None, None

    # Resolve selection
    try:
        sel = str(selection).strip()
        idx = None
        if sel.isdigit() and 0 <= int(sel) < len(context_options):
            idx = int(sel)
        else:
            idx = next(
                i for i, opt in enumerate(format_options)
                if opt.lower() == sel.lower()
            )
        fmt, file_url, _, _ = context_options[idx]
        logger.info({
            "level": "INFO",
            "type": "prompt",
            "message": f"User selected format: {str(fmt).upper()}",
            "session_id": session_id
        })
        # --- Actually download the file here ---
        # Use the page's URL as the base for relative links
        page_url = getattr(page, "url", target_url) if page is not None else target_url
        local_file_path = download_file(page_url, file_url)
        if not local_file_path or not os.path.exists(local_file_path):
            logger.error({
                "level": "ERROR",
                "type": "download",
                "message": f"Failed to download file: {file_url}",
                "session_id": session_id
            })
            return None, None
        # Route to handler as before, but now with local_file_path
        handler = route_format_handler(fmt)
        if not handler:
            logger.error({
                "level": "ERROR",
                "type": "router",
                "message": f"No handler for format: {fmt}",
                "session_id": session_id
            })
            return None, None
        result = safe_parse(
            handler,
            page=None,
            manual_file=local_file_path,
            source_url=file_url,
            logger=logger,
            session_id=session_id
        )
        valid = isinstance(result, tuple) and len(result) == 4
        if not valid:
            logger.error({
                "level": "ERROR",
                "type": "manual_override",
                "message": "[ManualOverride] Invalid result format.",
                "session_id": session_id
            })
            return None, False
        return result, True
    except Exception as e:
        logger.warning({
            "level": "WARNING",
            "type": "prompt",
            "message": f"Invalid selection. Skipping format download. ({e})",
            "session_id": session_id
        })
        return None, None