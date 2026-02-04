import os
import re
import tempfile
import time
from difflib import get_close_matches
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import requests

from ..config import ALLOW_GOOGLE_DOCS, DISABLE_HTML_FALLBACK, SUPPORTED_FORMATS, URL_MAX_REDIRECTS
from ..Context_Integration.Context_Library.constants import CONTEST_KEYWORDS
from ..handlers import fec_handler
from ..handlers.formats import csv_handler, json_handler, pdf_handler, txt_handler, xlsx_handler
from .browser_utils import (
    safe_click,
    safe_content,
    safe_context_library,
    safe_context_result,
    safe_get_attribute,
    safe_inner_text,
    safe_query_selector_all,
    safe_url,
    safe_wait_for_timeout,
)
from .download_utils import download_file, ensure_input_directory
from .html_scanner import append_pattern_kb, load_pattern_kb
from .logger_singleton import logger, prompt
from .shared_logic import safe_lower, safe_parse

FORMAT_KEYWORDS = [
    ("xlsx", {"xlsx", "excel", "spreadsheet"}),
    ("xls", {"xls", "excel"}),
    ("csv", {"csv", "comma", "delimited"}),
    ("json", {"json", "geojson", "api"}),
    ("pdf", {"pdf", "portable document", "report"}),
    ("txt", {"txt", "text", "plain"}),
]

CONTENT_TYPE_FORMAT_MAP = {
    "application/pdf": "pdf",
    "application/json": "json",
    "text/json": "json",
    "text/csv": "csv",
    "application/csv": "csv",
    "application/vnd.ms-excel": "xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/octet-stream": None,
    "text/plain": "txt",
}

FILENAME_FROM_DISPOSITION = re.compile(
    r"filename\*=UTF-8''(?P<utf8>[^;]+)|filename=\"?(?P<plain>[^\";]+)\"?",
    re.IGNORECASE,
)


def _normalize_text(text: Optional[str]) -> str:
    return (text or "").strip().lower()


def _infer_format_from_text(text: Optional[str]) -> Optional[str]:
    norm = _normalize_text(text)
    if not norm:
        return None
    for fmt, keywords in FORMAT_KEYWORDS:
        for kw in keywords:
            if kw in norm:
                return fmt
    return None


def _infer_format_from_attr_value(attr: str, value: str) -> Optional[str]:
    norm_val = _normalize_text(value)
    if not norm_val:
        return None
    if attr in {"data-format", "data-filetype", "data-extension", "data-export", "data-type", "data-value", "aria-label", "title"}:
        return _infer_format_from_text(norm_val)
    if attr == "class":
        return _infer_format_from_text(norm_val)
    return None


def _extract_candidate_urls(attr: str, raw_value: str) -> List[str]:
    if not raw_value:
        return []
    raw_value = raw_value.strip()
    if attr == "onclick":
        urls = []
        urls.extend(re.findall(r"https?://[^\s'\"<>]+", raw_value))
        urls.extend(re.findall(r"/[^\s'\"<>]+", raw_value))
        urls.extend(
            match
            for match in re.findall(r"['\"]([^'\"]+\.(?:json|csv|pdf|txt|xlsx|xls))['\"]", raw_value, re.IGNORECASE)
        )
        deduped = []
        for url in urls:
            if not url:
                continue
            lower = url.lower()
            if lower in {"javascript:void(0)", "#"}:
                continue
            if url not in deduped:
                deduped.append(url)
        return deduped
    if raw_value.lower().startswith("javascript"):
        return []
    return [raw_value]


def _clean_filename(name: str) -> str:
    name = unquote(name or "")
    name = name.strip()
    return name or "download"


def _guess_filename_from_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        path = parsed.path or ""
        filename = os.path.basename(path)
        if filename:
            return _clean_filename(filename)
        if parsed.query:
            for segment in parsed.query.split("&"):
                if "=" in segment:
                    _, val = segment.split("=", 1)
                    val = _clean_filename(val)
                    if any(val.lower().endswith(f".{ext}") for ext in ("csv", "json", "pdf", "txt", "xlsx", "xls")):
                        return val
        return _clean_filename(url.split("//")[-1].split("/")[-1])
    except Exception:
        return "download"


def _extract_filename_from_disposition(disposition: Optional[str]) -> Optional[str]:
    if not disposition:
        return None
    match = FILENAME_FROM_DISPOSITION.search(disposition)
    if not match:
        return None
    filename = match.group("utf8") or match.group("plain")
    return _clean_filename(filename)


def _extract_google_sheet_metadata(url: str) -> Optional[Dict[str, str]]:
    if not url:
        return None
    try:
        parsed = urlparse(url)
    except Exception:
        return None
    host = (parsed.hostname or "").lower()
    if host not in {"docs.google.com", "drive.google.com", "spreadsheets.google.com"}:
        return None

    spreadsheet_id = None
    gid = None
    path_parts = [p for p in (parsed.path or "").split("/") if p]
    if "spreadsheets" in path_parts and "d" in path_parts:
        try:
            d_idx = path_parts.index("d")
            spreadsheet_id = path_parts[d_idx + 1] if d_idx + 1 < len(path_parts) else None
        except Exception:
            spreadsheet_id = None
    if not spreadsheet_id:
        qs = parse_qs(parsed.query or "")
        spreadsheet_id = (qs.get("id") or [None])[0]
    if not spreadsheet_id:
        return None

    qs = parse_qs(parsed.query or "")
    gid = (qs.get("gid") or [None])[0]
    if not gid and parsed.fragment and "gid=" in parsed.fragment:
        frag_qs = parse_qs(parsed.fragment)
        gid = (frag_qs.get("gid") or [None])[0]
    gid = gid or "0"

    base = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export"
    export_csv = f"{base}?format=csv&gid={gid}"
    export_xlsx = f"{base}?format=xlsx&gid={gid}"
    return {
        "spreadsheet_id": spreadsheet_id,
        "gid": gid,
        "export_csv": export_csv,
        "export_xlsx": export_xlsx,
        "source_url": url,
    }


def _probe_remote_format(page, resolved_url: str, session_id: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    referer = getattr(page, "url", "") if page is not None else ""
    headers = {**_browser_headers(page, referer), **_cookies_header_from_page(page)}
    content_type = None
    disposition = None
    filename = None
    status = None
    try:
        response = None
        if page is not None and hasattr(page, "context") and hasattr(page.context, "request"):
            response = page.context.request.head(resolved_url, headers=headers, timeout=20_000)
            status = getattr(response, "status", None)
            if status and status >= 400:
                response = None
        if response is None:
            try:
                resp = requests.head(resolved_url, headers=headers, allow_redirects=True, timeout=15)
                status = getattr(resp, "status_code", None)
                if status and status >= 400:
                    resp = None
                if resp is not None and hasattr(resp, "history"):
                    history = resp.history or []
                    if len(history) > URL_MAX_REDIRECTS:
                        raise ValueError("Too many redirects during HEAD probe")
                response = resp
            except Exception:
                response = None
        if response is not None:
            headers_map = {}
            if hasattr(response, "headers"):
                headers_map = dict(response.headers)
            content_type = headers_map.get("content-type") or headers_map.get("Content-Type")
            disposition = headers_map.get("content-disposition") or headers_map.get("Content-Disposition")
            filename = _extract_filename_from_disposition(disposition)
    except Exception as exc:
        logger.debug({
            "level": "DEBUG",
            "type": "download",
            "message": f"[format_router] HEAD probe failed for {resolved_url}: {exc}",
            "session_id": session_id
        })
    fmt = None
    if content_type:
        content_type = content_type.split(";")[0].strip().lower()
        fmt = CONTENT_TYPE_FORMAT_MAP.get(content_type)
    if not fmt and disposition:
        fmt = _infer_format_from_text(disposition)
    if not fmt and filename:
        fmt = _infer_format_from_url(filename)
    return fmt, filename

def _browser_headers(page, referer: str) -> dict:
    try:
        ua = page.evaluate("() => navigator.userAgent") if page else None
    except Exception:
        ua = None
    origin = ""
    try:
        if referer:
            u = urlparse(referer)
            origin = f"{u.scheme}://{u.netloc}"
    except Exception:
        origin = ""
    return {
        "User-Agent": ua or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Referer": referer or "",
        "Origin": origin or "",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
def _build_download_url(base_url: str, href: str) -> str:
    """Resolve href against base_url safely."""
    try:
        return urljoin(base_url or "", href or "")
    except Exception:
        return (href or "")

def _cookies_header_from_page(page) -> dict:
    """Return {'Cookie': 'k=v; ...'} header synthesized from Playwright page.context cookies."""
    try:
        ctx = getattr(page, "context", None)
        if not ctx:
            return {}
        cookies = ctx.cookies() or []
        if not cookies:
            return {}
        cookie_str = "; ".join(f"{c.get('name','')}={c.get('value','')}" for c in cookies if c.get('name'))
        return {"Cookie": cookie_str}
    except Exception:
        return {}

def extract_contest_from_filename(filename: str) -> str:
    """
    Extracts contest/race/type from a filename using canonical keywords, regex, and fuzzy matching.
    Returns the best match or "Other".
    """
    if not filename:
        return "Other"
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

def _infer_format_from_url(url: str) -> Optional[str]:
    lowered = safe_lower(url or "")
    for ext in (".json", ".csv", ".pdf", ".txt", ".xlsx", ".xls"):
        if ext in lowered:
            return ext.strip(".")
    return None


def _expose_download_interfaces(page, session_id: Optional[str] = None) -> None:
    if page is None:
        return
    keywords = {"download", "export", "view", "export data", "save"}
    try:
        elements = safe_query_selector_all(page, "a, button") or []
    except Exception:
        elements = []
    seen: set[str] = set()
    for element in elements:
        try:
            href = (safe_get_attribute(element, "href", logger) or "").lower()
            if any(ext in href for ext in (".json", ".csv", ".pdf", ".txt", ".xlsx", ".xls")):
                continue
            text = (safe_inner_text(element, logger) or "").strip().lower()
            attrs = " ".join(
                filter(
                    None,
                    [
                        safe_get_attribute(element, "data-toggle", logger),
                        safe_get_attribute(element, "aria-haspopup", logger),
                        safe_get_attribute(element, "class", logger),
                    ],
                )
            ).lower()
            identifier = safe_get_attribute(element, "id", logger) or f"{text}:{attrs}"
            if identifier in seen:
                continue
            should_click = False
            if any(keyword in text for keyword in keywords):
                should_click = True
            if "dropdown" in attrs or safe_get_attribute(element, "data-target", logger):
                should_click = True
            if not should_click:
                continue
            if safe_click(element, logger):
                safe_wait_for_timeout(page, 250, logger)
                seen.add(identifier)
                logger.info({
                    "level": "INFO",
                    "type": "download",
                    "message": "[format_router] Expanded potential download menu.",
                    "session_id": session_id,
                    "identifier": identifier,
                })
        except Exception:
            continue


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
        if fmt in {"txt", "text"}:
            return txt_handler
        if fmt in {"xlsx", "xls"}:
            return xlsx_handler
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
    rejected_downloads: Optional[set] = None,
    session_id: Optional[str] = None,
    manual_upload_mode: bool = False,
    uploads_dir: Optional[str] = None,
    cancel_flag=None,
    **handler_kwargs,
) -> Tuple[Optional[tuple], bool]:
    """
    Extracts download links (from context library, DOM, and HTML), prompts user for format,
    downloads file, and routes to handler.
    Returns (result, handled) where handled=True if a format was selected and processed.
    """
    from .user_prompt import PromptCancelled
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
        try:
            files = [f for f in os.listdir(uploads_dir) if os.path.isfile(os.path.join(uploads_dir, f))]
        except Exception:
            files = []
        if not files:
            logger.error({
                "level": "ERROR",
                "type": "manual_override",
                "message": "[ManualOverride] No files found in uploads folder.",
                "session_id": session_id
            })
            return None, False

        # Prompt user to select file (index or filename)
        prompt_message = "[PROMPT] Select a file to parse from uploads (enter index or filename):"
        def validator(x):
            s = (str(x) or "").strip()
            if not s:
                return False
            if s.isdigit() and 0 <= int(s) < len(files):
                return True
            # allow exact or case-insensitive filename
            low = s.lower()
            return any(low == f.lower() for f in files)

        try:
            selection = prompt.prompt_input(
                prompt_message,
                validator=validator,
                session_id=session_id,
                context={
                    "title": "Select a File from Uploads",
                    "urls": files,  # Changed from "files" to "urls" to match frontend expectations
                    "options": files,  # Also provide as "options" for compatibility
                    "placeholder": "Enter index or filename"
                }
            )
            if not isinstance(selection, str):
                raise ValueError("Non-string selection")
            sel = selection.strip()
            if sel.isdigit():
                selected_index = int(sel)
            else:
                # resolve filename (case-insensitive)
                target = sel.lower()
                selected_index = next(i for i, f in enumerate(files) if f.lower() == target)
        except (PromptCancelled, ValueError, EOFError, KeyboardInterrupt, StopIteration):
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
        handler_kwargs.pop("cancel_flag", None)
        headers, rows, contest, metadata = safe_parse(
            handler,
            page=None,
            manual_file=full_path,
            source_url=target_url,
            logger=logger,
            session_id=session_id,
            cancel_flag=cancel_flag,
            **handler_kwargs,
        )
        if isinstance(metadata, dict) and metadata.get("error"):
            logger.error({
                "level": "ERROR",
                "type": "manual_override",
                "message": f"[ManualOverride] Handler error: {metadata.get('error')}",
                "session_id": session_id
            })
            return None, False
        return (headers, rows, contest, metadata), True

    # --- Not in manual upload mode: fallback to DOM/HTML logic ---
    logger.info({
        "level": "INFO",
        "type": "manual_override",
        "message": "[ManualOverride] Manual upload mode not engaged or failed. Falling back to DOM/HTML scan.",
        "session_id": session_id
    })

    _expose_download_interfaces(page, session_id=session_id)

    html = safe_content(page, session_id=session_id)
    page_url = safe_url(page)
    probe_cache: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    probe_budget = 6

    google_sheet_links: List[Dict[str, str]] = []
    google_sheet_meta = None
    if ALLOW_GOOGLE_DOCS:
        google_sheet_meta = _extract_google_sheet_metadata(target_url) or _extract_google_sheet_metadata(page_url)
    if google_sheet_meta:
        sheet_id = google_sheet_meta.get("spreadsheet_id", "sheet")
        gid = google_sheet_meta.get("gid", "0")
        google_sheet_links = [
            {
                "href": google_sheet_meta.get("export_csv"),
                "format": "csv",
                "source": "google_sheets",
                "label": "Google Sheets CSV Export",
                "filename": f"google_sheet_{sheet_id}_{gid}.csv",
            },
            {
                "href": google_sheet_meta.get("export_xlsx"),
                "format": "xlsx",
                "source": "google_sheets",
                "label": "Google Sheets XLSX Export",
                "filename": f"google_sheet_{sheet_id}_{gid}.xlsx",
            },
        ]
        logger.info({
            "level": "INFO",
            "type": "download",
            "message": "[format_router] Google Sheets export detected; offering CSV/XLSX export.",
            "session_id": session_id,
            "sheet_id": sheet_id,
            "gid": gid,
        })

    def probe_format_for_url(resolved_url: str) -> Tuple[Optional[str], Optional[str]]:
        nonlocal probe_budget
        if not resolved_url:
            return None, None
        if resolved_url in probe_cache:
            return probe_cache[resolved_url]
        if probe_budget <= 0:
            probe_cache[resolved_url] = (None, None)
            return probe_cache[resolved_url]
        probe_budget -= 1
        fmt_guess, remote_name = _probe_remote_format(page, resolved_url, session_id=session_id)
        probe_cache[resolved_url] = (fmt_guess, remote_name)
        return probe_cache[resolved_url]

    # 1) Context library links (if any)
    supported_links: List[Dict[str, str]] = []
    try:
        context_lib = safe_context_library(page, session_id=session_id) or {}
        download_links = context_lib.get("download_links", [])
        if isinstance(download_links, list):
            supported_links = [link for link in download_links if isinstance(link, dict)]
    except Exception:
        supported_links = []
    if google_sheet_links:
        supported_links.extend(google_sheet_links)
    # 2) DOM anchors for supported formats
    dom_links: List[Dict[str, str]] = []
    try:
        selectors = "a, button, [data-href], [data-url], [data-download], [data-file], [data-link], [data-src], [data-value]"
        anchors = safe_query_selector_all(page, selectors) or []
        base_url = page_url
        attribute_candidates = [
            "href",
            "data-href",
            "data-url",
            "data-download",
            "data-file",
            "data-link",
            "data-src",
            "data-value",
            "value",
            "formaction",
            "onclick",
        ]
        hint_attributes = [
            "aria-label",
            "title",
            "data-format",
            "data-filetype",
            "data-extension",
            "data-export",
            "data-type",
            "download",
            "class",
        ]
        for element in anchors:
            label_text = safe_inner_text(element, logger) or ""
            hint_values = {attr: safe_get_attribute(element, attr, logger) or "" for attr in hint_attributes}
            for attr in attribute_candidates:
                raw_value = safe_get_attribute(element, attr, logger) or ""
                if not raw_value:
                    continue
                label_fmt = _infer_format_from_text(label_text)
                for candidate in _extract_candidate_urls(attr, raw_value):
                    resolved = _build_download_url(base_url, candidate)
                    if not resolved or resolved in {"#", "javascript:void(0)", "about:blank"}:
                        continue
                    url_inferred_fmt = _infer_format_from_url(resolved)
                    fmt = url_inferred_fmt or _infer_format_from_attr_value(attr, raw_value)
                    fmt = fmt or label_fmt
                    if not fmt:
                        for hint_attr, hint_val in hint_values.items():
                            fmt = fmt or _infer_format_from_attr_value(hint_attr, hint_val)
                            if fmt:
                                break
                    remote_fmt = None
                    remote_filename = None
                    cached_probe = probe_cache.get(resolved)
                    if cached_probe:
                        remote_fmt, remote_filename = cached_probe
                    else:
                        remote_fmt, remote_filename = probe_format_for_url(resolved)
                        probe_cache[resolved] = (remote_fmt, remote_filename)
                    if remote_fmt:
                        fmt = remote_fmt
                    if not (url_inferred_fmt or remote_fmt) and fmt == label_fmt:
                        # Heuristic-only inference from visible text without remote confirmation; skip to allow HTML parsing
                        continue
                    if not fmt:
                        continue
                    dom_links.append({
                        "href": resolved,
                        "format": fmt,
                        "source": f"dom:{attr}",
                        "label": label_text,
                        "filename": remote_filename or _guess_filename_from_url(resolved),
                    })
    except Exception as e:
        logger.warning({
            "level": "WARNING",
            "type": "download",
            "message": f"[format_router] DOM scan failed: {e}",
            "session_id": session_id
        })

    # 3) HTML regex scan
    dynamic_links = extract_download_links_from_html(html, exts=[".json", ".csv", ".pdf", ".txt", ".xlsx", ".xls"])

    # 4) Merge/dedupe by (href, format)
    all_links: Dict[Tuple[str, str], Dict[str, str]] = {}
    for link in (supported_links + dom_links + dynamic_links):
        href = link.get("href")
        fmt = link.get("format")
        if not (href and fmt):
            continue
        key = (href, fmt)
        existing = all_links.get(key)
        if existing:
            if not existing.get("filename") and link.get("filename"):
                existing["filename"] = link.get("filename")
            continue
        all_links[key] = link
    merged_links = list(all_links.values())

    # 5) Remove rejected
    new_links = [
        link
        for link in merged_links
        if link.get("href") not in (rejected_downloads or set()) and link.get("format")
    ]
    if not new_links:
        logger.info({
            "level": "INFO",
            "type": "download",
            "message": "[format_router] No new downloadable links found.",
            "session_id": session_id
        })
        return None, False

    # 6) Update context metadata (best-effort)
    try:
        context_result = safe_context_result(page, session_id=session_id)
        if isinstance(context_result, dict):
            context_result.setdefault("metadata", {})["download_links"] = merged_links
            if google_sheet_meta:
                context_result["metadata"]["google_sheet"] = google_sheet_meta
            logger.debug({
                "level": "DEBUG",
                "type": "download",
                "message": f"[format_router][Session:{session_id}] Context metadata updated with download_links.",
                "session_id": session_id
            })
    except Exception:
        pass

    # 7) Add format patterns to KB (best-effort)
    try:
        existing_entries = load_pattern_kb() or []
        existing_ids = {
            entry.get("pattern_id")
            for entry in existing_entries
            if isinstance(entry, dict)
        }
        added = 0
        for link in merged_links:
            fmt = link.get("format", "")
            href = link.get("href", "")
            kb_entry = {
                "pattern_id": f"format_{fmt}_{os.path.basename(href)}",
                "label": "download_format",
                "format": fmt,
                "href": href,
                "source_url": getattr(page, "url", target_url),
                "timestamp": time.time(),
                "embedding": [],
                "session_id": session_id
            }
            if kb_entry["pattern_id"] in existing_ids:
                continue
            append_pattern_kb(kb_entry)
            added += 1
        logger.debug({
            "level": "DEBUG",
            "type": "download",
            "message": f"[format_router][Session:{session_id}] Pattern KB entries added: {added}",
            "session_id": session_id
        })
    except Exception:
        pass

    # 8) Build prompt with contest context
    context_options: List[Dict[str, str]] = []
    for link in new_links:
        fmt = str(link.get("format", "")).lower()
        url = str(link.get("href", ""))
        if not (fmt and url):
            continue
        filename = link.get("filename") or _guess_filename_from_url(url)
        contest = extract_contest_from_filename(filename or url)
        context_options.append({
            "format": fmt,
            "url": url,
            "contest": contest,
            "filename": filename,
        })

    if not context_options:
        logger.info({
            "level": "INFO",
            "type": "download",
            "message": "[format_router] Detected downloads lacked recognizable formats after probing.",
            "session_id": session_id
        })
        return None, False

    summary = summarize_downloads([(opt["format"], opt["url"]) for opt in context_options])
    logger.info({
        "level": "INFO",
        "type": "prompt",
        "message": f"Summary of detected downloads: {summary}",
        "session_id": session_id
    })

    format_options = [
        f"{opt['format'].upper()} ({opt['filename']}) [{opt['contest']}]"
        for opt in context_options
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
        s = str(x).strip().lower()
        valid_indices = [str(i) for i in range(len(format_options))]
        valid_options = [opt.lower() for opt in format_options]
        return (s in valid_indices or s in valid_options) if DISABLE_HTML_FALLBACK else (s in ("", "n") or s in valid_indices or s in valid_options)

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

    # Handle skip
    if selection is None or (not DISABLE_HTML_FALLBACK and str(selection).strip().lower() in ("", "n")):
        logger.info({
            "level": "INFO",
            "type": "prompt",
            "message": ("No selection made. Aborting as HTML fallback is disabled." if DISABLE_HTML_FALLBACK else "User chose to skip format download."),
            "session_id": session_id
        })
        return None, None

    # Resolve choice -> index
    try:
        sel = str(selection).strip()
        if sel.isdigit() and 0 <= int(sel) < len(context_options):
            idx = int(sel)
        else:
            idx = next(i for i, opt in enumerate(format_options) if opt.lower() == sel.lower())
        selected_option = context_options[idx]
        fmt = selected_option.get("format", "")
        file_url = selected_option.get("url", "")
        logger.info({
            "level": "INFO",
            "type": "prompt",
            "message": f"User selected format: {str(fmt).upper()}",
            "session_id": session_id
        })

        # Resolve URL relative to page.url
        page_url = getattr(page, "url", target_url) if page is not None else target_url
        resolved_url = _build_download_url(page_url, file_url)

        # Headers: cookies + browser-like
        cookie_hdr = _cookies_header_from_page(page)
        hdrs = {**_browser_headers(page, page_url), **cookie_hdr}

        local_file_path = None
        selected_filename = selected_option.get("filename") or ""

        # 1) Playwright request if available
        try:
            if page is not None and hasattr(page, "context") and hasattr(page.context, "request"):
                resp = page.context.request.get(resolved_url, headers=hdrs, timeout=60_000)
                if getattr(resp, "ok", False):
                    from ..config import INPUT_DIR
                    ensure_input_directory()
                    fname = selected_filename or os.path.basename(resolved_url) or f"download.{fmt}"
                    save_path = os.path.join(INPUT_DIR, fname)
                    with open(save_path, "wb") as f:
                        f.write(resp.body())
                    local_file_path = save_path
                else:
                    logger.error({
                        "level": "ERROR",
                        "type": "download",
                        "message": f"HTTP {getattr(resp, 'status', 'unknown')} via Playwright for {resolved_url}",
                        "session_id": session_id
                    })
        except Exception as e:
            logger.warning({
                "level": "WARNING",
                "type": "download",
                "message": f"Playwright request failed, will fallback to requests: {e}",
                "session_id": session_id
            })

        # 2) Fallback: our downloader
        if not local_file_path:
            try:
                local_file_path = download_file(
                    page_url,
                    resolved_url,
                    headers=hdrs,
                    check_hash=True,
                    filename_override=selected_filename or None,
                )
            except TypeError:
                # 3) Last resort: raw requests
                try:
                    r = requests.get(resolved_url, headers=hdrs, timeout=60, stream=True)
                    status = getattr(r, "status_code", None)
                    if status and status >= 400:
                        raise requests.HTTPError(f"HTTP {status}")
                    os.makedirs("downloads", exist_ok=True)
                    suffix = os.path.splitext(selected_filename or "")[1] or os.path.splitext(os.path.basename(resolved_url))[1] or f".{fmt}"
                    fd, tmp = tempfile.mkstemp(prefix="dl_", suffix=suffix, dir="downloads")
                    os.close(fd)
                    with open(tmp, "wb") as f:
                        for chunk in r.iterate_content(8192) if hasattr(r, "iterate_content") else r.iter_content(8192):
                            if chunk:
                                f.write(chunk)
                    local_file_path = tmp
                except Exception as e:
                    logger.error({
                        "level": "ERROR",
                        "type": "download",
                        "message": f"Requests fallback failed for {resolved_url}: {e}",
                        "session_id": session_id
                    })
                    local_file_path = None

        if not local_file_path or not os.path.exists(local_file_path):
            logger.error({
                "level": "ERROR",
                "type": "download",
                "message": f"Failed to download file: {resolved_url}",
                "session_id": session_id
            })
            return None, None

        # Dispatch to format handler with manual_file
        handler = None
        # Heuristic: prefer fec_handler for CSV/Excel that match FEC header patterns
        if fmt in ('csv', 'xlsx', 'xls') and local_file_path:
            try:
                ext = os.path.splitext(local_file_path)[1].lower().lstrip('.')
                if ext in ('xlsx', 'xls'):
                    # try to read headers via pandas if available
                    try:
                        import pandas as _pd
                        df = _pd.read_excel(local_file_path, sheet_name=0, nrows=0)
                        cols = [str(c).lower() for c in list(df.columns)]
                        hay = " ".join(cols)
                    except Exception:
                        hay = ""
                else:
                    with open(local_file_path, 'r', encoding='utf-8', errors='replace') as fh:
                        hay = fh.read(4096).lower()
                # look for distinctive FEC headers/tokens
                if any(tok in hay for tok in ('cand_id', 'cand_name', 'link_image', 'cand_party_affiliation')):
                    handler = fec_handler
            except Exception:
                handler = None
        if handler is None:
            handler = route_format_handler(fmt)
        if not handler:
            logger.error({
                "level": "ERROR",
                "type": "router",
                "message": f"No handler for format: {fmt}",
                "session_id": session_id
            })
            return None, None

        headers, rows, contest, metadata = safe_parse(
            handler,
            page=None,
            manual_file=local_file_path,
            source_url=file_url,
            logger=logger,
            session_id=session_id,
            cancel_flag=cancel_flag,
            **handler_kwargs,
        )
        if isinstance(metadata, dict) and metadata.get("error"):
            logger.error({
                "level": "ERROR",
                "type": "manual_override",
                "message": "[ManualOverride] Handler returned error.",
                "session_id": session_id
            })
            return None, False
        return (headers, rows, contest, metadata), True

    except Exception as e:
        logger.warning({
            "level": "WARNING",
            "type": "prompt",
            "message": f"Invalid selection. Skipping format download. ({e})",
            "session_id": session_id
        })
        return None, None
