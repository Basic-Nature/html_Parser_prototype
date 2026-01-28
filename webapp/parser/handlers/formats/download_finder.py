from __future__ import annotations

from typing import List, Any, Optional
from urllib.parse import urljoin

from ...utils.logger_singleton import logger


def find_download_links(page: Any, base_url: str | None = None, session_id: str | None = None) -> List[str]:
    """
    Heuristic finder for obvious CSV/JSON download links on a page.
    Returns absolute URLs where possible.
    """
    urls: List[str] = []
    try:
        if page is None:
            return []
        # Look for anchors ending with .csv/.json
        try:
            anchors = page.query_selector_all('a[href$=".csv"], a[href$=".json"]')
        except Exception:
            # Fallback to generic selector via evaluate
            anchors = None
        if anchors:
            for a in anchors:
                try:
                    href = a.get_attribute('href')
                    if href:
                        href = href.strip()
                        if base_url and not href.startswith(('http:', 'https:')):
                            href = urljoin(base_url, href)
                        urls.append(href)
                except Exception:
                    continue
        else:
            # Evaluate fallback for environments where query_selector_all isn't present
            try:
                found = page.evaluate(r"() => Array.from(document.querySelectorAll('a')).map(a=>a.href).filter(h=>h.match(/\.(csv|json)$/i))")
                if isinstance(found, list):
                    urls.extend(found)
            except Exception:
                pass
    except Exception as e:
        try:
            logger.warning({"level":"WARNING","type":"download_finder","message":f"Download finder failed: {e}","session_id":session_id})
        except Exception:
            pass
    return urls


def attempt_download_and_parse(
    page: Any,
    coordinator: Any = None,
    context: Optional[dict] = None,
    session_id: Optional[str] = None,
    logger: Any = logger,
    non_interactive: bool = True,
    cancel_flag=None,
    output_bypass: bool = False,
) -> Optional[tuple]:
    """
    Conservative non-interactive helper: detect an obvious downloadable export
    (CSV/JSON/XLSX/PDF/etc), download it to the input folder, and dispatch to
    the matching format handler using the repository's existing format router.

    Returns a handler result tuple (headers, data, contest, metadata) on
    success, otherwise returns None. This helper performs local imports to
    avoid circular import issues at module import time.
    """
    try:
        # Local imports to avoid circular/early import problems
        from ...utils.format_router import detect_format_from_links, route_format_handler
        from ...utils.download_utils import download_file, ensure_input_directory
        from ...utils.shared_logic import safe_parse
    except Exception as exc:
        try:
            logger.debug({"level": "DEBUG", "type": "download_finder", "message": f"Unable to import download helpers: {exc}", "session_id": session_id})
        except Exception:
            pass
        return None

    if page is None:
        return None

    page_url = getattr(page, "url", None)
    candidates = []
    try:
        # detect_format_from_links should return [(fmt, url), ...] or similar
        candidates = detect_format_from_links(page, base_url=page_url, auto_confirm=False) or []
    except Exception:
        candidates = []

    if not candidates:
        return None

    fmt, file_url = candidates[0] if isinstance(candidates[0], (list, tuple)) else (None, None)
    if not fmt or not file_url:
        return None

    handler = route_format_handler(fmt)
    if not handler:
        try:
            logger.info({"level": "INFO", "type": "download_finder", "message": f"No handler for format {fmt}", "session_id": session_id})
        except Exception:
            pass
        return None

    # Attempt download
    try:
        ensure_input_directory()
        local_path = download_file(page_url or "", file_url, headers=None, context_info=None, check_hash=True)
    except Exception as e:
        try:
            logger.warning({"level": "WARNING", "type": "download_finder", "message": f"Download failed for {file_url}: {e}", "session_id": session_id})
        except Exception:
            pass
        local_path = None

    if not local_path:
        return None

    # Dispatch to the handler using safe_parse (signature-aware)
    try:
        result = safe_parse(
            handler,
            None,
            coordinator,
            manual_file=local_path,
            source_url=file_url,
            logger=logger,
            session_id=session_id,
            cancel_flag=cancel_flag,
        )
        if isinstance(result, tuple) and len(result) == 4:
            return result
    except Exception:
        try:
            logger.warning({"level": "WARNING", "type": "download_finder", "message": f"Handler failed for downloaded file: {file_url}", "session_id": session_id})
        except Exception:
            pass
    return None
