from __future__ import annotations

from typing import List, Any
from urllib.parse import urljoin

from ...utils.logger_singleton import logger

## (A) Run the signature-normalization plan (scan handlers and auto-fix non-canonical signatures), then re-run the smoke test and full pytest;
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
                found = page.evaluate("() => Array.from(document.querySelectorAll('a')).map(a=>a.href).filter(h=>h.match(/\.(csv|json)$/i))")
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
