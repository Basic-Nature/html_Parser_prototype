"""DOM Snapshot Mode for Medium-Trust URLs

Provides a safer alternative to full browser navigation by capturing
static HTML content without JavaScript execution, protecting against
XSS attacks and reducing SSRF risk for medium-trust election data sources.

Trust Score Range: 50-79 (medium-trust)
- Too risky for full JS-enabled browser navigation
- Too valuable to reject outright
- Solution: Capture static DOM, extract tables server-side

Author: Smart Elections Team
Date: February 2026
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

try:
    from selectolax.parser import HTMLParser
    HAS_SELECTOLAX = True
except ImportError:
    HAS_SELECTOLAX = False

from ..utils.logger_singleton import logger
from ..utils.telemetry import emit_telemetry_event


def capture_dom_snapshot(
    page,
    *,
    wait_for_selector: str | None = None,
    max_wait_ms: int = 5000,
    session_id: str | None = None
) -> str:
    """Capture static HTML content from a Playwright page without JS execution.
    
    This is safer than full browser navigation because:
    - JavaScript is disabled/not executed
    - No dynamic content loading
    - Reduced XSS attack surface
    - Faster extraction (no waiting for JS frameworks)
    
    Args:
        page: Playwright page object (already navigated)
        wait_for_selector: Optional CSS selector to wait for before snapshot
        max_wait_ms: Maximum wait time for selector (default 5s)
        session_id: Optional session ID for logging
    
    Returns:
        Raw HTML content as string
    
    Raises:
        Exception if page is None or snapshot fails
    """
    if page is None:
        raise ValueError("Page object is None")
    
    start_time = time.time()
    
    # Wait for specific selector if provided (e.g., table, tbody)
    if wait_for_selector:
        try:
            page.wait_for_selector(
                wait_for_selector,
                timeout=max_wait_ms,
                state="attached"
            )
            logger.debug({
                "level": "DEBUG",
                "type": "dom_snapshot",
                "message": f"[DOMSnapshot] Selector '{wait_for_selector}' found",
                "session_id": session_id
            })
        except Exception as exc:
            logger.warning({
                "level": "WARNING",
                "type": "dom_snapshot",
                "message": f"[DOMSnapshot] Selector '{wait_for_selector}' not found: {exc}",
                "session_id": session_id
            })
    
    # Capture full HTML content
    try:
        html_content = page.content()
    except Exception as exc:
        logger.error({
            "level": "ERROR",
            "type": "dom_snapshot",
            "message": f"[DOMSnapshot] Failed to capture HTML: {exc}",
            "session_id": session_id
        })
        raise
    
    duration_ms = int((time.time() - start_time) * 1000)
    content_size = len(html_content)
    
    logger.info({
        "level": "INFO",
        "type": "dom_snapshot",
        "message": f"[DOMSnapshot] Captured {content_size} bytes in {duration_ms}ms",
        "session_id": session_id,
        "content_size": content_size,
        "duration_ms": duration_ms
    })
    
    # Emit telemetry
    try:
        emit_telemetry_event("dom_snapshot_captured", {
            "session_id": session_id,
            "content_size": content_size,
            "duration_ms": duration_ms,
            "wait_selector": wait_for_selector
        })
    except Exception:
        pass
    
    return html_content


def extract_tables_from_snapshot(
    html_content: str,
    context: Dict[str, Any] | None = None,
    session_id: str | None = None
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Extract tabular data from static HTML snapshot.
    
    Uses fast HTML parsing (selectolax preferred) to find table elements
    and convert them to structured data without JavaScript execution.
    
    Args:
        html_content: Raw HTML content from snapshot
        context: Optional context dict with state/county/contest hints
        session_id: Optional session ID for logging
    
    Returns:
        Tuple of (headers, data_rows) where:
        - headers: List of column names
        - data_rows: List of dicts with column name -> value mapping
    
    Raises:
        Exception if HTML parsing fails
    """
    if not html_content:
        logger.warning({
            "level": "WARNING",
            "type": "dom_snapshot",
            "message": "[DOMSnapshot] Empty HTML content",
            "session_id": session_id
        })
        return [], []
    
    start_time = time.time()
    context = context or {}
    
    # Parse HTML with selectolax (fast) or fallback to stdlib
    if HAS_SELECTOLAX:
        try:
            parser = HTMLParser(html_content)
        except Exception as exc:
            logger.error({
                "level": "ERROR",
                "type": "dom_snapshot",
                "message": f"[DOMSnapshot] Selectolax parsing failed: {exc}",
                "session_id": session_id
            })
            raise
    else:
        # Fallback: Use dynamic_table_extractor which has its own HTML parsing
        logger.debug({
            "level": "DEBUG",
            "type": "dom_snapshot",
            "message": "[DOMSnapshot] Selectolax not available, using fallback parser",
            "session_id": session_id
        })
        # Import here to avoid circular dependency
        from ..Context_Integration.context_coordinator import ContextCoordinator
        from ..utils.dynamic_table_extractor import dynamic_table_extractor
        
        coordinator = ContextCoordinator()
        headers, rows = dynamic_table_extractor(None, context, coordinator, table_html=html_content)
        
        duration_ms = int((time.time() - start_time) * 1000)
        logger.info({
            "level": "INFO",
            "type": "dom_snapshot",
            "message": f"[DOMSnapshot] Extracted {len(rows)} rows in {duration_ms}ms (fallback parser)",
            "session_id": session_id,
            "row_count": len(rows),
            "column_count": len(headers),
            "duration_ms": duration_ms
        })
        
        return headers, rows
    
    # Selectolax parsing path
    tables = parser.css("table")
    if not tables:
        logger.warning({
            "level": "WARNING",
            "type": "dom_snapshot",
            "message": "[DOMSnapshot] No tables found in HTML",
            "session_id": session_id
        })
        return [], []
    
    logger.debug({
        "level": "DEBUG",
        "type": "dom_snapshot",
        "message": f"[DOMSnapshot] Found {len(tables)} table(s)",
        "session_id": session_id,
        "table_count": len(tables)
    })
    
    # Extract from largest table (most likely to contain election results)
    largest_table = max(tables, key=lambda t: len(t.css("tr")))
    
    # Extract headers from first row (th or first tr)
    headers = []
    header_row = largest_table.css_first("thead tr") or largest_table.css_first("tr")
    if header_row:
        for cell in header_row.css("th, td"):
            text = cell.text(strip=True)
            if text:
                headers.append(text)
    
    if not headers:
        # Fallback: Generate generic column names
        first_row = largest_table.css_first("tbody tr") or largest_table.css("tr")[1] if len(largest_table.css("tr")) > 1 else None
        if first_row:
            cell_count = len(first_row.css("td"))
            headers = [f"Column{i+1}" for i in range(cell_count)]
    
    # Extract data rows (skip header row)
    data_rows = []
    body_rows = largest_table.css("tbody tr") if largest_table.css_first("tbody") else largest_table.css("tr")[1:]
    
    for row in body_rows:
        cells = row.css("td")
        if not cells:
            continue
        
        row_data = {}
        for idx, cell in enumerate(cells):
            text = cell.text(strip=True)
            col_name = headers[idx] if idx < len(headers) else f"Column{idx+1}"
            row_data[col_name] = text
        
        if row_data:
            data_rows.append(row_data)
    
    duration_ms = int((time.time() - start_time) * 1000)
    
    logger.info({
        "level": "INFO",
        "type": "dom_snapshot",
        "message": f"[DOMSnapshot] Extracted {len(data_rows)} rows in {duration_ms}ms",
        "session_id": session_id,
        "row_count": len(data_rows),
        "column_count": len(headers),
        "duration_ms": duration_ms
    })
    
    # Emit telemetry
    try:
        emit_telemetry_event("snapshot_tables_extracted", {
            "session_id": session_id,
            "row_count": len(data_rows),
            "column_count": len(headers),
            "duration_ms": duration_ms,
            "table_count": len(tables),
            "parser": "selectolax" if HAS_SELECTOLAX else "fallback"
        })
    except Exception:
        pass
    
    return headers, data_rows


def snapshot_mode_pipeline(
    page,
    context: Dict[str, Any] | None = None,
    session_id: str | None = None
) -> Tuple[List[str], List[Dict[str, Any]], str, Dict[str, Any]]:
    """Complete DOM snapshot extraction pipeline for medium-trust URLs.
    
    This replaces the full browser navigation + JS execution path with:
    1. Wait for table element to appear
    2. Capture static HTML snapshot
    3. Extract tables from snapshot
    4. Return structured data
    
    Args:
        page: Playwright page object (already navigated)
        context: Optional context dict with state/county/contest hints
        session_id: Optional session ID for logging
    
    Returns:
        Tuple of (headers, data_rows, contest, metadata) matching parser contract:
        - headers: List of column names
        - data_rows: List of dicts with column name -> value mapping
        - contest: Contest label/name (from context or "DOM Snapshot Extraction")
        - metadata: Dict with output paths, handler info, snapshot metrics
    """
    context = context or {}
    
    logger.info({
        "level": "INFO",
        "type": "dom_snapshot",
        "message": "[DOMSnapshot] Starting snapshot mode pipeline",
        "session_id": session_id,
        "url": context.get("url")
    })
    
    # Step 1: Capture DOM snapshot
    try:
        html_content = capture_dom_snapshot(
            page,
            wait_for_selector="table",
            max_wait_ms=5000,
            session_id=session_id
        )
    except Exception as exc:
        logger.error({
            "level": "ERROR",
            "type": "dom_snapshot",
            "message": f"[DOMSnapshot] Snapshot capture failed: {exc}",
            "session_id": session_id
        })
        return [], [], "DOM Snapshot Failed", {
            "error": str(exc),
            "handler": "dom_snapshot",
            "snapshot_mode": True,
            "session_id": session_id
        }
    
    # Step 2: Extract tables from snapshot
    try:
        headers, data_rows = extract_tables_from_snapshot(
            html_content,
            context,
            session_id
        )
    except Exception as exc:
        logger.error({
            "level": "ERROR",
            "type": "dom_snapshot",
            "message": f"[DOMSnapshot] Table extraction failed: {exc}",
            "session_id": session_id
        })
        return [], [], "DOM Snapshot Extraction Failed", {
            "error": str(exc),
            "handler": "dom_snapshot",
            "snapshot_mode": True,
            "session_id": session_id
        }
    
    # Step 3: Build contest label and metadata
    contest = context.get("contest") or context.get("state") or "DOM Snapshot Extraction"
    
    metadata = {
        "handler": "dom_snapshot",
        "snapshot_mode": True,
        "session_id": session_id,
        "url": context.get("url"),
        "state": context.get("state"),
        "county": context.get("county"),
        "row_count": len(data_rows),
        "column_count": len(headers),
        "content_size": len(html_content),
        "trust_score": context.get("trust_score"),
        "trust_factors": context.get("trust_factors"),
    }
    
    logger.info({
        "level": "INFO",
        "type": "dom_snapshot",
        "message": f"[DOMSnapshot] Pipeline complete: {len(data_rows)} rows, {len(headers)} columns",
        "session_id": session_id,
        "row_count": len(data_rows),
        "column_count": len(headers)
    })
    
    return headers, data_rows, contest, metadata
