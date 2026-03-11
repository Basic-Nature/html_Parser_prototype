from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright


def _safe_slug(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in (value or "").strip().lower())
    return cleaned.strip("_") or "preview"


def capture_url_glimpse(
    url: str,
    *,
    out_dir: Path,
    timeout_ms: int = 45_000,
    wait_ms: int = 1_800,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    host = _safe_slug(urlparse(url).netloc or "local")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"glimpse_{host}_{ts}"

    screenshot_path = out_dir / f"{stem}.png"
    html_path = out_dir / f"{stem}.html"
    json_path = out_dir / f"{stem}.json"

    result: dict = {
        "url": url,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "screenshot": str(screenshot_path),
        "html_snapshot": str(html_path),
        "json_report": str(json_path),
    }

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()
        console_msgs = []

        def on_console(msg):
            try:
                console_msgs.append({"type": msg.type, "text": msg.text})
            except Exception:
                pass

        page.on("console", on_console)

        try:
            response = page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
            if response is not None:
                result["status"] = response.status
                try:
                    result["content_type"] = response.headers.get("content-type")
                except Exception:
                    result["content_type"] = None
            page.wait_for_timeout(wait_ms)
        except Exception as exc:
            result["error"] = f"goto_failed: {exc}"

        try:
            metrics = page.evaluate(
                """
                () => {
                  const tableCount = document.querySelectorAll('table, [role="table"]').length;
                  const headingCount = document.querySelectorAll('h1,h2,h3,h4,h5,h6').length;
                  const rowsEstimate = Array.from(document.querySelectorAll('table')).reduce((acc, t) => acc + t.querySelectorAll('tr').length, 0);
                  const hasElectionTerms = /election|precinct|county|contest|candidate|votes?/i.test(document.body?.innerText || '');
                  return {
                    table_count: tableCount,
                    heading_count: headingCount,
                    table_rows_estimate: rowsEstimate,
                    has_election_terms: hasElectionTerms,
                    title: document.title || '',
                  };
                }
                """
            )
            result.update(metrics or {})
        except Exception as exc:
            result["metrics_error"] = str(exc)

        try:
            page.screenshot(path=str(screenshot_path), full_page=True)
        except Exception as exc:
            result["screenshot_error"] = str(exc)

        try:
            html_path.write_text(page.content(), encoding="utf-8")
        except Exception as exc:
            result["html_error"] = str(exc)

        if console_msgs:
            result["console"] = console_msgs[-50:]

        context.close()
        browser.close()

    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def build_glimpse_risk_flags(glimpse: dict) -> dict:
    content_type = str(glimpse.get("content_type") or "").lower()
    has_tables = int(glimpse.get("table_count") or 0) > 0
    has_election_terms = bool(glimpse.get("has_election_terms"))
    status = glimpse.get("status")

    content_type_supported = (
        ("text/html" in content_type)
        or ("application/xhtml+xml" in content_type)
        or not content_type
    )

    risk_level = "low"
    if status is None or int(status) >= 400:
        risk_level = "high"
    elif not content_type_supported:
        risk_level = "high"
    elif not has_tables and not has_election_terms:
        risk_level = "medium"

    return {
        "status_code": status,
        "content_type": content_type,
        "content_type_supported": content_type_supported,
        "tables_found": has_tables,
        "has_election_terms": has_election_terms,
        "risk_level": risk_level,
    }
