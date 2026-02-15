#!/usr/bin/env python3
"""Random navigation smoke test (navigation-only, no parsing/output).

Runs the navigation runner on a random sample of URLs from urls.txt and reports
basic outcomes. By default it does not persist navigation feedback logs.
"""

from __future__ import annotations

import argparse
import random
import time
from typing import List

from playwright.sync_api import sync_playwright
from webapp.parser.config import URL_LIST_FILE
from webapp.parser.Context_Integration.context_coordinator import ContextCoordinator
from webapp.parser.html_election_parser import NAVIGATION_RUNNER
from webapp.parser.utils.browser_utils import sync_browser_pipeline, sync_safe_browser_close
from webapp.parser.utils.logger_singleton import logger
from webapp.parser.utils.misc_utils import extract_url_and_label
from webapp.parser.utils.shared_logic import infer_state_county_from_url


def _load_urls(limit: int | None = None) -> List[str]:
    urls: List[str] = []
    if not URL_LIST_FILE.exists():
        return urls
    for raw in URL_LIST_FILE.read_text(encoding="utf-8").splitlines():
        url, _ = extract_url_and_label(raw)
        if url:
            urls.append(url)
        if limit and len(urls) >= limit:
            break
    return urls


def _run_navigation(url: str, *, session_id: str, persist_log: bool) -> bool:
    state, county = infer_state_county_from_url(url)
    coordinator = ContextCoordinator()
    nav_context = {"state": state, "county": county, "url": url}
    nav_context_before = dict(nav_context)

    with sync_playwright() as playwright:
        browser, context, page, _, _ = sync_browser_pipeline(
            playwright,
            url,
            session_id=session_id,
        )
        if page is None:
            logger.error({
                "level": "ERROR",
                "type": "navigation",
                "message": f"Failed to open page: {url}",
                "session_id": session_id,
            })
            sync_safe_browser_close(browser, session_id=session_id)
            return False

        try:
            if NAVIGATION_RUNNER is None:
                logger.warning({
                    "level": "WARNING",
                    "type": "navigation",
                    "message": "Navigation runner not available.",
                    "session_id": session_id,
                })
                return False
            nav_output = NAVIGATION_RUNNER.run(
                page,
                context=nav_context,
                coordinator=coordinator,
                session_id=session_id,
            )
            if nav_output and nav_output.context_updates:
                nav_context.update(nav_output.context_updates)
            if persist_log and bool(getattr(nav_output, "executed", False)):
                coordinator.record_navigation_feedback(
                    script_id=getattr(nav_output, "script_id", None),
                    success=bool(getattr(nav_output, "executed", False)),
                    context_before=nav_context_before,
                    context_after=dict(nav_context),
                    telemetry=getattr(nav_output, "telemetry", None),
                    metadata=getattr(nav_output, "metadata", None),
                )
            return bool(getattr(nav_output, "executed", False)) if nav_output is not None else False
        finally:
            try:
                if page is not None:
                    page.close()
            except Exception:
                pass
            try:
                if context is not None:
                    context.close()
            except Exception:
                pass
            sync_safe_browser_close(browser, session_id=session_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Navigation-only random smoke test.")
    parser.add_argument("--count", type=int, default=2, help="Number of URLs to sample.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--persist-log", action="store_true", help="Persist navigation learning log entries.")
    args = parser.parse_args()

    urls = _load_urls()
    if not urls:
        raise SystemExit("No URLs found in urls.txt")

    if args.seed is not None:
        random.seed(args.seed)

    sample = random.sample(urls, k=min(args.count, len(urls)))
    successes = 0
    for idx, url in enumerate(sample, start=1):
        session_id = f"nav_smoke_{int(time.time())}_{idx}"
        print(f"[{idx}/{len(sample)}] Navigating: {url}")
        if _run_navigation(url, session_id=session_id, persist_log=args.persist_log):
            successes += 1
            print("  -> navigation script executed")
        else:
            print("  -> no navigation script executed")

    print(f"Completed {len(sample)} runs, {successes} executed.")


if __name__ == "__main__":
    main()
