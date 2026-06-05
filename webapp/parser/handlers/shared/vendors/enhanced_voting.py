"""
Enhanced Voting vendor parser.

This module contains shared workflow logic for Enhanced Voting-style portals.
The goal is a reusable vendor engine that county adapters can call with
state/county metadata and a navigation recipe.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from playwright.sync_api import Page

from webapp.parser.Context_Integration.context_coordinator import ContextCoordinator
from webapp.parser.Context_Integration.librarian import clean_for_json
from webapp.parser.handlers.shared.navigation.recipe_runner import run_navigation_recipe
from webapp.parser.utils.browser_utils import (
    autoscroll_until_stable,
    safe_click,
    safe_is_enabled,
    safe_is_visible,
)
from webapp.parser.utils.contest_selector import select_contest_auto_first
from webapp.parser.utils.html_scanner import scan_html_for_context
from webapp.parser.utils.logger_singleton import logger
from webapp.parser.utils.shared_logic import safe_get
from webapp.parser.utils.table_core import robust_table_extraction

DEFAULT_ENHANCED_VOTING_RECIPE = [
    {
        "action": "select_contest",
        "strategy": "role_link_exact_or_fuzzy",
        "label_pattern": "View results by election district : {contest}",
    },
    {
        "action": "toggle_vote_method",
        "strategy": "button_or_tab_text",
        "keywords": ["Vote Method", "Ballot Type", "Method"],
    },
    {
        "action": "autoscroll_until_stable",
    },
]


def _ensure_contest_title(contest: Dict[str, Any]) -> str:
    if not isinstance(contest, dict):
        return ""
    return str(safe_get(contest, "title", "") or "").strip()


def parse_enhanced_voting(
    page: Page | None = None,
    state: str | None = None,
    county: str | None = None,
    vendor: str | None = None,
    html_context: Dict[str, Any] | None = None,
    coordinator: Any | None = None,
    context: Dict[str, Any] | None = None,
    session_id: str | None = None,
    vendor_rules: Dict[str, Any] | None = None,
    **kwargs,
) -> Tuple[List[str], List[Dict[str, Any]], str, Dict[str, Any]] | None:
    """Shared parse workflow for Enhanced Voting vendor pages."""
    html_context = html_context or context or {}
    state = state or safe_get(html_context, "state", "NY")
    county = county or safe_get(html_context, "county", "Unknown")
    session_id = session_id or safe_get(html_context, "session_id", None)

    if coordinator is None:
        coordinator = ContextCoordinator()

    logger.info(
        f"[EnhancedVoting] Starting shared parse for state={state} county={county}"
    )

    context_result = scan_html_for_context(
        target_url=getattr(page, "url", None) if page else html_context.get("url"),
        page=page,
        coordinator=coordinator,
        session_id=session_id,
        allow_duplicates=getattr(coordinator, "allow_duplicates", False),
        context_cache={},
        debug=html_context.get("debug", False),
        **kwargs,
    )

    state = context_result.get("state") or state
    county = context_result.get("county") or county
    year = context_result.get("year")
    for contest in safe_get(context_result, "contests", []):
        if safe_get(contest, "state", None) is None:
            contest["state"] = state
        if safe_get(contest, "county", None) is None:
            contest["county"] = county
        if safe_get(contest, "year", None) is None and year is not None:
            contest["year"] = year
        if session_id is not None:
            contest["session_id"] = session_id

    context_result = clean_for_json(context_result)
    coordinator.organize_and_enrich(context_result)

    context_for_selector = {
        "state": state,
        "county": county,
        "year": year,
        "contests": context_result.get("contests", []),
        "session_id": session_id,
        **{k: v for k, v in html_context.items() if k not in ("state", "county", "year", "contests")},
    }

    selected = select_contest_auto_first(
        coordinator=coordinator,
        context=context_for_selector,
        session_id=session_id,
        allow_multiple=False,
        force_interactive=html_context.get("force_interactive", False),
    )

    if not selected:
        logger.warning("[EnhancedVoting] No contest selected; skipping parse.")
        return None, None, None, {"skipped": True, "state": state, "county": county}

    if isinstance(selected, list) and selected:
        selected = selected[0]

    contest_title = _ensure_contest_title(selected)
    vendor_rules = vendor_rules or safe_get(html_context, "vendor_rules", {"steps": DEFAULT_ENHANCED_VOTING_RECIPE})
    recipe = safe_get(vendor_rules, "steps", DEFAULT_ENHANCED_VOTING_RECIPE)

    run_navigation_recipe(
        page=page,
        contest=selected,
        html_context=html_context,
        recipe=recipe,
        coordinator=coordinator,
        session_id=session_id,
    )

    if page is not None:
        autoscroll_until_stable(page, session_id=session_id)
        page.wait_for_timeout(2000)

    extraction_context = {
        **html_context,
        "selected_contest": selected,
        "state": state,
        "county": county,
        "vendor": vendor or "enhanced_voting",
        "handler": "shared.vendors.enhanced_voting",
    }

    result = robust_table_extraction(
        page=page,
        coordinator=coordinator,
        html_context=extraction_context,
        session_id=session_id,
        **kwargs,
    )

    headers = result.get("headers", []) if isinstance(result, dict) else []
    data_rows = result.get("data", []) if isinstance(result, dict) else []
    contest_title = contest_title or safe_get(selected, "title", "Unknown Contest")

    metadata = {
        "state": state,
        "county": county,
        "race": contest_title,
        "source": getattr(page, "url", "Unknown"),
        "handler": "enhanced_voting",
        "session_id": session_id,
        "vendor": vendor or "enhanced_voting",
    }
    if year is not None:
        metadata["year"] = year

    return headers, data_rows, contest_title, metadata


def click_enhancedvoting_contest(
    page: Page,
    contest_title: str,
    label_pattern: str | None = None,
) -> bool:
    """Click the contest detail link for an Enhanced Voting contest."""
    if not contest_title or page is None:
        return False

    pattern_template = label_pattern or "View results by election district : {contest}"
    literal_pattern = pattern_template.format(contest=re.escape(contest_title))
    try:
        regex = re.compile(rf"^{literal_pattern}$", re.I)
        link = page.get_by_role("link", name=regex)
        if link.count() == 0:
            return False
        link.first().click()
        page.wait_for_timeout(3000)
        return True
    except Exception:
        return False


def click_toggle_by_keywords(page: Page, keywords: List[str]) -> bool:
    if page is None or not keywords:
        return False

    for keyword in keywords:
        pattern = re.compile(re.escape(keyword), re.I)
        try:
            locator = page.get_by_role("button", name=pattern)
            if locator.count() > 0:
                locator.first().click()
                page.wait_for_timeout(2000)
                return True
        except Exception:
            pass

        try:
            locator = page.get_by_role("link", name=pattern)
            if locator.count() > 0:
                locator.first().click()
                page.wait_for_timeout(2000)
                return True
        except Exception:
            pass

    # Fall back to visible button-like labels
    candidates = page.locator("button, a, [role='button'], input[type='button'], input[type='submit']")
    for i in range(min(candidates.count(), 80)):
        element = candidates.nth(i)
        label = ""
        try:
            label = element.inner_text() or ""
        except Exception:
            pass
        if any(keyword.lower() in label.lower() for keyword in keywords):
            if safe_is_visible(element, logger) and safe_is_enabled(element, logger):
                try:
                    safe_click(element, logger)
                    page.wait_for_timeout(2000)
                    return True
                except Exception:
                    continue
    return False
