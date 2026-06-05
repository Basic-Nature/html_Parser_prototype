import re
from pathlib import Path
from typing import TYPE_CHECKING

from playwright.sync_api import Page

from webapp.parser.handlers.shared.vendors.enhanced_voting import parse_enhanced_voting
from .....Context_Integration.librarian import clean_for_json
from .....utils.browser_utils import (
    autoscroll_until_stable,
    safe_click,
    safe_is_enabled,
    safe_is_visible,
)
from .....utils.contest_selector import select_contest_auto_first
from .....utils.html_scanner import scan_html_for_context
from .....utils.logger_singleton import logger, prompt
from .....utils.output_utils import finalize_election_output
from .....utils.shared_logic import safe_get
from .....utils.table_builder import build_dynamic_table
from .....utils.table_core import harmonize_headers_and_data

if TYPE_CHECKING:
    from .....Context_Integration.context_coordinator import ContextCoordinator

BUTTON_SELECTORS = "button, a, [role='button'], input[type='button'], input[type='submit']"
context_cache = {}
accepted_buttons_cache = {}

DEBUG_OUTPUT_DIR = Path("tools") / "debug_headless_output"

TOGGLE_KEYWORDS = {
    "election_district": {
        "view results by election district": 5.0,
        "results by election district": 4.0,
        "election district": 3.5,
        "by election district": 3.0,
        "district results": 2.5,
        "district": 1.0,
    },
    "vote_method": {
        "vote method": 4.0,
        "method": 1.5,
        "ballot type": 1.0,
    },
}

KEYWORD_VOCAB = {
    "contest": {
        "president": 4.0,
        "vice president": 4.0,
        "county clerk": 3.5,
        "court": 3.0,
        "judge": 3.0,
        "proposition": 2.5,
        "amendment": 2.5,
        "referendum": 2.5,
        "ballot question": 2.0,
    },
    "precinct": {
        "ward": 2.5,
        "district": 2.5,
        "precinct": 2.5,
        "election district": 3.0,
        "ed": 1.0,
    },
    "candidate": {
        "candidate": 1.0,
        "vote": 1.0,
        "votes": 1.0,
        "percentage": 1.0,
        "party": 1.0,
        "democratic": 1.5,
        "republican": 1.5,
        "conservative": 1.0,
        "working families": 1.0,
        "independence": 1.0,
        "write-in": 1.0,
    },
}

def _write_debug_html(session_id: str | None, html: str) -> None:
    try:
        DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        suffix = session_id or "session"
        out_path = DEBUG_OUTPUT_DIR / f"rockland_debug_{suffix}.html"
        out_path.write_text(html, encoding="utf-8")
    except Exception:
        pass

def click_enhancedvoting_contest(page: Page, contest_title: str) -> bool:
    """Click a deterministic enhanced voting contest link in Rockland County."""
    try:
        pattern = re.compile(
            rf"^View results by election district\s*:\s*{re.escape(contest_title)}$",
            re.I,
        )
        link = page.get_by_role("link", name=pattern)
        if link.count() == 0:
            return False
        link.first().click()
        page.wait_for_timeout(3000)
        return True
    except Exception:
        return False

def _score_keyword_match(text: str, weights: dict[str, float]) -> float:
    if not text:
        return 0.0
    lowered = text.lower()
    score = 0.0
    for key, weight in weights.items():
        if key in lowered:
            score += weight
    return score

def _extract_button_label(element) -> str:
    try:
        label = element.inner_text() or ""
    except Exception:
        label = ""
    if not label:
        try:
            label = element.get_attribute("aria-label") or ""
        except Exception:
            label = ""
    if not label:
        try:
            label = element.get_attribute("title") or ""
        except Exception:
            label = ""
    return label.strip()

def _fallback_button_search(page: Page, weights: dict[str, float]) -> dict:
    try:
        candidates = page.locator(BUTTON_SELECTORS)
        best = None
        best_score = 0.0
        for i in range(min(candidates.count(), 80)):
            element = candidates.nth(i)
            label = _extract_button_label(element)
            if not label:
                continue
            score = _score_keyword_match(label, weights)
            if score > best_score:
                best = {"element_handle": element, "label": label, "selector": None}
                best_score = score
        if best and best_score >= 2.5:
            return best
    except Exception:
        pass
    return {}

def _score_keyword_groups(text: str, vocab: dict[str, dict[str, float]]) -> dict[str, float]:
    scores = {}
    for group, weights in vocab.items():
        scores[group] = _score_keyword_match(text, weights)
    return scores

def _flatten_panel_text(panel: dict) -> str:
    parts = []
    heading = safe_get(panel, "panel_heading", "")
    if heading:
        parts.append(str(heading))
    for table in safe_get(panel, "tables", []):
        html = safe_get(table, "table_html", "")
        if html:
            parts.append(str(html))
    return " ".join(parts)

def parse(page: Page = None, html_context: dict = None, coordinator: "ContextCoordinator" = None, context=None, session_id=None, logger=logger, **kwargs) -> tuple:
    """Rockland County lightweight adapter for Enhanced Voting pages."""
    if html_context is None:
        html_context = {}

    return parse_enhanced_voting(
        page=page,
        html_context=html_context,
        coordinator=coordinator,
        context=context,
        session_id=session_id,
        logger=logger,
        vendor="rockland",
        **kwargs,
    )