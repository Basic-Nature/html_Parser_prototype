"""
Westchester County Handler (New York)

Auto-generated county-level handler for Westchester County, New York.

County handlers extend state-level logic with county-specific:
- Custom navigation sequences
- Button/toggle interactions
- Vendor-specific UI patterns
- Table structure variations

Based on Rockland County, NY reference implementation.

To customize:
1. Implement button toggle logic in parse()
2. Add navigation recipe for automation
3. Override extraction strategy if needed

Generated with: python scripts/generate_county_handler.py "New York" "Westchester"
"""
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from playwright.sync_api import Page
from webapp.parser.Context_Integration.librarian import clean_for_json
from webapp.parser.handlers.shared.vendors.enhanced_voting import parse_enhanced_voting
from webapp.parser.utils.browser_utils import (
    autoscroll_until_stable,
)
from webapp.parser.utils.contest_selector import select_contest_auto_first
from webapp.parser.utils.html_scanner import scan_html_for_context
from webapp.parser.utils.logger_singleton import logger
from webapp.parser.utils.retry_utils import retry_with_snapshot
from webapp.parser.utils.shared_logic import safe_get
from webapp.parser.utils.table_core import robust_table_extraction

if TYPE_CHECKING:
    from webapp.parser.Context_Integration.context_coordinator import ContextCoordinator


# Context cache for this county
context_cache = {}
accepted_buttons_cache = {}

WESTCHESTER_VENDOR_RULES = {
    "steps": [
        {
            "action": "select_contest",
            "label_pattern": "View results by election district : {contest}",
        },
        {
            "action": "toggle_vote_method",
            "keywords": ["Vote Method", "Ballot Type", "Method"],
        },
        {
            "action": "autoscroll_until_stable",
        },
    ]
}


def parse(
    page: Page = None,
    html_context: Dict[str, Any] = None,
    coordinator: "ContextCoordinator" = None,
    context: Dict[str, Any] = None,
    session_id: str = None,
    **kwargs,
) -> Tuple[List[str], List[Dict[str, Any]], str, Dict[str, Any]]:
    """Westchester County handler wrapper for Enhanced Voting pages."""
    if html_context is None:
        html_context = {}

    return parse_enhanced_voting(
        page=page,
        html_context=html_context,
        coordinator=coordinator,
        context=context,
        session_id=session_id,
        logger=logger,
        vendor="westchester",
        county="Westchester",
        state="NY",
        vendor_rules=WESTCHESTER_VENDOR_RULES,
        **kwargs,
    )
