"""
Vendor dispatch handler.

Resolves a vendor base class per state and delegates parsing.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from webapp.parser.Context_Integration.librarian import get_state_abbr, lookup_state
from webapp.parser.handlers.shared.state_scaffold import parse as scaffold_parse
from webapp.parser.handlers.shared.vendors.clarity_base_handler import ClarityBaseHandler
from webapp.parser.handlers.shared.vendors.dominion_base_handler import DominionBaseHandler
from webapp.parser.handlers.shared.vendors.voteworks_base_handler import VoteWorksBaseHandler
from webapp.parser.handlers.vendor_state_map import get_vendor_for_state
from webapp.parser.utils.logger_singleton import logger

_VENDOR_CLASS_MAP = {
    "clarity": ClarityBaseHandler,
    "voteworks": VoteWorksBaseHandler,
    "dominion": DominionBaseHandler,
}

_HANDLER_CACHE: Dict[Tuple[str, str], Any] = {}


def _display_state_name(canonical_state: str) -> str:
    return canonical_state.replace("_", " ").title()


def _get_canonical_state(html_context: Dict[str, Any] | None, context: Dict[str, Any] | None) -> str | None:
    ctx = html_context or (context or {})
    state_input = (
        ctx.get("state")
        or ctx.get("state_abbr")
        or ctx.get("detected_state")
        or ctx.get("state_code")
    )
    if not state_input:
        return None
    return lookup_state(str(state_input))


def _get_handler(canonical_state: str) -> Any | None:
    vendor = get_vendor_for_state(canonical_state)
    if not vendor:
        return None

    base_class = _VENDOR_CLASS_MAP.get(vendor)
    if not base_class:
        return None

    state_abbr = get_state_abbr(canonical_state)
    if not state_abbr:
        return None

    cache_key = (vendor, canonical_state)
    cached = _HANDLER_CACHE.get(cache_key)
    if cached:
        return cached

    class VendorStateHandler(base_class):
        STATE_NAME = _display_state_name(canonical_state)
        STATE_CODE = state_abbr

    handler = VendorStateHandler()
    _HANDLER_CACHE[cache_key] = handler
    return handler


def parse(
    page: Any = None,
    html_context: Dict[str, Any] | None = None,
    coordinator: Any = None,
    context: Dict[str, Any] | None = None,
    session_id: str | None = None,
    **kwargs,
) -> Tuple[List[str], List[Dict[str, Any]], str, Dict[str, Any]] | None:
    """Dispatch to a vendor handler when mapped; otherwise fallback to scaffold."""
    canonical_state = _get_canonical_state(html_context, context)
    if not canonical_state:
        return scaffold_parse(page=page, coordinator=coordinator, context=(html_context or context or {}), session_id=session_id, **kwargs)

    handler = _get_handler(canonical_state)
    if not handler:
        logger.debug(f"[VendorDispatch] No vendor mapping for state={canonical_state}; using scaffold.")
        return scaffold_parse(page=page, coordinator=coordinator, context=(html_context or context or {}), session_id=session_id, **kwargs)

    return handler.parse(
        page=page,
        html_context=html_context,
        coordinator=coordinator,
        context=context,
        session_id=session_id,
        **kwargs,
    )
