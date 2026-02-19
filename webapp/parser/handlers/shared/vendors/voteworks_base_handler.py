"""
VoteWorks base handler.

Provides vendor-level URL pattern checks and default behavior.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from webapp.parser.handlers.shared.state_handler_base import StateHandlerBase
from webapp.parser.utils.logger_singleton import logger
from webapp.parser.utils.table_core import robust_table_extraction


class VoteWorksBaseHandler(StateHandlerBase):
    """Base handler for VoteWorks-hosted results sites."""

    VENDOR_NAME = "VoteWorks"
    VENDOR_URL_PATTERNS = [
        r"https://results\.voteworks\.com/.*",
        r"https://.*voteworks\.com/.*",
    ]

    def should_use_fallback(self, page: Any, html_context: Dict[str, Any]) -> bool:
        url = (html_context or {}).get("url") or getattr(page, "url", "")
        if not url:
            return False

        for pattern in self.VENDOR_URL_PATTERNS:
            if re.search(pattern, url, re.IGNORECASE):
                return False

        logger.info(
            f"[{self.STATE_NAME}] URL did not match {self.VENDOR_NAME} patterns; using fallback."
        )
        return True
    
    def extract_tables(
        self,
        page: Any,
        contest: Dict[str, Any],
        html_context: Dict[str, Any],
        coordinator: Any,
        session_id: str,
        **kwargs,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Delegate to robust_table_extraction for VoteWorks sites."""
        result = robust_table_extraction(
            page=page,
            coordinator=coordinator,
            html_context={**html_context, "selected_contest": contest},
            session_id=session_id,
            **kwargs,
        )
        
        headers = result.get("headers", [])
        data_rows = result.get("data", [])
        
        return headers, data_rows
