"""
Clarity Elections base handler.

Provides vendor-level URL pattern checks and default behavior.
"""
from __future__ import annotations

import re
from typing import Any, Dict

from webapp.parser.handlers.shared.state_handler_base import StateHandlerBase
from webapp.parser.utils.logger_singleton import logger


class ClarityBaseHandler(StateHandlerBase):
    """Base handler for Clarity Elections sites."""

    VENDOR_NAME = "Clarity"
    VENDOR_URL_PATTERNS = [
        r"https://results\.clarityelections\.com/.*",
        r"https://.*clarityelections\.com/.*",
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
