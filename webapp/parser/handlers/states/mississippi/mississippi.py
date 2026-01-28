from __future__ import annotations

from typing import Any, Dict

from webapp.parser.handlers.formats.html_dynamic_fallback import parse as dynamic_parse


def parse(page=None, html_context=None, coordinator=None, context=None, session_id=None, **kwargs):
    """State scaffold handler that delegates to the dynamic HTML fallback.
    This file was auto-generated. Replace with a state-specific implementation when ready.
    """
    ctx = html_context or (context or {})
    return dynamic_parse(page=page, coordinator=coordinator, context=ctx, session_id=session_id, **kwargs)
