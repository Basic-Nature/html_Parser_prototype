from __future__ import annotations

from typing import Any, Dict

from webapp.parser.handlers.shared.state_handler_base import SimpleTableHandler


class AlabamaHandler(SimpleTableHandler):
    """Handler for Alabama election data."""

    STATE_NAME = "Alabama"
    STATE_CODE = "AL"

# Create module-level parse function for router compatibility
_handler_instance = AlabamaHandler()

def parse(page: Any = None, html_context: Dict[str, Any] | None = None, coordinator: Any = None, context: Dict[str, Any] | None = None, session_id: str | None = None, **kwargs):
    """State handler for Alabama."""
    return _handler_instance.parse(
        page=page,
        html_context=html_context,
        coordinator=coordinator,
        context=context,
        session_id=session_id,
        **kwargs,
    )
