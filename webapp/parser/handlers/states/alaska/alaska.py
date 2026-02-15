from __future__ import annotations

from typing import Any, Dict

from webapp.parser.handlers.shared.state_handler_base import SimpleTableHandler


class AlaskaHandler(SimpleTableHandler):
    """Handler for Alaska election data."""

    STATE_NAME = "Alaska"
    STATE_CODE = "AK"

# Create module-level parse function for router compatibility
_handler_instance = AlaskaHandler()

def parse(page: Any = None, html_context: Dict[str, Any] | None = None, coordinator: Any = None, context: Dict[str, Any] | None = None, session_id: str | None = None, **kwargs):
    """State handler for Alaska."""
    return _handler_instance.parse(
        page=page,
        html_context=html_context,
        coordinator=coordinator,
        context=context,
        session_id=session_id,
        **kwargs,
    )
