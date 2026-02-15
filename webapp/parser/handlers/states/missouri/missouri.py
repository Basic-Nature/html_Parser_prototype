from __future__ import annotations

from typing import Any, Dict

from webapp.parser.handlers.shared.state_handler_base import SimpleTableHandler


class MissouriHandler(SimpleTableHandler):
    """Handler for Missouri election data."""

    STATE_NAME = "Missouri"
    STATE_CODE = "MO"

# Create module-level parse function for router compatibility
_handler_instance = MissouriHandler()

def parse(page: Any = None, html_context: Dict[str, Any] | None = None, coordinator: Any = None, context: Dict[str, Any] | None = None, session_id: str | None = None, **kwargs):
    """State handler for Missouri."""
    return _handler_instance.parse(
        page=page,
        html_context=html_context,
        coordinator=coordinator,
        context=context,
        session_id=session_id,
        **kwargs,
    )
