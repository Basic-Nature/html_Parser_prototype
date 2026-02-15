from __future__ import annotations

from typing import Any, Dict, Tuple

from webapp.parser.handlers.formats.html_dynamic_fallback import parse as dynamic_parse
from webapp.parser.handlers.shared.parity_hooks import (
    attach_parity_note_to_metadata,
    extract_router_parity_note,
)


def parse(
    page: Any = None,
    html_context: Dict[str, Any] | None = None,
    coordinator: Any = None,
    context: Dict[str, Any] | None = None,
    session_id: str | None = None,
    **kwargs,
) -> Tuple[list, list, str, dict] | None:
    """Shared scaffold handler that delegates to the dynamic HTML fallback."""
    ctx = html_context or (context or {})
    parity_note = extract_router_parity_note(ctx)
    result = dynamic_parse(page=page, coordinator=coordinator, context=ctx, session_id=session_id, **kwargs)
    if not result or not isinstance(result, tuple) or len(result) != 4:
        return result
    headers, rows, contest, metadata = result
    metadata = attach_parity_note_to_metadata(metadata, parity_note)
    return headers, rows, contest, metadata
