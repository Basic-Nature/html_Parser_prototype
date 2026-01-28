from __future__ import annotations

from typing import Any, Dict, Tuple

from webapp.parser.html_election_parser import generate_generic_html_result
from webapp.parser.utils.logger_singleton import logger


def parse(page: Any = None, coordinator: Any = None, context: Dict[str, Any] | None = None, session_id: str | None = None, **kwargs) -> Tuple[list, list, str, dict] | None:
    """
    Generic HTML dynamic fallback handler.
    If `page` is provided, it will attempt to extract page HTML, otherwise
    it will look for `html_text` in the provided `context`.
    """
    ctx = dict(context or {})
    html_text = ctx.get('html_text') or ctx.get('raw_html') or None
    try:
        result = generate_generic_html_result(
            page=page,
            coordinator=coordinator,
            context=ctx,
            session_id=session_id,
            html_text=html_text,
            log_type="html_dynamic_fallback",
        )
        return result
    except Exception as e:
        try:
            logger.error({"level":"ERROR","type":"handler","message":f"html_dynamic_fallback parse failed: {e}","session_id":session_id})
        except Exception:
            pass
        return None
