"""
California State Handler

Auto-generated handler for California election data extraction.

This handler uses SimpleTableHandler base class, which delegates
table extraction to the robust_table_extraction strategy pipeline.

To customize:
1. Override extract_tables() for custom extraction logic
2. Override pre_extraction_hook() for navigation/button clicks
3. Override should_use_fallback() for URL pattern checks

Generated with: python scripts/generate_state_handler.py California --simple
"""
from webapp.parser.handlers.shared.state_handler_base import SimpleTableHandler


class CaliforniaHandler(SimpleTableHandler):
    """Handler for California election data."""
    
    STATE_NAME = "California"
    STATE_CODE = "CA"
    
    # === Optional: Add custom behavior below ===
    
    # def pre_extraction_hook(self, contest, page, html_context, coordinator, session_id, **kwargs):
    #     """Run before table extraction. Override for custom navigation."""
    #     # Example: Click buttons, wait for elements, etc.
    #     pass
    
    # def should_use_fallback(self, page, html_context) -> bool:
    #     """Return True to delegate to generic HTML handler."""
    #     # Example: Check URL pattern
    #     url = html_context.get("url", "")
    #     if "expected-vendor.com" not in url:
    #         return True  # Not our expected vendor, use fallback
    #     return False


# Create module-level parse function for router compatibility
_handler_instance = CaliforniaHandler()

def parse(page=None, html_context=None, coordinator=None, context=None, session_id=None, **kwargs):
    """Module-level parse function called by state router."""
    return _handler_instance.parse(
        page=page,
        html_context=html_context,
        coordinator=coordinator,
        context=context,
        session_id=session_id,
        **kwargs,
    )
