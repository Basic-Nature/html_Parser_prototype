#!/usr/bin/env python3
"""
State Handler Generator

Generates a new state handler from template with proper scaffolding.

Usage:
    python scripts/generate_state_handler.py California
    python scripts/generate_state_handler.py "New York" --county Westchester
    python scripts/generate_state_handler.py Texas --simple
    python scripts/generate_state_handler.py Florida --vendor clarity

Options:
    --simple: Use SimpleTableHandler base (delegates to robust_table_extraction)
    --vendor NAME: Use vendor-specific base class (clarity, voteworks, dominion)
    --county NAME: Also generate county-level handler
    --force: Overwrite existing handler file
"""
import argparse
import sys
from pathlib import Path

# US state codes mapping
STATE_CODES = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY", "District of Columbia": "DC",
}

VENDOR_BASE_CLASSES = {
    "clarity": "ClarityBaseHandler",
    "voteworks": "VoteWorksBaseHandler",
    "dominion": "DominionBaseHandler",
}


def get_state_code(state_name: str) -> str:
    """Get 2-letter state code from full name."""
    # Try exact match first
    if state_name in STATE_CODES:
        return STATE_CODES[state_name]
    
    # Try case-insensitive match
    for name, code in STATE_CODES.items():
        if name.lower() == state_name.lower():
            return code
    
    # Try if user provided code directly
    if len(state_name) == 2 and state_name.upper() in STATE_CODES.values():
        return state_name.upper()
    
    raise ValueError(f"Unknown state: {state_name}. Use full name or 2-letter code.")


def get_template_simple(state_name: str, state_code: str) -> str:
    """Generate template using SimpleTableHandler."""
    return f'''"""
{state_name} State Handler

Auto-generated handler for {state_name} election data extraction.

This handler uses SimpleTableHandler base class, which delegates
table extraction to the robust_table_extraction strategy pipeline.

To customize:
1. Override extract_tables() for custom extraction logic
2. Override pre_extraction_hook() for navigation/button clicks
3. Override should_use_fallback() for URL pattern checks

Generated with: python scripts/generate_state_handler.py {state_name} --simple
"""
from webapp.parser.handlers.shared.state_handler_base import SimpleTableHandler


class {state_name.replace(" ", "")}Handler(SimpleTableHandler):
    """Handler for {state_name} election data."""
    
    STATE_NAME = "{state_name}"
    STATE_CODE = "{state_code}"
    
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
_handler_instance = {state_name.replace(" ", "")}Handler()

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
'''


def get_template_custom(state_name: str, state_code: str) -> str:
    """Generate template with custom extract_tables implementation."""
    snake_case = state_name.lower().replace(" ", "_")
    
    return f'''"""
{state_name} State Handler

Auto-generated handler for {state_name} election data extraction.

This handler uses StateHandlerBase with custom extract_tables() implementation.

To customize:
1. Implement extract_tables() with state-specific logic
2. Override hooks (pre_scan_hook, post_extraction_hook, etc.) as needed
3. Override should_use_fallback() for URL pattern checks

Generated with: python scripts/generate_state_handler.py {state_name}
"""
from typing import Any, Dict, List, Tuple

from webapp.parser.handlers.shared.state_handler_base import StateHandlerBase
from webapp.parser.utils.table_core import robust_table_extraction
from webapp.parser.utils.logger_singleton import logger


class {state_name.replace(" ", "")}Handler(StateHandlerBase):
    """Handler for {state_name} election data."""
    
    STATE_NAME = "{state_name}"
    STATE_CODE = "{state_code}"
    
    def extract_tables(
        self,
        page: Any,
        contest: Dict[str, Any],
        html_context: Dict[str, Any],
        coordinator: Any,
        session_id: str,
        **kwargs,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Extract election results tables for {state_name}.
        
        TODO: Implement state-specific extraction logic.
        
        Options:
        1. Delegate to robust_table_extraction (recommended to start):
           result = robust_table_extraction(page, coordinator, html_context, session_id)
           return result.get("headers", []), result.get("data", [])
        
        2. Custom DOM traversal with selectolax/Playwright:
           tables = page.query_selector_all("table.results")
           # ... custom parsing logic
        
        3. Use table_builder utilities:
           from webapp.parser.utils.table_builder import build_dynamic_table
           headers, data = build_dynamic_table(table_element, coordinator)
           return headers, data
        """
        logger.info(f"[{{self.STATE_NAME}}] Extracting tables for contest: {{contest.get('title')}}")
        
        # Default implementation: delegate to robust_table_extraction
        result = robust_table_extraction(
            page=page,
            coordinator=coordinator,
            html_context={{**html_context, "selected_contest": contest}},
            session_id=session_id,
            **kwargs,
        )
        
        headers = result.get("headers", [])
        data_rows = result.get("data", [])
        
        # TODO: Add {state_name}-specific transformations here
        # Example:
        # - Normalize party names
        # - Add precinct ID from contest title
        # - Filter out summary rows
        
        return headers, data_rows
    
    # === Optional: Override hooks for custom behavior ===
    
    # def pre_extraction_hook(self, contest, page, html_context, coordinator, session_id, **kwargs):
    #     """Run before table extraction. Add navigation logic here."""
    #     # Example: Click buttons specific to {state_name} sites
    #     pass
    
    # def should_use_fallback(self, page, html_context) -> bool:
    #     """Return True to delegate to generic HTML handler."""
    #     url = html_context.get("url", "")
    #     # Example: Check for expected URL pattern
    #     if "{snake_case}" not in url.lower():
    #         logger.warning(f"URL does not match {state_name} pattern, using fallback")
    #         return True
    #     return False


# Create module-level parse function for router compatibility
_handler_instance = {state_name.replace(" ", "")}Handler()

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
'''


def get_template_vendor(state_name: str, state_code: str, vendor: str) -> str:
    """Generate template using vendor-specific base class."""
    base_class = VENDOR_BASE_CLASSES.get(vendor.lower())
    if not base_class:
        raise ValueError(f"Unknown vendor: {vendor}. Use: {', '.join(VENDOR_BASE_CLASSES.keys())}")
    
    return f'''"""
{state_name} State Handler

Auto-generated handler for {state_name} election data extraction.

This handler uses {base_class} for {vendor.title()} election systems.

The vendor base class provides:
- Standard navigation patterns for {vendor.title()} systems
- Common JSON API extraction logic
- Typical button/toggle handling
- Result table structure knowledge

To customize:
1. Override vendor-specific methods as needed
2. Add state-specific URL patterns
3. Customize metadata fields

Generated with: python scripts/generate_state_handler.py {state_name} --vendor {vendor}
"""
from webapp.parser.handlers.shared.vendors.{vendor.lower()}_base_handler import {base_class}


class {state_name.replace(" ", "")}Handler({base_class}):
    """Handler for {state_name} election data using {vendor.title()} systems."""
    
    STATE_NAME = "{state_name}"
    STATE_CODE = "{state_code}"
    
    # === Optional: Add state-specific customization ===
    
    # def get_expected_url_patterns(self):
    #     """Return list of expected URL patterns for {state_name}."""
    #     return [
    #         r"https://results\\.{state_code.lower()}\\.gov/.*",
    #         r"https://.*\\.{state_code.lower()}elections\\..*",
    #     ]


# Create module-level parse function for router compatibility
_handler_instance = {state_name.replace(" ", "")}Handler()

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
'''


def generate_handler(state_name: str, simple: bool = False, vendor: str = None, force: bool = False) -> Path:
    """Generate state handler file."""
    # Get state code
    state_code = get_state_code(state_name)
    
    # Determine output path
    project_root = Path(__file__).parent.parent
    snake_case = state_name.lower().replace(" ", "_")
    handler_dir = project_root / "webapp" / "parser" / "handlers" / "states"
    handler_dir.mkdir(parents=True, exist_ok=True)
    
    handler_file = handler_dir / f"{snake_case}.py"
    
    # Check if exists
    if handler_file.exists() and not force:
        print(f"❌ Handler already exists: {handler_file}")
        print("   Use --force to overwrite")
        return None
    
    # Generate template
    if vendor:
        content = get_template_vendor(state_name, state_code, vendor)
    elif simple:
        content = get_template_simple(state_name, state_code)
    else:
        content = get_template_custom(state_name, state_code)
    
    # Write file
    handler_file.write_text(content, encoding="utf-8")
    
    return handler_file


def main():
    parser = argparse.ArgumentParser(
        description="Generate a new state handler from template",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple handler (delegates to robust_table_extraction)
  python scripts/generate_state_handler.py California --simple
  
  # Custom handler (implement extract_tables yourself)
  python scripts/generate_state_handler.py Texas
  
  # Vendor-specific handler
  python scripts/generate_state_handler.py Florida --vendor clarity
  
  # With county-level handler
  python scripts/generate_state_handler.py "New York" --county Westchester
  
  # Overwrite existing
  python scripts/generate_state_handler.py Ohio --force
        """,
    )
    parser.add_argument("state", help="State name or 2-letter code")
    parser.add_argument("--simple", action="store_true", help="Use SimpleTableHandler base")
    parser.add_argument("--vendor", choices=list(VENDOR_BASE_CLASSES.keys()), help="Use vendor-specific base")
    parser.add_argument("--county", help="Also generate county-level handler")
    parser.add_argument("--force", action="store_true", help="Overwrite existing handler")
    
    args = parser.parse_args()
    
    try:
        # Generate state handler
        handler_file = generate_handler(args.state, args.simple, args.vendor, args.force)
        
        if handler_file:
            print(f"✅ Generated state handler: {handler_file}")
            print(f"   State: {args.state}")
            
            if args.simple:
                print("   Type: SimpleTableHandler (automatic extraction)")
            elif args.vendor:
                print(f"   Type: {VENDOR_BASE_CLASSES[args.vendor]} ({args.vendor.title()} vendor)")
            else:
                print("   Type: Custom StateHandlerBase")
            
            print("\n📝 Next steps:")
            print(f"   1. Review and customize: {handler_file}")
            print("   2. Test with: python -m webapp.parser.html_election_parser --url <state-url>")
            
            if not args.simple and not args.vendor:
                print("   3. Implement extract_tables() method with state-specific logic")
            
            # Generate county handler if requested
            if args.county:
                print(f"\n🏛️  Generating county handler for {args.county}...")
                from subprocess import call
                call([
                    sys.executable,
                    "scripts/generate_county_handler.py",
                    args.state,
                    args.county,
                ])
        
        return 0
    
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
