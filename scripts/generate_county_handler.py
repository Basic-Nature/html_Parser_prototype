#!/usr/bin/env python3
"""
County Handler Generator

Generates a new county-level handler under a state handler directory.

Usage:
    python scripts/generate_county_handler.py "New York" Westchester
    python scripts/generate_county_handler.py CA "Los Angeles"
    python scripts/generate_county_handler.py Florida "Miami-Dade" --navigation-recipe

Options:
    --navigation-recipe: Also generate navigation recipe template
    --force: Overwrite existing handler file
"""
import argparse
import sys
from pathlib import Path
from typing import Tuple

# Import state codes from generate_state_handler
try:
    from generate_state_handler import STATE_CODES, get_state_code
except ImportError:
    # Fallback if not in path
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
    
    def get_state_code(state_name: str) -> str:
        """Get 2-letter state code from full name."""
        if state_name in STATE_CODES:
            return STATE_CODES[state_name]
        for name, code in STATE_CODES.items():
            if name.lower() == state_name.lower():
                return code
        if len(state_name) == 2 and state_name.upper() in STATE_CODES.values():
            return state_name.upper()
        raise ValueError(f"Unknown state: {state_name}")


def get_county_template(state_name: str, state_code: str, county_name: str) -> str:
    """Generate county handler template."""
    snake_case_county = county_name.lower().replace(" ", "_").replace("-", "_")
    
    return f'''"""
{county_name} County Handler ({state_name})

Auto-generated county-level handler for {county_name} County, {state_name}.

County handlers extend state-level logic with county-specific:
- Custom navigation sequences
- Button/toggle interactions
- Vendor-specific UI patterns
- Table structure variations

Based on Rockland County, NY reference implementation.

To customize:
1. Implement button toggle logic in parse()
2. Add navigation recipe for automation
3. Override extraction strategy if needed

Generated with: python scripts/generate_county_handler.py "{state_name}" "{county_name}"
"""
from typing import Any, Dict, Tuple, List, TYPE_CHECKING

from playwright.sync_api import Page

from webapp.parser.Context_Integration.librarian import clean_for_json
from webapp.parser.utils.browser_utils import (
    autoscroll_until_stable,
    safe_click,
    safe_is_enabled,
    safe_is_visible,
)
from webapp.parser.utils.contest_selector import select_contest_auto_first
from webapp.parser.utils.html_scanner import scan_html_for_context
from webapp.parser.utils.logger_singleton import logger
from webapp.parser.utils.table_core import robust_table_extraction
from webapp.parser.utils.shared_logic import safe_get

if TYPE_CHECKING:
    from webapp.parser.Context_Integration.context_coordinator import ContextCoordinator


# Context cache for this county
context_cache = {{}}
accepted_buttons_cache = {{}}


def parse(
    page: Page = None,
    html_context: Dict[str, Any] = None,
    coordinator: "ContextCoordinator" = None,
    context: Dict[str, Any] = None,
    session_id: str = None,
    **kwargs,
) -> Tuple[List[str], List[Dict[str, Any]], str, Dict[str, Any]]:
    """
    {county_name} County handler for {state_name}.
    
    Workflow:
    1. Scan HTML for context and contests
    2. Select contest
    3. Perform county-specific navigation (buttons, toggles, etc.)
    4. Extract tables
    5. Return results
    
    TODO: Customize this handler for {county_name} County's specific UI.
    """
    if html_context is None:
        html_context = {{}}
    
    from webapp.parser.Context_Integration.context_coordinator import ContextCoordinator
    
    if coordinator is None:
        coordinator = ContextCoordinator()
    
    logger.info(f"[bold cyan][{county_name} County, {state_code}] Starting parse...[/bold cyan]")
    
    # === 1. Scan HTML for context ===
    context_result = scan_html_for_context(
        target_url=getattr(page, "url", None) if page else html_context.get("url"),
        page=page,
        coordinator=coordinator,
        session_id=session_id or getattr(coordinator, "session_id", None),
        allow_duplicates=getattr(coordinator, "allow_duplicates", False),
        context_cache=context_cache,
        debug=html_context.get("debug", False),
        **kwargs,
    )
    
    # Ensure location fields
    state = context_result.get("state") or "{state_code}"
    county = context_result.get("county") or "{county_name}"
    year = context_result.get("year")
    
    # Propagate to contests
    for contest in safe_get(context_result, "contests", []):
        contest.setdefault("state", state)
        contest.setdefault("county", county)
        if year:
            contest.setdefault("year", year)
        if session_id:
            contest["session_id"] = session_id
    
    # Clean and enrich
    context_result = clean_for_json(context_result)
    enriched = coordinator.organize_and_enrich(context_result)
    
    # === 2. Select contest ===
    context_for_selector = {{
        "state": state,
        "county": county,
        "year": year,
        "contests": context_result.get("contests", []),
        "session_id": session_id,
        **{{k: v for k, v in html_context.items() if k not in ("state", "county", "year", "contests")}},
    }}
    
    selected = select_contest_auto_first(
        coordinator=coordinator,
        context=context_for_selector,
        session_id=session_id,
        allow_multiple=False,
        force_interactive=html_context.get("force_interactive", False),
    )
    
    if not selected:
        logger.warning(f"[{county_name} County] No contest selected")
        return None, None, None, {{"skipped": True, "county": county}}
    
    # Handle list return
    if isinstance(selected, list) and selected:
        selected = selected[0]
    
    logger.info(f"[{county_name} County] Processing contest: {{selected.get('title')}}")
    
    # === 3. County-specific navigation ===
    # TODO: Add button toggles, navigation sequences, etc. specific to {county_name} County
    #
    # Example button toggle pattern (based on Rockland County reference):
    #
    # toggle_keywords = ["View results by district", "Show detailed results"]
    # btn, idx = coordinator.get_best_button_advanced(
    #     page=page,
    #     contest=selected,
    #     keywords=toggle_keywords,
    #     context=html_context,
    #     learning_mode=True,
    # )
    # if btn and "element_handle" in btn:
    #     element = btn["element_handle"]
    #     if safe_is_visible(element, logger) and safe_is_enabled(element, logger):
    #         safe_click(element, logger)
    #         page.wait_for_timeout(2000)  # Wait for content to load
    
    # === 4. Autoscroll if needed ===
    if page and html_context.get("autoscroll", False):
        logger.info(f"[{county_name} County] Autoscrolling to load all content...")
        autoscroll_until_stable(page, max_scrolls=10, scroll_delay=1000)
    
    # === 5. Extract tables ===
    result = robust_table_extraction(
        page=page,
        coordinator=coordinator,
        html_context={{**html_context, "selected_contest": selected}},
        session_id=session_id,
        **kwargs,
    )
    
    headers = result.get("headers", [])
    data_rows = result.get("data", [])
    
    # === 6. Build metadata ===
    metadata = {{
        "state": state,
        "county": county,
        "year": year,
        "contest_title": selected.get("title"),
        "source_url": html_context.get("url") or context_result.get("url"),
        "session_id": session_id,
        "handler": f"{{state.lower()}}.county.{snake_case_county}",
        "row_count": len(data_rows),
        "column_count": len(headers),
    }}
    
    contest_title = selected.get("title", "Unknown Contest")
    logger.info(f"[green][{county_name} County] Extracted {{len(data_rows)}} rows[/green]")
    
    return headers, data_rows, contest_title, metadata
'''


def get_navigation_recipe_template(state_code: str, county_name: str, county_snake: str) -> dict:
    """Generate navigation recipe template for this county."""
    return {
        "name": f"{state_code.lower()}_{county_snake}_default",
        "description": f"Default navigation recipe for {county_name} County, {state_code}",
        "match_conditions": {
            "state": state_code,
            "county": county_name,
            "url_contains": [
                # TODO: Add URL patterns specific to this county
                # Examples:
                # f"{county_snake.replace('_', '')}.{state_code.lower()}.gov",
                # f"{county_snake}county.gov/elections",
            ],
        },
        "steps": [
            {
                "action": "wait_for_selector",
                "selector": "body",
                "timeout": 5000,
                "description": "Wait for page to load",
            },
            # TODO: Add navigation steps specific to this county
            # Examples:
            # {
            #     "action": "click",
            #     "selector": "button:has-text('View Results')",
            #     "timeout": 3000,
            #     "description": "Click View Results button",
            # },
            # {
            #     "action": "wait_for_selector",
            #     "selector": "table.results",
            #     "timeout": 5000,
            #     "description": "Wait for results table to appear",
            # },
            {
                "action": "autoscroll",
                "max_scrolls": 10,
                "scroll_delay": 1000,
                "description": "Scroll to load all content",
            },
            {
                "action": "scan_context",
                "description": "Scan HTML for contests and context",
            },
        ],
    }


def generate_county_handler(
    state_name: str,
    county_name: str,
    navigation_recipe: bool = False,
    force: bool = False,
) -> Tuple[Path, Path]:
    """Generate county handler file and optionally navigation recipe."""
    # Get state code
    state_code = get_state_code(state_name)
    
    # Determine output paths
    project_root = Path(__file__).parent.parent
    snake_case_state = state_name.lower().replace(" ", "_")
    snake_case_county = county_name.lower().replace(" ", "_").replace("-", "_")
    
    # County handler directory
    handler_dir = (
        project_root / "webapp" / "parser" / "handlers" / "states"
        / snake_case_state / "county"
    )
    handler_dir.mkdir(parents=True, exist_ok=True)
    
    # Create __init__.py files
    (handler_dir.parent.parent / "__init__.py").touch(exist_ok=True)
    (handler_dir.parent / "__init__.py").touch(exist_ok=True)
    (handler_dir / "__init__.py").touch(exist_ok=True)
    
    handler_file = handler_dir / f"{snake_case_county}.py"
    
    # Check if exists
    if handler_file.exists() and not force:
        print(f"❌ Handler already exists: {handler_file}")
        print("   Use --force to overwrite")
        return None, None
    
    # Generate handler template
    content = get_county_template(state_name, state_code, county_name)
    handler_file.write_text(content, encoding="utf-8")
    
    # Generate navigation recipe if requested
    recipe_file = None
    if navigation_recipe:
        recipe_dir = project_root / "webapp" / "parser" / "navigator"
        recipe_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing recipes or create new
        recipes_file = recipe_dir / "navigation_recipes.orjson"
        
        if recipes_file.exists():
            import orjson
            with open(recipes_file, "rb") as f:
                recipes = orjson.loads(f.read())
        else:
            recipes = {"recipes": []}
        
        # Add new recipe
        new_recipe = get_navigation_recipe_template(state_code, county_name, snake_case_county)
        recipes["recipes"].append(new_recipe)
        
        # Save with pretty formatting
        import orjson
        with open(recipes_file, "wb") as f:
            f.write(orjson.dumps(recipes, option=orjson.OPT_INDENT_2))
        
        recipe_file = recipes_file
    
    return handler_file, recipe_file


def main():
    parser = argparse.ArgumentParser(
        description="Generate a new county-level handler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic county handler
  python scripts/generate_county_handler.py "New York" Westchester
  
  # With navigation recipe
  python scripts/generate_county_handler.py California "Los Angeles" --navigation-recipe
  
  # Using state code
  python scripts/generate_county_handler.py FL "Miami-Dade"
  
  # Overwrite existing
  python scripts/generate_county_handler.py Texas Harris --force
        """,
    )
    parser.add_argument("state", help="State name or 2-letter code")
    parser.add_argument("county", help="County name (e.g., 'Los Angeles', 'Miami-Dade')")
    parser.add_argument("--navigation-recipe", action="store_true", help="Generate navigation recipe template")
    parser.add_argument("--force", action="store_true", help="Overwrite existing handler")
    
    args = parser.parse_args()
    
    try:
        handler_file, recipe_file = generate_county_handler(
            args.state,
            args.county,
            args.navigation_recipe,
            args.force,
        )
        
        if handler_file:
            print(f"✅ Generated county handler: {handler_file}")
            print(f"   State: {args.state}")
            print(f"   County: {args.county}")
            
            if recipe_file:
                print(f"✅ Updated navigation recipes: {recipe_file}")
                print(f"   Recipe name: {get_state_code(args.state).lower()}_{args.county.lower().replace(' ', '_').replace('-', '_')}_default")
            
            print("\n📝 Next steps:")
            print(f"   1. Review and customize: {handler_file}")
            print("   2. Add county-specific navigation logic (button toggles, etc.)")
            
            if recipe_file:
                print(f"   3. Update navigation recipe with actual selectors: {recipe_file}")
            
            print("   4. Test with: python -m webapp.parser.html_election_parser --url <county-url>")
        
        return 0
    
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
