"""Navigation recipe executor for shared handler workflows."""
from __future__ import annotations

from typing import Any, Dict, List

from webapp.parser.utils.browser_utils import autoscroll_until_stable
from webapp.parser.utils.logger_singleton import logger


def run_navigation_recipe(
    page: Any,
    contest: Dict[str, Any],
    html_context: Dict[str, Any],
    recipe: List[Dict[str, Any]],
    coordinator: Any,
    session_id: str | None = None,
) -> Dict[str, Any]:
    """Execute a navigation recipe for an enhanced voting portal."""
    if page is None or not recipe:
        return {"steps": []}

    contest_title = str(contest.get("title", "") or "")
    results: List[Dict[str, Any]] = []

    for step in recipe:
        action = step.get("action")
        step_result = {"action": action, "success": False}

        if action == "select_contest":
            from webapp.parser.handlers.shared.vendors.enhanced_voting import click_enhancedvoting_contest

            label_pattern = step.get("label_pattern")
            step_result["success"] = click_enhancedvoting_contest(
                page=page,
                contest_title=contest_title,
                label_pattern=label_pattern,
            )

        elif action == "toggle_vote_method":
            from webapp.parser.handlers.shared.vendors.enhanced_voting import click_toggle_by_keywords

            keywords = step.get("keywords", [])
            step_result["success"] = click_toggle_by_keywords(page=page, keywords=keywords)

        elif action == "autoscroll_until_stable":
            try:
                autoscroll_until_stable(page, session_id=session_id)
                step_result["success"] = True
            except Exception as e:
                logger.warning(
                    f"[NavigationRecipe] autoscroll failed: {e}",
                )

        else:
            logger.debug(f"[NavigationRecipe] Unsupported action: {action}")

        results.append(step_result)

    return {"steps": results}
