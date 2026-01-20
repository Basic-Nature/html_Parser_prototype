"""Dynamic navigation recipes for Smart Elections Parser.

Exposes the navigation runner and recipe store so handlers and the
HTML pipeline can execute context-aware instruction sets loaded from
`navigation_recipes.orjson`.
"""

from .navigation_recipes import NavigationRecipeStore, DEFAULT_RECIPE_PATH
from .navigation_runner import NavigationInstructionRunner

__all__ = [
    "NavigationInstructionRunner",
    "NavigationRecipeStore",
    "DEFAULT_RECIPE_PATH",
]
