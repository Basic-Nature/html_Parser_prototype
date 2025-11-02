"""Interpreter-level compatibility patches for the Smart Elections Parser.

This file is discovered automatically by Python during startup (when the
project root is on ``sys.path``) and lets us apply safe shims before any third
party modules are imported. Use this sparingly to keep surprises to a minimum.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any, Callable


def _alias_click_split_arg_string() -> None:
    """Expose ``click.parser.split_arg_string`` without triggering deprecation.

    Click 8.2 moved ``split_arg_string`` to ``click.shell_completion`` and now
    raises a ``DeprecationWarning`` when legacy imports access the attribute on
    ``click.parser``. spaCy/weasel still reach for the legacy location, so we
    install a direct alias ahead of time. This mirrors Click's own fallback but
    avoids the warning and keeps the behaviour identical.
    """

    try:
        parser_mod = import_module("click.parser")
        shell_completion = import_module("click.shell_completion")
    except Exception:
        return

    alias: Callable[[str], list[str]] | None = getattr(
        shell_completion, "split_arg_string", None
    )
    if alias is None:
        return

    # Assign unconditionally so that downstream imports find the attribute on
    # the module dict and never trigger the deprecated fallback pathway.
    setattr(parser_mod, "split_arg_string", alias)


_alias_click_split_arg_string()
