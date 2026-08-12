from __future__ import annotations

# webapp/parser/utils/logger_singleton.py
# ---------------------------------------------------------------
# Shared, import-safe logger/console singletons.
#
# Logging bootstrap is intentionally independent from interactive
# prompt/session initialization.
#
# Existing callers may continue importing ``prompt`` from this
# module. That compatibility attribute is resolved lazily through
# prompt_singleton.py.
#
# LOG_LEVEL is read directly from the environment so this low-level
# bootstrap module does not need to import config.py.
# ---------------------------------------------------------------

import importlib
import os
from typing import Any

from .shared_logger import RichConsoleProxy, SharedLogger


LOG_LEVEL = (
    os.environ.get("LOG_LEVEL", "INFO")
    .split(",")[0]
    .strip()
    .upper()
)


logger = SharedLogger(level=LOG_LEVEL)
console = RichConsoleProxy(logger=logger)


def set_log_level(level: str) -> None:
    """Update the process-wide shared logger level."""
    logger.set_level(level)


def get_shared_logger() -> SharedLogger:
    """Return the process-wide shared logger."""
    return logger


def get_prompt() -> Any:
    """Resolve and return the interactive prompt singleton lazily."""

    package = __package__ or "webapp.parser.utils"

    module = importlib.import_module(
        f"{package}.prompt_singleton"
    )

    return module.prompt


def __getattr__(name: str) -> Any:
    """Provide lazy compatibility for ``logger_singleton.prompt``."""

    if name == "prompt":
        value = get_prompt()

        globals()["prompt"] = value

        return value

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


def __dir__() -> list[str]:
    """Include the lazy compatibility attribute in introspection."""

    return sorted(
        set(globals()) | {"prompt"}
    )
