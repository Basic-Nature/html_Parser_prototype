from __future__ import annotations

# webapp/parser/utils/prompt_singleton.py
# ---------------------------------------------------------------
# Explicit owner of the process-wide UserPrompt instance.
#
# Importing logger_singleton alone must not initialize prompt/session
# machinery. Prompt initialization occurs only when this module is
# explicitly imported or lazily requested through logger_singleton.
# ---------------------------------------------------------------

from typing import Any


try:
    from .user_prompt import UserPrompt

    prompt: Any = UserPrompt()

except Exception:
    # Preserve historical optional-prompt behavior for reduced
    # runtimes where interactive dependencies cannot initialize.
    prompt = None


def get_prompt() -> Any:
    """Return the process-wide interactive prompt singleton."""

    return prompt
