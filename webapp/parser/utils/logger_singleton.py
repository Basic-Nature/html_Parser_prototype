from __future__ import annotations
# webapp/parser/utils/logger_singleton.py
# ---------------------------------------------------------------
# Shared instance for logging and prompting in Smart Elections Parser Webapp
# ---------------------------------------------------------------
from .shared_logger import SharedLogger, RichConsoleProxy
from ..config import LOG_LEVEL

logger = SharedLogger(level=LOG_LEVEL)
console = RichConsoleProxy(logger=logger)

from .user_prompt import UserPrompt
prompt = UserPrompt()