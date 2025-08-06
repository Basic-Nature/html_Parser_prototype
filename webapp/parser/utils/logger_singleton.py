# webapp/parser/utils/logger_singleton.py
# ---------------------------------------------------------------
# Shared instance for logging and prompting in Smart Elections Parser Webapp
# ---------------------------------------------------------------
from __future__ import annotations
from .shared_logger import SharedLogger, RichConsoleProxy

logger = SharedLogger()
console = RichConsoleProxy(logger=logger)

from .user_prompt import UserPrompt
prompt = UserPrompt()