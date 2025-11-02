from __future__ import annotations

# webapp/parser/utils/logger_singleton.py
# ---------------------------------------------------------------
# Shared, import-safe logger/console singletons.
# Reads LOG_LEVEL from env to avoid circular import with config.py.
# ---------------------------------------------------------------
import os

from .shared_logger import RichConsoleProxy, SharedLogger

# Resolve log level from environment (no config.py import)
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").split(",")[0].strip().upper()

# Singleton instances
logger = SharedLogger(level=LOG_LEVEL)
console = RichConsoleProxy(logger=logger)

# Optional helpers
def set_log_level(level: str) -> None:
    logger.set_level(level)

def get_shared_logger() -> SharedLogger:
    return logger

# Optional prompt singleton (if used by UI)
try:
    from .user_prompt import UserPrompt
    prompt = UserPrompt()
except Exception:
    prompt = None