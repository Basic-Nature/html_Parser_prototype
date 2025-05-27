import logging
import os
from rich.logging import RichHandler


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
level_mapping = {
    "TRACE": 5,  # Custom trace level
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}
logging.addLevelName(5, "TRACE")

logging.basicConfig(
    level=level_mapping.get(LOG_LEVEL, logging.INFO),
    format='[%(levelname)s] %(message)s',
    handlers=[RichHandler()]
)
logger = logging.getLogger("smart_elections")
logger.setLevel(level_mapping.get(LOG_LEVEL, logging.INFO))