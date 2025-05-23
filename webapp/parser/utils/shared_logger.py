import logging
import os
# Add rprint for visual CLI feedback via rich
from rich import print as rprint
from rich.logging import RichHandler

# Read log level from .env or default to INFO
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Convert to logging level
level_mapping = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}


logging.basicConfig(
    level=level_mapping.get(LOG_LEVEL, logging.INFO),
    format='[%(levelname)s] %(message)s',
    handlers=[RichHandler()]
)


logger = logging.getLogger("smart_elections")
# Set the logger to use the RichHandler for better formatting
logger.setLevel(level_mapping.get(LOG_LEVEL, logging.INFO))
# Set up a custom handler to use Rich for logging
# Add a RichHandler to the logger
# logger.addHandler(RichHandler())
# Set the logger to use the RichHandler for better formatting
# logger.setLevel(level_mapping.get(LOG_LEVEL, logging.INFO))


# Shared logging functions

def log_debug(msg):
    logger.debug(msg)

def log_info(msg):
    logger.info(msg)

def log_warning(msg):
    logger.warning(msg)

def log_error(msg):
    logger.error(msg)

def log_critical(msg):
    logger.critical(msg)
    
    



