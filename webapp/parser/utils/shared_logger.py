import logging
import os
import re
from pathlib import Path
from rich.logging import RichHandler
from rich import print as rprint
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn
import orjson

# --- Logger Setup ---
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

# Now configure logging with the correct level from .env
logging.basicConfig(
    level=level_mapping.get(LOG_LEVEL, logging.INFO),
    format='[%(levelname)s] %(message)s'
)

# Singleton logger instance
_logger_instance = None

def get_shared_logger(name="smart_elections"):
    global _logger_instance
    if _logger_instance is not None:
        return _logger_instance
    logging.basicConfig(
        level=level_mapping.get(LOG_LEVEL, logging.INFO),
        format='[%(levelname)s] %(message)s',
        handlers=[RichHandler(rich_tracebacks=True)]
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_mapping.get(LOG_LEVEL, logging.INFO))
    _logger_instance = logger
    return logger

logger = get_shared_logger()

# --- Rich Logging Helpers ---
def contains_rich_markup(msg):
    # Detects [red], [bold], [INFO], [ERROR], etc. at the start or inside the message
    return bool(re.search(r"\[[a-zA-Z0-9_]+\]", msg))
def extract_label_and_color(msg):
    """
    Detects a leading [LABEL] or [color] and returns (label, color, msg_without_label).
    Supports [INFO], [ERROR], [WARNING], [DEBUG], [CRITICAL], [red], [green], etc.
    """
    match = re.match(r"^\[([A-Z]+|[a-z]+)\]\s*(.*)", msg)
    if match:
        label = match.group(1)
        rest = match.group(2)
        # Map common log labels to colors
        color_map = {
            "INFO": "green",
            "DEBUG": "blue",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "magenta",
            "ALERT": "cyan",
            "TRACE": "cyan",
            # Rich colors
            "red": "red",
            "green": "green",
            "yellow": "yellow",
            "blue": "blue",
            "magenta": "magenta",
            "cyan": "cyan",
            "bold": "white",
        }
        color = color_map.get(label, None)
        return label, color, rest
    return None, None, msg

def log_trace(msg, context=None, *args, **kwargs):
    if logger.isEnabledFor(5):
        logger.log(5, msg, *args, **kwargs)
    _rich_log(msg, context, default_label="TRACE", default_color="cyan")

def log_debug(msg, context=None, *args, **kwargs):
    logger.debug(msg, *args, **kwargs)
    _rich_log(msg, context, default_label="DEBUG", default_color="blue")

def log_info(msg, context=None, *args, **kwargs):
    logger.info(msg, *args, **kwargs)
    _rich_log(msg, context, default_label="INFO", default_color="green")

def log_warning(msg, context=None, *args, **kwargs):
    logger.warning(msg, *args, **kwargs)
    _rich_log(msg, context, default_label="WARNING", default_color="yellow")

def log_error(msg, context=None, *args, **kwargs):
    logger.error(msg, *args, **kwargs)
    _rich_log(msg, context, default_label="ERROR", default_color="red")

def log_critical(msg, context=None, *args, **kwargs):
    logger.critical(msg, *args, **kwargs)
    _rich_log(msg, context, default_label="CRITICAL", default_color="magenta")

def log_alert(msg, context=None, alert_type="info"):
    style = {
        "info": "cyan",
        "warning": "yellow",
        "error": "red",
        "critical": "magenta"
    }.get(alert_type, "cyan")
    label = alert_type.upper()
    if context:
        rprint(Panel(f"[bold]{label} ALERT:[/bold] {msg}\n[dim]{context}[/dim]", style=style))
    elif contains_rich_markup(msg):
        rprint(msg)
    else:
        rprint(Panel(f"[bold]{label} ALERT:[/bold] {msg}", style=style))

def _rich_log(msg, context=None, default_label=None, default_color=None):
    """
    Advanced rich log rendering: detects [LABEL] or [color] and uses it for the panel.
    If rich markup is present, prints as-is. Otherwise, wraps in a colored panel.
    """
    if context:
        # Always show context in a panel with color
        label, color, msg_body = extract_label_and_color(msg)
        panel_label = label or default_label
        panel_color = color or default_color or "white"
        rprint(Panel(f"[bold {panel_color}]{panel_label}:[/bold {panel_color}] {msg_body}\n[dim]{context}[/dim]", style=panel_color))
    elif contains_rich_markup(msg):
        rprint(msg)
    else:
        # No markup, no context: use default label/color
        label = default_label or "LOG"
        color = default_color or "white"
        rprint(Panel(f"[bold {color}]{label}:[/bold {color}] {msg}", style=color))       

# --- Progress Bar Helper ---
def get_progress_bar(description="Processing", total=100):
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40, style="bold blue"),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        expand=True,
        transient=True
    )

# --- Log File Discovery and Safe JSONL Reading ---
def discover_log_files(log_dirs, suffixes=(".jsonl", ".json", ".log", ".html")):
    """Discover all log files in the given directories with the specified suffixes."""
    files = []
    for log_dir in log_dirs:
        p = Path(log_dir)
        if not p.exists():
            logger.warning(f"Log directory does not exist: {log_dir}")
            continue
        for suf in suffixes:
            files.extend(p.glob(f"*{suf}"))
    return files

def safe_read_jsonl(path):
    """Read a JSONL file, handling errors gracefully."""
    entries = []
    try:
        with open(path, "rb") as f:
            for line in f:
                try:
                    entries.append(orjson.loads(line))
                except Exception as e:
                    logger.warning(f"Corrupt line in {path}: {e}")
    except Exception as e:
        logger.error(f"Failed to read {path}: {e}")
    return entries