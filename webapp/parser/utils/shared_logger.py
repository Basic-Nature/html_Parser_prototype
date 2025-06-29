import logging
import os
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
def log_trace(msg, *args, **kwargs):
    if logger.isEnabledFor(5):
        logger.log(5, msg, *args, **kwargs)

def log_debug(msg, context=None):
    logger.debug(msg)
    if context:
        rprint(Panel(f"[bold blue]DEBUG:[/bold blue] {msg}\n[dim]{context}[/dim]", style="blue"))

def log_info(msg, context=None):
    logger.info(msg)
    if context:
        rprint(Panel(f"[bold green]INFO:[/bold green] {msg}\n[dim]{context}[/dim]", style="green"))

def log_warning(msg, context=None):
    logger.warning(msg)
    if context:
        rprint(Panel(f"[bold yellow]WARNING:[/bold yellow] {msg}\n[dim]{context}[/dim]", style="yellow"))

def log_error(msg, context=None):
    logger.error(msg)
    if context:
        rprint(Panel(f"[bold red]ERROR:[/bold red] {msg}\n[dim]{context}[/dim]", style="red"))

def log_critical(msg, context=None):
    logger.critical(msg)
    if context:
        rprint(Panel(f"[bold magenta]CRITICAL:[/bold magenta] {msg}\n[dim]{context}[/dim]", style="magenta"))

def log_alert(msg, context=None, alert_type="info"):
    style = {
        "info": "cyan",
        "warning": "yellow",
        "error": "red",
        "critical": "magenta"
    }.get(alert_type, "cyan")
    rprint(Panel(f"[bold]{alert_type.upper()} ALERT:[/bold] {msg}\n[dim]{context}[/dim]", style=style))

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