import logging
import os
from rich import print as rprint
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn
from ..Context_Integration.context_organizer import append_to_context_library

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
        # Optionally append to context library for learning
        append_to_context_library({"alerts": [{"level": "warning", "msg": msg, "context": context}]})

def log_error(msg, context=None):
    logger.error(msg)
    if context:
        rprint(Panel(f"[bold red]ERROR:[/bold red] {msg}\n[dim]{context}[/dim]", style="red"))
        append_to_context_library({"alerts": [{"level": "error", "msg": msg, "context": context}]})

def log_critical(msg, context=None):
    logger.critical(msg)
    if context:
        rprint(Panel(f"[bold magenta]CRITICAL:[/bold magenta] {msg}\n[dim]{context}[/dim]", style="magenta"))
        append_to_context_library({"alerts": [{"level": "critical", "msg": msg, "context": context}]})

def log_alert(msg, context=None, alert_type="info"):
    """General alert for unexpected or special-case events."""
    style = {
        "info": "cyan",
        "warning": "yellow",
        "error": "red",
        "critical": "magenta"
    }.get(alert_type, "cyan")
    rprint(Panel(f"[bold]{alert_type.upper()} ALERT:[/bold] {msg}\n[dim]{context}[/dim]", style=style))
    append_to_context_library({"alerts": [{"level": alert_type, "msg": msg, "context": context}]})

# Stylized progress bar for data flow
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

# Example usage in other scripts:
# with get_progress_bar("Downloading files", total=len(files)) as progress:
#     task = progress.add_task("Downloading...", total=len(files))
#     for file in files:
#         ... # download logic
#         progress.update(task, advance=1)