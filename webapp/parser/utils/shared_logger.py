from ..utils.logger_instance import logger
from rich import print as rprint
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn

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
    """General alert for unexpected or special-case events."""
    style = {
        "info": "cyan",
        "warning": "yellow",
        "error": "red",
        "critical": "magenta"
    }.get(alert_type, "cyan")
    rprint(Panel(f"[bold]{alert_type.upper()} ALERT:[/bold] {msg}\n[dim]{context}[/dim]", style=style))

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