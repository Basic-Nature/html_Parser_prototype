import logging
import os
import re
import time
from pathlib import Path
from rich.logging import RichHandler
from rich import print as rprint
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn
import orjson

def summarize_logs(log_path=None, max_lines=1000):
    """
    Return the last max_lines of the log file as a string.
    """
    log_path = log_path or "pipeline.log"
    if not os.path.exists(log_path):
        return ""
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return "".join(lines[-max_lines:])

SUPPRESS_RICH_LOGS = False
SOCKETIO_EMIT_FUNC = None
LOG_MODE = "cli"  # or "webapp"
LOG_FORMAT = "plain"  # or "json"

def set_log_mode(mode):
    global LOG_MODE
    LOG_MODE = mode

def set_log_format(fmt):
    global LOG_FORMAT
    LOG_FORMAT = fmt

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
    format='[%(levelname)s] %(message)s'
)

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

def set_socketio_emit_func(emit_func):
    global SOCKETIO_EMIT_FUNC
    SOCKETIO_EMIT_FUNC = emit_func

def contains_rich_markup(msg):
    return bool(re.search(r"\[[a-zA-Z0-9_]+\]", msg))

def extract_label_and_color(msg):
    match = re.match(r"^\[([a-zA-Z0-9_ ]+)\]\s*(.*)", msg)
    if match:
        label = match.group(1)
        rest = match.group(2)
        if " " in label:
            *style_parts, color = label.split()
            style = " ".join(style_parts)
        else:
            style = None
            color = label
        color_map = {
            "INFO": "green",
            "DEBUG": "blue",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "magenta",
            "ALERT": "cyan",
            "TRACE": "cyan",
            "red": "red",
            "green": "green",
            "yellow": "yellow",
            "blue": "blue",
            "magenta": "magenta",
            "cyan": "cyan",
            "bold": "white",
        }
        panel_color = color_map.get(color.upper(), color)
        panel_label = label
        return panel_label, panel_color, rest
    return None, None, msg

def _rich_log(msg, context=None, default_label=None, default_color=None):
    """
    Unified log rendering for CLI and webapp.
    - In CLI: uses rich panels and colors.
    - In webapp: emits plain text or structured JSON to SocketIO.
    """
    if SUPPRESS_RICH_LOGS:
        return

    # Compose the message and context
    if context:
        label, color, msg_body = extract_label_and_color(msg)
        panel_label = label or default_label or "LOG"
        panel_color = color or default_color or "white"
        text_msg = f"[{panel_label}] {msg_body}\n{context}"
    elif contains_rich_markup(msg):
        text_msg = msg
        panel_color = default_color or "white"
        panel_label = default_label or "LOG"
    else:
        panel_label = default_label or "LOG"
        panel_color = default_color or "white"
        text_msg = f"[{panel_label}] {msg}"

    # Output logic
    if LOG_MODE == "webapp" and SOCKETIO_EMIT_FUNC:
        if LOG_FORMAT == "json":
            log_obj = {
                "timestamp": time.time(),
                "level": panel_label,
                "color": panel_color,
                "message": re.sub(r"\[/?[a-zA-Z0-9_ ]+\]", "", text_msg).strip(),
                "context": context,
            }
            SOCKETIO_EMIT_FUNC(orjson.dumps(log_obj).decode("utf-8"))
        else:
            plain_msg = re.sub(r"\[/?[a-zA-Z0-9_ ]+\]", "", text_msg)
            SOCKETIO_EMIT_FUNC(plain_msg.strip())
    elif LOG_MODE == "cli":
        rprint(Panel(text_msg, style=panel_color))

# --- Unified log entry points ---

def log_trace(msg, context=None, *args, **kwargs):
    if SUPPRESS_RICH_LOGS:
        return
    if LOG_MODE == "webapp" and SOCKETIO_EMIT_FUNC:
        _rich_log(msg, context, default_label="TRACE", default_color="cyan")
        return
    _rich_log(msg, context, default_label="TRACE", default_color="cyan")

def log_debug(msg, context=None, *args, **kwargs):
    if SUPPRESS_RICH_LOGS:
        return
    if LOG_MODE == "webapp" and SOCKETIO_EMIT_FUNC:
        _rich_log(msg, context, default_label="DEBUG", default_color="blue")
        return
    _rich_log(msg, context, default_label="DEBUG", default_color="blue")

def log_info(msg, context=None, *args, **kwargs):
    if SUPPRESS_RICH_LOGS:
        return
    if LOG_MODE == "webapp" and SOCKETIO_EMIT_FUNC:
        _rich_log(msg, context, default_label="INFO", default_color="green")
        return
    _rich_log(msg, context, default_label="INFO", default_color="green")

def log_warning(msg, context=None, *args, **kwargs):
    if SUPPRESS_RICH_LOGS:
        return
    if LOG_MODE == "webapp" and SOCKETIO_EMIT_FUNC:
        _rich_log(msg, context, default_label="WARNING", default_color="yellow")
        return
    _rich_log(msg, context, default_label="WARNING", default_color="yellow")

def log_error(msg, context=None, *args, **kwargs):
    if SUPPRESS_RICH_LOGS:
        return
    if LOG_MODE == "webapp" and SOCKETIO_EMIT_FUNC:
        _rich_log(msg, context, default_label="ERROR", default_color="red")
        return
    _rich_log(msg, context, default_label="ERROR", default_color="red")

def log_critical(msg, context=None, *args, **kwargs):
    if SUPPRESS_RICH_LOGS:
        return
    if LOG_MODE == "webapp" and SOCKETIO_EMIT_FUNC:
        _rich_log(msg, context, default_label="CRITICAL", default_color="magenta")
        return
    _rich_log(msg, context, default_label="CRITICAL", default_color="magenta")

def log_alert(msg, context=None, alert_type_="info"):
    if SUPPRESS_RICH_LOGS:
        return
    style = {
        "info": "cyan",
        "warning": "yellow",
        "error": "red",
        "critical": "magenta"
    }.get(alert_type_, "cyan")
    label = alert_type_.upper()
    panel_msg = None
    if context:
        panel_msg = f"[bold]{label} ALERT:[/bold] {msg}\n[dim]{context}[/dim]"
    else:
        panel_msg = f"[bold]{label} ALERT:[/bold] {msg}"
    if LOG_MODE == "webapp" and SOCKETIO_EMIT_FUNC:
        if LOG_FORMAT == "json":
            log_obj = {
                "timestamp": time.time(),
                "level": label,
                "color": style,
                "message": re.sub(r"\[/?[a-zA-Z0-9_ ]+\]", "", msg).strip(),
                "context": context,
            }
            SOCKETIO_EMIT_FUNC(orjson.dumps(log_obj).decode("utf-8"))
        else:
            plain_msg = re.sub(r"\[/?[a-zA-Z0-9_ ]+\]", "", panel_msg)
            SOCKETIO_EMIT_FUNC(plain_msg.strip())
    elif LOG_MODE == "cli":
        rprint(Panel(panel_msg, style=style))

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