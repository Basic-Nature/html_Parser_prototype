import os
import re
import time
import logging
import inspect
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, Generator, ContextManager, Tuple
from rich import print as rprint
from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn
from rich.text import Text
from rich.markdown import Markdown
from rich.rule import Rule
import orjson
import json
from contextlib import contextmanager
from io import StringIO

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def get_log_mode() -> str:
    """Get log mode from environment or default to CLI."""
    return os.environ.get("LOG_MODE", os.environ.get("PROMPT_MODE", "cli")).lower()

def get_log_format() -> str:
    """Get log format from environment or default to plain."""
    return os.environ.get("LOG_FORMAT", "plain").lower()

def get_log_level() -> str:
    """Get log level from environment or default to INFO."""
    return os.environ.get("LOG_LEVEL", "INFO").upper()

class RichConsoleProxy:
    """
    Proxy class to provide a Console-like API using SharedLogger.
    Use this as a drop-in replacement for rich.console.Console.
    Handles routing of rich output (Table, Panel, Progress, etc.) to CLI or webapp (SocketIO).
    """
    def __init__(self, logger=None):
        from .shared_logger import _logger_instance as logger_instance
        self.logger = logger or logger_instance
        self._console = Console()

    def print(self, *args, **kwargs):
        """
        Print using rich's Console.print (for tables, panels, etc).
        In webapp mode, serialize the output and emit via SocketIO.
        """
        # If in webapp mode and socketio_emit_func is set, serialize and emit
        if getattr(self.logger, "mode", "cli") == "webapp" and getattr(self.logger, "socketio_emit_func", None):
            sio = self.logger.socketio_emit_func
            for obj in args:
                # Try to render as text
                sio(self._render_to_text(obj))
        else:
            self._console.print(*args, **kwargs)

    def _render_to_text(self, obj: RenderableType) -> str:
        """
        Render a rich object (Table, Panel, Progress, etc.) to plain text for webapp streaming.
        """
        # Use a temporary Console to capture output as string
        temp_console = Console(file=StringIO(), force_terminal=True, color_system=None, width=100)
        temp_console.print(obj)
        output = temp_console.file.getvalue()
        return output

    def rule(self, title: str = "", **kwargs):
        """Draw a rule (horizontal line) with optional title."""
        if getattr(self.logger, "mode", "webapp") and getattr(self.logger, "socketio_emit_func", None):
            sio = self.logger.socketio_emit_func
            sio("-" * 40 + f" {title} " + "-" * 40)
        else:
            self._console.rule(title, **kwargs)

    def panel(self, msg: str, title: str = "", style: str = "blue"):
        """Print a message in a rich Panel."""
        panel = Panel(msg, title=title, style=style)
        self.print(panel)

    def input(self, prompt: str = "") -> str:
        """Proxy for input, using rich's Console.input."""
        return self._console.input(prompt)

    def log(self, *args, **kwargs):
        """Log a message using rich's Console.log (for debug/info output)."""
        if getattr(self.logger, "mode", "webapp") and getattr(self.logger, "socketio_emit_func", None):
            sio = self.logger.socketio_emit_func
            for obj in args:
                sio(self._render_to_text(obj))
        else:
            self._console.log(*args, **kwargs)

# Usage:
# from .shared_logger import RichConsoleProxy
# console = RichConsoleProxy()
# console.print(table)  # Will route to SocketIO as text in webapp mode
# console.panel("Hello", title="Greeting")
# console.rule("Section")

class SharedLogger:
    """
    Unified logger supporting CLI/webapp and plain/json/file output.
    Mode, format, and level can be set via .env or at init.
    """

    # Compiled regex patterns for performance
    RICH_MARKUP_RE = re.compile(r"\[[a-zA-Z0-9_]+\]")
    LABEL_COLOR_RE = re.compile(r"^\[([a-zA-Z0-9_ ]+)\]\s*(.*)")

    _instance = None  # Singleton

    def __new__(cls, *args, **kwargs):
        # Singleton pattern (optional, can be removed for multi-instance)
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        mode: Optional[str] = None,
        fmt: Optional[str] = None,
        level: Optional[str] = None,
        socketio_emit_func: Optional[Callable[[str], None]] = None,
        suppress_rich_logs: bool = False,
        file_path: Optional[str] = None,
    ):
        """
        Initialize the logger.
        """
        self.mode = mode or get_log_mode()
        self.format = fmt or get_log_format()
        self.level = (level or get_log_level()).upper()
        self.socketio_emit_func = socketio_emit_func
        self.suppress_rich_logs = suppress_rich_logs
        self.file_path = file_path
        self._setup_python_logger()

    def _setup_python_logger(self) -> None:
        """Set up the internal Python logger with RichHandler."""
        self.level_mapping = {
            "TRACE": 5,
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        logging.addLevelName(5, "TRACE")
        self.logger = logging.getLogger("smart_elections")
        self.logger.setLevel(self.level_mapping.get(self.level, logging.INFO))
        # Remove all handlers before adding a new one (avoid duplication)
        self.logger.handlers.clear()
        from rich.logging import RichHandler
        handler = RichHandler(rich_tracebacks=True)
        self.logger.addHandler(handler)
        # Optional: add file handler if file_path is set
        if self.file_path:
            file_handler = logging.FileHandler(self.file_path, encoding="utf-8")
            file_handler.setLevel(self.level_mapping.get(self.level, logging.INFO))
            formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def set_mode(self, mode: Optional[str] = None) -> None:
        """Set the logger mode (cli/webapp)."""
        self.mode = mode or get_log_mode()

    def set_format(self, fmt: Optional[str] = None) -> None:
        """Set the logger format (plain/json)."""
        self.format = fmt or get_log_format()

    def set_level(self, level: Optional[str] = None) -> None:
        """Set the logger level."""
        self.level = (level or get_log_level()).upper()
        self.logger.setLevel(self.level_mapping.get(self.level, logging.INFO))

    def set_socketio_emit_func(self, emit_func: Callable[[str], None]) -> None:
        """Set the function to emit logs via socketio (for webapp mode)."""
        self.socketio_emit_func = emit_func

    def set_file_path(self, file_path: str) -> None:
        """Set the file path for file logging."""
        self.file_path = file_path
        self._setup_python_logger()

    def suppress(self, value: bool = True) -> None:
        """Suppress rich logs (for testing or silent mode)."""
        self.suppress_rich_logs = value

    def _contains_rich_markup(self, msg: str) -> bool:
        """Check if message contains rich markup."""
        return bool(self.RICH_MARKUP_RE.search(msg))

    def _extract_label_and_color(self, msg: str) -> Tuple[Optional[str], Optional[str], str]:
        """Extract label and color from message."""
        match = self.LABEL_COLOR_RE.match(msg)
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

    def _should_emit(self, level: str) -> bool:
        """Check if the log should be emitted based on the current log level."""
        return self.level_mapping.get(level, 100) >= self.level_mapping.get(self.level, 20)

    def _format_context(self, context: Any) -> str:
        """Format context for output."""
        if context is None:
            return ""
        if isinstance(context, dict):
            try:
                return json.dumps(context, indent=2, ensure_ascii=False)
            except Exception:
                return str(context)
        return str(context)

    def _get_caller_info(self) -> Dict[str, Any]:
        """Get structured caller info for advanced/structured logging."""
        frame = inspect.currentframe()
        if frame is not None:
            outer = inspect.getouterframes(frame, 3)
            if len(outer) > 3:
                caller = outer[3]
                return {
                    "module": caller.frame.f_globals.get("__name__", ""),
                    "function": caller.function,
                    "line": caller.lineno
                }
        return {}

    def _emit(self, level: str, msg: str, context: Any = None, color: str = "white") -> None:
        """
        Emit a log message to the appropriate destination(s).
        """
        if self.suppress_rich_logs or not self._should_emit(level):
            return

        # Compose the message and context
        context_str = self._format_context(context)
        if context_str:
            label, panel_color, msg_body = self._extract_label_and_color(msg)
            panel_label = label or level
            panel_color = panel_color or color
            text_msg = f"[{panel_label}] {msg_body}\n{context_str}"
        elif self._contains_rich_markup(msg):
            text_msg = msg
            panel_color = color
            panel_label = level
        else:
            panel_label = level
            panel_color = color
            text_msg = f"[{panel_label}] {msg}"

        # Structured logging info
        caller_info = self._get_caller_info()

        # Output logic
        if self.mode == "webapp" and self.socketio_emit_func:
            # --- Webapp: emit to socketio and log to Python logger ---
            if self.format == "json":
                log_obj = {
                    "timestamp": time.time(),
                    "level": panel_label,
                    "color": panel_color,
                    "message": re.sub(r"\[/?[a-zA-Z0-9_ ]+\]", "", text_msg).strip(),
                    "context": context_str,
                    **caller_info
                }
                self.socketio_emit_func(orjson.dumps(log_obj).decode("utf-8"))
            else:
                plain_msg = re.sub(r"\[/?[a-zA-Z0-9_ ]+\]", "", text_msg)
                self.socketio_emit_func(plain_msg.strip())
            # Also log to Python logger (RichHandler)
            if hasattr(self.logger, level.lower()):
                getattr(self.logger, level.lower())(re.sub(r"\[/?[a-zA-Z0-9_ ]+\]", "", text_msg).strip())
            else:
                self.logger.info(re.sub(r"\[/?[a-zA-Z0-9_ ]+\]", "", text_msg).strip())
        elif self.mode == "cli":
            # --- CLI: only print Rich panel, do NOT log to Python logger ---
            if not isinstance(text_msg, str):
                text_msg = str(text_msg)
            if not isinstance(panel_color, str):
                panel_color = str(panel_color)
            rprint(Panel(text_msg, style=panel_color))

        # File output (optional, always logs to file if file_path is set)
        if self.file_path:
            log_line = {
                "timestamp": time.time(),
                "level": panel_label,
                "color": panel_color,
                "message": re.sub(r"\[/?[a-zA-Z0-9_ ]+\]", "", text_msg).strip(),
                "context": context_str,
                **caller_info
            }
            with open(self.file_path, "a", encoding="utf-8") as f:
                if self.format == "json":
                    f.write(orjson.dumps(log_line).decode("utf-8") + "\n")
                else:
                    f.write(f"{log_line['timestamp']} [{log_line['level']}] {log_line['message']}\n")

    # --- Logging methods ---
    def trace(self, msg: str, context: Any = None) -> None:
        """Log a trace message."""
        self._emit("TRACE", msg, context, color="cyan")

    def debug(self, msg: str, context: Any = None) -> None:
        """Log a debug message."""
        self._emit("DEBUG", msg, context, color="blue")

    def info(self, msg: str, context: Any = None) -> None:
        """Log an info message."""
        self._emit("INFO", msg, context, color="green")

    def warning(self, msg: str, context: Any = None) -> None:
        """Log a warning message."""
        self._emit("WARNING", msg, context, color="yellow")

    def error(self, msg: str, context: Any = None) -> None:
        """Log an error message."""
        self._emit("ERROR", msg, context, color="red")

    def critical(self, msg: str, context: Any = None) -> None:
        """Log a critical message."""
        self._emit("CRITICAL", msg, context, color="magenta")

    def alert(self, msg: str, context: Any = None, alert_type: str = "info") -> None:
        """Log an alert message with a specific alert type."""
        style = {
            "info": "cyan",
            "warning": "yellow",
            "error": "red",
            "critical": "magenta"
        }.get(alert_type, "cyan")
        label = alert_type.upper()
        panel_msg = f"[bold]{label} ALERT:[/bold] {msg}"
        if context:
            panel_msg += f"\n[dim]{self._format_context(context)}[/dim]"
        self._emit(label, panel_msg, context, color=style)

    # --- Progress Bar Helper ---
    @contextmanager
    def progress_bar(
        self,
        description: str = "Processing",
        total: int = 100,
        emit_interval: float = 0.5,
        progress_callback: Optional[Callable[[dict], None]] = None,
    ) -> Generator[Any, None, None]:
        """
        Context manager for a rich progress bar.
        In CLI mode: shows a live progress bar.
        In webapp mode: emits progress updates via SocketIO as JSON.
        Optionally, accepts a progress_callback for custom handling.
        Usage:
            with logger.progress_bar("Processing", total=100) as update_progress:
                for i in range(total):
                    # ... your work ...
                    update_progress(i + 1)
        """
        if self.mode == "webapp" and self.socketio_emit_func:
            last_emit = 0

            def update_progress(completed: int, extra: Optional[dict] = None):
                now = time.time()
                if now - update_progress.last_emit >= emit_interval or completed == total:
                    percent = (completed / total) * 100 if total else 0
                    payload = {
                        "type": "progress",
                        "description": description,
                        "completed": completed,
                        "total": total,
                        "percent": percent,
                        "timestamp": now,
                    }
                    if extra:
                        payload.update(extra)
                    msg = orjson.dumps(payload).decode("utf-8")
                    self.socketio_emit_func(msg)
                    if progress_callback:
                        progress_callback(payload)
                    update_progress.last_emit = now
            update_progress.last_emit = 0
            try:
                yield update_progress
            finally:
                # Ensure 100% is emitted at the end
                update_progress(total)
        else:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=40, style="bold blue"),
                "[progress.percentage]{task.percentage:>3.0f}%",
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                expand=True,
                transient=True
            ) as progress:
                task_id = progress.add_task(description, total=total)
                def update_progress(completed: int, extra: Optional[dict] = None):
                    progress.update(task_id, completed=completed)
                yield update_progress

    # --- Log File Discovery and Safe JSONL Reading ---
    def discover_log_files(self, log_dirs: List[str], suffixes: tuple = (".jsonl", ".json", ".log", ".html")) -> List[Path]:
        """
        Discover log files in the given directories with specified suffixes.
        """
        files = []
        for log_dir in log_dirs:
            p = Path(log_dir)
            if not p.exists():
                self.warning(f"Log directory does not exist: {log_dir}")
                continue
            for suf in suffixes:
                files.extend(p.glob(f"*{suf}"))
        return files

    def safe_read_jsonl(self, path: str) -> List[Any]:
        """
        Safely read a JSONL file and return a list of entries.
        """
        entries = []
        try:
            with open(path, "rb") as f:
                for line in f:
                    try:
                        entries.append(orjson.loads(line))
                    except Exception as e:
                        self.warning(f"Corrupt line in {path}: {e}")
        except Exception as e:
            self.error(f"Failed to read {path}: {e}")
        return entries

    def summarize_logs(self, log_path: Optional[str] = None, max_lines: int = 1000) -> str:
        """
        Return the last max_lines of the log file as a string.
        """
        log_path = log_path or "pipeline.log"
        if not os.path.exists(log_path):
            return ""
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return "".join(lines[-max_lines:])

# --- Global logger instance and wrappers for backward compatibility ---

_logger_instance = SharedLogger()

def set_log_mode(mode: Optional[str]):
    """Set the global logger mode."""
    _logger_instance.set_mode(mode)

def set_log_format(fmt: Optional[str]):
    """Set the global logger format."""
    _logger_instance.set_format(fmt)

def set_log_level(level: Optional[str]):
    """Set the global logger level."""
    _logger_instance.set_level(level)

def set_socketio_emit_func(emit_func: Callable[[str], None]):
    """Set the global logger's socketio emit function."""
    _logger_instance.set_socketio_emit_func(emit_func)

def set_log_file_path(file_path: str):
    """Set the global logger's file output path."""
    _logger_instance.set_file_path(file_path)

def suppress_rich_logs(value: bool = True):
    """Suppress rich logs globally."""
    _logger_instance.suppress(value)

def log_trace(msg: str, context: Any = None, *args, **kwargs):
    _logger_instance.trace(msg, context)

def log_debug(msg: str, context: Any = None, *args, **kwargs):
    _logger_instance.debug(msg, context)

def log_info(msg: str, context: Any = None, *args, **kwargs):
    _logger_instance.info(msg, context)

def log_warning(msg: str, context: Any = None, *args, **kwargs):
    _logger_instance.warning(msg, context)

def log_error(msg: str, context: Any = None, *args, **kwargs):
    _logger_instance.error(msg, context)

def log_critical(msg: str, context: Any = None, *args, **kwargs):
    _logger_instance.critical(msg, context)

def log_alert(msg: str, context: Any = None, alert_type: str = "info"):
    _logger_instance.alert(msg, context, alert_type=alert_type)

@contextmanager
def progress_bar(description: str = "Processing", total: int = 100, **kwargs) -> Generator[Any, None, None]:
    """Context manager for a global progress bar (CLI or webapp)."""
    with _logger_instance.progress_bar(description, total, **kwargs) as update_progress:
        yield update_progress

def discover_log_files(log_dirs: List[str], suffixes: tuple = (".jsonl", ".json", ".log", ".html")) -> List[Path]:
    return _logger_instance.discover_log_files(log_dirs, suffixes)

def safe_read_jsonl(path: str) -> List[Any]:
    return _logger_instance.safe_read_jsonl(path)

def summarize_logs(log_path: Optional[str] = None, max_lines: int = 1000) -> str:
    return _logger_instance.summarize_logs(log_path, max_lines)