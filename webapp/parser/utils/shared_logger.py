import os
import re
import time
import logging
import inspect
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Generator, Tuple, Set
from rich import print as rprint
from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn
from rich.json import JSON
import orjson
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

def safe_getvalue(file_obj: StringIO) -> str:
    """
    Safely call .getvalue() on a file-like object.
    Returns the string value or an empty string if an error occurs.
    """
    try:
        return file_obj.getvalue()
    except Exception as e:
        logging.error(f"Error getting value from StringIO: {e}")
        return ""

class RichConsoleProxy(Console):
    """
    Proxy class to provide a Console-like API using SharedLogger.
    Use this as a drop-in replacement for rich.console.Console.
    Handles routing of rich output (Table, Panel, Progress, etc.) to CLI or webapp (SocketIO).
    """
    def __init__(self, logger=None) -> None:
        self.logger = logger or SharedLogger()
        self._console = Console()

    def print(self, *args, **kwargs) -> None:
        """
        Print using rich's Console.print (for tables, panels, JSON, etc).
        In webapp mode, serialize the output and emit via SocketIO.
        """
        renderables = []
        for obj in args:
            # If it's a dict, wrap in Rich JSON
            if isinstance(obj, dict):
                renderables.append(JSON(obj))
            # If it's bytes (from orjson), decode and wrap in Rich JSON
            elif isinstance(obj, bytes):
                try:
                    renderables.append(JSON(obj.decode("utf-8")))
                except Exception:
                    renderables.append(str(obj))
            # If it's a string that looks like JSON, wrap in Rich JSON
            elif isinstance(obj, str) and obj.strip().startswith("{"):
                try:
                    renderables.append(JSON(obj))
                except Exception:
                    renderables.append(obj)
            else:
                renderables.append(obj)

        if getattr(self.logger, "mode", "cli") == "webapp" and getattr(self.logger, "socketio_emit_func", None):
            sio = self.logger.socketio_emit_func
            for obj in renderables:
                sio(self._render_to_text(obj))
        else:
            self._console.print(*renderables, **kwargs)

    def _render_to_text(self, obj: RenderableType) -> str:
        """
        Render a rich object (Table, Panel, Progress, etc.) to plain text for webapp streaming.
        """
        temp_console = Console(file=StringIO(), force_terminal=True, color_system=None, width=100)
        temp_console.print(obj)
        output = safe_getvalue(temp_console.file)
        return output

    def rule(self, title: str = "", **kwargs) -> None:
        """Draw a rule (horizontal line) with optional title."""
        if getattr(self.logger, "mode", "webapp") and getattr(self.logger, "socketio_emit_func", None):
            sio = self.logger.socketio_emit_func
            sio("-" * 40 + f" {title} " + "-" * 40)
        else:
            self._console.rule(title, **kwargs)

    def panel(self, msg: str, title: str = "", style: str = "blue") -> None:
        """Print a message in a rich Panel."""
        panel = Panel(msg, title=title, style=style)
        self.print(panel)

    def table(self, *args, **kwargs) -> None:
        """
        Print a rich Table object.
        In webapp mode, serialize the table and emit via SocketIO.
        In CLI mode, print using rich's Console.
        """
        from rich.table import Table
        if args and isinstance(args[0], Table):
            table_obj = args[0]
        else:
            table_obj = Table(*args, **kwargs)
        if getattr(self.logger, "mode", "webapp") and getattr(self.logger, "socketio_emit_func", None):
            sio = self.logger.socketio_emit_func
            sio(self._render_to_text(table_obj))
        else:
            self._console.print(table_obj)

    def input(self, prompt: str = "") -> str:
        """Proxy for input, using rich's Console.input."""
        return self._console.input(prompt)

    def log(self, *args, **kwargs) -> None:
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

class SQLAlchemyToSharedLoggerHandler(logging.Handler):
    def __init__(self, shared_logger: "SharedLogger"):
        super().__init__()
        self.shared_logger = shared_logger

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        # You can choose the log level mapping here
        if record.levelno >= logging.ERROR:
            self.shared_logger.error(msg)
        elif record.levelno >= logging.WARNING:
            self.shared_logger.warning(msg)
        elif record.levelno >= logging.INFO:
            self.shared_logger.info(msg)
        else:
            self.shared_logger.debug(msg)
            
class SharedLogger(logging.Logger):
    """
    Unified logger supporting CLI/webapp and plain/json/file output.
    Mode, format, and level can be set via .env or at init.
    """

    # Compiled regex patterns for performance
    RICH_MARKUP_RE = re.compile(r"\[[a-zA-Z0-9_]+\]")
    LABEL_COLOR_RE = re.compile(r"^\[([a-zA-Z0-9_ ]+)\]\s*(.*)")

    _instance = None  # Singleton

    def __new__(cls, *args, **kwargs) -> "SharedLogger":
        # Singleton pattern (optional, can be removed for multi-instance)
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    _warned_sections: Set[str]
    
    def __init__(
        self,
        mode: Optional[str] = None,
        fmt: Optional[str] = None,
        level: Optional[str] = None,
        socketio_emit_func: Optional[Callable[[str], None]] = None,
        suppress_rich_logs: bool = False,
        file_path: Optional[str] = None,
    ) -> None:
        """
        Initialize the logger.
        """
        self.mode = mode or get_log_mode()
        self.format = fmt or get_log_format()
        self.level = (level or get_log_level()).upper()
        self.socketio_emit_func = socketio_emit_func
        self.suppress_rich_logs = suppress_rich_logs
        self.file_path = file_path
        self._warned_sections = set()
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
        if isinstance(context, (dict, list)):
            try:
                return orjson.dumps(context, option=orjson.OPT_INDENT_2).decode("utf-8")
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
                module = caller.frame.f_globals.get("__name__", "")
                if not isinstance(module, str):
                    module = str(module)
                return {
                    "module": module,
                    "function": caller.function,
                    "line": caller.lineno
                }
        return {}

    def trace(self, msg: str, context: Any = None, exc_info: Any = None) -> None:
        """Log a trace message."""
        msg = self._append_traceback(msg, exc_info)
        self._log("TRACE", msg, context, color="cyan")

    def debug(self, msg: str, context: Any = None, exc_info: Any = None) -> None:
        """Log a debug message."""
        msg = self._append_traceback(msg, exc_info)
        self._log("DEBUG", msg, context, color="blue")

    def info(self, msg: str, context: Any = None, exc_info: Any = None) -> None:
        """Log an info message."""
        msg = self._append_traceback(msg, exc_info)
        self._log("INFO", msg, context, color="green")

    def warning(self, msg: str, context: Any = None, exc_info: Any = None) -> None:
        """Log a warning message."""
        msg = self._append_traceback(msg, exc_info)
        self._log("WARNING", msg, context, color="yellow")

    def error(self, msg: str, context: Any = None, exc_info: Any = None) -> None:
        """Log an error message. Accepts exc_info for traceback compatibility."""
        msg = self._append_traceback(msg, exc_info)
        self._log("ERROR", msg, context, color="red")

    def critical(self, msg: str, context: Any = None, exc_info: Any = None) -> None:
        """Log a critical message."""
        msg = self._append_traceback(msg, exc_info)
        self._log("CRITICAL", msg, context, color="magenta")

    def alert(self, msg: str, context: Any = None, alert_type: str = "info", exc_info: Any = None) -> None:
        """Log an alert message with a specific alert type."""
        msg = self._append_traceback(msg, exc_info)
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
        self._log(label, panel_msg, context, color=style)

    def _append_traceback(self, msg: str, exc_info: Any = None) -> str:
        """Helper to append traceback info to message if exc_info is provided."""
        if exc_info:
            if exc_info is True:
                tb_str = traceback.format_exc()
            elif isinstance(exc_info, BaseException):
                tb_str = "".join(traceback.format_exception(type(exc_info), exc_info, exc_info.__traceback__))
            elif isinstance(exc_info, tuple):
                tb_str = "".join(traceback.format_exception(*exc_info))
            else:
                tb_str = str(exc_info)
            msg = f"{msg}\nTraceback:\n{tb_str}"
        return msg

    def _log(self, level: str, msg: str, context: Any = None, color: str = "white") -> None:
        """
        Robustly handle logging for both CLI and webapp GUI.
        Emits to SocketIO in webapp mode, prints rich panels in CLI.
        """
        from ..utils.shared_logic import safe_lower
        if self.suppress_rich_logs or not self._should_emit(level):
            return

        # Defensive: ensure msg is always a string
        if not isinstance(msg, (str, bytes)):
            try:
                msg = orjson.dumps(msg, option=orjson.OPT_INDENT_2).decode("utf-8")
            except Exception:
                msg = str(msg)

        context_str = self._format_context(context)
        # Compose message for panel or plain output
        text_msg = f"[{level}] {msg}"

        # Output logic
        if self.mode == "webapp" and self.socketio_emit_func:
            # Webapp: emit JSON if format is json, else plain text
            if self.format == "json":
                log_obj = {
                    "timestamp": time.time(),
                    "level": level,
                    "color": color,
                    "message": re.sub(r"\[/?[a-zA-Z0-9_ ]+\]", "", msg).strip(),
                    "context": context_str,
                }
                self.socketio_emit_func(orjson.dumps(log_obj).decode("utf-8"))
                # Also log to Python logger
                log_method = getattr(self.logger, safe_lower(level), None)
                if callable(log_method):
                    log_method(log_obj["message"])
                else:
                    self.logger.info(log_obj["message"])
            else:
                plain_msg = re.sub(r"\[/?[a-zA-Z0-9_ ]+\]", "", text_msg)
                self.socketio_emit_func(plain_msg.strip())
                # Also log to Python logger
                log_method = getattr(self.logger, safe_lower(level), None)
                if callable(log_method):
                    log_method(plain_msg.strip())
                else:
                    self.logger.info(plain_msg.strip())
        elif self.mode == "cli":
            try:
                # If msg is JSON, extract "message" field if present
                if isinstance(msg, str) and msg.strip().startswith("{"):
                    try:
                        msg_obj = orjson.loads(msg)
                        # If message is a list, print each item
                        if isinstance(msg_obj, dict) and "message" in msg_obj:
                            message = msg_obj["message"]
                            if isinstance(message, list):
                                for item in message:
                                    rprint(Panel(str(item), style=color))
                            else:
                                rprint(Panel(str(message), style=color))
                        else:
                            rprint(Panel(str(msg), style=color))
                    except Exception:
                        rprint(Panel(str(msg), style=color))
                else:
                    rprint(Panel(str(msg), style=color))
            except Exception:
                print(str(msg))
        # File output (optional)
        if self.file_path:
            log_line = {
                "timestamp": time.time(),
                "level": level,
                "color": color,
                "message": re.sub(r"\[/?[a-zA-Z0-9_ ]+\]", "", msg).strip(),
                "context": context_str,
            }
            with open(self.file_path, "a", encoding="utf-8") as f:
                if self.format == "json":
                    f.write(orjson.dumps(log_line).decode("utf-8") + "\n")
                else:
                    f.write(f"{log_line['timestamp']} [{log_line['level']}] {log_line['message']}\n")

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

            def update_progress(completed: int, extra: Optional[dict] = None) -> None:
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
                def update_progress(completed: int, extra: Optional[dict] = None) -> None:
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
        Handles both text and JSONL logs robustly.
        """
        from ..config import LOG_DIR
        
        # Default to pipeline_log.jsonl in LOG_DIR
        default_log_name = "pipeline_log.jsonl"
        log_path = log_path or os.path.join(LOG_DIR, default_log_name)

        if not os.path.exists(log_path):
            return ""

        # Try to detect if it's a JSONL file
        is_jsonl = log_path.endswith(".jsonl")

        lines = []
        try:
            with open(log_path, "rb" if is_jsonl else "r", encoding=None if is_jsonl else "utf-8") as f:
                if is_jsonl:
                    # Read last max_lines JSON objects
                    all_lines = f.readlines()
                    for line in all_lines[-max_lines:]:
                        try:
                            obj = orjson.loads(line)
                            lines.append(orjson.dumps(obj, option=orjson.OPT_INDENT_2).decode("utf-8"))
                        except Exception:
                            # If not valid JSON, include raw line
                            lines.append(line.decode("utf-8", errors="replace").strip())
                    return "\n".join(lines)
                else:
                    # Plain text log
                    all_lines = f.readlines()
                    return "".join(all_lines[-max_lines:])
        except Exception as e:
            self.error(f"Failed to summarize log file {log_path}: {e}")
            return ""