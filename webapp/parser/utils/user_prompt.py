from __future__ import annotations
# webapp/parser/utils/user_prompt.py
# -----------------------------------------------------------------------------------
# This file contains a unified user prompt handler for both CLI and webapp modes.
# It encapsulates all prompt logic as methods and provides a consistent interface
# for user interactions, including advanced features like session management and
# structured logging.
# -----------------------------------------------------------------------------------
import time
import threading
import datetime
import re
import orjson
import inspect
import traceback
from datetime import timezone
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn
from typing import Any, Callable, Dict, List, Optional, Union, Generator, ContextManager
from contextlib import contextmanager

def safe_lower(val) -> str:
    try:
        return val.lower() if isinstance(val, str) else str(val).lower()
    except Exception:
        return ""

def safe_strip(val) -> str:
    try:
        return val.strip() if isinstance(val, str) else str(val).strip()
    except Exception:
        return ""

class PromptCancelled(Exception):
    """Raised when the user cancels a prompt."""
    pass

class PromptSession(ContextManager):
    """Session object for webapp prompt responses."""
    def __init__(
        self,
        session_id: Optional[str] = None,
        context: Optional[dict] = None,
        expiry_seconds: int = 600
    ):
        self.event = threading.Event()
        self.response = None
        self.session_id = session_id
        self.context = context or {}
        self.created_at = datetime.datetime.now(timezone.utc)
        self.last_active = self.created_at
        self.expiry_seconds = expiry_seconds
        self.cancelled = False
        self.timed_out = False

    def __enter__(self) -> 'PromptSession':
        self.response = None
        self.event.clear()
        self.created_at = datetime.datetime.now(timezone.utc)
        self.last_active = self.created_at
        self.cancelled = False
        self.timed_out = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.event.set()
        if exc_type is not None:
            print(f"[PromptSession] Exception in context: {exc_type.__name__}: {exc_val}")
            traceback.print_tb(exc_tb)
        return False

    def is_resolved(self) -> bool:
        """Return True if the prompt session has a response, was cancelled, timed out, or expired."""
        return (
            self.response is not None
            or self.cancelled
            or self.timed_out
            or self.is_expired()
        )

    def wait_for_response(self, timeout: Optional[float] = None) -> Any:
        """Wait for a response with optional timeout."""
        self.last_active = datetime.datetime.now(timezone.utc)
        finished = self.event.wait(timeout)
        if not finished:
            self.timed_out = True
        return self.response

    def set_response(self, response: Any) -> None:
        """Set the response and notify waiting thread."""
        self.response = response
        self.last_active = datetime.datetime.now(timezone.utc)
        self.event.set()

    def cancel(self) -> None:
        """Cancel the prompt session."""
        self.cancelled = True
        self.last_active = datetime.datetime.now(timezone.utc)
        self.event.set()

    def is_expired(self, now: Optional[datetime.datetime] = None) -> bool:
        """Check if the session has expired."""
        now = now or datetime.datetime.now(timezone.utc)
        return (now - self.last_active).total_seconds() > self.expiry_seconds

    @property
    def status(self) -> str:
        """Return the current status of the prompt session."""
        if self.cancelled:
            return "cancelled"
        if self.timed_out:
            return "timed_out"
        if self.response is not None:
            return "responded"
        if self.is_expired():
            return "expired"
        return "pending"      

class UserPrompt(ContextManager):
    """
    Unified user prompt handler for CLI and webapp modes.
    All prompt logic is encapsulated as methods.
    Use .prompt(prompt_type, ...) to dispatch to a specific prompt type.
    """

    # Compiled regex patterns for performance
    RICH_MARKUP_RE = re.compile(r"\[[a-zA-Z0-9_]+\]")
    LABEL_COLOR_RE = re.compile(r"^\[([a-zA-Z0-9_ ]+)\]\s*(.*)")

    def __init__(
        self,
        timeout_sec: int = 180,
        mode: Optional[str] = None,
        socketio_emit_func: Optional[Callable[[str], None]] = None,
        file_path: Optional[str] = None,
    ) -> None:
        self.mode = mode
        self.socketio_emit_func = socketio_emit_func
        self.prompt_sessions: Dict[str, PromptSession] = {}
        self.timeout_sec = timeout_sec
        self.file_path = file_path
        self._pending_delete: Dict[str, float] = {}  # session_id -> delete_at timestamp
        self._cleanup_lock = threading.Lock()
        self._cleanup_thread_started = False
        self._start_cleanup_thread()      

    def _start_cleanup_thread(self):
        if self._cleanup_thread_started:
            return
        self._cleanup_thread_started = True
        def cleanup_loop():
            while True:
                now = time.time()
                with self._cleanup_lock:
                    # Remove sessions marked for delayed deletion
                    to_delete = [sid for sid, t in self._pending_delete.items() if now >= t]
                    for sid in to_delete:
                        self.prompt_sessions.pop(sid, None)
                        self._pending_delete.pop(sid, None)
                    # Remove sessions expired by inactivity
                    expired = [sid for sid, ps in self.prompt_sessions.items() if ps.is_expired()]
                    for sid in expired:
                        self.prompt_sessions.pop(sid, None)
                time.sleep(5)
        t = threading.Thread(target=cleanup_loop, daemon=True)
        t.start()

    def create_prompt_session(self, session_id: str, context: Optional[dict] = None) -> PromptSession:
        with self._cleanup_lock:
            ps = PromptSession(session_id, context, self.timeout_sec)
            self.prompt_sessions[session_id] = ps
            return ps

    def get_prompt_session(self, session_id: str, context: Optional[dict] = None) -> PromptSession:
        """
        Get or create a prompt session by session_id, with optional context.
        Expired sessions are replaced.
        """
        with self._cleanup_lock:
            session = self.prompt_sessions.get(session_id)
            if session is None or session.is_expired():
                session = PromptSession(session_id, context, self.timeout_sec)
                self.prompt_sessions[session_id] = session
            return session

    def clear_prompt_session(self, session_id: str, delay: float = 30.0) -> None:
        """Mark a prompt session for deletion after a delay (seconds)."""
        with self._cleanup_lock:
            if session_id in self.prompt_sessions:
                self._pending_delete[session_id] = time.time() + delay

    def cleanup_sessions(self) -> None:
        """Immediate cleanup of expired or completed sessions."""
        now = time.time()
        with self._cleanup_lock:
            to_remove = [
                sid for sid, sess in self.prompt_sessions.items()
                if sess.event.is_set() or sess.is_expired()
            ]
            for sid in to_remove:
                self.prompt_sessions.pop(sid, None)
            # Also remove any pending deletes that are now expired
            expired_pending = [sid for sid, t in self._pending_delete.items() if now >= t]
            for sid in expired_pending:
                self._pending_delete.pop(sid, None)
        
    def __enter__(self) -> 'UserPrompt':
        """
        Enter the UserPrompt context.
        Sets up resources for CLI or webapp mode, manages session state,
        and prepares for advanced/nested prompt handling.
        """
        from .logger_singleton import logger
        logger.info(f"[UserPrompt] Entering context at {datetime.datetime.now(timezone.utc).isoformat()} (mode={self.mode})")

        # Ensure prompt_sessions is always a dict
        if not hasattr(self, "prompt_sessions") or self.prompt_sessions is None:
            self.prompt_sessions = {}

        # Track nested prompt contexts (for advanced use)
        if not hasattr(self, "_context_stack"):
            self._context_stack = []
        self._context_stack.append({
            "timestamp": datetime.datetime.now(timezone.utc).isoformat(),
            "mode": self.mode,
            "file_path": self.file_path,
            "socketio_emit_func": bool(self.socketio_emit_func),
        })

        # CLI mode: prepare file logging if needed
        if self.mode == "cli":
            if self.file_path and not hasattr(self, "_file_handle"):
                try:
                    self._file_handle = open(self.file_path, "ab")
                    logger.info(f"[UserPrompt] Opened log file: {self.file_path}")
                except Exception as e:
                    logger.error(f"[UserPrompt] Failed to open log file: {e}")
                    self._file_handle = None

        # Webapp mode: ensure socketio emit function is set
        if self.mode == "webapp":
            if not self.socketio_emit_func:
                logger.warning("[UserPrompt] Webapp mode active but no socketio_emit_func set!")
            # Optionally, preload webapp-specific settings here
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        from .logger_singleton import logger
        # Cleanup logic (e.g., clear prompt sessions)
        self.prompt_sessions.clear()
        # Optionally log exceptions for debugging
        if exc_type is not None:
            logger.error(f"[UserPrompt] Exception in context: {exc_type.__name__}: {exc_val}")
            traceback.print_tb(exc_tb)
        # Return False to propagate exceptions
        return False

    def set_mode(self, mode: Optional[str] = None) -> None:
        """Set the prompt mode (cli/webapp)."""
        self.mode = mode

    def set_socketio_emit_func(self, emit_func: Callable[[str], None]) -> None:
        """Set the function to emit prompts via socketio (for webapp mode)."""
        self.socketio_emit_func = emit_func

    def set_file_path(self, file_path: str) -> None:
        """Set the file path for prompt logging."""
        self.file_path = file_path

    def _emit_cli_prompt(self, message: str, default: Optional[str] = None) -> str:
        """
        Centralized CLI prompt logic with consistent logging.
        """
        from .logger_singleton import logger
        logger.info(f"[CLI Prompt] {message}")
        try:
            resp = input(message)
            return resp if resp else default
        except EOFError:
            logger.warning("[CLI Prompt] EOFError encountered.")
            return default

    def _emit_webapp_prompt(self, message: str, session_id: str, context: Optional[dict] = None, status: str = "prompt", error: Optional[str] = None) -> None:
        """
        Centralized webapp prompt logic.
        Emits structured JSON messages for prompt, status, or error.
        """
        from .logger_singleton import logger
        payload = {
            "type": status,
            "session_id": session_id,
            "message": message,
            "context": context or {},
            "timestamp": datetime.datetime.now(timezone.utc).isoformat(),
        }
        if error:
            payload["error"] = error
        if self.socketio_emit_func:
            self.socketio_emit_func(orjson.dumps(payload).decode("utf-8"))
        else:
            logger.warning("[Webapp Prompt] socketio_emit_func not set.")

    def print_header(self, title: str = "USER INPUT REQUIRED", char: str = "=", width: int = 60) -> None:
        """Print a formatted header for prompts."""
        from .logger_singleton import logger
        logger.info("\n" + char * width)
        logger.info(f"{title.center(width)}")
        logger.info(char * width)

    def prompt(self, prompt_type: str, *args, **kwargs) -> Any:
        """
        Dispatcher for prompt types.
        Example: prompt("yes_no", message="Continue?")
        """
        method = getattr(self, f"prompt_{prompt_type}", None)
        if not method:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
        return method(*args, **kwargs)

    def _emit_prompt(self, message: str, session_id: Optional[str] = None, context: Optional[dict] = None, status: str = "prompt", error: Optional[str] = None) -> None:
        """
        Emit a prompt or status message in both CLI and webapp modes.
        """
        from .logger_singleton import logger, console
        payload = {
            "type": status,
            "session_id": session_id,
            "message": message,
            "context": context or {},
            "timestamp": datetime.datetime.now(timezone.utc).isoformat(),
        }
        if error:
            payload["error"] = error

        logger.info(orjson.dumps(payload, option=orjson.OPT_INDENT_2).decode("utf-8"))

        if self.mode == "webapp" and self.socketio_emit_func and session_id:
            self.socketio_emit_func(orjson.dumps(payload).decode("utf-8"))
        else:
            # For CLI, print only the message field, not the whole dict
            if status == "error":
                console.print(f"[ERROR] {message}")
            elif status == "status":
                console.print(f"[STATUS] {message}")
            else:
                console.print(str(message))

    def _should_emit(self, level: str = "INFO") -> bool:
        """
        Check if a message at the given level should be emitted, based on SharedLogger's current log level.
        """
        from .logger_singleton import logger
        # Use the logger's level mapping and current level
        logger_level = getattr(logger, "level", "INFO")
        level_mapping = getattr(logger, "level_mapping", {
            "TRACE": 5,
            "DEBUG": 10,
            "INFO": 20,
            "WARNING": 30,
            "ERROR": 40,
            "CRITICAL": 50
        })
        # Normalize level names
        level = level.upper()
        logger_level = logger_level.upper() if isinstance(logger_level, str) else "INFO"
        return level_mapping.get(level, 100) >= level_mapping.get(logger_level, 20)

    def _format_context(self, context: Any) -> str:
        """Format context for output using orjson."""
        if context is None:
            return ""
        if isinstance(context, dict):
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
                return {
                    "module": caller.frame.f_globals.get("__name__", ""),
                    "function": caller.function,
                    "line": caller.lineno
                }
        return {}

    def _log_to_file(self, msg: str, context: Any = None) -> None:
        """Log prompt interactions to a file if file_path is set, using orjson."""
        if not self.file_path:
            return
        log_line = {
            "timestamp": datetime.datetime.now().isoformat(),
            "message": msg,
            "context": self._format_context(context),
            **self._get_caller_info()
        }
        with open(self.file_path, "ab") as f:
            f.write(orjson.dumps(log_line, option=orjson.OPT_APPEND_NEWLINE))

    def prompt_input(
        self,
        message: str,
        session_id: Optional[str] = None,
        timeout: Optional[float] = None,
        default: Optional[str] = None,
        validator: Optional[Callable[[str], bool]] = None,
        allow_cancel: bool = True,
        on_error: Optional[Callable[[str], None]] = None,
        header: Optional[str] = None,
        log_func: Optional[Callable[[str], None]] = None,
        max_attempts: int = 5,
        context: Any = None,
        raise_on_max_attempts: bool = True,
    ) -> str:
        """
        Prompt the user for input, with optional default, validation, cancel, timeout, header, and logging.
        Returns the validated input or raises PromptCancelled if cancelled.
        """
        from .logger_singleton import logger
        print("Creating prompt session for", session_id)
        self.cleanup_sessions()
        def input_with_timeout(prompt: str, timeout: float) -> Optional[str]:
            result = [None]
            def inner():
                try:
                    result[0] = input(prompt)
                except Exception:
                    result[0] = None
            t = threading.Thread(target=inner)
            t.start()
            t.join(timeout)
            if t.is_alive():
                logger.warning("\n[Prompt] Timed out.")
                return None
            return result[0]

        attempts = 0
        if header:
            self.print_header(header)
        while True:
            prompt_str = f"{message}"
            if default is not None:
                prompt_str += f" [{default}]"
            if allow_cancel:
                prompt_str += " (type 'cancel' to abort)"
            prompt_str += " "
            try:
                if self.mode == "webapp" and self.socketio_emit_func and session_id:
                    # Emit prompt payload to frontend
                    payload = {
                        "type": "prompt",
                        "message": prompt_str,
                        "session_id": session_id,
                        "context": context
                    }
                    self.socketio_emit_func(orjson.dumps(payload).decode("utf-8"))
                    # Wait for frontend response
                    prompt_session = self.get_prompt_session(session_id, context)
                    print("TRACE: Waiting for frontend response for session", session_id)
                    response = prompt_session.wait_for_response(timeout)
                    print("TRACE: Received response from frontend:", response)
                    if response is None:
                        self._emit_prompt(
                            f"Prompt timed out or cancelled. Using default: {default}",
                            session_id,
                            context,
                            status="status",
                            error="timeout_or_cancel"
                        )
                        return default
                    self._emit_prompt(
                        f"Prompt response received: {response}",
                        session_id,
                        context,
                        status="status"
                    )
                    self.clear_prompt_session(session_id)
                else:
                    response = input_with_timeout(prompt_str, timeout) if timeout else input(prompt_str)
            except EOFError:
                logger.warning("\n[Prompt] No input available (EOF). Exiting prompt.")
                return default
            if response is None:
                if timeout:
                    if on_error:
                        on_error("Timed out.")
                    if log_func:
                        log_func(f"[PROMPT] Timed out at {datetime.datetime.now()}")
                    self._log_to_file(prompt_str + " [Timed out]", context)
                    return default
                continue
            if allow_cancel and safe_lower(safe_strip(response)) == "cancel":
                if log_func:
                    log_func(f"[PROMPT] User cancelled at {datetime.datetime.now()}")
                self._log_to_file(prompt_str + " [User cancelled]", context)
                raise PromptCancelled("User cancelled the prompt.")
            if not response and default is not None:
                response = default
            if validator:
                try:
                    if validator(response):
                        if log_func:
                            log_func(f"[PROMPT] User input: {response} at {datetime.datetime.now()}")
                        self._log_to_file(prompt_str + f" [User input: {response}]", context)
                        return response
                except Exception:
                    pass
                attempts += 1
                if on_error:
                    on_error("Invalid input.")
                logger.warning("Invalid input. Please try again.")
                if attempts >= max_attempts:
                    logger.warning("[Prompt] Too many invalid attempts.")
                    if log_func:
                        log_func(f"[PROMPT] Too many invalid attempts at {datetime.datetime.now()}")
                    self._log_to_file(prompt_str + " [Too many invalid attempts]", context)
                    if raise_on_max_attempts:
                        raise PromptCancelled("Too many invalid attempts.")
                    # Soft-fail: return default (or empty string if no default)
                    return default
            else:
                if log_func:
                    log_func(f"[PROMPT] User input: {response} at {datetime.datetime.now()}")
                self._log_to_file(prompt_str + f" [User input: {response}]", context)
                return response

    def prompt_yes_no(
        self,
        message: str,
        default: str = "y",
        allow_cancel: bool = True,
        timeout: Optional[float] = None,
        header: Optional[str] = None,
        log_func: Optional[Callable[[str], None]] = None,
        session_id: Optional[str] = None,
        context: Any = None,
    ) -> bool:
        """
        Prompt the user for a yes/no answer.
        """
        from .logger_singleton import logger
        self.cleanup_sessions()
        if header:
            self.print_header(header)
        prompt_str = f"{message} (y/n) [{default}]"
        if allow_cancel:
            prompt_str += " (type 'cancel' to abort)"
        prompt_str += ": "
        while True:
            if self.mode == "webapp" and session_id:
                resp = self.prompt_input(prompt_str, session_id=session_id, timeout=timeout, default=default, context=context)
            elif timeout:
                result = [None]
                def inner():
                    try:
                        result[0] = input(prompt_str)
                    except Exception:
                        result[0] = None
                t = threading.Thread(target=inner)
                t.start()
                t.join(timeout)
                if t.is_alive():
                    logger.warning("\n[Prompt] Timed out.")
                    return safe_lower(default or "") == "y"
                resp = result[0]
            else:
                resp = input(prompt_str)
            if resp is None or not safe_strip(resp):
                resp = default
            resp = safe_lower(safe_strip(resp))
            if allow_cancel and resp == "cancel":
                if log_func:
                    log_func(f"[PROMPT] User cancelled yes/no at {datetime.datetime.now()}")
                self._log_to_file(prompt_str + " [User cancelled]", context)
                raise PromptCancelled("User cancelled the prompt.")
            if resp in ("y", "yes"):
                if log_func:
                    log_func(f"[PROMPT] User input: YES at {datetime.datetime.now()}")
                self._log_to_file(prompt_str + " [YES]", context)
                return True
            if resp in ("n", "no"):
                if log_func:
                    log_func(f"[PROMPT] User input: NO at {datetime.datetime.now()}")
                self._log_to_file(prompt_str + " [NO]", context)
                return False
            logger.info("Please enter 'y' or 'n'.")

    def prompt_choice(
        self,
        message: str,
        options: List[str],
        default: Optional[int] = None,
        allow_cancel: bool = True,
        header: Optional[str] = None,
        log_func: Optional[Callable[[str], None]] = None,
        session_id: Optional[str] = None,
        context: Any = None,
    ) -> str:
        """
        Prompt the user to select from a list of options.
        """
        from .logger_singleton import logger
        if not options:
            raise ValueError("No options provided for selection.")
        if header:
            self.print_header(header)
        for idx, opt in enumerate(options):
            logger.info(f"  [{idx}] {opt}")
        def validator(x: str) -> bool:
            return x.isdigit() and 0 <= int(x) < len(options)
        selection = self.prompt_input(
            f"{message} (0-{len(options)-1})",
            default=str(default) if default is not None else "0",
            validator=validator,
            allow_cancel=allow_cancel,
            header=None,
            log_func=log_func,
            session_id=session_id,
            context=context
        )
        if log_func:
            log_func(f"[PROMPT] User selected option {selection} at {datetime.datetime.now()}")
        self._log_to_file(f"{message} [User selected: {selection}]", context)
        return options[int(selection)]

    def prompt_for_metadata_field(
        self,
        field_name: str,
        suggestions: Optional[List[str]] = None,
        default: Optional[str] = None,
        allow_cancel: bool = True,
        session_id: Optional[str] = None,
        context: Any = None,
    ) -> str:
        """
        Prompt for a metadata field, optionally with suggestions.
        """
        from .logger_singleton import logger
        if suggestions:
            logger.info(f"Suggestions for {field_name}:")
            for idx, s in enumerate(suggestions):
                logger.info(f"  [{idx}] {s}")
            def validator(x: str) -> bool:
                return (x.isdigit() and 0 <= int(x) < len(suggestions)) or bool(x.strip())
            response = self.prompt_input(
                f"Enter {field_name} or select a suggestion (0-{len(suggestions)-1}):",
                default=str(default) if default is not None else "",
                validator=validator,
                allow_cancel=allow_cancel,
                session_id=session_id,
                context=context
            )
            if response.isdigit():
                return suggestions[int(response)]
            return response
        else:
            return self.prompt_input(
                f"Enter {field_name}:",
                default=default,
                allow_cancel=allow_cancel,
                session_id=session_id,
                context=context
            )

    def prompt_for_metadata(
        self,
        metadata_fields: Dict[str, Dict[str, Any]],
        session_id: Optional[str] = None,
        context: Any = None,
    ) -> Dict[str, Any]:
        """
        Prompt for multiple metadata fields.
        """
        responses = {}
        for field, opts in metadata_fields.items():
            responses[field] = self.prompt_for_metadata_field(
                field,
                suggestions=opts.get("suggestions", []),
                default=opts.get("default", []),
                session_id=session_id,
                context=context
            )
        return responses

    def prompt_review_context(
        self,
        context: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Review and optionally edit a context dictionary.
        """
        from .logger_singleton import logger
        logger.info("\n[Context Review]")
        for k, v in context.items():
            logger.info(f"  {k}: {v}")
        if self.prompt_yes_no("Is this context correct?", default="y", session_id=session_id, context=context):
            return context
        for k in context:
            if self.prompt_yes_no(f"Edit {k}? (current: {context[k]})", default="n", session_id=session_id, context=context):
                context[k] = self.prompt_input(f"Enter new value for {k}:", default=str(context[k]), session_id=session_id, context=context)
        return context

    def prompt_resolve_conflict(
        self,
        conflict_type: str,
        options: List[str],
        session_id: Optional[str] = None,
        context: Any = None,
    ) -> str:
        """
        Prompt user to resolve a conflict by selecting an option.
        """
        from .logger_singleton import logger
        logger.info(f"\n[Conflict Detected: {conflict_type}]")
        for idx, opt in enumerate(options):
            logger.info(f"  [{idx}] {opt}")
        idx = self.prompt_input(
            f"Select the correct option (0-{len(options)-1}):",
            validator=lambda x: x.isdigit() and 0 <= int(x) < len(options),
            session_id=session_id,
            context=context
        )
        self._log_to_file(f"Conflict resolved: {conflict_type} -> {options[int(idx)]}", context)
        return options[int(idx)]

    def prompt_user_for_button(
        self,
        page: Any,
        candidates: List[Dict[str, Any]],
        toggle_name: str,
        session_id: Optional[str] = None,
        context: Any = None,
    ) -> Union[Dict[str, Any], None]:
        """
        Prompt user to select the correct button from candidates.
        The `page` argument is accepted for compatibility with advanced feedback/callbacks.
        """
        from .logger_singleton import logger
        logger.info(f"\n[FEEDBACK] Please select the correct button for '{toggle_name}':")
        for idx, btn in enumerate(candidates):
            logger.info(
                f"{idx}: label='{btn.get('label', '')}'"
                f" | class='{btn.get('class', '')}'"
                f" | tag='{btn.get('tag', '')}'"
                f" | context_heading='{btn.get('context_heading', '')}'"
                f" | context_anchor='{btn.get('context_anchor', '')}'"
                f" | visible={btn.get('is_visible', False)}"
                f" | enabled={btn.get('is_clickable', False)}"
            )
        # Optionally, add page info to context for logging/debugging
        if context is None:
            context = {}
        context = dict(context)  # shallow copy
        context["page_info"] = str(page)  # or extract relevant info if needed

        try:
            choice = self.prompt_input(
                "Enter the number of the correct button (or -1 to skip): ",
                session_id=session_id,
                context=context
            )
            choice = int(choice)
            if 0 <= choice < len(candidates):
                chosen_btn = candidates[choice]
                logger.info(f"[bold green][FEEDBACK] You selected: '{chosen_btn.get('label', '')}'[/bold green]")
                self._log_to_file(f"Button selected: {chosen_btn}", context)
                return chosen_btn, choice
            else:
                logger.warning("[yellow][FEEDBACK] Skipped manual correction.[/yellow]")
                self._log_to_file("Button selection skipped", context)
                return None, None
        except Exception as e:
            logger.error(f"[red][FEEDBACK ERROR] {e}[/red]")
            self._log_to_file(f"Button selection error: {e}", context)
            return None, None

    def confirm_button_callback(
        self,
        candidate: Dict[str, Any],
        session_id: Optional[str] = None,
        context: Any = None,
    ) -> bool:
        """
        Confirm with the user if a button should be clicked.
        """
        from .logger_singleton import logger
        label = candidate.get("label", "")
        selector = candidate.get("selector", "")
        logger.info(f"\n[CONFIRMATION] Candidate button found: '{label}'\nSelector: {selector}")
        try:
            resp = self.prompt_input(
                f"Do you want to click this button? (y/n): ",
                default="y",
                validator=lambda x: safe_lower(x or "") in {"y", "n", "yes", "no"},
                allow_cancel=True,
                header="BUTTON CONFIRMATION",
                session_id=session_id,
                context=context
            ).strip().lower()
        except PromptCancelled:
            logger.warning("[yellow]Button confirmation cancelled by user.[/yellow]")
            self._log_to_file("Button confirmation cancelled", context)
            return False
        self._log_to_file(f"Button confirmation: {resp}", context)
        return resp in {"y", "yes"}

    @contextmanager
    def progress_bar(
        self,
        description: str = "Processing",
        total: int = 100,
        emit_interval: float = 0.5,
        progress_callback: Optional[Callable[[dict], None]] = None,
    ) -> Generator[Any, None, None]:
        """
        Context manager for a progress bar.
        In CLI mode: shows a live progress bar.
        In webapp mode: emits progress updates via SocketIO as JSON.
        Optionally, accepts a progress_callback for custom handling.
        Usage:
            with prompt.progress_bar("Processing", total=100) as update_progress:
                for i in range(total):
                    # ... your work ...
                    update_progress(i + 1)
        """
        if self.mode == "webapp" and self.socketio_emit_func:
            last_emit = 0

            def update_progress(completed: int, extra: Optional[dict] = None) -> None:
                nonlocal last_emit
                now = time.time()
                should_emit = (now - last_emit >= emit_interval) or (completed == total)
                if should_emit:
                    percent = (completed / total) * 100 if total else 0
                    payload = {
                        "type": "progress",
                        "description": description,
                        "completed": completed,
                        "total": total,
                        "percent": percent,
                        "timestamp": now,
                    }
                    # Robustly merge extra if it's a dict
                    if isinstance(extra, dict):
                        payload.update(extra)
                    elif extra is not None:
                        payload["extra"] = str(extra)
                    msg = orjson.dumps(payload).decode("utf-8")
                    self.socketio_emit_func(msg)
                    if progress_callback:
                        progress_callback(payload)
                    last_emit = now
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
                    # Optionally, handle extra/progress_callback in CLI mode as well
                    if progress_callback:
                        percent = (completed / total) * 100 if total else 0
                        payload = {
                            "type": "progress",
                            "description": description,
                            "completed": completed,
                            "total": total,
                            "percent": percent,
                            "timestamp": time.time(),
                        }
                        if isinstance(extra, dict):
                            payload.update(extra)
                        elif extra is not None:
                            payload["extra"] = str(extra)
                        progress_callback(payload)
                yield update_progress

# Example usage:
# prompt = UserPrompt()  # mode will be set from .env PROMPT_MODE or default to "cli"
# answer = prompt.prompt("yes_no", message="Continue?", default="y")
# choice = prompt.prompt("choice", message="Pick one:", options=["A", "B", "C"])
# user_input = prompt.prompt("input", message="Enter your name:")