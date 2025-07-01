import sys
import threading
import datetime
from ..utils.shared_logger import log_info, log_warning, log_error
import os

# --- Mode and SocketIO integration ---
PROMPT_MODE = "cli"  # or "webapp"
SOCKETIO_EMIT_FUNC = None

def set_prompt_mode(mode):
    global PROMPT_MODE
    PROMPT_MODE = mode

def set_socketio_emit_func(emit_func):
    global SOCKETIO_EMIT_FUNC
    SOCKETIO_EMIT_FUNC = emit_func

# --- Prompt session management for webapp mode ---
class PromptSession:
    def __init__(self):
        self.event = threading.Event()
        self.response = None

    def wait_for_response(self, timeout=None):
        self.event.wait(timeout)
        return self.response

    def set_response(self, response):
        self.response = response
        self.event.set()

prompt_sessions = {}  # session_id -> PromptSession

def get_prompt_session(session_id):
    if session_id not in prompt_sessions:
        prompt_sessions[session_id] = PromptSession()
    return prompt_sessions[session_id]

def clear_prompt_session(session_id):
    if session_id in prompt_sessions:
        del prompt_sessions[session_id]

# --- Core prompt logic ---
def prompt_user(message, session_id=None, timeout=None, default=None):
    """
    Unified prompt function for CLI and webapp.
    - In CLI: uses input()
    - In webapp: emits prompt to SocketIO and waits for response
    """
    if PROMPT_MODE == "webapp" and SOCKETIO_EMIT_FUNC and session_id:
        SOCKETIO_EMIT_FUNC(message)
        prompt_session = get_prompt_session(session_id)
        response = prompt_session.wait_for_response(timeout)
        clear_prompt_session(session_id)
        if response is None and default is not None:
            return default
        return response
    else:
        try:
            resp = input(message)
            if not resp and default is not None:
                return default
            return resp
        except EOFError:
            return default

class PromptCancelled(Exception):
    """Raised when the user cancels a prompt."""
    pass

def print_header(title: str = "USER INPUT REQUIRED", char: str = "=", width: int = 60):
    log_info("\n" + char * width)
    log_info(f"{title.center(width)}")
    log_info(char * width)

def prompt_user_input(
    message,
    session_id=None,
    timeout=None,
    default=None,
    validator=None,
    allow_cancel=True,
    on_error=None,
    header=None,
    log_func=None,
    max_attempts=5
):
    """
    Prompt the user for input, with optional default, validation, cancel, timeout, header, and logging.
    Returns the validated input or raises PromptCancelled if cancelled.
    """
    def input_with_timeout(prompt, timeout):
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
            print("\n[Prompt] Timed out.")
            return None
        return result[0]

    attempts = 0
    if header:
        print_header(header)
    while True:
        prompt = f"{message}"
        if default is not None:
            prompt += f" [{default}]"
        if allow_cancel:
            prompt += " (type 'cancel' to abort)"
        prompt += " "
        # --- Use webapp/CLI unified prompt ---
        try:
            if PROMPT_MODE == "webapp" and session_id:
                response = prompt_user(prompt, session_id=session_id, timeout=timeout, default=default)
            else:
                response = input_with_timeout(prompt, timeout) if timeout else input(prompt)
        except EOFError:
            log_warning("\n[Prompt] No input available (EOF). Exiting prompt.")
            return default
        if response is None:
            if timeout:
                if on_error:
                    on_error("Timed out.")
                if log_func:
                    log_func(f"[PROMPT] Timed out at {datetime.datetime.now()}")
                return default
            continue
        if allow_cancel and response.strip().lower() == "cancel":
            if log_func:
                log_func(f"[PROMPT] User cancelled at {datetime.datetime.now()}")
            raise PromptCancelled("User cancelled the prompt.")
        if not response and default is not None:
            response = default
        if validator:
            try:
                if validator(response):
                    if log_func:
                        log_func(f"[PROMPT] User input: {response} at {datetime.datetime.now()}")
                    return response
            except Exception:
                pass
            attempts += 1
            if on_error:
                on_error("Invalid input.")
            log_warning("Invalid input. Please try again.")
            if attempts >= max_attempts:
                log_warning("[Prompt] Too many invalid attempts. Cancelling.")
                if log_func:
                    log_func(f"[PROMPT] Too many invalid attempts at {datetime.datetime.now()}")
                raise PromptCancelled("Too many invalid attempts.")
        else:
            if log_func:
                log_func(f"[PROMPT] User input: {response} at {datetime.datetime.now()}")
            return response

# --- Yes/No and choice helpers ---
def prompt_yes_no(
    message,
    default="y",
    allow_cancel=True,
    timeout=None,
    header=None,
    log_func=None,
    session_id=None
):
    if header:
        print_header(header)
    prompt_str = f"{message} (y/n) [{default}]"
    if allow_cancel:
        prompt_str += " (type 'cancel' to abort)"
    prompt_str += ": "
    while True:
        if PROMPT_MODE == "webapp" and session_id:
            resp = prompt_user(prompt_str, session_id=session_id, timeout=timeout, default=default)
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
                log_warning("\n[Prompt] Timed out.")
                return default.lower() == "y"
            resp = result[0]
        else:
            resp = input(prompt_str)
        if resp is None or not resp.strip():
            resp = default
        resp = resp.strip().lower()
        if allow_cancel and resp == "cancel":
            if log_func:
                log_func(f"[PROMPT] User cancelled yes/no at {datetime.datetime.now()}")
            raise PromptCancelled("User cancelled the prompt.")
        if resp in ("y", "yes"):
            if log_func:
                log_func(f"[PROMPT] User input: YES at {datetime.datetime.now()}")
            return True
        if resp in ("n", "no"):
            if log_func:
                log_func(f"[PROMPT] User input: NO at {datetime.datetime.now()}")
            return False
        log_info("Please enter 'y' or 'n'.")

def prompt_choice(
    message,
    options,
    default=None,
    allow_cancel=True,
    header=None,
    log_func=None,
    session_id=None
):
    if not options:
        raise ValueError("No options provided for selection.")
    if header:
        print_header(header)
    for idx, opt in enumerate(options):
        print(f"  [{idx}] {opt}")
    def validator(x):
        return x.isdigit() and 0 <= int(x) < len(options)
    selection = prompt_user_input(
        f"{message} (0-{len(options)-1})",
        default=str(default) if default is not None else "0",
        validator=validator,
        allow_cancel=allow_cancel,
        header=None,
        log_func=log_func,
        session_id=session_id
    )
    if log_func:
        log_func(f"[PROMPT] User selected option {selection} at {datetime.datetime.now()}")
    return options[int(selection)]

# --- Advanced Context Prompts ---
def prompt_for_metadata_field(field_name, suggestions=None, default=None, allow_cancel=True, session_id=None):
    if suggestions:
        print(f"Suggestions for {field_name}:")
        for idx, s in enumerate(suggestions):
            print(f"  [{idx}] {s}")
        def validator(x):
            return (x.isdigit() and 0 <= int(x) < len(suggestions)) or bool(x.strip())
        response = prompt_user_input(
            f"Enter {field_name} or select a suggestion (0-{len(suggestions)-1}):",
            default=str(default) if default is not None else "",
            validator=validator,
            allow_cancel=allow_cancel,
            session_id=session_id
        )
        if response.isdigit():
            return suggestions[int(response)]
        return response
    else:
        return prompt_user_input(
            f"Enter {field_name}:",
            default=default,
            allow_cancel=allow_cancel,
            session_id=session_id
        )

def prompt_for_metadata(metadata_fields, session_id=None):
    responses = {}
    for field, opts in metadata_fields.items():
        responses[field] = prompt_for_metadata_field(
            field,
            suggestions=opts.get("suggestions", []),
            default=opts.get("default", []),
            session_id=session_id
        )
    return responses

def prompt_review_context(context, session_id=None):
    log_info("\n[Context Review]")
    for k, v in context.items():
        log_info(f"  {k}: {v}")
    if prompt_yes_no("Is this context correct?", default="y", session_id=session_id):
        return context
    for k in context:
        if prompt_yes_no(f"Edit {k}? (current: {context[k]})", default="n", session_id=session_id):
            context[k] = prompt_user_input(f"Enter new value for {k}:", default=str(context[k]), session_id=session_id)
    return context

def prompt_resolve_conflict(conflict_type, options, session_id=None):
    log_info(f"\n[Conflict Detected: {conflict_type}]")
    for idx, opt in enumerate(options):
        log_info(f"  [{idx}] {opt}")
    idx = prompt_user_input(
        f"Select the correct option (0-{len(options)-1}):",
        validator=lambda x: x.isdigit() and 0 <= int(x) < len(options),
        session_id=session_id
    )
    return options[int(idx)]

def prompt_user_for_button(page, candidates, toggle_name, session_id=None):
    log_info(f"\n[FEEDBACK] Please select the correct button for '{toggle_name}':")
    for idx, btn in enumerate(candidates):
        print(
            f"{idx}: label='{btn.get('label', '')}'"
            f" | class='{btn.get('class', '')}'"
            f" | tag='{btn.get('tag', '')}'"
            f" | context_heading='{btn.get('context_heading', '')}'"
            f" | context_anchor='{btn.get('context_anchor', '')}'"
            f" | visible={btn.get('is_visible', False)}"
            f" | enabled={btn.get('is_clickable', False)}"
        )
    try:
        choice = prompt_user_input("Enter the number of the correct button (or -1 to skip): ", session_id=session_id)
        choice = int(choice)
        if 0 <= choice < len(candidates):
            chosen_btn = candidates[choice]
            log_info(f"[bold green][FEEDBACK] You selected: '{chosen_btn.get('label', '')}'[/bold green]")
            return chosen_btn, choice
        else:
            log_warning("[yellow][FEEDBACK] Skipped manual correction.[/yellow]")
            return None, None
    except Exception as e:
        log_error(f"[red][FEEDBACK ERROR] {e}[/red]")
        return None, None

def confirm_button_callback(candidate, session_id=None):
    label = candidate.get("label", "")
    selector = candidate.get("selector", "")
    log_info(f"\n[CONFIRMATION] Candidate button found: '{label}'\nSelector: {selector}")
    try:
        resp = prompt_user_input(
            f"Do you want to click this button? (y/n): ",
            default="y",
            validator=lambda x: x.lower() in {"y", "n", "yes", "no"},
            allow_cancel=True,
            header="BUTTON CONFIRMATION",
            session_id=session_id
        ).strip().lower()
    except PromptCancelled:
        log_warning("[yellow]Button confirmation cancelled by user.[/yellow]")
        return False
    return resp in {"y", "yes"}