# user_prompt.py
# ===================================================================
# User Prompt Utilities for Interactive CLI Pipelines
# -------------------------------------------------------------------
# - Robust input with validation, timeout, and cancellation
# - Yes/No and choice selection helpers
# - Optional headers, logging, and error callbacks
# - Designed for integration in multi-step pipelines
# ===================================================================

import sys
import threading
import datetime

class PromptCancelled(Exception):
    """Raised when the user cancels a prompt."""
    pass

def print_header(title: str = "USER INPUT REQUIRED", char: str = "=", width: int = 60):
    """
    Prints a formatted header for prompt sections.
    """
    print("\n" + char * width)
    print(f"{title.center(width)}")
    print(char * width)

def prompt_user_input(
    message,
    default=None,
    validator=None,
    allow_cancel=True,
    timeout=None,
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
        response = input_with_timeout(prompt, timeout) if timeout else input(prompt)
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
            print("Invalid input. Please try again.")
            if attempts >= max_attempts:
                print("[Prompt] Too many invalid attempts. Cancelling.")
                if log_func:
                    log_func(f"[PROMPT] Too many invalid attempts at {datetime.datetime.now()}")
                raise PromptCancelled("Too many invalid attempts.")
        else:
            if log_func:
                log_func(f"[PROMPT] User input: {response} at {datetime.datetime.now()}")
            return response

def prompt_yes_no(
    message,
    default="y",
    allow_cancel=True,
    timeout=None,
    header=None,
    log_func=None
):
    """
    Prompt the user for a yes/no answer, with optional cancel, timeout, header, and logging.
    """
    if header:
        print_header(header)
    while True:
        prompt = f"{message} (y/n) [{default}]"
        if allow_cancel:
            prompt += " (type 'cancel' to abort)"
        prompt += ": "
        resp = input(prompt) if not timeout else None
        if timeout:
            # Could implement timeout logic here if needed
            pass
        if resp is None:
            resp = default
        resp = resp.strip().lower()
        if allow_cancel and resp == "cancel":
            if log_func:
                log_func(f"[PROMPT] User cancelled yes/no at {datetime.datetime.now()}")
            raise PromptCancelled("User cancelled the prompt.")
        if not resp and default:
            resp = default
        if resp in ("y", "yes"):
            if log_func:
                log_func(f"[PROMPT] User input: YES at {datetime.datetime.now()}")
            return True
        if resp in ("n", "no"):
            if log_func:
                log_func(f"[PROMPT] User input: NO at {datetime.datetime.now()}")
            return False
        print("Please enter 'y' or 'n'.")

def prompt_choice(
    message,
    options,
    default=None,
    allow_cancel=True,
    header=None,
    log_func=None
):
    """
    Prompt the user to select from a list of options.
    Returns the selected option or raises PromptCancelled.
    """
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
        log_func=log_func
    )
    if log_func:
        log_func(f"[PROMPT] User selected option {selection} at {datetime.datetime.now()}")
    return options[int(selection)]