# user_prompt.py
# ===================================================================
# User Prompt Utilities for Interactive CLI Pipelines
# -------------------------------------------------------------------
# - Robust input with validation, timeout, and cancellation
# - Yes/No and choice selection helpers
# - Context-driven metadata and conflict prompts
# - Designed for integration in multi-step pipelines
# ===================================================================

import sys
import threading
import datetime
from utils.user_prompt import prompt_user_input, prompt_yes_no, prompt_choice, PromptCancelled
from utils.shared_logger import log_warning
class PromptCancelled(Exception):
    """Raised when the user cancels a prompt."""
    pass

def prompt_user_for_label(contest, prompt_func=prompt_user_input):
    try:
        label = prompt_func(
            f"Review contest: {contest['title']}\nLabel as (1) correct, (0) incorrect, or type annotation:",
            allow_cancel=True
        )
        return label
    except PromptCancelled:
        log_warning("User cancelled the labeling prompt.", context=contest)
        return None

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
    Returns True for yes, False for no.
    """
    if header:
        print_header(header)
    prompt_str = f"{message} (y/n) [{default}]"
    if allow_cancel:
        prompt_str += " (type 'cancel' to abort)"
    prompt_str += ": "
    while True:
        if timeout:
            # Timeout logic
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
                print("\n[Prompt] Timed out.")
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

# =========================
# Advanced Context Prompts
# =========================

def prompt_for_metadata_field(field_name, suggestions=None, default=None, allow_cancel=True):
    """
    Prompt the user to fill in a missing metadata field, with optional suggestions.
    """
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
            allow_cancel=allow_cancel
        )
        if response.isdigit():
            return suggestions[int(response)]
        return response
    else:
        return prompt_user_input(
            f"Enter {field_name}:",
            default=default,
            allow_cancel=allow_cancel
        )

def prompt_for_metadata(metadata_fields):
    """
    Prompt the user for multiple metadata fields.
    metadata_fields: dict of {field_name: {"suggestions": [...], "default": ...}}
    Returns a dict of user responses.
    """
    responses = {}
    for field, opts in metadata_fields.items():
        responses[field] = prompt_for_metadata_field(
            field,
            suggestions=opts.get("suggestions"),
            default=opts.get("default")
        )
    return responses

def prompt_review_context(context):
    """
    Display context summary and prompt user to confirm or edit.
    """
    print("\n[Context Review]")
    for k, v in context.items():
        print(f"  {k}: {v}")
    if prompt_yes_no("Is this context correct?", default="y"):
        return context
    # Optionally allow editing fields
    for k in context:
        if prompt_yes_no(f"Edit {k}? (current: {context[k]})", default="n"):
            context[k] = prompt_user_input(f"Enter new value for {k}:", default=str(context[k]))
    return context

def prompt_resolve_conflict(conflict_type, options):
    """
    Prompt the user to resolve a detected conflict.
    """
    print(f"\n[Conflict Detected: {conflict_type}]")
    for idx, opt in enumerate(options):
        print(f"  [{idx}] {opt}")
    idx = prompt_user_input(
        f"Select the correct option (0-{len(options)-1}):",
        validator=lambda x: x.isdigit() and 0 <= int(x) < len(options)
    )
    return options[int(idx)]