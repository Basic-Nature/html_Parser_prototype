def prompt_user_input(message, default=None, validator=None):
    """
    Prompt the user for input, with optional default and validation.
    """
    while True:
        prompt = f"{message}"
        if default is not None:
            prompt += f" [{default}]"
        prompt += " "
        response = input(prompt)
        if not response and default is not None:
            response = default
        if validator:
            try:
                if validator(response):
                    return response
            except Exception:
                pass
            print("Invalid input. Please try again.")
        else:
            return response

def prompt_yes_no(message, default="y"):
    """
    Prompt the user for a yes/no answer.
    """
    while True:
        resp = input(f"{message} (y/n) [{default}]: ").strip().lower()
        if not resp and default:
            resp = default
        if resp in ("y", "yes"):
            return True
        if resp in ("n", "no"):
            return False
        print("Please enter 'y' or 'n'.")