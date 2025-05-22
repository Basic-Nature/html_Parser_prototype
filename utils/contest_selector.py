# utiils/contest_selector.py
# ===================================================================
# Election Data Cleaner
# This script is part of the Election Data Cleaner project, which is licensed under the MIT License.
# ===================================================================
from rich import print as rprint

def select_contest(detected_races, prompt_message="[PROMPT] Enter contest index or leave blank to skip: "):
    """
    Presents a list of contest races to the user and prompts for selection.
    Returns the selected contest title, or None if skipped.
    """
    if not detected_races:
        rprint("[yellow]No contests detected. Skipping.[/yellow]")
        return None

    rprint("[bold #eb4f43]Contest Races Detected:[/bold #eb4f43]")
    for idx, race in enumerate(detected_races):
        rprint(f"  [{idx}] {race}")

    choice = input(prompt_message).strip()
    if not choice or not choice.isdigit() or int(choice) >= len(detected_races):
        rprint("[yellow]No contest selected. Skipping.[/yellow]")
        return None

    return detected_races[int(choice)]