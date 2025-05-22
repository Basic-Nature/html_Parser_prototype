# utiils/contest_selector.py
# ===================================================================
# Election Data Cleaner
# This script is part of the Election Data Cleaner project, which is licensed under the MIT License.
# ===================================================================
from rich import print as rprint
from utils.shared_logger import logger
import re

NOISY_LABELS = [
    "view results by election district",
    "summary by method",
    "download",
    "vote method",
    "voting method",
]
NOISY_LABEL_PATTERNS = [
    r"view results? by election district\s*[:\n]?$",
    r"summary by method\s*[:\n]?$",
    r"download\s*[:\n]?$",
    r"vote method\s*[:\n]?$",
    r"voting method\s*[:\n]?$",
    r"^vote for \d+$"
]
NOISY_LABELS = [label.lower() for label in NOISY_LABELS]

def is_noisy_label(label: str) -> bool:
    """
    Check if a label is considered noisy based on predefined patterns.
    """
    label = label.lower()
    for pattern in NOISY_LABEL_PATTERNS:
        if re.search(pattern, label):
            return True
    # Check for exact matches with noisy labels
    if label in NOISY_LABELS:
        return True
    # Check for any noisy label in the string
    # This is a more lenient check, but it can be useful for certain cases
        
        
    return any(noisy in label for noisy in NOISY_LABELS)

def extract_election_types(races):
    """
    Dynamically extract possible election types from the list of races.
    Returns a set of types (e.g., {'general', 'primary', 'runoff', ...}).
    """
    types = set()
    for race in races:
       # If race is a tuple, get the etype directly
        if isinstance(race, tuple) and len(race) == 3:
            types.add(race[1].strip().lower())
        else:
            # fallback for string input        
        # Look for words after the year, e.g., "2024 General", "2024 Primary Election"
            m = re.match(r'(19|20)\d{2}\s+([a-z ]+?)( election)?$', race.lower())
            if m:
                types.add(m.group(2).strip())
    return types

def is_noisy_contest_label(label: str, election_types=None) -> bool:
    # Normalize label
    label = label.strip().lower()
    if not election_types:
        election_types = {"general", "primary", "runoff", "special"}
    # Normalize election_types to lowercase and strip spaces
    types_pattern = "|".join(re.escape(t.strip().lower()) for t in election_types)
    # Remove extra whitespace in label
    label = re.sub(r'\s+', ' ', label)
    # Match "2024 General", "2024 General Election", or repeated patterns, any case
    pattern = rf'^((19|20)\d{{2}}\s+({types_pattern})( election)?\s*)+$'
    return bool(re.fullmatch(pattern, label, re.IGNORECASE))

def select_contest(
    detected_races,
    prompt_message="[PROMPT] Enter contest indices (comma-separated), 'all', or leave blank to skip: ",
    allow_multiple=True
):

    # Dynamically extract election types from the list
    election_types = extract_election_types(detected_races)
    # Filter out noisy/generic labels
    filtered_races = [
        (year, etype, race) for (year, etype, race) in detected_races 
        if not is_noisy_contest_label(race)
        and not is_noisy_label(race)
    ]
    logger.debug(f"[DEBUG] Filtered races (full): {filtered_races}")
    logger.debug(f"[DEBUG] Number of filtered races (full): {len(filtered_races)}")
    if not filtered_races:
        rprint("[yellow]No valid contests detected after filtering. Skipping.[/yellow]")
        return None
    # Group by (year, etype)
    from collections import defaultdict
    grouped = defaultdict(list)
    for year, etype, race in filtered_races:
        grouped[(year, etype)].append(race)

    # Display headings and contests
    idx = 0
    for (year, etype), races in sorted(grouped.items()):
        rprint(f"[bold cyan]{year} {etype}[/bold cyan]")
        for race in races:
            rprint(f"  [{idx}] {race}")
            idx += 1
    logger.debug(f"[DEBUG] Number of races displayed: {idx}")    
    
    # Auto-select if only one contest
    if len(filtered_races) == 1:
        rprint(f"[green]Only one contest found. Auto-selecting: {filtered_races[0]}[/green]")
        return [filtered_races[0]]

    choice = input(prompt_message).strip().lower()
    if not choice:
        rprint("[yellow]No contest selected. Skipping.[/yellow]")
        return None

    if choice == "all":
        return filtered_races

    # Parse comma-separated indices
    indices = []
    for part in choice.split(","):
        part = part.strip()
        if part.isdigit():
            idx = int(part)
            if 0 <= idx < len(filtered_races):
                indices.append(idx)
    if not indices:
        rprint("[yellow]No valid contest indices selected. Skipping.[/yellow]")
        return None

    return [filtered_races[i] for i in indices]