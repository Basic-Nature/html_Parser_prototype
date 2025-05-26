from rich import print as rprint
from ..utils.shared_logger import logger
from ..utils.user_prompt import prompt_user_input, PromptCancelled
import re
import json
import os

# Load noisy label config from context library
CONTEXT_LIBRARY_PATH = os.path.join(
    os.path.dirname(__file__), "..", "Context_Integration", "context_library.json"
)
if os.path.exists(CONTEXT_LIBRARY_PATH):
    with open(CONTEXT_LIBRARY_PATH, "r", encoding="utf-8") as f:
        CONTEXT_LIBRARY = json.load(f)
    DEFAULT_NOISY_LABELS = [label.lower() for label in CONTEXT_LIBRARY.get("default_noisy_labels", [])]
    DEFAULT_NOISY_LABEL_PATTERNS = CONTEXT_LIBRARY.get("default_noisy_label_patterns", [])
else:
    logger.error("[contest_selector] context_library.json not found. Noisy label filtering will be limited.")
    DEFAULT_NOISY_LABELS = []
    DEFAULT_NOISY_LABEL_PATTERNS = []

def is_noisy_label(label: str, noisy_labels=None, noisy_label_patterns=None) -> bool:
    """Check if a label is considered noisy based on patterns or substrings."""
    noisy_labels = noisy_labels or DEFAULT_NOISY_LABELS
    noisy_label_patterns = noisy_label_patterns or DEFAULT_NOISY_LABEL_PATTERNS
    label = label.lower()
    for pattern in noisy_label_patterns:
        if re.search(pattern, label):
            return True
    for noisy in noisy_labels:
        if noisy in label:
            return True
    return False

def extract_election_types(races):
    """Extract possible election types from the list of races."""
    types = set()
    for race in races:
        if isinstance(race, tuple) and len(race) == 3:
            types.add(race[1].strip().lower())
        else:
            m = re.match(r'(19|20)\d{2}\s+([a-z ]+?)( election)?$', race.lower())
            if m:
                types.add(m.group(2).strip())
    return types

def is_noisy_contest_label(label: str, election_types=None, noisy_label_patterns=None) -> bool:
    label = label.strip().lower()
    if not election_types:
        election_types = {"general", "primary", "runoff", "special"}
    types_pattern = "|".join(re.escape(t.strip().lower()) for t in election_types)
    label = re.sub(r'\s+', ' ', label)
    pattern = rf'^((19|20)\d{{2}}\s+({types_pattern})( election)?\s*)+$'
    if noisy_label_patterns:
        for pat in noisy_label_patterns:
            if re.fullmatch(pat, label, re.IGNORECASE):
                return True
    return bool(re.fullmatch(pattern, label, re.IGNORECASE))

def normalize_race_name(race):
    """Remove 'Vote for X' and extra whitespace, lowercased."""
    race = re.sub(r"\s*Vote for \d+\s*$", "", race, flags=re.IGNORECASE)
    return race.strip().lower()

def select_contest(
    detected_races,
    prompt_message="[PROMPT] Enter contest indices (comma-separated), 'all', or leave blank to skip: ",
    allow_multiple=True,
    noisy_labels=None,
    noisy_label_patterns=None,
    non_interactive=False,
    log_func=None
):
    """
    Prompts the user to select contests from detected races, filtering out noisy/generic labels.
    Returns a list of selected contest tuples or None if skipped/cancelled.
    """
    election_types = extract_election_types(detected_races)
    filtered_races = [
        (year, etype, race) for (year, etype, race) in detected_races
        if not is_noisy_contest_label(race, election_types, noisy_label_patterns)
        and not is_noisy_label(race, noisy_labels, noisy_label_patterns)
    ]
    logger.debug(f"[DEBUG] Filtered races (full): {filtered_races}")
    logger.debug(f"[DEBUG] Number of filtered races (full): {len(filtered_races)}")
    if not filtered_races:
        rprint("[yellow]No valid contests detected after filtering. Skipping.[/yellow]")
        return None

    # Deduplicate by normalized race name
    unique_races = []
    seen = set()
    for year, etype, race in filtered_races:
        norm = normalize_race_name(race)
        key = (year, etype, norm)
        if key not in seen:
            unique_races.append((year, etype, race))
            seen.add(key)
    filtered_races = unique_races

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
        label = f"{year or 'Unknown'} {etype or 'Unknown'}"
        rprint(f"[bold cyan]{label}[/bold cyan]")
        for race in races:
            rprint(f"  [{idx}] {race}")
            idx += 1
    logger.debug(f"[DEBUG] Number of races displayed: {idx}")

    # Auto-select if only one contest
    if len(filtered_races) == 1:
        rprint(f"[green]Only one contest found. Auto-selecting: {filtered_races[0]}[/green]")
        if log_func:
            log_func(f"[CONTEST] Auto-selected: {filtered_races[0]}")
        return [filtered_races[0]]

    if non_interactive:
        if log_func:
            log_func(f"[CONTEST] Non-interactive mode: selecting all contests.")
        return filtered_races

    try:
        choice = prompt_user_input(
            prompt_message,
            default="all",
            validator=lambda x: x == "all" or all(
                p.strip().isdigit() and 0 <= int(p.strip()) < len(filtered_races)
                for p in x.split(",") if p.strip()
            ),
            allow_cancel=True,
            header="CONTEST SELECTION",
            log_func=log_func
        ).strip().lower()
    except PromptCancelled:
        rprint("[yellow]Contest selection cancelled by user.[/yellow]")
        if log_func:
            log_func("[CONTEST] User cancelled contest selection.")
        return None

    if not choice:
        rprint("[yellow]No contest selected. Skipping.[/yellow]")
        if log_func:
            log_func("[CONTEST] No contest selected.")
        return None

    if choice == "all":
        if log_func:
            log_func("[CONTEST] User selected all contests.")
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
        if log_func:
            log_func("[CONTEST] No valid contest indices selected.")
        return None

    selected = [filtered_races[i] for i in indices]
    if log_func:
        log_func(f"[CONTEST] User selected contests: {selected}")
    return selected