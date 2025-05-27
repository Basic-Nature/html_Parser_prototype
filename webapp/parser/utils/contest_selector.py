from ..utils.shared_logger import rprint
from ..utils.logger_instance import logger
from ..utils.user_prompt import prompt_user_input, PromptCancelled

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..Context_Integration.context_coordinator import ContextCoordinator

def normalize_race_name(name):
    # Simple normalization for deduplication
    import re
    return re.sub(r"\W+", "", name.strip().lower()) if name else ""

def select_contest(
    coordinator: "ContextCoordinator",
    state=None,
    county=None,
    year=None,
    prompt_message="[PROMPT] Enter contest indices (comma-separated), 'all', or leave blank to skip: ",
    allow_multiple=True,
    non_interactive=False,
    log_func=None
):
    """
    Prompts the user to select contests from the organized context, filtering out noisy/generic labels.
    Returns a list of selected contest dicts or None if skipped/cancelled.
    """
    selector_data = coordinator.get_for_selector()
    contests = selector_data["contests"]
    noisy_patterns = selector_data["noisy_patterns"]

    # Filter contests by state/county/year and remove noisy patterns
    filtered_contests = [
        c for c in contests
        if (not state or c.get("state", "").lower() == state.lower())
        and (not county or c.get("county", "").lower() == county.lower())
        and (not year or str(c.get("year", "")) == str(year))
        and not any(pat.lower() in c.get("title", "").lower() for pat in noisy_patterns)
    ]
    logger.debug(f"[DEBUG] Filtered contests: {filtered_contests}")
    logger.debug(f"[DEBUG] Number of filtered contests: {len(filtered_contests)}")
    if not filtered_contests:
        rprint("[yellow]No valid contests detected after filtering. Skipping.[/yellow]")
        return None

    # Deduplicate by normalized race name, year, and type
    unique_contests = []
    seen = set()
    for c in filtered_contests:
        norm = normalize_race_name(c.get("title", ""))
        key = (c.get("year"), c.get("type"), norm)
        if key not in seen:
            unique_contests.append(c)
            seen.add(key)
    filtered_contests = unique_contests

    if not filtered_contests:
        rprint("[yellow]No valid contests detected after deduplication. Skipping.[/yellow]")
        return None

    # Group by (year, type)
    from collections import defaultdict
    grouped = defaultdict(list)
    for c in filtered_contests:
        grouped[(c.get("year"), c.get("type"))].append(c)

    # Display headings and contests
    idx = 0
    contest_indices = []
    for (year, etype), contests_in_group in sorted(grouped.items()):
        label = f"{year or 'Unknown'} {etype or 'Unknown'}"
        rprint(f"[bold cyan]{label}[/bold cyan]")
        for c in contests_in_group:
            rprint(f"  [{idx}] {c.get('title', '')}")
            contest_indices.append(c)
            idx += 1
    logger.debug(f"[DEBUG] Number of contests displayed: {idx}")

    # Auto-select if only one contest
    if len(filtered_contests) == 1:
        rprint(f"[green]Only one contest found. Auto-selecting: {filtered_contests[0]['title']}[/green]")
        if log_func:
            log_func(f"[CONTEST] Auto-selected: {filtered_contests[0]['title']}")
        return [filtered_contests[0]]

    if non_interactive:
        if log_func:
            log_func(f"[CONTEST] Non-interactive mode: selecting all contests.")
        return filtered_contests

    try:
        choice = prompt_user_input(
            prompt_message,
            default="all",
            validator=lambda x: x == "all" or all(
                p.strip().isdigit() and 0 <= int(p.strip()) < len(contest_indices)
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
        return filtered_contests

    # Parse comma-separated indices
    indices = []
    for part in choice.split(","):
        part = part.strip()
        if part.isdigit():
            idx = int(part)
            if 0 <= idx < len(contest_indices):
                indices.append(idx)
    if not indices:
        rprint("[yellow]No valid contest indices selected. Skipping.[/yellow]")
        if log_func:
            log_func("[CONTEST] No valid contest indices selected.")
        return None

    selected = [contest_indices[i] for i in indices]
    if log_func:
        log_func(f"[CONTEST] User selected contests: {[c.get('title', '') for c in selected]}")
    return selected

# Example usage (for testing/demo):
if __name__ == "__main__":
    from ..Context_Integration.context_coordinator import ContextCoordinator

    # Simulate loading and organizing a sample context
    sample_context = {
        "contests": [
            {"title": "2024 Presidential Election - New York", "year": 2024, "type": "Presidential", "state": "New York"},
            {"title": "2022 Senate Race - California", "year": 2022, "type": "Senate", "state": "California"},
            {"title": "2024 Mayoral Election - Houston, TX", "year": 2024, "type": "Mayoral", "state": "Texas"},
            {"title": "2023 School Board - Miami", "year": 2023, "type": "School Board", "state": "Florida"},
        ],
        "buttons": [
            {"label": "Show Results", "is_clickable": True, "is_visible": True},
            {"label": "Vote Method", "is_clickable": True, "is_visible": True},
            {"label": "Summary", "is_clickable": True, "is_visible": True}
        ]
    }
    coordinator = ContextCoordinator()
    coordinator.organize_and_enrich(sample_context)
    selected = select_contest(coordinator, state="New York", year=2024)
    rprint(f"[bold green]Selected contests:[/bold green] {selected}")