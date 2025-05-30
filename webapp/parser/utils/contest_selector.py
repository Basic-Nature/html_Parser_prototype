import re
from ..utils.shared_logger import rprint
from ..utils.shared_logic import normalize_state_name, normalize_county_name
from ..utils.logger_instance import logger
from ..utils.user_prompt import prompt_user_input, PromptCancelled
from collections import defaultdict

from typing import TYPE_CHECKING, List, Dict, Any, Optional
if TYPE_CHECKING:
    from ..Context_Integration.context_coordinator import ContextCoordinator

def normalize_race_name(name):
    import re
    return re.sub(r"\W+", "", name.strip().lower()) if name else ""

def normalize_contest_title(title: str) -> str:
    if not title:
        return ""
    title = re.sub(r'\s*[\r\n]*Vote for \d+\s*', '', title, flags=re.IGNORECASE)
    return title.strip()

def ml_verify_contest(contest: Dict[str, Any], coordinator: "ContextCoordinator", context: dict, threshold: float = 0.85) -> bool:
    """
    Use ML/NER to verify if the contest's year/type/title are likely correct.
    Returns True if above threshold, False otherwise.
    """
    title = contest.get("title", "")
    year = contest.get("year", "")
    ctype = contest.get("type", "")
    # Score year: must match a valid election year, not a timestamp
    year_score = 0.0
    if year and re.match(r"^(19|20)\d{2}$", str(year)):
        year_score = 1.0
    else:
        # Try to extract year from title using NER
        entities = coordinator.extract_entities(title)
        for ent, label in entities:
            if label == "DATE" and re.match(r"^(19|20)\d{2}$", ent):
                year_score = 0.9
                break
    # Score type: must match known election types
    known_types = [t.lower() for t in coordinator.get_election_types()]
    type_score = 1.0 if ctype and ctype.lower() in known_types else 0.0
    # Score title: must contain a known contest/election phrase
    contest_keywords = ["president", "senate", "congress", "governor", "mayor", "school board", "proposition", "referendum", "assembly", "council", "trustee", "justice", "clerk"]
    title_score = 1.0 if any(kw in title.lower() for kw in contest_keywords) else 0.0
    # ML/NER score for the whole header
    ml_score = coordinator.score_header(title, context)
    # Weighted average
    score = 0.4 * year_score + 0.2 * type_score + 0.2 * title_score + 0.2 * ml_score
    return score >= threshold

def feedback_loop_verify_contests(contests: List[Dict[str, Any]], coordinator: "ContextCoordinator", context: dict, max_loops: int = 3, threshold: float = 0.85) -> List[Dict[str, Any]]:
    """
    Feedback loop: rescans and verifies contests using ML/NER, retries if below threshold.
    Prompts user for clarification if still ambiguous after max_loops.
    """
    for loop in range(max_loops):
        verified = []
        for c in contests:
            if ml_verify_contest(c, coordinator, context, threshold=threshold):
                verified.append(c)
        if verified:
            logger.info(f"[CONTEST SELECTOR] Feedback loop {loop+1}: {len(verified)} contests passed ML/NER verification.")
            return verified
        logger.warning(f"[CONTEST SELECTOR] Feedback loop {loop+1}: No contests passed ML/NER verification. Retrying...")
    # If still ambiguous, prompt user for clarification
    rprint("[yellow]Unable to confidently identify valid contests after feedback loop. Please clarify selection.[/yellow]")
    grouped = defaultdict(list)
    for idx, c in enumerate(contests):
        grouped[(c.get('year', ''), c.get('type', ''))].append((idx, c))

    for (year, ctype), items in sorted(grouped.items()):
        rprint(f"[bold cyan]Year: {year or 'Unknown'}, Type: {ctype or 'Unknown'}[/bold cyan]")
        for idx, c in items:
            rprint(f"  [{idx}] {c.get('title', '')}")
    try:
        choice = prompt_user_input(
            "[PROMPT] Enter contest indices (comma-separated), 'all', 'skip', or leave blank to skip: ",
            default="all",
            validator=lambda x: x == "all" or x == "skip" or all(
                p.strip().isdigit() and 0 <= int(p.strip()) < len(contests)
                for p in x.split(",") if p.strip()
            ),
            allow_cancel=True,
            header="CONTEST FEEDBACK",
        ).strip().lower()
    except PromptCancelled:
        rprint("[yellow]Contest selection cancelled by user.[/yellow]")
        return []
    if not choice or choice == "skip":
        rprint("[yellow]No contest selected. Skipping.[/yellow]")
        return []
    if choice == "all":
        return contests
    indices = []
    for part in choice.split(","):
        part = part.strip()
        if part.isdigit():
            idx = int(part)
            if 0 <= idx < len(contests):
                indices.append(idx)
    selected = [contests[i] for i in indices]
    # Log user feedback for ML improvement
    for c in selected:
        coordinator.submit_user_feedback("contest", "contest_title", c.get("title", ""), context)
    return selected

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
    Uses ML/NER/regex feedback loop to verify correct year/type/title.
    Returns a list of selected contest dicts or None if skipped/cancelled.
    """
    selector_data = coordinator.get_for_selector()
    contests = selector_data["contests"]
    noisy_patterns = selector_data["noisy_patterns"]
    norm_state = normalize_state_name(state)
    norm_county = normalize_county_name(county)

    # Filter contests by state/county/year and remove noisy patterns
    filtered_contests = [
        c for c in contests
        if (not norm_state or normalize_state_name(c.get("state", "")) == norm_state)
        and (not norm_county or normalize_county_name(c.get("county", "")) == norm_county)
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
        norm_title = normalize_contest_title(c.get("title", ""))
        key = (c.get("year"), c.get("type"), norm_title)
        if key not in seen:
            unique_contests.append(c)
            seen.add(key)
    filtered_contests = unique_contests

    if not filtered_contests:
        rprint("[yellow]No valid contests detected after deduplication. Skipping.[/yellow]")
        return None

    # --- Feedback loop: ML/NER verification of contests ---
    context = {"state": state, "county": county, "year": year}
    verified_contests = feedback_loop_verify_contests(filtered_contests, coordinator, context)
    if not verified_contests:
        rprint("[yellow]No contests passed ML/NER verification. Skipping.[/yellow]")
        return None

    # Group by (year, type)
    
    grouped = defaultdict(list)
    for c in verified_contests:
        grouped[(c.get("year"), c.get("type"))].append(c)

    # --- Dynamic titling for selection prompt ---
    idx = 0
    contest_indices = []
    for (year_val, etype), contests_in_group in sorted(grouped.items()):
        # If multiple years, show state/county as heading
        if len(grouped) > 1:
            label = f"{state or 'Unknown State'} {county or ''} {year_val or 'Unknown'} {etype or 'Unknown'}"
        else:
            label = f"{year_val or 'Unknown'} {etype or 'Unknown'}"
        rprint(f"[bold cyan]{label.strip()}[/bold cyan]")
        for c in contests_in_group:
            rprint(f"  [{idx}] {c.get('title', '')}")
            contest_indices.append(c)
            idx += 1
    logger.debug(f"[DEBUG] Number of contests displayed: {idx}")

    # Auto-select if only one contest
    if len(verified_contests) == 1:
        rprint(f"[green]Only one contest found. Auto-selecting: {verified_contests[0]['title']}[/green]")
        if log_func:
            log_func(f"[CONTEST] Auto-selected: {verified_contests[0]['title']}")
        return [verified_contests[0]]

    if non_interactive:
        if log_func:
            log_func(f"[CONTEST] Non-interactive mode: selecting all contests.")
        return verified_contests

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
        return verified_contests

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
            {"title": "2024 General Election - Miami", "year": 2024, "type": "General", "state": "Florida"},
            {"title": "2022 Special Election - Miami", "year": 2022, "type": "Special", "state": "Florida"},
        ],
        "buttons": [
            {"label": "Show Results", "is_clickable": True, "is_visible": True},
            {"label": "Vote Method", "is_clickable": True, "is_visible": True},
            {"label": "Summary", "is_clickable": True, "is_visible": True}
        ]
    }
    coordinator = ContextCoordinator()
    coordinator.organize_and_enrich(sample_context)
    selected = select_contest(coordinator, state="Florida", year=2024)
    rprint(f"[bold green]Selected contests:[/bold green] {selected}")