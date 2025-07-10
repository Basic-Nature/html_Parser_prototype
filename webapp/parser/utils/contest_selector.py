import re
from ..utils.shared_logger import log_info, log_debug, log_warning
from ..utils.shared_logic import normalize_state_name, normalize_county_name
from ..utils.user_prompt import UserPrompt, PromptCancelled
from collections import defaultdict
from ..bots.librarian import (
    ELECTION_TYPES, CONTEST_KEYWORDS, KNOWN_COUNTY_TO_PRECINCTS_MAP
    )
user_prompt = UserPrompt()
from typing import TYPE_CHECKING, List, Dict, Any, Optional
if TYPE_CHECKING:
    from ..Context_Integration.context_coordinator import ContextCoordinator
coordinator = ContextCoordinator()

def extract_year_from_title(title) -> Optional[int]:
    import re
    if not title:
        return None
    # Find all years
    years = [int(y) for y in re.findall(r"(19|20)\d{2}", title)]
    if not years:
        return None
    # Lowercase title for type search
    title_lower = title.lower()
    # Find all valid types and their positions
    type_positions = []
    for t in ELECTION_TYPES:
        for m in re.finditer(re.escape(t), title_lower):
            type_positions.append((m.start(), t))
    # If no types, return the most recent year
    if not type_positions:
        return max(years)
    # Find year closest to a type
    best_year = None
    min_distance = float("inf")
    for y in years:
        for pos, t in type_positions:
            # Find position of year in string
            y_match = re.search(str(y), title)
            if y_match:
                dist = abs(y_match.start() - pos)
                if dist < min_distance:
                    min_distance = dist
                    best_year = y
    return best_year if best_year else max(years)

def normalize_race_name(name) -> str:
    import re
    return re.sub(r"\W+", "", name.strip().lower()) if name else ""

def normalize_contest_title(title: str) -> str:
    if not title:
        return ""
    title = re.sub(r'\s*[\r\n]*Vote for \d+\s*', '', title, flags=re.IGNORECASE)
    return title.strip()

def ml_verify_contest(contest: Dict[str, Any], coordinator: "ContextCoordinator", context: dict, threshold: float = 0.75) -> bool:
    """
    Use ML/NER to verify if the contest's year/type/title are likely correct.
    Returns True if above threshold, False otherwise.
    """
    title = contest.get("title", "")
    year = contest.get("year", "")
    ctype = contest.get("type_", "")
    year_score = 0.0
    if year and re.match(r"^(19|20)\d{2}$", str(year)):
        year_score = 1.0
    else:
        entities = coordinator.extract_entities(title)
        for ent, label in entities:
            if label == "DATE" and re.match(r"^(19|20)\d{2}$", ent):
                year_score = 0.9
                break

    # --- Election type detection ---
    known_types = [t.lower() for t in coordinator.get_election_types()]
    ctype_norm = ctype.lower().replace("election", "").strip()
    # Accept common election types even if not in known_types
    type_score = 0.0
    if ctype:
        if any(t in ctype_norm for t in known_types):
            type_score = 1.0
        elif any(v in ctype_norm for v in ELECTION_TYPES):
            type_score = 1.0
        else:
            # Partial match (e.g., "general" in "general election")
            if any(v in ctype_norm for v in ["general", "primary", "presidential", "special", "runoff"]):
                type_score = 0.8

    # --- Contest keywords: for office/position, not election type ---

    title_score = 1.0 if any(kw in title.lower() for kw in CONTEST_KEYWORDS) else 0.0

    # --- ML/NER header score ---
    ml_score = coordinator.score_header(title, context)

    # --- Tune weights: prioritize year and type ---
    score = 0.45 * year_score + 0.35 * type_score + 0.1 * title_score + 0.1 * ml_score

    if score < threshold:
        log_debug(f"[DEBUG][ml_verify_contest] Rejected contest: '{title}' | year: {year} | type_: {ctype}")
        log_info(f"  year_score={year_score}, type_score={type_score}, title_score={title_score}, ml_score={ml_score}, total={score:.2f}")
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
            log_info(f"[CONTEST SELECTOR] Feedback loop {loop+1}: {len(verified)} contests passed ML/NER verification.")
            return verified
        log_warning(f"[CONTEST SELECTOR] Feedback loop {loop+1}: No contests passed ML/NER verification. Retrying...")
    # If still ambiguous, prompt user for clarification
    log_warning("[yellow]Unable to confidently identify valid contests after feedback loop. Please clarify selection.[/yellow]")
    grouped = defaultdict(list)
    for idx, c in enumerate(contests):
        grouped[(c.get('year', ''), c.get('type_', ''))].append((idx, c))

    for (year, ctype), items in sorted(grouped.items()):
        log_info(f"[bold cyan]Year: {year or 'Unknown'}, Type: {ctype or 'Unknown'}[/bold cyan]")
        for idx, c in items:
            log_info(f"  [{idx}] {c.get('title', '')}")
    try:
        choice = user_prompt.prompt_input(
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
        log_warning("[yellow]Contest selection cancelled by user.[/yellow]")
        return []
    if not choice or choice == "skip":
        log_warning("[yellow]No contest selected. Skipping.[/yellow]")
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
    log_debug(f"norm_state: {context.get('state')}, norm_county: {context.get('county')}, year: {context.get('year')}")
    for c in selected:
        log_debug(f"Contest: {c.get('title', '')}, state: {c.get('state', '')}, county: {c.get('county', '')}, year: {c.get('year', '')}")
        coordinator.submit_user_feedback("contest", "contest_title", c.get("title", ""), context)
    return selected

def ensure_contest_title(contest) -> Dict[str, Any]:
    """
    Ensures the contest dict has a non-empty 'title' key.
    Falls back to 'name', or stringifies the contest if needed.
    """
    if not isinstance(contest, dict) or not contest:
        return {"title": str(contest)}
    title = contest.get("title")
    if title and isinstance(title, str) and title.strip():
        return contest
    # Try fallback keys
    for alt in ("name", "contest_name", "label"):
        alt_val = contest.get(alt)
        if alt_val and isinstance(alt_val, str) and alt_val.strip():
            contest["title"] = alt_val
            return contest
    # Fallback: stringified dict
    contest["title"] = str(contest)
    return contest

def select_contest(
    coordinator: "ContextCoordinator",
    state=None,
    county=None,
    year=None,
    prompt_message="[PROMPT] Enter contest indices (comma-separated), 'all', or leave blank to skip: ",
    allow_multiple=True,
    non_interactive=False,
    log_func=None,
    context=None
) -> Optional[List[Dict[str, Any]]]:
    """
    Prompts the user to select contests from the organized context, filtering out noisy/generic labels.
    Uses ML/NER/regex feedback loop to verify correct year/type/title.
    Returns a list of selected contest dicts or None if skipped/cancelled.
    """
    norm_state = normalize_state_name(state)
    norm_county = normalize_county_name(county)
    selector_data = coordinator.get_for_selector()
    contests = selector_data.get("contests", [])
    noisy_patterns = selector_data.get("noisy_patterns", [])
    known_county_to_precincts = KNOWN_COUNTY_TO_PRECINCTS_MAP

    # --- Fill in missing year/type from title using ML/NER ---
    for c in contests:
        # Fill year
        if not c.get("year"):
            year_from_title = extract_year_from_title(c.get("title", ""))
            if year_from_title:
                c["year"] = year_from_title
        # Fill type
        if not c.get("type_"):
            title = c.get("title", "").lower()
            found_type_ = None
            for t in ELECTION_TYPES:
                if t in title:
                    found_type_ = t
                    break
            if not found_type_:
                # Try ML/NER
                ents = coordinator.extract_entities(c.get("title", ""))
                for ent, label in ents:
                    if label == "EVENT" and ent.lower() in ELECTION_TYPES:
                        found_type_ = ent.lower()
                        break
            if found_type_:
                c["type_"] = found_type_.capitalize()

    context = {
        "state": norm_state,
        "county": norm_county,
        "year": year,
        "contests": contests,
        "url": getattr(coordinator, "last_url", None) if hasattr(coordinator, "last_url") else None
    }
    log_debug(f"DEBUG: selector_data['contests']: {selector_data.get('contests', None)}")
    log_debug(f"[DEBUG] norm_state: {norm_state}, norm_county: {norm_county}, year: {year}")
    log_debug(f"[DEBUG] noisy_patterns: {noisy_patterns}")
    log_debug(f"[DEBUG] contests before filtering: {contests}")

    # Helper for county matching
    def county_matches(contest_county):
        contest_county_norm = normalize_county_name(contest_county)
        if not norm_county:
            return True
        if contest_county_norm == norm_county:
            return True
        for parent_county, precincts in known_county_to_precincts.items():
            if normalize_county_name(parent_county) == norm_county:
                if contest_county_norm in [normalize_county_name(d) for d in precincts]:
                    return True
        return False

    # --- Filter contests ---
    filtered_contests = []
    for c in contests:
        skip_reason = None
        if norm_state and normalize_state_name(c.get("state", "")) != norm_state:
            skip_reason = "state mismatch"
        elif not county_matches(c.get("county", "")):
            skip_reason = "county mismatch"
        elif year and str(c.get("year", "")) != str(year):
            skip_reason = "year mismatch"
        elif any(pat.lower() in c.get("title", "").lower() for pat in noisy_patterns):
            skip_reason = "noisy pattern"
        elif not c.get("title") or c.get("title", "").strip().lower() in ["", "results", "summary"]:
            skip_reason = "empty/generic title"
        if skip_reason:
            log_debug(f"Skipping contest '{c.get('title', '')}': {skip_reason}")
            continue
        filtered_contests.append(c)

    log_debug(f"[DEBUG] Filtered contests: {filtered_contests}")
    log_debug(f"[DEBUG] Number of filtered contests: {len(filtered_contests)}")
    if not filtered_contests:
        log_warning("[yellow]No valid contests detected after filtering. Skipping.[/yellow]")
        return None

    # --- Deduplicate by normalized title, year, and type ---
    unique_contests = []
    seen = set()
    for c in filtered_contests:
        norm_title = normalize_contest_title(c.get("title", ""))
        key = (c.get("year"), c.get("type_"), norm_title)
        if key not in seen:
            unique_contests.append(c)
            seen.add(key)
    filtered_contests = unique_contests

    if not filtered_contests:
        log_warning("[yellow]No valid contests detected after deduplication. Skipping.[/yellow]")
        return None

    # --- ML/NER verification ---
    verified_contests = []
    for c in filtered_contests:
        if ml_verify_contest(c, coordinator, context, threshold=0.75):
            verified_contests.append(c)
    if not verified_contests:
        # Try feedback loop for user clarification
        verified_contests = feedback_loop_verify_contests(filtered_contests, coordinator, context)
        if not verified_contests:
            log_warning("[yellow]No contests passed ML/NER verification. Skipping.[/yellow]")
            return None

    # --- Group by (year, type) for display ---
    grouped = defaultdict(list)
    for c in verified_contests:
        grouped[(c.get("year"), c.get("type_"))].append(c)

    # --- Dynamic titling for selection prompt ---
    idx = 0
    contest_indices = []
    for (year_val, etype), contests_in_group in sorted(grouped.items()):
        if len(grouped) > 1:
            label = f"{state or 'Unknown State'} {county or ''} {year_val or 'Unknown'} {etype or 'Unknown'}"
        else:
            label = f"{year_val or 'Unknown'} {etype or 'Unknown'}"
        log_info(f"[bold cyan]{label.strip()}[/bold cyan]")
        for c in contests_in_group:
            log_info(f"  [{idx}] {c.get('title', '')}")
            contest_indices.append(c)
            idx += 1
    log_debug(f"[DEBUG] Number of contests displayed: {idx}")

    # --- Auto-select if only one contest ---
    if len(verified_contests) == 1:
        contest = ensure_contest_title(verified_contests[0])
        log_info(f"[green]Only one contest found. Auto-selecting: {contest['title']}[/green]")
        if log_func:
            log_func(f"[CONTEST] Auto-selected: {contest['title']}")
        return [contest]

    # --- Non-interactive mode: select all ---
    if non_interactive:
        if log_func:
            log_func(f"[CONTEST] Non-interactive mode: selecting all contests.")
        return [ensure_contest_title(c) for c in verified_contests]

    # --- Interactive prompt ---
    try:
        choice = user_prompt.prompt_input(
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
        log_warning("[yellow]Contest selection cancelled by user.[/yellow]")
        if log_func:
            log_func("[CONTEST] User cancelled contest selection.")
        return None

    if not choice:
        log_warning("[yellow]No contest selected. Skipping.[/yellow]")
        if log_func:
            log_func("[CONTEST] No contest selected.")
        return None

    if choice == "all":
        if log_func:
            log_func("[CONTEST] User selected all contests.")
        return [ensure_contest_title(c) for c in verified_contests]

    # --- Parse comma-separated indices ---
    indices = []
    for part in choice.split(","):
        part = part.strip()
        if part.isdigit():
            idx = int(part)
            if 0 <= idx < len(contest_indices):
                indices.append(idx)
    if not indices:
        log_warning("[yellow]No valid contest indices selected. Skipping.[/yellow]")
        if log_func:
            log_func("[CONTEST] No valid contest indices selected.")
        return None

    selected = [ensure_contest_title(contest_indices[i]) for i in indices]
    if log_func:
        log_func(f"[CONTEST] User selected contests: {[c.get('title', '') for c in selected]}")
    return selected

