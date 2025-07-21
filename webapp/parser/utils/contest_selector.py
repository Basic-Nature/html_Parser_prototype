import re
from ..utils.shared_logger import SharedLogger
from ..utils.shared_logic import normalize_state_name, normalize_county_name
from ..utils.user_prompt import UserPrompt, PromptCancelled
from collections import defaultdict
from difflib import get_close_matches
from ..bots.librarian import (
    ELECTION_TYPES, CONTEST_KEYWORDS, KNOWN_COUNTY_TO_PRECINCTS_MAP
    )

logger = SharedLogger()
prompt = UserPrompt()
from typing import TYPE_CHECKING, List, Dict, Any, Optional
if TYPE_CHECKING:
    from ..Context_Integration.context_coordinator import ContextCoordinator

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

def infer_election_type(title, context, contest, all_contests, coordinator) -> Optional[str]:
    """
    Dynamically infer the election type for a contest using:
    - Regex and fuzzy matching on the title
    - Context clues (year, known election dates, context type)
    - ML/NER entity extraction
    - Most common type among all contests for the same year/county
    """
    if not title:
        return None
    title_lower = title.lower()
    # 1. Regex for common election types
    regex_types = re.findall(r"\b(general|primary|special|runoff|municipal|presidential|school|bond|proposition|referendum)\b", title_lower)
    if regex_types:
        return regex_types[0].capitalize()
    # 2. Fuzzy match against ELECTION_TYPES
    close = get_close_matches(title_lower, [(t or "").lower() for t in ELECTION_TYPES], n=1, cutoff=0.8)
    if close:
        return close[0].capitalize()
    # 3. Use ML/NER
    if coordinator:
        ents = coordinator.extract_entities(title)
        for ent, label in ents:
            if label == "EVENT" and (ent or "").lower() in [(et or "").lower() for et in ELECTION_TYPES]:
                return ent.capitalize()
    # 4. Context clues: if context has a type, use it
    if context and context.get("type_"):
        return context["type_"].capitalize()
    # 5. If contest has a date, and it matches a known general/primary election date, infer type
    # (You can expand this with a lookup table of known election dates if available)
    # 6. Most common type among all contests for this year/county
    year = contest.get("year")
    county = contest.get("county")
    type_counts = defaultdict(int)
    for c in all_contests:
        if c.get("year") == year and c.get("county") == county and c.get("type_"):
            type_counts[c["type_"].lower()] += 1
    if type_counts:
        most_common = max(type_counts.items(), key=lambda x: x[1])[0]
        return most_common.capitalize()
    return None

def fuzzy_county_match(contest_county, norm_county, known_county_to_precincts) -> bool:
    """Return True if contest_county matches norm_county or any known precinct, using fuzzy matching."""
    contest_county_norm = normalize_county_name(contest_county)
    if not norm_county:
        return True
    if contest_county_norm == norm_county:
        return True
    # Fuzzy match against parent county and precincts
    all_names = [normalize_county_name(norm_county)]
    for parent_county, precincts in known_county_to_precincts.items():
        if normalize_county_name(parent_county) == norm_county:
            all_names += [normalize_county_name(d) for d in precincts]
    matches = get_close_matches(contest_county_norm, all_names, n=1, cutoff=0.85)
    if matches:
        logger.info(f"[FUZZY COUNTY MATCH] '{contest_county}' matched to '{matches[0]}' (target: '{norm_county}')")
        return True
    # If partial match, log for user review
    partials = [name for name in all_names if contest_county_norm in name or name in contest_county_norm]
    if partials:
        logger.info(f"[PARTIAL COUNTY MATCH] '{contest_county}' partially matched to '{partials[0]}' (target: '{norm_county}')")
        return True
    return False

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
    if coordinator is None:
        coordinator = ContextCoordinator()
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
    known_types = [(t or "").lower() for t in coordinator.get_election_types()]
    ctype_norm = (ctype or "").lower().replace("election", "").strip()
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

    title_score = 1.0 if any((kw or "") in (title or "").lower() for kw in CONTEST_KEYWORDS) else 0.0

    # --- ML/NER header score ---
    ml_score = coordinator.score_header(title, context)

    # --- Tune weights: prioritize year and type ---
    score = 0.45 * year_score + 0.35 * type_score + 0.1 * title_score + 0.1 * ml_score

    if score < threshold:
        logger.debug(f"[DEBUG][ml_verify_contest] Rejected contest: '{title}' | year: {year} | type_: {ctype}")
        logger.info(f"  year_score={year_score}, type_score={type_score}, title_score={title_score}, ml_score={ml_score}, total={score:.2f}")
    return score >= threshold

def feedback_loop_verify_contests(contests: List[Dict[str, Any]], coordinator: "ContextCoordinator", context: dict, max_loops: int = 3, threshold: float = 0.85) -> List[Dict[str, Any]]:
    """
    Feedback loop: rescans and verifies contests using ML/NER, retries if below threshold.
    Prompts user for clarification if still ambiguous after max_loops.
    """
    if coordinator is None:
        coordinator = ContextCoordinator()
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
    logger.warning("[yellow]Unable to confidently identify valid contests after feedback loop. Please clarify selection.[/yellow]")
    grouped = defaultdict(list)
    for idx, c in enumerate(contests):
        grouped[(c.get('year', ''), c.get('type_', ''))].append((idx, c))

    for (year, ctype), items in sorted(grouped.items()):
        logger.info(f"[bold cyan]Year: {year or 'Unknown'}, Type: {ctype or 'Unknown'}[/bold cyan]")
        for idx, c in items:
            logger.info(f"  [{idx}] {c.get('title', '')}")
    try:
        choice = prompt.prompt_input(
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
        logger.warning("[yellow]Contest selection cancelled by user.[/yellow]")
        return []
    if not choice or choice == "skip":
        logger.warning("[yellow]No contest selected. Skipping.[/yellow]")
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
    logger.debug(f"norm_state: {context.get('state')}, norm_county: {context.get('county')}, year: {context.get('year')}")
    for c in selected:
        logger.debug(f"Contest: {c.get('title', '')}, state: {c.get('state', '')}, county: {c.get('county', '')}, year: {c.get('year', '')}")
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
    if coordinator is None:
        coordinator = ContextCoordinator()
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
        # Fill type_
        found_type_ = None
        if not c.get("type_"):
            inferred_type = infer_election_type(
                c.get("title", ""),
                context,
                c,
                contests,
                coordinator
            )
            if inferred_type:
                c["type_"] = inferred_type
                # Try ML/NER
                ents = coordinator.extract_entities(c.get("title", "")) if coordinator else []
                for ent, label in ents:
                    if label == "EVENT" and (ent or "").lower() in [(et or "").lower() for et in ELECTION_TYPES]:
                        found_type_ = (ent or "").lower()
                        break
        # PATCH: Only use found_type_ if set
        if found_type_:
            c["type_"] = found_type_.capitalize()
        # PATCH: Fallback: if still missing, log and set to 'Unknown'
        if not c.get("type_"):
            logger.warning(f"[CONTEST SELECTOR] Could not infer type for contest: {c.get('title', '')}")
            c["type_"] = "Unknown"

    context = {
        "state": norm_state,
        "county": norm_county,
        "year": year,
        "contests": contests,
        "url": getattr(coordinator, "last_url", None) if hasattr(coordinator, "last_url") else None
    }
    logger.debug(f"DEBUG: selector_data['contests']: {selector_data.get('contests', None)}")
    logger.debug(f"[DEBUG] norm_state: {norm_state}, norm_county: {norm_county}, year: {year}")
    logger.debug(f"[DEBUG] noisy_patterns: {noisy_patterns}")
    logger.debug(f"[DEBUG] contests before filtering: {contests}")

    # --- Filter contests ---
    filtered_contests = []
    fallback_contests = []
    for c in contests:
        skip_reason = None
        contest_state = c.get("state", "")
        title = c.get("title")
        title_norm = (title or "").strip().lower()

        # Skip noisy/generic patterns
        if not title or not isinstance(title, str) or title_norm in ["", "results", "summary"]:
            skip_reason = "empty/generic title"
        elif any(pat in title_norm for pat in ["unofficial results", "summary", "results by election district"]):
            skip_reason = "noisy pattern"
        elif not c.get("year"):
            skip_reason = "missing year"
        elif not c.get("type_"):
            skip_reason = "missing type_"
        elif not c.get("state"):
            skip_reason = "missing state"
        elif not c.get("county"):
            skip_reason = "missing county"
        elif not fuzzy_county_match(c.get("county", ""), norm_county, known_county_to_precincts):
            skip_reason = "county mismatch"
        elif norm_state and normalize_state_name(contest_state) != norm_state:
            skip_reason = "state mismatch"

        if skip_reason:
            logger.debug(f"[DEBUG] Skipping contest due to {skip_reason}: {c}")
            if title and title_norm not in ["", "results", "summary"]:
                fallback_contests.append(c)
            continue
        filtered_contests.append(c)

    # If ambiguous county matches, prompt user
    if not filtered_contests and fallback_contests:
        ambiguous_counties = set(c.get("county", "") for c in fallback_contests)
        if len(ambiguous_counties) > 1:
            logger.warning(f"[COUNTY AMBIGUITY] Multiple possible counties found: {ambiguous_counties}")
            try:
                choice = prompt.prompt_input(
                    f"[PROMPT] Multiple counties found: {ambiguous_counties}. Enter county to use or 'all': ",
                    default="all",
                    validator=lambda x: x == "all" or x in ambiguous_counties,
                    allow_cancel=True,
                    header="COUNTY SELECTION"
                ).strip().lower()
                if choice and choice != "all":
                    filtered_contests = [c for c in fallback_contests if normalize_county_name(c.get("county", "")) == normalize_county_name(choice)]
                else:
                    filtered_contests = fallback_contests
            except PromptCancelled:
                logger.warning("[yellow]County selection cancelled by user.[/yellow]")
                return None
        else:
            filtered_contests = fallback_contests

    if not filtered_contests:
        logger.warning("[yellow]No valid contests detected after filtering. Skipping.[/yellow]")
        return None

    # --- Deduplicate by normalized title, year, and type ---
    unique_contests = []
    seen = set()
    for c in filtered_contests:
        norm_title = normalize_contest_title(c.get("title", "") or "")
        key = ((c.get("year") or ""), (c.get("type_") or ""), norm_title)
        if key not in seen:
            unique_contests.append(c)
            seen.add(key)
    filtered_contests = unique_contests

    if not filtered_contests:
        logger.warning("[yellow]No valid contests detected after deduplication. Skipping.[/yellow]")
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
            logger.warning("[yellow]No contests passed ML/NER verification. Skipping.[/yellow]")
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
        logger.info(f"[bold cyan]{label.strip()}[/bold cyan]")
        for c in contests_in_group:
            logger.info(f"  [{idx}] {c.get('title', '')}")
            contest_indices.append(c)
            idx += 1
    logger.debug(f"[DEBUG] Number of contests displayed: {idx}")

    # --- Auto-select if only one contest ---
    if len(verified_contests) == 1:
        contest = ensure_contest_title(verified_contests[0])
        logger.info(f"[green]Only one contest found. Auto-selecting: {contest['title']}[/green]")
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
        choice = prompt.prompt_input(
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
        logger.warning("[yellow]Contest selection cancelled by user.[/yellow]")
        if log_func:
            log_func("[CONTEST] User cancelled contest selection.")
        return None

    if not choice:
        logger.warning("[yellow]No contest selected. Skipping.[/yellow]")
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
        logger.warning("[yellow]No valid contest indices selected. Skipping.[/yellow]")
        if log_func:
            log_func("[CONTEST] No valid contest indices selected.")
        return None

    selected = [ensure_contest_title(contest_indices[i]) for i in indices]
    if log_func:
        log_func(f"[CONTEST] User selected contests: {[c.get('title', '') for c in selected]}")
    return selected

