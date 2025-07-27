import re
from ..utils.shared_logger import SharedLogger
from ..utils.shared_logic import (
    normalize_state_name, normalize_county_name, _sync_type_and_election_types,
    safe_get, safe_items, safe_lower, safe_split, safe_capitalize, safe_strip
)
from ..utils.user_prompt import UserPrompt, PromptCancelled
from collections import defaultdict
from difflib import get_close_matches
from ..Context_Integration.Context_Library.constants import (
    ELECTION_TYPES, CONTEST_KEYWORDS, KNOWN_COUNTY_TO_PRECINCTS_MAP,
    ELECTION_TYPE_REGEX_MAP, OFFICE_KEYWORDS
    )

try:
    from nltk.stem import PorterStemmer
    from nltk.corpus import stopwords
    STEMMER = PorterStemmer()
    STOPWORDS = set(stopwords.words('english'))
except ImportError:
    STEMMER = None
    STOPWORDS = set()

logger = SharedLogger()
prompt = UserPrompt()
from typing import TYPE_CHECKING, List, Dict, Any, Optional
if TYPE_CHECKING:
    from ..Context_Integration.context_coordinator import ContextCoordinator

def extract_year_from_title(title) -> Optional[int]:
    if not title:
        return None
    # Find all years
    years = [int(y) for y in re.findall(r"(19|20)\d{2}", title)]
    if not years:
        return None
    # Lowercase title for type search
    title_lower = safe_lower(title)
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
    Robustly infer the election type for a contest using:
    - Regex and keyword matching on the title
    - Fuzzy matching against known types
    - Context clues (year, known election dates, context type)
    - ML/NER entity extraction
    - Most common type among all contests for the same year/county
    """
    if not title:
        return None
    title_lower = safe_lower(title)

    # --- 1. Robust regex/keyword matching for common types ---
    for pattern, forced_type in ELECTION_TYPE_REGEX_MAP:
        if re.search(pattern, title_lower):
            if forced_type:
                return forced_type
            match = re.search(pattern, title_lower)
            if match and match.lastindex is not None:
                return safe_capitalize(match.group(1))
            elif match:
                return safe_capitalize(match.group(0))

    # --- 2. Fuzzy match against ELECTION_TYPES ---
    close = get_close_matches(title_lower, [safe_lower(t or "") for t in ELECTION_TYPES], n=1, cutoff=0.8)
    if close:
        return safe_capitalize(close[0])

    # --- 3. ML/NER entity extraction ---
    if coordinator:
        ents = (coordinator or ContextCoordinator()).extract_entities(title)
        for ent, label in ents:
            if label == "EVENT" and safe_lower(ent or "") in [safe_lower(et or "") for et in ELECTION_TYPES]:
                return safe_capitalize(ent)

    # --- 4. Context clues: if context has a type, use it ---
    if context and safe_get(context, "type_"):
        return safe_capitalize(safe_get(context, "type_") or "")

    # --- 5. Most common type among all contests for this year/county ---
    year = safe_get(contest, "year")
    county = safe_get(contest, "county")
    type_counts = defaultdict(int)
    for c in all_contests:
        if (
            safe_get(c, "year") == year and
            safe_lower(safe_get(c, "county") or "") == safe_lower(county or "") and
            safe_lower(safe_get(c, "type_") or "")
        ):
            type_counts[safe_lower(safe_get(c, "type_") or "")] += 1
    if type_counts:
        most_common = max(type_counts.items(), key=lambda x: x[1])[0]
        return safe_capitalize(most_common or "")

    # --- 6. Fallback: try to infer from common office/role keywords ---
    for kw, typ in OFFICE_KEYWORDS:
        if kw in title_lower:
            return typ

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
    for parent_county, precincts in safe_items(known_county_to_precincts or {}):
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

def _remove_boilerplate(text: str) -> str:
    """
    Remove boilerplate phrases using regex and constants.
    Handles noisy phrases, election instructions, and patterns from constants.
    """
    patterns = [
        r'\s*[\r\n]*Vote for \d+\s*',
        r'\s*[\r\n]*Select \d+\s*',
        r'\s*[\r\n]*Choose \d+\s*',
        r'\s*[\r\n]*Pick \d+\s*',
        r'\s*[\r\n]*Ballot Item \d+\s*',
        r'\s*[\r\n]*Ballot Position \d+\s*',
        r'\s*[\r\n]*For Election Use Only\s*',
        r'\s*[\r\n]*Unofficial Results\s*',
        r'\s*[\r\n]*Summary\s*',
        r'\s*[\r\n]*Results by Election District\s*',
    ]
    # Add patterns from ELECTION_TYPE_REGEX_MAP and OFFICE_KEYWORDS if relevant
    patterns += [pat for pat, _ in ELECTION_TYPE_REGEX_MAP if "vote" in pat or "select" in pat or "ballot" in pat]
    patterns += [rf'\b{re.escape(kw)}\b' for kw, _ in OFFICE_KEYWORDS]
    for pat in patterns:
        text = re.sub(pat, '', text, flags=re.IGNORECASE)
    return text

def _remove_keywords(text: str, keywords) -> str:
    """
    Remove contest/office keywords from text, including plural and possessive forms.
    """
    for kw in keywords:
        # Remove keyword, plural, and possessive forms
        text = re.sub(rf'\b{re.escape(kw)}(\'s|s)?\b', '', text, flags=re.IGNORECASE)
    return text

def _stem_and_remove_stopwords(text: str) -> str:
    """
    Apply stemming and remove stopwords for advanced normalization.
    """
    if not STEMMER or not STOPWORDS:
        return text
    words = re.findall(r'\w+', text, flags=re.UNICODE)
    stemmed = [STEMMER.stem(w) for w in words if w.lower() not in STOPWORDS]
    return ' '.join(stemmed)

def normalize_race_name(name: str, advanced: bool = False) -> str:
    """
    Advanced normalization for race/contest names:
    - Lowercase
    - Remove boilerplate phrases and contest/office keywords
    - Remove common suffixes
    - Remove non-alphanumeric (Unicode-aware)
    - Collapse whitespace
    - Optionally stem and remove stopwords
    """
    if not name:
        return ""
    name = safe_strip(name)
    name = safe_lower(name)
    name = _remove_boilerplate(name)
    name = _remove_keywords(name, CONTEST_KEYWORDS)
    name = _remove_keywords(name, [kw for kw, _ in OFFICE_KEYWORDS])
    # Remove common suffixes and extra words
    name = re.sub(r"\b(race|contest|seat|position|ballot|item|office|role)\b", "", name)
    # Remove non-alphanumeric (preserve Unicode letters/numbers)
    name = re.sub(r"[^\w\d]", " ", name, flags=re.UNICODE)
    # Collapse whitespace
    name = re.sub(r"\s+", " ", name)
    name = name.strip()
    if advanced:
        name = _stem_and_remove_stopwords(name)
    return name

def normalize_contest(title: str, advanced: bool = False) -> str:
    """
    Advanced normalization for contest titles:
    - Remove boilerplate phrases using regex and constants
    - Remove contest/office keywords
    - Remove trailing/leading numbers and punctuation
    - Remove extra whitespace
    - Lowercase for deduplication
    - Optionally stem and remove stopwords
    """
    if not title:
        return ""
    title = safe_strip(title)
    title = _remove_boilerplate(title)
    title = _remove_keywords(title, CONTEST_KEYWORDS)
    title = _remove_keywords(title, [kw for kw, _ in OFFICE_KEYWORDS])
    # Remove trailing/leading numbers and punctuation
    title = re.sub(r'^[\d\W]+|[\d\W]+$', '', title, flags=re.UNICODE)
    # Collapse whitespace
    title = re.sub(r'\s+', ' ', title)
    title = title.strip().lower()
    if advanced:
        title = _stem_and_remove_stopwords(title)
    return title

def ml_verify_contest(contest: Dict[str, Any], coordinator: "ContextCoordinator", context: dict, threshold: float = 0.75) -> bool:
    """
    Enhanced ML/NER contest verification using context, constants, and semantic scoring.
    Returns True if above threshold, False otherwise.
    """
    if coordinator is None:
        coordinator = ContextCoordinator()
    title = safe_strip(contest.get("title", ""))
    year = safe_strip(contest.get("year", ""))
    ctype = safe_strip(contest.get("type_", ""))
    year_score = 0.0

    # Year scoring
    if year and re.match(r"^(19|20)\d{2}$", str(year)):
        year_score = 1.0
    else:
        entities = coordinator.extract_entities(title)
        for ent, label in entities:
            if label == "DATE" and re.match(r"^(19|20)\d{2}$", ent):
                year_score = 0.9
                break

    # Election type scoring
    known_types = [safe_lower(t or "") for t in coordinator.get_election_types()]
    ctype_norm = safe_lower(ctype).replace("election", "").strip()
    type_score = 0.0
    detected_type = None

    if ctype:
        # Direct match to known types or constants
        if any(t in ctype_norm for t in known_types):
            type_score = 1.0
            detected_type = ctype
        elif any(safe_lower(v) in ctype_norm for v in ELECTION_TYPES):
            type_score = 1.0
            detected_type = ctype
        else:
            # Regex/keyword match from constants, utilize forced_type
            for pattern, forced_type in ELECTION_TYPE_REGEX_MAP:
                match = re.search(pattern, ctype_norm)
                if match:
                    type_score = 0.9
                    detected_type = forced_type if forced_type else match.group(0)
                    break
            # Accept other common types
            if type_score == 0.0 and ctype_norm in {
                "judicial", "proposition", "amendment", "state legislature", "federal legislature"
            }:
                type_score = 0.8
                detected_type = ctype_norm
            # Partial match
            elif type_score == 0.0 and any(safe_lower(v) in ctype_norm for v in ["general", "primary", "presidential", "special", "runoff"]):
                type_score = 0.8
                detected_type = ctype_norm

    # Advanced: If type is still ambiguous, use semantic similarity to known types
    if type_score == 0.0 and hasattr(coordinator, "_semantic_model"):
        model = getattr(coordinator, "_semantic_model", None)
        best_sim = 0.0
        best_type = None
        for t in known_types:
            sim = coordinator.score_header(ctype_norm, {"known_labels": [t]})
            if sim > best_sim:
                best_sim = sim
                best_type = t
        if best_sim > 0.7:
            type_score = 0.7
            detected_type = best_type

    # Title scoring (office/position keywords)
    title_score = 1.0 if any(safe_lower(kw or "") in safe_lower(title or "") for kw in CONTEST_KEYWORDS) else 0.0

    # Semantic/ML scoring
    ml_score = coordinator.score_header(title, context)

    # Fuzzy/semantic boost for ambiguous cases
    fuzzy_boost = 0.0
    if hasattr(coordinator, "fuzzy_score"):
        fuzzy_boost = coordinator.fuzzy_score(title, ctype) * 0.1

    # Aggregate score
    score = (
        0.4 * year_score +
        0.3 * type_score +
        0.2 * title_score +
        0.1 * ml_score +
        fuzzy_boost
    )

    # Log detected type for debugging/feedback
    logger.debug(f"[ml_verify_contest] Detected type: {detected_type} | year_score={year_score}, type_score={type_score}, title_score={title_score}, ml_score={ml_score}, fuzzy_boost={fuzzy_boost:.2f}, total={score:.2f}")

    if score < threshold:
        logger.debug(f"[DEBUG][ml_verify_contest] Rejected contest: '{title}' | year: {year} | type_: {ctype}")
        logger.info(f"  year_score={year_score}, type_score={type_score}, title_score={title_score}, ml_score={ml_score}, fuzzy_boost={fuzzy_boost:.2f}, total={score:.2f}")

    # If year and title are strong, allow type_ to be unknown
    if year_score == 1.0 and title_score == 1.0 and score >= 0.55:
        return True
    return score >= threshold

def feedback_loop_verify_contests(contests: List[Dict[str, Any]], coordinator: "ContextCoordinator", context: dict, max_loops: int = 3, threshold: float = 0.85) -> List[Dict[str, Any]]:
    """
    Enhanced feedback loop: rescans and verifies contests using ML/NER, fuzzy/semantic scoring, and user feedback.
    """
    if coordinator is None:
        coordinator = ContextCoordinator()
    for loop in range(max_loops):
        verified = []
        for c in contests:
            # Lower threshold on last loop
            if loop == max_loops - 1:
                if ml_verify_contest(c, coordinator, context, threshold=0.5):
                    verified.append(c)
            else:
                if ml_verify_contest(c, coordinator, context, threshold=threshold):
                    verified.append(c)
        if verified:
            logger.info(f"[CONTEST SELECTOR] Feedback loop {loop+1}: {len(verified)} contests passed ML/NER verification.")
            return verified
        logger.warning(f"[CONTEST SELECTOR] Feedback loop {loop+1}: No contests passed ML/NER verification. Retrying...")

    # Fallback: select contests with strong title match or semantic similarity
    fallback_verified = [
        c for c in contests
        if safe_get(c, "title") and len(safe_get(c, "title")) > 10
        or (hasattr(coordinator, "score_header") and coordinator.score_header(safe_get(c, "title", ""), context) > 0.6)
    ]
    if fallback_verified:
        logger.info("[CONTEST SELECTOR] Fallback: selecting contests by title/semantic score.")
        return fallback_verified

    # If still ambiguous, prompt user for clarification
    logger.warning("[yellow]Unable to confidently identify valid contests after feedback loop. Please clarify selection.[/yellow]")
    grouped = defaultdict(list)
    for idx, c in enumerate(contests):
        grouped[(safe_get(c, 'year', ''), safe_get(c, 'type_', ''))].append((idx, c))

    for (year, ctype), items in sorted(grouped.items()):
        logger.info(f"[bold cyan]Year: {year or 'Unknown'}, Type: {ctype or 'Unknown'}[/bold cyan]")
        for idx, c in items:
            logger.info(f"  [{idx}] {safe_get(c, 'title', '')}")

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
        return [_sync_type_and_election_types(c) or c for c in contests]
    indices = []
    for part in choice.split(","):
        part = part.strip()
        if part.isdigit():
            idx = int(part)
            if 0 <= idx < len(contests):
                indices.append(idx)
    selected = [_sync_type_and_election_types(contests[i]) or contests[i] for i in indices]
    # Log user feedback for ML improvement
    logger.debug(f"norm_state: {safe_get(context, 'state')}, norm_county: {safe_get(context, 'county')}, year: {safe_get(context, 'year')}")
    for c in selected:
        logger.debug(f"Contest: {safe_get(c, 'title', '')}, state: {safe_get(c, 'state', '')}, county: {safe_get(c, 'county', '')}, year: {safe_get(c, 'year', '')}")
        coordinator.submit_user_feedback("contest", "contest", safe_get(c, "title", ""), context)
    return selected

def ensure_contest(contest) -> Dict[str, Any]:
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
    session_id=None,
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
    contests = safe_get(selector_data, "contests", [])
    noisy_patterns = safe_get(selector_data, "noisy_patterns", [])
    known_county_to_precincts = KNOWN_COUNTY_TO_PRECINCTS_MAP

    # --- Fill in missing year/type from title using ML/NER ---
    for c in contests:
        # Fill year
        if not safe_get(c, "year"):
            year_from_title = extract_year_from_title(safe_get(c, "title", ""))
            if year_from_title:
                c["year"] = year_from_title
        # Fill type_
        found_type_ = None
        if not safe_get(c, "type_"):
            inferred_type = infer_election_type(
                safe_get(c, "title", ""),
                context,
                c,
                contests,
                coordinator
            )
            if inferred_type:
                c["type_"] = inferred_type
                # Try ML/NER
                ents = coordinator.extract_entities(safe_get(c, "title", "")) if coordinator else []
                for ent, label in ents:
                    if label == "EVENT" and safe_lower(ent) in [safe_lower(et) for et in ELECTION_TYPES]:
                        found_type_ = safe_lower(ent)
                        break
        # Only use found_type_ if set
        if found_type_:
            c["type_"] = found_type_.capitalize()
        # Fallback: if still missing, log and set to 'Unknown'
        if not safe_get(c, "type_"):
            logger.warning(f"[CONTEST SELECTOR] Could not infer type for contest: {safe_get(c, 'title', '')}")
            c["type_"] = "Unknown"
        _sync_type_and_election_types(c)
        # Attach session_id if provided
        if session_id is not None:
            c["session_id"] = session_id

    context = {
        "state": norm_state,
        "county": norm_county,
        "year": year,
        "contests": contests,
        "url": getattr(coordinator, "last_url", None) if hasattr(coordinator, "last_url") else None,
        "session_id": session_id
    }
    logger.debug(f"DEBUG: selector_data['contests']: {safe_get(selector_data, 'contests', None)}")
    logger.debug(f"[DEBUG] norm_state: {norm_state}, norm_county: {norm_county}, year: {year}")
    logger.debug(f"[DEBUG] noisy_patterns: {noisy_patterns}")
    logger.debug(f"[DEBUG] contests before filtering: {contests}")

    # --- Filter contests ---
    filtered_contests = []
    fallback_contests = []
    for c in contests:
        skip_reason = None
        contest_state = safe_get(c, "state", "")
        title = safe_get(c, "title", "")
        title_norm = safe_lower(title).strip() if isinstance(title, str) else ""

        # Skip noisy/generic patterns
        if not title or not isinstance(title, str) or title_norm in ["", "results", "summary"]:
            skip_reason = "empty/generic title"
        elif any(pat in title_norm for pat in ["unofficial results", "summary", "results by election district"]):
            skip_reason = "noisy pattern"
        elif not safe_get(c, "year"):
            skip_reason = "missing year"
        elif not safe_get(c, "type_"):
            skip_reason = "missing type_"
        elif not safe_get(c, "state"):
            skip_reason = "missing state"
        elif not safe_get(c, "county"):
            skip_reason = "missing county"
        elif not fuzzy_county_match(safe_get(c, "county", ""), norm_county, known_county_to_precincts):
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
        ambiguous_counties = set(safe_get(c, "county", "") for c in fallback_contests)
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
                    filtered_contests = [c for c in fallback_contests if normalize_county_name(safe_get(c, "county", "")) == normalize_county_name(choice)]
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
        norm_title = normalize_contest(safe_get(c, "title", "") or "")
        key = (safe_get(c, "year", ""), safe_get(c, "type_", ""), norm_title)
        if key not in seen:
            unique_contests.append(c)
            seen.add(key)
    unique_contests = []
    seen = set()
    for c in filtered_contests:
        norm_title = normalize_contest(safe_get(c, "title", "") or "")
        key = (safe_get(c, "year", ""), safe_get(c, "type_", ""), norm_title)
        if key not in seen:
            unique_contests.append(c)
            seen.add(key)
    # --- Deduplicate and sort by year, type, title ---
    filtered_contests = sorted(
        unique_contests,
        key=lambda c: (
            safe_get(c, "year", ""),
            safe_get(c, "type_", ""),
            safe_lower(safe_get(c, "title", ""))
        )
    )

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
        grouped[
            (
                safe_get(c, "year"),
                safe_get(c, "type_"),
                tuple(safe_get(c, "election_types", [])) if safe_get(c, "election_types") else ()
            )
        ].append(c)

    # --- Dynamic titling for selection prompt ---
    idx = 0
    contest_indices = []
    for (year_val, etype, etypes), contests_in_group in sorted(grouped.items()):
        etypes_str = ", ".join(etypes) if etypes else ""
        if len(grouped) > 1:
            label = f"{state or 'Unknown State'} {county or ''} {year_val or 'Unknown'} {etype or 'Unknown'}"
            if etypes_str:
                label += f" [{etypes_str}]"
        else:
            label = f"{year_val or 'Unknown'} {etype or 'Unknown'}"
            if etypes_str:
                label += f" [{etypes_str}]"
        logger.info(f"[bold cyan]{label.strip()}[/bold cyan]")
        for c in contests_in_group:
            logger.info(f"  [{idx}] {safe_get(c, 'title', '')}")
            contest_indices.append(c)
            idx += 1
    logger.debug(f"[DEBUG] Number of contests displayed: {idx}")

    # --- Auto-select if only one contest ---
    if len(verified_contests) == 1:
        contest = ensure_contest(verified_contests[0])
        logger.info(f"[green]Only one contest found. Auto-selecting: {safe_get(contest, 'title', '')}[/green]")
        if log_func:
            log_func(f"[CONTEST] Auto-selected: {safe_get(contest, 'title', '')}")
        return [contest]

    # --- Non-interactive mode: select all ---
    if non_interactive:
        # For webapp GUI, return empty and let frontend/API handle selection.
        if getattr(logger, "mode", None) == "webapp":
            if log_func:
                log_func("[CONTEST] Non-interactive mode (webapp): awaiting selection from frontend/API.")
            return []
        # For CLI, select all contests automatically.
        if log_func:
            log_func(f"[CONTEST] Non-interactive mode (CLI): selecting all contests.")
        selected = [ensure_contest(c) for c in verified_contests]
        # Attach session_id to selected contests if provided
        if session_id is not None:
            for c in selected:
                c["session_id"] = session_id
        return selected

    # --- Interactive prompt ---
    try:
        choice = prompt.prompt_input(
            prompt_message,
            default="all",
            validator=lambda x: (
                x == "all"
                or (allow_multiple and all(
                    p.strip().isdigit() and 0 <= int(p.strip()) < len(contest_indices)
                    for p in safe_split(x, ",") if p.strip()
                ))
                or (not allow_multiple and x.strip().isdigit() and 0 <= int(x.strip()) < len(contest_indices))
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
        selected = [ensure_contest(_sync_type_and_election_types(c) or c) for c in verified_contests]
    else:
        indices = []
        if allow_multiple:
            for part in choice.split(","):
                part = part.strip()
                if part.isdigit():
                    idx = int(part)
                    if 0 <= idx < len(contest_indices):
                        indices.append(idx)
        else:
            if choice.isdigit():
                idx = int(choice)
                if 0 <= idx < len(contest_indices):
                    indices.append(idx)
        if not indices:
            logger.warning("[yellow]No valid contest indices selected. Skipping.[/yellow]")
            if log_func:
                log_func("[CONTEST] No valid contest indices selected.")
            return None
        selected = [ensure_contest(_sync_type_and_election_types(contest_indices[i]) or contest_indices[i]) for i in indices]

    # Attach session_id to selected contests if provided
    if session_id is not None:
        for c in selected:
            c["session_id"] = session_id

    if log_func:
        log_func(f"[CONTEST] User selected contests: {[safe_get(c, 'title', '') for c in selected]}")
    return selected