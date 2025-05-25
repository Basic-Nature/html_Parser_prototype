import os
import re
import time
from collections import defaultdict
from .contest_selector import is_noisy_label, is_noisy_contest_label
from ..state_router import STATE_MODULE_MAP
from typing import Dict, Any
from ..utils.shared_logic import ( 
    COMMON_CONTEST_LABELS, COMMON_PRECINCT_HEADERS, 
    IGNORE_SUBSTRINGS, SCAN_WAIT_SECONDS,
    ALL_SELECTORS, YEAR_REGEX, SYMBOL_REGEX, is_contest_label 
)
from ..utils.shared_logger import logger

def normalize_text(text: str) -> str:
    text = SYMBOL_REGEX.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def score_state_county_from_url(url: str, text_lower: str):
    """
    Score state and county from the URL and text, similar to DOM scoring.
    Returns (best_state, state_score, best_county, county_score)
    """
    url = url.lower()
    state_scores = {}
    for abbr, state_name in STATE_MODULE_MAP.items():
        score = 0
        state_name_clean = state_name.replace("_", " ")
        if state_name_clean in url:
            score += 2
        if state_name_clean in text_lower:
            score += 1
        abbr_pattern = rf"(?<![a-z]){re.escape(abbr)}(?![a-z])"
        if re.search(abbr_pattern, url):
            score += 2
        if re.search(abbr_pattern, text_lower):
            score += 1
        abbr_sep_pattern = rf"([/\-_\.]){re.escape(abbr)}([/\-_\.])"
        if re.search(abbr_sep_pattern, url):
            score += 2
        for clue in ["county", "election", "results", "votes"]:
            if f"{state_name_clean} {clue}" in url or f"{abbr} {clue}" in url:
                score += 1
            if f"{state_name_clean}-{clue}" in url or f"{abbr}-{clue}" in url:
                score += 1
            if f"{state_name_clean}_{clue}" in url or f"{abbr}_{clue}" in url:
                score += 1
        for header in COMMON_PRECINCT_HEADERS:
            if header.lower() in text_lower and (state_name_clean in text_lower or abbr in text_lower):
                score += 1
        state_scores[abbr] = score
    best_state = max(state_scores, key=state_scores.get)
    state_score = state_scores[best_state]

    # County scoring (only if state found)
    best_county = None
    county_score = 0
    county_folder = os.path.join(
        os.path.dirname(__file__), "..", "handlers", "states", best_state, "county"
    )
    if os.path.isdir(county_folder):
        possible_counties = [
            f[:-3] for f in os.listdir(county_folder)
            if f.endswith(".py") and f != "__init__.py"
        ]
        for county in possible_counties:
            score = 0
            county_pattern = re.compile(rf"\b{re.escape(county)}\b", re.IGNORECASE)
            if county_pattern.search(url):
                score += 2
            if county_pattern.search(text_lower):
                score += 1
            if f"{county}-county" in url or f"{county}_county" in url or f"{county} county" in url:
                score += 1
            if score > county_score:
                best_county = county
                county_score = score
    return best_state, state_score, best_county, county_score

def scan_html_for_context(page, debug=False) -> Dict[str, Any]:
    context_result = {
        "available_races": defaultdict(lambda: defaultdict(list)),
        "_source_map": defaultdict(list),
        "raw_text": "",
        "indicators": [],
        "state": "Unknown",
        "county": None,
        "election_type": None,
        "error": None,
        "debug_elements": []
    }

    try:
        logger.info(f"[SCAN] Waiting {SCAN_WAIT_SECONDS} seconds for page content to load...")
        time.sleep(SCAN_WAIT_SECONDS)
        text = page.inner_text("body")
        context_result["raw_text"] = text

        url = page.url.lower()
        text_lower = text.lower()

        # --- 1. Try to detect state/county from URL first, using scoring ---
        best_state, state_score, best_county, county_score = score_state_county_from_url(url, text_lower)
        if state_score >= 3:
            context_result["state"] = best_state
            logger.info(f"[SCAN] (URL) Detected state '{best_state}' from URL with score {state_score}.")
        else:
            logger.info(f"[SCAN] (URL) State score too low ({state_score}), will try DOM context.")

        if county_score >= 3:
            context_result["county"] = best_county
            logger.info(f"[SCAN] (URL) Detected county '{best_county}' from URL with score {county_score}.")
        else:
            logger.info(f"[SCAN] (URL) County score too low ({county_score}), will try DOM context.")

        # --- 2. Fallback: DOM-based state detection if not found in URL ---
        if context_result["state"] == "Unknown":
            state_scores = {}
            for abbr, state_name in STATE_MODULE_MAP.items():
                score = 0
                state_name_clean = state_name.replace("_", " ")

                if state_name_clean in url:
                    score += 2
                if state_name_clean in text_lower:
                    score += 1

                abbr_pattern = rf"(?<![a-z]){re.escape(abbr)}(?![a-z])"
                if re.search(abbr_pattern, url):
                    score += 2
                if re.search(abbr_pattern, text_lower):
                    score += 1

                abbr_sep_pattern = rf"([/\-_\.]){re.escape(abbr)}([/\-_\.])"
                if re.search(abbr_sep_pattern, url):
                    score += 2

                for clue in ["county", "election", "results", "votes"]:
                    if f"{state_name_clean} {clue}" in url or f"{abbr} {clue}" in url:
                        score += 1
                    if f"{state_name_clean}-{clue}" in url or f"{abbr}-{clue}" in url:
                        score += 1
                    if f"{state_name_clean}_{clue}" in url or f"{abbr}_{clue}" in url:
                        score += 1

                for header in COMMON_PRECINCT_HEADERS:
                    if header.lower() in text_lower and (state_name_clean in text_lower or abbr in text_lower):
                        score += 1

                state_scores[abbr] = score

            best_state_dom = max(state_scores, key=state_scores.get)
            if state_scores[best_state_dom] >= 3:
                context_result["state"] = best_state_dom
                logger.info(f"[SCAN] (DOM) Detected state '{best_state_dom}' with score {state_scores[best_state_dom]}.")
            else:
                logger.info(f"[SCAN] No state detected with high confidence. Best score: {state_scores[best_state_dom]}")

        # --- 3. Fallback: DOM-based county detection if not found in URL ---
        if not context_result["county"] and context_result["state"] != "Unknown":
            county_folder = os.path.join(
                os.path.dirname(__file__), "..", "handlers", "states", context_result["state"], "county"
            )
            if os.path.isdir(county_folder):
                possible_counties = [
                    f[:-3] for f in os.listdir(county_folder)
                    if f.endswith(".py") and f != "__init__.py"
                ]
                best_county_dom = None
                best_score_dom = 0
                for county in possible_counties:
                    score = 0
                    county_pattern = re.compile(rf"\b{re.escape(county)}\b", re.IGNORECASE)
                    if county_pattern.search(url):
                        score += 2
                    if county_pattern.search(text_lower):
                        score += 1
                    if f"{county}-county" in url or f"{county}_county" in url or f"{county} county" in url:
                        score += 1
                    if score > best_score_dom:
                        best_county_dom = county
                        best_score_dom = score
                if best_score_dom >= 3:
                    context_result["county"] = best_county_dom
                    logger.info(f"[SCAN] (DOM) Detected county '{best_county_dom}' with score {best_score_dom}.")

        # --- Query all possible elements ---
        elements = page.query_selector_all(ALL_SELECTORS)
        last_detected_year = None
        last_detected_type = None

        for el in elements:
            try:
                raw_segment = el.inner_text().strip()
                tag_name = el.evaluate("e => e.tagName").lower()
                norm_segment = normalize_text(raw_segment)
                context_result["_source_map"][tag_name].append(raw_segment)
                if debug:
                    context_result["debug_elements"].append(
                        {"tag": tag_name, "raw": raw_segment, "norm": norm_segment}
                    )
            except Exception:
                continue

            # --- EARLY FILTERING: skip noisy or irrelevant segments ---
            if (
                not norm_segment
                or any(ignore in norm_segment for ignore in IGNORE_SUBSTRINGS)
                or is_noisy_label(norm_segment)
                or is_noisy_contest_label(norm_segment)
                or not is_contest_label(norm_segment)
            ):
                continue

            # Detect year
            year_matches = YEAR_REGEX.findall(raw_segment)
            if year_matches:
                last_detected_year = max(year_matches)
                logger.debug(f"[SCAN] Detected year: {last_detected_year} from line: '{raw_segment[:50]}...'")

            # Detect contest/race keywords
            for keyword in COMMON_CONTEST_LABELS:
                if normalize_text(keyword) in norm_segment:
                    tag_year = last_detected_year if last_detected_year else "Unknown"
                    # Detect election type
                    if "general" in norm_segment:
                        detected_type = "General"
                    elif "primary" in norm_segment:
                        detected_type = "Primary"
                    elif "special" in norm_segment:
                        detected_type = "Special"
                    elif "runoff" in norm_segment:
                        detected_type = "Runoff"
                    else:
                        detected_type = context_result.get("election_type") or last_detected_type or "Unknown"
                    context_result["election_type"] = last_detected_type = detected_type
                    context_result["available_races"][tag_year][detected_type].append(raw_segment)
                    break  # Only match first keyword per segment

            # Detect precinct headers
            for header in COMMON_PRECINCT_HEADERS:
                if normalize_text(header) in norm_segment:
                    context_result.setdefault("precinct_headers", []).append(raw_segment)
                    break

        # Flatten all detected races for indicators
        all_races = []
        for year_group in context_result["available_races"].values():
            for races in year_group.values():
                all_races.extend(races)
        context_result["indicators"] = sorted(set(all_races))

        # Debug print of all elements and matches
        if debug:
            print("\n[DEBUG] Elements scanned and normalized:")
            for elem in context_result["debug_elements"]:
                print(f"<{elem['tag']}> {elem['raw']}  -->  [{elem['norm']}]")
            print("\n[DEBUG] Contest keyword matches:")
            for year, types in context_result["available_races"].items():
                for etype, races in types.items():
                    print(f"  {year} / {etype}: {races}")

    except Exception as e:
        err_msg = f"[SCAN ERROR] HTML parsing failed: {e}"
        logger.error(err_msg)
        context_result["error"] = err_msg
    logger.info(f"[DEBUG] html_context before routing: {context_result}")
    return context_result  # END scan_html_for_context

def get_detected_races_from_context(context_result):
    flat = []
    seen = set()
    for year, year_group in context_result.get("available_races", {}).items():
        for etype, races in year_group.items():
            for race in races:
                key = (year, etype, race.strip())
                # Filter out noisy/generic races
                if key not in seen and race.strip() and not race.strip().lower().startswith("vote for"):
                    if not is_noisy_label(race) and not is_noisy_contest_label(race):
                        flat.append(key)
                        seen.add(key)
    return sorted(flat)