import re
import time
from collections import defaultdict
from typing import Dict, Any
from ..utils.shared_logic import ( 
    COMMON_CONTEST_LABELS, COMMON_PRECINCT_HEADERS, 
    IGNORE_SUBSTRINGS, SCAN_WAIT_SECONDS,
    ALL_SELECTORS, YEAR_REGEX, SYMBOL_REGEX
)
from ..utils.shared_logger import logger


def normalize_text(text: str) -> str:
    """Normalize text for matching: strip, lower, remove excess whitespace/symbols."""
    text = SYMBOL_REGEX.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def scan_html_for_context(page, debug=False) -> Dict[str, Any]:
    """
    Scans the page HTML for:
    - All possible elements (headers, divs, spans, roles, etc.)
    - Election years, contest/race keywords, and other metadata
    - Returns a context dictionary with all structured insights
    - In debug mode, prints all elements and their normalized text
    """
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

        # Query all possible elements
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

            if not norm_segment or any(ignore in norm_segment for ignore in IGNORE_SUBSTRINGS):
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

            # Detect state/county hints (if you have a state/county map, add logic here)

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

    return context_result

def get_detected_races_from_context(context_result):
    """
    Returns a flat, sorted list of (year, election_type, race) tuples, deduplicated and filtered.
    """
    flat = []
    seen = set()
    for year, year_group in context_result.get("available_races", {}).items():
        for etype, races in year_group.items():
            for race in races:
                key = (year, etype, race.strip())
                if key not in seen and race.strip() and not race.strip().lower().startswith("vote for"):
                    flat.append(key)
                    seen.add(key)
    return sorted(flat)