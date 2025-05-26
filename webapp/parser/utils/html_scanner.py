import os
import re
import time
import json
from collections import defaultdict
from typing import Dict, Any
import difflib

from .contest_selector import is_noisy_label, is_noisy_contest_label
from ..state_router import STATE_MODULE_MAP
from ..utils.shared_logger import logger
from ..Context_Integration.context_organizer import organize_context_with_cache, append_to_context_library

# --- Load config from context library ---
CONTEXT_LIBRARY_PATH = os.path.join(
    os.path.dirname(__file__), "..", "Context_Integration", "context_library.json"
)
if os.path.exists(CONTEXT_LIBRARY_PATH):
    with open(CONTEXT_LIBRARY_PATH, "r", encoding="utf-8") as f:
        CONTEXT_LIBRARY = json.load(f)
    COMMON_CONTEST_LABELS = CONTEXT_LIBRARY.get("common_contest_labels", [])
    COMMON_PRECINCT_HEADERS = CONTEXT_LIBRARY.get("common_precinct_headers", [])
    IGNORE_SUBSTRINGS = CONTEXT_LIBRARY.get("ignore_substrings", [])
    CONTEST_PARTS = CONTEXT_LIBRARY.get("contest_parts", [])
    ALL_SELECTORS = ", ".join(CONTEXT_LIBRARY.get("selectors", {}).get("all_selectors", []))
    YEAR_REGEX = re.compile(CONTEXT_LIBRARY.get("regex", {}).get("year", r"\b((?:19|20)\d{2})\b"))
    SYMBOL_REGEX = re.compile(CONTEXT_LIBRARY.get("regex", {}).get("symbol", r"[\s\n\r\t\$0]+"))
    SCAN_WAIT_SECONDS = CONTEXT_LIBRARY.get("scan_wait_seconds", 5)
else:
    logger.error("[html_scanner] context_library.json not found. Using limited defaults.")
    COMMON_CONTEST_LABELS = []
    COMMON_PRECINCT_HEADERS = []
    IGNORE_SUBSTRINGS = []
    CONTEST_PARTS = []
    ALL_SELECTORS = "h1, h2, h3, h4, h5, h6, strong, b, span, div"
    YEAR_REGEX = re.compile(r"\b((?:19|20)\d{2})\b")
    SYMBOL_REGEX = re.compile(r"[\s\n\r\t\$0]+")
    SCAN_WAIT_SECONDS = 5

def normalize_text(text: str) -> str:
    text = SYMBOL_REGEX.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def all_patterns(name, extra_terms=None):
    patterns = set()
    base = name.lower().replace("_", " ").replace("-", " ").strip()
    base = re.sub(r"\s+", " ", base)
    patterns.add(base)
    terms = ["county", "state", "election", "results", "votes"]
    if extra_terms:
        terms += extra_terms
    for part in terms:
        part_norm = part.lower().replace("_", " ").replace("-", " ").strip()
        patterns.add(f"{base} {part_norm}")
        patterns.add(f"{base}-{part_norm}")
        patterns.add(f"{base}_{part_norm}")
        patterns.add(f"{base}.{part_norm}")
        patterns.add(f"{base}{part_norm}")
        patterns.add(f"{part_norm} {base}")
        patterns.add(f"{part_norm}-{base}")
        patterns.add(f"{part_norm}_{base}")
        patterns.add(f"{part_norm}.{base}")
        patterns.add(f"{part_norm}{base}")
    for sep in ["/", "-", "_", ".", ""]:
        patterns.add(f"{sep}{base}{sep}")
        patterns.add(f"{sep}{base}")
        patterns.add(f"{base}{sep}")
    return patterns

def score_state_county_from_url(url: str, text_lower: str):
    url = url.lower()
    text_lower = text_lower.lower()
    state_scores = {}
    for abbr, state_name in STATE_MODULE_MAP.items():
        score = 0
        state_name_clean = state_name.replace("_", " ").replace("-", " ").strip().lower()
        abbr_clean = abbr.lower()
        patterns = all_patterns(state_name_clean, extra_terms=CONTEST_PARTS)
        for pattern in patterns:
            if pattern and pattern in url:
                score += 2
            if pattern and pattern in text_lower:
                score += 1
        abbr_regex = re.compile(rf"\b{re.escape(abbr_clean)}\b")
        if abbr_regex.search(url):
            score += 1
        if abbr_regex.search(text_lower):
            score += 0.5
        state_scores[abbr] = score
    best_state = max(state_scores, key=state_scores.get)
    state_score = state_scores[best_state]
    best_county = None
    county_score = 0
    county_folder = os.path.join(
        os.path.dirname(__file__), "..", "handlers", "states", STATE_MODULE_MAP.get(best_state, best_state), "county"
    )
    if os.path.isdir(county_folder):
        possible_counties = [
            f[:-3] for f in os.listdir(county_folder)
            if f.endswith(".py") and f != "__init__.py"
        ]
        url_segments = re.split(r"[\/\-_\.]", url)
        for county in possible_counties:
            score = 0
            county_clean = county.lower().replace("_", " ").replace("-", " ").strip()
            patterns = {
                county_clean,
                f"{county_clean} county",
                county_clean.replace(" ", ""),
                f"{county_clean.replace(' ', '')}county",
                f"{county_clean}-county",
                f"{county_clean}_county",
                f"{county_clean}county"
            }
            for pattern in patterns:
                if pattern and pattern in url:
                    score += 2
            if county_clean in url_segments or f"{county_clean} county" in url_segments:
                score += 4
            if score < 4:
                matches = difflib.get_close_matches(county_clean, url_segments, n=1, cutoff=0.85)
                matches += difflib.get_close_matches(f"{county_clean} county", url_segments, n=1, cutoff=0.85)
                if matches:
                    score += 3
            if score > county_score:
                best_county = county
                county_score = score
    return best_state, state_score, best_county, county_score

def scan_html_for_context(page, debug=False) -> Dict[str, Any]:
    """
    Scans the HTML for state/county/race info, available downloadable formats, 
    and button-like elements. Returns a context dict ready for organize_context.
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
        "debug_elements": [],
        "available_formats": [],
        "button_elements": [],
        "contests": [],
        "buttons": [],
        "panels": [],
        "tables": [],
        "metadata": {},
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
        if county_score >= 3:
            context_result["county"] = best_county

        # --- 2. Detect downloadable formats (CSV, JSON, PDF, etc.) ---
        known_exts = CONTEXT_LIBRARY.get("supported_formats", [".csv", ".json", ".pdf", ".xlsx", ".xls"])
        for a in page.query_selector_all("a[href]"):
            try:
                href = a.get_attribute("href")
                if not href:
                    continue
                for ext in known_exts:
                    if href.lower().endswith(ext):
                        context_result["available_formats"].append((ext.lstrip('.'), page.urljoin(href)))
                        break
            except Exception:
                continue

        # --- 3. Detect button-like elements (for later toggling) ---
        button_selectors = ["button", "[role=button]", "a.button", ".btn", ".button"]
        for sel in button_selectors:
            for btn in page.query_selector_all(sel):
                try:
                    label = btn.inner_text().strip()
                    if label:
                        context_result["button_elements"].append(label)
                        context_result["buttons"].append({"label": label, "selector": sel})
                except Exception:
                    continue

        # --- 4. Scan for races, headers, etc. ---
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

            # Early break for noisy/irrelevant labels
            if (
                not norm_segment
                or any(ignore in norm_segment for ignore in IGNORE_SUBSTRINGS)
                or is_noisy_label(norm_segment)
                or is_noisy_contest_label(norm_segment)
            ):
                continue

            year_matches = YEAR_REGEX.findall(raw_segment)
            year_in_label = None
            if year_matches:
                last_detected_year = max(year_matches)
                year_in_label = last_detected_year
            else:
                year_matches_label = YEAR_REGEX.findall(norm_segment)
                if year_matches_label:
                    last_detected_year = max(year_matches_label)
                    year_in_label = last_detected_year

            # Early break for contest label match
            for keyword in COMMON_CONTEST_LABELS:
                if normalize_text(keyword) in norm_segment:
                    tag_year = year_in_label if year_in_label else (last_detected_year if last_detected_year else "Unknown")
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
                    context_result["contests"].append({
                        "title": raw_segment,
                        "year": tag_year,
                        "type": detected_type,
                        "raw": raw_segment
                    })
                    break

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

        # --- 5. Metadata for context organizer ---
        context_result["metadata"] = {
            "state": context_result.get("state"),
            "county": context_result.get("county"),
            "source_url": page.url,
            "election_type": context_result.get("election_type"),
            "scrape_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # --- 6. Optionally update context library ---
        # append_to_context_library(context_result, path=CONTEXT_LIBRARY_PATH)

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
    flat = []
    seen = set()
    for year, year_group in context_result.get("available_races", {}).items():
        for etype, races in year_group.items():
            for race in races:
                key = (year, etype, normalize_text(race))
                if key not in seen and race.strip() and not race.strip().lower().startswith("vote for"):
                    if not is_noisy_label(race) and not is_noisy_contest_label(race):
                        flat.append((year, etype, race.strip()))
                        seen.add(key)
    return sorted(flat)