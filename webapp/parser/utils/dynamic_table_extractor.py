"""
dynamic_table_extractor.py

Candidate Table Generator & Scorer for Election Data Extraction Pipeline

This module is responsible ONLY for:
- Finding all plausible tabular data candidates on a page (tables, repeated DOM, patterns)
- Scoring and ranking candidates using ML/NLP and heuristics
- Providing diagnostics, advanced extraction, and pattern learning utilities

All harmonization, entity annotation, structure verification, and user feedback
are handled centrally in table_core.py and table_builder.py.

This ensures a single source of truth for table structure and learning.
"""

import os
import re
import json
import unicodedata

from typing import List, Dict
from ..utils.shared_logger import logger, rprint
from ..config import BASE_DIR
from typing import TYPE_CHECKING
from ..utils.table_core import (
    TOTAL_KEYWORDS,
    LOCATION_KEYWORDS,
    MISC_FOOTER_KEYWORDS,
    extract_rows_and_headers_from_dom,
    extract_with_patterns,
    guess_headers_from_row,
    extract_table_data,
    detect_table_structure,
    get_safe_log_path,
    load_dom_patterns,
    normalize_text,
    robust_table_extraction,
    is_date_like,
)

if TYPE_CHECKING:
    from ..Context_Integration.context_coordinator import ContextCoordinator

# --- Constants and Paths ---
CANDIDATE_KEYWORDS = {"candidate", "candidates", "name", "nominee"}
BALLOT_TYPE_KEYWORDS = {"election day", "early voting", "absentee", "mail", "provisional", "affidavit", "other", "void"}

LOG_PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "log"))

# --- Main Candidate Generator/Scorer ---

def dynamic_table_extractor(page, context, coordinator, table_html=None):
    """
    Finds and scores candidate tables, returning the best (headers, data) for further processing.
    Does NOT run harmonization, annotation, or feedback loop.
    """
    if table_html:
        # Use BeautifulSoup to extract headers and rows from the HTML snippet
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(table_html, "html.parser")
        table = soup.find("table")
        if table:
            rows = table.find_all("tr")
            if not rows:
                return [], []
            headers = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]
            data = []
            for row in rows[1:]:
                cells = row.find_all(["td", "th"])
                data.append({headers[i]: cells[i].get_text(strip=True) if i < len(cells) else "" for i in range(len(headers))})
            return headers, data
    candidates = find_tabular_candidates(page)
    enriched_candidates = []
    for cand in candidates:
        cand = analyze_candidate_nlp(cand, coordinator)
        cand['score'], cand['rationale'] = score_candidate(cand, context, coordinator)
        enriched_candidates.append(cand)
    enriched_candidates.sort(key=lambda c: c['score'], reverse=True)
    best = enriched_candidates[0] if enriched_candidates else None
    if best:
        logger.info(f"[DYNAMIC_TABLE_EXTRACTOR] Best candidate source: {best.get('source')}, score: {best.get('score'):.2f}")
        return best['headers'], best['rows']
    logger.warning("[DYNAMIC_TABLE_EXTRACTOR] No suitable table candidates found.")
    return [], []

# --- Candidate Generation & Scoring ---

def find_tabular_candidates(page):
    """
    Find all DOM elements that look like tables or repeated row structures.
    Returns a list of candidate dicts with 'headers' and 'rows'.
    """
    candidates = []
    # 1. Standard HTML tables
    tables = page.locator("table")
    for i in range(tables.count()):
        table = tables.nth(i)
        if table is None:
            continue
        headers, data, _ = extract_table_data(table)
        if headers and data:
            candidates.append({"headers": headers, "rows": data, "source": "table"})
    # 2. Repeated DOM structures (divs, lists, etc.)
    headers, data = extract_rows_and_headers_from_dom(page)
    if headers and data:
        candidates.append({"headers": headers, "rows": data, "source": "repeated_dom"})
    # 3. Pattern-based extraction (if any patterns are approved)
    pattern_rows = extract_with_patterns(page)
    # Only use rows where row is not None
    pattern_rows = [tup for tup in pattern_rows if tup[1] is not None]
    if pattern_rows:
        headers = guess_headers_from_row(pattern_rows[0][1])
        data = []
        for heading, row, pat in pattern_rows:
            if row is None:
                continue
            cells = row.locator("> *")
            row_data = {}
            for idx in range(cells.count()):
                row_data[headers[idx] if idx < len(headers) else f"Column {idx+1}"] = cells.nth(idx).inner_text().strip()
            if row_data:
                data.append(row_data)
        if headers and data:
            candidates.append({"headers": headers, "rows": data, "source": "pattern"})
    return candidates

def analyze_candidate_nlp(candidate, coordinator):
    """
    Enrich a candidate dict with NLP/NER analysis for headers.
    Adds 'header_entities' and 'header_scores' fields.
    """
    headers = candidate.get("headers", [])
    header_entities = []
    header_scores = []
    for h in headers:
        ents = coordinator.extract_entities(h)
        header_entities.append(ents)
        score = coordinator.score_header(h, {})
        header_scores.append(score)
    candidate["header_entities"] = header_entities
    candidate["header_scores"] = header_scores
    return candidate

def score_candidate(candidate, context, coordinator):
    """
    Score a candidate table structure using ML/NLP and heuristics.
    Returns (score, rationale).
    """
    headers = candidate.get("headers", [])
    rows = candidate.get("rows", [])
    rationale = []

    # 1. ML/NLP header confidence
    ml_scores = []
    for h in headers:
        score = coordinator.score_header(h, context)
        ml_scores.append(score)
    avg_ml_score = sum(ml_scores) / len(ml_scores) if ml_scores else 0
    rationale.append(f"ML header avg score: {avg_ml_score:.2f}")

    # 2. Heuristic: prefer more rows and columns (but not too many)
    n_rows = len(rows)
    n_cols = len(headers)
    row_score = min(n_rows / 10.0, 1.0)  # up to 1.0 for 10+ rows
    col_score = min(n_cols / 8.0, 1.0)   # up to 1.0 for 8+ columns
    rationale.append(f"Rows: {n_rows}, Cols: {n_cols}, row_score: {row_score:.2f}, col_score: {col_score:.2f}")

    # 3. Heuristic: penalize if too many empty cells
    total_cells = n_rows * n_cols if n_rows and n_cols else 1
    non_empty_cells = sum(1 for row in rows for v in row.values() if v not in ("", None))
    fill_ratio = non_empty_cells / total_cells if total_cells else 0
    rationale.append(f"Fill ratio: {fill_ratio:.2f}")
    fill_penalty = 0.0 if fill_ratio > 0.7 else -0.5

    # 4. Heuristic: bonus if headers match known keywords/entities
    entity_bonus = 0.0
    entity_hits = 0
    for h in headers:
        ents = coordinator.extract_entities(h)
        if ents:
            entity_hits += 1
    if headers:
        entity_bonus = 0.2 * (entity_hits / len(headers))
    rationale.append(f"Entity bonus: {entity_bonus:.2f} ({entity_hits}/{len(headers)} headers)")

    # 5. Penalty for generic headers (Column 1, etc.)
    generic_headers = sum(1 for h in headers if re.match(r"Column \d+", h))
    generic_penalty = -0.2 * (generic_headers / len(headers)) if headers else 0
    if generic_penalty:
        rationale.append(f"Generic header penalty: {generic_penalty:.2f}")

    # 6. Final score
    score = (
        0.5 * avg_ml_score +
        0.2 * row_score +
        0.2 * col_score +
        fill_penalty +
        entity_bonus +
        generic_penalty
    )
    score = max(0.0, min(1.0, score))
    rationale.append(f"Final score: {score:.2f}")

    return score, "; ".join(rationale)

# --- Column/Row Filtering & Type Inference ---

def remove_low_signal_columns(headers, data, min_unique=2, min_non_empty_ratio=0.05):
    """
    Remove columns with low variance or too many repeated values.
    """
    keep = []
    n_rows = len(data)
    for h in headers:
        col_vals = [row.get(h, "") for row in data]
        unique_vals = set(col_vals)
        non_empty = [v for v in col_vals if v not in ("", None)]
        if len(unique_vals) >= min_unique and len(non_empty) / n_rows >= min_non_empty_ratio:
            keep.append(h)
    return keep, [{h: row.get(h, "") for h in keep} for row in data]

def infer_column_types(headers, data):
    types = {}
    for h in headers:
        col_vals = [row.get(h, "") for row in data]
        if all(re.fullmatch(r"\d{1,3}(,\d{3})*", v) or v == "" for v in col_vals):
            types[h] = "int"
        elif all(re.fullmatch(r"\d+(\.\d+)?%", v) or v == "" for v in col_vals):
            types[h] = "percent"
        else:
            types[h] = "str"
    return types

def infer_column_types_advanced(headers, data):
    """
    Use statistics to infer column types: numeric, categorical, date, etc.
    """
    import numpy as np
    import dateutil.parser
    types = {}
    for h in headers:
        col_vals = [row.get(h, "") for row in data]
        non_empty = [v for v in col_vals if v not in ("", None)]
        try:
            nums = [float(v.replace(",", "")) for v in non_empty if v.replace(",", "").replace(".", "", 1).isdigit()]
        except Exception:
            nums = []
        if len(nums) > 0 and len(nums) / len(non_empty) > 0.7:
            types[h] = "numeric"
        elif all(is_date_like(v) for v in non_empty):
            types[h] = "date"
        elif len(set(non_empty)) < 10:
            types[h] = "categorical"
        else:
            types[h] = "string"
    return types

# --- Heuristics & Entity Linking ---

def advanced_party_candidate_detection(headers, coordinator):
    """
    Use NER and context to better distinguish between candidate, party, and location columns.
    """
    result = {"candidate": [], "party": [], "location": []}
    for idx, h in enumerate(headers):
        ents = coordinator.extract_entities(h)
        for ent, label in ents:
            if label in {"PERSON"}:
                result["candidate"].append(idx)
            elif label in {"ORG", "NORP"}:
                result["party"].append(idx)
            elif label in {"GPE", "LOC", "FAC"}:
                result["location"].append(idx)
    return result

def normalize_header(header, lang="en"):
    """
    Normalize header for comparison: lower, strip, remove accents, and translate if needed.
    """
    header = header.strip().lower()
    header = unicodedata.normalize('NFKD', header).encode('ascii', 'ignore').decode('ascii')
    return header

def extract_candidates_and_parties(headers: List[str], coordinator: "ContextCoordinator") -> Dict[str, Dict[str, List[str]]]:
    """
    Returns a dict: {party: {candidate: [ballot_types]}}
    """
    known_parties = [
        "Democratic", "DEM", "dem", 
        "Republican", "REP", "rep", 
        "Working Families", "WOR", "wor",
        "Conservative", "CON", "con", 
        "Green", "GRN", "grn", 
        "Libertarian", "LIB", "lib", 
        "Independent", "IND", "ind",
        "Larouche", "Write-In", "Other"                     
    ]
    ballot_types = BALLOT_TYPE_KEYWORDS

    candidate_party_map = {}
    for h in headers:
        m = re.match(r"(.+?)\s*\((.+?)\)\s*-\s*(.+)", h)
        if m:
            candidate, party, ballot_type = m.groups()
        else:
            m = re.match(r"(.+?)\s*-\s*(.+)", h)
            if m:
                candidate, ballot_type = m.groups()
                party = ""
            else:
                candidate, party, ballot_type = h, "", ""
        candidate = candidate.strip()
        party = party.strip()
        ballot_type = ballot_type.strip()
        if party:
            best_party, score = max(((p, coordinator.fuzzy_score(party, p)) for p in known_parties), key=lambda x: x[1])
            if score > 80:
                party = best_party
        else:
            entities = coordinator.extract_entities(candidate)
            for ent, label in entities:
                if label in {"ORG", "NORP"}:
                    party = ent
                    break
        if not party:
            party = "Other"
        if party not in candidate_party_map:
            candidate_party_map[party] = {}
        if candidate not in candidate_party_map[party]:
            candidate_party_map[party][candidate] = []
        if ballot_type and ballot_type not in candidate_party_map[party][candidate]:
            candidate_party_map[party][candidate].append(ballot_type)
    return candidate_party_map

def entity_linking(header, known_entities):
    """
    Link header to known candidates/parties for normalization.
    """
    import difflib
    best, score = None, 0
    for ent in known_entities:
        s = difflib.SequenceMatcher(None, normalize_header(header), normalize_header(ent)).ratio()
        if s > score:
            best, score = ent, s
    return best if score > 0.8 else header

# --- Pattern/Selector Discovery & Logging ---

def find_tables_with_headings(page, dom_segments=None, heading_tags=None, include_section_context=True):
    """
    Finds all tables on the page and pairs each with its nearest heading or ARIA landmark.
    Returns a list of (heading, table_locator) tuples.
    """
    if heading_tags is None:
        heading_tags = ("h1", "h2", "h3", "h4", "h5", "h6")

    results = []

    def extract_text_from_html(html: str) -> str:
        """
        Extracts visible text from an HTML string.
        - Handles tags like <span>, <div>, <a>, <li>, <b>, <strong>, <em>, <u>, <i>, <p>, <br>, <th>, <td>, <button>, <label>, <h1>-<h6>.
        - Strips all tags and returns the concatenated text.
        - Handles nested tags and ignores script/style.
        """
        # Remove script and style blocks
        html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
        # Replace <br> and <br/> with newlines
        html = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
        # Remove all other tags, keeping their content
        text = re.sub(r"<[^>]+>", "", html)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    if dom_segments:
        tables = [seg for seg in dom_segments if seg.get("tag") == "table"]
        for i, table_seg in enumerate(tables):
            heading = None
            section_context = None
            idx = table_seg.get("_idx", None)
            # 1. Walk backwards for nearest heading
            if idx is not None:
                for j in range(idx-1, -1, -1):
                    tag = dom_segments[j].get("tag", "")
                    if tag in heading_tags:
                        heading_html = dom_segments[j].get("html", "")
                        heading = extract_text_from_html(heading_html)
                        break
            # 2. If not found, walk up for ARIA landmarks or section/fieldset
            if not heading and idx is not None:
                # Walk up the DOM tree for section/fieldset/region
                parent_idx = table_seg.get("_parent_idx", None)
                visited = set()
                while parent_idx is not None and parent_idx not in visited:
                    visited.add(parent_idx)
                    parent_seg = dom_segments[parent_idx]
                    tag = parent_seg.get("tag", "")
                    attrs = parent_seg.get("attrs", {})
                    # ARIA region/landmark
                    aria_label = attrs.get("aria-label") or attrs.get("aria-labelledby")
                    role = attrs.get("role", "")
                    if role in ("region", "complementary", "main", "navigation", "search") or aria_label:
                        section_context = aria_label or role
                        break
                    # Section/fieldset/legend
                    if tag in ("section", "fieldset"):
                        # Try to find a legend or heading inside this section
                        for k in range(parent_idx+1, len(dom_segments)):
                            if dom_segments[k].get("_parent_idx") == parent_idx:
                                child_tag = dom_segments[k].get("tag", "")
                                if child_tag == "legend":
                                    heading = extract_text_from_html(dom_segments[k].get("html", ""))
                                    break
                                if child_tag in heading_tags:
                                    heading = extract_text_from_html(dom_segments[k].get("html", ""))
                                    break
                        if heading:
                            break
                        section_context = tag
                        break
                    parent_idx = parent_seg.get("_parent_idx", None)
            # 3. Compose heading with section context if desired
            if not heading:
                heading = f"Precinct {i+1}"
            if include_section_context and section_context:
                heading = f"{section_context}: {heading}"
            # Use Playwright to get the table locator by index
            table_locator = page.locator("table").nth(i)
            if table_locator is not None:
                results.append((heading, table_locator))
    else:
        # Fallback: Use Playwright only
        tables = page.locator("table")
        for i in range(tables.count()):
            table = tables.nth(i)
            if table is None:
                continue
            heading = None
            section_context = None
            try:
                parent = table
                for _ in range(5):
                    parent = parent.locator("xpath=..")
                    attrs = parent.evaluate("el => ({'role': el.getAttribute('role'), 'aria-label': el.getAttribute('aria-label'), 'aria-labelledby': el.getAttribute('aria-labelledby'), 'tag': el.tagName.toLowerCase()})")
                    if attrs.get("role") in ("region", "complementary", "main", "navigation", "search") or attrs.get("aria-label"):
                        section_context = attrs.get("aria-label") or attrs.get("role")
                        break
                    if attrs.get("tag") in ("section", "fieldset"):
                        legend = parent.locator("legend")
                        if legend.count() > 0:
                            heading = legend.nth(0).inner_text().strip()
                            break
                        for tag in heading_tags:
                            h = parent.locator(tag)
                            if h.count() > 0:
                                heading = h.nth(0).inner_text().strip()
                                break
                        if heading:
                            break
                        section_context = attrs.get("tag")
                        break
                if not heading:
                    header_locator = table.locator("xpath=preceding-sibling::*[self::h1 or self::h2 or self::h3 or self::h4 or self::h5 or self::h6][1]")
                    if header_locator.count() > 0:
                        heading = header_locator.nth(0).inner_text().strip()
            except Exception:
                pass
            if not heading:
                heading = f"Precinct {i+1}"
            if include_section_context and section_context:
                heading = f"{section_context}: {heading}"
            results.append((heading, table))
    return results

def discover_container_selectors(page, extra_keywords=None, min_row_count=2):
    """
    Dynamically discovers container selectors (divs, sections, etc.) with relevant keywords or tabular structure.
    Returns a list of selectors, ranked by likelihood.
    """
    if extra_keywords is None:
        extra_keywords = ["vote", "result", "candidate", "precinct", "choice", "option", "ballot", "row", "table", "summary"]
    selectors = set()
    class_scores = {}

    all_divs = page.locator("div")
    for i in range(all_divs.count()):
        div = all_divs.nth(i)
        if div is None:
            continue
        cls = div.get_attribute("class") or ""
        id_ = div.get_attribute("id") or ""
        text = div.inner_text().strip().lower()
        score = 0

        # Score based on keywords in class/id/text
        for kw in extra_keywords:
            if kw in cls.lower() or kw in id_.lower() or kw in text:
                score += 2
        # Score based on number of children (tabular structure)
        children = div.locator("> *")
        if children.count() >= min_row_count:
            score += 2
        # Score based on presence of numbers (votes)
        if any(char.isdigit() for char in text):
            score += 1

        # Build selector and store score
        if cls:
            sel = "div." + ".".join(cls.split())
            class_scores[sel] = class_scores.get(sel, 0) + score
        if id_:
            sel = f"div#{id_}"
            class_scores[sel] = class_scores.get(sel, 0) + score

    # Return selectors sorted by score
    sorted_selectors = [sel for sel, _ in sorted(class_scores.items(), key=lambda x: -x[1])]
    # Add some generic selectors as fallback
    sorted_selectors += ["section", "ul", "ol"]
    return sorted_selectors
        
def log_new_dom_pattern(example_html, selector, context=None, log_path=None):
    """
    Logs a new DOM pattern for future learning/updating of extraction logic.
    Uses a safe log path.
    """
    if log_path is None:
        log_path = get_safe_log_path()
    entry = {
        "selector": selector,
        "example_html": example_html,
        "context": context or {}
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

def review_dom_patterns(log_path=None):
    """
    CLI to review, approve, or delete learned DOM patterns.
    """
    if log_path is None:
        log_path = get_safe_log_path()
    if not os.path.exists(log_path):
        print("No learned DOM patterns found.")
        return

    with open(log_path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    for idx, entry in enumerate(entries):
        print(f"\n[{idx}] Selector: {entry.get('selector')}")
        print(f"    Example HTML: {entry.get('example_html')[:200]}...")
        print(f"    Context: {entry.get('context')}")
        print("-" * 40)

    while True:
        cmd = input("\nEnter entry number to approve/delete, or 'q' to quit: ").strip()
        if cmd.lower() == "q":
            break
        if cmd.isdigit():
            idx = int(cmd)
            if 0 <= idx < len(entries):
                action = input("Approve (a) or Delete (d) this entry? [a/d]: ").strip().lower()
                if action == "d":
                    entries.pop(idx)
                    print("Entry deleted.")
                elif action == "a":
                    entries[idx]["approved"] = True
                    print("Entry approved.")
                else:
                    print("Unknown action.")
            else:
                print("Invalid entry number.")
        # Save changes
        with open(log_path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        print("Changes saved.")

def auto_approve_dom_pattern(selector, log_path=None, min_count=2):
    """
    Auto-approves a pattern if it appears at least min_count times.
    """
    patterns = load_dom_patterns(log_path)
    count = sum(1 for p in patterns if p.get("selector") == selector)
    for p in patterns:
        if p.get("selector") == selector and count >= min_count:
            p["approved"] = True
    # Save back
    if log_path is None:
        log_path = get_safe_log_path()
    with open(log_path, "w", encoding="utf-8") as f:
        for p in patterns:
            f.write(json.dumps(p) + "\n")

# --- Structure Detection & Classification ---

def find_tables_with_panel_headings(page, panel_selector="p-panel.ballot-item", header_selector="h1.panel-header span.ng-star-inserted", fallback_header_selector="h1.panel-header", table_selector="table.contest-table"):
    """
    Finds all tables inside panels, associates each with the panel's heading.
    Returns a list of (district_name, table_locator) tuples.
    """
    results = []
    panels = page.locator(panel_selector)
    for i in range(panels.count()):
        panel = panels.nth(i)
        # Try to get the span inside the h1.panel-header
        district_name = ""
        header_span = panel.locator(header_selector)
        if header_span.count() > 0:
            district_name = header_span.nth(0).inner_text().strip()
        else:
            header = panel.locator(fallback_header_selector)
            if header.count() > 0:
                district_name = header.nth(0).inner_text().strip()
        # Find the table inside this panel
        table = panel.locator(table_selector)
        if table.count() == 0:
            continue
        results.append((district_name, table.nth(0)))
    return results

def find_tables_with_section_headings(page, heading_tags=None, extra_heading_selectors=None, max_depth=6):
    """
    For each table on the page, walk up the DOM to find the nearest section heading.
    Returns a list of (section_name, table_locator) tuples.
    - heading_tags: tuple of heading tags to consider (default: h1-h6, span, strong, b)
    - extra_heading_selectors: list of additional selectors (e.g., ".ng-star-inserted")
    - max_depth: how many parent levels to walk up
    """
    if heading_tags is None:
        heading_tags = ("h1", "h2", "h3", "h4", "h5", "h6", "span", "strong", "b")
    if extra_heading_selectors is None:
        extra_heading_selectors = [".ng-star-inserted", ".section-title", ".panel-header", ".fw-bold"]

    results = []
    tables = page.locator("table")
    for i in range(tables.count()):
        table = tables.nth(i)
        section_name = None

        # 1. Walk up DOM for heading tags
        parent = table
        for _ in range(max_depth):
            parent = parent.locator("xpath=..")
            # Try heading tags
            for tag in heading_tags:
                headings = parent.locator(tag)
                if headings.count() > 0:
                    section_name = headings.nth(0).inner_text().strip()
                    if section_name:
                        break
            if section_name:
                break
            # Try extra selectors
            for sel in extra_heading_selectors:
                extra = parent.locator(sel)
                if extra.count() > 0:
                    section_name = extra.nth(0).inner_text().strip()
                    if section_name:
                        break
            if section_name:
                break
            # Try ARIA label
            try:
                aria_label = parent.evaluate("el => el.getAttribute('aria-label')")
                if aria_label:
                    section_name = aria_label.strip()
                    break
            except Exception:
                pass

        # 2. Fallback: preceding sibling heading
        if not section_name:
            for tag in heading_tags:
                sibling = table.locator(f"xpath=preceding-sibling::{tag}[1]")
                if sibling.count() > 0:
                    section_name = sibling.nth(0).inner_text().strip()
                    if section_name:
                        break

        # 3. Fallback: use table index
        if not section_name:
            section_name = f"Section {i+1}"

        results.append((section_name, table))
    return results

def is_candidate_major_row(headers, data, coordinator, context):
    # First column is candidate, rest are vote types or totals
    if not headers or not data:
        headers, data = robust_table_extraction(context.get("page"), context)
        if not headers or not data:
            logger.error("[TABLE BUILDER] No data could be extracted from the page.")
            return [], []
    candidate_major_headers = {"Candidate", "Election Day", "Early Voting", "Absentee Mail", "Total Votes"}
    if set(headers) == candidate_major_headers:
        structure_info = {"type": "candidate-major", "candidate_col": 0, "ballot_type_cols": [1, 2, 3]}
    else:
        structure_info = detect_table_structure(headers, data, coordinator)
    logger.info(f"[TABLE BUILDER] Detected table structure: {structure_info}")        
    first_col = normalize_text(headers[0])
    return first_col in CANDIDATE_KEYWORDS and len(data) > 1

def is_candidate_major_col(headers, data, context):
    # First row is vote type, columns are candidates (not location)
    if not headers or not data:
        headers, data = robust_table_extraction(context.get("page"), context)
        return False
    return (
        all(normalize_text(h) not in LOCATION_KEYWORDS for h in headers)
        and any(normalize_text(h) in CANDIDATE_KEYWORDS for h in headers)
    )

def is_precinct_major(headers, coordinator):
    # First column is a location/precinct/district
    location_patterns = set(coordinator.library.get("location_patterns", LOCATION_KEYWORDS))
    return headers and normalize_text(headers[0]) in location_patterns

def is_flat_candidate_table(headers):
    # Only candidate and total columns (no locations)
    if not headers:
        rprint("[red][ERROR] No headers extracted from table. Skipping this table.[/red]")
        return False
    first_col = normalize_text(headers[0])
    return (
        first_col in CANDIDATE_KEYWORDS and
        all(
            any(kw in normalize_text(h) for kw in TOTAL_KEYWORDS.union(CANDIDATE_KEYWORDS))
            for h in headers
        )
    )

def is_single_row_summary(data):
    # Only one row, likely a summary
    return len(data) == 1

def is_candidate_footer(data):
    # Last row contains candidate or misc footer keywords
    if not data or not data[-1]:
        return False
    last_row = data[-1]
    return any(
        any(kw in normalize_text(str(v)) for kw in CANDIDATE_KEYWORDS.union(MISC_FOOTER_KEYWORDS))
        for v in last_row.values()
    )

def detect_wide_vs_long(headers, data):
    """
    Detect if table is wide or long format.
    """
    # Heuristic: if there are many columns and few rows, it's wide
    if len(headers) > 10 and len(data) < 10:
        return "wide"
    # If there are few columns and many rows, it's long
    if len(headers) <= 5 and len(data) > 10:
        return "long"
    return "ambiguous"

def classify_ambiguous_tables(headers, data, coordinator):
    """
    Use ML or rules to classify ambiguous structures.
    """
    # Example: Use ML model or rules
    # For now, use NER and heuristics
    col_types = advanced_party_candidate_detection(headers, coordinator)
    if col_types["candidate"] and col_types["location"]:
        return "precinct-major"
    elif col_types["candidate"]:
        return "candidate-major"
    else:
        return "ambiguous"

# ===================================================================
# END OF FILE
# ===================================================================