import os
import re
import json
from typing import List, Dict, Any, Tuple
from .shared_logger import logger, rprint
import unicodedata
import time
from rich.table import Table
from ..config import BASE_DIR
import hashlib

LOG_PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "log"))

from .table_core import (
    robust_table_extraction,
    extract_table_data,
    harmonize_headers_and_data,
    detect_table_structure,
    normalize_text,
    normalize_header_name,
    load_dom_patterns,
    BALLOT_TYPES,
    LOCATION_KEYWORDS,
    TOTAL_KEYWORDS,
    MISC_FOOTER_KEYWORDS,
    extract_table_data,
    get_safe_log_path,
    detect_table_structure
)

from ..config import BASE_DIR
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..Context_Integration.context_coordinator import ContextCoordinator
context_cache = {}

# --- Robust Table Type Detection Helpers ---

CANDIDATE_KEYWORDS = {"candidate", "candidates", "name", "nominee"}
BALLOT_TYPE_KEYWORDS = {"election day", "early voting", "absentee", "mail", "provisional", "affidavit", "other", "void"}
TABLE_STRUCTURE_CACHE_PATH = os.path.join(BASE_DIR, "parser", "Context_Integration", "Context_Library", "table_structure_cache.json")


def dynamic_table_extractor(page, context, coordinator):
    # 1. Find all tabular candidates in the DOM
    candidates = find_tabular_candidates(page)
    enriched_candidates = []
    for cand in candidates:
        # 2. Run NLP analysis on headers and sample rows
        cand = analyze_candidate_nlp(cand, coordinator)
        # 3. Score/rank using DOM+NLP features
        cand['score'], cand['rationale'] = score_candidate(cand, context, coordinator)
        enriched_candidates.append(cand)
    # 4. Sort and select best
    enriched_candidates.sort(key=lambda c: c['score'], reverse=True)
    best = enriched_candidates[0] if enriched_candidates else None
    # 5. Feedback loop
    if best:
        # PATCH: Add progressive verification and interactive feedback loop
        from .table_core import progressive_table_verification, interactive_feedback_loop
        headers, data, structure_info = progressive_table_verification(best['headers'], best['rows'], coordinator, context)
        if not structure_info.get("verified"):
            headers, data, structure_info = interactive_feedback_loop(headers, data, structure_info)
        return headers, data
    return [], []

# Support functions (case-based, not monolithic)
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
        from .table_core import extract_table_data
        headers, data = extract_table_data(table)
        if headers and data:
            candidates.append({"headers": headers, "rows": data, "source": "table"})
    # 2. Repeated DOM structures (divs, lists, etc.)
    from .table_core import extract_rows_and_headers_from_dom
    headers, data = extract_rows_and_headers_from_dom(page)
    if headers and data:
        candidates.append({"headers": headers, "rows": data, "source": "repeated_dom"})
    # 3. Pattern-based extraction (if any patterns are approved)
    from .table_core import extract_with_patterns, guess_headers_from_row
    pattern_rows = extract_with_patterns(page)
    if pattern_rows:
        headers = guess_headers_from_row(pattern_rows[0][1])
        data = []
        for heading, row, pat in pattern_rows:
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

def feedback_and_confirm(candidate, context, coordinator):
    """
    Interactive feedback loop for user to confirm or correct table structure.
    Returns possibly corrected headers and data.
    """
    from .table_core import progressive_table_verification, interactive_feedback_loop
    headers = candidate.get("headers", [])
    data = candidate.get("rows", [])
    headers, data, structure_info = progressive_table_verification(headers, data, coordinator, context)
    if not structure_info.get("verified"):
        headers, data, structure_info = interactive_feedback_loop(headers, data, structure_info)
    return headers, data

## Core Extraction & Harmonization ##

def deduplicate_headers(headers, data):
    """Remove duplicate headers by normalized name, keep first occurrence."""
    seen = set()
    new_headers = []
    for h in headers:
        norm = normalize_header_name(h)
        if norm not in seen:
            new_headers.append(h)
            seen.add(norm)
    new_data = [{h: row.get(h, "") for h in new_headers} for row in data]
    return new_headers, new_data





# --- ROW FILTERING ---


# --- COLUMN FILTERING ---

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

## DOM/NLP Analysis & Scoring ## 

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

# --- COLUMN TYPE INFERENCE ---

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
            mean = np.mean(nums)
            std = np.std(nums)
            types[h] = "numeric"
        elif all(is_date_like(v) for v in non_empty):
            types[h] = "date"
        elif len(set(non_empty)) < 10:
            types[h] = "categorical"
        else:
            types[h] = "string"
    return types

# --- HEADER DETECTION HEURISTICS ---

def normalize_header(header, lang="en"):
    """
    Normalize header for comparison: lower, strip, remove accents, and translate if needed.
    """
    header = header.strip().lower()
    header = unicodedata.normalize('NFKD', header).encode('ascii', 'ignore').decode('ascii')
    # Optionally: add translation for non-English headers here using a translation dictionary or service
    # Example: if lang != "en": header = translate(header, lang)
    return header

def extract_candidates_and_parties(headers: List[str], coordinator: "ContextCoordinator") -> Dict[str, Dict[str, List[str]]]:
    """
    Returns a dict: {party: {candidate: [ballot_types]}}
    """
    # Use coordinator to extract all known parties and ballot types
    known_parties = [
        "Democratic", "DEM", "dem", 
        "Republican", "REP", "rep", 
        "Working Families", "WOR", "wor" 
        "Conservative", "CON", "con", 
        "Green", "GRN", "grn", 
        "Libertarian", "LIB", "lib", 
        "Independent", "IND", "ind",
        "Larouche", "Write-In", "Other"                     
    ]
    ballot_types = BALLOT_TYPES

    # Group headers by candidate/party/ballot type
    candidate_party_map = {}
    for h in headers:
        # Try to parse: Candidate (Party) - BallotType
        m = re.match(r"(.+?)\s*\((.+?)\)\s*-\s*(.+)", h)
        if m:
            candidate, party, ballot_type = m.groups()
        else:
            # Try: Candidate - BallotType
            m = re.match(r"(.+?)\s*-\s*(.+)", h)
            if m:
                candidate, ballot_type = m.groups()
                party = ""
            else:
                candidate, party, ballot_type = h, "", ""
        candidate = candidate.strip()
        party = party.strip()
        ballot_type = ballot_type.strip()
        # Fuzzy match party
        if party:
            best_party, score = max(((p, coordinator.fuzzy_score(party, p)) for p in known_parties), key=lambda x: x[1])
            if score > 80:
                party = best_party
        else:
            # Try to infer party from candidate name using NER
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
    # Clamp score to [0, 1]
    score = max(0.0, min(1.0, score))
    rationale.append(f"Final score: {score:.2f}")

    return score, "; ".join(rationale)

## Candidate Selection & Feedback ## 
# --- USER FEEDBACK LOOP ---

def prompt_user_to_confirm_table_structure(headers, data, domain, contest_title, coordinator):
    import copy

    should_log = True
    columns_changed = False
    new_headers = copy.deepcopy(headers)
    # PATCH: Use log parent dir for denied structures
    denied_structures_path = os.path.join(LOG_PARENT_DIR, "denied_table_structures.json")
    denied_structures = {}
    denied_structures_dir = os.path.dirname(denied_structures_path)
    os.makedirs(denied_structures_dir, exist_ok=True)
    if os.path.exists(denied_structures_path):
        with open(denied_structures_path, "r", encoding="utf-8") as f:
            denied_structures = json.load(f)
    sig = f"{domain}:{table_signature(headers)}"
    denied_count = denied_structures.get(sig, 0)

    # PATCH: Use log parent dir for removed columns log
    removed_columns_log_path = os.path.join(LOG_PARENT_DIR, "removed_columns_log.json")
    removed_columns_log_dir = os.path.dirname(removed_columns_log_path)
    os.makedirs(removed_columns_log_dir, exist_ok=True)
    if os.path.exists(removed_columns_log_path):
        with open(removed_columns_log_path, "r", encoding="utf-8") as f:
            removed_columns_log = json.load(f)
    else:
        removed_columns_log = {}

    # ML/NLP suggestions
    ml_scores = []
    nlp_suggestions = []
    for h in new_headers:
        score = coordinator.score_header(h, {"contest_title": contest_title})
        ml_scores.append(score)
        ents = coordinator.extract_entities(h)
        if ents:
            ent, label = ents[0]
            nlp_suggestions.append((h, ent, label))
        else:
            nlp_suggestions.append((h, None, None))

    avg_score = sum(ml_scores) / len(ml_scores) if ml_scores else 0
    auto_accept_threshold = 0.93  # Accept automatically if ML is very confident

    # PATCH: If ML confidence is low and NLP suggests better header names, auto-apply those suggestions
    if avg_score < 0.7 and any(ent and ent != h for h, ent, label in nlp_suggestions):
        logger.info("[TABLE BUILDER] ML confidence low and NLP suggests better header names. Auto-applying suggestions.")
        for idx, (h, ent, label) in enumerate(nlp_suggestions):
            if ent and ent != h:
                new_headers[idx] = ent
        new_headers, data = harmonize_headers_and_data(new_headers, data)
        # Recompute ML scores after change
        ml_scores = [coordinator.score_header(h, {"contest_title": contest_title}) for h in new_headers]
        avg_score = sum(ml_scores) / len(ml_scores) if ml_scores else 0

    # Multiple structure candidates (if available)
    structure_candidates = [new_headers]
    # Optionally, try to generate alternative header orders/types using ML/NLP
    alt_headers = []
    for idx, (h, ent, label) in enumerate(nlp_suggestions):
        if ent and ent != h:
            alt = copy.deepcopy(new_headers)
            alt[idx] = ent
            alt_headers.append(alt)
    if alt_headers:
        structure_candidates += alt_headers

    candidate_idx = 0
    while True:
        candidate_headers = structure_candidates[candidate_idx]
        # Show ML/NLP confidence and suggestions
        rprint(f"\n[bold yellow][Table Builder] Candidate structure {candidate_idx+1}/{len(structure_candidates)} for '{contest_title}':[/bold yellow]")
        preview_table = Table(show_header=True, header_style="bold magenta")
        N = min(5, len(data))  # Show up to 5 values, or all if fewer rows
        rprint(f"[bold green]Column content preview (first {N} rows):[/bold green]")
        for h in candidate_headers:
            preview_table.add_column(h)
            values = [str(row.get(h, "")) for row in data[:N]]
            preview_vals = [v if len(v) < 30 else v[:27] + "..." for v in values]
            rprint(f"[cyan]{h}[/cyan]: {preview_vals}")
        for row in data[:5]:
            preview_table.add_row(*(str(row.get(h, "")) for h in candidate_headers))
        rprint(preview_table)
        rprint(f"[cyan]ML average confidence: {avg_score:.2f}[/cyan]")
        if nlp_suggestions:
            rprint("[cyan]NLP suggestions:[/cyan]")
            for h, ent, label in nlp_suggestions:
                if ent and ent != h:
                    rprint(f"  [green]{h}[/green] → [yellow]{ent}[/yellow] ({label})")
        if len(structure_candidates) > 1:
            rprint(f"[cyan]Use [N]ext/[P]revious to cycle through {len(structure_candidates)} candidates.[/cyan]")

        # Auto-accept if ML is very confident
        if avg_score >= auto_accept_threshold:
            rprint("[green]ML confidence is high. Auto-accepting this structure.[/green]")
            new_headers = candidate_headers
            break

        rprint("[bold cyan]Options:[/bold cyan]")
        rprint("  [Y] Accept as correct")
        rprint("  [N] Reject (log as denied structure)")
        rprint("  [C] Mark columns as incorrect (remove)")
        rprint("  [O] Reorder columns")
        rprint("  [R] Rename columns")
        rprint("  [A] Add missing columns")
        if len(structure_candidates) > 1:
            rprint("  [Next] Show next candidate structure")
            rprint("  [Prev] Show previous candidate structure")
        resp = input("Accept, Reject, mark Columns, reorder, Rename, Add, Next, or Prev? [Y/n/c/o/r/a/next/prev]: ").strip().lower()
        if resp in ("", "y", "yes"):
            new_headers = candidate_headers
            should_log = True
            break
        elif resp in ("n", "no"):
            denied_structures[sig] = denied_structures.get(sig, 0) + 1
            with open(denied_structures_path, "w", encoding="utf-8") as f:
                json.dump(denied_structures, f, indent=2)
            logger.info(f"[TABLE BUILDER] User declined to log table structure for '{contest_title}'. Denied {denied_structures[sig]} times.")
            if denied_structures[sig] >= 3:
                logger.warning(f"[TABLE BUILDER] Structure for '{contest_title}' denied {denied_structures[sig]} times. Will not auto-apply in future.")
            retry = input("Would you like to retry correction? [y/N]: ").strip().lower()
            if retry in ("y", "yes"):
                continue
            else:
                return headers, data
        elif resp == "c":
            rprint("Enter column numbers (comma-separated) that are incorrect (starting from 1):")
            for idx, h in enumerate(candidate_headers):
                rprint(f"  {idx+1}: {h}")
            wrong_cols = input("Columns to mark as incorrect: ").strip()
            if wrong_cols:
                wrong_idxs = [int(i)-1 for i in wrong_cols.split(",") if i.strip().isdigit()]
                for idx in wrong_idxs:
                    if 0 <= idx < len(candidate_headers):
                        rprint(f"[red]Column '{candidate_headers[idx]}' marked as incorrect.[/red]")
                        col_name = candidate_headers[idx]
                        removed_columns_log.setdefault(contest_title, {})
                        removed_columns_log[contest_title][col_name] = removed_columns_log[contest_title].get(col_name, 0) + 1
                candidate_headers = [h for i, h in enumerate(candidate_headers) if i not in wrong_idxs]
                data = [{h: row.get(h, "") for h in candidate_headers} for row in data]
                columns_changed = True
                structure_candidates[candidate_idx] = candidate_headers
            with open(removed_columns_log_path, "w", encoding="utf-8") as f:
                json.dump(removed_columns_log, f, indent=2)
        elif resp == "o":
            rprint("Enter new order of columns as space/comma-separated numbers (starting from 1):")
            for idx, h in enumerate(candidate_headers):
                rprint(f"  {idx+1}: {h}")
            order = input("New order: ").replace(",", " ").split()
            try:
                new_order = [candidate_headers[int(i)-1] for i in order if i.strip().isdigit() and 0 < int(i) <= len(candidate_headers)]
                if new_order:
                    candidate_headers = new_order
                    data = [{h: row.get(h, "") for h in candidate_headers} for row in data]
                    columns_changed = True
                    structure_candidates[candidate_idx] = candidate_headers
                    rprint(f"[green]Columns reordered.[/green]")
            except Exception as e:
                rprint(f"[red]Invalid order: {e}[/red]")
        elif resp == "r":
            rprint("Enter column numbers (comma-separated) to rename (starting from 1):")
            for idx, h in enumerate(candidate_headers):
                rprint(f"  {idx+1}: {h}")
            col_nums = input("Columns to rename: ").strip()
            if col_nums:
                rename_idxs = [int(i)-1 for i in col_nums.split(",") if i.strip().isdigit() and 0 <= int(i)-1 < len(candidate_headers)]
                for idx in rename_idxs:
                    old_name = candidate_headers[idx]
                    new_name = input(f"Rename column '{old_name}' to: ").strip()
                    if new_name:
                        rprint(f"[yellow]Renamed '{old_name}' to '{new_name}'[/yellow]")
                        candidate_headers[idx] = new_name
                data = [{h: row.get(h, "") for h in candidate_headers} for row in data]
                columns_changed = True
                structure_candidates[candidate_idx] = candidate_headers
        elif resp == "a":
            rprint("Enter names of columns to add, separated by commas:")
            add_cols = input("Columns to add: ").split(",")
            for col in add_cols:
                col = col.strip()
                if col and col not in candidate_headers:
                    candidate_headers.append(col)
                    for row in data:
                        row[col] = ""
                    rprint(f"[green]Added column '{col}'[/green]")
            columns_changed = True
            structure_candidates[candidate_idx] = candidate_headers
        elif resp in ("next", "nxt"):
            candidate_idx = (candidate_idx + 1) % len(structure_candidates)
            continue
        elif resp in ("prev", "previous"):
            candidate_idx = (candidate_idx - 1) % len(structure_candidates)
            continue
        else:
            rprint("[red]Unknown option. Please try again.[/red]")

        # Always harmonize after user modification
        candidate_headers, data = harmonize_headers_and_data(candidate_headers, data)

    # Save user-confirmed structure for future ML learning
    if should_log and hasattr(coordinator, "log_table_structure"):
        coordinator.log_table_structure(contest_title, new_headers, context={"domain": domain})
        cache_table_structure(domain, new_headers, new_headers)
        logger.info(f"[TABLE BUILDER] Logged confirmed table structure for '{contest_title}'.")
        if hasattr(coordinator, "save_table_structure_to_db"):
            coordinator.save_table_structure_to_db(
                contest_title=contest_title,
                headers=new_headers,
                context={"domain": domain},
                ml_confidence=avg_score if 'avg_score' in locals() else None,
                confirmed_by_user=True
            )
    # Always harmonize before returning
    new_headers, data = harmonize_headers_and_data(new_headers, data)
    return new_headers, data

def interactive_batch_operations(headers, data):
    """
    Allow batch renaming, reordering, or removal of columns in the CLI.
    """
    import copy
    history = []
    while True:
        rprint("\n[bold cyan]Batch Operations: [R]ename, [O]rder, [D]elete, [U]ndo, [Q]uit[/bold cyan]")
        cmd = input("Choose operation: ").strip().lower()
        if cmd == "r":
            rprint("Enter column numbers (comma-separated) to rename:")
            for idx, h in enumerate(headers):
                rprint(f"  {idx+1}: {h}")
            col_nums = input("Columns to rename: ").strip()
            if col_nums:
                rename_idxs = [int(i)-1 for i in col_nums.split(",") if i.strip().isdigit() and 0 <= int(i)-1 < len(headers)]
                history.append((copy.deepcopy(headers), copy.deepcopy(data)))
                for idx in rename_idxs:
                    old_name = headers[idx]
                    new_name = input(f"Rename column '{old_name}' to: ").strip()
                    if new_name:
                        headers[idx] = new_name
                data = [{h: row.get(h, "") for h in headers} for row in data]
        elif cmd == "o":
            rprint("Enter new order of columns as space/comma-separated numbers (starting from 1):")
            for idx, h in enumerate(headers):
                rprint(f"  {idx+1}: {h}")
            order = input("New order: ").replace(",", " ").split()
            try:
                new_order = [headers[int(i)-1] for i in order if i.strip().isdigit() and 0 < int(i) <= len(headers)]
                if new_order:
                    history.append((copy.deepcopy(headers), copy.deepcopy(data)))
                    headers = new_order
                    data = [{h: row.get(h, "") for h in headers} for row in data]
            except Exception as e:
                rprint(f"[red]Invalid order: {e}[/red]")
        elif cmd == "d":
            rprint("Enter column numbers (comma-separated) to delete:")
            for idx, h in enumerate(headers):
                rprint(f"  {idx+1}: {h}")
            del_nums = input("Columns to delete: ").strip()
            if del_nums:
                del_idxs = [int(i)-1 for i in del_nums.split(",") if i.strip().isdigit() and 0 <= int(i)-1 < len(headers)]
                history.append((copy.deepcopy(headers), copy.deepcopy(data)))
                headers = [h for i, h in enumerate(headers) if i not in del_idxs]
                data = [{h: row.get(h, "") for h in headers} for row in data]
        elif cmd == "u":
            if history:
                headers, data = history.pop()
                rprint("[green]Undo successful.[/green]")
            else:
                rprint("[yellow]Nothing to undo.[/yellow]")
        elif cmd == "q":
            break
        else:
            rprint("[red]Unknown option.[/red]")
    return headers, data

def auto_suggest_corrections(headers, data, coordinator):
    """
    Suggest likely corrections based on previous user feedback or ML confidence.
    """
    suggestions = []
    for h in headers:
        score = coordinator.score_header(h, {})
        if score < 0.7:
            suggestions.append((h, "Low ML confidence"))
    # Add more suggestions based on previous feedback logs if available
    return suggestions

# --- ML/NLP INTEGRATION ---

def dynamic_confidence_threshold(history, default=0.93):
    """
    Adjust threshold for auto-accepting structures based on past accuracy.
    """
    # Example: If last 5 were correct, raise threshold, else lower
    if not history:
        return default
    correct = sum(1 for h in history[-5:] if h["accepted"])
    if correct >= 4:
        return min(0.98, default + 0.02)
    elif correct <= 2:
        return max(0.85, default - 0.05)
    return default

def rescan_and_verify(headers: List[str], data: List[Dict[str, Any]], coordinator: "ContextCoordinator", context: dict, threshold: float = 0.85) -> Tuple[List[str], List[Dict[str, Any]], bool]:
    """
    Rescans headers and data, verifies with ML/NER, and retries if below threshold.
    Returns (headers, data, passed)
    """
    # Use coordinator's ML/NER to score headers
    scores = []
    for h in headers:
        score = coordinator.score_header(h, context)
        scores.append(score)
    avg_score = sum(scores) / len(scores) if scores else 0
    passed = avg_score >= threshold
    if not passed:
        # Attempt to re-extract or re-map headers using NER/ML
        new_headers = []
        for h in headers:
            entities = coordinator.extract_entities(h)
            if entities:
                # Use the most likely entity label
                ent, label = entities[0]
                new_headers.append(ent)
            else:
                new_headers.append(h)
        headers = new_headers
        # Optionally, re-harmonize data
        headers, data = harmonize_headers_and_data(headers, data)
    logger.info(f"[TABLE BUILDER] Rescan and verify final table: {len(data)} rows, {len(headers)} columns (learned structure).")
    return headers, data, passed

## Pattern/Selector Discovery & Logging ## 

def find_tables_with_headings(page, dom_segments=None, heading_tags=None, include_section_context=True):
    """
    Finds all tables on the page and pairs each with its nearest heading or ARIA landmark.
    - If dom_segments is provided (from scan_html_for_context), uses that for robust matching.
    - Otherwise, falls back to Playwright DOM traversal.
    - Supports nested sections, ARIA landmarks, and fieldset legends.
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
            results.append((heading, table_locator))
    else:
        # Fallback: Use Playwright only
        tables = page.locator("table")
        for i in range(tables.count()):
            table = tables.nth(i)
            heading = None
            section_context = None
            try:
                # Try ARIA landmarks/regions
                parent = table
                for _ in range(5):  # Walk up to 5 ancestors
                    parent = parent.locator("xpath=..")
                    attrs = parent.evaluate("el => ({'role': el.getAttribute('role'), 'aria-label': el.getAttribute('aria-label'), 'aria-labelledby': el.getAttribute('aria-labelledby'), 'tag': el.tagName.toLowerCase()})")
                    if attrs.get("role") in ("region", "complementary", "main", "navigation", "search") or attrs.get("aria-label"):
                        section_context = attrs.get("aria-label") or attrs.get("role")
                        break
                    if attrs.get("tag") in ("section", "fieldset"):
                        # Try to find a legend or heading inside this section
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
                # Try previous heading sibling
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



## Structure Handling ## 
# auto-detection in detect_table_structure ---

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
    
# --- STRUCTURE DETECTION ---

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

## Other Utilities ## 

# --- PERFORMANCE & LOGGING ---

def profile_extraction_step(func):
    """
    Decorator to profile extraction speed.
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"[PROFILE] {func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper

def log_decision(decision, context=None):
    """
    Log not just errors but also decisions made by heuristics for later review.
    """
    logger.info(f"[DECISION] {decision} | Context: {context}")

def robust_html_fallback(page):
    """
    Add more robust fallbacks for broken or inconsistent markup.
    """
    try:
        html = page.content()
        # Try to parse with BeautifulSoup as a fallback
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        tables = soup.find_all("table")
        all_tables = []
        for table in tables:
            rows = table.find_all("tr")
            headers = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]
            data = []
            for row in rows[1:]:
                cells = row.find_all(["td", "th"])
                data.append({headers[i]: cells[i].get_text(strip=True) if i < len(cells) else "" for i in range(len(headers))})
            all_tables.append((headers, data))
        return all_tables
    except Exception as e:
        logger.error(f"[HTML FALLBACK] Error: {e}")
        return []
    
# --- EDGE CASES ---

def handle_nested_tables(page):
    """
    Handle tables within tables or complex nested DOM structures.
    """
    tables = page.locator("table table")
    results = []
    for i in range(tables.count()):
        table = tables.nth(i)
        headers, data = extract_table_data(table)
        results.append((headers, data))
    return results

def review_learned_table_structures(log_path=None):
    """
    CLI to review/edit learned table structures.
    """
    # PATCH: Use log directory parent to webapp for default path
    if log_path is None:
        from ..config import BASE_DIR
        import os
        LOG_PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "log"))
        log_path = os.path.join(LOG_PARENT_DIR, "table_structure_learning_log.jsonl")
    if not os.path.exists(log_path):
        print("No learned table structures found.")
        return

    entries = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                entries.append(entry)
            except Exception:
                continue

    for idx, entry in enumerate(entries):
        print(f"\n[{idx}] Contest: {entry.get('contest_title')}")
        print(f"    Headers: {entry.get('headers')}")
        print(f"    Context: {entry.get('context')}")
        print(f"    Result: {entry.get('result')}")
        print("-" * 40)

    while True:
        cmd = input("\nEnter entry number to delete/edit, or 'q' to quit: ").strip()
        if cmd.lower() == "q":
            break
        if cmd.isdigit():
            idx = int(cmd)
            if 0 <= idx < len(entries):
                action = input("Delete (d) or Edit (e) this entry? [d/e]: ").strip().lower()
                if action == "d":
                    entries.pop(idx)
                    print("Entry deleted.")
                elif action == "e":
                    new_headers = input("Enter new headers as comma-separated values: ").strip().split(",")
                    entries[idx]["headers"] = [h.strip() for h in new_headers]
                    print("Headers updated.")
                else:
                    print("Unknown action.")
            else:
                print("Invalid entry number.")
        # Save changes
        with open(log_path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        print("Changes saved.")

def table_signature(headers):
    return hashlib.md5(json.dumps(headers, sort_keys=True).encode()).hexdigest()

def load_table_structure_cache():
    if os.path.exists(TABLE_STRUCTURE_CACHE_PATH):
        with open(TABLE_STRUCTURE_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_table_structure_cache(cache):
    with open(TABLE_STRUCTURE_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

def cache_table_structure(domain, headers, structure):
    cache = load_table_structure_cache()
    sig = f"{domain}:{table_signature(headers)}"
    cache[sig] = structure
    save_table_structure_cache(cache)

def get_cached_table_structure(domain, headers):
    cache = load_table_structure_cache()
    sig = f"{domain}:{table_signature(headers)}"
    return cache.get(sig)

def guess_contest_title(table_headers, known_titles):
    """
    Try to match table headers to known contest titles using fuzzy matching.
    """
    import difflib
    for header in table_headers:
        matches = difflib.get_close_matches(header, known_titles, n=1, cutoff=0.7)
        if matches:
            return matches[0]
    return None

def extract_title_from_html_near_table(table_idx, dom_nodes, window=5):
    """
    Scan nearby DOM nodes for likely contest titles.
    """
    idx_range = range(max(0, table_idx - window), min(len(dom_nodes), table_idx + window + 1))
    for idx in idx_range:
        node = dom_nodes[idx]
        if node.get("tag", "").lower() in {"h1", "h2", "h3", "caption"}:
            text = node.get("html", "").strip()
            if text and len(text.split()) > 2:
                return text
    return None

## Optional/Advanced ## 

def merge_multirow_headers(header_rows):
    """
    Merge multiple header rows (e.g., stacked headers) into a single header list.
    """
    merged = []
    for cols in zip(*header_rows):
        merged_col = " ".join([c for c in cols if c and c.strip() and not c.strip().isdigit()])
        merged.append(merged_col.strip())
    return merged

def fuzzy_merge_headers(headers, threshold=0.85):
    """
    Merge similar headers using fuzzy matching.
    """
    import difflib
    merged = []
    used = set()
    for i, h in enumerate(headers):
        if i in used:
            continue
        group = [h]
        for j, h2 in enumerate(headers):
            if i != j and j not in used:
                score = difflib.SequenceMatcher(None, normalize_header(h), normalize_header(h2)).ratio()
                if score > threshold:
                    group.append(h2)
                    used.add(j)
        merged.append(group[0])  # Keep the first as canonical
        used.add(i)
    return merged

def detect_language(headers):
    """
    Detect language of headers (very basic, can be replaced with langdetect).
    """
    try:
        from langdetect import detect
        text = " ".join(headers)
        return detect(text)
    except Exception:
        return "en"

def is_date_like(val):
    import dateutil.parser
    try:
        dateutil.parser.parse(val)
        return True
    except Exception:
        return False

def dynamic_required_columns(context, default_required=None):
    """
    Adjust required columns based on context.
    """
    if default_required is None:
        default_required = {"Grand Total", "Precinct", "Location"}
    # Example: if context says percent reported is not present, remove it
    if not context.get("has_percent_reported", True):
        default_required.discard("Percent Reported")
    return default_required





