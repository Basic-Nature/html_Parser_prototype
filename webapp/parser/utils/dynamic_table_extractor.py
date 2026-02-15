from __future__ import annotations

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
import difflib
import os
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import dateutil.parser
import numpy as np
import orjson
from selectolax.parser import HTMLParser

from ..config import ENTITY_LINKING_THRESHOLD
from ..Context_Integration.Context_Library.constants import (
    BALLOT_TYPES,
    BALLOT_TYPES_SORT_ORDER,
    CANDIDATE_KEYWORDS,
    CONTAINER_EXTRA_KEYWORDS,
    CONTAINER_FALLBACK_SELECTORS,
    CONTEST_KEYWORDS,
    EXTRA_HEADING_TAGS,
    HEADING_TAGS,
    LOCATION_ABBREVIATIONS,
    LOCATION_KEYWORDS,
    MISC_FOOTER_KEYWORDS,
    NLP_SKIP_PHRASES,
    PANEL_TAGS,
    PARTY_KEYWORDS,
    TOTAL_KEYWORDS,
)
from ..Context_Integration.librarian import (
    extend_heading_tags,
    extend_panel_tags,
    get_safe_log_path,
    log_unknown_tag,
)
from .browser_utils import (
    safe_count,
    safe_evaluate,
    safe_get_attribute,
    safe_inner_text,
    safe_locator,
    safe_nth,
)
from .date_utils import is_date_like
from .detect import extract_table_data, is_location_header, normalize_header, normalize_text
from .dom_extractor import extract_rows_and_headers_from_dom, guess_headers_from_row
from .logger_singleton import logger
from .pattern_extractor import extract_with_patterns, load_dom_patterns
from .shared_logic import (
    safe_append,
    safe_copy,
    safe_get,
    safe_lower,
    safe_replace,
    safe_split,
    safe_strip,
    safe_values,
)
from .table_core import (
    robust_table_extraction,
)

if TYPE_CHECKING:
    from ..Context_Integration.context_coordinator import ContextCoordinator

# -------------------------------------------------------------------
# Structured logging helper aligned with SharedLogger and frontend
# -------------------------------------------------------------------

def _emit(level: str, msg_type: str, message: str, session_id: Optional[str] = None, **fields):
    """
    Emit a structured log payload. Keys:
      - level: uppercased level
      - type: short subsystem label (e.g., "extractor")
      - message: human-readable message
      - session_id: passthrough for frontend correlation
      - additional fields included as provided (non-None)
    """
    payload = {
        "level": level.upper(),
        "type": msg_type,
        "message": message,
        "session_id": session_id,
    }
    for k, v in fields.items():
        if v is not None:
            payload[k] = v
    # Delegate to SharedLogger (CLI/webapp aware)
    getattr(logger, level.lower(), logger.info)(payload)

# --- Main Candidate Generator/Scorer ---

def dynamic_table_extractor(page, context, coordinator, table_html=None) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Finds and scores candidate tables, returning the best (headers, data) for further processing.
    Does NOT run harmonization, annotation, or feedback loop.
    Uses selectolax for HTML parsing if table_html is provided.
    """
    session_id = safe_get(context, "session_id", None)
    _emit("info", "extractor", "[EXTRACTOR] Starting dynamic table extraction", session_id,
          has_table_html=bool(table_html))

    # HTML string path (selectolax)
    if table_html:
        try:
            soup = HTMLParser(table_html)
            table = soup.css_first("table")
            if not table:
                _emit("warning", "extractor", "[EXTRACTOR] No <table> found in provided table_html.", session_id)
                return [], []
            # Find all rows (tr) in the table
            rows = table.css("tr")
            if not rows:
                _emit("warning", "extractor", "[EXTRACTOR] No <tr> rows found in table_html.", session_id)
                return [], []
            # Extract headers from first row (th or td)
            header_cells = rows[0].css("th") or rows[0].css("td")
            headers = [cell.text(strip=True) for cell in header_cells]
            data = []
            for row in rows[1:]:
                cells = row.css("td") or row.css("th")
                row_dict = {}
                for i in range(len(headers)):
                    val = cells[i].text(strip=True) if i < len(cells) else ""
                    row_dict[headers[i]] = val
                data.append(row_dict)
            # Attach context to each row if Precinct/panel_heading present
            precinct = safe_get(context, "panel_heading") or safe_get(context, "Precinct")
            if precinct:
                if "Precinct" not in headers:
                    headers = ["Precinct"] + headers
                for row in data:
                    row["Precinct"] = precinct
            _emit("info", "extractor", "[EXTRACTOR] Extracted rows from HTML table", session_id,
                  rows=len(data), cols=len(headers))
            return headers, data
        except Exception as e:
            _emit("error", "extractor", "[EXTRACTOR] Failed to parse table_html with selectolax", session_id, error=str(e))
            return [], []

    # Playwright/DOM path
    try:
        candidates = find_tabular_candidates(page, context=context, session_id=session_id)
    except Exception as e:
        _emit("error", "extractor", "[EXTRACTOR] Candidate discovery failed", session_id, error=str(e))
        candidates = []

    enriched_candidates = []
    for cand in candidates:
        try:
            cand = analyze_candidate_nlp(cand, coordinator, session_id=session_id)
            score, rationale = score_candidate(cand, context, coordinator, session_id=session_id)
            cand["score"], cand["rationale"] = score, rationale
            enriched_candidates.append(cand)
        except Exception as e:
            _emit("warning", "extractor", "[EXTRACTOR] Candidate NLP/score step failed", session_id, error=str(e))

    enriched_candidates.sort(key=lambda c: c.get("score", 0.0), reverse=True)
    best = enriched_candidates[0] if enriched_candidates else None
    if best:
        _emit("info", "extractor", "[EXTRACTOR] Best candidate selected", session_id,
              source=safe_get(best, "source"), score=round(safe_get(best, "score", 0.0), 3),
              rows=len(safe_get(best, "rows", [])), cols=len(safe_get(best, "headers", [])))
        # Attach context to each row if Precinct/panel_heading is present
        precinct = safe_get(context, "panel_heading") or safe_get(context, "Precinct")
        if precinct and "Precinct" not in safe_get(best, "headers", []):
            best["headers"] = ["Precinct"] + best["headers"]
            for row in best["rows"]:
                row["Precinct"] = precinct
        return best["headers"], best["rows"]

    _emit("warning", "extractor", "[EXTRACTOR] No suitable table candidates found.", session_id)
    return [], []

# --- Candidate Generation & Scoring ---

def find_tabular_candidates(page, context=None, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Find all DOM elements that look like tables or repeated row structures.
    Returns a list of candidate dicts with 'headers' and 'rows'.
    Uses safe_locator, safe_count, safe_nth, and safe_copy for robustness.
    """
    candidates = []
    # 1. Standard HTML tables
    try:
        tables = safe_locator(page, "table", logger)
        table_count = safe_count(tables, logger)
        _emit("debug", "extractor", "[EXTRACTOR] Scanning <table> elements", session_id, count=table_count)
        for i in range(table_count):
            table = safe_nth(tables, i, logger)
            if table is None:
                continue
            # Pass context to extract_table_data for consistency
            headers, data, _ = extract_table_data(table, structure_info={"context": context or {}})
            if headers and data:
                candidate = {"headers": headers, "rows": data, "source": "table"}
                if context:
                    candidate["context"] = safe_copy(context)
                candidates.append(candidate)
        _emit("debug", "extractor", "[EXTRACTOR] Table candidates collected", session_id, found=len(candidates))
    except Exception as e:
        _emit("warning", "extractor", "[EXTRACTOR] Error while scanning <table> elements", session_id, error=str(e))

    # 2. Repeated DOM structures (divs, lists, etc.)
    try:
        headers, data, _ = extract_rows_and_headers_from_dom(page, context=context)
        if headers and data:
            candidate = {"headers": headers, "rows": data, "source": "repeated_dom"}
            if context:
                candidate["context"] = safe_copy(context)
            candidates.append(candidate)
            _emit("debug", "extractor", "[EXTRACTOR] Repeated DOM candidate added", session_id, rows=len(data), cols=len(headers))
    except Exception as e:
        _emit("warning", "extractor", "[EXTRACTOR] DOM extraction failed", session_id, error=str(e))

    # 3. Pattern-based extraction (if any patterns are approved)
    try:
        pattern_rows = extract_with_patterns(page, context=context)
        pattern_rows = [tup for tup in pattern_rows if len(tup) > 1 and tup[1] is not None]
        if pattern_rows:
            headers = []
            for tup in pattern_rows:
                row = tup[1]
                if hasattr(row, "locator"):
                    cells = safe_locator(row, "> *", logger)
                    if safe_count(cells, logger) > 0:
                        headers, _ = guess_headers_from_row(row, context=context)
                        break
            if headers:
                data = []
                for heading, row, pat in pattern_rows:
                    if row is None:
                        continue
                    cells = safe_locator(row, "> *", logger)
                    if safe_count(cells, logger) < len(headers):
                        continue
                    row_data = {}
                    for idx in range(safe_count(cells, logger)):
                        cell = safe_nth(cells, idx, logger)
                        val = cell.inner_text().strip() if cell else ""
                        col_name = headers[idx] if idx < len(headers) else f"Column {idx+1}"
                        row_data[col_name] = val
                    # Attach pattern/heading info (diagnostics)
                    if heading is not None:
                        row_data["_pattern_heading"] = heading
                    if pat is not None:
                        row_data["_pattern_id"] = str(pat)
                    if row_data:
                        data.append(row_data)
                if headers and data:
                    candidate = {"headers": headers, "rows": data, "source": "pattern"}
                    if context:
                        candidate["context"] = safe_copy(context)
                    candidates.append(candidate)
                    _emit("debug", "extractor", "[EXTRACTOR] Pattern-based candidate added", session_id, rows=len(data), cols=len(headers))
    except Exception as e:
        _emit("warning", "extractor", "[EXTRACTOR] Pattern extraction failed", session_id, error=str(e))

    _emit("info", "extractor", "[EXTRACTOR] Candidate discovery complete", session_id, candidates=len(candidates))
    return candidates

def analyze_candidate_nlp(candidate, coordinator, session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Enrich a candidate dict with NLP/NER analysis for headers.
    Adds 'header_entities' and 'header_scores' fields.
    Uses the provided coordinator; falls back to default if None.
    """
    from ..Context_Integration.context_coordinator import ContextCoordinator
    coordinator = coordinator or ContextCoordinator()
    headers = safe_get(candidate, "headers", [])
    header_entities = []
    header_scores = []
    for h in headers:
        try:
            ents = coordinator.extract_entities(h)
        except Exception:
            ents = []
        header_entities.append(ents)
        try:
            score = coordinator.score_header(h, {})
        except Exception:
            score = 0.0
        header_scores.append(score)
    candidate["header_entities"] = header_entities
    candidate["header_scores"] = header_scores
    return candidate

def score_candidate(candidate, context, coordinator, session_id: Optional[str] = None) -> Tuple[float, str]:
    """
    Score a candidate table structure using ML/NLP and heuristics.
    Returns (score, rationale).
    Adds a bonus if a location column is present (using is_location_header),
    and penalizes if missing when context expects one.
    Uses provided coordinator; falls back if None.
    """
    from ..Context_Integration.context_coordinator import ContextCoordinator
    coordinator = coordinator or ContextCoordinator()
    headers = safe_get(candidate, "headers", [])
    rows = safe_get(candidate, "rows", [])
    rationale = []

    # 1. ML/NLP header confidence
    ml_scores = []
    for h in headers:
        try:
            score = coordinator.score_header(h, context or {})
        except Exception:
            score = 0.0
        ml_scores.append(score)
    avg_ml_score = sum(ml_scores) / len(ml_scores) if ml_scores else 0.0
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
        try:
            ents = coordinator.extract_entities(h)
        except Exception:
            ents = []
        if ents:
            entity_hits += 1
    if headers:
        entity_bonus = 0.2 * (entity_hits / len(headers))
    rationale.append(f"Entity bonus: {entity_bonus:.2f} ({entity_hits}/{len(headers)} headers)")

    # 4b. Bonus for location column, penalty if missing and context expects one
    has_location_col = any(is_location_header(h) for h in headers)
    location_bonus = 0.15 if has_location_col else 0.0
    location_penalty = 0.0
    if not has_location_col and (context and safe_get(context, "require_location_column", True)):
        location_penalty = -0.15
    if location_bonus:
        rationale.append("Location column bonus: +0.15 (location column detected)")
    if location_penalty:
        rationale.append("Location column penalty: -0.15 (location column missing, expected)")

    # 5. Penalty for generic headers (Column 1, etc.)
    generic_headers = sum(1 for h in headers if re.match(r"Column \d+", h))
    generic_penalty = -0.2 * (generic_headers / len(headers)) if headers else 0.0
    if generic_penalty:
        rationale.append(f"Generic header penalty: {generic_penalty:.2f}")

    # 6. Final score
    score = (
        0.5 * avg_ml_score +
        0.2 * row_score +
        0.2 * col_score +
        fill_penalty +
        entity_bonus +
        generic_penalty +
        location_bonus +
        location_penalty
    )
    score = max(0.0, min(1.0, score))
    rationale.append(f"Final score: {score:.2f}")

    return score, "; ".join(rationale)

# --- Column/Row Filtering & Type Inference ---

def remove_low_signal_columns(headers, data, min_unique=2, min_non_empty_ratio=0.05) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Remove columns with low variance or too many repeated values.
    Uses safe_get for robustness.
    """
    keep = []
    n_rows = len(data)
    for h in headers:
        col_vals = [safe_get(row, h, "") for row in data]
        unique_vals = set(col_vals)
        non_empty = [v for v in col_vals if v not in ("", None)]
        if len(unique_vals) >= min_unique and (len(non_empty) / n_rows if n_rows else 0) >= min_non_empty_ratio:
            keep.append(h)
    return keep, [{h: safe_get(row, h, "") for h in keep} for row in data]

def infer_column_types(headers, data) -> Dict[str, str]:
    """
    Infer column types using safe_get and safe_replace for robustness.
    Uses statistics, regex, numpy, and dateutil for advanced inference.
    Types: int, float, percent, date, categorical, string.
    """
    types = {}
    for h in headers:
        col_vals = [safe_replace(safe_get(row, h, ""), ",", "") for row in data]
        non_empty = [v for v in col_vals if v not in ("", None, "NA", "N/A", "-")]

        # Try integer
        if all(re.fullmatch(r"\d+", v) or v == "" for v in non_empty):
            types[h] = "int"
            continue

        # Try float
        try:
            floats = [float(v) for v in non_empty if v.replace(".", "", 1).isdigit()]
        except Exception:
            floats = []
        if len(floats) == len(non_empty) and len(non_empty) > 0:
            types[h] = "float"
            continue

        # Try percent
        if all(re.fullmatch(r"\d+(\.\d+)?%", v) or v == "" for v in non_empty):
            types[h] = "percent"
            continue

        # Try date
        date_count = 0
        for v in non_empty:
            try:
                if is_date_like(v):
                    date_count += 1
                else:
                    dateutil.parser.parse(v, fuzzy=True)
                    date_count += 1
            except Exception:
                continue
        if len(non_empty) > 0 and date_count / len(non_empty) > 0.7:
            types[h] = "date"
            continue

        # Try categorical (few unique values)
        unique_vals = set(non_empty)
        if 0 < len(unique_vals) < 10:
            types[h] = "categorical"
            continue

        # Try numeric by numpy (robust for mixed int/float)
        try:
            arr = np.array([float(v) for v in non_empty])
            if arr.size > 0 and not np.isnan(arr).any():
                types[h] = "numeric"
                continue
        except Exception:
            pass

        # Default to string
        types[h] = "string"
    return types

# --- Heuristics & Entity Linking ---

def advanced_party_candidate_detection(headers, coordinator) -> Dict[str, List[Tuple[int, str]]]:
    """
    Use NER and context to better distinguish between candidate, party, and location columns.
    Returns a dict with lists of (index, entity) tuples for each type.
    """
    from ..Context_Integration.context_coordinator import ContextCoordinator
    coordinator = coordinator or ContextCoordinator()
    result = {"candidate": [], "party": [], "location": []}
    for idx, h in enumerate(headers):
        ents = coordinator.extract_entities(h)
        for ent, label in ents:
            if label in {"PERSON"}:
                result["candidate"].append((idx, ent))
            elif label in {"ORG", "NORP"}:
                result["party"].append((idx, ent))
            elif label in {"GPE", "LOC", "FAC"}:
                result["location"].append((idx, ent))
    return result

def extract_candidates_and_parties(headers: List[str], coordinator: "ContextCoordinator") -> Dict[str, Dict[str, List[str]]]:
    """
    Returns a dict: {party: {candidate: [ballot_types]}}
    Uses safe_append for robust list appending.
    """
    from ..Context_Integration.context_coordinator import ContextCoordinator
    coordinator = coordinator or ContextCoordinator()
    known_parties = PARTY_KEYWORDS
    ballot_types = BALLOT_TYPES

    candidate_party_map = {}
    for h in headers:
        m = re.match(r"(.+?)\s*\((.+?)\)\s*-\s*(.+)", h)
        if m:
            candidate, party, ballot_types = m.groups()
        else:
            m = re.match(r"(.+?)\s*-\s*(.+)", h)
            if m:
                candidate, ballot_types = m.groups()
                party = ""
            else:
                candidate, party, ballot_types = h, "", ""
        candidate = candidate.strip()
        party = party.strip()
        ballot_types = ballot_types.strip()
        if party:
            try:
                best_party, score = max(((p, coordinator.fuzzy_score(party, p)) for p in known_parties), key=lambda x: x[1])
            except Exception:
                best_party, score = party, 0
            if score > 80:
                party = best_party
        else:
            try:
                entities = coordinator.extract_entities(candidate)
            except Exception:
                entities = []
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
        if ballot_types and ballot_types not in candidate_party_map[party][candidate]:
            safe_append(candidate_party_map[party][candidate], ballot_types)
    return candidate_party_map

def entity_linking(
    header,
    known_entities,
    threshold=ENTITY_LINKING_THRESHOLD,
    return_score=False,
    allow_substring=True,
    allow_token_match=True,
) -> str:
    """
    Link header to known candidates/parties/entities for normalization.
    Uses robust normalization, fuzzy, substring, and token-based matching.
    Returns the best match if above threshold, else the original header.
    If return_score is True, returns (best_match, score).
    """   
    header_norm = normalize_header(header)
    best, best_score = None, 0

    # 1. Exact and substring match (case-insensitive, normalized)
    for ent in known_entities:
        ent_norm = normalize_header(ent)
        if allow_substring and (ent_norm in header_norm or header_norm in ent_norm):
            if return_score:
                return ent, 1.0
            return ent

    # 2. Token-based match
    if allow_token_match:
        header_tokens = set(header_norm.split())
        for ent in known_entities:
            ent_norm = normalize_header(ent)
            ent_tokens = set(ent_norm.split())
            if header_tokens and ent_tokens and (header_tokens <= ent_tokens or ent_tokens <= header_tokens):
                if return_score:
                    return ent, 0.95
                return ent

    # 3. Fuzzy match (difflib)
    for ent in known_entities:
        ent_norm = normalize_header(ent)
        s = difflib.SequenceMatcher(None, header_norm, ent_norm).ratio()
        if s > best_score:
            best, best_score = ent, s

    if best_score >= threshold:
        if return_score:
            return best, best_score
        return best

    if return_score:
        return header, best_score
    return header

# --- Pattern/Selector Discovery & Logging ---

def find_tables_with_headings(page, dom_segments=None, heading_tags=None, include_section_context=True) -> List[Tuple[str, Any]]:
    """
    Finds all tables on the page and pairs each with its nearest heading or ARIA landmark.
    Returns a list of (heading, table_locator) tuples.
    """
    if heading_tags is None:
        heading_tags = HEADING_TAGS + EXTRA_HEADING_TAGS

    results = []

    def extract_text_from_html(html: str) -> str:
        """
        Extracts visible text from an HTML string.
        - Handles tags like <span>, <div>, <a>, <li>, <b>, <strong>, <em>, <u>, <i>, <p>, <br>, <th>, <td>, <button>, <label>, <h1>-<h6>.
        - Strips all tags and returns the concatenated text.
        - Handles nested tags and ignores script/style.
        """
        html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", "", html)
        text = re.sub(r"\s+", " ", text)
        return safe_strip(text)

    if dom_segments:
        tables = [seg for seg in dom_segments if safe_get(seg, "tag") == "table"]
        for i, table_seg in enumerate(tables):
            heading = None
            section_context = None
            idx = safe_get(table_seg, "_idx")
            # 1. Walk backwards for nearest heading
            if idx is not None:
                for j in range(idx-1, -1, -1):
                    tag = safe_get(dom_segments[j], "tag", "")
                    if tag in heading_tags:
                        heading_html = safe_get(dom_segments[j], "html", "")
                        heading = extract_text_from_html(heading_html)
                        break
            # 2. If not found, walk up for ARIA landmarks or section/fieldset
            if not heading and idx is not None:
                parent_idx = safe_get(table_seg, "_parent_idx")
                visited = set()
                while parent_idx is not None and parent_idx not in visited:
                    visited.add(parent_idx)
                    parent_seg = dom_segments[parent_idx]
                    tag = safe_get(parent_seg, "tag", "")
                    attrs = safe_get(parent_seg, "attrs", {})
                    aria_label = safe_get(attrs, "aria-label") or safe_get(attrs, "aria-labelledby")
                    role = safe_get(attrs, "role", "")
                    if role in ("region", "complementary", "main", "navigation", "search") or aria_label:
                        section_context = aria_label or role
                        break
                    if tag in ("section", "fieldset"):
                        for k in range(parent_idx+1, len(dom_segments)):
                            if safe_get(dom_segments[k], "_parent_idx") == parent_idx:
                                child_tag = safe_get(dom_segments[k], "tag", "")
                                if child_tag == "legend":
                                    heading = extract_text_from_html(safe_get(dom_segments[k], "html", ""))
                                    break
                                if child_tag in heading_tags:
                                    heading = extract_text_from_html(safe_get(dom_segments[k], "html", ""))
                                    break
                        if heading:
                            break
                        section_context = tag
                        break
                    parent_idx = safe_get(parent_seg, "_parent_idx")
            if not heading:
                heading = f"Precinct {i+1}"
            if include_section_context and section_context:
                heading = f"{section_context}: {heading}"
            table_locator = safe_locator(page, "table")
            table_nth = safe_nth(table_locator, i, logger) if table_locator else None
            if table_nth is not None:
                results.append((heading, table_nth))
    else:
        tables = safe_locator(page, "table")
        for i in range(safe_count(tables, logger)):
            table = safe_nth(tables, i, logger)
            if table is None:
                continue
            heading = None
            section_context = None
            try:
                parent = table
                for _ in range(5):
                    parent = safe_locator(parent, "xpath=..", logger)
                    attrs = safe_evaluate(parent, "el => ({'role': el.getAttribute('role'), 'aria-label': el.getAttribute('aria-label'), 'aria-labelledby': el.getAttribute('aria-labelledby'), 'tag': el.tagName.toLowerCase()})", logger)
                    if safe_get(attrs, "role") in ("region", "complementary", "main", "navigation", "search") or safe_get(attrs, "aria-label"):
                        section_context = safe_get(attrs, "aria-label") or safe_get(attrs, "role")
                        break
                    if safe_get(attrs, "tag") in ("section", "fieldset"):
                        legend = safe_locator(parent, "legend", logger)
                        if safe_count(legend, logger) > 0:
                            heading = safe_strip(safe_inner_text(safe_nth(legend, 0, logger), logger))
                            break
                        for tag in heading_tags:
                            h = safe_locator(parent, tag, logger)
                            if safe_count(h, logger) > 0:
                                heading = safe_strip(safe_inner_text(safe_nth(h, 0, logger), logger))
                                break
                        if heading:
                            break
                        section_context = safe_get(attrs, "tag")
                        break
                if not heading:
                    header_locator = safe_locator(table, "xpath=preceding-sibling::*[self::h1 or self::h2 or self::h3 or self::h4 or self::h5 or self::h6][1]", logger)
                    if safe_count(header_locator, logger) > 0:
                        heading = safe_strip(safe_inner_text(safe_nth(header_locator, 0, logger), logger))
            except Exception:
                pass
            if not heading:
                heading = f"Section {i+1}"
            if include_section_context and section_context:
                heading = f"{section_context}: {heading}"
            results.append((heading, table))
    return results

def discover_container_selectors(page, extra_keywords=None, min_row_count=2):
    """
    Dynamically discovers container selectors (divs, sections, etc.) with relevant keywords or tabular structure.
    Returns a list of selectors, ranked by likelihood.
    Uses safe_* utilities for all DOM/string operations.
    """
    if extra_keywords is None:
        extra_keywords = CONTAINER_EXTRA_KEYWORDS
    class_scores = {}

    all_divs = safe_locator(page, "div")
    for i in range(safe_count(all_divs)):
        div = safe_nth(all_divs, i)
        if div is None:
            continue
        cls = safe_get_attribute(div, "class") or ""
        id_ = safe_get_attribute(div, "id") or ""
        text = safe_lower(safe_strip(safe_inner_text(div)))
        score = 0

        # Score based on keywords in class/id/text
        for kw in extra_keywords:
            kw_l = safe_lower(kw)
            if kw_l in safe_lower(cls) or kw_l in safe_lower(id_) or kw_l in text:
                score += 2
        # Score based on number of children (tabular structure)
        children = safe_locator(div, "> *")
        if safe_count(children) >= min_row_count:
            score += 2
        # Score based on presence of numbers (votes)
        if any(char.isdigit() for char in text):
            score += 1

        # Build selector and store score
        if cls:
            sel = "div." + ".".join(safe_split(cls))
            class_scores[sel] = class_scores.get(sel, 0) + score
        if id_:
            sel = f"div#{id_}"
            class_scores[sel] = class_scores.get(sel, 0) + score

    # Return selectors sorted by score
    sorted_selectors = [sel for sel, _ in sorted(class_scores.items(), key=lambda x: -x[1])]
    # Add some generic selectors as fallback
    sorted_selectors += CONTAINER_FALLBACK_SELECTORS
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
    with open(log_path, "ab") as f:
        f.write(orjson.dumps(entry) + b"\n")

def review_dom_patterns(log_path=None):
    """
    CLI to review, approve, or delete learned DOM patterns.
    Uses safe_get for robust dict access.
    """
    if log_path is None:
        log_path = get_safe_log_path()
    if not os.path.exists(log_path):
        _emit("warning", "extractor", "No learned DOM patterns found.")
        return

    with open(log_path, "rb") as f:
        entries = [orjson.loads(line) for line in f if line.strip()]

    for idx, entry in enumerate(entries):
        _emit("info", "extractor", f"[{idx}] Selector preview", selector=safe_get(entry, "selector"))
        example_html = safe_get(entry, "example_html", "")
        _emit("info", "extractor", "Example HTML (truncated)", preview=example_html[:200] + "...")
        _emit("info", "extractor", "Context", context=safe_get(entry, "context"))
        _emit("info", "extractor", "-" * 40)

    while True:
        cmd = input("\nEnter entry number to approve/delete, or 'q' to quit: ")
        cmd = cmd.strip()
        if cmd.lower() == "q":
            break
        if cmd.isdigit():
            idx = int(cmd)
            if 0 <= idx < len(entries):
                action = input("Approve (a) or Delete (d) this entry? [a/d]: ").strip().lower()
                if action == "d":
                    entries.pop(idx)
                    _emit("warning", "extractor", "Entry deleted.")
                elif action == "a":
                    entries[idx]["approved"] = True
                    _emit("info", "extractor", "Entry approved.")
                else:
                    _emit("warning", "extractor", "Unknown action.")
            else:
                _emit("warning", "extractor", "Invalid entry number.")
        # Save changes
        with open(log_path, "wb") as f:
            for entry in entries:
                f.write(orjson.dumps(entry) + b"\n")
        _emit("info", "extractor", "Changes saved.")

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
    with open(log_path, "wb") as f:
        for p in patterns:
            f.write(orjson.dumps(p) + b"\n")

# --- Structure Detection & Classification ---

def find_tables_with_panel_headings(
    page,
    panel_selectors=None,
    header_selectors=None,
    table_selectors=None,
    context_library=None
):
    """
    Finds all tables inside panels, associates each with the panel's heading.
    Returns a list of (district_name, table_locator) tuples.
    Uses safe_* utilities for robustness and canonical panel/header tags.
    Dynamically extends selectors using librarian/context_library and librarian utilities.
    """
    # Use canonical and user-provided selectors, and extend with librarian/context_library
    if panel_selectors is None:
        panel_selectors = list(PANEL_TAGS)
    if header_selectors is None:
        header_selectors = [
            "h1.panel-header span.ng-star-inserted",
            "h1.panel-header"
        ] + list(EXTRA_HEADING_TAGS)
    if table_selectors is None:
        table_selectors = ["table.contest-table"] + list(CONTAINER_FALLBACK_SELECTORS)

    # Use librarian utilities to extend selectors
    panel_selectors = extend_panel_tags(panel_selectors)
    header_selectors = extend_heading_tags(header_selectors)

    # Optionally extend selectors from context_library/librarian
    if context_library:
        panel_selectors = list(set(panel_selectors) | set(context_library.get("panel_tags", [])))
        header_selectors = list(set(header_selectors) | set(context_library.get("heading_tags", [])))
        table_selectors = list(set(table_selectors) | set(context_library.get("table_tags", [])))

    # Try all panel selectors in order
    results = []
    found_panels = False
    for panel_selector in panel_selectors:
        panels = safe_locator(page, panel_selector)
        if safe_count(panels) == 0:
            continue
        found_panels = True
        for i in range(safe_count(panels)):
            panel = safe_nth(panels, i)
            if panel is None:
                continue
            district_name = ""
            found_header = False
            for hsel in header_selectors:
                header_span = safe_locator(panel, hsel)
                if safe_count(header_span) > 0:
                    district_name = safe_strip(safe_inner_text(safe_nth(header_span, 0)))
                    found_header = True
                    break
            if not found_header and context_library:
                log_unknown_tag(panel_selector, context_library)
            # Try all table selectors in order
            found_table = False
            for tsel in table_selectors:
                table = safe_locator(panel, tsel)
                if safe_count(table) > 0:
                    results.append((district_name, safe_nth(table, 0)))
                    found_table = True
                    break
            if not found_table and context_library:
                log_unknown_tag("table", context_library)
        if found_panels:
            break  # Only use the first panel selector that matches
    return results

def find_tables_with_section_headings(
    page,
    heading_tags=None,
    extra_heading_selectors=None,
    max_depth=6
):
    """
    For each table on the page, walk up the DOM to find the nearest section heading.
    Returns a list of (section_name, table_locator) tuples.
    Uses safe_* utilities for robustness and canonical heading tags.
    Dynamically extends heading tags using librarian utilities.
    """
    if heading_tags is None:
        heading_tags = list(HEADING_TAGS)
    if extra_heading_selectors is None:
        extra_heading_selectors = list(EXTRA_HEADING_TAGS)

    # Use librarian utility to extend heading tags/selectors
    heading_tags = extend_heading_tags(heading_tags)
    extra_heading_selectors = extend_heading_tags(extra_heading_selectors)

    results = []
    tables = safe_locator(page, "table")
    for i in range(safe_count(tables)):
        table = safe_nth(tables, i)
        if table is None:
            continue
        section_name = None

        # 1. Walk up DOM for heading tags
        parent = table
        for _ in range(max_depth):
            parent = safe_locator(parent, "xpath=..")
            # Try heading tags
            for tag in heading_tags:
                headings = safe_locator(parent, tag)
                if safe_count(headings) > 0:
                    section_name = safe_strip(safe_inner_text(safe_nth(headings, 0)))
                    if section_name:
                        break
            if section_name:
                break
            # Try extra selectors
            for sel in extra_heading_selectors:
                extra = safe_locator(parent, sel)
                if safe_count(extra) > 0:
                    section_name = safe_strip(safe_inner_text(safe_nth(extra, 0)))
                    if section_name:
                        break
            if section_name:
                break
            # Try ARIA label
            try:
                aria_label = safe_evaluate(parent, "el => el.getAttribute('aria-label')")
                if aria_label:
                    section_name = safe_strip(aria_label)
                    break
            except Exception:
                pass

        # 2. Fallback: preceding sibling heading
        if not section_name:
            for tag in heading_tags:
                sibling = safe_locator(table, f"xpath=preceding-sibling::{tag}[1]")
                if safe_count(sibling) > 0:
                    section_name = safe_strip(safe_inner_text(safe_nth(sibling, 0)))
                    if section_name:
                        break

        # 3. Fallback: use table index
        if not section_name:
            section_name = f"Section {i+1}"

        results.append((section_name, table))
    return results

def is_candidate_major_row(headers, data, coordinator, context):
    """
    Detect if table is candidate-major (first column is candidate, rest are vote types or totals).
    Uses spaCy NER, canonical keywords, abbreviations, and context for robust detection.
    """
    from ..Context_Integration.context_coordinator import ContextCoordinator
    coordinator = coordinator or ContextCoordinator()
    headers = headers or []
    data = data or []
    if not headers or not data:
        page = safe_get(context, "page")
        headers, data = robust_table_extraction(page, context)
        if not headers or not data:
            logger.error("[TABLE BUILDER] No data could be extracted from the page.")
            return False

    first_col = safe_get(headers, 0, "")
    first_col_norm = normalize_text(first_col)
    first_row_val = safe_get(safe_get(data, 0, {}), first_col, "")
    entities = coordinator.extract_entities(first_col)
    data_entities = coordinator.extract_entities(first_row_val)
    is_person_header = any(label == "PERSON" for _, label in entities + data_entities)
    is_candidate_keyword = (
        first_col_norm in (normalize_text(k) for k in CANDIDATE_KEYWORDS)
        or first_col_norm in (normalize_text(k) for k in CONTEST_KEYWORDS)
    )
    if is_person_header or is_candidate_keyword:
        non_candidate_headers = headers[1:]
        ballot_type_hits = 0
        for h in non_candidate_headers:
            h_norm = normalize_text(h)
            if (
                h_norm in (normalize_text(bt) for bt in BALLOT_TYPES + BALLOT_TYPES_SORT_ORDER)
                or h_norm in (normalize_text(tk) for tk in TOTAL_KEYWORDS)
                or h_norm in (normalize_text(pk) for pk in PARTY_KEYWORDS)
                or h_norm in (normalize_text(lk) for lk in LOCATION_KEYWORDS)
                or h_norm in (normalize_text(abbr) for abbr in LOCATION_ABBREVIATIONS)
                or h_norm in (normalize_text(k) for k in NLP_SKIP_PHRASES)
            ):
                ballot_type_hits += 1
        if non_candidate_headers and ballot_type_hits / len(non_candidate_headers) >= 0.5:
            return True
    return False

def is_candidate_major_col(headers, data, context):
    """
    Detect if table is candidate-major by columns (first row is vote type, columns are candidates).
    Uses spaCy NER, canonical keywords, and context for robust detection.
    """
    from ..Context_Integration.context_coordinator import ContextCoordinator
    coordinator = ContextCoordinator()
    headers = headers or []
    data = data or []
    if not headers or not data:
        page = safe_get(context, "page")
        headers, data = robust_table_extraction(page, context)
        if not headers or not data:
            return False
    no_location_or_party = all(
        normalize_text(h) not in LOCATION_KEYWORDS
        and normalize_text(h) not in PARTY_KEYWORDS
        for h in headers
    )
    has_candidate = False
    for h in headers:
        ents = coordinator.extract_entities(h)
        if any(label == "PERSON" for _, label in ents):
            has_candidate = True
            break
        if normalize_text(h) in (normalize_text(k) for k in CANDIDATE_KEYWORDS):
            has_candidate = True
            break
    return no_location_or_party and has_candidate

def is_precinct_major(headers, coordinator):
    """
    Detect if table is precinct-major (first column is a location/precinct/district).
    Uses canonical keywords and context_coordinator patterns.
    """
    from ..Context_Integration.context_coordinator import ContextCoordinator
    coordinator = coordinator or ContextCoordinator()
    headers = headers or []
    if not headers:
        return False
    location_patterns = set(
        normalize_text(p) for p in coordinator.library.get("location_patterns", LOCATION_KEYWORDS)
    )
    first_col = normalize_text(headers[0])
    ents = coordinator.extract_entities(headers[0])
    is_location = any(label in {"GPE", "LOC", "FAC"} for _, label in ents)
    return first_col in location_patterns or is_location

def is_flat_candidate_table(headers, coordinator=None):
    """
    Detect if table is a flat candidate table (only candidate and total columns, no locations).
    Uses canonical keywords and NER.
    """
    from ..Context_Integration.context_coordinator import ContextCoordinator
    coordinator = coordinator or ContextCoordinator()
    headers = headers or []
    if not headers:
        logger.error("[red][ERROR] No headers extracted from table. Skipping this table.[/red]")
        return False
    first_col = normalize_text(headers[0])
    ents = coordinator.extract_entities(headers[0])
    is_candidate = (
        first_col in (normalize_text(k) for k in CANDIDATE_KEYWORDS)
        or any(label == "PERSON" for _, label in ents)
    )
    all_valid = all(
        any(
            kw in normalize_text(h)
            for kw in list(TOTAL_KEYWORDS) + list(CANDIDATE_KEYWORDS)
        ) or any(label == "PERSON" for _, label in coordinator.extract_entities(h))
        for h in headers
    )
    return is_candidate and all_valid

def is_single_row_summary(data):
    """
    Detect if table is a single-row summary.
    """
    return len(data) == 1

def is_candidate_footer(data, coordinator=None):
    """
    Detect if last row contains candidate or misc footer keywords.
    Uses canonical keywords and NER.
    """
    from ..Context_Integration.context_coordinator import ContextCoordinator
    coordinator = coordinator or ContextCoordinator()
    if not data or not data[-1]:
        return False
    last_row = data[-1]
    for v in safe_values(last_row):
        v_norm = normalize_text(str(v))
        ents = coordinator.extract_entities(str(v))
        if any(
            kw in v_norm for kw in list(CANDIDATE_KEYWORDS) + list(MISC_FOOTER_KEYWORDS)
        ) or any(label == "PERSON" for _, label in ents):
            return True
    return False

def detect_wide_vs_long(headers, data):
    """
    Detect if table is wide or long format.
    Uses heuristics.
    """
    if len(headers) > 10 and len(data) < 10:
        return "wide"
    if len(headers) <= 5 and len(data) > 10:
        return "long"
    return "ambiguous"

def classify_ambiguous_tables(headers, data, coordinator):
    """
    Use ML, NER, rules, and data content to classify ambiguous structures.
    Returns: "precinct-major", "candidate-major", "candidate-major-with-party", or "ambiguous".
    """
    from ..Context_Integration.context_coordinator import ContextCoordinator
    coordinator = coordinator or ContextCoordinator()
    col_types = advanced_party_candidate_detection(headers, coordinator)
    has_candidate = bool(col_types["candidate"])
    has_location = bool(col_types["location"])
    has_party = bool(col_types["party"])

    first_col = headers[0] if headers else ""
    first_col_values = [row.get(first_col, "") for row in data if first_col]
    location_like_count = 0
    candidate_like_count = 0
    party_like_count = 0
    for val in first_col_values:
        ents = coordinator.extract_entities(val)
        if any(label in {"GPE", "LOC", "FAC"} for _, label in ents):
            location_like_count += 1
        if any(label == "PERSON" for _, label in ents):
            candidate_like_count += 1
        if any(label in {"ORG", "NORP"} for _, label in ents):
            party_like_count += 1
        norm_val = normalize_text(val)
        if norm_val in (normalize_text(k) for k in LOCATION_KEYWORDS):
            location_like_count += 1
        if norm_val in (normalize_text(k) for k in CANDIDATE_KEYWORDS):
            candidate_like_count += 1
        if norm_val in (normalize_text(k) for k in PARTY_KEYWORDS):
            party_like_count += 1

    n_rows = len(first_col_values)
    location_ratio = location_like_count / n_rows if n_rows else 0
    candidate_ratio = candidate_like_count / n_rows if n_rows else 0
    party_ratio = party_like_count / n_rows if n_rows else 0

    if location_ratio > 0.5:
        return "precinct-major"
    if candidate_ratio > 0.5:
        if party_ratio > 0.3:
            return "candidate-major-with-party"
        return "candidate-major"
    if has_candidate and has_location:
        return "precinct-major"
    elif has_candidate and not has_location and has_party:
        return "candidate-major-with-party"
    elif has_candidate:
        return "candidate-major"
    elif has_location:
        return "precinct-major"
    else:
        return "ambiguous"

# ===================================================================
# END OF FILE
# ===================================================================