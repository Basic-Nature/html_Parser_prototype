# ===================================================================
# table_builder.py
# Election Data Cleaner - Table Extraction and Cleaning Orchestrator
# Centralizes user feedback, ML learning, and structure confirmation.
# ===================================================================

import os
import json
from rich.table import Table
from typing import List, Dict, Tuple, Any, Optional, TYPE_CHECKING
from .logger_instance import logger
from ..utils.shared_logger import rprint
from ..config import BASE_DIR

LOG_PARENT_DIR = os.path.join(BASE_DIR, "logs")

from .table_core import (
    extract_table_data,
    extract_rows_and_headers_from_dom,
    extract_with_patterns,
    harmonize_headers_and_data,
    detect_table_structure,
    nlp_entity_annotate_table,
    verify_table_structure,
    interactive_feedback_loop,
    pivot_to_wide_format,
    table_signature,
    cache_table_structure,
)

if TYPE_CHECKING:
    from ..Context_Integration.context_coordinator import ContextCoordinator

# ===================================================================
# MAIN TABLE BUILDING PIPELINE
# ===================================================================

def build_dynamic_table(
    domain: str,
    headers: List[str],
    data: List[Dict[str, Any]],
    coordinator: "ContextCoordinator",
    context: dict = None,
    max_feedback_loops: int = 3,
    learning_mode: bool = True,
    confirm_table_structure_callback=None,
    pivot_to_wide: bool = True,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Orchestrates robust, multi-source, entity-aware table extraction and harmonization.
    - Always attempts DOM extraction first.
    - Uses pattern-based and NLP-based entity detection to annotate and enrich data.
    - Harmonizes and verifies table structure.
    - Handles user/ML feedback and learning.
    - Only transforms/pivots to wide format if explicitly requested.
    """
    if context is None:
        context = {}

    # 1. DOM Extraction (always first)
    page = context.get("page")
    dom_headers, dom_data = [], []
    try:
        if page and page.locator("table").count() > 0:
            dom_headers, dom_data = extract_table_data(page.locator("table").first)
            logger.info(f"[TABLE_BUILDER] DOM table extraction: {len(dom_headers)} headers, {len(dom_data)} rows.")
    except Exception as e:
        logger.warning(f"[TABLE_BUILDER] DOM table extraction failed: {e}")

    # 2. Pattern-based Extraction (divs, lists, etc.)
    pattern_headers, pattern_data = [], []
    try:
        pattern_headers, pattern_data = extract_rows_and_headers_from_dom(page, coordinator=coordinator)
        logger.info(f"[TABLE_BUILDER] Pattern-based extraction: {len(pattern_headers)} headers, {len(pattern_data)} rows.")
    except Exception as e:
        logger.warning(f"[TABLE_BUILDER] Pattern-based extraction failed: {e}")

    # 3. Pattern-based DOM heuristics (custom patterns)
    custom_headers, custom_data = [], []
    try:
        custom_headers, custom_data = extract_with_patterns(page, context)
        logger.info(f"[TABLE_BUILDER] Custom pattern extraction: {len(custom_headers)} headers, {len(custom_data)} rows.")
    except Exception as e:
        logger.warning(f"[TABLE_BUILDER] Custom pattern extraction failed: {e}")

    # 4. NLP Entity Annotation (always used, not fallback)
    # Combine all extracted data for annotation
    all_headers = list({h for h in (dom_headers + pattern_headers + custom_headers) if h})
    all_data = (dom_data or []) + (pattern_data or []) + (custom_data or [])
    all_headers, all_data = harmonize_headers_and_data(all_headers, all_data)
    logger.info(f"[TABLE_BUILDER] Combined extraction: {len(all_headers)} headers, {len(all_data)} rows.")

    # NLP entity annotation: enriches data with detected people, locations, ballot types, numbers
    try:
        annotated_headers, annotated_data, entity_info = nlp_entity_annotate_table(
            all_headers, all_data, context=context, coordinator=coordinator
        )
        logger.info(f"[TABLE_BUILDER] NLP entity annotation complete. Entities: {entity_info}")
    except Exception as e:
        logger.warning(f"[TABLE_BUILDER] NLP entity annotation failed: {e}")
        annotated_headers, annotated_data = all_headers, all_data
        entity_info = {}

    # 5. Harmonize headers/data after annotation
    headers, data = harmonize_headers_and_data(annotated_headers, annotated_data)
    logger.info(f"[TABLE_BUILDER] Harmonized headers: {headers}")
    logger.info(f"[TABLE_BUILDER] Harmonized sample row: {data[0] if data else 'NO DATA'}")

    # 6. Structure Analysis (annotation only, no transformation)
    try:
        structure_info = detect_table_structure(headers, data, coordinator, entity_info=entity_info)
        logger.info(f"[TABLE_BUILDER] Detected table structure: {structure_info}")
    except Exception as e:
        logger.warning(f"[TABLE_BUILDER] Structure detection failed: {e}")
        structure_info = {"type": "ambiguous", "verified": False}

    # 7. Verification (required columns/entities)
    try:
        verified, missing = verify_table_structure(headers, data, entity_info, coordinator, context)
        if not verified:
            logger.warning(f"[TABLE_BUILDER] Table verification failed. Missing: {missing}")
            headers, data, structure_info = interactive_feedback_loop(headers, data, structure_info)
        else:
            logger.info("[TABLE_BUILDER] Table verification passed.")
    except Exception as e:
        logger.warning(f"[TABLE_BUILDER] Table verification error: {e}")
        headers, data, structure_info = interactive_feedback_loop(headers, data, structure_info)

    # 8. User/ML confirmation and learning (if enabled)
    if learning_mode:
        contest_title = context.get("contest_title") or "Unknown Contest"
        headers, data = prompt_user_to_confirm_table_structure(
            headers, data, domain, contest_title, coordinator
        )

    # 9. Only transform/pivot to wide format if requested
    if pivot_to_wide:
        try:
            wide_headers, wide_data = pivot_to_wide_format(headers, data, entity_info, coordinator, context)
            logger.info(f"[TABLE_BUILDER] Pivoted to wide format: {len(wide_headers)} headers, {len(wide_data)} rows.")
            return wide_headers, wide_data
        except Exception as e:
            logger.warning(f"[TABLE_BUILDER] Pivot to wide format failed: {e}")
            # Return harmonized, annotated data as fallback

    return headers, data

# ===================================================================
# USER FEEDBACK, CONFIRMATION, AND LEARNING
# ===================================================================

def prompt_user_to_confirm_table_structure(headers, data, domain, contest_title, coordinator):
    """
    Interactive CLI for user to confirm, correct, or reject table structure.
    Handles ML/NLP suggestions, logs user feedback for learning.
    """
    import copy

    should_log = True
    columns_changed = False
    new_headers = copy.deepcopy(headers)
    denied_structures_path = os.path.join(LOG_PARENT_DIR, "denied_table_structures.json")
    denied_structures = {}
    denied_structures_dir = os.path.dirname(denied_structures_path)
    os.makedirs(denied_structures_dir, exist_ok=True)
    if os.path.exists(denied_structures_path):
        with open(denied_structures_path, "r", encoding="utf-8") as f:
            denied_structures = json.load(f)
    sig = f"{domain}:{table_signature(headers)}"
    denied_count = denied_structures.get(sig, 0)

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

    # If ML confidence is low and NLP suggests better header names, auto-apply those suggestions
    if avg_score < 0.7 and any(ent and ent != h for h, ent, label in nlp_suggestions):
        logger.info("[TABLE BUILDER] ML confidence low and NLP suggests better header names. Auto-applying suggestions.")
        for idx, (h, ent, label) in enumerate(nlp_suggestions):
            if ent and ent != h:
                new_headers[idx] = ent
        new_headers, data = harmonize_headers_and_data(new_headers, data)
        ml_scores = [coordinator.score_header(h, {"contest_title": contest_title}) for h in new_headers]
        avg_score = sum(ml_scores) / len(ml_scores) if ml_scores else 0

    # Multiple structure candidates (if available)
    structure_candidates = [new_headers]
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
        N = min(5, len(data))
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

# ===================================================================
# OPTIONAL: BATCH OPERATIONS AND SUGGESTIONS
# ===================================================================

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

def dynamic_confidence_threshold(history, default=0.93):
    """
    Adjust threshold for auto-accepting structures based on past accuracy.
    """
    if not history:
        return default
    correct = sum(1 for h in history[-5:] if h["accepted"])
    if correct >= 4:
        return min(0.98, default + 0.02)
    elif correct <= 2:
        return max(0.85, default - 0.05)
    return default

# ===================================================================
# END OF FILE
# ===================================================================