# ============================================================
# 🗳️ Smart Elections: JSON Handler for Dynamic State Parsing
# ============================================================

# This script processes structured election result JSON files.
# It dynamically detects contests, candidates, vote methods, and precinct-level vote tallies,
# formats the extracted data into a clean matrix, and prepares it for export into CSV.
# It supports custom state handlers or a generic fallback logic if no handler is defined.

import csv
import json
import os
import pandas as pd
import re
from collections import defaultdict
from state_router import get_handler_from_context, resolve_state_handler
from utils.shared_logger import logger, rprint
from utils.output_utils import get_output_path, format_timestamp

def parse(page, html_context):
    # Respect early skip signal from calling context
    if html_context.get("skip_format") or html_context.get("manual_skip"):
        return None, None, None, {"skipped": True}
    # Step 1: Detect available JSON files in 'input' folder
    input_folder = "input"
    files = [f for f in os.listdir(input_folder) if f.endswith(".json")]

    if not files:
        rprint("[red][ERROR] No JSON files found in 'input' folder.[/red]")
        logger.error("No JSON files found in 'input' folder.")
        return None

    files.sort(key=lambda f: os.path.getmtime(os.path.join(input_folder, f)), reverse=True)
    selected_file = html_context.get("filename", "").strip()
    matched_file = selected_file if selected_file in files else None

    if matched_file:
        rprint(f"\n[green]Matched downloaded file:[/green] {matched_file}")
        use_file = input(f"[PROMPT] Proceed with parsing {matched_file}? (y/n): ").strip().lower()
        if use_file != "y":
            rprint("[yellow]Okay, choose a different file from the list.[/yellow]")
            matched_file = None
    if not matched_file:
        rprint("\n[yellow]Available JSON files in 'input' folder:[/yellow]")
        for i, f in enumerate(files):
            rprint(f"  [bold cyan][{i}][/bold cyan] {f}")
        idx = input("\n[PROMPT] Enter file index or press Enter to cancel: ").strip()
        if not idx.isdigit():
            rprint("[yellow]No file selected. Skipping JSON parsing.[/yellow]")
            return None
        try:
            selected_idx = int(idx)
            matched_file = files[selected_idx]
        except (IndexError, ValueError):
            rprint("[red]Invalid index. Skipping JSON parsing.[/red]")
            return None
          
    # Step 2: Load JSON content
    file_path = os.path.join(input_folder, matched_file)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract reported timestamp if available
    report_time = data.get("reportGeneratedAt") or data.get("timestamp") or data.get("lastUpdated")
    if report_time:
        html_context["report_time"] = report_time

    # Step 3: Try resolving a state-specific handler
    if not html_context.get("state") or html_context["state"].lower() == "unknown":
        handler = resolve_state_handler(html_context.get("source_url", ""))
        if handler:
            html_context["state"] = handler.__name__.split(".")[-1].upper()
    else:
        handler = get_handler_from_context(html_context)
        if not handler:
            handler = resolve_state_handler(html_context.get("source_url", ""))

    if handler and hasattr(handler, "parse_json"):
        logger.info(f"Using custom state handler '{handler.__name__}' for JSON parsing.")
        return handler.parse_json(data, html_context)
    else:
        logger.warning("[Router] No handler resolved or handler lacks 'parse_json'.")

    # Step 4: Fallback logic for generic JSON structures
    logger.info("[State Router] No custom handler found. Using fallback parser.")
    ballot_items = data.get("results", {}).get("ballotItems", [])
    if not ballot_items:
        rprint("[red][ERROR] No contests found in JSON structure.[/red]")
        return None

    # Step 5: Detect available contests and prompt user
    contests = sorted({item.get("name", "").strip() for item in ballot_items if item.get("name")})
    rprint("\n[yellow]Available contests:[/yellow]")
    for i, name in enumerate(contests, 1):
        rprint(f" [bold cyan]{i:2d}[/bold cyan]. {name}")

    rprint("\nEnter the contest name (exactly as shown), or type its number:")
    user_input = input("> ").strip()

    if user_input.isdigit():
        idx = int(user_input)
        try:
            target_contest = contests[idx - 1]
        except IndexError:
            raise ValueError("Invalid contest number.")
    else:
        if user_input not in contests:
            rprint(f"[red][ERROR] Contest name '{user_input}' not found.[/red]")
            raise ValueError("Contest name not found.")
        target_contest = user_input

    rprint(f"\n🔍 [bold]Parsing contest:[/bold] [green]{target_contest}[/green]\n")

    # Step 6: Normalize candidate-party labels and vote method groupings
    group_rename = {
        "Election Day": "Election Day",
        "Early Voting": "Early",
        "Absentee Mail": "Mail-In",
        "Provisional": "Provisional"
    }
    vote_methods = list(group_rename.values())

    raw_candidates = {}
    for item in ballot_items:
        if item["name"].strip() != target_contest:
            continue
        for opt in item.get("ballotOptions", []):
            raw = opt.get("name", "").strip()
            party = opt.get("politicalParty", "Unknown")
            label = f"{raw} ({party})"
            raw_candidates[raw] = label

    normalization_map = raw_candidates.copy()
    candidate_order = sorted(set(normalization_map.values()))

    # Step 7: Build nested results: precinct → candidate → vote method
    results_nested = defaultdict(lambda: defaultdict(dict))
    for item in ballot_items:
        if item["name"].strip() != target_contest:
            continue
        for opt in item.get("ballotOptions", []):
            raw_label = opt.get("name", "").strip()
            for precinct in opt.get("precinctResults", []):
                p = precinct.get("name", "").strip()
                if p == "-" or not p:
                    p = pd.NA
                results_nested[p][raw_label]["Total"] = precinct.get("voteCount")
                for grp in precinct.get("groupResults", []):
                    g = grp.get("groupName", "").strip()
                    norm_g = group_rename.get(g, g)
                    results_nested[p][raw_label][norm_g] = grp.get("voteCount")

    # Step 8: Flatten nested results into a DataFrame
    rows = {}
    cols = set()
    for precinct, cands in results_nested.items():
        row = {}
        for raw_label, methods in cands.items():
            for method, count in methods.items():
                cols.add((raw_label, method))
                row[(raw_label, method)] = count
        rows[precinct] = row

    df_raw = pd.DataFrame.from_dict(rows, orient="index")
    index_label = "Precinct"
    # Look for common alternatives in metadata for better labeling
    default_keywords = os.getenv("JSON_HEADER_KEYWORDS", "ward,district,town,precinct").split(",")
    handler_keywords = default_keywords
    if handler and hasattr(handler, "header_keywords"):
        handler_keywords = getattr(handler, "header_keywords")
    index_label = "Precinct"
    context_source = json.dumps(data).lower()
    for label in handler_keywords:
        if label in context_source:
            index_label = label.capitalize()
            logger.info(f"[Label Detection] Using '{index_label}' as the row label based on JSON context.")
            break
    possible_labels = handler_keywords
    context_source = json.dumps(data).lower()
    for label in possible_labels:
        if label in context_source:
            index_label = label.capitalize()
            logger.info(f"[Label Detection] Using '{index_label}' as the row label based on JSON context.")
            break
    df_raw.index.name = index_label
    df_raw = df_raw.reindex(columns=pd.MultiIndex.from_tuples(sorted(cols)))

    # Step 9: Normalize all candidate-method columns, ensuring blanks are filled with NA
    for raw_label in normalization_map:
        for method in vote_methods + ["Total"]:
            col = (raw_label, method)
            if col not in df_raw.columns:
                df_raw[col] = pd.NA

    df_clean = pd.DataFrame(index=df_raw.index)

    # Log missing data counts per row and column
    if os.getenv("OUTPUT_NULL_MAP", "false").lower() == "true":
        df_raw.isna().to_csv("output/null_map.csv")
        logger.info("[Output] Missing data map exported to output/null_map.csv")
    missing_summary = df_raw.isna().sum()
    if missing_summary.any():
        if os.getenv("LOG_WARNINGS", "true").lower() == "true":
            logger.warning("[Missing Data] Some values are missing in the parsed DataFrame:")
            for col, count in missing_summary.items():
                if count > 0:
                    logger.warning(f" - Column {col}: {count} missing value(s)")
    for norm_label in candidate_order:
        for method in vote_methods + ["Total"]:
            cols_to_sum = [
                col for col in df_raw.columns
                if normalization_map.get(col[0], "Other") == norm_label and col[1] == method
            ]
            if cols_to_sum:
                df_clean[(norm_label, method)] = df_raw[cols_to_sum].sum(axis=1, min_count=1)
            else:
                df_clean[(norm_label, method)] = pd.NA

    final_cols = [(cand, method) for cand in candidate_order for method in vote_methods + ["Total"]]
    df_clean = df_clean.reindex(columns=pd.MultiIndex.from_tuples(final_cols))

    # Step 10: Return the parsed structure in pipeline-compatible format
    contest_title = target_contest
    headers = df_clean.columns.tolist()
    df_clean = df_clean.reset_index()

    # Append Grand Totals row
    grand_totals = {col: df_clean[col].sum(min_count=1) for col in df_clean.columns if pd.api.types.is_numeric_dtype(df_clean[col])}
    grand_totals[index_label] = "Grand Totals"
    df_clean = pd.concat([df_clean, pd.DataFrame([grand_totals])], ignore_index=True)
    df_clean.columns = [
        f"{col[0]} - {col[1]}" if isinstance(col, tuple) else col
        for col in df_clean.columns
    ]
    headers = df_clean.columns.tolist()
    data = df_clean.to_dict(orient="records")
    metadata = {
        "state": html_context.get("state", "Unknown"),
        "county": html_context.get("county", "Unknown"),
        "race": contest_title,
        "report_time": html_context.get("report_time", None)
    }
    output_path = get_output_path(metadata["state"], metadata["county"], "parsed")
    timestamp = format_timestamp() if os.getenv("INCLUDE_TIMESTAMP_IN_FILENAME", "true").lower() == "true" else ""
    safe_title = re.sub(r"[\\/*?\"<>|]", "_", contest_title)
    filename = f"{safe_title.replace(' ', '_')}_results_{timestamp}.csv" if timestamp else f"{safe_title.replace(' ', '_')}_results.csv"

    filepath = os.path.join(output_path, filename)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        data = [{str(k): v for k, v in row.items()} for row in data]
        writer.writerows(data)
        f.write(f"\n# Generated at: {format_timestamp()}")    
    metadata["output_file"] = filepath

    # Optionally write metadata side-by-side as JSON
    json_meta_path = filepath.replace(".csv", "_metadata.json")
    with open(json_meta_path, "w", encoding="utf-8") as jf:
        json.dump(metadata, jf, indent=2)
    logger.info(f"[OUTPUT] Metadata written to: {json_meta_path}")
    return headers, data, contest_title, metadata
