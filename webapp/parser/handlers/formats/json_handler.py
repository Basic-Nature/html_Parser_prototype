import json
import os
from ...state_router import get_handler_from_context, resolve_state_handler
from ...utils.shared_logger import logger, rprint
from ...utils.output_utils import finalize_election_output
from ...utils.table_builder import harmonize_rows, calculate_grand_totals, clean_candidate_name
from ...utils.shared_logic import normalize_text
from collections import defaultdict

def detect_json_files(input_folder="input"):
    """Return a list of JSON files in the input folder, sorted by modified time (newest first)."""
    try:
        json_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".json")]
        json_files.sort(key=lambda x: os.path.getmtime(os.path.join(input_folder, x)), reverse=True)
        return [os.path.join(input_folder, f) for f in json_files]
    except Exception as e:
        logger.error(f"[ERROR] Failed to list JSON files: {e}")
        return []

def prompt_file_selection(json_files):
    """Prompt user to select a JSON file from the list."""
    rprint("\n[yellow]Available JSON files in 'input' folder:[/yellow]")
    for i, f in enumerate(json_files):
        rprint(f"  [bold cyan][{i}][/bold cyan] {os.path.basename(f)}")
    idx = input("\n[PROMPT] Enter file index or press Enter to cancel: ").strip()
    if not idx.isdigit():
        rprint("[yellow]No file selected. Skipping JSON parsing.[/yellow]")
        return None
    try:
        return json_files[int(idx)]
    except (IndexError, ValueError):
        rprint("[red]Invalid index. Skipping JSON parsing.[/red]")
        return None

def parse(page, html_context):
    # Respect early skip signal from calling context
    if html_context.get("skip_format") or html_context.get("manual_skip"):
        return None, None, None, {"skipped": True}

    json_path = html_context.get("json_source")
    if not json_path:
        json_files = detect_json_files()
        if not json_files:
            logger.error("[ERROR] No JSON files found in the input directory.")
            return None, None, None, {"error": "No JSON in input folder"}
        json_path = prompt_file_selection(json_files)
        if not json_path:
            return None, None, None, {"skipped": True}

    try:
        rprint("[yellow]Available JSON file detected:[/yellow]")
        rprint(f"  [bold cyan]{os.path.basename(json_path)}[/bold cyan]")
        user_input = input("[PROMPT] Parse this file? (y/n): ").strip().lower()
        if user_input != 'y':
            logger.info("[INFO] User declined JSON parse. Skipping.")
            return None, None, None, {"skip_json": True}
    except Exception as e:
        logger.warning(f"[WARN] Skipping user input prompt due to error: {e}")
        return None, None, None, {"error": str(e)}

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Try to resolve a state-specific handler
        handler = get_handler_from_context(html_context) or resolve_state_handler(html_context.get("source_url", ""))
        if handler and hasattr(handler, "parse_json"):
            logger.info(f"Using custom state handler '{handler.__name__}' for JSON parsing.")
            return handler.parse_json(data, html_context)

        # Fallback: generic JSON structure
        ballot_items = data.get("results", {}).get("ballotItems", [])
        if not ballot_items:
            rprint("[red][ERROR] No contests found in JSON structure.[/red]")
            return None, None, None, {"error": "No contests in JSON"}

        # Detect available contests and prompt user
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
                rprint("[red]Invalid contest number.[/red]")
                return None, None, None, {"error": "Invalid contest number"}
        else:
            if user_input not in contests:
                rprint(f"[red][ERROR] Contest name '{user_input}' not found.[/red]")
                return None, None, None, {"error": "Contest name not found"}
            target_contest = user_input

        # --- Data normalization and cleaning logic ---
        group_rename = {
            "Election Day": "Election Day",
            "Early Voting": "Early Voting",
            "Absentee Mail": "Absentee Mail",
            "Mail-In": "Absentee Mail",
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
                label = f"{clean_candidate_name(raw)} ({party})"
                raw_candidates[raw] = label

        normalization_map = raw_candidates.copy()
        candidate_order = sorted(set(normalization_map.values()))

        # Build nested results: precinct → candidate → vote method
        results_nested = defaultdict(lambda: defaultdict(dict))
        for item in ballot_items:
            if item["name"].strip() != target_contest:
                continue
            for opt in item.get("ballotOptions", []):
                raw_label = opt.get("name", "").strip()
                for precinct in opt.get("precinctResults", []):
                    p = precinct.get("name", "").strip()
                    if p == "-" or not p:
                        p = None
                    results_nested[p][raw_label]["Total"] = precinct.get("voteCount")
                    for grp in precinct.get("groupResults", []):
                        g = grp.get("groupName", "").strip()
                        norm_g = group_rename.get(g, g)
                        results_nested[p][raw_label][norm_g] = grp.get("voteCount")

        # Flatten nested results into rows
        headers = []
        data_rows = []
        for precinct, cands in results_nested.items():
            row = {"Precinct": precinct}
            for raw_label, methods in cands.items():
                canonical_label = normalization_map.get(raw_label, raw_label)
                for method, count in methods.items():
                    col = f"{canonical_label} - {method}"
                    row[col] = count
                    if col not in headers:
                        headers.append(col)
            data_rows.append(row)
        headers = ["Precinct"] + sorted([h for h in headers if h != "Precinct"])

        # Harmonize rows and add grand total
        data_rows = harmonize_rows(headers, data_rows)
        grand_total = calculate_grand_totals(data_rows)
        data_rows.append(grand_total)

        # Metadata
        metadata = {
            "state": html_context.get("state", "Unknown"),
            "county": html_context.get("county", "Unknown"),
            "race": target_contest,
            "handler": "json_handler",
            "source": json_path
        }

        # Output via finalize_election_output
        result = finalize_election_output(headers, data_rows, target_contest, metadata)
        contest_title = result.get("contest_title", target_contest)
        metadata = result.get("metadata", metadata)
        return headers, data_rows, contest_title, metadata

    except Exception as e:
        logger.error(f"[ERROR] Failed to parse JSON: {e}")
        return None, None, None, {"error": str(e)}