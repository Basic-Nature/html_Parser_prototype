# ============================================================
# 🗳️ Smart Elections: Universal JSON Election Results Parser
# ============================================================

import json
import os
import csv
from collections import defaultdict
from ...config import BASE_DIR
from ...bots.librarian import (
    LOCATION_KEYWORDS, CANDIDATE_KEYWORDS, BALLOT_TYPES, PARTY_KEYWORDS, TOTAL_KEYWORDS,
    MISC_FOOTER_KEYWORDS, CONTEST_KEYWORDS
)
from ...utils.table_core import harmonize_headers_and_data

def get_input_folder():
    # Parent of webapp, then 'input'
    return os.path.join(os.path.dirname(BASE_DIR), "input")

def get_output_folder():
    # Parent of webapp, then 'output'
    return os.path.join(os.path.dirname(BASE_DIR), "output")

def list_json_files(input_folder):
    return [f for f in os.listdir(input_folder) if f.lower().endswith(".json")]

def prompt_for_json_file(input_folder):
    json_files = list_json_files(input_folder)
    if not json_files:
        print("[ERROR] No JSON files found in the input directory.")
        return None
    print("\nAvailable JSON files in 'input' folder:")
    for i, fname in enumerate(json_files):
        print(f"  [{i}] {fname}")
    print("\n[PROMPT] Enter file index or press Enter to cancel:", end=" ")
    user_input = input().strip()
    if not user_input:
        return None
    if user_input.isdigit():
        idx = int(user_input)
        if 0 <= idx < len(json_files):
            return os.path.join(input_folder, json_files[idx])
    print("[ERROR] Invalid selection.")
    return None

def parse_json_election_results(json_path, output_dir=None):
    # === Load JSON ===
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # === List Available Contests ===
    contests = set()
    for item in data.get("results", {}).get("ballotItems", []):
        name = item.get("name", "").strip()
        if name:
            contests.add(name)

    print("\nAvailable contests:")
    for i, name in enumerate(sorted(contests), 1):
        print(f" {i:2d}. {name}")

    # === Prompt for Contest Name ===
    print("\nEnter the contest name (exactly as shown), or type its number:")
    user_input = input("> ").strip()

    # Resolve numeric index to name
    if user_input.isdigit():
        idx = int(user_input)
        try:
            target_contest = sorted(contests)[idx - 1]
        except IndexError:
            raise ValueError("Invalid contest number.")
    else:
        if user_input not in contests:
            raise ValueError("Contest name not found.")
        target_contest = user_input

    print(f"\n🔍 Parsing contest: {target_contest}\n")

    # === Setup Output Paths ===
    if output_dir is None:
        output_dir = get_output_folder()
    os.makedirs(output_dir, exist_ok=True)
    safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in target_contest).replace(" ", "_")
    output_csv = os.path.join(output_dir, f"{safe_title}_parsed.csv")
    output_meta = os.path.join(output_dir, f"{safe_title}_metadata.json")

    # === Group Rename Map ===
    group_rename = {
        "Election Day": "Election Day",
        "Early Voting": "Early",
        "Absentee Mail": "Mail-In",
        "Provisional": "Provisional"
    }
    vote_methods = list(group_rename.values())

    # === Candidate Normalization ===
    raw_candidates = {}
    for item in data["results"]["ballotItems"]:
        if item["name"].strip() != target_contest:
            continue
        for opt in item["ballotOptions"]:
            raw = opt["name"].strip()
            party = opt.get("politicalParty", "Unknown")
            label = f"{raw} ({party})"
            raw_candidates[raw] = label

    normalization_map = {k: v for k, v in raw_candidates.items()}
    candidate_order = sorted(set(normalization_map.values()))

    # === Parse JSON Data ===
    results_nested = defaultdict(lambda: defaultdict(dict))

    for item in data["results"]["ballotItems"]:
        if item["name"].strip() != target_contest:
            continue
        for opt in item["ballotOptions"]:
            raw_label = opt["name"].strip()
            for precinct in opt["precinctResults"]:
                p = precinct["name"].strip()
                results_nested[p][raw_label]["Total"] = precinct.get("voteCount")
                for grp in precinct.get("groupResults", []):
                    g = grp["groupName"].strip()
                    norm_g = group_rename.get(g, g)
                    results_nested[p][raw_label][norm_g] = grp.get("voteCount")

    # === Build Wide-format Rows (harmonized with librarian context) ===
    rows = []
    all_keys = set()
    for precinct, cands in results_nested.items():
        row = {"Precinct": precinct}
        for raw_label, methods in cands.items():
            norm_label = normalization_map.get(raw_label, raw_label)
            for method, count in methods.items():
                col_name = f"{norm_label} - {method}"
                row[col_name] = count
                all_keys.add(col_name)
        rows.append(row)

    headers = ["Precinct"] + sorted(all_keys)

    # Harmonize and add grand total if needed
    headers, rows = harmonize_headers_and_data(headers, rows)

    # === Write Output CSV ===
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # === Write Metadata JSON ===
    metadata = {
        "race": target_contest,
        "input_file": os.path.basename(json_path),
        "output_file": os.path.basename(output_csv),
        "headers": headers,
        "row_count": len(rows),
        "handler": "json_handler"
    }
    with open(output_meta, "w", encoding="utf-8") as jf:
        json.dump(metadata, jf, indent=2)

    print("✅ Completed!")
    print(" - Output CSV:", output_csv)
    print(" - Metadata:", output_meta)

    return headers, rows, target_contest, metadata

# --- Entry point for pipeline integration ---
def parse(page=None, coordinator=None, html_context=None, non_interactive=False, manual_file=None, **kwargs):
    """
    Universal pipeline entry: Accepts a JSON file path (manual_file) from the format router,
    or prompts user to select a file from the input folder.
    Returns: headers, data, contest_title, metadata
    """
    html_context = html_context or {}
    if html_context.get("skip_format") or html_context.get("manual_skip"):
        return None, None, None, {"skipped": True}

    input_folder = get_input_folder()
    json_path = None

    # 1. Use file handed over from format router if provided
    if manual_file and os.path.isfile(manual_file):
        json_path = manual_file
    else:
        # 2. Otherwise, prompt user to select from input folder
        json_path = prompt_for_json_file(input_folder)
        if not json_path:
            print("[INFO] No file selected. Exiting JSON parser.")
            return None, None, None, {"skipped": True}

    print(f"\n[INFO] Using JSON file: {json_path}")

    # Run the contest parser pipeline
    return parse_json_election_results(json_path)

# If run as a script, allow standalone use
if __name__ == "__main__":
    input_folder = get_input_folder()
    json_path = prompt_for_json_file(input_folder)
    if json_path:
        parse_json_election_results(json_path)
