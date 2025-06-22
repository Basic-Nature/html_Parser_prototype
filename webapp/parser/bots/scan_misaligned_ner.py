import os
import sys
import spacy
from spacy.training import offsets_to_biluo_tags
import orjson
import subprocess
import time
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
def scan_misaligned(jsonl_path="log/spacy_ner_train_data.jsonl", verbose=False):
    nlp = spacy.blank("en")
    misaligned = []
    total = 0
    if not os.path.exists(jsonl_path):
        print(f"[ERROR] File not found: {jsonl_path}")
        return 1
    with open(jsonl_path, "rb") as f:
        for line in f:
            entry = orjson.loads(line)
            text = entry.get("text")
            entities = entry.get("entities")
            total += 1
            try:
                tags = offsets_to_biluo_tags(nlp.make_doc(text), entities)
                if "-" in tags:
                    misaligned.append((text, entities))
                    if verbose:
                        print(f"MISALIGNED: {text} {entities}")
            except Exception as e:
                misaligned.append((text, entities))
                if verbose:
                    print(f"ERROR: {text} {entities} ({e})")
    print(f"\n[SUMMARY] {len(misaligned)} misaligned out of {total} examples.")
    if misaligned:
        print("Run the manual_correction_bot to review and clean these examples.")
    return 0 if not misaligned else 2

def self_heal_loop(jsonl_path, verbose, max_retries=3, cooldown=2):
    """Loop: scan -> correct -> rescan, until clean or max_retries reached."""
    for attempt in range(1, max_retries + 1):
        print(f"\n[SELF-HEAL] Attempt {attempt}...")
        exit_code = scan_misaligned(jsonl_path, verbose)
        if exit_code == 0:
            print("[SELF-HEAL] Data is clean. Exiting self-heal mode.")
            return 0
        print("[SELF-HEAL] Misalignments found. Launching manual_correction_bot...")
        subprocess.run([sys.executable, "-m", "webapp.parser.bots.manual_correction_bot", "--fields", "tables", "--enhanced"], check=True, cwd=project_root)
        print(f"[SELF-HEAL] Sleeping {cooldown}s before rescanning...")
        time.sleep(cooldown)
    print("[SELF-HEAL] Max retries reached. Some misalignments may remain.")
    return 2

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scan spaCy NER training data for misaligned examples.")
    parser.add_argument("--jsonl", type=str, default="log/spacy_ner_train_data.jsonl", help="Path to NER training data JSONL")
    parser.add_argument("--verbose", action="store_true", help="Print all misaligned examples")
    parser.add_argument("--auto-correct", action="store_true", help="Automatically run manual_correction_bot if misaligned examples are found")
    parser.add_argument("--self-heal", action="store_true", help="Loop: scan -> correct -> rescan until clean or max retries")
    parser.add_argument("--max-retries", type=int, default=3, help="Max self-heal attempts")
    parser.add_argument("--cooldown", type=int, default=2, help="Seconds to wait between self-heal attempts")
    args = parser.parse_args()

    if args.self_heal:
        exit_code = self_heal_loop(args.jsonl, args.verbose, args.max_retries, args.cooldown)
    else:
        exit_code = scan_misaligned(args.jsonl, args.verbose)
        if args.auto_correct and exit_code == 2:
            print("\n[INFO] Launching manual_correction_bot for review...")
            subprocess.run([sys.executable, "-m", "webapp.parser.bots.manual_correction_bot", "--fields", "tables", "--enhanced"], check=True, cwd=project_root)
    sys.exit(exit_code)