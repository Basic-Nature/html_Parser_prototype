import os
import sys
import spacy
from spacy.training import offsets_to_biluo_tags
import orjson
import subprocess
import time
from pathlib import Path
from ..config import LOG_DIR, PROJECT_ROOT
from ..utils.shared_logger import log_info, log_error, log_warning, log_debug

def resolve_jsonl_path(jsonl_path):
    # If absolute, use as-is; else, join with LOG_DIR
    p = Path(jsonl_path)
    if not p.is_absolute():
        p = Path(LOG_DIR) / p
    return str(p)

def scan_misaligned(jsonl_path=None, verbose=False, output_misaligned=True, top_n=10):
    """
    Scan a spaCy NER JSONL file for misaligned examples.
    Optionally writes misaligned examples to spacy_ner_misaligned.jsonl for correction.
    Returns:
        0 if no misaligned examples found,
        1 if file missing or unreadable,
        2 if misaligned examples found.
    """
    nlp = spacy.blank("en")
    misaligned = []
    total = 0
    if jsonl_path is None:
        jsonl_path = os.path.join(LOG_DIR, "spacy_ner_train_data.jsonl")
    else:
        jsonl_path = resolve_jsonl_path(jsonl_path)
    if not os.path.exists(jsonl_path):
        log_error(f"[ERROR] File not found: {jsonl_path}")
        return 1
    try:
        with open(jsonl_path, "rb") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = orjson.loads(line)
                    text = entry.get("text")
                    entities = entry.get("entities")
                    total += 1
                    try:
                        tags = offsets_to_biluo_tags(nlp.make_doc(text), entities)
                        if "-" in tags:
                            misaligned.append({"text": text, "entities": entities})
                            if verbose:
                                log_info(f"MISALIGNED: {text} {entities}")
                    except Exception as e:
                        misaligned.append({"text": text, "entities": entities, "error": str(e)})
                        if verbose:
                            log_error(f"ERROR: {text} {entities} ({e})")
                except Exception as e:
                    log_warning(f"[CORRUPT] Could not parse line: {e}")
    except Exception as e:
        log_error(f"[ERROR] Could not read file: {jsonl_path} ({e})")
        return 1

    log_info(f"\n[SUMMARY] {len(misaligned)} misaligned out of {total} examples.")

    misaligned_path = os.path.join(LOG_DIR, "spacy_ner_misaligned.jsonl")
    if misaligned and output_misaligned:
        with open(misaligned_path, "wb") as f:
            for entry in misaligned:
                f.write(orjson.dumps(entry, option=orjson.OPT_APPEND_NEWLINE))
        log_info(f"[INFO] Misaligned examples written to {misaligned_path}")
        # Summarize top misaligned texts
        from collections import Counter
        counter = Counter()
        for entry in misaligned:
            text = entry.get("text", "")
            if text:
                counter[text] += 1
        if counter:
            log_warning(f"\n[MISALIGNED] Top {top_n} most frequent misaligned NER texts:")
            for text, count in counter.most_common(top_n):
                log_warning(f"  {repr(text)}: {count} times")
            log_warning("[MISALIGNED] Consider cleaning or pattern-excluding these from your training data.")
        log_warning("Run the manual_correction_bot to review and clean these examples before retraining.")
        log_warning("If you see spaCy entity alignment warnings, consider cleaning your training data or using the provided validation function.")
        return 2
    elif not misaligned:
        log_info("[INFO] All NER training examples are aligned and ready for retraining.")
        # Remove old misaligned file if exists
        if os.path.exists(misaligned_path):
            try:
                os.remove(misaligned_path)
                log_info(f"[INFO] Removed old misaligned file: {misaligned_path}")
            except Exception as e:
                log_warning(f"[WARN] Could not remove old misaligned file: {e}")
        return 0

def self_heal_loop(jsonl_path, verbose, max_retries=3, cooldown=2):
    """
    Loop: scan -> correct -> rescan, until clean or max_retries reached.
    Calls manual_correction_bot for misaligned NER correction.
    """
    for attempt in range(1, max_retries + 1):
        log_info(f"\n[SELF-HEAL] Attempt {attempt}...")
        exit_code = scan_misaligned(jsonl_path, verbose)
        if exit_code == 0:
            log_info("[SELF-HEAL] Data is clean. Exiting self-heal mode.")
            return 0
        log_warning("[SELF-HEAL] Misalignments found. Launching manual_correction_bot for spacy_ner_misaligned...")
        # Always use the special field for misaligned NER
        result = subprocess.run([
            sys.executable, "-m", "webapp.parser.bots.manual_correction_bot",
            "--fields", "spacy_ner_misaligned", "--enhanced"
        ], cwd=PROJECT_ROOT)
        if result.returncode != 0:
            log_warning(f"[SELF-HEAL] manual_correction_bot exited with code {result.returncode}")
        log_warning(f"[SELF-HEAL] Sleeping {cooldown}s before rescanning...")
        time.sleep(cooldown)
    log_warning("[SELF-HEAL] Max retries reached. Some misalignments may remain.")
    return 2

def main():
    """Main entry point for the script.
Scans for misaligned NER examples and optionally runs manual correction."""
    import argparse
    parser = argparse.ArgumentParser(
        description=(
            "Scan spaCy NER training data for misaligned examples. "
            "Misaligned examples are written to spacy_ner_misaligned.jsonl for correction. "
            "After correction, retrain_table_structure_models will use only aligned data."
        )
    )
    parser.add_argument("--jsonl", type=str, default=None, help="Path to NER training data JSONL (default: LOG_DIR/spacy_ner_train_data.jsonl)")
    parser.add_argument("--verbose", action="store_true", help="Print all misaligned examples")
    parser.add_argument("--auto-correct", action="store_true", help="Automatically run manual_correction_bot if misaligned examples are found")
    parser.add_argument("--self-heal", action="store_true", help="Loop: scan -> correct -> rescan until clean or max retries")
    parser.add_argument("--max-retries", type=int, default=3, help="Max self-heal attempts")
    parser.add_argument("--cooldown", type=int, default=2, help="Seconds to wait between self-heal attempts")
    args = parser.parse_args()

    jsonl_path = args.jsonl if args.jsonl else os.path.join(LOG_DIR, "spacy_ner_train_data.jsonl")
    jsonl_path = resolve_jsonl_path(jsonl_path)
    if args.self_heal:
        exit_code = self_heal_loop(jsonl_path, args.verbose, args.max_retries, args.cooldown)
    else:
        exit_code = scan_misaligned(jsonl_path, args.verbose)
        if args.auto_correct and exit_code == 2:
            log_info("\n[INFO] Launching manual_correction_bot for review of misaligned NER...")
            subprocess.run([
                sys.executable, "-m", "webapp.parser.bots.manual_correction_bot",
                "--fields", "spacy_ner_misaligned", "--enhanced"
            ], cwd=PROJECT_ROOT)
    sys.exit(exit_code)
    
if __name__ == "__main__":
    try:
        nlp = spacy.blank("en")
    except OSError:
        log_error("[ERROR] spaCy model not found. Please install the 'en_core_web_sm' model.")
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True, cwd=PROJECT_ROOT)
        nlp = spacy.blank("en")
    main()