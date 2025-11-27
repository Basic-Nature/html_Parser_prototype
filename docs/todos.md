# TODO/FIXME index — webapp

Total annotations: 526

## High Priority

### `webapp\parser\utils\shared_logic.py` (High Priority)

- L2699 *FIXME*: ', 'BUG'\]

## Medium Priority

### `webapp\parser\health\manual_correction_bot.py` (Medium Priority)

- L750 *TODO*: Add JSON schema validation here if desired

### `webapp\parser\utils\shared_logic.py` (Medium Priority)

- L2204 *TODO*: /FIXME/WARN and similar keywords (case-insensitive). Returns list of (lineno, keyword, cleaned_text)."""
- L2206 *TODO*: |FIXME|WARN|WARNING|NOTE|HACK|XXX|BUG)\b", re.IGNORECASE)
- L2619 *TODO*: /FIXME/WARN
- L2622 *TODO*: /FIXME/WARN:")
- L2689 *TODO*: /FIXME/WARN lines from webapp/ into a compact index.
- L2700 *TODO*: ', 'HACK', 'XXX'\]
- L2736 *TODO*: /FIXME index — webapp\n")
- L3080 *TODO*: /FIXME/WARN")

## Low Priority

### `webapp\Smart_Elections_Parser_Webapp.py` (Low Priority)

- L210 *WARNING*: ").upper().split(","))
- L480 *WARNING*: , ERROR, CRITICAL, TRACE
- L519 *WARNING*: ", "ERROR", "CRITICAL", "TRACE"}
- L555 *WARNING*: " in mlow:
- L954 *WARNING*:         # For websocket handshake only: add Cache-Control so webhint stops warning
- L1229 *WARNING*: ({"type": "sec", "message": "Favicon path escape blocked", "requested": ico_path})
- L1331 *WARNING*: ({
- L1332 *WARNING*: ",
- L1641 *WARNING*: ",
- L1724 *WARNING*: (
- L1726 *WARNING*: ",
- L1736 *WARNING*: (
- L1738 *WARNING*: ",
- L1768 *WARNING*: (
- L1770 *WARNING*: ",
- L2055 *WARNING*: ({
- L2056 *WARNING*: ",
- L2120 *WARNING*: ({
- L2121 *WARNING*: ",
- L2171 *WARNING*: ({
- L2172 *WARNING*: ",
- L2194 *WARNING*: ({
- L2195 *WARNING*: ",
- L2203 *WARNING*: ({
- L2204 *WARNING*: ",
- L2211 *WARNING*: ({
- L2212 *WARNING*: ",

### `webapp\parser\Context_Integration\Context_Library\constants.py` (Low Priority)

- L1831 *NOTE*: .*$",                     # Note
- L2020 *WARNING*: ",
- L2111 *WARNING*: ", "info_box", "navigation", "pagination", "tab", "modal", "tooltip", "ignore", "unknown"
- L2144 *NOTE*: ", "comment",
- L2220 *NOTE*: ", "Comment", "Feedback", "Suggestion", "Recommendation",
- L2236 *NOTE*: ", "Comment", "Feedback", "Suggestion",

### `webapp\parser\Context_Integration\context_coordinator.py` (Low Priority)

- L788 *WARNING*: ("\[ALERT MONITOR\] Thread did not stop cleanly.")
- L876 *WARNING*: ({
- L877 *WARNING*: ",
- L995 *WARNING*: (f"\[yellow\]Integrity issues:\[/yellow\] {issues\['integrity_issues'\]}")
- L1234 *WARNING*: (f"\[ContextCoordinator\] No table structure found for contest: {contest}")
- L1403 *WARNING*: (f"\[get_feedback_pattern_kb\] Skipping corrupt line: {e}")
- L1515 *WARNING*: ("\[group_dom_nodes_by_label\] No organized DOM parts. (Further warnings suppressed)")
- L1517 *WARNING*: (f"\[group_dom_nodes_by_label\] No organized DOM parts. (Occurred {ContextCoordinator._dom_parts_warning_count} times)")
- L1522 *WARNING*: ("\[group_dom_nodes_by_label\] No DOM nodes found.")
- L1540 *WARNING*: ("\[submit_user_feedback\] ContextOrganizer has no submit_user_feedback method.")
- L1568 *WARNING*: (f"\[correct_and_update_contest\] Contest {contest_id} missing type/election_types after sync.")
- L1592 *WARNING*: ("\[print_contest_summary\] No organized contests to summarize.")
- L1605 *WARNING*: ("\[plot_contest_distribution\] No organized contests to plot.")
- L1656 *WARNING*: ("No organized DOM parts.")
- L1659 *WARNING*: ("No organized DOM parts. (Further warnings suppressed)")
- L1670 *WARNING*: ("\[get_contest_groups\] No contest groups found.")
- L1679 *WARNING*: ("\[get_panel_groups\] No panel groups found.")
- L1688 *WARNING*: ("\[get_button_groups\] No button groups found.")
- L1697 *WARNING*: ("\[get_table_groups\] No table groups found.")
- L1706 *WARNING*: ("\[get_relationships\] No organized context.")
- L1814 *WARNING*: (f"\[fuzzy_score\] One or both inputs are empty: a='{a_str}', b='{b_str}'")
- L1820 *WARNING*: (f"\[fuzzy_score\] One or both inputs are too short: a='{a_str}', b='{b_str}'")
- L2266 *WARNING*: (f"\[extract_field\] Unknown field_type: {field_type}")
- L2524 *WARNING*: (f"\[get_full_contest\] Contest {contest_id} missing type/election_types after sync.")
- L2609 *WARNING*: (f"\[list_tables\] Table '{tbl}' missing metadata or columns.")
- L2641 *WARNING*: (f"\[get_table_metadata\] Table '{table_name}' missing columns.")
- L2659 *WARNING*: (f"\[check_missing_tables\] Missing tables: {missing}")
- L2720 *WARNING*: (f"\[save_table_structure\] Failed to save structure for contest: {contest}")
- L2897 *WARNING*: (f"\[get_best_button_advanced\] Contest argument was not a dict. Converted to: {contest}")
- L2901 *WARNING*: (f"\[get_best_button_advanced\] Keywords argument was not a list. Converted to: {keywords}")
- L2905 *WARNING*: (f"\[get_best_button_advanced\] Context argument was not a dict. Converted to: {context}")
- L2912 *WARNING*: ("\[get_best_button_advanced\]_semantic_model is not set or is not an object. Using None.")
- L3057 *WARNING*: (f"\[yellow\]\[Coordinator\] Button '{cand.get('label')}' rejected, retrying...\[/yellow\]")

### `webapp\parser\Context_Integration\context_organizer.py` (Low Priority)

- L282 *WARNING*: (
- L407 *WARNING*: (f"\[CONTEST\] Skipping contest with suspiciously large or missing title: {str(title)\[:100\]}...")
- L495 *WARNING*: (f"\[CONTEST\] Filtered out {len(filtered_out)} contests due to missing required fields.")
- L497 *WARNING*: (f"  \[Filtered\] {reason}: {str(c)\[:100\]}...")
- L500 *WARNING*: ("\[CONTEST\] No contests with required fields for downstream output.")
- L816 *WARNING*: (f"\[ML\] Anomaly index {idx} out of range for contests list of length {len(contests)}")
- L1500 *WARNING*: (f"  \[yellow\]{title}\[/yellow\]: {fixes}")
- L1505 *WARNING*: (f"\[bold yellow\]\[INTEGRITY\]\[/bold yellow\] Duplicate contest detected.\n  \[dim\]Context:\[/dim\] {contest}")
- L1507 *WARNING*: (f"\[bold yellow\]\[INTEGRITY\]\[/bold yellow\] Contest missing location info.\n  \[dim\]Context:\[/dim\] {contest}")
- L1509 *WARNING*: (f"\[bold yellow\]\[INTEGRITY\]\[/bold yellow\] Contest missing year.\n  \[dim\]Context:\[/dim\] {contest}")
- L1972 *WARNING*: (f"\[ContextOrganizer\] Could not update context library with feedback: {e}")
- L2049 *WARNING*: (f"\[CONTEXT ORGANIZER\] No table structure found for contest: {contest}")

### `webapp\parser\Context_Integration\librarian.py` (Low Priority)

- L652 *WARNING*: (f"\n\[LIBRARIAN SELF-HEAL\] Attempt {attempt}...")
- L658 *WARNING*: ("\[LIBRARIAN SELF-HEAL\] Misalignments found. Launching manual_correction...")
- L661 *WARNING*: (f"\[LIBRARIAN SELF-HEAL\] Sleeping {cooldown}s before rescanning...")

### `webapp\parser\config.py` (Low Priority)

- L328 *WARNING*: ("\[DB\]\[AAD\] Falling back to password auth.")

### `webapp\parser\data_manager.py` (Low Priority)

- L83 *WARNING*: (f"\[REMOVED\] {popped}")
- L90 *WARNING*: (f"\[REMOVED\] {index_or_value}")
- L129 *WARNING*: (f"\[DELETED\] {files\[idx\]}")

### `webapp\parser\handlers\batch_handler.py` (Low Priority)

- L134 *WARNING*: ({
- L135 *WARNING*: ",
- L426 *WARNING*: ({
- L427 *WARNING*: ",

### `webapp\parser\handlers\formats\html_handler.py` (Low Priority)

- L216 *WARNING*: (f"\[HTML Handler\] County '{county}' not found. Closest matches: {matches}")
- L220 *WARNING*: (f"\[HTML Handler\] Detected county '{county}' is not in known counties for state '{suggested_state or state}'.")
- L241 *WARNING*: (f"\[HTML Handler\] State '{user_state}' not found. Closest matches: {matches}")
- L285 *WARNING*: (f"\[HTML Handler\] County '{user_county}' not found. Closest matches: {matches}")

### `webapp\parser\handlers\formats\json_handler.py` (Low Priority)

- L376 *WARNING*: ({
- L377 *WARNING*: ",
- L489 *WARNING*: ({
- L490 *WARNING*: ",

### `webapp\parser\handlers\formats\pdf_handler.py` (Low Priority)

- L421 *WARNING*: ({
- L422 *WARNING*: ",
- L425 *WARN*: \] Detected PyMuPDF %s. Upgrade to %s or newer to avoid parser instability."
- L1787 *WARNING*: ({
- L1788 *WARNING*: ",
- L1790 *WARN*: \] Poppler binaries not detected; skipping pdf2image and using PyMuPDF fallback.",
- L1808 *WARNING*: ({
- L1809 *WARNING*: ",
- L1812 *WARN*: \] pdf2image conversion failed; "
- L2184 *WARNING*: ({
- L2185 *WARNING*: ",
- L2187 *WARN*: \] Multi-mode text extraction failed: {e}",
- L3283 *WARNING*: ({
- L3284 *WARNING*: ",
- L3286 *WARN*: \] fitz text extraction failed: {e}",
- L3315 *WARNING*: ({
- L3316 *WARNING*: ",
- L3318 *WARN*: \] ENABLE_OCR_FORCE is set but Tesseract is unavailable; skipping OCR fallback.",
- L3366 *WARNING*: ({
- L3367 *WARNING*: ",
- L3369 *WARN*: \] Low-signal text detected but OCR is unavailable or disabled.",
- L3586 *WARNING*: ({
- L3587 *WARNING*: ",
- L3589 *WARN*: \] No contest selected. Using filename fallback.",
- L4034 *WARNING*: ({
- L4035 *WARNING*: ",
- L4037 *WARN*: \] Selected contest '{contest}' not found in column '{contest_column}'. Skipping row filter.",
- L4136 *WARNING*: ({
- L4137 *WARNING*: ",
- L4139 *WARN*: \] No structured rows matched the inferred column count of {len(headers)}. Total lines scanned: {unmatched_count}",
- L4178 *WARNING*: ({
- L4179 *WARNING*: ",
- L4367 *WARNING*: ({
- L4368 *WARNING*: ",

### `webapp\parser\handlers\states\arizona\arizona.py` (Low Priority)

- L25 *WARNING*: ("\[WARN\] context_library.json not found. Using fallback config for Arizona handler.")
- L51 *WARNING*: (f"\[WARN\] Could not expand card {i+1}: {e}")
- L64 *WARNING*: (f"\[WARN\] Vote Type toggle failed: {e}")
- L77 *WARNING*: (f"\[WARN\] County toggle failed: {e}")
- L164 *WARNING*: ("\[FALLBACK\] No tables were parsed. Either no results are published yet or the structure has changed.")
- L165 *WARNING*: ("\[FALLBACK\] Please verify that the site has posted election data.")

### `webapp\parser\handlers\states\example state\example_county\example_county.py` (Low Priority)

- L123 *WARNING*: ("\[yellow\]\[WARNING\] No ballot items found by div selectors. Trying table-based extraction...\[/yellow\]")

### `webapp\parser\handlers\states\example state\example_state.py` (Low Priority)

- L51 *WARNING*: (f"\[Example Handler\] No specific parser implemented for county: '{county}'. Continuing with state-level logic.")
- L152 *WARNING*: ("\[yellow\]\[WARNING\] No ballot items found by div selectors. Trying table-based extraction...\[/yellow\]")

### `webapp\parser\handlers\states\new_york\county\rockland.py` (Low Priority)

- L72 *WARNING*: ("\[WARNING\] dom_parts missing after organize_and_enrich.")
- L95 *WARNING*: ("\[red\]No contest selected. Skipping.\[/red\]")
- L139 *WARNING*: (f"\[yellow\]\[WARNING\] Button '{btn1.get('label', '')}' is not clickable (visible={safe_is_visible(element, logger)}, enabled={safe_is_enabled(element, logger)})\[/yellow\]")
- L176 *WARNING*: (f"\[yellow\]\[WARNING\] Button '{btn2.get('label', '')}' is not clickable (visible={safe_is_visible(element, logger)}, enabled={safe_is_enabled(element, logger)})\[/yellow\]")

### `webapp\parser\handlers\states\new_york\new_york.py` (Low Priority)

- L27 *WARNING*: ("\[NY Handler\] No county specified in html_context.")
- L43 *WARNING*: (f"\[NY Handler\] No specific parser implemented for county: '{county}'. Please add it under {module_path}.py")

### `webapp\parser\handlers\states\pennsylvania\pennsylvania.py` (Low Priority)

- L44 *WARNING*: (f"\[NAV\] Step failed: {step} — {e}")
- L55 *WARNING*: (f"\[bold yellow\]Detected election:\[/bold yellow\] {header_text}")
- L76 *WARNING*: ("\[PA\] Invalid index input for election selection.")
- L78 *WARNING*: ("\[PA\] Elections dropdown not found.")
- L80 *WARNING*: (f"\[PA\] Failed to expand Elections menu or load selection: {e}")
- L96 *WARNING*: ("\[PA\] County Breakdown link not found.")
- L98 *WARNING*: (f"\[PA\] Failed to click County Breakdown link: {e}")
- L113 *WARNING*: ("\[yellow\]Multiple CSV files found in input. Please select one:\[/yellow\]")

### `webapp\parser\health\health_router.py` (Low Priority)

- L252 *WARNING*: (f"\[health_router\] manual_correction failed (attempt {attempt}): {result.stderr}")
- L336 *WARNING*: ("\[SELF-HEAL\] Misalignments found. Launching manual_correction...")
- L338 *WARNING*: (f"\[SELF-HEAL\] Sleeping {cooldown}s before rescanning...")
- L340 *WARNING*: ("\[SELF-HEAL\] Max retries reached. Some misalignments may remain.")
- L375 *WARNING*: (f"\[PIPELINE\] Could not fix corrupted JSON files: {e}")
- L380 *WARNING*: ("\[PIPELINE\] Misaligned NER examples found. Self-heal loop will be handled by scan_misaligned_ner.")
- L382 *WARNING*: ("\[PIPELINE\] scan_misaligned_ner failed or file missing. Proceeding with caution.")
- L414 *WARNING*: ("\[PIPELINE\] Model retraining failed.")

### `webapp\parser\health\log_cache_cleaner_bot.py` (Low Priority)

- L151 *WARNING*: (f"Skipping non-dict entry in spacy_ner_train_data.jsonl: {entry}")
- L460 *WARNING*: ("\[DB\]\[WARNING\] No user tables found in schema 'public'.")
- L503 *WARNING*: ("\[CLEAN\]\[WARNING\] The following files are still too large after cleaning:")
- L507 *WARNING*: ("\[MISALIGNED\] Consider cleaning or pattern-excluding these from your training data:")

### `webapp\parser\health\manual_correction_bot.py` (Low Priority)

- L322 *WARNING*: (f"Coordinator ML scoring failed: {e}")
- L343 *WARNING*: (f"Coordinator field suggestion failed: {e}")
- L355 *WARNING*: (f"Log file not found: {path}")
- L364 *WARNING*: (f"\[CORRUPT\] {path} line {i}: {e}")
- L396 *WARNING*: (f"\[SKIP\] File not found: {file}")
- L400 *WARNING*: (f"\[SKIP\] File too large: {file}")
- L422 *WARNING*: (f"\[CORRUPT-LINE\] {file} line {i+1}: {line\[:80\]}... ({e})")
- L434 *WARNING*: (f"\[CORRUPT\] {len(corrupt_items)} lines saved to {corrupt_path}")
- L439 *WARNING*: (f"\[FIXED\] All lines invalid, recreated empty .jsonl file: {file}")
- L453 *WARNING*: (f"\[CORRUPT\] {file}: {e}")
- L465 *WARNING*: (f"\[CORRUPT\] Corrupt JSON saved to {corrupt_path}")
- L471 *WARNING*: (f"\[FIXED\] All content invalid, recreated minimal valid JSON in {file}")
- L476 *WARNING*: (f"\[CORRUPT\] {file}: {e}")
- L485 *WARNING*: (f"\[QUARANTINED\] {file} -&gt; {quarantine_dir / file.name}")
- L489 *WARNING*: (f"\[DELETED\] {file}")
- L492 *WARNING*: (f"\[SKIP-DELETE\] File already missing: {file}")
- L537 *WARNING*: (f"\[FIND-LOGS\] Skipped {d}: {e}")
- L562 *WARNING*: (f"\[CORRUPT\] {path} line {line_num}: {e}")
- L717 *WARNING*: (f"Invalid JSON, skipping edit: {e}")
- L989 *WARNING*: (
- L1079 *WARN*: if schema version mismatches.
- L1098 *WARNING*: (f"Schema version mismatch: found {context_lib.get('schema_version')}, expected {SCHEMA_VERSION}. Consider migrating.")
- L1141 *WARNING*: (f"\[AUTO\] Could not delete log file {log_file}: {e}")
- L1257 *WARNING*: (f"\[SKIP\] Could not load {log_file}: {e}")
- L1273 *WARNING*: ("No log files matched any of the specified fields. Will attempt to process all log files for all fields.")
- L1356 *WARNING*: (f"Could not delete log file {log_file}: {e}")
- L1376 *WARNING*: ("\[WARNING\] No entries were processed. Check your log file naming, field configuration, or use --dry-run for debugging.")

### `webapp\parser\health\retrain_table_structure_models.py` (Low Priority)

- L178 *WARNING*: (f"\[CLEAN\] File not found: {jsonl_path}")
- L186 *WARNING*: (f"\[CLEAN\] Could not parse line: {e}")
- L201 *WARNING*: (f"\[CLEAN\] Alignment check failed for text: {text\[:50\]}... ({e})")
- L274 *WARNING*: (f"Failed to load {path}: {e}")
- L403 *WARNING*: (f"Skipping misaligned entity in: {text}")
- L408 *WARNING*: (f"Error validating entity alignment: {e}")
- L434 *WARNING*: (f"\[spaCy\] Could not check GPU availability: {e}")
- L450 *WARNING*: (f"\[spaCy\] Could not load lexeme normalization table. You may ignore this for English. Error: {e}")
- L536 *WARNING*: (f"\[NER\] Skipped {misaligned_count} misaligned examples. Saved to {misaligned_path}")
- L550 *WARNING*: ("No NER training examples found. Skipping spaCy NER retraining.")
- L619 *WARNING*: ("\[SUGGESTION\] Consider lowering min_delta or increasing patience if you want longer training.")
- L621 *WARNING*: ("\[SUGGESTION\] Model improved until the last epoch. Consider increasing epochs for further improvement.")
- L622 *WARNING*: (f"\[SUGGESTION\] Next run: patience={patience}, min_delta={min_delta:.2f}, epochs={epochs}")
- L708 *WARNING*: ("No training examples found. Aborting retraining.")
- L727 *WARNING*: (f"\[WARN\] Could not delete old model directory {oldest_path}: {e}")
- L739 *WARNING*: (f"\[WARN\] Failed to load existing model: {e}")
- L742 *WARNING*: ("Falling back to base model (all-MiniLM-L6-v2).")
- L782 *WARNING*: (f"\[WARN\] Could not update canonical model directory: {e}")
- L810 *WARNING*: (f"MISALIGNED: {text} {annots\['entities'\]}")
- L840 *WARNING*: ("\[DB\] Base.metadata.tables is empty. No models registered? Did you import all model classes?")

### `webapp\parser\health\scan_misaligned_ner.py` (Low Priority)

- L62 *WARNING*: (f"\[CORRUPT\] Could not parse line: {e}")
- L83 *WARNING*: (f"\n\[MISALIGNED\] Top {top_n} most frequent misaligned NER texts:")
- L85 *WARNING*: (f"  {repr(text)}: {count} times")
- L86 *WARNING*: ("\[MISALIGNED\] Consider cleaning or pattern-excluding these from your training data.")
- L87 *WARNING*: ("Run the manual_correction to review and clean these examples before retraining.")
- L88 *WARNING*: ("If you see spaCy entity alignment warnings, consider cleaning your training data or using the provided validation function.")
- L98 *WARNING*: (f"\[WARN\] Could not remove old misaligned file: {e}")
- L112 *WARNING*: ("\[SELF-HEAL\] Misalignments found. Launching manual_correction for spacy_ner_misaligned...")
- L119 *WARNING*: (f"\[SELF-HEAL\] manual_correction exited with code {result.returncode}")
- L120 *WARNING*: (f"\[SELF-HEAL\] Sleeping {cooldown}s before rescanning...")
- L122 *WARNING*: ("\[SELF-HEAL\] Max retries reached. Some misalignments may remain.")

### `webapp\parser\html_election_parser.py` (Low Priority)

- L56 *WARNING*: ("Deleting .processed_urls cache for fresh start...")
- L393 *WARNING*: ({
- L394 *WARNING*: ",
- L408 *WARNING*: ({
- L409 *WARNING*: ",
- L469 *WARNING*: ({
- L470 *WARNING*: ",
- L543 *WARNING*: (payload_2)
- L870 *WARNING*: ({
- L871 *WARNING*: ",
- L917 *WARNING*: ({
- L918 *WARNING*: ",
- L971 *WARNING*: ({
- L972 *WARNING*: ",
- L1076 *WARNING*: ",
- L1081 *WARNING*: (payload)
- L1106 *WARN*: if nothing found
- L1166 *WARNING*: ",
- L1171 *WARNING*: (payload)
- L1249 *WARNING*: ({
- L1250 *WARNING*: ",
- L1267 *WARNING*: ",
- L1272 *WARNING*: (payload)
- L1283 *WARNING*: ",
- L1288 *WARNING*: (payload)
- L1290 *WARN*: \] No output file path returned from parser and no output files found."
- L1292 *WARNING*: ",
- L1297 *WARNING*: (payload)
- L1302 *WARNING*: ",
- L1307 *WARNING*: (payload)
- L1425 *WARNING*: ({
- L1426 *WARNING*: ",
- L1486 *WARNING*: ({
- L1487 *WARNING*: ",

### `webapp\parser\state_router.py` (Low Priority)

- L49 *WARNING*: ("\[Router\] handlers/states directory not found.")
- L66 *WARNING*: (f"\[Router\] counties directory not found for state: {state_key}")
- L137 *WARNING*: (f"\[Fallback\]\[Session:{session_id}\] No handler states available for manual selection.")
- L154 *WARNING*: (f"\[Fallback\]\[Session:{session_id}\] Aborted by user.")
- L157 *WARNING*: (f"\[Fallback\]\[Session:{session_id}\] Aborted by user.")
- L160 *WARNING*: (f"\[Fallback\]\[Session:{session_id}\] State '{state}' not found. Please try again.")
- L179 *WARNING*: (f"\[Fallback\]\[Session:{session_id}\] Aborted by user.")
- L182 *WARNING*: (f"\[Fallback\]\[Session:{session_id}\] County '{county}' not found for state '{state}'. Please try again.")
- L189 *WARNING*: (f"\[Fallback\]\[Session:{session_id}\] Too many failed attempts. Exiting fallback.")
- L205 *WARNING*: (f"\[Router\] Requested state '{state_name}' not found on disk. Skipping restrict filter.")
- L512 *WARNING*: (f"No counties found for state '{state}'. Try --fuzzy for fuzzy matching.")
- L523 *WARNING*: (f"Failed to load context from file: {e}")
- L533 *WARNING*: ("No suitable handler found.")
- L540 *WARNING*: ("No handler selected. Exiting.")
- L547 *WARNING*: ("Still could not import a suitable handler.")

### `webapp\parser\utils\browser_utils.py` (Low Priority)

- L89 *WARNING*: (f"\[browser_utils\] Failed to safely parse context_library value for key '{key}'")
- L91 *WARNING*: (f"\[browser_utils\] Skipping unsafe context_library value for key '{key}'")
- L295 *WARNING*: (f"\[safe_attributes\] Playwright JS extraction failed: {e}")
- L309 *WARNING*: (f"\[safe_attributes\] Playwright fallback extraction failed: {e}")
- L395 *WARNING*: (f"\[safe_count\] Object is not countable: {type(obj)}")
- L441 *WARNING*: (f"\[safe_launch\] browser_type is not a SyncBrowserType: {type(browser_type)}")
- L461 *WARNING*: (f"\[async_safe_launch\] browser_type is not an AsyncBrowserType: {type(browser_type)}")
- L540 *WARNING*: ({
- L541 *WARNING*: ",
- L569 *WARNING*: (f"\[CAPTCHA\] Detected Cloudflare CAPTCHA indicator: '{indicator}'")
- L578 *WARNING*: (f"\[CAPTCHA\] CAPTCHA detected in async mode. Manual intervention not implemented. (Session: {session_id})")
- L602 *WARNING*: (f"\[CAPTCHA\] Detected Cloudflare CAPTCHA indicator: '{indicator}'")
- L611 *WARNING*: ({
- L612 *WARNING*: ",
- L623 *WARNING*: (f"\[CAPTCHA\] CAPTCHA detected in sync mode. Manual intervention not implemented. (Session: {session_id})")
- L712 *WARNING*: ("\[SCROLL\] User aborted scrolling.")
- L733 *WARNING*: ("\[SCROLL\] Max scroll time/attempts exceeded. Page may not be fully loaded.")

### `webapp\parser\utils\captcha_tools.py` (Low Priority)

- L118 *WARNING*: (f"\[CAPTCHA\] Foreground window fallback failed: {e}")
- L154 *WARNING*: ("\[CAPTCHA\] CAPTCHA not resolved within timeout.")

### `webapp\parser\utils\contest_selector.py` (Low Priority)

- L635 *WARNING*: ":
- L636 *WARNING*: (entry)
- L1029 *WARNING*: ", "selector", f"Feedback loop {loop+1}: verifying contests", session_id=session_id,
- L1565 *WARNING*: ({"level": "WARNING", "type": "selector", "message": "Empty search term", "session_id": session_id})
- L1570 *WARNING*: ({"level": "WARNING", "type": "selector", "message": f"No matches for '{term}'", "session_id": session_id})
- L1642 *WARNING*: ({"level": "WARNING", "type": "selector", "message": "No match; try again.", "session_id": session_id})

### `webapp\parser\utils\dom_extractor.py` (Low Priority)

- L153 *WARNING*: (f"\[DOM_EXTRACTOR\] failure: {e}")

### `webapp\parser\utils\dynamic_table_extractor.py` (Low Priority)

- L124 *WARNING*: ", "extractor", "\[EXTRACTOR\] No &lt;table&gt; found in provided table_html.", session_id)
- L129 *WARNING*: ", "extractor", "\[EXTRACTOR\] No &lt;tr&gt; rows found in table_html.", session_id)
- L171 *WARNING*: ", "extractor", "\[EXTRACTOR\] Candidate NLP/score step failed", session_id, error=str(e))
- L187 *WARNING*: ", "extractor", "\[EXTRACTOR\] No suitable table candidates found.", session_id)
- L217 *WARNING*: ", "extractor", "\[EXTRACTOR\] Error while scanning &lt;table&gt; elements", session_id, error=str(e))
- L229 *WARNING*: ", "extractor", "\[EXTRACTOR\] DOM extraction failed", session_id, error=str(e))
- L272 *WARNING*: ", "extractor", "\[EXTRACTOR\] Pattern extraction failed", session_id, error=str(e))
- L776 *WARNING*: ", "extractor", "No learned DOM patterns found.")
- L800 *WARNING*: ", "extractor", "Entry deleted.")
- L805 *WARNING*: ", "extractor", "Unknown action.")
- L807 *WARNING*: ", "extractor", "Invalid entry number.")

### `webapp\parser\utils\embedding_cache.py` (Low Priority)

- L178 *WARNING*: (msg)

### `webapp\parser\utils\extraction_strategies.py` (Low Priority)

- L68 *WARNING*: (f"\[STRATEGY\] {name} failed: {e}")

### `webapp\parser\utils\format_router.py` (Low Priority)

- L374 *WARNING*: ({
- L375 *WARNING*: ",
- L377 *WARN*: \] No supported file formats found on the page.",
- L402 *WARNING*: ({
- L403 *WARNING*: ",
- L405 *WARN*: \] Unsupported format requested: {format_str}",
- L409 *WARNING*: ({
- L410 *WARNING*: ",
- L654 *WARNING*: ({
- L655 *WARNING*: ",
- L874 *WARNING*: ({
- L875 *WARNING*: ",
- L950 *WARNING*: ({
- L951 *WARNING*: ",

### `webapp\parser\utils\html_scanner.py` (Low Priority)

- L163 *WARNING*: ",
- L167 *WARNING*: (payload)
- L189 *WARNING*: ",
- L193 *WARNING*: (payload)
- L288 *WARNING*: ",
- L292 *WARNING*: (payload)
- L315 *WARNING*: ",
- L319 *WARNING*: (payload)
- L353 *WARNING*: ",
- L357 *WARNING*: (payload)
- L380 *WARNING*: ",
- L384 *WARNING*: (payload)
- L579 *WARNING*: ",
- L583 *WARNING*: (payload)
- L784 *WARNING*: (f"\[ML SIMILARITY\] No embedding computed for segment: {safe_get(segment, 'segment_hash', None)}")
- L807 *WARNING*: (f"\[ML SIMILARITY\] No embedding computed for segment: {safe_get(segment, 'segment_hash', None)}")
- L1034 *WARNING*: ",
- L1038 *WARNING*: (payload)
- L1045 *WARNING*: ",
- L1049 *WARNING*: (payload)
- L1376 *WARNING*: ",
- L1380 *WARNING*: (payload)
- L1438 *WARNING*: ",
- L1442 *WARNING*: (payload)
- L1691 *WARNING*: ({"level": "WARNING", "type": "dom_segments", "message": msg_warn})
- L1747 *WARNING*: ({"level": "WARNING", "type": "page_hash", "message": msg})
- L1754 *WARNING*: ({"level": "WARNING", "type": "page_hash", "message": msg})
- L1766 *WARNING*: ({"level": "WARNING", "type": "page_hash", "message": msg})
- L1789 *WARNING*: ({"level": "WARNING", "type": "cache", "message": msg})
- L1824 *WARNING*: ({"level": "WARNING", "type": "cache", "message": msg})
- L2003 *WARNING*: ({"level": "WARNING", "type": "segment_review", "message": msg})
- L2012 *WARNING*: ({
- L2013 *WARNING*: ",
- L2129 *WARNING*: ",
- L2133 *WARNING*: (payload)
- L2145 *WARNING*: ",
- L2149 *WARNING*: (payload)
- L2158 *WARNING*: ",
- L2162 *WARNING*: (payload)
- L2177 *WARNING*: ",
- L2181 *WARNING*: (payload)
- L2193 *WARNING*: ",
- L2197 *WARNING*: (payload)
- L2206 *WARNING*: ",
- L2210 *WARNING*: (payload)
- L2219 *WARNING*: ",
- L2223 *WARNING*: (payload)
- L2233 *WARNING*: ",
- L2237 *WARNING*: (payload)
- L2248 *WARNING*: ",
- L2252 *WARNING*: (payload)
- L2262 *WARNING*: ",
- L2266 *WARNING*: (payload)
- L2278 *WARNING*: ",
- L2282 *WARNING*: (payload)
- L2293 *WARNING*: ",
- L2297 *WARNING*: (payload)
- L2307 *WARNING*: ",
- L2311 *WARNING*: (payload)
- L2321 *WARNING*: ",
- L2325 *WARNING*: (payload)
- L2335 *WARNING*: ",
- L2339 *WARNING*: (payload)
- L2349 *WARNING*: ",
- L2353 *WARNING*: (payload)
- L2369 *WARNING*: ",
- L2373 *WARNING*: (payload)
- L2384 *WARNING*: ",
- L2388 *WARNING*: (payload)
- L2399 *WARNING*: ",
- L2403 *WARNING*: (payload)
- L2414 *WARNING*: ",
- L2418 *WARNING*: (payload)
- L2429 *WARNING*: ",
- L2433 *WARNING*: (payload)
- L2441 *WARNING*: ",
- L2445 *WARNING*: (payload)
- L2454 *WARNING*: ",
- L2458 *WARNING*: (payload)
- L2472 *WARNING*: ",
- L2476 *WARNING*: (payload)
- L2486 *WARNING*: ",
- L2490 *WARNING*: (payload)
- L2501 *WARNING*: ",
- L2505 *WARNING*: (payload)
- L2515 *WARNING*: ",
- L2519 *WARNING*: (payload)
- L2529 *WARNING*: ",
- L2533 *WARNING*: (payload)
- L2543 *WARNING*: ",
- L2547 *WARNING*: (payload)
- L2790 *WARNING*: ({"level": "WARNING", "type": "context", "message": msg})
- L2806 *WARNING*: ({"level": "WARNING", "type": "context", "message": msg})
- L2815 *WARNING*: ({"level": "WARNING", "type": "context", "message": msg})
- L2827 *WARNING*: ({"level": "WARNING", "type": "context", "message": msg})
- L2837 *WARNING*: ({"level": "WARNING", "type": "context", "message": msg})
- L2847 *WARNING*: ({"level": "WARNING", "type": "context", "message": msg})
- L2869 *WARNING*: ({"level": "WARNING", "type": "scan_html", "message": msg})
- L3297 *WARNING*: ({"level": "WARNING", "type": "dom_debug", "message": msg_warn})

### `webapp\parser\utils\model_registry.py` (Low Priority)

- L389 *WARNING*: (f"Failed loading local override for SentenceTransformer: {e}")
- L409 *WARNING*: ("TRANSFORMERS_OFFLINE/HUGGINGFACE_HUB_OFFLINE set; skipping HF download. Embeddings disabled.")
- L426 *WARNING*: for noisy environments
- L429 *WARNING*: (f"Failed to load base SentenceTransformer (network/DNS). Running without embeddings. Error: {e}")

### `webapp\parser\utils\output_utils.py` (Low Priority)

- L105 *WARNING*: ("\[yellow\]\[OUTPUT\] Year could not be verified. Using 'Unknown'.\[/yellow\]")
- L108 *WARNING*: ("\[yellow\]\[OUTPUT\] contests could not be verified. Using 'unknown_contests'.\[/yellow\]")
- L531 *WARNING*: (f"\[OUTPUT_UTILS\] Enrichment build failed: {e}")
- L607 *WARNING*: (f"\[OUTPUT_UTILS\] XLSX export failed: {e}")

### `webapp\parser\utils\pattern_extractor.py` (Low Priority)

- L26 *WARNING*: (f"\[PATTERN\] load fail {e}")
- L95 *WARNING*: (f"\[PATTERN\] pattern error {pat.get('name')}: {e}")

### `webapp\parser\utils\pivot.py` (Low Priority)

- L1353 *WARNING*: ("\[PIVOT\] No candidates detected – verify headers and candidate column extraction.")

### `webapp\parser\utils\shared_logger.py` (Low Priority)

- L159 *WARNING*:         elif record.levelno &gt;= logging.WARNING:
- L160 *WARNING*: (msg)
- L236 *WARNING*: ": logging.WARNING,
- L307 *WARNING*: ": "yellow",
- L369 *WARNING*: (self, msg, context=None, exc_info=None):
- L371 *WARNING*: ", msg, context, color="yellow")
- L385 *WARNING*: ": "yellow",
- L598 *WARNING*: (f"Log directory does not exist: {log_dir}")
- L615 *WARNING*: (f"Corrupt line in {path}: {e}")

### `webapp\parser\utils\shared_logic.py` (Low Priority)

- L236 *WARNING*: (f"\[safe_query\] session.query({model}) failed: {e}")
- L259 *WARNING*: (f"\[safe_filter_by\] No mapper found for model {model}")
- L265 *WARNING*: (f"\[safe_filter_by\] Could not inspect model {model}: {e}")
- L279 *WARNING*: (f"\[safe_filter_by\] filter_by failed: {e}")
- L292 *WARNING*: (f"\[safe_first\] query.first() failed: {e}")
- L362 *WARNING*: (f"\[PLUGIN EXTRACTION\] Plugin {plugin} has no callable 'extract' method.")
- L496 *WARNING*: (f"\[WARN\] Model save failed (attempt {attempt}): {e}")
- L710 *WARNING*: (f"\[safe_append\] Target is not a list: {type(lst)}; coercing to list.")
- L732 *WARNING*: (f"\[safe_update\] Target is not a dict: {type(dct)}")
- L736 *WARNING*: (f"\[safe_update\] Updates is not a dict: {type(updates)}")
- L756 *WARNING*: (f"\[safe_extend\] Target is not a list: {type(lst)}; coercing to list.")
- L1096 *WARNING*: (f"\[DOM_PARTS\] '{label}' is not a list for URL: {url} (type: {type(lst).__name__})")
- L1359 *WARNING*: (f"State '{state_norm}' not found in county map")
- L2137 *WARNING*: (f"\[inventory\] architecture.md not found at {md_file}")
- L2143 *WARNING*: ("\[inventory\] Markers not found in architecture.md; aborting replace.")
- L2158 *WARNING*: ("\[inventory\] generate_project_map completed with warnings; check markers and path.")
- L2701 *WARN*: ', 'WARNING', 'NOTE'\]
- L2784 *WARNING*: (f"\[noise\] No suggestions file found at {path}")

### `webapp\parser\utils\strategy_concurrency.py` (Low Priority)

- L37 *WARNING*: (f"\[CONCURRENCY\] DOM strategy {name} failed: {e}")
- L65 *WARNING*: (f"\[CONCURRENCY\] Strategy {name} error: {e}")
- L73 *WARNING*: (f"\[CONCURRENCY\] {_safe_run_strategy.__name__} {name} failed: {e}")
- L102 *WARNING*: (f"\[CONCURRENCY\]\[ASYNC\] DOM strategy {name} failed: {e}")
- L120 *WARNING*: (f"\[CONCURRENCY\]\[ASYNC\] Strategy {name} error: {e}")

### `webapp\parser\utils\table_builder.py` (Low Priority)

- L816 *WARNING*: ", "builder", "\[TABLE_BUILDER\] dynamic_table_extractor failed for panel table", session_id, error=str(e))
- L828 *WARNING*: ", "builder", "\[TABLE_BUILDER\] dynamic_table_extractor failed (no panels path)", session_id, error=str(e))
- L836 *WARNING*: ", "builder", "\[TABLE_BUILDER\] all_panel_tables was not a list; coercing to empty list", session_id, got_type=str(type(all_panel_tables)))
- L845 *WARNING*: ", "builder", "\[TABLE_BUILDER\] Dropping invalid table entry", session_id, entry_type=str(type(item)))
- L862 *WARNING*: ", "builder", "\[TABLE_BUILDER\] sanitize failed", session_id, error=str(e))
- L867 *WARNING*: ", "builder", "\[TABLE_BUILDER\] harmonize failed", session_id, error=str(e))
- L873 *WARNING*: ", "builder", "\[TABLE_BUILDER\] collapse_ballot_synonym_columns failed", session_id, error=str(e))
- L925 *WARNING*: ",
- L950 *WARNING*: ", "builder", "\[TABLE_BUILDER\] entity annotate failed", session_id, error=str(e))
- L955 *WARNING*: ", "builder", "\[TABLE_BUILDER\] stringify entity_info failed", session_id, error=str(e))
- L975 *WARNING*: ", "builder", "\[TABLE_BUILDER\] pivot_to_wide failed", session_id, error=str(e))
- L995 *WARNING*: ", "builder", "\[TABLE_BUILDER\] ensure division totals failed", session_id, error=str(e))
- L1288 *WARNING*: ", "builder", f"\[TABLE_BUILDER\] Column marked incorrect: {col_name}", session_id, contest=contest)
- L1361 *WARNING*: ", "builder", "\[TABLE_BUILDER\] Failed to persist table structure logs", session_id, error=str(e))
- L1376 *WARNING*: ", "builder", "\[TABLE_BUILDER\] Failed to persist coordinator DB log", session_id, error=str(e))

### `webapp\parser\utils\table_core.py` (Low Priority)

- L231 *WARNING*: (f"\[TABLE BUILDER\] Concurrent strategies execution failed: {e}")
- L288 *WARNING*: (f"\[TABLE BUILDER\] RawJSON pivot failed: {e}")
- L296 *WARNING*: (f"\[TABLE BUILDER\] pivot_to_wide signature mismatch (skipped): {e}")
- L298 *WARNING*: (f"\[TABLE BUILDER\] pivot_to_wide failed (skipped): {e}")
- L349 *WARNING*: (f"\[TABLE BUILDER\] finalize output failed: {e}")
- L414 *WARNING*: (f"\[TABLE BUILDER\]\[ASYNC\] Concurrent strategies execution failed: {e}")
- L477 *WARNING*: (f"\[TABLE BUILDER\]\[ASYNC\] finalize output failed: {e}")

### `webapp\parser\utils\user_prompt.py` (Low Priority)

- L312 *WARNING*: ("\[UserPrompt\] Webapp mode active but no socketio_emit_func set!")
- L349 *WARNING*: ("\[CLI Prompt\] EOFError encountered.")
- L370 *WARNING*: ("\[Webapp Prompt\] socketio_emit_func not set.")
- L428 *WARNING*: ": 30,
- L507 *WARNING*: ("\n\[Prompt\] Timed out.")
- L558 *WARNING*: ("\n\[Prompt\] No input available (EOF). Exiting prompt.")
- L592 *WARNING*: ("Invalid input. Please try again.")
- L594 *WARNING*: ("\[Prompt\] Too many invalid attempts.")
- L659 *WARNING*: ("\[Prompt Queue\] Invalid queued yes/no response; falling back to interactive prompt.")
- L674 *WARNING*: ("\n\[Prompt\] Timed out.")
- L881 *WARNING*: ("\[yellow\]\[FEEDBACK\] Skipped manual correction.\[/yellow\]")
- L913 *WARNING*: ("\[yellow\]Button confirmation cancelled by user.\[/yellow\]")

### `webapp\parser\web_pipeline.py` (Low Priority)

- L49 *WARNING*: ({
- L50 *WARNING*: ",
- L66 *WARNING*: ({
- L67 *WARNING*: ",
- L83 *WARNING*: ({
- L84 *WARNING*: ",

### `webapp\tests\conftest.py` (Low Priority)

- L17 *WARNING*: escalates to
- L21 *WARNING*: category so it never escalates.
- L23 *WARNING*: was ignored.

### `webapp\tests\test_schema_validation_warnings.py` (Low Priority)

- L17 *WARNING*:     # Build a table that lacks candidate/ballot/total columns, forcing a 'normalized schema weak' warning
- L39 *WARNING*: " and msg.get("status")=="weak"
- L44 *WARNING*: " and inner.get("status")=="weak"
- L49 *WARNING*: in captured logs; got: {captured}"

### `webapp\tests\test_webapp_app.py` (Low Priority)

- L17 *WARNING*: ,ERROR")
