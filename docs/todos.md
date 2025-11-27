# TODO/FIXME index — webapp

Total annotations: 519

## `webapp\Smart_Elections_Parser_Webapp.py`

- L210: WEBAPP_CONSOLE_LEVELS = set(os.environ.get("WEBAPP_CONSOLE_LEVELS", "ERROR,WARNING").upper().split(","))
- L480:     Levels: INFO, DEBUG, WARNING, ERROR, CRITICAL, TRACE
- L519:     LEVELS = {"INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL", "TRACE"}
- L555:             elif "warning" in mlow:
- L954:         # For websocket handshake only: add Cache-Control so webhint stops warning
- L1229:         logger.warning({"type": "sec", "message": "Favicon path escape blocked", "requested": ico_path})
- L1331:             logger.warning({
- L1332:                 "level": "WARNING",
- L1641:             "level": "WARNING",
- L1724:         logger.warning(
- L1726:                 "level": "WARNING",
- L1736:         logger.warning(
- L1738:                 "level": "WARNING",
- L1768:         logger.warning(
- L1770:                 "level": "WARNING",
- L2055:         logger.warning({
- L2056:             "level": "WARNING",
- L2120:         logger.warning({
- L2121:             "level": "WARNING",
- L2171:             logger.warning({
- L2172:                 "level": "WARNING",
- L2194:                 logger.warning({
- L2195:                     "level": "WARNING",
- L2203:         logger.warning({
- L2204:             "level": "WARNING",
- L2211:         logger.warning({
- L2212:             "level": "WARNING",

## `webapp\parser\Context_Integration\Context_Library\constants.py`

- L2020:         "icon-bg-dark", "icon-bg-primary", "icon-bg-secondary", "icon-bg-success", "icon-bg-danger", "icon-bg-warning",
- L2111:     "warning", "info_box", "navigation", "pagination", "tab", "modal", "tooltip", "ignore", "unknown"

## `webapp\parser\Context_Integration\context_coordinator.py`

- L788:                     logger.warning("\[ALERT MONITOR\] Thread did not stop cleanly.")
- L876:             logger.warning({
- L877:                 "level": "WARNING",
- L995:             logger.warning(f"\[yellow\]Integrity issues:\[/yellow\] {issues\['integrity_issues'\]}")
- L1234:                 logger.warning(f"\[ContextCoordinator\] No table structure found for contest: {contest}")
- L1403:                     logger.warning(f"\[get_feedback_pattern_kb\] Skipping corrupt line: {e}")
- L1515:                 logger.warning("\[group_dom_nodes_by_label\] No organized DOM parts. (Further warnings suppressed)")
- L1517:                 logger.warning(f"\[group_dom_nodes_by_label\] No organized DOM parts. (Occurred {ContextCoordinator._dom_parts_warning_count} times)")
- L1522:             logger.warning("\[group_dom_nodes_by_label\] No DOM nodes found.")
- L1540:                 logger.warning("\[submit_user_feedback\] ContextOrganizer has no submit_user_feedback method.")
- L1568:                 logger.warning(f"\[correct_and_update_contest\] Contest {contest_id} missing type/election_types after sync.")
- L1592:             logger.warning("\[print_contest_summary\] No organized contests to summarize.")
- L1605:             logger.warning("\[plot_contest_distribution\] No organized contests to plot.")
- L1656:                 logger.warning("No organized DOM parts.")
- L1659:                 logger.warning("No organized DOM parts. (Further warnings suppressed)")
- L1670:             logger.warning("\[get_contest_groups\] No contest groups found.")
- L1679:             logger.warning("\[get_panel_groups\] No panel groups found.")
- L1688:             logger.warning("\[get_button_groups\] No button groups found.")
- L1697:             logger.warning("\[get_table_groups\] No table groups found.")
- L1706:             logger.warning("\[get_relationships\] No organized context.")
- L1814:                 logger.warning(f"\[fuzzy_score\] One or both inputs are empty: a='{a_str}', b='{b_str}'")
- L1820:                 logger.warning(f"\[fuzzy_score\] One or both inputs are too short: a='{a_str}', b='{b_str}'")
- L2266:             logger.warning(f"\[extract_field\] Unknown field_type: {field_type}")
- L2524:                     logger.warning(f"\[get_full_contest\] Contest {contest_id} missing type/election_types after sync.")
- L2609:                         logger.warning(f"\[list_tables\] Table '{tbl}' missing metadata or columns.")
- L2641:                 logger.warning(f"\[get_table_metadata\] Table '{table_name}' missing columns.")
- L2659:                 logger.warning(f"\[check_missing_tables\] Missing tables: {missing}")
- L2720:                 logger.warning(f"\[save_table_structure\] Failed to save structure for contest: {contest}")
- L2897:             logger.warning(f"\[get_best_button_advanced\] Contest argument was not a dict. Converted to: {contest}")
- L2901:             logger.warning(f"\[get_best_button_advanced\] Keywords argument was not a list. Converted to: {keywords}")
- L2905:             logger.warning(f"\[get_best_button_advanced\] Context argument was not a dict. Converted to: {context}")
- L2912:             logger.warning("\[get_best_button_advanced\]_semantic_model is not set or is not an object. Using None.")
- L3057:                             logger.warning(f"\[yellow\]\[Coordinator\] Button '{cand.get('label')}' rejected, retrying...\[/yellow\]")

## `webapp\parser\Context_Integration\context_organizer.py`

- L282:             logger.warning(
- L407:                 logger.warning(f"\[CONTEST\] Skipping contest with suspiciously large or missing title: {str(title)\[:100\]}...")
- L495:             logger.warning(f"\[CONTEST\] Filtered out {len(filtered_out)} contests due to missing required fields.")
- L497:                 logger.warning(f"  \[Filtered\] {reason}: {str(c)\[:100\]}...")
- L500:             logger.warning("\[CONTEST\] No contests with required fields for downstream output.")
- L816:                             logger.warning(f"\[ML\] Anomaly index {idx} out of range for contests list of length {len(contests)}")
- L1500:                     logger.warning(f"  \[yellow\]{title}\[/yellow\]: {fixes}")
- L1505:                     logger.warning(f"\[bold yellow\]\[INTEGRITY\]\[/bold yellow\] Duplicate contest detected.\n  \[dim\]Context:\[/dim\] {contest}")
- L1507:                     logger.warning(f"\[bold yellow\]\[INTEGRITY\]\[/bold yellow\] Contest missing location info.\n  \[dim\]Context:\[/dim\] {contest}")
- L1509:                     logger.warning(f"\[bold yellow\]\[INTEGRITY\]\[/bold yellow\] Contest missing year.\n  \[dim\]Context:\[/dim\] {contest}")
- L1972:             logger.warning(f"\[ContextOrganizer\] Could not update context library with feedback: {e}")
- L2049:                 logger.warning(f"\[CONTEXT ORGANIZER\] No table structure found for contest: {contest}")

## `webapp\parser\Context_Integration\librarian.py`

- L652:         logger.warning(f"\n\[LIBRARIAN SELF-HEAL\] Attempt {attempt}...")
- L658:         logger.warning("\[LIBRARIAN SELF-HEAL\] Misalignments found. Launching manual_correction...")
- L661:         logger.warning(f"\[LIBRARIAN SELF-HEAL\] Sleeping {cooldown}s before rescanning...")

## `webapp\parser\config.py`

- L328:                 logger.warning("\[DB\]\[AAD\] Falling back to password auth.")

## `webapp\parser\data_manager.py`

- L83:             logger.warning(f"\[REMOVED\] {popped}")
- L90:             logger.warning(f"\[REMOVED\] {index_or_value}")
- L129:                     logger.warning(f"\[DELETED\] {files\[idx\]}")

## `webapp\parser\handlers\batch_handler.py`

- L134:                 logger.warning({
- L135:                     "level": "WARNING",
- L426:             logger.warning({
- L427:                 "level": "WARNING",

## `webapp\parser\handlers\formats\html_handler.py`

- L216:                 app_logger.warning(f"\[HTML Handler\] County '{county}' not found. Closest matches: {matches}")
- L220:                 app_logger.warning(f"\[HTML Handler\] Detected county '{county}' is not in known counties for state '{suggested_state or state}'.")
- L241:                     app_logger.warning(f"\[HTML Handler\] State '{user_state}' not found. Closest matches: {matches}")
- L285:                         app_logger.warning(f"\[HTML Handler\] County '{user_county}' not found. Closest matches: {matches}")

## `webapp\parser\handlers\formats\json_handler.py`

- L376:         logger.warning({
- L377:             "level": "WARNING",
- L489:         logger.warning({
- L490:             "level": "WARNING",

## `webapp\parser\handlers\formats\pdf_handler.py`

- L421:         logger.warning({
- L422:             "level": "WARNING",
- L425:                 "\[WARN\] Detected PyMuPDF %s. Upgrade to %s or newer to avoid parser instability."
- L1787:                     logger.warning({
- L1788:                         "level": "WARNING",
- L1790:                         "message": "\[WARN\] Poppler binaries not detected; skipping pdf2image and using PyMuPDF fallback.",
- L1808:             logger.warning({
- L1809:                 "level": "WARNING",
- L1812:                     "\[WARN\] pdf2image conversion failed; "
- L2184:         logger.warning({
- L2185:             "level": "WARNING",
- L2187:             "message": f"\[WARN\] Multi-mode text extraction failed: {e}",
- L3283:         logger.warning({
- L3284:             "level": "WARNING",
- L3286:             "message": f"\[WARN\] fitz text extraction failed: {e}",
- L3315:         logger.warning({
- L3316:             "level": "WARNING",
- L3318:             "message": "\[WARN\] ENABLE_OCR_FORCE is set but Tesseract is unavailable; skipping OCR fallback.",
- L3366:             logger.warning({
- L3367:                 "level": "WARNING",
- L3369:                 "message": "\[WARN\] Low-signal text detected but OCR is unavailable or disabled.",
- L3586:         logger.warning({
- L3587:             "level": "WARNING",
- L3589:             "message": "\[WARN\] No contest selected. Using filename fallback.",
- L4034:                 logger.warning({
- L4035:                     "level": "WARNING",
- L4037:                     "message": f"\[WARN\] Selected contest '{contest}' not found in column '{contest_column}'. Skipping row filter.",
- L4136:             logger.warning({
- L4137:                 "level": "WARNING",
- L4139:                 "message": f"\[WARN\] No structured rows matched the inferred column count of {len(headers)}. Total lines scanned: {unmatched_count}",
- L4178:             logger.warning({
- L4179:                 "level": "WARNING",
- L4367:     logger.warning({
- L4368:         "level": "WARNING",

## `webapp\parser\handlers\states\arizona\arizona.py`

- L25:     logger.warning("\[WARN\] context_library.json not found. Using fallback config for Arizona handler.")
- L51:                 logger.warning(f"\[WARN\] Could not expand card {i+1}: {e}")
- L64:             logger.warning(f"\[WARN\] Vote Type toggle failed: {e}")
- L77:             logger.warning(f"\[WARN\] County toggle failed: {e}")
- L164:         logger.warning("\[FALLBACK\] No tables were parsed. Either no results are published yet or the structure has changed.")
- L165:         logger.warning("\[FALLBACK\] Please verify that the site has posted election data.")

## `webapp\parser\handlers\states\example state\example_county\example_county.py`

- L123:         logger.warning("\[yellow\]\[WARNING\] No ballot items found by div selectors. Trying table-based extraction...\[/yellow\]")

## `webapp\parser\handlers\states\example state\example_state.py`

- L51:             logger.warning(f"\[Example Handler\] No specific parser implemented for county: '{county}'. Continuing with state-level logic.")
- L152:         logger.warning("\[yellow\]\[WARNING\] No ballot items found by div selectors. Trying table-based extraction...\[/yellow\]")

## `webapp\parser\handlers\states\new_york\county\rockland.py`

- L72:         logger.warning("\[WARNING\] dom_parts missing after organize_and_enrich.")
- L95:         logger.warning("\[red\]No contest selected. Skipping.\[/red\]")
- L139:                         logger.warning(f"\[yellow\]\[WARNING\] Button '{btn1.get('label', '')}' is not clickable (visible={safe_is_visible(element, logger)}, enabled={safe_is_enabled(element, logger)})\[/yellow\]")
- L176:                         logger.warning(f"\[yellow\]\[WARNING\] Button '{btn2.get('label', '')}' is not clickable (visible={safe_is_visible(element, logger)}, enabled={safe_is_enabled(element, logger)})\[/yellow\]")

## `webapp\parser\handlers\states\new_york\new_york.py`

- L27:         logger.warning("\[NY Handler\] No county specified in html_context.")
- L43:         logger.warning(f"\[NY Handler\] No specific parser implemented for county: '{county}'. Please add it under {module_path}.py")

## `webapp\parser\handlers\states\pennsylvania\pennsylvania.py`

- L44:             logger.warning(f"\[NAV\] Step failed: {step} — {e}")
- L55:     logger.warning(f"\[bold yellow\]Detected election:\[/bold yellow\] {header_text}")
- L76:                     logger.warning("\[PA\] Invalid index input for election selection.")
- L78:                 logger.warning("\[PA\] Elections dropdown not found.")
- L80:             logger.warning(f"\[PA\] Failed to expand Elections menu or load selection: {e}")
- L96:                 logger.warning("\[PA\] County Breakdown link not found.")
- L98:             logger.warning(f"\[PA\] Failed to click County Breakdown link: {e}")
- L113:         logger.warning("\[yellow\]Multiple CSV files found in input. Please select one:\[/yellow\]")

## `webapp\parser\health\health_router.py`

- L252:                     logger.warning(f"\[health_router\] manual_correction failed (attempt {attempt}): {result.stderr}")
- L336:             logger.warning("\[SELF-HEAL\] Misalignments found. Launching manual_correction...")
- L338:             logger.warning(f"\[SELF-HEAL\] Sleeping {cooldown}s before rescanning...")
- L340:         logger.warning("\[SELF-HEAL\] Max retries reached. Some misalignments may remain.")
- L375:                 logger.warning(f"\[PIPELINE\] Could not fix corrupted JSON files: {e}")
- L380:                 logger.warning("\[PIPELINE\] Misaligned NER examples found. Self-heal loop will be handled by scan_misaligned_ner.")
- L382:                 logger.warning("\[PIPELINE\] scan_misaligned_ner failed or file missing. Proceeding with caution.")
- L414:                 logger.warning("\[PIPELINE\] Model retraining failed.")

## `webapp\parser\health\log_cache_cleaner_bot.py`

- L151:                     logger.warning(f"Skipping non-dict entry in spacy_ner_train_data.jsonl: {entry}")
- L460:                 logger.warning("\[DB\]\[WARNING\] No user tables found in schema 'public'.")
- L503:         logger.warning("\[CLEAN\]\[WARNING\] The following files are still too large after cleaning:")
- L507:         logger.warning("\[MISALIGNED\] Consider cleaning or pattern-excluding these from your training data:")

## `webapp\parser\health\manual_correction_bot.py`

- L322:             logger.warning(f"Coordinator ML scoring failed: {e}")
- L343:             logger.warning(f"Coordinator field suggestion failed: {e}")
- L355:         logger.warning(f"Log file not found: {path}")
- L364:                     logger.warning(f"\[CORRUPT\] {path} line {i}: {e}")
- L396:                             logger.warning(f"\[SKIP\] File not found: {file}")
- L400:                             logger.warning(f"\[SKIP\] File too large: {file}")
- L422:                                         logger.warning(f"\[CORRUPT-LINE\] {file} line {i+1}: {line\[:80\]}... ({e})")
- L434:                                 logger.warning(f"\[CORRUPT\] {len(corrupt_items)} lines saved to {corrupt_path}")
- L439:                                 logger.warning(f"\[FIXED\] All lines invalid, recreated empty .jsonl file: {file}")
- L453:                                 logger.warning(f"\[CORRUPT\] {file}: {e}")
- L465:                                 logger.warning(f"\[CORRUPT\] Corrupt JSON saved to {corrupt_path}")
- L471:                                 logger.warning(f"\[FIXED\] All content invalid, recreated minimal valid JSON in {file}")
- L476:                         logger.warning(f"\[CORRUPT\] {file}: {e}")
- L485:                                         logger.warning(f"\[QUARANTINED\] {file} -&gt; {quarantine_dir / file.name}")
- L489:                                         logger.warning(f"\[DELETED\] {file}")
- L492:                                     logger.warning(f"\[SKIP-DELETE\] File already missing: {file}")
- L537:             logger.warning(f"\[FIND-LOGS\] Skipped {d}: {e}")
- L562:                     logger.warning(f"\[CORRUPT\] {path} line {line_num}: {e}")
- L717:                     logger.warning(f"Invalid JSON, skipping edit: {e}")
- L750:     # TODO: Add JSON schema validation here if desired
- L989:         logger.warning(
- L1079:     If missing, create with DEFAULT_STRUCTURE. Warn if schema version mismatches.
- L1098:         logger.warning(f"Schema version mismatch: found {context_lib.get('schema_version')}, expected {SCHEMA_VERSION}. Consider migrating.")
- L1141:                     logger.warning(f"\[AUTO\] Could not delete log file {log_file}: {e}")
- L1257:             logger.warning(f"\[SKIP\] Could not load {log_file}: {e}")
- L1273:         logger.warning("No log files matched any of the specified fields. Will attempt to process all log files for all fields.")
- L1356:                 logger.warning(f"Could not delete log file {log_file}: {e}")
- L1376:         logger.warning("\[WARNING\] No entries were processed. Check your log file naming, field configuration, or use --dry-run for debugging.")

## `webapp\parser\health\retrain_table_structure_models.py`

- L178:         logger.warning(f"\[CLEAN\] File not found: {jsonl_path}")
- L186:                 logger.warning(f"\[CLEAN\] Could not parse line: {e}")
- L201:                 logger.warning(f"\[CLEAN\] Alignment check failed for text: {text\[:50\]}... ({e})")
- L274:             logger.warning(f"Failed to load {path}: {e}")
- L403:                     logger.warning(f"Skipping misaligned entity in: {text}")
- L408:                 logger.warning(f"Error validating entity alignment: {e}")
- L434:         logger.warning(f"\[spaCy\] Could not check GPU availability: {e}")
- L450:         logger.warning(f"\[spaCy\] Could not load lexeme normalization table. You may ignore this for English. Error: {e}")
- L536:         logger.warning(f"\[NER\] Skipped {misaligned_count} misaligned examples. Saved to {misaligned_path}")
- L550:         logger.warning("No NER training examples found. Skipping spaCy NER retraining.")
- L619:         logger.warning("\[SUGGESTION\] Consider lowering min_delta or increasing patience if you want longer training.")
- L621:         logger.warning("\[SUGGESTION\] Model improved until the last epoch. Consider increasing epochs for further improvement.")
- L622:     logger.warning(f"\[SUGGESTION\] Next run: patience={patience}, min_delta={min_delta:.2f}, epochs={epochs}")
- L708:         logger.warning("No training examples found. Aborting retraining.")
- L727:             logger.warning(f"\[WARN\] Could not delete old model directory {oldest_path}: {e}")
- L739:             logger.warning(f"\[WARN\] Failed to load existing model: {e}")
- L742:         logger.warning("Falling back to base model (all-MiniLM-L6-v2).")
- L782:             logger.warning(f"\[WARN\] Could not update canonical model directory: {e}")
- L810:                     logger.warning(f"MISALIGNED: {text} {annots\['entities'\]}")
- L840:             logger.warning("\[DB\] Base.metadata.tables is empty. No models registered? Did you import all model classes?")

## `webapp\parser\health\scan_misaligned_ner.py`

- L62:                     logger.warning(f"\[CORRUPT\] Could not parse line: {e}")
- L83:             logger.warning(f"\n\[MISALIGNED\] Top {top_n} most frequent misaligned NER texts:")
- L85:                 logger.warning(f"  {repr(text)}: {count} times")
- L86:             logger.warning("\[MISALIGNED\] Consider cleaning or pattern-excluding these from your training data.")
- L87:         logger.warning("Run the manual_correction to review and clean these examples before retraining.")
- L88:         logger.warning("If you see spaCy entity alignment warnings, consider cleaning your training data or using the provided validation function.")
- L98:                 logger.warning(f"\[WARN\] Could not remove old misaligned file: {e}")
- L112:         logger.warning("\[SELF-HEAL\] Misalignments found. Launching manual_correction for spacy_ner_misaligned...")
- L119:             logger.warning(f"\[SELF-HEAL\] manual_correction exited with code {result.returncode}")
- L120:         logger.warning(f"\[SELF-HEAL\] Sleeping {cooldown}s before rescanning...")
- L122:     logger.warning("\[SELF-HEAL\] Max retries reached. Some misalignments may remain.")

## `webapp\parser\html_election_parser.py`

- L56:     logger.warning("Deleting .processed_urls cache for fresh start...")
- L393:                     logger.warning({
- L394:                         "level": "WARNING",
- L408:             logger.warning({
- L409:                 "level": "WARNING",
- L469:                 logger.warning({
- L470:                     "level": "WARNING",
- L543:                 logger.warning(payload_2)
- L870:                     logger.warning({
- L871:                         "level": "WARNING",
- L917:         logger.warning({
- L918:             "level": "WARNING",
- L971:         logger.warning({
- L972:             "level": "WARNING",
- L1076:                         "level": "WARNING",
- L1081:                     logger.warning(payload)
- L1106:                 # Soft-fail: continue; downstream will warn if nothing found
- L1166:                     "level": "WARNING",
- L1171:                 logger.warning(payload)
- L1249:                                 logger.warning({
- L1250:                                     "level": "WARNING",
- L1267:                             "level": "WARNING",
- L1272:                         logger.warning(payload)
- L1283:                             "level": "WARNING",
- L1288:                         logger.warning(payload)
- L1290:                         msg = "\[WARN\] No output file path returned from parser and no output files found."
- L1292:                             "level": "WARNING",
- L1297:                         logger.warning(payload)
- L1302:                     "level": "WARNING",
- L1307:                 logger.warning(payload)
- L1425:                     logger.warning({
- L1426:                         "level": "WARNING",
- L1486:         logger.warning({
- L1487:             "level": "WARNING",

## `webapp\parser\state_router.py`

- L49:         logger.warning("\[Router\] handlers/states directory not found.")
- L66:             logger.warning(f"\[Router\] counties directory not found for state: {state_key}")
- L137:         logger.warning(f"\[Fallback\]\[Session:{session_id}\] No handler states available for manual selection.")
- L154:             logger.warning(f"\[Fallback\]\[Session:{session_id}\] Aborted by user.")
- L157:             logger.warning(f"\[Fallback\]\[Session:{session_id}\] Aborted by user.")
- L160:             logger.warning(f"\[Fallback\]\[Session:{session_id}\] State '{state}' not found. Please try again.")
- L179:                 logger.warning(f"\[Fallback\]\[Session:{session_id}\] Aborted by user.")
- L182:                 logger.warning(f"\[Fallback\]\[Session:{session_id}\] County '{county}' not found for state '{state}'. Please try again.")
- L189:     logger.warning(f"\[Fallback\]\[Session:{session_id}\] Too many failed attempts. Exiting fallback.")
- L205:                 logger.warning(f"\[Router\] Requested state '{state_name}' not found on disk. Skipping restrict filter.")
- L512:             logger.warning(f"No counties found for state '{state}'. Try --fuzzy for fuzzy matching.")
- L523:                     logger.warning(f"Failed to load context from file: {e}")
- L533:             logger.warning("No suitable handler found.")
- L540:                 logger.warning("No handler selected. Exiting.")
- L547:                 logger.warning("Still could not import a suitable handler.")

## `webapp\parser\utils\browser_utils.py`

- L89:                     logger.warning(f"\[browser_utils\] Failed to safely parse context_library value for key '{key}'")
- L91:                 logger.warning(f"\[browser_utils\] Skipping unsafe context_library value for key '{key}'")
- L295:                     logger.warning(f"\[safe_attributes\] Playwright JS extraction failed: {e}")
- L309:                 logger.warning(f"\[safe_attributes\] Playwright fallback extraction failed: {e}")
- L395:         logger and logger.warning(f"\[safe_count\] Object is not countable: {type(obj)}")
- L441:             logger.warning(f"\[safe_launch\] browser_type is not a SyncBrowserType: {type(browser_type)}")
- L461:             logger.warning(f"\[async_safe_launch\] browser_type is not an AsyncBrowserType: {type(browser_type)}")
- L540:             logger.warning({
- L541:                 "level": "WARNING",
- L569:             logger.warning(f"\[CAPTCHA\] Detected Cloudflare CAPTCHA indicator: '{indicator}'")
- L578:     logger.warning(f"\[CAPTCHA\] CAPTCHA detected in async mode. Manual intervention not implemented. (Session: {session_id})")
- L602:             logger.warning(f"\[CAPTCHA\] Detected Cloudflare CAPTCHA indicator: '{indicator}'")
- L611:             logger.warning({
- L612:                 "level": "WARNING",
- L623:     logger.warning(f"\[CAPTCHA\] CAPTCHA detected in sync mode. Manual intervention not implemented. (Session: {session_id})")
- L712:                         logger and logger.warning("\[SCROLL\] User aborted scrolling.")
- L733:         logger and logger.warning("\[SCROLL\] Max scroll time/attempts exceeded. Page may not be fully loaded.")

## `webapp\parser\utils\captcha_tools.py`

- L118:         logger.warning(f"\[CAPTCHA\] Foreground window fallback failed: {e}")
- L154:     logger.warning("\[CAPTCHA\] CAPTCHA not resolved within timeout.")

## `webapp\parser\utils\contest_selector.py`

- L635:     elif lvl == "warning":
- L636:         logger.warning(entry)
- L1029: _log("warning", "selector", f"Feedback loop {loop+1}: verifying contests", session_id=session_id,
- L1565:                 logger.warning({"level": "WARNING", "type": "selector", "message": "Empty search term", "session_id": session_id})
- L1570:                 logger.warning({"level": "WARNING", "type": "selector", "message": f"No matches for '{term}'", "session_id": session_id})
- L1642:         logger.warning({"level": "WARNING", "type": "selector", "message": "No match; try again.", "session_id": session_id})

## `webapp\parser\utils\dom_extractor.py`

- L153:         logger.warning(f"\[DOM_EXTRACTOR\] failure: {e}")

## `webapp\parser\utils\dynamic_table_extractor.py`

- L124: _emit("warning", "extractor", "\[EXTRACTOR\] No &lt;table&gt; found in provided table_html.", session_id)
- L129: _emit("warning", "extractor", "\[EXTRACTOR\] No &lt;tr&gt; rows found in table_html.", session_id)
- L171: _emit("warning", "extractor", "\[EXTRACTOR\] Candidate NLP/score step failed", session_id, error=str(e))
- L187: _emit("warning", "extractor", "\[EXTRACTOR\] No suitable table candidates found.", session_id)
- L217: _emit("warning", "extractor", "\[EXTRACTOR\] Error while scanning &lt;table&gt; elements", session_id, error=str(e))
- L229: _emit("warning", "extractor", "\[EXTRACTOR\] DOM extraction failed", session_id, error=str(e))
- L272: _emit("warning", "extractor", "\[EXTRACTOR\] Pattern extraction failed", session_id, error=str(e))
- L776: _emit("warning", "extractor", "No learned DOM patterns found.")
- L800: _emit("warning", "extractor", "Entry deleted.")
- L805: _emit("warning", "extractor", "Unknown action.")
- L807: _emit("warning", "extractor", "Invalid entry number.")

## `webapp\parser\utils\embedding_cache.py`

- L178:                 logger.warning(msg)

## `webapp\parser\utils\extraction_strategies.py`

- L68:             logger.warning(f"\[STRATEGY\] {name} failed: {e}")

## `webapp\parser\utils\format_router.py`

- L374:         logger.warning({
- L375:             "level": "WARNING",
- L377:             "message": "\[WARN\] No supported file formats found on the page.",
- L402:         logger.warning({
- L403:             "level": "WARNING",
- L405:             "message": f"\[WARN\] Unsupported format requested: {format_str}",
- L409:         logger.warning({
- L410:             "level": "WARNING",
- L654:         logger.warning({
- L655:             "level": "WARNING",
- L874:             logger.warning({
- L875:                 "level": "WARNING",
- L950:         logger.warning({
- L951:             "level": "WARNING",

## `webapp\parser\utils\html_scanner.py`

- L163:                 "level": "WARNING",
- L167:             logger.warning(payload)
- L189:                             "level": "WARNING",
- L193:                         logger.warning(payload)
- L288:                 "level": "WARNING",
- L292:             logger.warning(payload)
- L315:                             "level": "WARNING",
- L319:                         logger.warning(payload)
- L353:                 "level": "WARNING",
- L357:             logger.warning(payload)
- L380:                             "level": "WARNING",
- L384:                         logger.warning(payload)
- L579:                     "level": "WARNING",
- L583:                 logger.warning(payload)
- L784:                 logger.warning(f"\[ML SIMILARITY\] No embedding computed for segment: {safe_get(segment, 'segment_hash', None)}")
- L807:                 logger.warning(f"\[ML SIMILARITY\] No embedding computed for segment: {safe_get(segment, 'segment_hash', None)}")
- L1034:                     "level": "WARNING",
- L1038:                 logger.warning(payload)
- L1045:                     "level": "WARNING",
- L1049:                 logger.warning(payload)
- L1376:                         "level": "WARNING",
- L1380:                     logger.warning(payload)
- L1438:                         "level": "WARNING",
- L1442:                     logger.warning(payload)
- L1691:                 logger.warning({"level": "WARNING", "type": "dom_segments", "message": msg_warn})
- L1747:                     logger.warning({"level": "WARNING", "type": "page_hash", "message": msg})
- L1754:                 logger.warning({"level": "WARNING", "type": "page_hash", "message": msg})
- L1766:                 logger.warning({"level": "WARNING", "type": "page_hash", "message": msg})
- L1789:             logger.warning({"level": "WARNING", "type": "cache", "message": msg})
- L1824:             logger.warning({"level": "WARNING", "type": "cache", "message": msg})
- L2003:         logger.warning({"level": "WARNING", "type": "segment_review", "message": msg})
- L2012:             else logger.warning({
- L2013:                 "level": "WARNING",
- L2129:                         "level": "WARNING",
- L2133:                     logger.warning(payload)
- L2145:                         "level": "WARNING",
- L2149:                     logger.warning(payload)
- L2158:                         "level": "WARNING",
- L2162:                     logger.warning(payload)
- L2177:                             "level": "WARNING",
- L2181:                         logger.warning(payload)
- L2193:                                 "level": "WARNING",
- L2197:                             logger.warning(payload)
- L2206:                                 "level": "WARNING",
- L2210:                             logger.warning(payload)
- L2219:                                 "level": "WARNING",
- L2223:                             logger.warning(payload)
- L2233:                                     "level": "WARNING",
- L2237:                                 logger.warning(payload)
- L2248:                                         "level": "WARNING",
- L2252:                                     logger.warning(payload)
- L2262:                                     "level": "WARNING",
- L2266:                                 logger.warning(payload)
- L2278:                                     "level": "WARNING",
- L2282:                                 logger.warning(payload)
- L2293:                                     "level": "WARNING",
- L2297:                                 logger.warning(payload)
- L2307:                                     "level": "WARNING",
- L2311:                                 logger.warning(payload)
- L2321:                                     "level": "WARNING",
- L2325:                                 logger.warning(payload)
- L2335:                                     "level": "WARNING",
- L2339:                                 logger.warning(payload)
- L2349:                                     "level": "WARNING",
- L2353:                                 logger.warning(payload)
- L2369:                                                     "level": "WARNING",
- L2373:                                                 logger.warning(payload)
- L2384:                                     "level": "WARNING",
- L2388:                                 logger.warning(payload)
- L2399:                                     "level": "WARNING",
- L2403:                                 logger.warning(payload)
- L2414:                                     "level": "WARNING",
- L2418:                                 logger.warning(payload)
- L2429:                                     "level": "WARNING",
- L2433:                                 logger.warning(payload)
- L2441:                 "level": "WARNING",
- L2445:             logger.warning(payload)
- L2454:                     "level": "WARNING",
- L2458:                 logger.warning(payload)
- L2472:                             "level": "WARNING",
- L2476:                         logger.warning(payload)
- L2486:                     "level": "WARNING",
- L2490:                 logger.warning(payload)
- L2501:                         "level": "WARNING",
- L2505:                     logger.warning(payload)
- L2515:                     "level": "WARNING",
- L2519:                 logger.warning(payload)
- L2529:                     "level": "WARNING",
- L2533:                 logger.warning(payload)
- L2543:                     "level": "WARNING",
- L2547:                 logger.warning(payload)
- L2790:                 logger.warning({"level": "WARNING", "type": "context", "message": msg})
- L2806:                 logger.warning({"level": "WARNING", "type": "context", "message": msg})
- L2815:                 logger.warning({"level": "WARNING", "type": "context", "message": msg})
- L2827:                     logger.warning({"level": "WARNING", "type": "context", "message": msg})
- L2837:                     logger.warning({"level": "WARNING", "type": "context", "message": msg})
- L2847:                     logger.warning({"level": "WARNING", "type": "context", "message": msg})
- L2869:             logger.warning({"level": "WARNING", "type": "scan_html", "message": msg})
- L3297:                     logger.warning({"level": "WARNING", "type": "dom_debug", "message": msg_warn})

## `webapp\parser\utils\model_registry.py`

- L389:                     logger.warning(f"Failed loading local override for SentenceTransformer: {e}")
- L409:                 logger.warning("TRANSFORMERS_OFFLINE/HUGGINGFACE_HUB_OFFLINE set; skipping HF download. Embeddings disabled.")
- L426:                 # Downgrade DNS/network errors to WARNING for noisy environments
- L429:                     logger.warning(f"Failed to load base SentenceTransformer (network/DNS). Running without embeddings. Error: {e}")

## `webapp\parser\utils\output_utils.py`

- L105:         logger.warning("\[yellow\]\[OUTPUT\] Year could not be verified. Using 'Unknown'.\[/yellow\]")
- L108:         logger.warning("\[yellow\]\[OUTPUT\] contests could not be verified. Using 'unknown_contests'.\[/yellow\]")
- L531:         logger.warning(f"\[OUTPUT_UTILS\] Enrichment build failed: {e}")
- L607:         logger.warning(f"\[OUTPUT_UTILS\] XLSX export failed: {e}")

## `webapp\parser\utils\pattern_extractor.py`

- L26:         logger.warning(f"\[PATTERN\] load fail {e}")
- L95:             logger.warning(f"\[PATTERN\] pattern error {pat.get('name')}: {e}")

## `webapp\parser\utils\pivot.py`

- L1353:         logger.warning("\[PIVOT\] No candidates detected – verify headers and candidate column extraction.")

## `webapp\parser\utils\shared_logger.py`

- L159:         elif record.levelno &gt;= logging.WARNING:
- L160:             self.shared_logger.warning(msg)
- L236:             "WARNING": logging.WARNING,
- L307:                 "WARNING": "yellow",
- L369:     def warning(self, msg, context=None, exc_info=None):
- L371:         self._log("WARNING", msg, context, color="yellow")
- L385:             "warning": "yellow",
- L598:                 self.warning(f"Log directory does not exist: {log_dir}")
- L615:                         self.warning(f"Corrupt line in {path}: {e}")

## `webapp\parser\utils\shared_logic.py`

- L236:         logger.warning(f"\[safe_query\] session.query({model}) failed: {e}")
- L259:             logger.warning(f"\[safe_filter_by\] No mapper found for model {model}")
- L265:         logger.warning(f"\[safe_filter_by\] Could not inspect model {model}: {e}")
- L279:         logger.warning(f"\[safe_filter_by\] filter_by failed: {e}")
- L292:         logger.warning(f"\[safe_first\] query.first() failed: {e}")
- L362:             logger.warning(f"\[PLUGIN EXTRACTION\] Plugin {plugin} has no callable 'extract' method.")
- L496:                 logger.warning(f"\[WARN\] Model save failed (attempt {attempt}): {e}")
- L710:                     logger.warning(f"\[safe_append\] Target is not a list: {type(lst)}; coercing to list.")
- L732:             logger.warning(f"\[safe_update\] Target is not a dict: {type(dct)}")
- L736:             logger.warning(f"\[safe_update\] Updates is not a dict: {type(updates)}")
- L756:                     logger.warning(f"\[safe_extend\] Target is not a list: {type(lst)}; coercing to list.")
- L1096:         logger.warning(f"\[DOM_PARTS\] '{label}' is not a list for URL: {url} (type: {type(lst).__name__})")
- L1359:             logger.warning(f"State '{state_norm}' not found in county map")
- L2137:             logger.warning(f"\[inventory\] architecture.md not found at {md_file}")
- L2143:             logger.warning("\[inventory\] Markers not found in architecture.md; aborting replace.")
- L2158:         logger.warning("\[inventory\] generate_project_map completed with warnings; check markers and path.")
- L2204:     """Find lines containing TODO/FIXME/WARN (case-insensitive). Returns list of (lineno, text)."""
- L2206:     pat = re.compile(r"\b(TODO|FIXME|WARN|WARNING)\b", re.IGNORECASE)
- L2613:         # TODO/FIXME/WARN
- L2616:             lines.append("- TODO/FIXME/WARN:")
- L2683:     """Aggregate TODO/FIXME/WARN lines from webapp/ into a compact index.
- L2692:         lines.append("# TODO/FIXME index — webapp\n")
- L2743:             logger.warning(f"\[noise\] No suggestions file found at {path}")
- L3039:                 lines.append("### ⚠️ TODO/FIXME/WARN")

## `webapp\parser\utils\strategy_concurrency.py`

- L37:             logger.warning(f"\[CONCURRENCY\] DOM strategy {name} failed: {e}")
- L65:                 logger.warning(f"\[CONCURRENCY\] Strategy {name} error: {e}")
- L73:         logger.warning(f"\[CONCURRENCY\] {_safe_run_strategy.__name__} {name} failed: {e}")
- L102:             logger.warning(f"\[CONCURRENCY\]\[ASYNC\] DOM strategy {name} failed: {e}")
- L120:             logger.warning(f"\[CONCURRENCY\]\[ASYNC\] Strategy {name} error: {e}")

## `webapp\parser\utils\table_builder.py`

- L816: _emit("warning", "builder", "\[TABLE_BUILDER\] dynamic_table_extractor failed for panel table", session_id, error=str(e))
- L828: _emit("warning", "builder", "\[TABLE_BUILDER\] dynamic_table_extractor failed (no panels path)", session_id, error=str(e))
- L836: _emit("warning", "builder", "\[TABLE_BUILDER\] all_panel_tables was not a list; coercing to empty list", session_id, got_type=str(type(all_panel_tables)))
- L845: _emit("warning", "builder", "\[TABLE_BUILDER\] Dropping invalid table entry", session_id, entry_type=str(type(item)))
- L862: _emit("warning", "builder", "\[TABLE_BUILDER\] sanitize failed", session_id, error=str(e))
- L867: _emit("warning", "builder", "\[TABLE_BUILDER\] harmonize failed", session_id, error=str(e))
- L873: _emit("warning", "builder", "\[TABLE_BUILDER\] collapse_ballot_synonym_columns failed", session_id, error=str(e))
- L925:                 "info" if status == "ok" else "warning",
- L950: _emit("warning", "builder", "\[TABLE_BUILDER\] entity annotate failed", session_id, error=str(e))
- L955: _emit("warning", "builder", "\[TABLE_BUILDER\] stringify entity_info failed", session_id, error=str(e))
- L975: _emit("warning", "builder", "\[TABLE_BUILDER\] pivot_to_wide failed", session_id, error=str(e))
- L995: _emit("warning", "builder", "\[TABLE_BUILDER\] ensure division totals failed", session_id, error=str(e))
- L1288: _emit("warning", "builder", f"\[TABLE_BUILDER\] Column marked incorrect: {col_name}", session_id, contest=contest)
- L1361: _emit("warning", "builder", "\[TABLE_BUILDER\] Failed to persist table structure logs", session_id, error=str(e))
- L1376: _emit("warning", "builder", "\[TABLE_BUILDER\] Failed to persist coordinator DB log", session_id, error=str(e))

## `webapp\parser\utils\table_core.py`

- L231:         logger.warning(f"\[TABLE BUILDER\] Concurrent strategies execution failed: {e}")
- L288:             logger.warning(f"\[TABLE BUILDER\] RawJSON pivot failed: {e}")
- L296:             logger.warning(f"\[TABLE BUILDER\] pivot_to_wide signature mismatch (skipped): {e}")
- L298:             logger.warning(f"\[TABLE BUILDER\] pivot_to_wide failed (skipped): {e}")
- L349:                 logger.warning(f"\[TABLE BUILDER\] finalize output failed: {e}")
- L414:         logger.warning(f"\[TABLE BUILDER\]\[ASYNC\] Concurrent strategies execution failed: {e}")
- L477:                 logger.warning(f"\[TABLE BUILDER\]\[ASYNC\] finalize output failed: {e}")

## `webapp\parser\utils\user_prompt.py`

- L312:                 logger.warning("\[UserPrompt\] Webapp mode active but no socketio_emit_func set!")
- L349:             logger.warning("\[CLI Prompt\] EOFError encountered.")
- L370:             logger.warning("\[Webapp Prompt\] socketio_emit_func not set.")
- L428:             "WARNING": 30,
- L507:                 logger.warning("\n\[Prompt\] Timed out.")
- L558:                 logger.warning("\n\[Prompt\] No input available (EOF). Exiting prompt.")
- L592:                 logger.warning("Invalid input. Please try again.")
- L594:                     logger.warning("\[Prompt\] Too many invalid attempts.")
- L659:             logger.warning("\[Prompt Queue\] Invalid queued yes/no response; falling back to interactive prompt.")
- L674:                     logger.warning("\n\[Prompt\] Timed out.")
- L881:                 logger.warning("\[yellow\]\[FEEDBACK\] Skipped manual correction.\[/yellow\]")
- L913:             logger.warning("\[yellow\]Button confirmation cancelled by user.\[/yellow\]")

## `webapp\parser\web_pipeline.py`

- L49:                 logger.warning({
- L50:                     "level": "WARNING",
- L66:                     logger.warning({
- L67:                         "level": "WARNING",
- L83:                     logger.warning({
- L84:                         "level": "WARNING",

## `webapp\tests\conftest.py`

- L17:  alive.  When we run with ⁣`⁣`PYTHONWARNINGS=error⁣`⁣` this warning escalates to
- L21:  1. Ignore the specific warning category so it never escalates.
- L23:     otherwise raise an ⁣`⁣`ExceptionGroup⁣`⁣` even if the warning was ignored.

## `webapp\tests\test_schema_validation_warnings.py`

- L17:     # Build a table that lacks candidate/ballot/total columns, forcing a 'normalized schema weak' warning
- L39:                 return lvl=="warning" and msg.get("status")=="weak"
- L44:                     return lvl=="warning" and inner.get("status")=="weak"
- L49:     assert found, f"Expected a schema weak warning in captured logs; got: {captured}"

## `webapp\tests\test_webapp_app.py`

- L17:     mp.setenv("WEBAPP_CONSOLE_LEVELS", "INFO,WARNING,ERROR")
