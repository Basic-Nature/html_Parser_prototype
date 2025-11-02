# TODO/FIXME index — webapp

Total annotations: 465

## `C:/Users/edu-loaner/html_Parser_prototype/webapp/Smart_Elections_Parser_Webapp.py`

- L88: WEBAPP_CONSOLE_LEVELS = set(os.environ.get("WEBAPP_CONSOLE_LEVELS", "ERROR,WARNING").upper().split(","))
- L346:     Levels: INFO, DEBUG, WARNING, ERROR, CRITICAL, TRACE
- L385:     LEVELS = {"INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL", "TRACE"}
- L421:             elif "warning" in mlow:
- L795:         # For websocket handshake only: add Cache-Control so webhint stops warning
- L1068:         logger.warning({"type": "sec", "message": "Favicon path escape blocked", "requested": ico_path})
- L1160:             logger.warning({
- L1161:                 "level": "WARNING",
- L1466:             "level": "WARNING",
- L1549:         logger.warning(
- L1551:                 "level": "WARNING",
- L1578:         logger.warning(
- L1580:                 "level": "WARNING",
- L1805:         logger.warning({
- L1806:             "level": "WARNING",
- L1874:         logger.warning({
- L1875:             "level": "WARNING",

## `C:/Users/edu-loaner/html_Parser_prototype/webapp/parser/Context_Integration/Context_Library/constants.py`

- L1920:         "icon-bg-dark", "icon-bg-primary", "icon-bg-secondary", "icon-bg-success", "icon-bg-danger", "icon-bg-warning",
- L2011:     "warning", "info_box", "navigation", "pagination", "tab", "modal", "tooltip", "ignore", "unknown"

## `C:/Users/edu-loaner/html_Parser_prototype/webapp/parser/Context_Integration/context_coordinator.py`

- L746:                     logger.warning("[ALERT MONITOR] Thread did not stop cleanly.")
- L884:             logger.warning(f"[yellow]Integrity issues:[/yellow] {issues['integrity_issues']}")
- L1123:                 logger.warning(f"[ContextCoordinator] No table structure found for contest: {contest}")
- L1292:                     logger.warning(f"[get_feedback_pattern_kb] Skipping corrupt line: {e}")
- L1405:                 logger.warning("[group_dom_nodes_by_label] No organized DOM parts. (Further warnings suppressed)")
- L1407:                 logger.warning(f"[group_dom_nodes_by_label] No organized DOM parts. (Occurred {ContextCoordinator._dom_parts_warning_count} times)")
- L1412:             logger.warning("[group_dom_nodes_by_label] No DOM nodes found.")
- L1430:                 logger.warning("[submit_user_feedback] ContextOrganizer has no submit_user_feedback method.")
- L1458:                 logger.warning(f"[correct_and_update_contest] Contest {contest_id} missing type/election_types after sync.")
- L1482:             logger.warning("[print_contest_summary] No organized contests to summarize.")
- L1495:             logger.warning("[plot_contest_distribution] No organized contests to plot.")
- L1546:                 logger.warning("No organized DOM parts.")
- L1549:                 logger.warning("No organized DOM parts. (Further warnings suppressed)")
- L1560:             logger.warning("[get_contest_groups] No contest groups found.")
- L1569:             logger.warning("[get_panel_groups] No panel groups found.")
- L1578:             logger.warning("[get_button_groups] No button groups found.")
- L1587:             logger.warning("[get_table_groups] No table groups found.")
- L1596:             logger.warning("[get_relationships] No organized context.")
- L1704:                 logger.warning(f"[fuzzy_score] One or both inputs are empty: a='{a_str}', b='{b_str}'")
- L1710:                 logger.warning(f"[fuzzy_score] One or both inputs are too short: a='{a_str}', b='{b_str}'")
- L2156:             logger.warning(f"[extract_field] Unknown field_type: {field_type}")
- L2414:                     logger.warning(f"[get_full_contest] Contest {contest_id} missing type/election_types after sync.")
- L2499:                         logger.warning(f"[list_tables] Table '{tbl}' missing metadata or columns.")
- L2531:                 logger.warning(f"[get_table_metadata] Table '{table_name}' missing columns.")
- L2549:                 logger.warning(f"[check_missing_tables] Missing tables: {missing}")
- L2610:                 logger.warning(f"[save_table_structure] Failed to save structure for contest: {contest}")
- L2787:             logger.warning(f"[get_best_button_advanced] Contest argument was not a dict. Converted to: {contest}")
- L2791:             logger.warning(f"[get_best_button_advanced] Keywords argument was not a list. Converted to: {keywords}")
- L2795:             logger.warning(f"[get_best_button_advanced] Context argument was not a dict. Converted to: {context}")
- L2802:             logger.warning("[get_best_button_advanced] _semantic_model is not set or is not an object. Using None.")
- L2946:                             logger.warning(f"[yellow][Coordinator] Button '{cand.get('label')}' rejected, retrying...[/yellow]")

## `C:/Users/edu-loaner/html_Parser_prototype/webapp/parser/Context_Integration/context_organizer.py`

- L224:                 # If it's a method, class, or something else, warn and set to None
- L225:                 logger.warning(f"[CONTEXT ORGANIZER] Provided embedding_model is not a recognized model instance or string. Type: {type(self.embedding_model)}. Setting to None.")
- L978:                     logger.warning(f"[CONTEST] Skipping contest with suspiciously large or missing title: {str(title)[:100]}...")
- L1071:                 logger.warning(f"[CONTEST] Filtered out {len(filtered_out)} contests due to missing required fields.")
- L1073:                     logger.warning(f"  [Filtered] {reason}: {str(c)[:100]}...")
- L1076:                 logger.warning("[CONTEST] No contests with required fields for downstream output.")
- L1329:                                 logger.warning(f"[ML] Anomaly index {idx} out of range for contests list of length {len(contests)}")
- L1362:                     logger.warning(f"  [yellow]{entry['title']}[/yellow]: {', '.join(entry['fixes'])}")
- L1366:                     logger.warning(f"[bold yellow][INTEGRITY][/bold yellow] Duplicate contest detected.\n  [dim]Context:[/dim] {contest}")
- L1368:                     logger.warning(f"[bold yellow][INTEGRITY][/bold yellow] Contest missing location info.\n  [dim]Context:[/dim] {contest}")
- L1370:                     logger.warning(f"[bold yellow][INTEGRITY][/bold yellow] Contest missing year.\n  [dim]Context:[/dim] {contest}")
- L1798:             logger.warning(f"[ContextOrganizer] Could not update context library with feedback: {e}")
- L1878:                 logger.warning(f"[CONTEXT ORGANIZER] No table structure found for contest: {contest}")

## `C:/Users/edu-loaner/html_Parser_prototype/webapp/parser/Context_Integration/librarian.py`

- L637:         logger.warning(f"\n[LIBRARIAN SELF-HEAL] Attempt {attempt}...")
- L643:         logger.warning("[LIBRARIAN SELF-HEAL] Misalignments found. Launching manual_correction...")
- L646:         logger.warning(f"[LIBRARIAN SELF-HEAL] Sleeping {cooldown}s before rescanning...")

## `C:/Users/edu-loaner/html_Parser_prototype/webapp/parser/config.py`

- L318:                 logger.warning("[DB][AAD] Falling back to password auth.")

## `C:/Users/edu-loaner/html_Parser_prototype/webapp/parser/data_manager.py`

- L82:             logger.warning(f"[REMOVED] {popped}")
- L89:             logger.warning(f"[REMOVED] {index_or_value}")
- L128:                     logger.warning(f"[DELETED] {files[idx]}")

## `C:/Users/edu-loaner/html_Parser_prototype/webapp/parser/handlers/formats/html_handler.py`

- L118:             logger.warning(f"[HTML Handler] County '{county}' not found. Closest matches: {matches}")
- L123:             logger.warning(f"[HTML Handler] Detected county '{county}' is not in known counties for state '{suggested_state or state}'.")
- L136:                 logger.warning(f"[HTML Handler] State '{user_state}' not found. Closest matches: {matches}")
- L166:                     logger.warning(f"[HTML Handler] County '{user_county}' not found. Closest matches: {matches}")

## `C:/Users/edu-loaner/html_Parser_prototype/webapp/parser/handlers/formats/pdf_handler.py`

- L1276:         logger.warning({
- L1277:             "level": "WARNING",
- L1279:             "message": f"[WARN] Multi-mode text extraction failed: {e}",
- L1410:         logger.warning({
- L1411:             "level": "WARNING",
- L1413:             "message": f"[WARN] fitz text extraction failed: {e}",
- L1441:         logger.warning({
- L1442:             "level": "WARNING",
- L1444:             "message": "[WARN] ENABLE_OCR_FORCE is set but Tesseract is unavailable; skipping OCR fallback.",
- L1500:             logger.warning({
- L1501:                 "level": "WARNING",
- L1503:                 "message": "[WARN] Low-signal text detected but OCR is unavailable or disabled.",
- L1599:         logger.warning({
- L1600:             "level": "WARNING",
- L1602:             "message": "[WARN] No contest selected. Using filename fallback.",
- L1870:                 logger.warning({
- L1871:                     "level": "WARNING",
- L1873:                     "message": f"[WARN] Selected contest '{contest}' not found in column '{contest_column}'. Skipping row filter.",
- L1957:             logger.warning({
- L1958:                 "level": "WARNING",
- L1960:                 "message": f"[WARN] No structured rows matched the inferred column count of {len(headers)}. Total lines scanned: {unmatched_count}",
- L1987:             logger.warning({
- L1988:                 "level": "WARNING",
- L2058:     logger.warning({
- L2059:         "level": "WARNING",

## `C:/Users/edu-loaner/html_Parser_prototype/webapp/parser/handlers/states/arizona/arizona.py`

- L23:     logger.warning("[WARN] context_library.json not found. Using fallback config for Arizona handler.")
- L49:                 logger.warning(f"[WARN] Could not expand card {i+1}: {e}")
- L62:             logger.warning(f"[WARN] Vote Type toggle failed: {e}")
- L75:             logger.warning(f"[WARN] County toggle failed: {e}")
- L162:         logger.warning("[FALLBACK] No tables were parsed. Either no results are published yet or the structure has changed.")
- L163:         logger.warning("[FALLBACK] Please verify that the site has posted election data.")

## `C:/Users/edu-loaner/html_Parser_prototype/webapp/parser/handlers/states/example state/example_county/example_county.py`

- L119:         logger.warning(f"[yellow][WARNING] No ballot items found by div selectors. Trying table-based extraction...[/yellow]")

## `C:/Users/edu-loaner/html_Parser_prototype/webapp/parser/handlers/states/example state/example_state.py`

- L47:             logger.warning(f"[Example Handler] No specific parser implemented for county: '{county}'. Continuing with state-level logic.")
- L148:         logger.warning(f"[yellow][WARNING] No ballot items found by div selectors. Trying table-based extraction...[/yellow]")

## `C:/Users/edu-loaner/html_Parser_prototype/webapp/parser/handlers/states/new_york/county/rockland.py`

- L73:         logger.warning("[WARNING] dom_parts missing after organize_and_enrich.")
- L96:         logger.warning("[red]No contest selected. Skipping.[/red]")
- L140:                         logger.warning(f"[yellow][WARNING] Button '{btn1.get('label', '')}' is not clickable (visible={safe_is_visible(element, logger)}, enabled={safe_is_enabled(element, logger)})[/yellow]")
- L177:                         logger.warning(f"[yellow][WARNING] Button '{btn2.get('label', '')}' is not clickable (visible={safe_is_visible(element, logger)}, enabled={safe_is_enabled(element, logger)})[/yellow]")

## `C:/Users/edu-loaner/html_Parser_prototype/webapp/parser/handlers/states/new_york/new_york.py`

- L20:         logger.warning("[NY Handler] No county specified in html_context.")
- L36:         logger.warning(f"[NY Handler] No specific parser implemented for county: '{county}'. Please add it under {module_path}.py")

## `C:/Users/edu-loaner/html_Parser_prototype/webapp/parser/handlers/states/pennsylvania/pennsylvania.py`

- L36:             logger.warning(f"[NAV] Step failed: {step} — {e}")
- L47:     logger.warning(f"[bold yellow]Detected election:[/bold yellow] {header_text}")
- L68:                     logger.warning("[PA] Invalid index input for election selection.")
- L70:                 logger.warning("[PA] Elections dropdown not found.")
- L72:             logger.warning(f"[PA] Failed to expand Elections menu or load selection: {e}")
- L88:                 logger.warning("[PA] County Breakdown link not found.")
- L90:             logger.warning(f"[PA] Failed to click County Breakdown link: {e}")
- L105:         logger.warning("[yellow]Multiple CSV files found in input. Please select one:[/yellow]")

## `C:/Users/edu-loaner/html_Parser_prototype/webapp/parser/health/health_router.py`

- L224:                     logger.warning(f"[health_router] manual_correction failed (attempt {attempt}): {result.stderr}")
- L308:             logger.warning(f"[SELF-HEAL] Misalignments found. Launching manual_correction...")
- L310:             logger.warning(f"[SELF-HEAL] Sleeping {cooldown}s before rescanning...")
- L312:         logger.warning("[SELF-HEAL] Max retries reached. Some misalignments may remain.")
- L347:                 logger.warning(f"[PIPELINE] Could not fix corrupted JSON files: {e}")
- L352:                 logger.warning("[PIPELINE] Misaligned NER examples found. Self-heal loop will be handled by scan_misaligned_ner.")
- L354:                 logger.warning("[PIPELINE] scan_misaligned_ner failed or file missing. Proceeding with caution.")
- L386:                 logger.warning("[PIPELINE] Model retraining failed.")

## `C:/Users/edu-loaner/html_Parser_prototype/webapp/parser/health/log_cache_cleaner_bot.py`

- L151:                     logger.warning(f"Skipping non-dict entry in spacy_ner_train_data.jsonl: {entry}")
- L460:                 logger.warning("[DB][WARNING] No user tables found in schema 'public'.")
- L503:         logger.warning("[CLEAN][WARNING] The following files are still too large after cleaning:")
- L507:         logger.warning("[MISALIGNED] Consider cleaning or pattern-excluding these from your training data:")

## `C:/Users/edu-loaner/html_Parser_prototype/webapp/parser/health/manual_correction_bot.py`

- L313:             logger.warning(f"Coordinator ML scoring failed: {e}")
- L335:             logger.warning(f"Coordinator field suggestion failed: {e}")
- L347:         logger.warning(f"Log file not found: {path}")
- L356:                     logger.warning(f"[CORRUPT] {path} line {i}: {e}")
- L388:                             logger.warning(f"[SKIP] File not found: {file}")
- L392:                             logger.warning(f"[SKIP] File too large: {file}")
- L414:                                         logger.warning(f"[CORRUPT-LINE] {file} line {i+1}: {line[:80]}... ({e})")
- L426:                                 logger.warning(f"[CORRUPT] {len(corrupt_items)} lines saved to {corrupt_path}")
- L431:                                 logger.warning(f"[FIXED] All lines invalid, recreated empty .jsonl file: {file}")
- L445:                                 logger.warning(f"[CORRUPT] {file}: {e}")
- L457:                                 logger.warning(f"[CORRUPT] Corrupt JSON saved to {corrupt_path}")
- L463:                                 logger.warning(f"[FIXED] All content invalid, recreated minimal valid JSON in {file}")
- L468:                         logger.warning(f"[CORRUPT] {file}: {e}")
- L477:                                         logger.warning(f"[QUARANTINED] {file} -> {quarantine_dir / file.name}")
- L481:                                         logger.warning(f"[DELETED] {file}")
- L484:                                     logger.warning(f"[SKIP-DELETE] File already missing: {file}")
- L529:             logger.warning(f"[FIND-LOGS] Skipped {d}: {e}")
- L554:                     logger.warning(f"[CORRUPT] {path} line {line_num}: {e}")
- L711:                     logger.warning(f"Invalid JSON, skipping edit: {e}")
- L744:     # TODO: Add JSON schema validation here if desired
- L975:         logger.warning("Could not import integrity_check for anomaly highlighting.")
- L1063:     If missing, create with DEFAULT_STRUCTURE. Warn if schema version mismatches.
- L1082:         logger.warning(f"Schema version mismatch: found {context_lib.get('schema_version')}, expected {SCHEMA_VERSION}. Consider migrating.")
- L1125:                     logger.warning(f"[AUTO] Could not delete log file {log_file}: {e}")
- L1241:             logger.warning(f"[SKIP] Could not load {log_file}: {e}")
- L1257:         logger.warning("No log files matched any of the specified fields. Will attempt to process all log files for all fields.")
- L1340:                 logger.warning(f"Could not delete log file {log_file}: {e}")
- L1360:         logger.warning("[WARNING] No entries were processed. Check your log file naming, field configuration, or use --dry-run for debugging.")

## `C:/Users/edu-loaner/html_Parser_prototype/webapp/parser/health/retrain_table_structure_models.py`

- L148:         logger.warning(f"[CLEAN] File not found: {jsonl_path}")
- L156:                 logger.warning(f"[CLEAN] Could not parse line: {e}")
- L171:                 logger.warning(f"[CLEAN] Alignment check failed for text: {text[:50]}... ({e})")
- L244:             logger.warning(f"Failed to load {path}: {e}")
- L373:                     logger.warning(f"Skipping misaligned entity in: {text}")
- L378:                 logger.warning(f"Error validating entity alignment: {e}")
- L404:         logger.warning(f"[spaCy] Could not check GPU availability: {e}")
- L420:         logger.warning(f"[spaCy] Could not load lexeme normalization table. You may ignore this for English. Error: {e}")
- L502:         logger.warning(f"[NER] Skipped {misaligned_count} misaligned examples. Saved to {misaligned_path}")
- L516:         logger.warning("No NER training examples found. Skipping spaCy NER retraining.")
- L585:         logger.warning(f"[SUGGESTION] Consider lowering min_delta or increasing patience if you want longer training.")
- L587:         logger.warning(f"[SUGGESTION] Model improved until the last epoch. Consider increasing epochs for further improvement.")
- L588:     logger.warning(f"[SUGGESTION] Next run: patience={patience}, min_delta={min_delta:.2f}, epochs={epochs}")
- L674:         logger.warning("No training examples found. Aborting retraining.")
- L693:             logger.warning(f"[WARN] Could not delete old model directory {oldest_path}: {e}")
- L705:             logger.warning(f"[WARN] Failed to load existing model: {e}")
- L708:         logger.warning("Falling back to base model (all-MiniLM-L6-v2).")
- L748:             logger.warning(f"[WARN] Could not update canonical model directory: {e}")
- L776:                     logger.warning(f"MISALIGNED: {text} {annots['entities']}")
- L806:             logger.warning("[DB] Base.metadata.tables is empty. No models registered? Did you import all model classes?")

## `C:/Users/edu-loaner/html_Parser_prototype/webapp/parser/health/scan_misaligned_ner.py`

- L60:                     logger.warning(f"[CORRUPT] Could not parse line: {e}")
- L81:             logger.warning(f"\n[MISALIGNED] Top {top_n} most frequent misaligned NER texts:")
- L83:                 logger.warning(f"  {repr(text)}: {count} times")
- L84:             logger.warning("[MISALIGNED] Consider cleaning or pattern-excluding these from your training data.")
- L85:         logger.warning("Run the manual_correction to review and clean these examples before retraining.")
- L86:         logger.warning("If you see spaCy entity alignment warnings, consider cleaning your training data or using the provided validation function.")
- L96:                 logger.warning(f"[WARN] Could not remove old misaligned file: {e}")
- L110:         logger.warning("[SELF-HEAL] Misalignments found. Launching manual_correction for spacy_ner_misaligned...")
- L117:             logger.warning(f"[SELF-HEAL] manual_correction exited with code {result.returncode}")
- L118:         logger.warning(f"[SELF-HEAL] Sleeping {cooldown}s before rescanning...")
- L120:     logger.warning("[SELF-HEAL] Max retries reached. Some misalignments may remain.")

## `C:/Users/edu-loaner/html_Parser_prototype/webapp/parser/html_election_parser.py`

- L39:     logger.warning("Deleting .processed_urls cache for fresh start...")
- L372:                 logger.warning({
- L373:                     "level": "WARNING",
- L443:                 logger.warning(payload_2)
- L553:                         "level": "WARNING",
- L558:                     logger.warning(payload)
- L582:                 # Soft-fail: continue; downstream will warn if nothing found
- L642:                     "level": "WARNING",
- L647:                 logger.warning(payload)
- L711:                                 logger.warning({
- L712:                                     "level": "WARNING",
- L729:                             "level": "WARNING",
- L734:                         logger.warning(payload)
- L745:                             "level": "WARNING",
- L750:                         logger.warning(payload)
- L752:                         msg = "[WARN] No output file path returned from parser and no output files found."
- L754:                             "level": "WARNING",
- L759:                         logger.warning(payload)
- L764:                     "level": "WARNING",
- L769:                 logger.warning(payload)
- L861:                     logger.warning({
- L862:                         "level": "WARNING",
- L914:         logger.warning({
- L915:             "level": "WARNING",

## `C:/Users/edu-loaner/html_Parser_prototype/webapp/parser/state_router.py`

- L42:         logger.warning("[Router] handlers/states directory not found.")
- L59:             logger.warning(f"[Router] counties directory not found for state: {state_key}")
- L141:             logger.warning(f"[Fallback][Session:{session_id}] Aborted by user.")
- L144:             logger.warning(f"[Fallback][Session:{session_id}] Aborted by user.")
- L147:             logger.warning(f"[Fallback][Session:{session_id}] State '{state}' not found. Please try again.")
- L164:                 logger.warning(f"[Fallback][Session:{session_id}] Aborted by user.")
- L167:                 logger.warning(f"[Fallback][Session:{session_id}] County '{county}' not found for state '{state}'. Please try again.")
- L174:     logger.warning(f"[Fallback][Session:{session_id}] Too many failed attempts. Exiting fallback.")
- L482:             logger.warning(f"No counties found for state '{state}'. Try --fuzzy for fuzzy matching.")
- L493:                     logger.warning(f"Failed to load context from file: {e}")
- L503:             logger.warning("No suitable handler found.")
- L510:                 logger.warning("No handler selected. Exiting.")
- L517:                 logger.warning("Still could not import a suitable handler.")

## `C:/Users/edu-loaner/html_Parser_prototype/webapp/parser/utils/browser_utils.py`

- L73:                     logger.warning(f"[browser_utils] Failed to safely parse context_library value for key '{key}'")
- L75:                 logger.warning(f"[browser_utils] Skipping unsafe context_library value for key '{key}'")
- L271:                     logger.warning(f"[safe_attributes] Playwright JS extraction failed: {e}")
- L285:                 logger.warning(f"[safe_attributes] Playwright fallback extraction failed: {e}")
- L368:         logger and logger.warning(f"[safe_count] Object is not countable: {type(obj)}")
- L413:             logger.warning(f"[safe_launch] browser_type is not a SyncBrowserType: {type(browser_type)}")
- L432:             logger.warning(f"[async_safe_launch] browser_type is not an AsyncBrowserType: {type(browser_type)}")
- L510:             logger.warning({
- L511:                 "level": "WARNING",
- L536:             logger.warning(f"[CAPTCHA] Detected Cloudflare CAPTCHA indicator: '{indicator}'")
- L545:     logger.warning(f"[CAPTCHA] CAPTCHA detected in async mode. Manual intervention not implemented. (Session: {session_id})")
- L566:             logger.warning(f"[CAPTCHA] Detected Cloudflare CAPTCHA indicator: '{indicator}'")
- L575:             logger.warning({
- L576:                 "level": "WARNING",
- L587:     logger.warning(f"[CAPTCHA] CAPTCHA detected in sync mode. Manual intervention not implemented. (Session: {session_id})")
- L671:                     logger and logger.warning("[SCROLL] User aborted scrolling.")
- L681:         logger and logger.warning("[SCROLL] Max scroll time/attempts exceeded. Page may not be fully loaded.")

## `C:/Users/edu-loaner/html_Parser_prototype/webapp/parser/utils/captcha_tools.py`

- L113:         logger.warning(f"[CAPTCHA] Foreground window fallback failed: {e}")
- L149:     logger.warning("[CAPTCHA] CAPTCHA not resolved within timeout.")

## `C:/Users/edu-loaner/html_Parser_prototype/webapp/parser/utils/contest_selector.py`

- L266:     elif lvl == "warning":
- L267:         logger.warning(entry)
- L668:         _log("warning", "selector", f"Feedback loop {loop+1}: verifying contests", session_id=session_id,
- L1119:                 logger.warning({"level": "WARNING", "type": "selector", "message": "Empty search term", "session_id": session_id})
- L1124:                 logger.warning({"level": "WARNING", "type": "selector", "message": f"No matches for '{term}'", "session_id": session_id})
- L1186:         logger.warning({"level": "WARNING", "type": "selector", "message": "No match; try again.", "session_id": session_id})

## `C:/Users/edu-loaner/html_Parser_prototype/webapp/parser/utils/dom_extractor.py`

- L148:         logger.warning(f"[DOM_EXTRACTOR] failure: {e}")

## `C:/Users/edu-loaner/html_Parser_prototype/webapp/parser/utils/dynamic_table_extractor.py`

- L117:                 _emit("warning", "extractor", "[EXTRACTOR] No <table> found in provided table_html.", session_id)
- L122:                 _emit("warning", "extractor", "[EXTRACTOR] No <tr> rows found in table_html.", session_id)
- L164:             _emit("warning", "extractor", "[EXTRACTOR] Candidate NLP/score step failed", session_id, error=str(e))
- L180:     _emit("warning", "extractor", "[EXTRACTOR] No suitable table candidates found.", session_id)
- L210:         _emit("warning", "extractor", "[EXTRACTOR] Error while scanning <table> elements", session_id, error=str(e))
- L222:         _emit("warning", "extractor", "[EXTRACTOR] DOM extraction failed", session_id, error=str(e))
- L265:         _emit("warning", "extractor", "[EXTRACTOR] Pattern extraction failed", session_id, error=str(e))
- L769:         _emit("warning", "extractor", "No learned DOM patterns found.")
- L793:                     _emit("warning", "extractor", "Entry deleted.")
- L798:                     _emit("warning", "extractor", "Unknown action.")
- L800:                 _emit("warning", "extractor", "Invalid entry number.")

## `C:/Users/edu-loaner/html_Parser_prototype/webapp/parser/utils/embedding_cache.py`

- L176:                 logger.warning(msg)

## `C:/Users/edu-loaner/html_Parser_prototype/webapp/parser/utils/extraction_strategies.py`

- L71:             logger.warning(f"[STRATEGY] {name} failed: {e}")

## `C:/Users/edu-loaner/html_Parser_prototype/webapp/parser/utils/format_router.py`

- L148:         logger.warning({
- L149:             "level": "WARNING",
- L151:             "message": "[WARN] No supported file formats found on the page.",
- L172:         logger.warning({
- L173:             "level": "WARNING",
- L175:             "message": f"[WARN] Unsupported format requested: {format_str}",
- L179:         logger.warning({
- L180:             "level": "WARNING",
- L354:         logger.warning({
- L355:             "level": "WARNING",
- L532:             logger.warning({
- L533:                 "level": "WARNING",
- L608:         logger.warning({
- L609:             "level": "WARNING",

## `C:/Users/edu-loaner/html_Parser_prototype/webapp/parser/utils/html_scanner.py`

- L105:                 "level": "WARNING",
- L109:             logger.warning(payload)
- L131:                             "level": "WARNING",
- L135:                         logger.warning(payload)
- L230:                 "level": "WARNING",
- L234:             logger.warning(payload)
- L257:                             "level": "WARNING",
- L261:                         logger.warning(payload)
- L295:                 "level": "WARNING",
- L299:             logger.warning(payload)
- L322:                             "level": "WARNING",
- L326:                         logger.warning(payload)
- L521:                     "level": "WARNING",
- L525:                 logger.warning(payload)
- L726:                 logger.warning(f"[ML SIMILARITY] No embedding computed for segment: {safe_get(segment, 'segment_hash', None)}")
- L749:                 logger.warning(f"[ML SIMILARITY] No embedding computed for segment: {safe_get(segment, 'segment_hash', None)}")
- L976:                     "level": "WARNING",
- L980:                 logger.warning(payload)
- L987:                     "level": "WARNING",
- L991:                 logger.warning(payload)
- L1303:                         "level": "WARNING",
- L1307:                     logger.warning(payload)
- L1365:                         "level": "WARNING",
- L1369:                     logger.warning(payload)
- L1618:                 logger.warning({"level": "WARNING", "type": "dom_segments", "message": msg_warn})
- L1674:                     logger.warning({"level": "WARNING", "type": "page_hash", "message": msg})
- L1681:                 logger.warning({"level": "WARNING", "type": "page_hash", "message": msg})
- L1693:                 logger.warning({"level": "WARNING", "type": "page_hash", "message": msg})
- L1716:             logger.warning({"level": "WARNING", "type": "cache", "message": msg})
- L1751:             logger.warning({"level": "WARNING", "type": "cache", "message": msg})
- L1930:         logger.warning({"level": "WARNING", "type": "segment_review", "message": msg})
- L1939:             else logger.warning({
- L1940:                 "level": "WARNING",
- L2056:                         "level": "WARNING",
- L2060:                     logger.warning(payload)
- L2072:                         "level": "WARNING",
- L2076:                     logger.warning(payload)
- L2085:                         "level": "WARNING",
- L2089:                     logger.warning(payload)
- L2104:                             "level": "WARNING",
- L2108:                         logger.warning(payload)
- L2120:                                 "level": "WARNING",
- L2124:                             logger.warning(payload)
- L2133:                                 "level": "WARNING",
- L2137:                             logger.warning(payload)
- L2146:                                 "level": "WARNING",
- L2150:                             logger.warning(payload)
- L2160:                                     "level": "WARNING",
- L2164:                                 logger.warning(payload)
- L2175:                                         "level": "WARNING",
- L2179:                                     logger.warning(payload)
- L2189:                                     "level": "WARNING",
- L2193:                                 logger.warning(payload)
- L2205:                                     "level": "WARNING",
- L2209:                                 logger.warning(payload)
- L2220:                                     "level": "WARNING",
- L2224:                                 logger.warning(payload)
- L2234:                                     "level": "WARNING",
- L2238:                                 logger.warning(payload)
- L2248:                                     "level": "WARNING",
- L2252:                                 logger.warning(payload)
- L2262:                                     "level": "WARNING",
- L2266:                                 logger.warning(payload)
- L2276:                                     "level": "WARNING",
- L2280:                                 logger.warning(payload)
- L2296:                                                     "level": "WARNING",
- L2300:                                                 logger.warning(payload)
- L2311:                                     "level": "WARNING",
- L2315:                                 logger.warning(payload)
- L2326:                                     "level": "WARNING",
- L2330:                                 logger.warning(payload)
- L2341:                                     "level": "WARNING",
- L2345:                                 logger.warning(payload)
- L2356:                                     "level": "WARNING",
- L2360:                                 logger.warning(payload)
- L2368:                 "level": "WARNING",
- L2372:             logger.warning(payload)
- L2381:                     "level": "WARNING",
- L2385:                 logger.warning(payload)
- L2399:                             "level": "WARNING",
- L2403:                         logger.warning(payload)
- L2413:                     "level": "WARNING",
- L2417:                 logger.warning(payload)
- L2428:                         "level": "WARNING",
- L2432:                     logger.warning(payload)
- L2442:                     "level": "WARNING",
- L2446:                 logger.warning(payload)
- L2456:                     "level": "WARNING",
- L2460:                 logger.warning(payload)
- L2470:                     "level": "WARNING",
- L2474:                 logger.warning(payload)
- L2718:                 logger.warning({"level": "WARNING", "type": "context", "message": msg})
- L2734:                 logger.warning({"level": "WARNING", "type": "context", "message": msg})
- L2743:                 logger.warning({"level": "WARNING", "type": "context", "message": msg})
- L2755:                     logger.warning({"level": "WARNING", "type": "context", "message": msg})
- L2765:                     logger.warning({"level": "WARNING", "type": "context", "message": msg})
- L2775:                     logger.warning({"level": "WARNING", "type": "context", "message": msg})
- L2797:             logger.warning({"level": "WARNING", "type": "scan_html", "message": msg})
- L3225:                     logger.warning({"level": "WARNING", "type": "dom_debug", "message": msg_warn})

## `C:/Users/edu-loaner/html_Parser_prototype/webapp/parser/utils/model_registry.py`

- L387:                     logger.warning(f"Failed loading local override for SentenceTransformer: {e}")
- L407:                 logger.warning("TRANSFORMERS_OFFLINE/HUGGINGFACE_HUB_OFFLINE set; skipping HF download. Embeddings disabled.")
- L424:                 # Downgrade DNS/network errors to WARNING for noisy environments
- L427:                     logger.warning(f"Failed to load base SentenceTransformer (network/DNS). Running without embeddings. Error: {e}")

## `C:/Users/edu-loaner/html_Parser_prototype/webapp/parser/utils/output_utils.py`

- L104:         logger.warning("[yellow][OUTPUT] Year could not be verified. Using 'Unknown'.[/yellow]")
- L107:         logger.warning("[yellow][OUTPUT] contests could not be verified. Using 'unknown_contests'.[/yellow]")
- L504:         logger.warning(f"[OUTPUT_UTILS] Enrichment build failed: {e}")
- L562:         logger.warning(f"[OUTPUT_UTILS] XLSX export failed: {e}")

## `C:/Users/edu-loaner/html_Parser_prototype/webapp/parser/utils/pattern_extractor.py`

- L22:         logger.warning(f"[PATTERN] load fail {e}")
- L87:             logger.warning(f"[PATTERN] pattern error {pat.get('name')}: {e}")

## `C:/Users/edu-loaner/html_Parser_prototype/webapp/parser/utils/pivot.py`

- L807:         logger.warning("[PIVOT] No candidates detected – verify headers and candidate column extraction.")

## `C:/Users/edu-loaner/html_Parser_prototype/webapp/parser/utils/shared_logger.py`

- L148:         elif record.levelno >= logging.WARNING:
- L149:             self.shared_logger.warning(msg)
- L207:             "WARNING": logging.WARNING,
- L280:                 "WARNING": "yellow",
- L342:     def warning(self, msg, context=None, exc_info=None):
- L344:         self._log("WARNING", msg, context, color="yellow")
- L358:             "warning": "yellow",
- L554:                 self.warning(f"Log directory does not exist: {log_dir}")
- L571:                         self.warning(f"Corrupt line in {path}: {e}")

## `C:/Users/edu-loaner/html_Parser_prototype/webapp/parser/utils/shared_logic.py`

- L209:         logger.warning(f"[safe_query] session.query({model}) failed: {e}")
- L232:             logger.warning(f"[safe_filter_by] No mapper found for model {model}")
- L238:         logger.warning(f"[safe_filter_by] Could not inspect model {model}: {e}")
- L252:         logger.warning(f"[safe_filter_by] filter_by failed: {e}")
- L265:         logger.warning(f"[safe_first] query.first() failed: {e}")
- L335:             logger.warning(f"[PLUGIN EXTRACTION] Plugin {plugin} has no callable 'extract' method.")
- L469:                 logger.warning(f"[WARN] Model save failed (attempt {attempt}): {e}")
- L682:                 try: logger.warning(f"[safe_append] Target is not a list: {type(lst)}; coercing to list.")
- L703:             logger.warning(f"[safe_update] Target is not a dict: {type(dct)}")
- L707:             logger.warning(f"[safe_update] Updates is not a dict: {type(updates)}")
- L726:                 try: logger.warning(f"[safe_extend] Target is not a list: {type(lst)}; coercing to list.")
- L1049:         logger.warning(f"[DOM_PARTS] '{label}' is not a list for URL: {url} (type: {type(lst).__name__})")
- L1320:             logger.warning(f"State '{state_norm}' not found in county map")
- L1751:             logger.warning(f"[inventory] architecture.md not found at {md_file}")
- L1757:             logger.warning("[inventory] Markers not found in architecture.md; aborting replace.")
- L1772:         logger.warning("[inventory] generate_project_map completed with warnings; check markers and path.")
- L1818:     """Find lines containing TODO/FIXME/WARN (case-insensitive). Returns list of (lineno, text)."""
- L1820:     pat = re.compile(r"\b(TODO|FIXME|WARN|WARNING)\b", re.IGNORECASE)
- L2182:         # TODO/FIXME/WARN
- L2185:             lines.append("- TODO/FIXME/WARN:")
- L2237:     """Aggregate TODO/FIXME/WARN lines from webapp/ into a compact index.
- L2246:         lines.append("# TODO/FIXME index — webapp\n")
- L2284:             logger.warning(f"[noise] No suggestions file found at {path}")

## `C:/Users/edu-loaner/html_Parser_prototype/webapp/parser/utils/strategy_concurrency.py`

- L35:             logger.warning(f"[CONCURRENCY] DOM strategy {name} failed: {e}")
- L63:                 logger.warning(f"[CONCURRENCY] Strategy {name} error: {e}")
- L71:         logger.warning(f"[CONCURRENCY] {_safe_run_strategy.__name__} {name} failed: {e}")
- L100:             logger.warning(f"[CONCURRENCY][ASYNC] DOM strategy {name} failed: {e}")
- L118:             logger.warning(f"[CONCURRENCY][ASYNC] Strategy {name} error: {e}")

## `C:/Users/edu-loaner/html_Parser_prototype/webapp/parser/utils/table_builder.py`

- L442:                     _emit("warning", "builder", "[TABLE_BUILDER] dynamic_table_extractor failed for panel table", session_id, error=str(e))
- L454:             _emit("warning", "builder", "[TABLE_BUILDER] dynamic_table_extractor failed (no panels path)", session_id, error=str(e))
- L462:         _emit("warning", "builder", "[TABLE_BUILDER] all_panel_tables was not a list; coercing to empty list", session_id, got_type=str(type(all_panel_tables)))
- L471:                 _emit("warning", "builder", "[TABLE_BUILDER] Dropping invalid table entry", session_id, entry_type=str(type(item)))
- L488:         _emit("warning", "builder", "[TABLE_BUILDER] sanitize failed", session_id, error=str(e))
- L493:         _emit("warning", "builder", "[TABLE_BUILDER] harmonize failed", session_id, error=str(e))
- L510:         _emit("warning", "builder", "[TABLE_BUILDER] entity annotate failed", session_id, error=str(e))
- L515:         _emit("warning", "builder", "[TABLE_BUILDER] stringify entity_info failed", session_id, error=str(e))
- L530:             _emit("warning", "builder", "[TABLE_BUILDER] pivot_to_wide failed", session_id, error=str(e))
- L822:                         _emit("warning", "builder", f"[TABLE_BUILDER] Column marked incorrect: {col_name}", session_id, contest=contest)
- L904:             _emit("warning", "builder", "[TABLE_BUILDER] Failed to persist table structure logs", session_id, error=str(e))

## `C:/Users/edu-loaner/html_Parser_prototype/webapp/parser/utils/table_core.py`

- L213:         logger.warning(f"[TABLE BUILDER] Concurrent strategies execution failed: {e}")
- L270:             logger.warning(f"[TABLE BUILDER] RawJSON pivot failed: {e}")
- L278:             logger.warning(f"[TABLE BUILDER] pivot_to_wide signature mismatch (skipped): {e}")
- L280:             logger.warning(f"[TABLE BUILDER] pivot_to_wide failed (skipped): {e}")
- L331:                 logger.warning(f"[TABLE BUILDER] finalize output failed: {e}")
- L396:         logger.warning(f"[TABLE BUILDER][ASYNC] Concurrent strategies execution failed: {e}")
- L459:                 logger.warning(f"[TABLE BUILDER][ASYNC] finalize output failed: {e}")

## `C:/Users/edu-loaner/html_Parser_prototype/webapp/parser/utils/user_prompt.py`

- L242:                 logger.warning("[UserPrompt] Webapp mode active but no socketio_emit_func set!")
- L279:             logger.warning("[CLI Prompt] EOFError encountered.")
- L300:             logger.warning("[Webapp Prompt] socketio_emit_func not set.")
- L358:             "WARNING": 30,
- L438:                 logger.warning("\n[Prompt] Timed out.")
- L486:                 logger.warning("\n[Prompt] No input available (EOF). Exiting prompt.")
- L516:                 logger.warning("Invalid input. Please try again.")
- L518:                     logger.warning("[Prompt] Too many invalid attempts.")
- L568:                     logger.warning("\n[Prompt] Timed out.")
- L775:                 logger.warning("[yellow][FEEDBACK] Skipped manual correction.[/yellow]")
- L807:             logger.warning("[yellow]Button confirmation cancelled by user.[/yellow]")

## `C:/Users/edu-loaner/html_Parser_prototype/webapp/parser/web_pipeline.py`

- L50:                 logger.warning({
- L51:                     "level": "WARNING",
- L67:                     logger.warning({
- L68:                         "level": "WARNING",
- L84:                     logger.warning({
- L85:                         "level": "WARNING",
