---
layout: default
title: "Comprehensive Pipeline Audit & Map"
---

Comprehensive pipeline audit for `webapp/parser/`.

## 📋 Table of Contents

- [Overview](#overview)
- [Interactive Pipeline Graph](#interactive-pipeline-graph)
- [File Connection Map](#file-connection-map)
- [Detailed Module Contexts](#detailed-module-contexts)

## Overview

- **Total Modules Audited:** 72
- **Total Connections:** 88
- **Clusters:** Entry, Pipeline, Routing, State Handlers, Format Handlers,
Shared Handlers, Services, Utils, Context Integration, Health
- **Audit Scope:** All `webapp/parser/` files with full context, imports,
dependencies, and optimization insights.

## Interactive Pipeline Graph

```mermaid
graph TD
  subgraph Entry["Entry"]
    html_election_parser["html_election_parser"]
  end
  subgraph Pipeline["Pipeline"]
    web_pipeline["web_pipeline"]
  end
  subgraph Routing["Routing"]
    state_router["state_router"]
  end
  subgraph State_Handlers["State Handlers"]
    example_county["example_county"]
    example_state["example_state"]
    rockland["rockland"]
  end
  subgraph Format_Handlers["Format Handlers"]
    pdf_handler["pdf_handler"]
    csv_handler["csv_handler"]
    html_handler["html_handler"]
    json_handler["json_handler"]
    xlsx_handler["xlsx_handler"]
  end
  subgraph Services["Services"]
    election_data_services["election_data_services"]
  end
  subgraph Utils["Utils"]
    browser_utils["browser_utils"]
    contest_selector["contest_selector"]
    detect["detect"]
    dynamic_table_extractor["dynamic_table_extractor"]
    html_scanner["html_scanner"]
    json_export_loader["json_export_loader"]
    models["models"]
    pattern_extractor["pattern_extractor"]
    pivot["pivot"]
    shared_logic["shared_logic"]
    table_builder["table_builder"]
    user_prompt["user_prompt"]
  end
  subgraph Context_Integration["Context Integration"]
    Integrity_check["Integrity_check"]
    context_coordinator["context_coordinator"]
    librarian["librarian"]
    loader["loader"]
    vocab_loader["vocab_loader"]
    constants["constants"]
    context_organizer["context_organizer"]
  end
  subgraph Health["Health"]
    manual_correction_bot["manual_correction_bot"]
    quarantine_queue["quarantine_queue"]
    session_branching["session_branching"]
    session_manager["session_manager"]
    dataset_promotion["dataset_promotion"]
    health_router["health_router"]
    integrity_check_runner["integrity_check_runner"]
    log_cache_cleaner_bot["log_cache_cleaner_bot"]
    promotion_helpers["promotion_helpers"]
    retrain_table_structure_models["retrain_table_structure_models"]
  end
  table_builder -->|37| dynamic_table_extractor
  manual_correction_bot -->|36| librarian
  detect -->|18| browser_utils
  loader -->|13| vocab_loader
  pivot -->|12| contest_selector
  pivot -->|11| json_export_loader
  dynamic_table_extractor -->|10| context_coordinator
  html_scanner -->|9| librarian
  user_prompt -->|9| shared_logic
  pattern_extractor -->|7| browser_utils
  election_data_services -->|6| models
  html_scanner -->|6| context_coordinator
  verification_endpoints -->|5| local_dl_sync
  html_election_parser -->|4| quarantine_queue
  web_pipeline -->|4| session_branching
  pdf_handler -->|4| config
  session_manager -->|4| session_branching
  table_builder -->|4| pivot
  table_builder -->|4| context_coordinator
  html_election_parser -->|3| Integrity_check
```

**✨ Legend:** Colors indicate module categories with metallic accents. Click
nodes for details below.

## Connection Highlights

Key integration points across major parser aspects to simplify tracking
relevance.

### Top Module Links

- `table_builder` → `dynamic_table_extractor` (37 refs, Utils → Utils) —
review `dynamic_table_extractor` whenever `table_builder` changes.
- `manual_correction_bot` → `librarian` (36 refs, Health → Context
Integration) — review `librarian` whenever `manual_correction_bot` changes.
- `detect` → `browser_utils` (18 refs, Utils → Utils) — review `browser_utils`
whenever `detect` changes.
- `loader` → `vocab_loader` (13 refs, Context Integration → Context
Integration) — review `vocab_loader` whenever `loader` changes.
- `pivot` → `contest_selector` (12 refs, Utils → Utils) — review
`contest_selector` whenever `pivot` changes.
- `pivot` → `json_export_loader` (11 refs, Utils → Utils) — review
`json_export_loader` whenever `pivot` changes.
- `dynamic_table_extractor` → `context_coordinator` (10 refs, Utils → Context
Integration) — review `context_coordinator` whenever `dynamic_table_extractor`
changes.
- `html_scanner` → `librarian` (9 refs, Utils → Context Integration) — review
`librarian` whenever `html_scanner` changes.
- `user_prompt` → `shared_logic` (9 refs, Utils → Utils) — review
`shared_logic` whenever `user_prompt` changes.
- `pattern_extractor` → `browser_utils` (7 refs, Utils → Utils) — review
`browser_utils` whenever `pattern_extractor` changes.

### Cluster Flow Summary

- Utils → Utils: 124 edges (intra-cluster flow to monitor.)
- Health → Context Integration: 39 edges (cross-cluster flow to monitor.)
- Utils → Context Integration: 38 edges (cross-cluster flow to monitor.)
- Context Integration → Context Integration: 14 edges (intra-cluster flow to
monitor.)
- Format Handlers → Other: 11 edges (cross-cluster flow to monitor.)
- Other → Other: 7 edges (intra-cluster flow to monitor.)
- Health → Entry: 7 edges (cross-cluster flow to monitor.)
- Entry → Context Integration: 6 edges (cross-cluster flow to monitor.)
- Services → Utils: 6 edges (cross-cluster flow to monitor.)
- Pipeline → Health: 5 edges (cross-cluster flow to monitor.)

## File Connection Map

Detailed import/export relationships and dependencies.

## Detailed Module Contexts

Click to expand each module for full audit details.

### Context\_Integration/Context\_Library/constants.py {#webapp-parser-context-integration-context-library-constants-py}

#### 🔧 Key Functions & Classes (Context_Integration_Context_Library_constants)

- `build_state_to_division_type_map` (function, line 691)
- `get_party_code_info` (function, line 1358)
- `_sanitize_party_token` (function, line 2573)
- `normalize_party_code` (function, line 2592)
- `canonical_ballot_group` (function, line 2619)
- `split_and_normalize_ballot_groups` (function, line 2646)
- `normalize_result_group_label` (function, line 2665)
- `normalize_party_label` (function, line 2683)
- `is_pseudo_result_party` (function, line 2713)
- `_iter_strings` (function, line 2884)
- `_compile_union` (function, line 2895)
- `_norm_state_key` (function, line 2938)
- `_norm_county_key` (function, line 2949)
- `_collect_layered_patterns` (function, line 2958)
- `get_camelot_title_regex` (function, line 2969)
- `get_camelot_row_regex` (function, line 2979)
- `build_camelot_row_filter` (function, line 2992)

#### 📦 Key Imports (Context_Integration_Context_Library_constants)

- `re`
- `functools`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`

#### ⚠️ Task markers (Context_Integration_Context_Library_constants)

- L2005 **NOTE**: ._$",                     # Note
- L2194 **WARNING**: ",
- L2285 **WARNING**: ", "info_box", "navigation", "pagination", "tab",
"modal", "tooltip", "ignore", "unknown"
- L2318 **NOTE**: ", "comment",
- L2394 **NOTE**: ", "Comment", "Feedback", "Suggestion", "Recommendation",
- L2410 **NOTE**: ", "Comment", "Feedback", "Suggestion",

### Context\_Integration/Integrity\_check.py {#webapp-parser-context-integration-integrity-check-py}

#### 🔧 Key Functions & Classes (Context_Integration_Integrity_check)

- `_trim_monitor_log` (function, line 47)
- `log_integrity_monitor` (function, line 70)
- `_ensure_alerts_table` (function, line 80)
- `find_date_anomalies` (function, line 87)
- `detect_anomalies_with_ml` (function, line 95)
- `election_integrity_checks` (function, line 182)
- `advanced_cross_field_validation` (function, line 203)
- `summarize_context_entities` (function, line 212)
- `analyze_contests` (function, line 221)
- `auto_tune_contamination` (function, line 267)
- `print_issues_table` (function, line 288)
- `print_entity_summary` (function, line 308)
- `print_ml_anomalies` (function, line 316)
- `print_date_anomalies` (function, line 346)
- `print_auto_tune_result` (function, line 364)
- `print_analyze_contests` (function, line 370)
- `monitor_db_for_alerts` (function, line 382)
- `log_integrity_issues` (function, line 428)
- `detect_statistical_outliers` (function, line 444)
- `print_integrity_summary` (function, line 480)

#### 📦 Key Imports (Context_Integration_Integrity_check)

- `__future__`
- `re`
- `threading`
- `time`
- `collections`
- `pathlib`
- `typing`
- `typing`
- `typing`
- `typing`
- `matplotlib`
- `numpy`
- `orjson`
- `rich.panel`
- `rich.table`
- `sklearn.cluster`
- `sklearn.ensemble`
- `sklearn.preprocessing`
- `sqlalchemy`
- `config`

### Context\_Integration/\_\_init\_\_.py {#webapp-parser-context-integration-init-py}

> Context integration module for election results.

### Context\_Integration/context\_coordinator.py {#webapp-parser-context-integration-context-coordinator-py}

> context*coordinator.py

#### 🔧 Key Functions & Classes (Context_Integration_context_coordinator)

- `get_semantic_score` (function, line 97)
- `merge_and_rank_candidates` (function, line 166)
- `dynamic_state_county_detection` (function, line 256)
- `ContextCoordinator` (class, line 857)

#### 📦 Key Imports (Context_Integration_context_coordinator)

- `__future__`
- `difflib`
- `numbers`
- `os`
- `re`
- `subprocess`
- `threading`
- `collections`
- `collections`
- `datetime`
- `datetime`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `numpy`
- `orjson`
- `rapidfuzz`

#### ⚠️ Task markers (Context_Integration_context_coordinator)

- L896 **WARNING**: ("\[ALERT MONITOR\] Thread did not stop cleanly.")
- L984 **WARNING**: ({
- L985 **WARNING**: ",
- L1383 **WARNING**: (f"\[yellow\]Integrity issues:\[/yellow\]
{issues\['integrity_issues'\]}")
- L1622 **WARNING**: (f"\[ContextCoordinator\] No table structure found for
contest: {contest}")
- L1799 **WARNING**: (f"\[get_feedback_pattern_kb\] Skipping corrupt line:
{e}")
- L1911 **WARNING**: ("\[group_dom_nodes_by_label\] No organized DOM parts.
(Further warnings suppressed)")
- L1913 **WARNING**: (f"\[group_dom_nodes_by_label\] No organized DOM parts.
(Occurred {ContextCoordinator._dom_parts_warning_count} times)")
- L1918 **WARNING**: ("\[group_dom_nodes_by_label\] No DOM nodes found.")
- L1936 **WARNING**: ("\[submit_user_feedback\] ContextOrganizer has no
submit_user_feedback method.")
- L1964 **WARNING**: (f"\[correct_and_update_contest\] Contest {contest_id}
missing type/election_types after sync.")
- L1988 **WARNING**: ("\[print_contest_summary\] No organized contests to
summarize.")
- L2001 **WARNING**: ("\[plot_contest_distribution\] No organized contests to
plot.")
- L2052 **WARNING**: ("No organized DOM parts.")
- L2055 **WARNING**: ("No organized DOM parts. (Further warnings suppressed)")
- L2066 **WARNING**: ("\[get_contest_groups\] No contest groups found.")
- L2075 **WARNING**: ("\[get_panel_groups\] No panel groups found.")
- L2084 **WARNING**: ("\[get_button_groups\] No button groups found.")
- L2093 **WARNING**: ("\[get_table_groups\] No table groups found.")
- L2102 **WARNING**: ("\[get_relationships\] No organized context.")

### Context\_Integration/context\_organizer.py {#webapp-parser-context-integration-context-organizer-py}

> context*organizer.py

#### 🔧 Key Functions & Classes (Context_Integration_context_organizer)

- `get_loading_indicator` (function, line 63)
- `ensure_dict` (function, line 66)
- `remove_functions` (function, line 79)
- `contest_hash` (function, line 87)
- `repair_dom_segments` (function, line 99)
- `_defensive_dom_check` (function, line 161)
- `ContextOrganizer` (class, line 182)

#### 📦 Key Imports (Context_Integration_context_organizer)

- `__future__`
- `itertools`
- `os`
- `re`
- `types`
- `collections`
- `collections`
- `collections.abc`
- `datetime`
- `datetime`
- `difflib`
- `typing`
- `matplotlib.pyplot`
- `numpy`
- `orjson`
- `rich.table`
- `sqlalchemy.exc`
- `config`
- `config`
- `config`

#### ⚠️ Task markers (Context_Integration_context_organizer)

- L282 **WARNING**: (
- L407 **WARNING**: (f"\[CONTEST\] Skipping contest with suspiciously large or
missing title: {str(title)\[:100\]}...")
- L495 **WARNING**: (f"\[CONTEST\] Filtered out {len(filtered_out)} contests
due to missing required fields.")
- L497 **WARNING**: (f"  \[Filtered\] {reason}: {str(c)\[:100\]}...")
- L500 **WARNING**: ("\[CONTEST\] No contests with required fields for
downstream output.")
- L816 **WARNING**: (f"\[ML\] Anomaly index {idx} out of range for contests
list of length {len(contests)}")
- L1602 **WARNING**: (f"  \[yellow\]{title}\[/yellow\]: {fixes}")
- L1608 **WARNING**: (f"\[bold yellow\]\[INTEGRITY\]\[/bold yellow\] Duplicate
contest detected.\n \[dim\]Context:\[/dim\] {contest}")
- L1610 **WARNING**: (f"\[bold yellow\]\[INTEGRITY\]\[/bold yellow\] Contest
missing location info.\n \[dim\]Context:\[/dim\] {contest}")
- L1612 **WARNING**: (f"\[bold yellow\]\[INTEGRITY\]\[/bold yellow\] Contest
missing year.\n \[dim\]Context:\[/dim\] {contest}")
- L2082 **WARNING**: (f"\[ContextOrganizer\] Could not update context library
with feedback: {e}")
- L2159 **WARNING**: (f"\[CONTEXT ORGANIZER\] No table structure found for
contest: {contest}")

### Context\_Integration/librarian.py {#webapp-parser-context-integration-librarian-py}

#### 🔧 Key Functions & Classes (Context_Integration_librarian)

- `safe_path` (function, line 74)
- `get_safe_log_path` (function, line 103)
- `atomic_write_json` (function, line 125)
- `extend_panel_tags` (function, line 188)
- `extend_heading_tags` (function, line 192)
- `extend_html_tags` (function, line 196)
- `extend_custom_attr_patterns` (function, line 200)
- `extend_location_keywords` (function, line 208)
- `extend_candidate_keywords` (function, line 212)
- `extend_ballot_types` (function, line 216)
- `safe_join` (function, line 220)
- `clean_for_json` (function, line 236)
- `robust_orjson_loads` (function, line 252)
- `load_context_library` (function, line 260)
- `update_context_library` (function, line 352)
- `backup_context_library` (function, line 368)
- `save_context_library` (function, line 426)
- `merge_and_save_context_library` (function, line 480)
- `update_context_library_field` (function, line 489)
- `update_domain_selector_cache` (function, line 501)
- `get_domain_selectors` (function, line 522)
- `log_selector_attempt` (function, line 527)
- `_get_log_path` (function, line 551)
- `_deduplicate_jsonl_log` (function, line 567)
- `log_unknown_tag` (function, line 602)

#### 📦 Key Imports (Context_Integration_librarian)

- `__future__`
- `argparse`
- `os`
- `re`
- `shutil`
- `subprocess`
- `sys`
- `tempfile`
- `threading`
- `time`
- `datetime`
- `datetime`
- `pathlib`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `numpy`
- `orjson`

#### 💬 Top-of-file Comments (Context_Integration_librarian)

```python

# webapp/parser/Context\_Integration/librarian.py

# -----------------------------------------------------------------------------------

# This file contains functions to manage the context library for the HTML parser,

# including loading, saving, and updating the context library, as well as

# It also includes utilities for logging unknown HTML tags and attributes,

# extending context library structures, and handling ML/LLM feedback.

#

# SECURITY: All file operations are validated using safe\_path() to prevent path traversal attacks.

# -----------------------------------------------------------------------------------

```

#### ⚠️ Task markers (Context_Integration_librarian)

- L763 **WARNING**: (f"\n\[LIBRARIAN SELF-HEAL\] Attempt {attempt}...")
- L773 **WARNING**: ("\[LIBRARIAN SELF-HEAL\] Misalignments found. Launching
manual_correction...")
- L776 **WARNING**: (f"\[LIBRARIAN SELF-HEAL\] Sleeping {cooldown}s before
rescanning...")

### Context\_Integration/library/entity\_confidence\_map.py {#webapp-parser-context-integration-library-entity-confidence-map-py}

> Entity Confidence Mapping: Weighted Signal Catalog for Decision Gates

#### 🔧 Key Functions & Classes (Context_Integration_library_entity_confidence_map)

- `DecisionCode` (class, line 23)
- `SignalType` (class, line 30)
- `AnomalyType` (class, line 44)
- `OverrideTrigger` (class, line 56)
- `SignalCoefficient` (class, line 67)
- `AnomalyCoefficient` (class, line 77)
- `ConfidenceCautionResult` (class, line 87)
- `EntityConfidenceMap` (class, line 289)
- `get_confidence_map` (function, line 468)

#### 📦 Key Imports (Context_Integration_library_entity_confidence_map)

- `__future__`
- `dataclasses`
- `enum`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `orjson`

### Context\_Integration/location\_inference.py {#webapp-parser-context-integration-location-inference-py}

#### 🔧 Key Functions & Classes (Context_Integration_location_inference)

- `infer_county_from_lines` (function, line 11)

#### 📦 Key Imports (Context_Integration_location_inference)

- `__future__`
- `re`
- `collections`
- `typing`
- `typing`
- `utils.shared_logic`
- `utils.shared_logic`
- `Context_Library.constants`

### Context\_Integration/vocab/loader.py {#webapp-parser-context-integration-vocab-loader-py}

> Vocab Loader: Safe, audited vocabulary file management for confidence/caution
framework.

#### 🔧 Key Functions & Classes (Context_Integration_vocab_loader)

- `VocabLoaderError` (class, line 33)
- `VocabSecurityError` (class, line 38)
- `VocabFileNotFound` (class, line 43)
- `VocabIntegrityError` (class, line 48)
- `RateLimitError` (class, line 53)
- `VocabLoader` (class, line 68)
- `get_vocab_loader` (function, line 356)

#### 📦 Key Imports (Context_Integration_vocab_loader)

- `hashlib`
- `os`
- `time`
- `pathlib`
- `typing`
- `typing`
- `typing`
- `typing`
- `webapp.parser.utils.logger_singleton`

### Context\_Integration/vocab\_loader.py {#webapp-parser-context-integration-vocab-loader-py}

> VocabLoader: Secure, auditable vocabulary file loader for election integrity.

#### 🔧 Key Functions & Classes (Context_Integration_vocab_loader)

- `VocabLoaderError` (class, line 22)
- `VocabFileNotFound` (class, line 27)
- `VocabIntegrityError` (class, line 32)
- `VocabSecurityError` (class, line 37)
- `RateLimitError` (class, line 42)
- `VocabLoader` (class, line 47)
- `get_vocab_loader` (function, line 413)

#### 📦 Key Imports (Context_Integration_vocab_loader)

- `__future__`
- `hashlib`
- `os`
- `threading`
- `time`
- `datetime`
- `datetime`
- `pathlib`
- `typing`
- `typing`
- `typing`
- `typing`
- `webapp.parser.utils.logger_singleton`

### config.py {#webapp-parser-config-py}

> Central configuration module for the Smart Elections Parser Webapp.

#### 🔧 Key Functions & Classes (config)

- `get_subprocess_env` (function, line 336)
- `get_supported_formats` (function, line 345)
- `get_sqlalchemy_engine` (function, line 381)
- `get_ocr_config_dict` (function, line 616)
- `log_ocr_config_summary` (function, line 668)
- `build_extraction_quality_metrics` (function, line 686)
- `log_extraction_quality` (function, line 881)

#### 📦 Key Imports (config)

- `os`
- `threading`
- `urllib.parse`
- `pathlib`
- `orjson`
- `psycopg2`
- `azure.identity`
- `sqlalchemy`
- `utils.logger_singleton`

#### ⚠️ Task markers (config)

- L911 **WARNING**: ({
- L912 **WARNING**: ",
- L930 **NOTE**: Both DL1 and DL2 are now stored in
CONTEXT_LIBRARY_DIR/verification

### config/\_ocr\_helpers.py {#webapp-parser-config-ocr-helpers-py}

> OCR Configuration Helper Functions

#### 🔧 Key Functions & Classes (config__ocr_helpers)

- `get_ocr_config_dict` (function, line 8)
- `log_ocr_config_summary` (function, line 43)

### config/ocr\_tuning.py {#webapp-parser-config-ocr-tuning-py}

> OCR Tuning Parameters — Centralized Configuration

#### 🔧 Key Functions & Classes (config_ocr_tuning)

- `OcrTuningConfig` (class, line 46)

#### 📦 Key Imports (config_ocr_tuning)

- `os`
- `typing`

### data\_manager.py {#webapp-parser-data-manager-py}

#### 🔧 Key Functions & Classes (data_manager)

- `_ensure_parent` (function, line 9)
- `_atomic_write_lines` (function, line 16)
- `load_urls` (function, line 25)
- `save_urls` (function, line 41)
- `add_url` (function, line 60)
- `remove_url` (function, line 75)
- `replace_urls` (function, line 96)
- `list_urls_cli` (function, line 99)
- `list_files` (function, line 109)
- `copy_file_to_folder` (function, line 133)
- `run_manager` (function, line 147)

#### 📦 Key Imports (data_manager)

- `os`
- `re`
- `config`
- `config`
- `config`
- `utils.logger_singleton`
- `utils.logger_singleton`
- `utils.logger_singleton`

### election\_fixtures.py {#webapp-parser-election-fixtures-py}

> Election results fixture loader with lazy caching (mirrors fec*lookup.py
pattern).

#### 🔧 Key Functions & Classes (election_fixtures)

- `_get_fixture_dir` (function, line 39)
- `load_election_results_index` (function, line 44)
- `load_election_results_shards` (function, line 79)
- `get_results_by_state` (function, line 113)
- `get_results_by_contest` (function, line 168)
- `find_candidate_by_name` (function, line 209)
- `get_cache_metrics` (function, line 285)
- `clear_cache` (function, line 291)
- `reset_metrics` (function, line 301)

#### 📦 Key Imports (election_fixtures)

- `json`
- `os`
- `threading`
- `functools`
- `pathlib`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`

### fec\_lookup.py {#webapp-parser-fec-lookup-py}

#### 🔧 Key Functions & Classes (fec_lookup)

- `_normalize_name` (function, line 17)
- `load_fec_candidates` (function, line 38)
- `get_candidate_by_id` (function, line 56)
- `_build_name_index` (function, line 63)
- `find_candidate_by_name` (function, line 79)

#### 📦 Key Imports (fec_lookup)

- `__future__`
- `json`
- `os`
- `typing`
- `typing`
- `typing`
- `config`
- `config`

### handlers/batch\_handler.py {#webapp-parser-handlers-batch-handler-py}

#### 🔧 Key Functions & Classes (handlers_batch_handler)

- `_normalize_label` (function, line 14)
- `BatchProcessor` (class, line 24)

#### 📦 Key Imports (handlers_batch_handler)

- `__future__`
- `copy`
- `time`
- `uuid`
- `concurrent.futures`
- `concurrent.futures`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `utils.logger_singleton`
- `utils.logger_singleton`
- `utils.shared_logic`
- `utils.shared_logic`
- `utils.shared_logic`
- `utils.shared_logic`
- `utils.user_prompt`

#### ⚠️ Task markers (handlers_batch_handler)

- L134 **WARNING**: ({
- L135 **WARNING**: ",
- L426 **WARNING**: ({
- L427 **WARNING**: ",

### handlers/fec\_handler.py {#webapp-parser-handlers-fec-handler-py}

#### 🔧 Key Functions & Classes (handlers_fec_handler)

- `parse` (function, line 22)

#### 📦 Key Imports (handlers_fec_handler)

- `__future__`
- `csv`
- `os`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `webapp.parser.fec_lookup`
- `webapp.parser.fec_lookup`
- `webapp.parser.utils.fec_utils`
- `webapp.parser.utils.fec_utils`
- `webapp.parser.utils.fec_utils`
- `webapp.parser.utils.fec_utils`
- `webapp.parser.utils.fec_utils`

### handlers/formats/csv\_handler.py {#webapp-parser-handlers-formats-csv-handler-py}

#### 🔧 Key Functions & Classes (handlers_formats_csv_handler)

- `parse_csv_election_results` (function, line 44)
- `parse` (function, line 327)

#### 📦 Key Imports (handlers_formats_csv_handler)

- `__future__`
- `csv`
- `os`
- `re`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `config`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.librarian`
- `utils.contest_detection`
- `utils.contest_detection`
- `utils.contest_detection`
- `utils.contest_selector`
- `utils.header_utils`
- `utils.location_helpers`
- `utils.location_helpers`

### handlers/formats/download\_finder.py {#webapp-parser-handlers-formats-download-finder-py}

#### 🔧 Key Functions & Classes (handlers_formats_download_finder)

- `find_download_links` (function, line 9)

#### 📦 Key Imports (handlers_formats_download_finder)

- `__future__`
- `typing`
- `typing`
- `urllib.parse`
- `utils.logger_singleton`

#### ⚠️ Task markers (handlers_formats_download_finder)

- L45 **WARNING**:
({"level":"WARNING","type":"download_finder","message":f"Download finder
failed: {e}","session_id":session_id})

### handlers/formats/html\_dynamic\_fallback.py {#webapp-parser-handlers-formats-html-dynamic-fallback-py}

#### 🔧 Key Functions & Classes (handlers_formats_html_dynamic_fallback)

- `parse` (function, line 9)

#### 📦 Key Imports (handlers_formats_html_dynamic_fallback)

- `__future__`
- `typing`
- `typing`
- `typing`
- `webapp.parser.html_election_parser`
- `webapp.parser.utils.logger_singleton`

### handlers/formats/html\_handler.py {#webapp-parser-handlers-formats-html-handler-py}

#### 🔧 Key Functions & Classes (handlers_formats_html_handler)

- `_attempt_generic_fallback` (function, line 19)
- `parse` (function, line 80)

#### 📦 Key Imports (handlers_formats_html_handler)

- `__future__`
- `importlib`
- `os`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `orjson`
- `Context_Integration.Context_Library.constants`
- `state_router`
- `state_router`
- `state_router`
- `utils.contest_selector`
- `utils.logger_singleton`
- `utils.logger_singleton`
- `utils.shared_logic`
- `utils.shared_logic`
- `utils.shared_logic`

#### ⚠️ Task markers (handlers_formats_html_handler)

- L216 **WARNING**: (f"\[HTML Handler\] County '{county}' not found. Closest
matches: {matches}")
- L220 **WARNING**: (f"\[HTML Handler\] Detected county '{county}' is not in
known counties for state '{suggested_state or state}'.")
- L241 **WARNING**: (f"\[HTML Handler\] State '{user_state}' not found.
Closest matches: {matches}")
- L285 **WARNING**: (f"\[HTML Handler\] County '{user_county}' not found.
Closest matches: {matches}")

### handlers/formats/json\_handler.py {#webapp-parser-handlers-formats-json-handler-py}

#### 🔧 Key Functions & Classes (handlers_formats_json_handler)

- `_build_contest_regex` (function, line 53)
- `_canonical_contest_key` (function, line 86)
- `_split_primary_title_for_grouping` (function, line 91)
- `_format_county_preview` (function, line 121)
- `_format_scope_label` (function, line 148)
- `_collect_contest_groups` (function, line 168)
- `find_key_by_keywords` (function, line 290)
- `_is_dict_list` (function, line 308)
- `_state_key_for_county` (function, line 313)
- `_extract_first_str` (function, line 324)
- `_derive_location_metadata` (function, line 332)
- `_fastpath_county_results` (function, line 360)
- `parse_json_election_results` (function, line 977)
- `parse` (function, line 1350)

#### 📦 Key Imports (handlers_formats_json_handler)

- `__future__`
- `os`
- `re`
- `collections`
- `collections`
- `collections`
- `pathlib`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `orjson`
- `config`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.Context_Library.constants`

#### ⚠️ Task markers (handlers_formats_json_handler)

- L377 **WARNING**: ({
- L378 **WARNING**: ",
- L501 **WARNING**: ({
- L502 **WARNING**: ",

### handlers/formats/pdf\_handler.py {#webapp-parser-handlers-formats-pdf-handler-py}

#### 🔧 Key Functions & Classes (handlers_formats_pdf_handler)

- `_env_truthy` (function, line 190)
- `PDFParseCancelled` (class, line 212)
- `_cleanup_pdf_resources` (function, line 216)
- `_register_pdf_cleanup` (function, line 261)
- `_sanitize_cache_get` (function, line 270)
- `_sanitize_cache_set` (function, line 281)
- `_normalize_angle` (function, line 292)
- `_quantize_angle` (function, line 300)
- `_collect_page_orientation` (function, line 310)
- `_get_page_orientation_map` (function, line 390)
- `_log_orientation_application` (function, line 454)
- `_apply_page_orientation` (function, line 467)
- `_expand_focus_windows` (function, line 498)
- `_normalize_contest_key` (function, line 524)
- `_contest_title_tokens` (function, line 531)
- `_ensure_not_cancelled` (function, line 537)
- `_cancelled_result` (function, line 598)
- `_estimate_ocr_time_budgets` (function, line 623)
- `_refine_focus_windows_for_contest` (function, line 634)
- `_focus_windows_from_line_records` (function, line 677)
- `_merge_focus_windows` (function, line 735)
- `_autopick_contest_from_probe` (function, line 762)
- `_compute_sample_page_indices` (function, line 811)
- `_contest_probe_scan` (function, line 843)
- `_yield_full_pass_batches` (function, line 943)

#### 📦 Key Imports (handlers_formats_pdf_handler)

- `__future__`
- `os`
- `re`
- `csv`
- `time`
- `math`
- `platform`
- `shutil`
- `importlib`
- `hashlib`
- `atexit`
- `gc`
- `tempfile`
- `typing`
- `collections`
- `collections`
- `collections`
- `concurrent.futures`
- `PIL`
- `PIL`

#### ⚠️ Task markers (handlers_formats_pdf_handler)

- L1004 **WARNING**: ({
- L1005 **WARNING**: ",
- L1007 **WARN**: \] Skipping page {page_index} during OCR batch render:
{exc}",
- L1160 **WARNING**: ({
- L1161 **WARNING**: ",
- L1164 **WARN**: \] Detected PyMuPDF %s. Upgrade to %s or newer to avoid
parser instability."
- L2756 **WARNING**: ({
- L2757 **WARNING**: ",
- L2759 **WARN**: \] Poppler binaries not detected; skipping pdf2image and
using PyMuPDF fallback.",
- L2799 **WARNING**: ({
- L2800 **WARNING**: ",
- L2803 **WARN**: \] pdf2image conversion failed; "
- L3187 **WARNING**: ({
- L3188 **WARNING**: ",
- L3190 **WARN**: \] Skipping full-document OCR pass due to expired sample
budget.",
- L3238 **WARNING**: ({
- L3239 **WARNING**: ",
- L3241 **WARN**: \] Aborting full-document OCR pass due to timeout budget.",
- L3268 **WARNING**: ({
- L3269 **WARNING**: ",

### handlers/formats/txt\_handler.py {#webapp-parser-handlers-formats-txt-handler-py}

#### 🔧 Key Functions & Classes (handlers_formats_txt_handler)

- `_read_delimited_file` (function, line 44)
- `parse_txt_election_results` (function, line 75)
- `parse` (function, line 327)

#### 📦 Key Imports (handlers_formats_txt_handler)

- `__future__`
- `csv`
- `os`
- `re`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `config`
- `Context_Integration.Context_Library.constants`
- `utils.contest_detection`
- `utils.contest_detection`
- `utils.contest_detection`
- `utils.contest_selector`
- `utils.location_helpers`
- `utils.location_helpers`
- `utils.logger_singleton`
- `utils.output_utils`

### handlers/formats/xlsx\_handler.py {#webapp-parser-handlers-formats-xlsx-handler-py}

#### 🔧 Key Functions & Classes (handlers_formats_xlsx_handler)

- `_dataframe_to_records` (function, line 48)
- `parse_xlsx_election_results` (function, line 65)
- `parse` (function, line 354)

#### 📦 Key Imports (handlers_formats_xlsx_handler)

- `__future__`
- `os`
- `re`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `config`
- `Context_Integration.Context_Library.constants`
- `utils.contest_detection`
- `utils.contest_detection`
- `utils.contest_detection`
- `utils.contest_selector`
- `utils.location_helpers`
- `utils.location_helpers`
- `utils.logger_singleton`
- `utils.output_utils`
- `utils.pivot`

### handlers/states/alabama/alabama.py {#webapp-parser-handlers-states-alabama-alabama-py}

#### 🔧 Key Functions & Classes (handlers_states_alabama_alabama)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_alabama_alabama)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/alaska/alaska.py {#webapp-parser-handlers-states-alaska-alaska-py}

#### 🔧 Key Functions & Classes (handlers_states_alaska_alaska)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_alaska_alaska)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/american\_samoa/american\_samoa.py {#webapp-parser-handlers-states-american-samoa-american-samoa-py}

#### 🔧 Key Functions & Classes (handlers_states_american_samoa_american_samoa)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_american_samoa_american_samoa)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/arizona/\_\_init\_\_.py {#webapp-parser-handlers-states-arizona-init-py}

#### 📦 Key Imports (handlers_states_arizona___init__)

- `arizona`

### handlers/states/arizona/arizona.py {#webapp-parser-handlers-states-arizona-arizona-py}

#### 🔧 Key Functions & Classes (handlers_states_arizona_arizona)

- `parse` (function, line 33)

#### 📦 Key Imports (handlers_states_arizona_arizona)

- `os`
- `orjson`
- `config`
- `Context_Integration.context_organizer`
- `utils.logger_singleton`
- `utils.output_utils`

#### 💬 Top-of-file Comments (handlers_states_arizona_arizona)

```python

# handlers/arizona.py

# ==============================================================

# Handler for Arizona election result sites with expandable cards

# and toggles between 'Vote Type' and 'By County' views.

# ==============================================================

```

#### ⚠️ Task markers (handlers_states_arizona_arizona)

- L25 **WARNING**: ("\[WARN\] context_library.json not found. Using fallback
config for Arizona handler.")
- L51 **WARNING**: (f"\[WARN\] Could not expand card {i+1}: {e}")
- L64 **WARNING**: (f"\[WARN\] Vote Type toggle failed: {e}")
- L77 **WARNING**: (f"\[WARN\] County toggle failed: {e}")
- L164 **WARNING**: ("\[FALLBACK\] No tables were parsed. Either no results
are published yet or the structure has changed.")
- L165 **WARNING**: ("\[FALLBACK\] Please verify that the site has posted
election data.")

### handlers/states/arkansas/arkansas.py {#webapp-parser-handlers-states-arkansas-arkansas-py}

#### 🔧 Key Functions & Classes (handlers_states_arkansas_arkansas)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_arkansas_arkansas)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/california/california.py {#webapp-parser-handlers-states-california-california-py}

#### 🔧 Key Functions & Classes (handlers_states_california_california)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_california_california)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/colorado/colorado.py {#webapp-parser-handlers-states-colorado-colorado-py}

#### 🔧 Key Functions & Classes (handlers_states_colorado_colorado)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_colorado_colorado)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/connecticut/connecticut.py {#webapp-parser-handlers-states-connecticut-connecticut-py}

#### 🔧 Key Functions & Classes (handlers_states_connecticut_connecticut)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_connecticut_connecticut)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/delaware/delaware.py {#webapp-parser-handlers-states-delaware-delaware-py}

#### 🔧 Key Functions & Classes (handlers_states_delaware_delaware)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_delaware_delaware)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/district\_of\_columbia/district\_of\_columbia.py {#webapp-parser-handlers-states-district-of-columbia-district-of-columbia-py}

#### 🔧 Key Functions & Classes (handlers_states_district_of_columbia_district_of_columbia)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_district_of_columbia_district_of_columbia)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/example state/example\_county/example\_county.py {#webapp-parser-handlers-states-example-state-example-county-example-county-py}

#### 🔧 Key Functions & Classes (handlers_states_example state_example_county_example_county)

- `parse` (function, line 16)
- `parse_single_contest_dynamic` (function, line 75)

#### 📦 Key Imports (handlers_states_example state_example_county_example_county)

- `typing`
- `playwright.sync_api`
- `utils.contest_selector`
- `utils.html_scanner`
- `utils.logger_singleton`
- `utils.output_utils`
- `utils.table_builder`
- `utils.table_core`

#### ⚠️ Task markers (handlers_states_example state_example_county_example_county)

- L123 **WARNING**: ("\[yellow\]\[WARNING\] No ballot items found by div
selectors. Trying table-based extraction...\[/yellow\]")

### handlers/states/example state/example\_state.py {#webapp-parser-handlers-states-example-state-example-state-py}

#### 🔧 Key Functions & Classes (handlers_states_example state_example_state)

- `parse` (function, line 24)
- `parse_single_contest_dynamic` (function, line 104)

#### 📦 Key Imports (handlers_states_example state_example_state)

- `importlib`
- `typing`
- `playwright.sync_api`
- `utils.contest_selector`
- `utils.html_scanner`
- `utils.logger_singleton`
- `utils.output_utils`
- `utils.shared_logic`
- `utils.shared_logic`
- `utils.shared_logic`
- `utils.shared_logic`
- `utils.table_builder`
- `utils.table_core`

#### ⚠️ Task markers (handlers_states_example state_example_state)

- L51 **WARNING**: (f"\[Example Handler\] No specific parser implemented for
county: '{county}'. Continuing with state-level logic.")
- L152 **WARNING**: ("\[yellow\]\[WARNING\] No ballot items found by div
selectors. Trying table-based extraction...\[/yellow\]")

### handlers/states/florida/florida.py {#webapp-parser-handlers-states-florida-florida-py}

#### 🔧 Key Functions & Classes (handlers_states_florida_florida)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_florida_florida)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/georgia/georgia.py {#webapp-parser-handlers-states-georgia-georgia-py}

#### 🔧 Key Functions & Classes (handlers_states_georgia_georgia)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_georgia_georgia)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/guam/guam.py {#webapp-parser-handlers-states-guam-guam-py}

#### 🔧 Key Functions & Classes (handlers_states_guam_guam)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_guam_guam)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/hawaii/hawaii.py {#webapp-parser-handlers-states-hawaii-hawaii-py}

#### 🔧 Key Functions & Classes (handlers_states_hawaii_hawaii)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_hawaii_hawaii)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/idaho/idaho.py {#webapp-parser-handlers-states-idaho-idaho-py}

#### 🔧 Key Functions & Classes (handlers_states_idaho_idaho)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_idaho_idaho)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/illinois/illinois.py {#webapp-parser-handlers-states-illinois-illinois-py}

#### 🔧 Key Functions & Classes (handlers_states_illinois_illinois)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_illinois_illinois)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/indiana/indiana.py {#webapp-parser-handlers-states-indiana-indiana-py}

#### 🔧 Key Functions & Classes (handlers_states_indiana_indiana)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_indiana_indiana)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/iowa/iowa.py {#webapp-parser-handlers-states-iowa-iowa-py}

#### 🔧 Key Functions & Classes (handlers_states_iowa_iowa)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_iowa_iowa)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/kansas/kansas.py {#webapp-parser-handlers-states-kansas-kansas-py}

#### 🔧 Key Functions & Classes (handlers_states_kansas_kansas)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_kansas_kansas)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/kentucky/kentucky.py {#webapp-parser-handlers-states-kentucky-kentucky-py}

#### 🔧 Key Functions & Classes (handlers_states_kentucky_kentucky)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_kentucky_kentucky)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/louisiana/louisiana.py {#webapp-parser-handlers-states-louisiana-louisiana-py}

#### 🔧 Key Functions & Classes (handlers_states_louisiana_louisiana)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_louisiana_louisiana)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/maine/maine.py {#webapp-parser-handlers-states-maine-maine-py}

#### 🔧 Key Functions & Classes (handlers_states_maine_maine)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_maine_maine)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/maryland/maryland.py {#webapp-parser-handlers-states-maryland-maryland-py}

#### 🔧 Key Functions & Classes (handlers_states_maryland_maryland)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_maryland_maryland)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/massachusetts/massachusetts.py {#webapp-parser-handlers-states-massachusetts-massachusetts-py}

#### 🔧 Key Functions & Classes (handlers_states_massachusetts_massachusetts)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_massachusetts_massachusetts)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/michigan/michigan.py {#webapp-parser-handlers-states-michigan-michigan-py}

#### 🔧 Key Functions & Classes (handlers_states_michigan_michigan)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_michigan_michigan)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/minnesota/minnesota.py {#webapp-parser-handlers-states-minnesota-minnesota-py}

#### 🔧 Key Functions & Classes (handlers_states_minnesota_minnesota)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_minnesota_minnesota)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/mississippi/mississippi.py {#webapp-parser-handlers-states-mississippi-mississippi-py}

#### 🔧 Key Functions & Classes (handlers_states_mississippi_mississippi)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_mississippi_mississippi)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/missouri/missouri.py {#webapp-parser-handlers-states-missouri-missouri-py}

#### 🔧 Key Functions & Classes (handlers_states_missouri_missouri)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_missouri_missouri)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/montana/montana.py {#webapp-parser-handlers-states-montana-montana-py}

#### 🔧 Key Functions & Classes (handlers_states_montana_montana)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_montana_montana)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/nebraska/nebraska.py {#webapp-parser-handlers-states-nebraska-nebraska-py}

#### 🔧 Key Functions & Classes (handlers_states_nebraska_nebraska)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_nebraska_nebraska)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/nevada/nevada.py {#webapp-parser-handlers-states-nevada-nevada-py}

#### 🔧 Key Functions & Classes (handlers_states_nevada_nevada)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_nevada_nevada)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/new\_hampshire/new\_hampshire.py {#webapp-parser-handlers-states-new-hampshire-new-hampshire-py}

#### 🔧 Key Functions & Classes (handlers_states_new_hampshire_new_hampshire)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_new_hampshire_new_hampshire)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/new\_jersey/new\_jersey.py {#webapp-parser-handlers-states-new-jersey-new-jersey-py}

#### 🔧 Key Functions & Classes (handlers_states_new_jersey_new_jersey)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_new_jersey_new_jersey)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/new\_mexico/new\_mexico.py {#webapp-parser-handlers-states-new-mexico-new-mexico-py}

#### 🔧 Key Functions & Classes (handlers_states_new_mexico_new_mexico)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_new_mexico_new_mexico)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/new\_york/county/rockland.py {#webapp-parser-handlers-states-new-york-county-rockland-py}

#### 🔧 Key Functions & Classes (handlers_states_new_york_county_rockland)

- `parse` (function, line 27)

#### 📦 Key Imports (handlers_states_new_york_county_rockland)

- `typing`
- `playwright.sync_api`
- `Context_Integration.librarian`
- `utils.browser_utils`
- `utils.browser_utils`
- `utils.browser_utils`
- `utils.browser_utils`
- `utils.contest_selector`
- `utils.html_scanner`
- `utils.logger_singleton`
- `utils.logger_singleton`
- `utils.output_utils`
- `utils.shared_logic`
- `utils.table_builder`
- `utils.table_core`

#### ⚠️ Task markers (handlers_states_new_york_county_rockland)

- L72 **WARNING**: ("\[WARNING\] dom_parts missing after
organize_and_enrich.")
- L95 **WARNING**: ("\[red\]No contest selected. Skipping.\[/red\]")
- L139 **WARNING**: (f"\[yellow\]\[WARNING\] Button '{btn1.get('label', '')}'
is not clickable (visible={safe_is_visible(element, logger)},
enabled={safe_is_enabled(element, logger)})\[/yellow\]")
- L176 **WARNING**: (f"\[yellow\]\[WARNING\] Button '{btn2.get('label', '')}'
is not clickable (visible={safe_is_visible(element, logger)},
enabled={safe_is_enabled(element, logger)})\[/yellow\]")

### handlers/states/new\_york/new\_york.py {#webapp-parser-handlers-states-new-york-new-york-py}

#### 🔧 Key Functions & Classes (handlers_states_new_york_new_york)

- `parse` (function, line 15)

#### 📦 Key Imports (handlers_states_new_york_new_york)

- `importlib`
- `typing`
- `typing`
- `typing`
- `playwright.sync_api`
- `utils.logger_singleton`
- `utils.shared_logic`
- `utils.shared_logic`
- `utils.shared_logic`
- `utils.shared_logic`

#### ⚠️ Task markers (handlers_states_new_york_new_york)

- L27 **WARNING**: ("\[NY Handler\] No county specified in html_context.")
- L43 **WARNING**: (f"\[NY Handler\] No specific parser implemented for
county: '{county}'. Please add it under {module_path}.py")

### handlers/states/north\_carolina/north\_carolina.py {#webapp-parser-handlers-states-north-carolina-north-carolina-py}

#### 🔧 Key Functions & Classes (handlers_states_north_carolina_north_carolina)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_north_carolina_north_carolina)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/north\_dakota/north\_dakota.py {#webapp-parser-handlers-states-north-dakota-north-dakota-py}

#### 🔧 Key Functions & Classes (handlers_states_north_dakota_north_dakota)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_north_dakota_north_dakota)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/northern\_mariana\_islands/northern\_mariana\_islands.py {#webapp-parser-handlers-states-northern-mariana-islands-northern-mariana-islands-py}

#### 🔧 Key Functions & Classes (handlers_states_northern_mariana_islands_northern_mariana_islands)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_northern_mariana_islands_northern_mariana_islands)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/ohio/ohio.py {#webapp-parser-handlers-states-ohio-ohio-py}

#### 🔧 Key Functions & Classes (handlers_states_ohio_ohio)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_ohio_ohio)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/oklahoma/oklahoma.py {#webapp-parser-handlers-states-oklahoma-oklahoma-py}

#### 🔧 Key Functions & Classes (handlers_states_oklahoma_oklahoma)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_oklahoma_oklahoma)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/oregon/oregon.py {#webapp-parser-handlers-states-oregon-oregon-py}

#### 🔧 Key Functions & Classes (handlers_states_oregon_oregon)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_oregon_oregon)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/pennsylvania/\_\_init\_\_.py {#webapp-parser-handlers-states-pennsylvania-init-py}

#### 📦 Key Imports (handlers_states_pennsylvania___init__)

- `pennsylvania`

### handlers/states/pennsylvania/pennsylvania.py {#webapp-parser-handlers-states-pennsylvania-pennsylvania-py}

#### 🔧 Key Functions & Classes (handlers_states_pennsylvania_pennsylvania)

- `apply_navigation_steps` (function, line 25)
- `parse` (function, line 46)

#### 📦 Key Imports (handlers_states_pennsylvania_pennsylvania)

- `csv`
- `os`
- `pathlib`
- `config`
- `utils.browser_utils`
- `utils.browser_utils`
- `utils.browser_utils`
- `utils.browser_utils`
- `utils.logger_singleton`
- `utils.output_utils`
- `utils.shared_logic`
- `utils.shared_logic`
- `utils.shared_logic`
- `utils.shared_logic`
- `utils.shared_logic`

#### ⚠️ Task markers (handlers_states_pennsylvania_pennsylvania)

- L44 **WARNING**: (f"\[NAV\] Step failed: {step} — {e}")
- L55 **WARNING**: (f"\[bold yellow\]Detected election:\[/bold yellow\]
{header_text}")
- L76 **WARNING**: ("\[PA\] Invalid index input for election selection.")
- L78 **WARNING**: ("\[PA\] Elections dropdown not found.")
- L80 **WARNING**: (f"\[PA\] Failed to expand Elections menu or load
selection: {e}")
- L96 **WARNING**: ("\[PA\] County Breakdown link not found.")
- L98 **WARNING**: (f"\[PA\] Failed to click County Breakdown link: {e}")
- L113 **WARNING**: ("\[yellow\]Multiple CSV files found in input. Please
select one:\[/yellow\]")

### handlers/states/puerto\_rico/puerto\_rico.py {#webapp-parser-handlers-states-puerto-rico-puerto-rico-py}

#### 🔧 Key Functions & Classes (handlers_states_puerto_rico_puerto_rico)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_puerto_rico_puerto_rico)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/rhode\_island/rhode\_island.py {#webapp-parser-handlers-states-rhode-island-rhode-island-py}

#### 🔧 Key Functions & Classes (handlers_states_rhode_island_rhode_island)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_rhode_island_rhode_island)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/south\_carolina/south\_carolina.py {#webapp-parser-handlers-states-south-carolina-south-carolina-py}

#### 🔧 Key Functions & Classes (handlers_states_south_carolina_south_carolina)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_south_carolina_south_carolina)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/south\_dakota/south\_dakota.py {#webapp-parser-handlers-states-south-dakota-south-dakota-py}

#### 🔧 Key Functions & Classes (handlers_states_south_dakota_south_dakota)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_south_dakota_south_dakota)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/tennessee/tennessee.py {#webapp-parser-handlers-states-tennessee-tennessee-py}

#### 🔧 Key Functions & Classes (handlers_states_tennessee_tennessee)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_tennessee_tennessee)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/texas/texas.py {#webapp-parser-handlers-states-texas-texas-py}

#### 🔧 Key Functions & Classes (handlers_states_texas_texas)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_texas_texas)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/us\_virgin\_islands/us\_virgin\_islands.py {#webapp-parser-handlers-states-us-virgin-islands-us-virgin-islands-py}

#### 🔧 Key Functions & Classes (handlers_states_us_virgin_islands_us_virgin_islands)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_us_virgin_islands_us_virgin_islands)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/utah/utah.py {#webapp-parser-handlers-states-utah-utah-py}

#### 🔧 Key Functions & Classes (handlers_states_utah_utah)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_utah_utah)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/vermont/vermont.py {#webapp-parser-handlers-states-vermont-vermont-py}

#### 🔧 Key Functions & Classes (handlers_states_vermont_vermont)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_vermont_vermont)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/virginia/virginia.py {#webapp-parser-handlers-states-virginia-virginia-py}

#### 🔧 Key Functions & Classes (handlers_states_virginia_virginia)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_virginia_virginia)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/washington/washington.py {#webapp-parser-handlers-states-washington-washington-py}

#### 🔧 Key Functions & Classes (handlers_states_washington_washington)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_washington_washington)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/west\_virginia/west\_virginia.py {#webapp-parser-handlers-states-west-virginia-west-virginia-py}

#### 🔧 Key Functions & Classes (handlers_states_west_virginia_west_virginia)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_west_virginia_west_virginia)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/wisconsin/wisconsin.py {#webapp-parser-handlers-states-wisconsin-wisconsin-py}

#### 🔧 Key Functions & Classes (handlers_states_wisconsin_wisconsin)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_wisconsin_wisconsin)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### handlers/states/wyoming/wyoming.py {#webapp-parser-handlers-states-wyoming-wyoming-py}

#### 🔧 Key Functions & Classes (handlers_states_wyoming_wyoming)

- `parse` (function, line 8)

#### 📦 Key Imports (handlers_states_wyoming_wyoming)

- `__future__`
- `typing`
- `typing`
- `webapp.parser.handlers.formats.html_dynamic_fallback`

### health/context\_migration.py {#webapp-parser-health-context-migration-py}

#### 🔧 Key Functions & Classes (health_context_migration)

- `table_structure_exists` (function, line 30)
- `create_table_structure` (function, line 37)
- `migrate_table_structures_from_jsonl` (function, line 50)
- `migrate_table_structures_from_json` (function, line 74)
- `load_migration_state` (function, line 109)
- `save_migration_state` (function, line 115)
- `_normalize_geo` (function, line 120)
- `_coerce_year` (function, line 128)
- `_ensure_contest_for_snapshot` (function, line 137)
- `migrate_context_snapshot_from_metadata` (function, line 191)
- `migrate_all` (function, line 281)
- `migrate_context_cache_to_db` (function, line 320)

#### 📦 Key Imports (health_context_migration)

- `datetime`
- `datetime`
- `pathlib`
- `typing`
- `typing`
- `typing`
- `orjson`
- `config`
- `config`
- `config`
- `config`
- `Context_Integration.librarian`
- `utils.db_utils`
- `utils.db_utils`
- `utils.db_utils`
- `utils.html_scanner`
- `utils.logger_singleton`
- `utils.models`
- `utils.models`
- `utils.models`

### health/dataset\_promotion.py {#webapp-parser-health-dataset-promotion-py}

#### 🔧 Key Functions & Classes (health_dataset_promotion)

- `discover_dataset_dirs` (function, line 68)
- `resolve_dataset_path` (function, line 80)
- `_load_metadata` (function, line 95)
- `_load_rows` (function, line 102)
- `_has_value` (function, line 111)
- `_match_field` (function, line 119)
- `_coerce_text` (function, line 138)
- `_coerce_votes` (function, line 145)
- `_resolve_election_date` (function, line 169)
- `build_warehouse_records` (function, line 194)
- `promote_dataset` (function, line 243)
- `_build_arg_parser` (function, line 351)
- `main` (function, line 375)

#### 📦 Key Imports (health_dataset_promotion)

- `__future__`
- `argparse`
- `csv`
- `datetime`
- `datetime`
- `pathlib`
- `typing`
- `typing`
- `typing`
- `orjson`
- `webapp.parser.config`
- `webapp.parser.Context_Integration.librarian`
- `webapp.parser.utils.db_utils`
- `webapp.parser.utils.db_utils`
- `webapp.parser.utils.db_utils`
- `webapp.parser.utils.logger_singleton`
- `webapp.parser.utils.models`
- `webapp.parser.health.promotion_helpers`
- `webapp.parser.health.promotion_helpers`

#### ⚠️ Task markers (health_dataset_promotion)

- L295 **WARNING**: (f"\[PROMOTE\] Skipping blocked URL: {source_url}")

### health/health\_config.py {#webapp-parser-health-health-config-py}

> health*config.py

#### 📦 Key Imports (health_health_config)

- `pathlib`
- `config`
- `config`
- `config`

### health/health\_router.py {#webapp-parser-health-health-router-py}

#### 🔧 Key Functions & Classes (health_health_router)

- `LocalLearningEngine` (class, line 74)
- `get_learning_engine` (function, line 131)
- `register_orchestration_plugin` (function, line 140)
- `run_orchestration_plugins` (function, line 143)
- `preclean_json_logs` (function, line 152)
- `BotPipeline` (class, line 207)

#### 📦 Key Imports (health_health_router)

- `errno`
- `glob`
- `os`
- `re`
- `subprocess`
- `sys`
- `time`
- `datetime`
- `datetime`
- `pathlib`
- `orjson`
- `sqlalchemy`
- `config`
- `config`
- `config`
- `config`
- `config`
- `config`
- `config`
- `config`

#### ⚠️ Task markers (health_health_router)

- L99 **WARNING**: (f"\[LocalLearning\] Failed to record training signal:
{e}")
- L337 **WARNING**: (f"\[health_router\] manual_correction failed (attempt
{attempt}): {result.stderr}")
- L421 **WARNING**: ("\[SELF-HEAL\] Misalignments found. Launching
manual_correction...")
- L423 **WARNING**: (f"\[SELF-HEAL\] Sleeping {cooldown}s before
rescanning...")
- L425 **WARNING**: ("\[SELF-HEAL\] Max retries reached. Some misalignments
may remain.")
- L460 **WARNING**: (f"\[PIPELINE\] Could not fix corrupted JSON files: {e}")
- L477 **WARNING**: ("\[PIPELINE\] Misaligned NER examples found. Self-heal
loop will be handled by scan_misaligned_ner.")
- L479 **WARNING**: ("\[PIPELINE\] scan_misaligned_ner failed or file missing.
Proceeding with caution.")
- L511 **WARNING**: ("\[PIPELINE\] Model retraining failed.")

### health/integrity\_check\_runner.py {#webapp-parser-health-integrity-check-runner-py}

#### 🔧 Key Functions & Classes (health_integrity_check_runner)

- `load_contests` (function, line 13)
- `run_integrity_summary` (function, line 23)
- `_build_arg_parser` (function, line 45)
- `main` (function, line 68)

#### 📦 Key Imports (health_integrity_check_runner)

- `__future__`
- `argparse`
- `pathlib`
- `typing`
- `webapp.parser.config`
- `webapp.parser.Context_Integration.Integrity_check`
- `webapp.parser.Context_Integration.librarian`
- `webapp.parser.utils.logger_singleton`

#### ⚠️ Task markers (health_integrity_check_runner)

- L18 **WARNING**: ("\[INTEGRITY\] Context library at %s is missing contest
data", context_path)

### health/integrity\_monitor.py {#webapp-parser-health-integrity-monitor-py}

> integrity*monitor.py

#### 🔧 Key Functions & Classes (health_integrity_monitor)

- `IntegrityNeuralNetwork` (class, line 59)
- `HuggingFaceNLPAnalyzer` (class, line 96)
- `IntegrityMonitor` (class, line 203)
- `get_integrity_monitor` (function, line 542)

#### 📦 Key Imports (health_integrity_monitor)

- `__future__`
- `asyncio`
- `hashlib`
- `time`
- `datetime`
- `datetime`
- `pathlib`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `orjson`
- `config`
- `config`
- `config`
- `Context_Integration.librarian`
- `Context_Integration.librarian`
- `utils.logger_singleton`

#### ⚠️ Task markers (health_integrity_monitor)

- L264 **WARNING**: (f"\[IntegrityMonitor\] Hash mismatch for
{file_path.name}: expected {expected_hash}, got {file_hash}")

### health/log\_cache\_cleaner\_bot.py {#webapp-parser-health-log-cache-cleaner-bot-py}

> log*cache*cleaner*bot.py

#### 🔧 Key Functions & Classes (health_log_cache_cleaner_bot)

- `is_jsonl_file` (function, line 44)
- `is_json_file` (function, line 47)
- `is_html_file` (function, line 50)
- `safe_path` (function, line 53)
- `log_empty_entry` (function, line 61)
- `clean_jsonl` (function, line 72)
- `clean_json` (function, line 175)
- `clean_html` (function, line 295)
- `human_size` (function, line 388)
- `clean_dir` (function, line 395)
- `run_db_maintenance` (function, line 441)
- `run_log_cache_cleaner` (function, line 486)
- `schedule_log_cache_cleaner` (function, line 520)
- `main` (function, line 530)

#### 📦 Key Imports (health_log_cache_cleaner_bot)

- `argparse`
- `os`
- `threading`
- `time`
- `pathlib`
- `orjson`
- `sqlalchemy`
- `sqlalchemy.exc`
- `config`
- `config`
- `config`
- `utils.db_utils`
- `utils.logger_singleton`
- `context_migration`

#### ⚠️ Task markers (health_log_cache_cleaner_bot)

- L151 **WARNING**: (f"Skipping non-dict entry in spacy_ner_train_data.jsonl:
{entry}")
- L460 **WARNING**: ("\[DB\]\[WARNING\] No user tables found in schema
'public'.")
- L503 **WARNING**: ("\[CLEAN\]\[WARNING\] The following files are still too
large after cleaning:")
- L507 **WARNING**: ("\[MISALIGNED\] Consider cleaning or pattern-excluding
these from your training data:")

### health/manual\_correction\_bot.py {#webapp-parser-health-manual-correction-bot-py}

> manual*correction.py

#### 🔧 Key Functions & Classes (health_manual_correction_bot)

- `safe_path` (function, line 76)
- `load_cache` (function, line 105)
- `close_cache` (function, line 120)
- `write_audit_log` (function, line 124)
- `process_logs_with_cache` (function, line 139)
- `process_and_sync` (function, line 151)
- `discover_field_types_from_logs` (function, line 195)
- `atomic_write_json` (function, line 228)
- `llm_suggest_action` (function, line 297)
- `ml_score_entry` (function, line 349)
- `ml_suggest_field` (function, line 372)
- `load_jsonl` (function, line 391)
- `check_and_fix_json_files` (function, line 407)
- `find_log_files` (function, line 569)
- `load_jsonl_incremental` (function, line 636)
- `save_jsonl` (function, line 654)
- `deduplicate_entries` (function, line 667)
- `entry_key` (function, line 681)
- `aggregate_successful_field_entries` (function, line 692)
- `feedback_loop` (function, line 733)
- `trim_log_file` (function, line 821)
- `update_context_with_new_entries` (function, line 828)
- `validate_context_schema` (function, line 845)
- `extract_year` (function, line 870)
- `extract_state` (function, line 884)

#### 📦 Key Imports (health_manual_correction_bot)

- `argparse`
- `importlib`
- `os`
- `re`
- `shelve`
- `shutil`
- `subprocess`
- `sys`
- `time`
- `collections`
- `collections`
- `datetime`
- `datetime`
- `pathlib`
- `openai`
- `orjson`
- `config`
- `config`
- `config`
- `config`

#### ⚠️ Task markers (health_manual_correction_bot)

- L361 **WARNING**: (f"Coordinator ML scoring failed: {e}")
- L382 **WARNING**: (f"Coordinator field suggestion failed: {e}")
- L395 **WARNING**: (f"Log file not found: {path}")
- L404 **WARNING**: (f"\[CORRUPT\] {path} line {i}: {e}")
- L434 **WARNING**: (f"\[SECURITY\] Skipping invalid directory: {directory} -
{e}")
- L448 **WARNING**: (f"\[SECURITY\] Skipping file outside allowed directories:
{file} - {e}")
- L454 **WARNING**: (f"\[SKIP\] File not found: {file}")
- L458 **WARNING**: (f"\[SKIP\] File too large: {file}")
- L483 **WARNING**: (f"\[CORRUPT-LINE\] {file} line {i+1}: {line\[:80\]}...
({e})")
- L497 **WARNING**: (f"\[CORRUPT\] {len(corrupt_items)} lines saved to
{corrupt_path}")
- L502 **WARNING**: (f"\[FIXED\] All lines invalid, recreated empty .jsonl
file: {file}")
- L516 **WARNING**: (f"\[CORRUPT\] {file}: {e}")
- L530 **WARNING**: (f"\[CORRUPT\] Corrupt JSON saved to {corrupt_path}")
- L536 **WARNING**: (f"\[FIXED\] All content invalid, recreated minimal valid
JSON in {file}")
- L541 **WARNING**: (f"\[CORRUPT\] {file}: {e}")
- L555 **WARNING**: (f"\[QUARANTINED\] {file} -&gt; {dest_path}")
- L559 **WARNING**: (f"\[DELETED\] {file}")
- L562 **WARNING**: (f"\[SKIP-DELETE\] File already missing: {file}")
- L597 **WARNING**: (f"\[SECURITY\] Skipping invalid directory: {d} - {e}")
- L615 **WARNING**: (f"\[SECURITY\] Skipping file outside allowed directories:
{f} - {e}")

### health/navigation\_feedback\_ingest.py {#webapp-parser-health-navigation-feedback-ingest-py}

#### 🔧 Key Functions & Classes (health_navigation_feedback_ingest)

- `ingest_navigation_feedback` (function, line 24)
- `_read_offset` (function, line 66)
- `_write_offset` (function, line 75)
- `_format_entry` (function, line 82)

#### 📦 Key Imports (health_navigation_feedback_ingest)

- `__future__`
- `pathlib`
- `typing`
- `typing`
- `orjson`

### health/promotion\_helpers.py {#webapp-parser-health-promotion-helpers-py}

> Helper functions for dataset promotion with verification gating.

#### 🔧 Key Functions & Classes (health_promotion_helpers)

- `check_exact_duplicate` (function, line 8)
- `get_url_verification_tier` (function, line 33)

#### 📦 Key Imports (health_promotion_helpers)

- `webapp.parser.utils.logger_singleton`

#### ⚠️ Task markers (health_promotion_helpers)

- L54 **WARNING**: (f"\[URL_TIER\] Failed to compute trust score: {exc}")

### health/quarantine\_queue.py {#webapp-parser-health-quarantine-queue-py}

> Quarantine Queue: Transparent URL quarantine workflow with audit trails.

#### 🔧 Key Functions & Classes (health_quarantine_queue)

- `QuarantineReason` (class, line 37)
- `ReviewStatus` (class, line 77)
- `DataCollectionNotice` (class, line 89)
- `QuarantineEntry` (class, line 101)
- `QuarantineQueue` (class, line 181)
- `get_quarantine_queue` (function, line 444)

#### 📦 Key Imports (health_quarantine_queue)

- `__future__`
- `hashlib`
- `json`
- `os`
- `threading`
- `time`
- `dataclasses`
- `dataclasses`
- `dataclasses`
- `datetime`
- `datetime`
- `enum`
- `pathlib`
- `typing`
- `typing`
- `typing`
- `typing`
- `config`
- `utils.logger_singleton`

#### ⚠️ Task markers (health_quarantine_queue)

- L295 **WARNING**: ({
- L296 **WARNING**: ",

### health/retrain\_table\_structure\_models.py {#webapp-parser-health-retrain-table-structure-models-py}

#### 🔧 Key Functions & Classes (health_retrain_table_structure_models)

- `NERPipeProtocol` (class, line 86)
- `MakeDocProtocol` (class, line 89)
- `normalize_entity` (function, line 98)
- `normalize_entity_list` (function, line 103)
- `update_advanced_entities` (function, line 107)
- `is_misaligned_text` (function, line 152)
- `clean_misaligned_ner_jsonl` (function, line 158)
- `append_training_data` (function, line 218)
- `save_training_data_jsonl` (function, line 244)
- `cluster_container_patterns` (function, line 257)
- `auto_label_header` (function, line 297)
- `extract_candidates_from_context` (function, line 319)
- `entity_frequency_analysis` (function, line 327)
- `update_db_with_new_entities` (function, line 334)
- `load_spacy_ner_examples` (function, line 350)
- `remove_overlapping_entities` (function, line 366)
- `validate_training_data` (function, line 388)
- `retrain_spacy_ner_advanced` (function, line 411)
- `get_all_confirmed_structures` (function, line 648)
- `run_manual_correction` (function, line 670)
- `retrain_sentence_transformer` (function, line 689)
- `segment_hash` (function, line 786)
- `load_cached_segment_hashes` (function, line 796)
- `scan_in_memory_ner_examples` (function, line 800)
- `ensure_table_structures_exists` (function, line 817)

#### 📦 Key Imports (health_retrain_table_structure_models)

- `copy`
- `datetime`
- `gc`
- `glob`
- `hashlib`
- `os`
- `random`
- `re`
- `shutil`
- `subprocess`
- `sys`
- `collections`
- `importlib.util`
- `types`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`

#### ⚠️ Task markers (health_retrain_table_structure_models)

- L178 **WARNING**: (f"\[CLEAN\] File not found: {jsonl_path}")
- L186 **WARNING**: (f"\[CLEAN\] Could not parse line: {e}")
- L201 **WARNING**: (f"\[CLEAN\] Alignment check failed for text:
{text\[:50\]}... ({e})")
- L274 **WARNING**: (f"Failed to load {path}: {e}")
- L403 **WARNING**: (f"Skipping misaligned entity in: {text}")
- L408 **WARNING**: (f"Error validating entity alignment: {e}")
- L434 **WARNING**: (f"\[spaCy\] Could not check GPU availability: {e}")
- L450 **WARNING**: (f"\[spaCy\] Could not load lexeme normalization table.
You may ignore this for English. Error: {e}")
- L536 **WARNING**: (f"\[NER\] Skipped {misaligned_count} misaligned examples.
Saved to {misaligned_path}")
- L550 **WARNING**: ("No NER training examples found. Skipping spaCy NER
retraining.")
- L619 **WARNING**: ("\[SUGGESTION\] Consider lowering min_delta or increasing
patience if you want longer training.")
- L621 **WARNING**: ("\[SUGGESTION\] Model improved until the last epoch.
Consider increasing epochs for further improvement.")
- L622 **WARNING**: (f"\[SUGGESTION\] Next run: patience={patience},
min_delta={min_delta:.2f}, epochs={epochs}")
- L708 **WARNING**: ("No training examples found. Aborting retraining.")
- L727 **WARNING**: (f"\[WARN\] Could not delete old model directory
{oldest_path}: {e}")
- L739 **WARNING**: (f"\[WARN\] Failed to load existing model: {e}")
- L742 **WARNING**: ("Falling back to base model (all-MiniLM-L6-v2).")
- L782 **WARNING**: (f"\[WARN\] Could not update canonical model directory:
{e}")
- L810 **WARNING**: (f"MISALIGNED: {text} {annots\['entities'\]}")
- L840 **WARNING**: ("\[DB\] Base.metadata.tables is empty. No models
registered? Did you import all model classes?")

### health/scan\_misaligned\_ner.py {#webapp-parser-health-scan-misaligned-ner-py}

#### 🔧 Key Functions & Classes (health_scan_misaligned_ner)

- `resolve_jsonl_path` (function, line 15)
- `scan_misaligned` (function, line 22)
- `self_heal_loop` (function, line 101)
- `main` (function, line 125)

#### 📦 Key Imports (health_scan_misaligned_ner)

- `os`
- `subprocess`
- `sys`
- `time`
- `pathlib`
- `orjson`
- `spacy`
- `spacy.training`
- `config`
- `config`
- `utils.logger_singleton`

#### ⚠️ Task markers (health_scan_misaligned_ner)

- L62 **WARNING**: (f"\[CORRUPT\] Could not parse line: {e}")
- L83 **WARNING**: (f"\n\[MISALIGNED\] Top {top_n} most frequent misaligned
NER texts:")
- L85 **WARNING**: (f"  {repr(text)}: {count} times")
- L86 **WARNING**: ("\[MISALIGNED\] Consider cleaning or pattern-excluding
these from your training data.")
- L87 **WARNING**: ("Run the manual_correction to review and clean these
examples before retraining.")
- L88 **WARNING**: ("If you see spaCy entity alignment warnings, consider
cleaning your training data or using the provided validation function.")
- L98 **WARNING**: (f"\[WARN\] Could not remove old misaligned file: {e}")
- L112 **WARNING**: ("\[SELF-HEAL\] Misalignments found. Launching
manual_correction for spacy_ner_misaligned...")
- L119 **WARNING**: (f"\[SELF-HEAL\] manual_correction exited with code
{result.returncode}")
- L120 **WARNING**: (f"\[SELF-HEAL\] Sleeping {cooldown}s before
rescanning...")
- L122 **WARNING**: ("\[SELF-HEAL\] Max retries reached. Some misalignments
may remain.")

### health/session\_branching.py {#webapp-parser-health-session-branching-py}

> Session Branching and Multi-Tenant Isolation for Smart Elections Parser

#### 🔧 Key Functions & Classes (health_session_branching)

- `SessionBranch` (class, line 22)
- `get_isolated_branch` (function, line 153)
- `validate_url_access` (function, line 171)
- `add_url_to_isolation` (function, line 229)
- `get_isolation_summary` (function, line 263)
- `list_all_isolation_branches` (function, line 278)
- `cleanup_principal_isolation` (function, line 291)

#### 📦 Key Imports (health_session_branching)

- `__future__`
- `threading`
- `typing`
- `typing`
- `typing`
- `utils.logger_singleton`
- `utils.privilege_tiers`
- `utils.privilege_tiers`

#### ⚠️ Task markers (health_session_branching)

- L219 **WARNING**: ({
- L220 **WARNING**: ",
- L284 **WARNING**:     WARNING:

### health/session\_manager.py {#webapp-parser-health-session-manager-py}

#### 🔧 Key Functions & Classes (health_session_manager)

- `SessionManager` (class, line 15)

#### 📦 Key Imports (health_session_manager)

- `__future__`
- `os`
- `time`
- `datetime`
- `datetime`
- `queue`
- `threading`
- `threading`
- `typing`
- `typing`
- `typing`
- `typing`
- `webapp.parser.utils.session_state`
- `webapp.parser.utils.session_state`
- `webapp.parser.utils.session_state`

### html\_election\_parser.py {#webapp-parser-html-election-parser-py}

#### 🔧 Key Functions & Classes (html_election_parser)

- `_close_browser_quietly` (function, line 83)
- `_count_dom_table_rows` (function, line 105)
- `load_urls` (function, line 136)
- `mark_url_processed` (function, line 196)
- `prompt_url_selection` (function, line 257)
- `process_format_override` (function, line 425)
- `ai_analyze_results` (function, line 621)
- `stream_results` (function, line 721)
- `_read_text_file_with_fallback` (function, line 768)
- `_extract_text_blocks` (function, line 784)
- `generate_generic_html_result` (function, line 972)
- `orchestrate_url` (function, line 1198)
- `_orchestrate_url_worker` (function, line 2098)
- `main` (function, line 2115)

#### 📦 Key Imports (html_election_parser)

- `__future__`
- `os`
- `re`
- `threading`
- `collections`
- `collections`
- `datetime`
- `multiprocessing`
- `typing`
- `typing`
- `typing`
- `orjson`
- `psycopg2`
- `playwright.sync_api`
- `sqlalchemy.exc`
- `config`
- `config`
- `config`
- `config`
- `config`

#### ⚠️ Task markers (html_election_parser)

- L76 **WARNING**: ("Deleting .processed_urls cache for fresh start...")
- L94 **WARNING**: ({
- L95 **WARNING**: ",
- L520 **WARNING**: ({
- L521 **WARNING**: ",
- L535 **WARNING**: ({
- L536 **WARNING**: ",
- L598 **WARNING**: ({
- L599 **WARNING**: ",
- L698 **WARNING**: (payload_2)
- L1026 **WARNING**: ({
- L1027 **WARNING**: ",
- L1073 **WARNING**: ({
- L1074 **WARNING**: ",
- L1127 **WARNING**: ({
- L1128 **WARNING**: ",
- L1288 **WARNING**: ({
- L1289 **WARNING**: ",
- L1344 **WARNING**: ({
- L1345 **WARNING**: ",

### navigator/\_\_init\_\_.py {#webapp-parser-navigator-init-py}

> Dynamic navigation recipes for Smart Elections Parser.

#### 📦 Key Imports (navigator___init__)

- `navigation_recipes`
- `navigation_recipes`
- `navigation_runner`

### navigator/dom\_snapshot.py {#webapp-parser-navigator-dom-snapshot-py}

> DOM Snapshot Mode for Medium-Trust URLs

#### 🔧 Key Functions & Classes (navigator_dom_snapshot)

- `capture_dom_snapshot` (function, line 31)
- `extract_tables_from_snapshot` (function, line 123)
- `snapshot_mode_pipeline` (function, line 282)

#### 📦 Key Imports (navigator_dom_snapshot)

- `__future__`
- `time`
- `typing`
- `typing`
- `typing`
- `typing`
- `utils.logger_singleton`
- `utils.telemetry`

#### ⚠️ Task markers (navigator_dom_snapshot)

- L78 **WARNING**: ({
- L79 **WARNING**: ",
- L147 **WARNING**: ({
- L148 **WARNING**: ",
- L201 **WARNING**: ({
- L202 **WARNING**: ",

### navigator/keyword\_bias.py {#webapp-parser-navigator-keyword-bias-py}

#### 🔧 Key Functions & Classes (navigator_keyword_bias)

- `_iter_lines` (function, line 16)
- `load_keyword_bias` (function, line 35)

#### 📦 Key Imports (navigator_keyword_bias)

- `__future__`
- `threading`
- `pathlib`
- `typing`
- `typing`
- `typing`
- `orjson`

### navigator/navigation\_recipes.py {#webapp-parser-navigator-navigation-recipes-py}

#### 🔧 Key Functions & Classes (navigator_navigation_recipes)

- `NavigationRecipeStore` (class, line 12)

#### 📦 Key Imports (navigator_navigation_recipes)

- `__future__`
- `threading`
- `pathlib`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `orjson`

### navigator/navigation\_runner.py {#webapp-parser-navigator-navigation-runner-py}

#### 🔧 Key Functions & Classes (navigator_navigation_runner)

- `NavigationResult` (class, line 18)
- `NavigationInstructionRunner` (class, line 26)

#### 📦 Key Imports (navigator_navigation_runner)

- `__future__`
- `threading`
- `concurrent.futures`
- `concurrent.futures`
- `dataclasses`
- `typing`
- `typing`
- `typing`
- `typing`
- `utils.browser_utils`
- `utils.browser_utils`
- `utils.html_scanner`
- `utils.logger_singleton`
- `keyword_bias`
- `navigation_recipes`
- `navigation_recipes`

#### ⚠️ Task markers (navigator_navigation_runner)

- L202 **WARNING**: ({
- L203 **WARNING**: ",

### navigator/training\_data.py {#webapp-parser-navigator-training-data-py}

#### 🔧 Key Functions & Classes (navigator_training_data)

- `iter_navigation_feedback` (function, line 14)
- `build_training_dataset` (function, line 32)
- `export_training_dataset` (function, line 56)
- `main` (function, line 69)

#### 📦 Key Imports (navigator_training_data)

- `__future__`
- `argparse`
- `pathlib`
- `typing`
- `typing`
- `typing`
- `orjson`
- `config`

### quality\_assurance/\_\_init\_\_.py {#webapp-parser-quality-assurance-init-py}

> Quality Assurance Module: Data Classification & Verification Pipeline

#### 📦 Key Imports (quality_assurance___init__)

- `data_classifier`
- `data_classifier`
- `data_classifier`
- `data_classifier`
- `data_classifier`
- `data_classifier`
- `data_classifier`
- `data_classifier`
- `data_classifier`
- `data_classifier`
- `data_classifier`
- `qa_endpoints`

### quality\_assurance/data\_classifier.py {#webapp-parser-quality-assurance-data-classifier-py}

> Data Classifier: DL1/DL2 Quality Assurance Pipeline

#### 🔧 Key Functions & Classes (quality_assurance_data_classifier)

- `DLStatus` (class, line 30)
- `QAIssueType` (class, line 38)
- `IssureSeverity` (class, line 50)
- `ActionType` (class, line 58)
- `QAIssue` (class, line 72)
- `ClassificationResult` (class, line 86)
- `DatasetMetadata` (class, line 97)
- `get_db_connection` (function, line 115)
- `classify_as_dl1` (function, line 137)
- `detect_quality_issues` (function, line 253)
- `promote_to_dl2` (function, line 367)
- `get_pending_dl2_reviews` (function, line 457)
- `get_dl2_inventory` (function, line 490)
- `get_dataset_lineage` (function, line 537)

#### 📦 Key Imports (quality_assurance_data_classifier)

- `__future__`
- `json`
- `dataclasses`
- `dataclasses`
- `dataclasses`
- `datetime`
- `datetime`
- `enum`
- `typing`
- `typing`
- `typing`
- `typing`
- `uuid`
- `psycopg2`
- `psycopg2.extras`
- `config`
- `config`
- `config`
- `config`
- `config`

#### ⚠️ Task markers (quality_assurance_data_classifier)

- L53 **WARNING**: = "WARNING"
- L285 **WARNING**: .value,
- L328 **WARNING**: .value,
- L356 **WARNING**: .value,

### quality\_assurance/qa\_endpoints.py {#webapp-parser-quality-assurance-qa-endpoints-py}

> Data Assurance Endpoints: REST API for DL1/DL2 Classification & Review

#### 🔧 Key Functions & Classes (quality_assurance_qa_endpoints)

- `_require_qa_enabled` (function, line 39)
- `_get_reviewer_principal` (function, line 50)
- `_require_reviewer` (function, line 56)
- `parse_and_classify` (function, line 90)
- `get_pending_reviews` (function, line 185)
- `verify_and_promote` (function, line 227)
- `get_inventory` (function, line 291)
- `get_lineage` (function, line 345)
- `export_dl2_data` (function, line 394)
- `get_stats` (function, line 462)

#### 📦 Key Imports (quality_assurance_qa_endpoints)

- `__future__`
- `csv`
- `io`
- `json`
- `datetime`
- `datetime`
- `io`
- `functools`
- `flask`
- `flask`
- `flask`
- `flask`
- `config`
- `config`
- `utils.cert_utils`
- `utils.shared_logic`
- `utils.shared_logic`
- `data_classifier`
- `data_classifier`
- `data_classifier`

#### ⚠️ Task markers (quality_assurance_qa_endpoints)

- L485 **TODO**: Query for rejected count

### quarantine\_endpoints.py {#webapp-parser-quarantine-endpoints-py}

> Quarantine Review Endpoints: Transparent UI for URL quarantine review.

#### 🔧 Key Functions & Classes (quarantine_endpoints)

- `_require_quarantine_enabled` (function, line 28)
- `_get_reviewer_principal` (function, line 38)
- `_require_reviewer` (function, line 44)
- `get_pending_quarantines` (function, line 60)
- `get_quarantine_detail` (function, line 114)
- `submit_quarantine_review` (function, line 164)
- `get_quarantine_stats` (function, line 232)

#### 📦 Key Imports (quarantine_endpoints)

- `__future__`
- `functools`
- `flask`
- `flask`
- `flask`
- `config`
- `health.quarantine_queue`
- `health.quarantine_queue`
- `utils.cert_utils`
- `utils.shared_logic`
- `utils.shared_logic`

### services/context\_service.py {#webapp-parser-services-context-service-py}

#### 🔧 Key Functions & Classes (services_context_service)

- `ContextBasedPredictor` (class, line 35)
- `ContextService` (class, line 215)

#### 📦 Key Imports (services_context_service)

- `hashlib`
- `json`
- `os`
- `re`
- `datetime`
- `datetime`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.librarian`
- `services.election_data_services`

### services/election\_data\_services.py {#webapp-parser-services-election-data-services-py}

> ElectionDataService: Service layer for all election DB operations.

#### 🔧 Key Functions & Classes (services_election_data_services)

- `DictConvertible` (class, line 66)
- `get_decl_class_registry` (function, line 83)
- `iter_orm_classes` (function, line 92)
- `get_orm_class_by_tablename` (function, line 100)
- `get_table_columns` (function, line 109)
- `get_row_table` (function, line 119)
- `iter_row_columns` (function, line 125)
- `row_to_dict` (function, line 134)
- `_get_contest_id` (function, line 146)
- `columns_to_names` (function, line 164)
- `get_metadata_tables` (function, line 170)
- `ElectionDataService` (class, line 183)

#### 📦 Key Imports (services_election_data_services)

- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `sqlalchemy`
- `sqlalchemy.engine`
- `sqlalchemy.orm`
- `sqlalchemy.orm`
- `sqlalchemy.sql.schema`
- `sqlalchemy.sql.schema`
- `Context_Integration.librarian`
- `utils.db_utils`
- `utils.db_utils`
- `utils.db_utils`
- `utils.db_utils`
- `utils.db_utils`

### state\_router.py {#webapp-parser-state-router-py}

#### 🔧 Key Functions & Classes (state_router)

- `list_available_states` (function, line 45)
- `list_available_counties` (function, line 57)
- `import_handler` (function, line 76)
- `prompt_for_handler_fallback` (function, line 120)
- `preload_handler_map` (function, line 192)
- `reload_handler_map` (function, line 219)
- `scan_url_for_state_county` (function, line 226)
- `fuzzy_match_handler` (function, line 263)
- `list_available_handlers` (function, line 277)
- `get_handler` (function, line 322)
- `cli` (function, line 482)

#### 📦 Key Imports (state_router)

- `difflib`
- `importlib`
- `os`
- `time`
- `traceback`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `orjson`
- `config`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.Context_Library.constants`
- `utils.logger_singleton`
- `utils.logger_singleton`
- `utils.logger_singleton`
- `utils.shared_logic`
- `utils.shared_logic`
- `utils.shared_logic`

#### 💬 Top-of-file Comments (state_router)

```python

# state\_router.py

# ===============================================

# Dynamically routes to the correct state or county-specific handler module

# Uses importlib for auto-resolution from folder structure.

# Now uses librarian.py for state/county mapping.

# Also provides state/county info for format\_router and download\_utils.

# ===============================================

```

#### ⚠️ Task markers (state_router)

- L49 **WARNING**: ("\[Router\] handlers/states directory not found.")
- L66 **WARNING**: (f"\[Router\] counties directory not found for state:
{state_key}")
- L137 **WARNING**: (f"\[Fallback\]\[Session:{session_id}\] No handler states
available for manual selection.")
- L154 **WARNING**: (f"\[Fallback\]\[Session:{session_id}\] Aborted by user.")
- L157 **WARNING**: (f"\[Fallback\]\[Session:{session_id}\] Aborted by user.")
- L160 **WARNING**: (f"\[Fallback\]\[Session:{session_id}\] State '{state}'
not found. Please try again.")
- L179 **WARNING**: (f"\[Fallback\]\[Session:{session_id}\] Aborted by user.")
- L182 **WARNING**: (f"\[Fallback\]\[Session:{session_id}\] County '{county}'
not found for state '{state}'. Please try again.")
- L189 **WARNING**: (f"\[Fallback\]\[Session:{session_id}\] Too many failed
attempts. Exiting fallback.")
- L205 **WARNING**: (f"\[Router\] Requested state '{state_name}' not found on
disk. Skipping restrict filter.")
- L512 **WARNING**: (f"No counties found for state '{state}'. Try --fuzzy for
fuzzy matching.")
- L523 **WARNING**: (f"Failed to load context from file: {e}")
- L533 **WARNING**: ("No suitable handler found.")
- L540 **WARNING**: ("No handler selected. Exiting.")
- L547 **WARNING**: ("Still could not import a suitable handler.")

### tests/test\_extract\_url.py {#webapp-parser-tests-test-extract-url-py}

#### 🔧 Key Functions & Classes (tests_test_extract_url)

- `test_extract_url_and_label_cases` (function, line 23)
- `test_load_urls_integration` (function, line 28)

#### 📦 Key Imports (tests_test_extract_url)

- `tempfile`
- `pathlib`
- `importlib`
- `pytest`
- `webapp.parser.utils.misc_utils`

### tests/test\_fec\_handler.py {#webapp-parser-tests-test-fec-handler-py}

#### 🔧 Key Functions & Classes (tests_test_fec_handler)

- `test_party_normalize` (function, line 7)
- `test_money_and_date_normalize` (function, line 13)
- `test_handler_parse_fixture` (function, line 19)

#### 📦 Key Imports (tests_test_fec_handler)

- `os`
- `webapp.parser.handlers`
- `webapp.parser.utils`

### utils/audit\_trail\_router.py {#webapp-parser-utils-audit-trail-router-py}

> Audit Trail Router - Multi-Tier Compliance Logging

#### 🔧 Key Functions & Classes (utils_audit_trail_router)

- `_ensure_audit_logs` (function, line 46)
- `AuditEntry` (class, line 69)
- `ComplianceMetadata` (class, line 116)
- `log_decision_with_tier` (function, line 148)
- `add_event_chain_id` (function, line 211)
- `summarize_daily_compliance` (function, line 223)
- `write_compliance_summary` (function, line 306)
- `get_audit_entries_for_chain` (function, line 347)
- `get_principal_decisions` (function, line 377)

#### 📦 Key Imports (utils_audit_trail_router)

- `__future__`
- `json`
- `os`
- `threading`
- `uuid`
- `dataclasses`
- `datetime`
- `datetime`
- `pathlib`
- `typing`
- `typing`
- `typing`
- `config`
- `utils.logger_singleton`

### utils/browser\_utils.py {#webapp-parser-utils-browser-utils-py}

#### 🔧 Key Functions & Classes (utils_browser_utils)

- `Closable` (class, line 117)
- `get_random_user_agent` (function, line 122)
- `safe_url` (function, line 129)
- `safe_inner_text` (function, line 138)
- `safe_locator` (function, line 157)
- `safe_evaluate` (function, line 168)
- `safe_wait_for_timeout` (function, line 202)
- `safe_content` (function, line 214)
- `safe_nth` (function, line 237)
- `safe_is_visible` (function, line 244)
- `safe_is_enabled` (function, line 255)
- `safe_click` (function, line 266)
- `capture_page_diagnostics` (function, line 279)
- `safe_click_with_retry` (function, line 326)
- `safe_get_attribute` (function, line 432)
- `safe_attributes` (function, line 444)
- `safe_query_selector_all` (function, line 514)
- `safe_context_library` (function, line 525)
- `safe_count` (function, line 537)
- `safe_context_result` (function, line 572)
- `safe_launch` (function, line 598)
- `async_safe_launch` (async_function, line 618)
- `safe_new_context` (function, line 637)
- `async_safe_new_context` (async_function, line 648)
- `safe_new_page` (function, line 659)

#### 📦 Key Imports (utils_browser_utils)

- `__future__`
- `asyncio`
- `inspect`
- `json`
- `os`
- `random`
- `re`
- `time`
- `datetime`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `playwright.async_api`
- `playwright.async_api`

#### ⚠️ Task markers (utils_browser_utils)

- L105 **WARNING**: (f"\[browser_utils\] Failed to safely parse
context_library value for key '{key}'")
- L107 **WARNING**: (f"\[browser_utils\] Skipping unsafe context_library value
for key '{key}'")
- L279 **NOTE**: str = "click_failure") -&gt; dict:
- L291 **NOTE**: }\_\_{ts}.html")
- L299 **NOTE**: }\_\_{ts}.png")
- L365 **WARNING**: (f"\[safe_click_with_retry\] Re-query failed: {e} (attempt
{attempt})")
- L368 **WARNING**: (f"\[safe_click_with_retry\] No element found for
selector={selector} (attempt {attempt})")
- L408 **WARNING**: ({"level": "WARNING", "type": "browser", "message":
f"Click attempt failed (attempt {attempt}/{max_retries}): {e}", "session_id":
session_id})
- L414 **WARNING**: (f"\[safe_click_with_retry\] Element has no click()
(attempt {attempt})")
- L420 **WARNING**: ({"level": "WARNING", "type": "browser", "message":
f"Exception during click helper (attempt {attempt}): {e}", "session_id":
session_id})
- L426 **NOTE**: =(selector or 'element_click').replace('/', '_'))
- L465 **WARNING**: (f"\[safe_attributes\] Playwright JS extraction failed:
{e}")
- L479 **WARNING**: (f"\[safe_attributes\] Playwright fallback extraction
failed: {e}")
- L565 **WARNING**: (f"\[safe_count\] Object is not countable: {type(obj)}")
- L611 **WARNING**: (f"\[safe_launch\] browser_type is not a SyncBrowserType:
{type(browser_type)}")
- L631 **WARNING**: (f"\[async_safe_launch\] browser_type is not an
AsyncBrowserType: {type(browser_type)}")
- L710 **WARNING**: ({
- L711 **WARNING**: ",
- L739 **WARNING**: (f"\[CAPTCHA\] Detected Cloudflare CAPTCHA indicator:
'{indicator}'")
- L748 **WARNING**: (f"\[CAPTCHA\] CAPTCHA detected in async mode. Manual
intervention not implemented. (Session: {session_id})")

### utils/camelot\_utils.py {#webapp-parser-utils-camelot-utils-py}

#### 🔧 Key Functions & Classes (utils_camelot_utils)

- `_normalize_headers` (function, line 22)
- `_row_is_title_noise` (function, line 40)
- `_table_to_rows` (function, line 44)
- `_score_table` (function, line 67)
- `attempt_camelot_extraction` (function, line 83)
- `hybrid_fill_camelot` (function, line 118)

#### 📦 Key Imports (utils_camelot_utils)

- `__future__`
- `re`
- `typing`
- `typing`
- `typing`
- `typing`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.Context_Library.constants`
- `salvage`

### utils/captcha\_tools.py {#webapp-parser-utils-captcha-tools-py}

#### 🔧 Key Functions & Classes (utils_captcha_tools)

- `HasContent` (class, line 22)
- `HasPageSource` (class, line 28)
- `HasBringToFront` (class, line 35)
- `HasMaximizeWindow` (class, line 41)
- `detect_cloudflare_challenge` (function, line 57)
- `get_page_content` (function, line 70)
- `bring_to_front` (function, line 80)
- `is_cloudflare_captcha_present` (function, line 120)
- `wait_for_user_to_solve_captcha` (function, line 131)

#### 📦 Key Imports (utils_captcha_tools)

- `__future__`
- `ctypes`
- `os`
- `platform`
- `time`
- `typing`
- `typing`
- `typing`
- `orjson`
- `config`
- `config`
- `logger_singleton`
- `shared_logic`
- `shared_logic`

#### ⚠️ Task markers (utils_captcha_tools)

- L118 **WARNING**: (f"\[CAPTCHA\] Foreground window fallback failed: {e}")
- L154 **WARNING**: ("\[CAPTCHA\] CAPTCHA not resolved within timeout.")

### utils/cert\_utils.py {#webapp-parser-utils-cert-utils-py}

#### 🔧 Key Functions & Classes (utils_cert_utils)

- `_sha256_hex` (function, line 28)
- `_decode_base64` (function, line 34)
- `_extract_cert_metadata` (function, line 41)
- `extract_client_cert_fingerprint` (function, line 135)
- `extract_sso_principal` (function, line 163)
- `extract_client_principal` (function, line 189)

#### 📦 Key Imports (utils_cert_utils)

- `__future__`
- `base64`
- `hashlib`
- `json`
- `datetime`
- `datetime`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`

### utils/confidence\_scorer.py {#webapp-parser-utils-confidence-scorer-py}

> Confidence Scoring System with Harmonic Ranking & Immediate Flush

#### 🔧 Key Functions & Classes (utils_confidence_scorer)

- `ConfidenceLevel` (class, line 55)
- `RunConfidence` (class, line 68)
- `HarmonicScore` (class, line 148)
- `MaliciousActFlag` (class, line 181)
- `CriticalErrorSnapshot` (class, line 219)
- `_ensure_log_files` (function, line 255)
- `flush_run_confidence` (function, line 273)
- `flush_malicious_act` (function, line 309)
- `flush_critical_error_snapshot` (function, line 347)
- `flush_harmonic_score` (function, line 384)
- `store_traceback_in_memory` (function, line 414)
- `get_traceback_from_memory` (function, line 440)
- `clear_traceback_memory` (function, line 448)
- `compute_extraction_confidence` (function, line 458)
- `compute_factor_integrity_confidence` (function, line 478)
- `detect_malicious_act` (function, line 506)
- `create_run_confidence` (function, line 535)

#### 📦 Key Imports (utils_confidence_scorer)

- `__future__`
- `os`
- `threading`
- `time`
- `uuid`
- `dataclasses`
- `dataclasses`
- `datetime`
- `datetime`
- `pathlib`
- `typing`
- `typing`
- `typing`
- `typing`
- `config`
- `utils.logger_singleton`

#### ⚠️ Task markers (utils_confidence_scorer)

- L290 **WARNING**: ({
- L291 **WARNING**: ",
- L363 **WARNING**: ({
- L364 **WARNING**: ",

### utils/contest\_detection.py {#webapp-parser-utils-contest-detection-py}

#### 🔧 Key Functions & Classes (utils_contest_detection)

- `_build_contest_regex` (function, line 19)
- `_should_drop_contest_title` (function, line 81)
- `detect_contest_titles_from_text` (function, line 99)
- `gather_lines_for_contest_detection` (function, line 186)

#### 📦 Key Imports (utils_contest_detection)

- `__future__`
- `os`
- `re`
- `collections`
- `typing`
- `typing`
- `Context_Integration.Context_Library.constants`

### utils/contest\_normalization.py {#webapp-parser-utils-contest-normalization-py}

> Utilities for normalizing contest titles (referenda, propositions, etc.).

#### 🔧 Key Functions & Classes (utils_contest_normalization)

- `_split_referendum_title` (function, line 25)
- `_normalize_candidate_label` (function, line 57)
- `normalize_contest_label` (function, line 63)

#### 📦 Key Imports (utils_contest_normalization)

- `__future__`
- `re`
- `typing`
- `typing`

### utils/contest\_selector.py {#webapp-parser-utils-contest-selector-py}

#### 🔧 Key Functions & Classes (utils_contest_selector)

- `_env_truthy` (function, line 59)
- `ContestRecord` (class, line 74)
- `_bundle_key` (function, line 88)
- `_collect_bundle_members` (function, line 101)
- `_should_bundle` (function, line 181)
- `_inject_bundle_records` (function, line 217)
- `_merge_contest_metadata` (function, line 272)
- `_extract_first_int` (function, line 371)
- `_contest_sort_key` (function, line 383)
- `_extract_display_details` (function, line 410)
- `_extract_year_tokens` (function, line 448)
- `_strip_years` (function, line 451)
- `_base_canonical_key` (function, line 454)
- `_expand_contests_from_context` (function, line 464)
- `_merge_expanded_contests` (function, line 521)
- `_cluster_titles_by_base` (function, line 540)
- `_pick_rep_title` (function, line 557)
- `_score_title` (function, line 569)
- `_chunk_log_options` (function, line 580)
- `_render_paginated_contest_menu` (function, line 594)
- `_log` (function, line 631)
- `_norm_key` (function, line 656)
- `_tokens` (function, line 662)
- `_jaccard` (function, line 665)
- `_cluster_titles` (function, line 670)

#### 📦 Key Imports (utils_contest_selector)

- `__future__`
- `json`
- `math`
- `os`
- `re`
- `collections`
- `dataclasses`
- `dataclasses`
- `difflib`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `numpy`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.Context_Library.constants`

#### ⚠️ Task markers (utils_contest_selector)

- L645 **WARNING**: ":
- L646 **WARNING**: (entry)
- L1039 **WARNING**: ", "selector", f"Feedback loop {loop+1}: verifying
contests", session_id=session_id,
- L1709 **WARNING**: ({"level": "WARNING", "type": "selector", "message":
"Empty search term", "session_id": session_id})
- L1714 **WARNING**: ({"level": "WARNING", "type": "selector", "message": f"No
matches for '{term}'", "session_id": session_id})
- L1786 **WARNING**: ({"level": "WARNING", "type": "selector", "message": "No
match; try again.", "session_id": session_id})

### utils/coordinator\_protocol.py {#webapp-parser-utils-coordinator-protocol-py}

#### 🔧 Key Functions & Classes (utils_coordinator_protocol)

- `CoordinatorProtocol` (class, line 7)

#### 📦 Key Imports (utils_coordinator_protocol)

- `__future__`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`

### utils/date\_utils.py {#webapp-parser-utils-date-utils-py}

> date*utils.py

#### 🔧 Key Functions & Classes (utils_date_utils)

- `is_date_like` (function, line 13)

#### 📦 Key Imports (utils_date_utils)

- `__future__`
- `re`

### utils/db\_utils.py {#webapp-parser-utils-db-utils-py}

#### 🔧 Key Functions & Classes (utils_db_utils)

- `robust_orjson_loads` (function, line 42)
- `get_session` (function, line 53)
- `get_engine` (function, line 65)
- `update_contest_in_db` (function, line 72)
- `fetch_contests_by_filter` (function, line 97)
- `create_all_tables` (function, line 131)
- `create_batch_metadata` (function, line 135)
- `update_batch_metadata` (function, line 142)
- `get_batch_metadata` (function, line 151)
- `create_staging_election_result` (function, line 156)
- `get_staging_results_by_batch` (function, line 163)
- `create_warehouse_election_result` (function, line 168)
- `get_warehouse_results_by_batch` (function, line 175)
- `create_table_structure` (function, line 179)
- `update_table_structure` (function, line 192)
- `get_table_structure_by_id` (function, line 201)
- `fetch_table_structures` (function, line 205)
- `search_table_structures` (function, line 219)
- `update_table_structure_fields` (function, line 235)
- `select_table_structures_by_title` (function, line 250)
- `save_table_structure_to_db` (function, line 258)
- `get_table_structure_from_db` (function, line 286)
- `upsert_contest` (function, line 307)
- `get_or_create_state` (function, line 367)
- `get_or_create_county` (function, line 375)

#### 📦 Key Imports (utils_db_utils)

- `__future__`
- `os`
- `contextlib`
- `typing`
- `typing`
- `typing`
- `orjson`
- `sqlalchemy`
- `sqlalchemy`
- `sqlalchemy`
- `sqlalchemy`
- `sqlalchemy`
- `sqlalchemy`
- `sqlalchemy`
- `sqlalchemy.exc`
- `sqlalchemy.orm`
- `sqlalchemy.orm`
- `config`
- `Context_Integration.librarian`
- `logger_singleton`

### utils/detect.py {#webapp-parser-utils-detect-py}

> detect.py

#### 🔧 Key Functions & Classes (utils_detect)

- `emit_metric` (function, line 42)
- `EntityInfo` (class, line 50)
- `StructureInfo` (class, line 64)
- `_norm` (function, line 74)
- `normalize_text` (function, line 80)
- `normalize_for_matching` (function, line 83)
- `extract_percent_reported_from_heading` (function, line 89)
- `_is_percent_header` (function, line 100)
- `_should_exclude_as_location` (function, line 106)
- `_is_bad_location_fallback` (function, line 109)
- `is_location_header` (function, line 115)
- `dynamic_detect_location_header` (function, line 124)
- `detect_candidate_column` (function, line 178)
- `nlp_entity_annotate_table` (function, line 239)
- `harmonize_headers_and_data` (function, line 280)
- `find_best_header` (function, line 379)
- `is_likely_header` (function, line 393)
- `parse_numeric` (function, line 409)
- `extract_table_data` (function, line 424)
- `normalize_header` (function, line 466)
- `dedupe_headers_with_suffix` (function, line 491)
- `is_total_column` (function, line 504)

#### 📦 Key Imports (utils_detect)

- `__future__`
- `difflib`
- `re`
- `unicodedata`
- `dataclasses`
- `dataclasses`
- `functools`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.Context_Library.constants`
- `logger_singleton`

### utils/detector.py {#webapp-parser-utils-detector-py}

> detector.py

#### 🔧 Key Functions & Classes (utils_detector)

- `_norm` (function, line 28)
- `_numeric_like` (function, line 33)
- `EntityAnnotation` (class, line 40)
- `Detector` (class, line 46)

#### 📦 Key Imports (utils_detector)

- `__future__`
- `difflib`
- `re`
- `unicodedata`
- `dataclasses`
- `dataclasses`
- `functools`
- `typing`
- `typing`
- `typing`
- `typing`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.Context_Library.constants`
- `shared_logic`

### utils/dom\_extractor.py {#webapp-parser-utils-dom-extractor-py}

> dom*extractor.py

#### 🔧 Key Functions & Classes (utils_dom_extractor)

- `_row_score` (function, line 17)
- `_extract_row_cells` (function, line 23)
- `_pick_header` (function, line 36)
- `extract_rows_and_headers_from_dom` (function, line 72)
- `guess_headers_from_row` (function, line 156)

#### 📦 Key Imports (utils_dom_extractor)

- `__future__`
- `statistics`
- `typing`
- `typing`
- `typing`
- `typing`
- `browser_utils`
- `browser_utils`
- `browser_utils`
- `browser_utils`
- `detect`
- `detect`
- `logger_singleton`

#### ⚠️ Task markers (utils_dom_extractor)

- L153 **WARNING**: (f"\[DOM_EXTRACTOR\] failure: {e}")

### utils/download\_utils.py {#webapp-parser-utils-download-utils-py}

#### 🔧 Key Functions & Classes (utils_download_utils)

- `ensure_input_directory` (function, line 21)
- `ensure_output_directory` (function, line 25)
- `load_download_manifest` (function, line 29)
- `update_download_manifest` (function, line 45)
- `is_already_downloaded` (function, line 50)
- `download_file` (function, line 70)
- `download_multiple_files` (function, line 153)
- `download_confirmed_file` (function, line 169)
- `summarize_downloads` (function, line 179)
- `get_downloaded_files_by_status` (function, line 190)

#### 📦 Key Imports (utils_download_utils)

- `__future__`
- `os`
- `datetime`
- `urllib.parse`
- `urllib.parse`
- `orjson`
- `requests`
- `config`
- `config`
- `config`
- `config`
- `config`
- `Context_Integration.context_organizer`
- `utils.logger_singleton`
- `utils.misc_utils`
- `utils.shared_logic`
- `utils.shared_logic`

### utils/dynamic\_table\_extractor.py {#webapp-parser-utils-dynamic-table-extractor-py}

#### 🔧 Key Functions & Classes (utils_dynamic_table_extractor)

- `_emit` (function, line 85)
- `dynamic_table_extractor` (function, line 108)
- `find_tabular_candidates` (function, line 192)
- `analyze_candidate_nlp` (function, line 277)
- `score_candidate` (function, line 303)
- `remove_low_signal_columns` (function, line 391)
- `infer_column_types` (function, line 406)
- `advanced_party_candidate_detection` (function, line 472)
- `extract_candidates_and_parties` (function, line 491)
- `entity_linking` (function, line 542)
- `find_tables_with_headings` (function, line 589)
- `discover_container_selectors` (function, line 706)
- `log_new_dom_pattern` (function, line 753)
- `review_dom_patterns` (function, line 768)
- `auto_approve_dom_pattern` (function, line 814)
- `find_tables_with_panel_headings` (function, line 832)
- `find_tables_with_section_headings` (function, line 902)
- `is_candidate_major_row` (function, line 978)
- `is_candidate_major_col` (function, line 1022)
- `is_precinct_major` (function, line 1052)
- `is_flat_candidate_table` (function, line 1070)
- `is_single_row_summary` (function, line 1096)
- `is_candidate_footer` (function, line 1102)
- `detect_wide_vs_long` (function, line 1121)
- `classify_ambiguous_tables` (function, line 1132)

#### 📦 Key Imports (utils_dynamic_table_extractor)

- `__future__`
- `difflib`
- `os`
- `re`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `dateutil.parser`
- `numpy`
- `orjson`
- `selectolax.parser`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.Context_Library.constants`

#### ⚠️ Task markers (utils_dynamic_table_extractor)

- L124 **WARNING**: ", "extractor", "\[EXTRACTOR\] No &lt;table&gt; found in
provided table_html.", session_id)
- L129 **WARNING**: ", "extractor", "\[EXTRACTOR\] No &lt;tr&gt; rows found in
table_html.", session_id)
- L171 **WARNING**: ", "extractor", "\[EXTRACTOR\] Candidate NLP/score step
failed", session_id, error=str(e))
- L187 **WARNING**: ", "extractor", "\[EXTRACTOR\] No suitable table
candidates found.", session_id)
- L217 **WARNING**: ", "extractor", "\[EXTRACTOR\] Error while scanning
&lt;table&gt; elements", session_id, error=str(e))
- L229 **WARNING**: ", "extractor", "\[EXTRACTOR\] DOM extraction failed",
session_id, error=str(e))
- L272 **WARNING**: ", "extractor", "\[EXTRACTOR\] Pattern extraction failed",
session_id, error=str(e))
- L776 **WARNING**: ", "extractor", "No learned DOM patterns found.")
- L800 **WARNING**: ", "extractor", "Entry deleted.")
- L805 **WARNING**: ", "extractor", "Unknown action.")
- L807 **WARNING**: ", "extractor", "Invalid entry number.")

### utils/embedding\_cache.py {#webapp-parser-utils-embedding-cache-py}

#### 🔧 Key Functions & Classes (utils_embedding_cache)

- `_log_cache_status` (function, line 116)
- `ensure_embedding_cache_table` (function, line 134)
- `_db_write_allowed` (function, line 179)
- `compute_embedding_for_hash` (function, line 195)
- `save_embedding` (function, line 209)
- `load_embedding` (function, line 233)
- `get_embedding_from_memory` (function, line 261)
- `save_embeddings_batch` (function, line 280)
- `load_embeddings_batch` (function, line 342)
- `fix_missing_embeddings` (function, line 397)

#### 📦 Key Imports (utils_embedding_cache)

- `__future__`
- `atexit`
- `logging`
- `os`
- `threading`
- `functools`
- `numpy`
- `orjson`
- `sqlalchemy`
- `sqlalchemy`
- `sqlalchemy.dialects.postgresql`
- `sqlalchemy.exc`
- `sqlalchemy.orm.exc`
- `config`
- `config`
- `db_utils`
- `db_utils`
- `db_utils`
- `logger_singleton`
- `logger_singleton`

### utils/extraction\_strategies.py {#webapp-parser-utils-extraction-strategies-py}

> extraction*strategies.py

#### 🔧 Key Functions & Classes (utils_extraction_strategies)

- `register_strategy` (function, line 36)
- `run_registered_strategies` (function, line 45)
- `strategy_html_tables` (function, line 81)
- `strategy_dom_repetition` (function, line 94)
- `strategy_pattern_based` (function, line 100)
- `strategy_heading_associated` (function, line 104)
- `strategy_ml_detection` (function, line 157)
- `strategy_selectolax_fallback` (function, line 173)
- `strategy_nlp_fallback` (function, line 196)
- `_normalized_header_tuple` (function, line 236)
- `_merge_similar_tables` (function, line 239)

#### 📦 Key Imports (utils_extraction_strategies)

- `__future__`
- `re`
- `time`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `selectolax.parser`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.Context_Library.constants`
- `browser_utils`
- `browser_utils`
- `browser_utils`
- `browser_utils`
- `browser_utils`
- `detect`
- `detect`
- `detect`
- `detect`

#### ⚠️ Task markers (utils_extraction_strategies)

- L68 **WARNING**: (f"\[STRATEGY\] {name} failed: {e}")

### utils/factor\_chain\_tracker.py {#webapp-parser-utils-factor-chain-tracker-py}

> Deterministic Factor Chain Tracker with Breaking-Chain Detection

#### 🔧 Key Functions & Classes (utils_factor_chain_tracker)

- `_ensure_anomaly_log` (function, line 41)
- `FactorSnapshot` (class, line 57)
- `FactorChain` (class, line 71)
- `detect_breaking_chains` (function, line 175)
- `flush_factor_chain_analysis` (function, line 306)
- `create_factor_chain` (function, line 349)
- `finalize_factor_chain` (function, line 369)

#### 📦 Key Imports (utils_factor_chain_tracker)

- `__future__`
- `uuid`
- `dataclasses`
- `dataclasses`
- `datetime`
- `datetime`
- `pathlib`
- `typing`
- `typing`
- `typing`
- `config`
- `utils.logger_singleton`

### utils/fec\_utils.py {#webapp-parser-utils-fec-utils-py}

#### 🔧 Key Functions & Classes (utils_fec_utils)

- `_load_json` (function, line 16)
- `_append_ambiguous_log` (function, line 64)
- `canonicalize_headers` (function, line 74)
- `money_normalize` (function, line 104)
- `date_normalize` (function, line 126)
- `party_normalize` (function, line 152)
- `incumbent_normalize` (function, line 166)

#### 📦 Key Imports (utils_fec_utils)

- `__future__`
- `json`
- `os`
- `re`
- `datetime`
- `typing`
- `typing`
- `typing`
- `typing`
- `webapp.parser.config`

### utils/format\_router.py {#webapp-parser-utils-format-router-py}

#### 🔧 Key Functions & Classes (utils_format_router)

- `_normalize_text` (function, line 58)
- `_infer_format_from_text` (function, line 62)
- `_infer_format_from_attr_value` (function, line 73)
- `_extract_candidate_urls` (function, line 84)
- `_clean_filename` (function, line 111)
- `_guess_filename_from_url` (function, line 117)
- `_extract_filename_from_disposition` (function, line 136)
- `_extract_google_sheet_metadata` (function, line 146)
- `_probe_remote_format` (function, line 191)
- `_browser_headers` (function, line 242)
- `_build_download_url` (function, line 263)
- `_cookies_header_from_page` (function, line 270)
- `extract_contest_from_filename` (function, line 284)
- `summarize_downloads` (function, line 323)
- `_infer_format_from_url` (function, line 333)
- `_expose_download_interfaces` (function, line 341)
- `detect_format_from_links` (function, line 390)
- `route_format_handler` (function, line 441)
- `extract_download_links_from_html` (function, line 468)
- `prompt_and_handle_download` (function, line 488)

#### 📦 Key Imports (utils_format_router)

- `os`
- `re`
- `tempfile`
- `time`
- `difflib`
- `typing`
- `typing`
- `typing`
- `typing`
- `urllib.parse`
- `urllib.parse`
- `urllib.parse`
- `urllib.parse`
- `requests`
- `config`
- `config`
- `config`
- `config`
- `Context_Integration.Context_Library.constants`
- `handlers`

#### ⚠️ Task markers (utils_format_router)

- L426 **WARNING**: ({
- L427 **WARNING**: ",
- L429 **WARN**: \] No supported file formats found on the page.",
- L454 **WARNING**: ({
- L455 **WARNING**: ",
- L457 **WARN**: \] Unsupported format requested: {format_str}",
- L461 **WARNING**: ({
- L462 **WARNING**: ",
- L756 **WARNING**: ({
- L757 **WARNING**: ",
- L979 **WARNING**: ({
- L980 **WARNING**: ",
- L1085 **WARNING**: ({
- L1086 **WARNING**: ",

### utils/header\_confidence.py {#webapp-parser-utils-header-confidence-py}

> Header mapping confidence scoring and validation.

#### 🔧 Key Functions & Classes (utils_header_confidence)

- `get_header_confidence` (function, line 34)
- `validate_row_headers` (function, line 89)
- `should_insert_row` (function, line 126)

#### 📦 Key Imports (utils_header_confidence)

- `typing`
- `typing`
- `typing`
- `logging`

### utils/header\_utils.py {#webapp-parser-utils-header-utils-py}

#### 🔧 Key Functions & Classes (utils_header_utils)

- `build_candidate_group_hierarchical` (function, line 10)
- `normalize_headers_list` (function, line 37)
- `_clean_header_fragment` (function, line 46)
- `_assemble_header_label` (function, line 57)
- `compact_header_tokens` (function, line 84)
- `collapse_multiline_header` (function, line 147)
- `_register_header_mapping` (function, line 171)
- `normalize_table_headers` (function, line 178)

#### 📦 Key Imports (utils_header_utils)

- `__future__`
- `re`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `detect`
- `detect`
- `salvage`

### utils/html\_scanner.py {#webapp-parser-utils-html-scanner-py}

#### 🔧 Key Functions & Classes (utils_html_scanner)

- `robust_orjson_loads` (function, line 126)
- `_get_label_cache_path` (function, line 146)
- `_load_label_cache` (function, line 199)
- `_save_label_cache` (function, line 219)
- `cache_segment_label` (function, line 230)
- `get_cached_segment_label` (function, line 239)
- `safe_cache_path` (function, line 267)
- `safe_log_path` (function, line 328)
- `is_trivial_segment` (function, line 393)
- `segment_identity_hash` (function, line 470)
- `embedding_cache_hash` (function, line 496)
- `get_segment_embedding` (function, line 515)
- `batch_get_segment_embeddings` (function, line 617)
- `deduplicate_pattern_kb` (function, line 689)
- `prune_embedding_cache` (function, line 699)
- `submit_segment_correction` (function, line 711)
- `auto_label_segment` (function, line 720)
- `_extract_clean_text` (function, line 928)
- `_label_in` (function, line 943)
- `_extract_segments_by_label` (function, line 951)
- `extract_year_and_type` (function, line 1053)
- `is_update_panel` (function, line 1130)
- `split_possible_contests` (function, line 1147)
- `extract_tagged_segments_with_attrs` (function, line 1171)
- `get_page_hash` (function, line 1730)

#### 📦 Key Imports (utils_html_scanner)

- `__future__`
- `concurrent.futures`
- `datetime`
- `hashlib`
- `os`
- `re`
- `tempfile`
- `threading`
- `time`
- `traceback`
- `collections`
- `difflib`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `numpy`
- `orjson`

#### ⚠️ Task markers (utils_html_scanner)

- L163 **WARNING**: ",
- L167 **WARNING**: (payload)
- L189 **WARNING**: ",
- L193 **WARNING**: (payload)
- L288 **WARNING**: ",
- L292 **WARNING**: (payload)
- L315 **WARNING**: ",
- L319 **WARNING**: (payload)
- L353 **WARNING**: ",
- L357 **WARNING**: (payload)
- L380 **WARNING**: ",
- L384 **WARNING**: (payload)
- L579 **WARNING**: ",
- L583 **WARNING**: (payload)
- L784 **WARNING**: (f"\[ML SIMILARITY\] No embedding computed for segment:
{safe_get(segment, 'segment_hash', None)}")
- L807 **WARNING**: (f"\[ML SIMILARITY\] No embedding computed for segment:
{safe_get(segment, 'segment_hash', None)}")
- L1034 **WARNING**: ",
- L1038 **WARNING**: (payload)
- L1045 **WARNING**: ",
- L1049 **WARNING**: (payload)

### utils/json\_export\_loader.py {#webapp-parser-utils-json-export-loader-py}

#### 🔧 Key Functions & Classes (utils_json_export_loader)

- `_safe_int` (function, line 34)
- `_collapse_spaces` (function, line 50)
- `_strip_party_from_name` (function, line 54)
- `_normalize_candidate` (function, line 73)
- `NormalizedResultRow` (class, line 87)
- `ContestCoverage` (class, line 101)
- `NormalizedExport` (class, line 120)
- `_iter_county_contests` (function, line 132)
- `_normalize_group_labels` (function, line 139)
- `_derive_division_metadata` (function, line 143)
- `_build_context_snapshot` (function, line 179)
- `load_state_export` (function, line 223)
- `load_json_export` (function, line 405)

#### 📦 Key Imports (utils_json_export_loader)

- `__future__`
- `json`
- `re`
- `collections`
- `dataclasses`
- `dataclasses`
- `pathlib`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.librarian`
- `contest_normalization`

### utils/location\_helpers.py {#webapp-parser-utils-location-helpers-py}

#### 🔧 Key Functions & Classes (utils_location_helpers)

- `_normalize_location_text` (function, line 75)
- `_location_phrases` (function, line 84)
- `is_strict_location_header` (function, line 127)
- `collect_location_headers` (function, line 149)
- `format_location_fragment` (function, line 189)
- `attach_precinct_column` (function, line 238)

#### 📦 Key Imports (utils_location_helpers)

- `__future__`
- `re`
- `functools`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.Context_Library.constants`
- `detect`

### utils/logger\_singleton.py {#webapp-parser-utils-logger-singleton-py}

#### 🔧 Key Functions & Classes (utils_logger_singleton)

- `set_log_level` (function, line 20)
- `get_shared_logger` (function, line 23)

#### 📦 Key Imports (utils_logger_singleton)

- `__future__`
- `os`
- `shared_logger`
- `shared_logger`

### utils/merge\_utils.py {#webapp-parser-utils-merge-utils-py}

> merge*utils.py

#### 🔧 Key Functions & Classes (utils_merge_utils)

- `merge_table_data` (function, line 19)

#### 📦 Key Imports (utils_merge_utils)

- `__future__`
- `typing`
- `typing`
- `typing`
- `typing`
- `salvage`

### utils/metrics\_prom.py {#webapp-parser-utils-metrics-prom-py}

> Prometheus metrics integration (optional).

#### 🔧 Key Functions & Classes (utils_metrics_prom)

- `increment_test_counter` (function, line 53)
- `_push_registry_async` (function, line 68)
- `increment_prom_counter` (function, line 85)

#### 📦 Key Imports (utils_metrics_prom)

- `os`
- `threading`
- `typing`

### utils/misc\_utils.py {#webapp-parser-utils-misc-utils-py}

#### 🔧 Key Functions & Classes (utils_misc_utils)

- `load_processed_urls` (function, line 29)
- `safe_db_path` (function, line 48)
- `load_output_cache` (function, line 51)
- `file_hash` (function, line 60)
- `is_safe_path` (function, line 75)
- `extract_url_and_label` (function, line 92)

#### 📦 Key Imports (utils_misc_utils)

- `__future__`
- `hashlib`
- `os`
- `re`
- `pathlib`
- `typing`
- `typing`
- `typing`
- `orjson`
- `config`
- `config`
- `config`
- `config`
- `config`
- `config`
- `config`
- `logger_singleton`
- `shared_logic`
- `shared_logic`

#### ⚠️ Task markers (utils_misc_utils)

- L126 **WARNING**: ({
- L127 **WARNING**: ",

### utils/ml\_table\_detector.py {#webapp-parser-utils-ml-table-detector-py}

#### 🔧 Key Functions & Classes (utils_ml_table_detector)

- `_llm_detect_tables` (function, line 53)
- `detect_tables_ml` (function, line 119)
- `_ml_detect_tables` (function, line 192)
- `_vision_detect_tables` (function, line 211)
- `_extract_table_from_selectolax` (function, line 222)
- `_looks_like_table_selectolax` (function, line 265)
- `_extract_table_from_selectolax` (function, line 290)
- `_looks_like_table_selectolax` (function, line 331)
- `_extract_table_like_structure_selectolax` (function, line 361)
- `_regex_table_detection` (function, line 404)
- `_normalize_header` (function, line 443)

#### 📦 Key Imports (utils_ml_table_detector)

- `__future__`
- `re`
- `collections`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `orjson`
- `selectolax.parser`
- `config`
- `config`
- `config`
- `config`
- `config`
- `config`
- `browser_utils`
- `browser_utils`
- `logger_singleton`
- `model_registry`

### utils/model\_registry.py {#webapp-parser-utils-model-registry-py}

#### 🔧 Key Functions & Classes (utils_model_registry)

- `_hf_offline` (function, line 41)
- `load_vocab_from_file` (function, line 50)
- `build_reverse_vocab` (function, line 68)
- `advanced_tokenizer` (function, line 92)
- `ModelRegistry` (class, line 255)

#### 📦 Key Imports (utils_model_registry)

- `__future__`
- `os`
- `re`
- `subprocess`
- `sys`
- `threading`
- `collections`
- `typing`
- `typing`
- `typing`
- `selectolax.parser`
- `config`
- `config`
- `config`
- `config`
- `Context_Integration.librarian`
- `logger_singleton`

#### ⚠️ Task markers (utils_model_registry)

- L425 **WARNING**: (f"Failed loading local override for SentenceTransformer:
{e}")
- L445 **WARNING**: ("TRANSFORMERS_OFFLINE/HUGGINGFACE_HUB_OFFLINE set;
skipping HF download. Embeddings disabled.")
- L462 **WARNING**: for noisy environments
- L465 **WARNING**: (f"Failed to load base SentenceTransformer (network/DNS).
Running without embeddings. Error: {e}")

### utils/models.py {#webapp-parser-utils-models-py}

#### 🔧 Key Functions & Classes (utils_models)

- `MetaDataProtocol` (class, line 36)
- `DeclarativeBaseProtocol` (class, line 40)
- `ElectionTypeEnum` (class, line 45)
- `OfficeLevelEnum` (class, line 51)
- `StatusEnum` (class, line 57)
- `State` (class, line 64)
- `County` (class, line 76)
- `District` (class, line 89)
- `Office` (class, line 104)
- `Party` (class, line 115)
- `Candidate` (class, line 125)
- `Contest` (class, line 143)
- `Result` (class, line 171)
- `Panel` (class, line 189)
- `Button` (class, line 204)
- `CandidatePanel` (class, line 217)
- `LocationPanel` (class, line 234)
- `Heading` (class, line 251)
- `BallotType` (class, line 267)
- `ResultsTimestamp` (class, line 284)
- `PartyLabel` (class, line 299)
- `VoteMethod` (class, line 314)
- `Entity` (class, line 331)
- `MiscEntity` (class, line 341)
- `TableStructure` (class, line 353)

#### 📦 Key Imports (utils_models)

- `__future__`
- `enum`
- `uuid`
- `datetime`
- `datetime`
- `typing`
- `typing`
- `sqlalchemy`
- `sqlalchemy`
- `sqlalchemy`
- `sqlalchemy`
- `sqlalchemy`
- `sqlalchemy`
- `sqlalchemy`
- `sqlalchemy`
- `sqlalchemy`
- `sqlalchemy`
- `sqlalchemy`
- `sqlalchemy`
- `sqlalchemy`

### utils/output\_utils.py {#webapp-parser-utils-output-utils-py}

#### 🔧 Key Functions & Classes (utils_output_utils)

- `coerce_percent_strings` (function, line 42)
- `get_project_root` (function, line 50)
- `get_output_root` (function, line 54)
- `safe_join` (function, line 66)
- `get_output_path` (function, line 89)
- `format_timestamp` (function, line 189)
- `update_output_cache` (function, line 192)
- `check_existing_output` (function, line 213)
- `convert_sets_to_lists` (function, line 255)
- `deep_merge_dicts` (function, line 265)
- `_slug` (function, line 282)
- `build_filename_triplet` (function, line 292)
- `_ensure_dir` (function, line 306)
- `_coerce_headers` (function, line 312)
- `apply_results_conditional_formatting` (function, line 324)
- `export_dataframe_with_format` (function, line 361)
- `_compute_structure_hash` (function, line 370)
- `finalize_election_output` (function, line 384)

#### 📦 Key Imports (utils_output_utils)

- `__future__`
- `csv`
- `datetime`
- `hashlib`
- `os`
- `re`
- `collections`
- `datetime`
- `typing`
- `typing`
- `typing`
- `typing`
- `orjson`
- `pandas`
- `config`
- `config`
- `config`
- `config`
- `logger_singleton`
- `pivot`

#### ⚠️ Task markers (utils_output_utils)

- L136 **WARNING**: ("\[yellow\]\[OUTPUT\] Year could not be verified. Using
'Unknown'.\[/yellow\]")
- L139 **WARNING**: ("\[yellow\]\[OUTPUT\] contests could not be verified.
Using 'unknown_contests'.\[/yellow\]")
- L610 **WARNING**: (f"\[OUTPUT_UTILS\] Enrichment build failed: {e}")
- L688 **WARNING**: (f"\[OUTPUT_UTILS\] XLSX export failed: {e}")

### utils/pattern\_extractor.py {#webapp-parser-utils-pattern-extractor-py}

> pattern*extractor.py

#### 🔧 Key Functions & Classes (utils_pattern_extractor)

- `load_dom_patterns` (function, line 17)
- `extract_with_patterns` (function, line 29)

#### 📦 Key Imports (utils_pattern_extractor)

- `__future__`
- `json`
- `os`
- `typing`
- `typing`
- `typing`
- `typing`
- `detect`
- `logger_singleton`
- `shared_logic`

#### ⚠️ Task markers (utils_pattern_extractor)

- L26 **WARNING**: (f"\[PATTERN\] load fail {e}")
- L95 **WARNING**: (f"\[PATTERN\] pattern error {pat.get('name')}: {e}")

### utils/pdf\_table\_utils.py {#webapp-parser-utils-pdf-table-utils-py}

#### 🔧 Key Functions & Classes (utils_pdf_table_utils)

- `_recon_debug_enabled` (function, line 72)
- `_record_recon_event` (function, line 80)
- `consume_reconstruction_debug_events` (function, line 86)
- `detect_district_heading` (function, line 156)
- `build_contest_regex` (function, line 223)
- `normalize_text_token` (function, line 246)
- `token_set` (function, line 252)
- `header_signature` (function, line 256)
- `looks_like_candidate_header` (function, line 262)
- `compute_header_richness` (function, line 276)
- `is_numeric_like` (function, line 301)
- `normalize_numeric_token` (function, line 312)
- `compute_numeric_fill` (function, line 321)
- `evaluate_table_candidate_quality` (function, line 344)
- `find_best_header_match` (function, line 428)
- `normalize_anchor_value` (function, line 449)
- `merge_camelot_with_text` (function, line 455)
- `best_title_match_idx` (function, line 519)
- `extract_contest_block` (function, line 543)
- `parse_candidate_line` (function, line 663)
- `extract_candidate_totals_from_lines` (function, line 751)
- `_split_crammed_numeric_row` (function, line 789)
- `split_ws_blocks` (function, line 831)
- `is_bad_header_line` (function, line 849)
- `table_looks_bad` (function, line 887)

#### 📦 Key Imports (utils_pdf_table_utils)

- `__future__`
- `os`
- `re`
- `collections`
- `typing`
- `typing`
- `typing`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.Context_Library.constants`
- `header_utils`

### utils/pivot.py {#webapp-parser-utils-pivot-py}

> pivot.py

#### 🔧 Key Functions & Classes (utils_pivot)

- `_token_to_pattern` (function, line 149)
- `_build_division_token_patterns` (function, line 157)
- `_parse_numeric_token` (function, line 235)
- `_coerce_int` (function, line 269)
- `_normalized_header_cache` (function, line 285)
- `_natural_key` (function, line 288)
- `_sort_precincts` (function, line 299)
- `_infer_division_type_by_suffix` (function, line 318)
- `_extract_municipality` (function, line 327)
- `_numeric_ratio` (function, line 352)
- `_is_numeric_column` (function, line 363)
- `_fast_path_already_wide` (function, line 367)
- `debug_dump_pivot_state` (function, line 416)
- `_strip_party_fragment` (function, line 420)
- `_normalize_candidate_label` (function, line 469)
- `_collect_ballot_types` (function, line 494)
- `_derive_party_map` (function, line 528)
- `_normalize_division_name` (function, line 542)
- `_division_type_for` (function, line 550)
- `_s` (function, line 566)
- `_safe_col_name` (function, line 573)
- `_norm_text` (function, line 581)
- `_normalize_state_key` (function, line 585)
- `_detect_division_type_for_precinct` (function, line 597)
- `_detect_division_name_for_precinct` (function, line 656)

#### 📦 Key Imports (utils_pivot)

- `__future__`
- `hashlib`
- `math`
- `os`
- `re`
- `collections`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `orjson`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.Context_Library.constants`

#### ⚠️ Task markers (utils_pivot)

- L1353 **WARNING**: ("\[PIVOT\] No candidates detected – verify headers and
candidate column extraction.")

### utils/privilege\_tiers.py {#webapp-parser-utils-privilege-tiers-py}

> 4-Tier Privilege System for Election Results Parser

#### 🔧 Key Functions & Classes (utils_privilege_tiers)

- `PrivilegeTier` (class, line 23)
- `get_tier_trust_thresholds` (function, line 48)
- `get_principal_tier` (function, line 71)
- `should_apply_admin_boost` (function, line 187)
- `is_domain_in_allowlist` (function, line 228)
- `_parse_env_list` (function, line 242)
- `require_minimum_tier` (function, line 254)

#### 📦 Key Imports (utils_privilege_tiers)

- `__future__`
- `os`
- `enum`
- `typing`
- `typing`
- `utils.logger_singleton`

#### ⚠️ Task markers (utils_privilege_tiers)

- L167 **WARNING**: ({
- L168 **WARNING**: ",

### utils/rawjson\_utils.py {#webapp-parser-utils-rawjson-utils-py}

#### 🔧 Key Functions & Classes (utils_rawjson_utils)

- `_rj_first` (function, line 17)
- `_rj_as_dict` (function, line 29)
- `_rj_ensure_list` (function, line 44)
- `_infer_party_from_name` (function, line 49)
- `extract_rawjson_enrichment_from_rows` (function, line 58)
- `offload_rawjson_to_ndjson` (function, line 183)

#### 📦 Key Imports (utils_rawjson_utils)

- `__future__`
- `os`
- `typing`
- `orjson`

### utils/root\_admin\_session.py {#webapp-parser-utils-root-admin-session-py}

> Root Admin Session Management for Smart Elections Parser

#### 🔧 Key Functions & Classes (utils_root_admin_session)

- `generate_root_admin_token` (function, line 39)
- `hash_token` (function, line 62)
- `verify_root_admin_token` (function, line 74)
- `check_is_root_uid` (function, line 101)
- `create_root_admin_session` (function, line 119)
- `is_root_admin_session` (function, line 185)
- `get_root_admin_session_info` (function, line 214)
- `revoke_root_admin_session` (function, line 238)
- `cleanup_expired_root_admin_sessions` (function, line 268)
- `list_active_root_admin_sessions` (function, line 294)

#### 📦 Key Imports (utils_root_admin_session)

- `__future__`
- `hashlib`
- `os`
- `secrets`
- `time`
- `typing`
- `typing`
- `logger_singleton`

#### ⚠️ Task markers (utils_root_admin_session)

- L84 **NOTE**:     Note:
- L107 **NOTE**:     Note:
- L258 **WARNING**: ({
- L259 **WARNING**: ",
- L300 **WARNING**:     WARNING:

### utils/safe\_decide.py {#webapp-parser-utils-safe-decide-py}

> Safe Decision Helpers: Confidence/Caution Gates for Election Entities

#### 🔧 Key Functions & Classes (utils_safe_decide)

- `_emit_decision_log` (function, line 30)
- `safe_decide_jurisdiction` (function, line 79)
- `safe_decide_office` (function, line 127)
- `safe_decide_party` (function, line 162)
- `safe_decide_source` (function, line 196)
- `should_proceed` (function, line 230)
- `should_caution` (function, line 235)
- `should_stop` (function, line 240)

#### 📦 Key Imports (utils_safe_decide)

- `__future__`
- `time`
- `datetime`
- `datetime`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `webapp.parser.Context_Integration.library.entity_confidence_map`
- `webapp.parser.Context_Integration.library.entity_confidence_map`
- `webapp.parser.Context_Integration.library.entity_confidence_map`
- `webapp.parser.Context_Integration.library.entity_confidence_map`
- `webapp.parser.Context_Integration.library.entity_confidence_map`
- `webapp.parser.Context_Integration.library.entity_confidence_map`
- `webapp.parser.utils.logger_singleton`
- `webapp.parser.utils.shared_logic`

#### ⚠️ Task markers (utils_safe_decide)

- L69 **WARNING**: ({
- L70 **WARNING**: ",

### utils/salvage.py {#webapp-parser-utils-salvage-py}

> salvage.py

#### 🔧 Key Functions & Classes (utils_salvage)

- `_to_int_or_none` (function, line 35)
- `normalize_ballot_column_name` (function, line 39)
- `collapse_ballot_synonym_columns` (function, line 96)
- `merge_multiline_candidate_rows` (function, line 183)
- `combine_panel_tables_by_precinct` (function, line 216)
- `_salvage_rows_from_rawjson` (function, line 237)
- `remove_footer_and_summary_rows` (function, line 333)
- `remove_outlier_and_empty_rows` (function, line 354)

#### 📦 Key Imports (utils_salvage)

- `__future__`
- `re`
- `typing`
- `typing`
- `typing`
- `typing`
- `detect`

### utils/seleniumbase\_launcher.py {#webapp-parser-utils-seleniumbase-launcher-py}

#### 🔧 Key Functions & Classes (utils_seleniumbase_launcher)

- `_MissingDriver` (class, line 20)
- `launch_browser` (function, line 38)
- `relaunch_browser_fullscreen_if_needed` (function, line 55)
- `relaunch_browser_stealth` (function, line 95)
- `close_driver` (function, line 112)

#### 📦 Key Imports (utils_seleniumbase_launcher)

- `__future__`
- `time`
- `typing`
- `config`
- `logger_singleton`

### utils/session\_state.py {#webapp-parser-utils-session-state-py}

#### 🔧 Key Functions & Classes (utils_session_state)

- `SessionState` (class, line 7)
- `PipelinePhase` (class, line 21)
- `export_session_enums` (function, line 44)

#### 📦 Key Imports (utils_session_state)

- `__future__`
- `enum`
- `typing`

### utils/shared\_logger.py {#webapp-parser-utils-shared-logger-py}

#### 🔧 Key Functions & Classes (utils_shared_logger)

- `safe_getvalue` (function, line 39)
- `RichConsoleProxy` (class, line 50)
- `SQLAlchemyToSharedLoggerHandler` (class, line 150)
- `SharedLogger` (class, line 167)

#### 📦 Key Imports (utils_shared_logger)

- `__future__`
- `inspect`
- `logging`
- `os`
- `re`
- `threading`
- `time`
- `traceback`
- `contextlib`
- `io`
- `pathlib`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `orjson`

#### ⚠️ Task markers (utils_shared_logger)

- L160 **WARNING**:         elif record.levelno &gt;= logging.WARNING:
- L161 **WARNING**: (msg)
- L286 **WARNING**: ": logging.WARNING,
- L357 **WARNING**: ": "yellow",
- L419 **WARNING**: (self, msg, context=None, exc_info=None):
- L421 **WARNING**: ", msg, context, color="yellow")
- L435 **WARNING**: ": "yellow",
- L667 **WARNING**: (f"Log directory does not exist: {log_dir}")
- L684 **WARNING**: (f"Corrupt line in {path}: {e}")

### utils/shared\_logic.py {#webapp-parser-utils-shared-logic-py}

#### 🔧 Key Functions & Classes (utils_shared_logic)

- `DecisionTuple` (class, line 76)
- `ExtractPlugin` (class, line 107)
- `Saveable` (class, line 110)
- `GCModule` (class, line 113)
- `ShutilModule` (class, line 116)
- `TimeModule` (class, line 120)
- `HasItem` (class, line 124)
- `HasAllMethod` (class, line 129)
- `PredictionResult` (class, line 136)
- `EventLike` (class, line 158)
- `Predictable` (class, line 167)
- `safe_filename` (function, line 193)
- `is_path_safe` (function, line 279)
- `safe_resolve_path` (function, line 312)
- `safe_join_path` (function, line 343)
- `validate_directory_path` (function, line 371)
- `safe_slug` (function, line 387)
- `safe_query` (function, line 403)
- `safe_key` (function, line 414)
- `_filter_valid_kwargs` (function, line 425)
- `safe_filter_by` (function, line 443)
- `safe_first` (function, line 457)
- `get_or_create` (function, line 470)
- `safe_translate` (function, line 493)
- `safe_scheme` (function, line 505)

#### 📦 Key Imports (utils_shared_logic)

- `__future__`
- `copy`
- `difflib`
- `gc`
- `inspect`
- `ipaddress`
- `os`
- `platform`
- `re`
- `shutil`
- `socket`
- `textwrap`
- `time`
- `pathlib`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`

#### ⚠️ Task markers (utils_shared_logic)

- L411 **WARNING**: (f"\[safe_query\] session.query({model}) failed: {e}")
- L434 **WARNING**: (f"\[safe_filter_by\] No mapper found for model {model}")
- L440 **WARNING**: (f"\[safe_filter_by\] Could not inspect model {model}:
{e}")
- L454 **WARNING**: (f"\[safe_filter_by\] filter_by failed: {e}")
- L467 **WARNING**: (f"\[safe_first\] query.first() failed: {e}")
- L619 **WARNING**: ({
- L620 **WARNING**: ",
- L646 **WARNING**: (f"\[PLUGIN EXTRACTION\] Plugin {plugin} has no callable
'extract' method.")
- L780 **WARNING**: (f"\[WARN\] Model save failed (attempt {attempt}): {e}")
- L994 **WARNING**: (f"\[safe_append\] Target is not a list: {type(lst)};
coercing to list.")
- L1016 **WARNING**: (f"\[safe_update\] Target is not a dict: {type(dct)}")
- L1020 **WARNING**: (f"\[safe_update\] Updates is not a dict:
{type(updates)}")
- L1040 **WARNING**: (f"\[safe_extend\] Target is not a list: {type(lst)};
coercing to list.")
- L1380 **WARNING**: (f"\[DOM_PARTS\] '{label}' is not a list for URL: {url}
(type: {type(lst).\_\_name\_\_})")
- L1702 **WARNING**: (f"State '{state_norm}' not found in county map")
- L2568 **WARNING**: (f"\[inventory\] architecture.md not found at {md_file}")
- L2574 **WARNING**: ("\[inventory\] Markers not found in architecture.md;
aborting replace.")
- L2589 **WARNING**: ("\[inventory\] generate_project_map completed with
warnings; check markers and path.")
- L2635 **WARN**: ) and return their metadata."""
- L2637 **WARN**: ", "WARNING", "NOTE", "HA" + "CK", "X"_3, "BUG")

### utils/spacy\_utils.py {#webapp-parser-utils-spacy-utils-py}

#### 🔧 Key Functions & Classes (utils_spacy_utils)

- `_get_nlp` (function, line 27)
- `extract_entities` (function, line 45)
- `get_sentences` (function, line 94)
- `clean_text` (function, line 101)
- `extract_entities_from_list` (function, line 104)
- `extract_entity_labels` (function, line 107)
- `is_location_entity` (function, line 114)
- `extract_locations` (function, line 117)
- `extract_dates` (function, line 124)
- `filter_entities_by_type` (function, line 131)
- `entity_frequency` (function, line 138)
- `get_entity_context` (function, line 150)
- `similarity_score` (function, line 160)
- `extract_persons` (function, line 170)
- `extract_organizations` (function, line 177)
- `extract_money` (function, line 184)
- `extract_emails` (function, line 191)
- `extract_urls` (function, line 194)
- `load_known_states_counties` (function, line 200)
- `normalize_location` (function, line 211)
- `is_known_state` (function, line 219)
- `is_known_county` (function, line 222)
- `detect_noisy_or_ambiguous_entities` (function, line 225)
- `canonicalize_entity` (function, line 245)
- `validate_contest` (function, line 251)

#### 📦 Key Imports (utils_spacy_utils)

- `__future__`
- `os`
- `re`
- `sys`
- `collections`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `orjson`
- `Context_Integration.Context_Library.constants`
- `logger_singleton`
- `shared_logic`
- `shared_logic`

#### ⚠️ Task markers (utils_spacy_utils)

- L40 **WARNING**: (f"spaCy unavailable or model load failed: {e}")

### utils/strategy\_concurrency.py {#webapp-parser-utils-strategy-concurrency-py}

> strategy*concurrency.py

#### 🔧 Key Functions & Classes (utils_strategy_concurrency)

- `run_strategies_concurrently` (function, line 19)
- `_safe_run_strategy` (function, line 68)
- `run_strategies_concurrently_async` (async_function, line 76)

#### 📦 Key Imports (utils_strategy_concurrency)

- `__future__`
- `asyncio`
- `concurrent.futures`
- `concurrent.futures`
- `functools`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `browser_utils`
- `logger_singleton`

#### ⚠️ Task markers (utils_strategy_concurrency)

- L37 **WARNING**: (f"\[CONCURRENCY\] DOM strategy {name} failed: {e}")
- L65 **WARNING**: (f"\[CONCURRENCY\] Strategy {name} error: {e}")
- L73 **WARNING**: (f"\[CONCURRENCY\] {_safe_run_strategy.\_\_name\_\_} {name}
failed: {e}")
- L102 **WARNING**: (f"\[CONCURRENCY\]\[ASYNC\] DOM strategy {name} failed:
{e}")
- L120 **WARNING**: (f"\[CONCURRENCY\]\[ASYNC\] Strategy {name} error: {e}")

### utils/structure\_cache.py {#webapp-parser-utils-structure-cache-py}

> structure*cache.py

#### 🔧 Key Functions & Classes (utils_structure_cache)

- `table_signature` (function, line 14)
- `cache_table_structure` (function, line 19)
- `get_cached_structure` (function, line 25)

#### 📦 Key Imports (utils_structure_cache)

- `__future__`
- `hashlib`
- `typing`
- `typing`
- `typing`
- `detect`

### utils/table\_builder.py {#webapp-parser-utils-table-builder-py}

#### 🔧 Key Functions & Classes (utils_table_builder)

- `_normalize_header_cached` (function, line 71)
- `_norm_header` (function, line 76)
- `_percent_norms` (function, line 86)
- `_percent_reported_norm` (function, line 100)
- `_looks_like_location_header` (function, line 170)
- `_location_priority_score` (function, line 178)
- `_candidate_header_info` (function, line 189)
- `_extract_candidate_blocks` (function, line 208)
- `_coerce_int_for_total` (function, line 219)
- `_ensure_division_totals` (function, line 242)
- `_apply_canonical_order` (function, line 319)
- `_emit` (function, line 401)
- `_salvage_promote_best_row_as_header` (function, line 420)
- `_salvage_promote_first_row_as_header` (function, line 475)
- `_sanitize_headers_and_rows` (function, line 504)
- `_stringify_for_pivot` (function, line 595)
- `_stringify_entity_info` (function, line 618)
- `_drop_title_noise_rows` (function, line 643)
- `build_dynamic_table` (function, line 746)
- `build_table_noninteractive` (function, line 1037)
- `_get_table_builder_cache_dir` (function, line 1071)
- `_save_table_builder_cache` (function, line 1079)
- `_list_table_builder_cache` (function, line 1103)
- `_load_table_builder_cache` (function, line 1116)
- `prompt_user_to_confirm_table_structure` (function, line 1138)

#### 📦 Key Imports (utils_table_builder)

- `__future__`
- `copy`
- `os`
- `re`
- `time`
- `collections`
- `functools`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `orjson`
- `rich.table`
- `config`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.Context_Library.constants`
- `Context_Integration.Context_Library.constants`

#### ⚠️ Task markers (utils_table_builder)

- L816 **WARNING**: ", "builder", "\[TABLE_BUILDER\] dynamic_table_extractor
failed for panel table", session_id, error=str(e))
- L828 **WARNING**: ", "builder", "\[TABLE_BUILDER\] dynamic_table_extractor
failed (no panels path)", session_id, error=str(e))
- L836 **WARNING**: ", "builder", "\[TABLE_BUILDER\] all_panel_tables was not
a list; coercing to empty list", session_id,
got_type=str(type(all_panel_tables)))
- L845 **WARNING**: ", "builder", "\[TABLE_BUILDER\] Dropping invalid table
entry", session_id, entry_type=str(type(item)))
- L862 **WARNING**: ", "builder", "\[TABLE_BUILDER\] sanitize failed",
session_id, error=str(e))
- L867 **WARNING**: ", "builder", "\[TABLE_BUILDER\] harmonize failed",
session_id, error=str(e))
- L873 **WARNING**: ", "builder", "\[TABLE_BUILDER\]
collapse_ballot_synonym_columns failed", session_id, error=str(e))
- L925 **WARNING**: ",
- L950 **WARNING**: ", "builder", "\[TABLE_BUILDER\] entity annotate failed",
session_id, error=str(e))
- L955 **WARNING**: ", "builder", "\[TABLE_BUILDER\] stringify entity_info
failed", session_id, error=str(e))
- L975 **WARNING**: ", "builder", "\[TABLE_BUILDER\] pivot_to_wide failed",
session_id, error=str(e))
- L995 **WARNING**: ", "builder", "\[TABLE_BUILDER\] ensure division totals
failed", session_id, error=str(e))
- L1298 **WARNING**: ", "builder", f"\[TABLE_BUILDER\] Column marked
incorrect: {col_name}", session_id, contest=contest)
- L1371 **WARNING**: ", "builder", "\[TABLE_BUILDER\] Failed to persist table
structure logs", session_id, error=str(e))
- L1386 **WARNING**: ", "builder", "\[TABLE_BUILDER\] Failed to persist
coordinator DB log", session_id, error=str(e))

### utils/table\_core.py {#webapp-parser-utils-table-core-py}

> table*core.py (refactored orchestrator)

#### 🔧 Key Functions & Classes (utils_table_core)

- `_stringify_for_pivot` (function, line 83)
- `_deduplicate_tables` (function, line 100)
- `_log_extraction_summary` (function, line 114)
- `_annotate_entities_via_detector` (function, line 123)
- `robust_table_extraction` (function, line 142)
- `_sanitize_headers` (function, line 317)
- `build_table_from_page` (function, line 333)
- `robust_table_extraction_async` (async_function, line 352)
- `build_table_from_page_async` (async_function, line 464)
- `auto_table_build` (function, line 480)

#### 📦 Key Imports (utils_table_core)

- `__future__`
- `asyncio`
- `re`
- `time`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `detect`
- `detect`
- `detect`
- `detector`
- `extraction_strategies`
- `extraction_strategies`
- `extraction_strategies`
- `extraction_strategies`
- `extraction_strategies`
- `extraction_strategies`
- `extraction_strategies`

#### ⚠️ Task markers (utils_table_core)

- L231 **WARNING**: (f"\[TABLE BUILDER\] Concurrent strategies execution
failed: {e}")
- L288 **WARNING**: (f"\[TABLE BUILDER\] RawJSON pivot failed: {e}")
- L296 **WARNING**: (f"\[TABLE BUILDER\] pivot_to_wide signature mismatch
(skipped): {e}")
- L298 **WARNING**: (f"\[TABLE BUILDER\] pivot_to_wide failed (skipped): {e}")
- L349 **WARNING**: (f"\[TABLE BUILDER\] finalize output failed: {e}")
- L414 **WARNING**: (f"\[TABLE BUILDER\]\[ASYNC\] Concurrent strategies
execution failed: {e}")
- L477 **WARNING**: (f"\[TABLE BUILDER\]\[ASYNC\] finalize output failed:
{e}")

### utils/telemetry.py {#webapp-parser-utils-telemetry-py}

#### 🔧 Key Functions & Classes (utils_telemetry)

- `_derive_url_fields` (function, line 23)
- `emit_telemetry_event` (function, line 35)

#### 📦 Key Imports (utils_telemetry)

- `hashlib`
- `json`
- `os`
- `time`
- `typing`
- `typing`

### utils/telemetry\_agg.py {#webapp-parser-utils-telemetry-agg-py}

#### 🔧 Key Functions & Classes (utils_telemetry_agg)

- `_read` (function, line 14)
- `_write` (function, line 23)
- `get_counters` (function, line 38)
- `increment_counter` (function, line 41)
- `set_counter` (function, line 63)
- `reset_counters` (function, line 69)

#### 📦 Key Imports (utils_telemetry_agg)

- `json`
- `os`
- `time`
- `typing`
- `typing`

### utils/url\_trust\_scorer.py {#webapp-parser-utils-url-trust-scorer-py}

> URL Trust Scoring System for Smart Elections Parser

#### 🔧 Key Functions & Classes (utils_url_trust_scorer)

- `_load_verified_domains` (function, line 85)
- `_load_trust_history` (function, line 113)
- `_log_trust_decision` (function, line 184)
- `get_domain_trust_factors` (function, line 211)
- `detect_domain_mimicry` (function, line 310)
- `compute_trust_score` (function, line 376)
- `should_use_snapshot_mode` (function, line 586)
- `should_quarantine` (function, line 599)
- `should_reject` (function, line 630)

#### 📦 Key Imports (utils_url_trust_scorer)

- `__future__`
- `json`
- `re`
- `time`
- `pathlib`
- `typing`
- `typing`
- `typing`
- `typing`
- `urllib.parse`
- `config`
- `config`
- `config`
- `config`
- `logger_singleton`
- `privilege_tiers`
- `privilege_tiers`
- `privilege_tiers`
- `telemetry`

#### ⚠️ Task markers (utils_url_trust_scorer)

- L104 **WARNING**: ({
- L105 **WARNING**: ",
- L459 **WARNING**: ({
- L460 **WARNING**: ",
- L470 **WARNING**: ({
- L471 **WARNING**: ",
- L484 **WARNING**: ({
- L485 **WARNING**: ",
- L497 **WARNING**: ({
- L498 **WARNING**: ",
- L648 **NOTE**:     Security Note:
- L658 **WARNING**: ({
- L659 **WARNING**: ",

### utils/user\_prompt.py {#webapp-parser-utils-user-prompt-py}

#### 🔧 Key Functions & Classes (utils_user_prompt)

- `safe_lower` (function, line 33)
- `safe_strip` (function, line 39)
- `PromptCancelled` (class, line 50)
- `PromptSession` (class, line 54)
- `UserPrompt` (class, line 135)

#### 📦 Key Imports (utils_user_prompt)

- `__future__`
- `datetime`
- `inspect`
- `os`
- `re`
- `threading`
- `time`
- `traceback`
- `contextlib`
- `datetime`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `orjson`
- `rich.progress`

#### ⚠️ Task markers (utils_user_prompt)

- L318 **WARNING**: ("\[UserPrompt\] Webapp mode active but no
socketio_emit_func set!")
- L355 **WARNING**: ("\[CLI Prompt\] EOFError encountered.")
- L376 **WARNING**: ("\[Webapp Prompt\] socketio_emit_func not set.")
- L434 **WARNING**: ": 30,
- L515 **WARNING**: ("\n\[Prompt\] Timed out.")
- L566 **WARNING**: ("\n\[Prompt\] No input available (EOF). Exiting prompt.")
- L600 **WARNING**: ("Invalid input. Please try again.")
- L602 **WARNING**: ("\[Prompt\] Too many invalid attempts.")
- L667 **WARNING**: ("\[Prompt Queue\] Invalid queued yes/no response; falling
back to interactive prompt.")
- L682 **WARNING**: ("\n\[Prompt\] Timed out.")
- L889 **WARNING**: ("\[yellow\]\[FEEDBACK\] Skipped manual
correction.\[/yellow\]")
- L921 **WARNING**: ("\[yellow\]Button confirmation cancelled by
user.\[/yellow\]")

### utils/verification\_framework.py {#webapp-parser-utils-verification-framework-py}

> Dual-Truth Verification Framework for Smart Elections Parser

#### 🔧 Key Functions & Classes (utils_verification_framework)

- `VerificationStatus` (class, line 41)
- `VerificationConfidence` (class, line 49)
- `AnomalyType` (class, line 57)
- `VerificationLineageEntry` (class, line 69)
- `VerificationLog` (class, line 162)
- `classify_anomaly` (function, line 295)

#### 📦 Key Imports (utils_verification_framework)

- `__future__`
- `hashlib`
- `json`
- `datetime`
- `datetime`
- `enum`
- `pathlib`
- `typing`
- `typing`
- `typing`
- `typing`
- `typing`
- `logger_singleton`

### utils/xlsx\_exporter.py {#webapp-parser-utils-xlsx-exporter-py}

#### 🔧 Key Functions & Classes (utils_xlsx_exporter)

- `_auto_width` (function, line 13)
- `_apply_styles` (function, line 26)
- `export_candidate_group_pivot_xlsx` (function, line 50)

#### 📦 Key Imports (utils_xlsx_exporter)

- `__future__`
- `re`
- `typing`
- `typing`
- `typing`
- `typing`
- `openpyxl`
- `openpyxl.formatting.rule`
- `openpyxl.styles`
- `openpyxl.styles`
- `openpyxl.styles`
- `openpyxl.styles`
- `openpyxl.styles`
- `openpyxl.utils`

### verification/local\_dl\_sync.py {#webapp-parser-verification-local-dl-sync-py}

> Local DL1/DL2 File System Sync Implementation

#### 🔧 Key Functions & Classes (verification_local_dl_sync)

- `LocalStorageSync` (class, line 22)

#### 📦 Key Imports (verification_local_dl_sync)

- `hashlib`
- `os`
- `shutil`
- `threading`
- `datetime`
- `datetime`
- `pathlib`
- `typing`
- `typing`
- `typing`
- `orjson`

### verification\_endpoints.py {#webapp-parser-verification-endpoints-py}

> Verification Framework API Endpoints

#### 🔧 Key Functions & Classes (verification_endpoints)

- `_require_verification_enabled` (function, line 49)
- `_get_verifier_principal` (function, line 59)
- `_require_verifier_tier` (function, line 66)
- `_require_principal` (function, line 86)
- `get_system_mission` (function, line 97)
- `get_verification_stats` (function, line 115)
- `get_verification_entries` (function, line 152)
- `submit_verification` (function, line 220)
- `compare_dl1_dl2` (function, line 333)
- `export_dl1_verified` (function, line 419)
- `sync_status` (function, line 500)
- `sync_list_dl2` (function, line 545)
- `sync_list_dl1` (function, line 600)
- `sync_stage_dl2` (function, line 654)
- `sync_promote` (function, line 714)

#### 📦 Key Imports (verification_endpoints)

- `__future__`
- `os`
- `datetime`
- `datetime`
- `functools`
- `typing`
- `flask`
- `flask`
- `flask`
- `flask`
- `webapp.parser.config`
- `webapp.parser.config`
- `webapp.parser.config`
- `webapp.parser.config`
- `webapp.parser.utils.logger_singleton`
- `webapp.parser.utils.shared_logic`
- `webapp.parser.utils.shared_logic`
- `webapp.parser.utils.verification_framework`
- `webapp.parser.utils.verification_framework`
- `webapp.parser.utils.verification_framework`

#### ⚠️ Task markers (verification_endpoints)

- L79 **TODO**: Check principal's tier from privilege_tiers module
- L762 **WARNING**: ({
- L763 **WARNING**: ",
- L769 **WARNING**: ({
- L770 **WARNING**: ",

### web\_pipeline.py {#webapp-parser-web-pipeline-py}

#### 🔧 Key Functions & Classes (web_pipeline)

- `CancellationManager` (class, line 22)
- `heartbeat` (function, line 97)
- `save_pipeline_report` (function, line 111)
- `process_urls_for_web` (function, line 122)
- `cancel_processing` (function, line 713)

#### 📦 Key Imports (web_pipeline)

- `os`
- `threading`
- `time`
- `traceback`
- `orjson`
- `config`
- `config`
- `config`
- `config`
- `html_election_parser`
- `utils.logger_singleton`
- `utils.logger_singleton`
- `utils.shared_logic`
- `utils.shared_logic`
- `utils.shared_logic`

#### ⚠️ Task markers (web_pipeline)

- L53 **WARNING**: ({
- L54 **WARNING**: ",
- L70 **WARNING**: ({
- L71 **WARNING**: ",
- L87 **WARNING**: ({
- L88 **WARNING**: ",
- L172 **WARNING**: ({
- L173 **WARNING**: ",
- L321 **WARNING**: ({
- L322 **WARNING**: ",
- L332 **WARNING**: ({
- L333 **WARNING**: ",
- L386 **WARNING**: ({
- L387 **WARNING**: ",
- L397 **WARNING**: ({
- L398 **WARNING**: ",
- L601 **WARNING**: ({
- L602 **WARNING**: ",
- L619 **WARNING**: ({
- L620 **WARNING**: ",
