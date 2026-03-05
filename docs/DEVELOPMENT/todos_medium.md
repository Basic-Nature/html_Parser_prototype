---
layout: default
title: "TODO/FIXME Index — Medium"
---

<!-- markdownlint-disable-file MD001 MD004 MD011 MD022 MD024 MD025 MD033 MD034 MD037 MD050 MD052 -->

Index scope: TODO/FIXME/HACK/XXX annotations under `webapp, scripts, docs`.
Generated: 2026-03-05 05:27:58Z
Total annotations: 45
High: 0, Medium: 45, Low: 0

## Scan Profile

- Roots: webapp, scripts, docs
- Tracked markers: TODO, FIXME, HACK, XXX
- Priority map: high: FIXME; medium: HACK, TODO, XXX; low: none
- Exclusions (sample): docs/DEVELOPMENT/project_audit.md, docs/DEVELOPMENT/todos_medium.md, docs/DEVELOPMENT/todos_low.md, webapp/static/vendor/xlsx.full.min.js, docs/DEVELOPMENT/pipeline_map.md
- Regex: \b(TODO|FIXME|HACK|XXX)\b (case-insensitive)
- Actionable marker syntax: `TODO:`/`FIXME:` and common comment-prefix forms only

## Marker Breakdown

- HACK: 2 (medium)
- TODO: 41 (medium)
- XXX: 2 (medium)

## Root Coverage

- docs: 30
- scripts: 7
- webapp: 8

## Files

### docs/DEVELOPMENT/TODOS_OVERVIEW.md

- L22 **TODO**: | `TODO` | General improvements and future work | `# TODO: Refactor event loop` |
- L24 **HACK**: | `HACK` | Temporary workarounds that need cleanup | `# HACK: Suppress type error` |
- L25 **XXX**: | `XXX` | Dangerous code requiring attention | `# XXX: SQL injection risk here` |
- L46 **TODO**: # TODO: HIGH - Validate user input before parsing
- L56 **HACK**: # HACK: Low - Suppress mypy error, needs proper type annotation
- L59 **XXX**: # XXX: CRITICAL - SQL injection vulnerability, sanitize input!
- L105 **TODO**: # TODO: Improve error handling in retry logic
- L111 **TODO**: # TODO: CRITICAL - Validate certificate before auth check
- L124 **TODO**: 3. Commit the change with reference to the TODO:
- L127 **TODO**: git commit -m "fix: Address HIGH TODO - validate certificates (#42)"
- L172 **TODO**: - Reference issue numbers when applicable: `TODO: Fix #42 - ...`
- L179 **TODO**: - Commit with "TODO: test this" type placeholders
- L202 **TODO**: - Ensure marker format is correct: `# TODO: Description` or `# FIXME: Description`

### docs/FEATURES/ML_TRAINING_ENHANCEMENTS.md

- L404 **TODO**: # (TODO: Implement evaluation metrics in fine_tune_bert_ner.py)

### docs/FEATURES/NLP_ML_TRAINING_ASSESSMENT.md

- L596 **TODO**: - 🔄 TODO: Wire corrections into `ner_training_data` table
- L600 **TODO**: - 🔬 TODO: Implement `incremental_train_spacy_ner(new_examples)`
- L601 **TODO**: - 🔬 TODO: Avoid full retraining (just update weights on new data)
- L602 **TODO**: - 🔬 TODO: Add version tracking for models (metadata)
- L639 **TODO**: - 🔬 TODO: Implement `extract_anonymized_patterns(context_library)`
- L640 **TODO**: - 🔬 TODO: Hash state/county/contest combos + aggregate confidence scores
- L641 **TODO**: - 🔬 TODO: Export to `pattern_export.jsonl`
- L645 **TODO**: - 🔬 TODO: Implement `sync_patterns_to_pool(pattern_export.jsonl, remote_url)`
- L646 **TODO**: - 🔬 TODO: Download remote patterns via REST API
- L647 **TODO**: - 🔬 TODO: Merge into local `context_library.json` (keep highest confidence)
- L791 **TODO**: _**TODO: Add Test Dataset Evaluation**_

### docs/QUALITY/DATA_COMPARISON_ROADMAP.md

- L287 **TODO**: # TODO: replace with regression script once implemented

### docs/QUALITY/GOOGLE_SHEETS_MIGRATION.md

- L480 **TODO**: # TODO: replace with load script once implemented

### docs/STATE_HANDLER_INTEGRATION.md

- L342 **TODO**: ### TODO: High Priority
- L348 **TODO**: ### TODO: Medium Priority
- L354 **TODO**: ### TODO: Low Priority

### scripts/generate_county_handler.py

- L122 **TODO**: TODO: Customize this handler for {county_name} County's specific UI.
- L193 **TODO**: # TODO: Add button toggles, navigation sequences, etc. specific to {county_name} County
- L257 **TODO**: # TODO: Add URL patterns specific to this county
- L270 **TODO**: # TODO: Add navigation steps specific to this county

### scripts/generate_state_handler.py

- L166 **TODO**: TODO: Implement state-specific extraction logic.
- L196 **TODO**: # TODO: Add {state_name}-specific transformations here

### scripts/migrate_google_sheets.py

- L335 **TODO**: # TODO: Migrate DL1/DL2 data sheets (requires discovery first to identify them)

### webapp/Smart_Elections_Parser_Webapp.py

- L9116 **TODO**: entities = []  # TODO: Use spaCy NER to detect entities

### webapp/parser/handlers/states/new_york/county/westchester.py

- L61 **TODO**: TODO: Customize this handler for Westchester County's specific UI.
- L132 **TODO**: # TODO: Add button toggles, navigation sequences, etc. specific to Westchester County

### webapp/parser/handlers/vendor_state_map.py

- L38 **TODO**: "notes": "TODO: enhancedvoting.com domain; confirm vendor",
- L45 **TODO**: "notes": "TODO: enhancedvoting.com domain; confirm vendor",

### webapp/parser/health/fine_tune_bert_ner.py

- L80 **TODO**: # TODO: Improve token alignment with actual character offsets (start, end)
- L112 **TODO**: # TODO: Improve token alignment (start, end offsets)

### webapp/static/js/__tests__/ballot_lens_modern.placeholder.test.js

- L10 **TODO**: // TODO: evaluate the script in a jsdom context and exercise the helpers

## Likely Stubs/Placeholders

### docs/DEVELOPMENT/TODOS_OVERVIEW.md

- L24 **HACK**: | `HACK` | Temporary workarounds that need cleanup | `# HACK: Suppress type error` |
- L56 **HACK**: # HACK: Low - Suppress mypy error, needs proper type annotation

### docs/FEATURES/ML_TRAINING_ENHANCEMENTS.md

- L404 **TODO**: # (TODO: Implement evaluation metrics in fine_tune_bert_ner.py)

### docs/FEATURES/NLP_ML_TRAINING_ASSESSMENT.md

- L596 **TODO**: - 🔄 TODO: Wire corrections into `ner_training_data` table
- L600 **TODO**: - 🔬 TODO: Implement `incremental_train_spacy_ner(new_examples)`
- L602 **TODO**: - 🔬 TODO: Add version tracking for models (metadata)
- L639 **TODO**: - 🔬 TODO: Implement `extract_anonymized_patterns(context_library)`
- L645 **TODO**: - 🔬 TODO: Implement `sync_patterns_to_pool(pattern_export.jsonl, remote_url)`
- L791 **TODO**: _**TODO: Add Test Dataset Evaluation**_

### scripts/generate_county_handler.py

- L122 **TODO**: TODO: Customize this handler for {county_name} County's specific UI.
- L193 **TODO**: # TODO: Add button toggles, navigation sequences, etc. specific to {county_name} County
- L257 **TODO**: # TODO: Add URL patterns specific to this county
- L270 **TODO**: # TODO: Add navigation steps specific to this county

### scripts/generate_state_handler.py

- L166 **TODO**: TODO: Implement state-specific extraction logic.
- L196 **TODO**: # TODO: Add {state_name}-specific transformations here

### scripts/migrate_google_sheets.py

- L335 **TODO**: # TODO: Migrate DL1/DL2 data sheets (requires discovery first to identify them)

### webapp/parser/handlers/states/new_york/county/westchester.py

- L61 **TODO**: TODO: Customize this handler for Westchester County's specific UI.
- L132 **TODO**: # TODO: Add button toggles, navigation sequences, etc. specific to Westchester County
