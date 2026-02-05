---
layout: default
title: "TODO/FIXME Index — Medium"
---

<!-- markdownlint-disable-file MD001 MD004 MD011 MD022 MD024 MD025 MD033 MD034 MD037 MD050 MD052 -->

Index scope: TODO/FIXME/HACK/XXX annotations under `webapp, scripts, docs`.
Generated: 2026-02-05 23:48:37Z
Total annotations: 39
High: 0, Medium: 39, Low: 0

## Scan Profile

- Roots: webapp, scripts, docs
- Tracked markers: TODO, FIXME, HACK, XXX
- Priority map: high: FIXME; medium: HACK, TODO, XXX; low: none
- Exclusions (sample): docs/DEVELOPMENT/todos_high.md, webapp/static/vendor/xlsx.full.min.js, docs/DEVELOPMENT/todos_low.md, docs/DEVELOPMENT/todos_medium.md, docs/DEVELOPMENT/todos.md
- Regex: \b(TODO|FIXME|HACK|XXX)\b (case-insensitive)

## Marker Breakdown

- HACK: 5 (medium)
- TODO: 30 (medium)
- XXX: 4 (medium)

## Root Coverage

- docs\DEVELOPMENT\TODOS_OVERVIEW.md: 26
- docs\README.md: 3
- docs\_data\navigation.yml: 1
- scripts\generate_todo_index.py: 2
- webapp\parser\fixtures\candidate_summary_index.json: 3
- webapp\parser\quality_assurance\qa_endpoints.py: 1
- webapp\parser\utils\shared_logic.py: 1
- webapp\parser\verification_endpoints.py: 1
- webapp\static\js\__tests__\ballot_lens_modern.placeholder.test.js: 1

## Files

### docs/DEVELOPMENT/TODOS_OVERVIEW.md

- L3 **TODO**: title: TODO System Overview
- L6 **TODO**: ## Development TODO System
- L8 **TODO**: ⚠️ **This page contains auto-generated documentation**. While this overview is manually maintained, the
  TODO lists below are automatically generated from your codebase. See [Auto-Generated Files](#auto-
  generated-files) for details.
- L18 **TODO**: The TODO system automatically scans Python and JavaScript files for the following markers:
- L22 **TODO**: | `TODO` | General improvements and future work | `# TODO: Refactor event loop` |
- L24 **HACK**: | `HACK` | Temporary workarounds that need cleanup | `# HACK: Suppress type error` |
- L25 **XXX**: | `XXX` | Dangerous code requiring attention | `# XXX: SQL injection risk here` |
- L46 **TODO**: # TODO: HIGH - Validate user input before parsing
- L56 **HACK**: # HACK: Low - Suppress mypy error, needs proper type annotation
- L59 **XXX**: # XXX: CRITICAL - SQL injection vulnerability, sanitize input!
- L65 **TODO**: The TODO system generates four markdown files automatically on each build:
- L81 **TODO**: 1. Extracts all TODO/FIXME/HACK/XXX markers with context
- L99 **TODO**: ### Adding a New TODO
- L105 **TODO**: # TODO: Improve error handling in retry logic
- L111 **TODO**: # TODO: CRITICAL - Validate certificate before auth check
- L120 **TODO**: ### Completing a TODO
- L123 **TODO**: 2. Remove the TODO marker from the source code
- L124 **TODO**: 3. Commit the change with reference to the TODO:
- L127 **TODO**: git commit -m "fix: Address HIGH TODO - validate certificates (#42)"
- L130 **TODO**: 4. The next generation automatically removes it from TODO lists
- L141 **TODO**: The TODO system is integrated into the automated build pipeline via:
- L156 **TODO**: The TODO files automatically include:
- L164 **TODO**: ## Guidelines for TODO Contributors
- L172 **TODO**: - Reference issue numbers when applicable: `TODO: Fix #42 - ...`
- L179 **TODO**: - Commit with "TODO: test this" type placeholders
- L202 **TODO**: - Ensure marker format is correct: `# TODO: Description` or `# FIXME: Description`

### docs/README.md

- L53 **TODO**: - [TODO Overview](./DEVELOPMENT/TODOS_OVERVIEW.md) - How-to guide for the TODO system
- L61 **TODO**: > **Note**: Files in `DEVELOPMENT/` are auto-generated from code analysis. See [TODO
  Overview](./DEVELOPMENT/TODOS_OVERVIEW.md) for the TODO system.
- L131 **TODO**: This documentation is actively maintained and includes consolidated content from 70+ source documents.
  The [DEVELOPMENT/](./DEVELOPMENT/) directory includes auto-generated TODO items that track outstanding
  work.

### docs/_data/navigation.yml

- L58 **TODO**: - title: TODO Overview

### scripts/generate_todo_index.py

- L261 **HACK**: f"Index scope: {_TASK}/{_FIXME}/HACK/XXX annotations under `{roots_text}`.",
- L309 **HACK**: lines.append(f"No {_TASK}/{_FIXME}/HACK/XXX annotations found under specified roots.")

### webapp/parser/fixtures/candidate_summary_index.json

- L480195 **XXX**: "CLYMER": "XXX, XXX",
- L480199 **XXX**: "Cand_Party_Affiliation": "XXX",
- L631839 **HACK**: "CLYMER": "HACK, HELMUTH",

### webapp/parser/quality_assurance/qa_endpoints.py

- L485 **TODO**: "rejected_count": 0,  # TODO: Query for rejected count

### webapp/parser/utils/shared_logic.py

- L3976 **TODO**: """Generate project audit + pipeline map, optionally a basic TODO index."""

### webapp/parser/verification_endpoints.py

- L79 **TODO**: # TODO: Check principal's tier from privilege_tiers module

### webapp/static/js/__tests__/ballot_lens_modern.placeholder.test.js

- L10 **TODO**: // TODO: evaluate the script in a jsdom context and exercise the helpers
