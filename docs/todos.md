---
layout: default
title: "TODO/FIXME Index"
---

<!-- markdownlint-disable-file MD001 MD004 MD011 MD022 MD024 MD025 MD033 MD034 MD037 MD050 MD052 -->

Index scope: TODO/FIXME/HACK/XXX annotations under `webapp, scripts, docs`.
Generated: 2026-01-15 02:57:51Z
Total annotations: 3
High: 0, Medium: 3, Low: 0

## Scan Profile

- Roots: webapp, scripts, docs
- Tracked markers: TODO, FIXME, HACK, XXX
- Priority map: high: FIXME; medium: HACK, TODO, XXX; low: none
- Exclusions (sample): webapp/parser/Context_Integration/Context_Library/cache/context_cache.json, docs/todos_medium.md, docs/project_audit.md, docs/pipeline_map.md, docs/todos_high.md
- Regex: \b(TODO|FIXME|HACK|XXX)\b (case-insensitive)

## Marker Breakdown

- HACK: 2 (medium)
- TODO: 1 (medium)

## Root Coverage

- docs\roadmap.md: 1
- scripts\generate_todo_index.py: 2

## Files

### docs/roadmap.md

- L142 *TODO*: ## 🧭 Working TODO List

### scripts/generate_todo_index.py

- L259 *HACK*: f"Index scope: {_TASK}/{_FIXME}/HACK/XXX annotations under `{roots_text}`.",
- L307 *HACK*: lines.append(f"No {_TASK}/{_FIXME}/HACK/XXX annotations found under specified roots.")
