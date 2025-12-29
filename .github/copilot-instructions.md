# Smart Elections Parser – Copilot Guide

**Read first:** [readme.md](../readme.md), [docs/architecture.md](../docs/architecture.md) for flow, [docs/handlers.md](../docs/handlers.md) for contracts, [docs/project_audit.md](../docs/project_audit.md) for hotspots, [docs/index.md](../docs/index.md) for doc entry points.

**Entrypoints & routing**
- CLI orchestrator [webapp/parser/html_election_parser.py](../webapp/parser/html_election_parser.py); Flask UI [webapp/Smart_Elections_Parser_Webapp.py](../webapp/Smart_Elections_Parser_Webapp.py). Both require a filled `.env` (DB + secrets).
- Routing: [webapp/parser/state_router.py](../webapp/parser/state_router.py) → state/county handlers; fallback format routing via [webapp/parser/utils/format_router.py](../webapp/parser/utils/format_router.py) using `html_scanner` and `prompt_user_for_format`.
- Navigation runs first: recipes in [webapp/parser/navigator/navigation_recipes.orjson](../webapp/parser/navigator/navigation_recipes.orjson) execute before routing; telemetry to `log/navigation_learning_log.jsonl` feeds health/retraining.

**Handler contract**
- Return `(headers, data_rows, contest, metadata)`; always gather input via `prompt_user_input()` from [webapp/parser/utils/user_prompt.py](../webapp/parser/utils/user_prompt.py).
- State/county handlers live under [webapp/parser/handlers/states](../webapp/parser/handlers/states); format fallbacks under [webapp/parser/handlers/formats](../webapp/parser/handlers/formats). Reuse shared helpers in [webapp/parser/handlers/shared](../webapp/parser/handlers/shared).
- Metadata drives output paths: results stored as `output/{state}/{county}/{race}/` with CSV + JSON metadata.

**Table + context pipeline**
- All formats flow through [webapp/parser/utils/table_builder.py](../webapp/parser/utils/table_builder.py) and [webapp/parser/utils/dynamic_table_extractor.py](../webapp/parser/utils/dynamic_table_extractor.py); supply `provided_tables` + `skip_pivot` via `html_context` when pre-extracted rows exist.
- Context/ML integrity orchestration in [webapp/parser/Context_Integration/context_coordinator.py](../webapp/parser/Context_Integration/context_coordinator.py) and [webapp/parser/Context_Integration/context_organizer.py](../webapp/parser/Context_Integration/context_organizer.py); knowledge base [webapp/parser/Context_Integration/Context_Library/context_library.json](../webapp/parser/Context_Integration/Context_Library/context_library.json); anomaly checks in [webapp/parser/Context_Integration/Integrity_check.py](../webapp/parser/Context_Integration/Integrity_check.py).
- Use safety helpers from [webapp/parser/utils/shared_logic.py](../webapp/parser/utils/shared_logic.py) for slugs/paths/audit logging; avoid ad-hoc `os.path`.

**Automation, tests, quality gates**
- `python automate.py` runs pipeline map generation, health bots, JS/TS lint + type checks, sample tests, and webapp import check. Flags: `--skip-web`, `--skip-health`, `--skip-tests`, `--skip-webapp-check`.
- Node scripts in [package.json](../package.json): `npm run check-js`, `lint:web`, `verify:python`, `verify:all`. Lint/type config in [pyproject.toml](../pyproject.toml); mypy targets format handlers/tests, ruff lenient elsewhere.
- Quick PDF smoke: `python run_statement_test.py`; fuller coverage under [webapp/tests](../webapp/tests).

**Health + navigation feedback**
- Health bots orchestrated via [webapp/parser/health/health_router.py](../webapp/parser/health/health_router.py); surfaced at `/azure_health` with streaming logs. Navigation feedback ingestion: [webapp/parser/health/navigation_feedback_ingest.py](../webapp/parser/health/navigation_feedback_ingest.py) converts navigation logs for correction/retraining.

**Front-end rules (UI is neon/metallic)**
- JS logic: [webapp/static/js/run_parser.js](../webapp/static/js/run_parser.js); styles: [webapp/static/css/run_parser.css](../webapp/static/css/run_parser.css). No inline styles; extend via classes/tokens under `@layer tokens`.
- Contest modal & bundles depend on `deriveOfficeTitle`/`deriveOfficeKey` + `expandedBundles/expandedOffices`; keep show/hide + bundleChildren in sync. Busy state: any backend prompt must call `PendingOverlay.show(...)`, `modalRestore.setBusyForSession`, and clear on `parser_output`/`session_state`.

**Ops and perf**
- `sitecustomize.py` installs Click parser shim; keep until upstream warnings resolved.
- PDFs: set `POPPLER_PATH` on Windows or install `poppler-utils` on Linux/Azure for faster pdf2image.

**Data safety & integrity**
- Prefer `.env` for secrets; path traversal protections already present—do not bypass. Ensure outputs include metadata and contest context; if extraction returns boilerplate, abort and prompt rather than emitting empty CSV.

**Roadmap anchors**
- Focus on schema consistency (party/division/jurisdiction columns), richer metadata (source/confidence), unified contest selection across formats, and multi-contest PDF regression fixtures. High/medium TODOs: [webapp/parser/utils/shared_logic.py](../webapp/parser/utils/shared_logic.py), [webapp/parser/health/manual_correction_bot.py](../webapp/parser/health/manual_correction_bot.py); see [docs/todos.md](../docs/todos.md).
