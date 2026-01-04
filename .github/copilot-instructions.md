# Smart Elections Parser – Copilot Guide

**Read first**
- Docs live in `docs/` (architecture, handlers, project_audit, index). Open the relevant doc before touching code; follow described contracts and hotspots.
- Repo entry points: CLI [webapp/parser/html_election_parser.py](../webapp/parser/html_election_parser.py); Flask UI [webapp/Smart_Elections_Parser_Webapp.py](../webapp/Smart_Elections_Parser_Webapp.py). `.env` must be populated.

**Workflow (keep edits minimal)**
- Locate the handler/format/state router before adding logic; reuse helpers in `utils/shared_logic.py`, `Context_Integration`, and `handlers/shared` instead of ad-hoc code.
- Preserve existing logging style (`logger.mode` CLI vs non-CLI) and avoid noisy warnings; favor info/debug for non-critical paths.
- Respect path/slug safety helpers; never bypass traversal guards.

**Contracts & routing**
- Handlers return `(headers, data_rows, contest, metadata)` and collect input via `prompt_user_input()`.
- Routing: `state_router.py` → state/county handlers; fallback formats via `utils/format_router.py` using `html_scanner` + `prompt_user_for_format`.
- Navigation recipes (`navigator/navigation_recipes.orjson`) run before routing; telemetry to `log/navigation_learning_log.jsonl` powers health/retraining.

**Context/ML pipeline**
- Context integrity lives in `Context_Integration/*` (coordinator/organizer/library/integrity checks). Keep `table_builder.py`/`dynamic_table_extractor.py` contracts intact; pass `provided_tables`/`skip_pivot` via `html_context` when pre-extracted.
- Use canonical labels/validators; prefer shared utilities for normalization, hashing, and safe updates.

**Front-end (neon/metallic)**
- JS: `static/js/run_parser.js`; CSS: `static/css/run_parser.css`. No inline styles—extend via classes/tokens under `@layer tokens`.
- Keep contest modal logic (`deriveOfficeTitle`/`deriveOfficeKey`, bundle expansion) and busy state hooks (`PendingOverlay`, `modalRestore`) in sync with backend events.

**Automation & tests**
- Primary check: `python automate.py` (flags: `--skip-web`, `--skip-health`, `--skip-tests`, `--skip-webapp-check`).
- JS/TS: `npm run check-js`, `lint:web`; Python: `npm run verify:python`, `verify:all`; quick PDF smoke: `python run_statement_test.py`; deeper coverage: `webapp/tests`.
- TODO index: `python scripts/generate_todo_index.py --root webapp --root scripts --root docs [--ruff-json report.json] [--max-total N --max-high M]` writes `docs/todos.md`.

**Ops & perf**
- `sitecustomize.py` installs Click parser shim—keep until upstream resolves warnings.
- PDFs: set `POPPLER_PATH` on Windows or install `poppler-utils` on Linux/Azure for pdf2image.

**Data safety**
- Keep secrets in `.env`; outputs must include metadata/contest context. If extraction is boilerplate or empty, prompt/abort instead of emitting empty CSVs.

**Roadmap anchors**
- Prioritize schema consistency (party/division/jurisdiction), richer metadata (source/confidence), unified contest selection, and multi-contest PDF regression fixtures. High/medium TODOs: `utils/shared_logic.py`, `health/manual_correction_bot.py`, see `docs/todos.md`.

**Version control & assistant limits**
- Use `git status`, `git diff`, `git log -p`, and `git reflog` to audit changes and reconcile temp or prior edits.
- Assistant visibility is limited to the current workspace and this conversation; it cannot recall past sessions or unseen temp copies.

**Hard stop (edits)**
- Do NOT rewrite or truncate whole files. Always read the file, preserve existing content, and apply minimal, targeted diffs.
- For larger changes, create a brief plan first and ensure context is restored before finishing; avoid dropping or reordering unrelated code.
