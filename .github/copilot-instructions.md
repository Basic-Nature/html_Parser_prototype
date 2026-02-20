# Smart Elections Parser – Copilot Guide

**Read first**
- Docs live in `docs/` (architecture, handlers, project_audit, index). Open the relevant doc before touching code; follow described contracts and hotspots.
- Repo entry points: CLI [webapp/parser/html_election_parser.py]`(../webapp/parser/html_election_parser.py)`; Flask UI [webapp/Smart_Elections_Parser_Webapp.py]`(../webapp/Smart_Elections_Parser_Webapp.py)`. `.env` must be populated.

**Note on links**
- Keep links repo-relative (no `file:///` absolute paths). Some editors show diagnostics in preview/virtual docs, but absolute paths are not portable and should never be committed to public repos.
- Fallback: when a preview/virtual editor flags a missing link, assume the project root (`html_Parser_prototype/`) and resolve the relative path locally (mentally replace the path in parentheses with the repo root). This silences the warning without changing committed links.

**Repository Structure & File Creation Rules**

Root has been cleaned to keep essentials only. Follow these strict placement rules:

**Root Directory (DO NOT clutter)**
```
html_Parser_prototype/
├── .github/              # CI/CD, workflows, copilot instructions
├── webapp/               # Main application code
├── docs/                 # Documentation (versioned + temp)
├── tests/                # Root-level tests (gitignored - use for experiments)
├── tools/                # Development tools, scripts, utilities
├── scripts/              # Automation scripts (install, maintenance)
├── constraints/          # Dependency constraints
├── automate.py           # Main test runner
├── pyproject.toml        # Python project config
├── package.json          # Node/npm config
└── [config files]        # .gitignore, .env.template, etc.
```

**File Creation Guidelines (CRITICAL)**

1. **Tests - Three Categories:**
   - `webapp/tests/` → **Production tests** (committed to git)
     - Unit tests: `test_*.py`
     - Integration tests: subfolders ok
     - Use for: Parser logic, handlers, utils, models
   
   - `tests/` (root) → **Experimental tests** (gitignored)
     - Quick validation scripts
     - Temporary test files
     - Agent-generated test explorations
     - Use when unsure of permanence
   
   - `tools/` → **Development test scripts** (committed selectively)
     - Smoke tests, headless checks
     - UI validation scripts
     - Use for: CI/CD tooling, debugging

2. **Webapp Structure (NEVER create at root)**
   - `webapp/parser/` → Parser engine, handlers, state routers
   - `webapp/parser/utils/` → Parser utilities (shared_logic, detect, etc.)
   - `webapp/static/` → CSS, JS, images
   - `webapp/templates/` → Jinja2 HTML templates
   - `webapp/tests/` → Webapp unit tests (committed)
   - `webapp/tools/` → Webapp-specific dev tools

3. **Documentation Placement:**
   - `docs/` → **Versioned documentation** (committed to GitHub Pages)
     - `docs/CORE/` → Core architecture
     - `docs/FEATURES/` → Feature documentation
     - `docs/DEPLOYMENT/` → Deployment guides
     - `docs/DEVELOPMENT/` → Development guides
   
   - `docs/temp/` → **Temporary/working docs** (gitignored, add to .gitignore)
     - Session notes, draft documents
     - Experimental design docs
     - Files for "just you and I" collaboration
     - **Always create temp docs here, NOT at root**

4. **Tools & Scripts:**
   - `tools/` → Development utilities (headless checks, smoke tests)
   - `scripts/` → Installation and maintenance scripts
   - **Never create `.py` scripts at root** unless it's a top-level entry point like `automate.py`

**Gitignore Patterns (know before creating)**

Automatically ignored (never committed):
- `tests/` (root) - experimental tests
- `tools/tmp/` - temporary tool outputs
- `tools/debug_headless_output/` - debug artifacts
- `tools/screenshots/` - UI test screenshots
- `input/`, `output/`, `uploads/` - runtime data directories
- `*.log`, `*.csv`, `*.pdf` - runtime files
- `__pycache__/`, `.pytest_cache/`, `.mypy_cache/` - Python caches
- `.env`, `google_service_account.json` - secrets (NEVER commit)
- `node_modules/` - npm packages

Selectively ignored (check .gitignore):
- `webapp/parser/Context_Integration/Context_Library/log/` - context logs
- `webapp/parser/Context_Integration/Context_Library/cache/` - context cache
- `selenium-screenshots/` - Selenium artifacts

**Location Decision Tree**

Creating a test?
  → Permanent parser/webapp test? → `webapp/tests/test_*.py`
  → Quick experiment/validation? → `tests/test_*.py` (gitignored)
  → CI/CD smoke test? → `tools/smoke_*.py`

Creating documentation?
  → Official feature/architecture doc? → `docs/FEATURES/`, `docs/CORE/`
  → Temporary working notes? → `docs/temp/` (add to .gitignore if not exists)
  → Session summary? → `docs/session-logs/`

Creating a utility?
  → Parser helper (CSV, NER, tables)? → `webapp/parser/utils/*.py`
  → Webapp helper (auth, validators)? → `webapp/tools/*.py`
  → Dev tool (smoke test, screenshot)? → `tools/*.py`
  → Build/install script? → `scripts/*.sh` or `scripts/*.py`

**Agent Execution Context**

When running through an agent or subagent:
- **Default cwd**: Repository root (`html_Parser_prototype/`)
- **Avoid creating files at root** - always use proper subdirectories
- **Check .gitignore** before creating test/temp files
- **Use relative paths** from root: `webapp/tests/`, `docs/temp/`, `tools/tmp/`
- **Never assume** files will be gitignored - verify first

**Common Mistakes to Avoid**
- ❌ Creating `test_*.py` at root (use `tests/` or `webapp/tests/`)
- ❌ Creating `*.md` docs at root (use `docs/` or `docs/temp/`)
- ❌ Creating utility scripts at root (use `tools/` or `scripts/`)
- ❌ Creating folders like `tmp/`, `temp/`, `scratch/` at root
- ❌ Leaving experimental files scattered (centralize in `tests/` or `tools/tmp/`)

**Before Creating Any File:**
1. Determine category: test, doc, tool, parser code, webapp code?
2. Check appropriate subdirectory from rules above
3. Verify gitignore status if temporary/experimental
4. Use proper naming conventions (`test_*.py`, `smoke_*.py`, etc.)
5. Never create at root unless it's a top-level config/entry point

**NEW: Database Comparison (Jan 2026)**
- URLs are now checked against Google Sheets + warehouse DB BEFORE parsing to avoid re-processing finalized data
- See [DATABASE_COMPARISON.md]`(../docs/FEATURES/DATABASE_COMPARISON.md)` for details
- Controlled via `skip_database_check` kwarg (default: False = checks enabled)
- URLs with existing data are marked `status="skipped_data_exists"` in `.processed_urls`

**NEW: Selenium-NLP Integration** (Feb 2026)
- Selenium is now a **strategic NLP training data collector**, not just CAPTCHA fallback
- Enabled by default (`ENABLE_SELENIUM_FALLBACK=true`) to capture entity-rich data from Cloudflare-protected government sites
- New logs: `selenium_ner_training.jsonl`, `captcha_resolution_log.jsonl`, `captcha_transition_log.jsonl`
- See [SELENIUM_NLP_INTEGRATION.md]`(../docs/FEATURES/SELENIUM_NLP_INTEGRATION.md)` for architecture and Phase 2 roadmap

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

**CI / Headless UI Stability Guidance**

- Visibility & timing:
	- Prefer robust visibility tests in headless checks: evaluate `getComputedStyle(el)` (display/visibility/opacity) and `el.getBoundingClientRect()` area instead of relying only on `classList` or `page.is_visible()`.
	- Poll briefly (e.g., 200ms intervals, up to ~1s) for state changes to allow CSS transitions/animations to complete; avoid brittle single-shot asserts.
	- Avoid forcing inline styles or classes in production code; use forced mutations only in CI diagnostics and mark them explicitly as such.

- Programmatic hooks & testability:
	- Expose small, documented helpers on `window` for tests: `openLeft()`, `openRight()`, `closeAll()`, `toggleNavDropdown()`, `setOverlayVisible(bool)`.
	- Handlers should be idempotent and return boolean success so tests can assert call outcomes without brittle UI clicks.

- Accessibility & event robustness:
	- Ensure interactive icons use `type="button"` and a descriptive `aria-label` or visible text to avoid webhint warnings and platform-specific click differences.
	- Avoid replacing native DOM methods (e.g., `document.addEventListener`) with brittle polyfills. If a wrapper is needed, preserve the original return values and behavior and always provide a cleanup/unsubscribe.

- CSP & third-party asset fallbacks:
	- When enabling relaxed CSP for dev/CI (e.g., allowing `https://cdn.jsdelivr.net`), gate it behind an env toggle and document the intent. Prefer `connect-src` for source-map fetching if necessary.

- Logging, artifacts & diagnostics:
	- On test failures, write a snapshot bundle: HTML (`.html`), full-page PNG, and a small JSON with computed-style diagnostics for key selectors (bounding rects, computed styles, class lists, body/html overflow). Store under `tools/debug_headless_output/`.
	- Capture console messages and page errors into the JSON bundle to correlate runtime exceptions with UI failures.

- CI integration & progressive hardening:
	- Integrate the headless check into `automate.py` as an optional `--self-check` step that runs after other validations and fails the run when requested.
	- Start with CI allowing a diagnostic fallback (accepting diagnostic-detected visibility) while stabilizing the UI; once stable, remove CI-only DOM mutations and tighten assertions.

- Tests & reviewers' checklist:
	- When creating or changing UI controls, add a short test checklist in the PR description: targeted viewport sizes, the programmatic hooks to verify, which selectors are critical, and expected overlay/scroll-lock behavior.
	- Prefer small, focused PRs for UI/behavior changes with attached headless run artifacts to speed review.

- Things to watch for (tips):
	- Headless differs from headed browsers on compositing and layout—zero-size bounding rects with non-none display often indicate offscreen transforms or z-index issues.
	- Synthetic `dispatchEvent('click')` vs `element.click()` can behave differently; provide an exposed helper instead of relying on synthetic events in tests.
	- Long-running animations, CSS transforms, or third-party widgets can delay reflow; use targeted polling instead of long global sleeps.

Add these guidelines to PR templates or CODEOWNERS notes when tying UI handler changes to tests to reduce regressions.
