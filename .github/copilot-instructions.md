# Smart Elections Parser – Copilot Guide

**Read first**
- Docs live in `docs/` (architecture, handlers, project_audit, index). Open the relevant doc before touching code; follow described contracts and hotspots.
- Repo entry points: CLI [webapp/parser/html_election_parser.py](../webapp/parser/html_election_parser.py); Flask UI [webapp/Smart_Elections_Parser_Webapp.py](../webapp/Smart_Elections_Parser_Webapp.py). `.env` must be populated.

**NEW: Selenium-NLP Integration** (Feb 2026)
- Selenium is now a **strategic NLP training data collector**, not just CAPTCHA fallback
- Enabled by default (`ENABLE_SELENIUM_FALLBACK=true`) to capture entity-rich data from Cloudflare-protected government sites
- New logs: `selenium_ner_training.jsonl`, `captcha_resolution_log.jsonl`, `captcha_transition_log.jsonl`
- See [SELENIUM_NLP_INTEGRATION.md](../docs/FEATURES/SELENIUM_NLP_INTEGRATION.md) for architecture and Phase 2 roadmap

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
