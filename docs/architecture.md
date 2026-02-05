---
layout: default
---

# Smart Elections Parser — Architecture Overview

This document provides a high-level overview of the architecture and responsibilities across modules in the Smart Elections Parser repository, reflecting the latest modular, ML/NLP-integrated, and integrity-focused design.

---

## 🧱 Project Layers

### 1. **Entry Point**

- **`html_election_parser.py`**
  - Main orchestrator: delegates all specialized logic, never implements scraping/parsing directly.
  - Handles browser setup, CAPTCHA detection, user input (via `prompt_user_input`), and URL cycling.
  - Delegates parsing to state- or format-specific handlers.
  - Supports batch mode, multiprocessing, and bot/web integration.
  - Logs all actions for auditability.

### 2. **Router Layer**

- **`state_router.py`**
  - Matches URLs to a specific state handler in `handlers/`.
  - If no match is found, falls back to format detection.
  - Handles dynamic routing, including county-level and format-level delegation.

- **`format_router.py`** (in `utils/`)
  - Uses `html_scanner.py` and link metadata to detect HTML, PDF, JSON, or CSV.
  - Handles user prompting for format selection (via `prompt_user_for_format`).
  - Dispatches to a format handler.

### 3. **Handlers**

- **`handlers/states/`**
  - Contains one file per U.S. state (e.g., `arizona.py`, `new_york.py`).
  - Each state script must export a `parse(page, html_context)` method and return `(headers, data, contest, metadata)`.
  - County-level handlers live in `handlers/states/<state>/county/`.

- **`handlers/formats/`**
  - Generic format parsers: `pdf_handler.py`, `json_handler.py`, `csv_handler.py`, `html_handler.py`.
  - Used when no specialized state handler exists.
  - Must also return `(headers, data, contest, metadata)`.

- **`handlers/shared/`**
  - Shared logic, normalizers, and templates for use across handlers.

### 4. **Utilities**

- **`utils/table_core.py`**
  - Centralized table extraction, harmonization, and feedback logic.
  - Implements multi-strategy extraction (panel, section, ML/NER, plugin).
  - Dynamic scoring and patching: combines results from multiple extraction strategies, fills in missing info, and scores each method.
  - Keyword libraries for location, percent, and other election-specific columns.

- **`utils/table_builder.py`**
  - Normalizes, merges, annotates, and pivots tables for every handler (CSV, JSON, TXT, PDF, and state pipelines).
  - Applies cached header normalization, row-salvage heuristics, and canonical column ordering so that downstream tests and exports stay consistent.
  - Invoked via `build_table_noninteractive()` from format handlers and tests, ensuring the webapp and CLI share the same table reconstruction logic.

- **`utils/dynamic_table_extractor.py`**
  - Finds tables using both panel and section heading strategies.
  - Supports plugin-based and ML/NER-based extraction.
  - Returns candidate tables with associated context for further harmonization.

- **`utils/ml_table_detector.py`**
  - ML/LLM-powered table detection and structure learning.
  - Used for advanced extraction and anomaly detection.

- **`utils/spacy_utils.py`**
  - NLP-powered entity recognition and context enrichment.

- **`utils/browser_utils.py`**
  - Launches Playwright by default with optional Selenium fallback (when installed), supports headless/GUI, and user-agent spoofing.

- **`utils/captcha_tools.py`**
  - Detects and handles CAPTCHA pages.
  - Supports browser un-hiding and user intervention.

- **`utils/download_utils.py`**
  - Handles file downloads and directory creation for parsed content.

- **`utils/html_scanner.py`**
  - Performs early scan of HTML content to detect election year, races, counties.
  - Critical for routing and user prompt generation.

- **`utils/contest_selector.py`**
  - Handles contest/race filtering and selection, supports custom noisy label patterns per handler.

- **`utils/user_prompt.py`**
  - All user input is routed through `prompt_user_input()` for CLI/web UI modularity.

- **`utils/output_utils.py`**
  - Handles output formatting, metadata, and audit trail generation.

- **`utils/shared_logger.py`**
  - Centralized logging for all modules, supports both CLI and Web UI.

- **`utils/shared_logic.py`**
  - Safety wrappers for filesystem, SQLAlchemy, and ML helpers used across routers, handlers, and the web pipeline.
  - Provides the “audit backbone” that keeps cross-module interactions defensive (e.g., `safe_get`, `safe_slug`, coordinator feedback helpers).

---

## � Table Builder Flow

- **Single orchestration surface**: All generic format handlers (`csv_handler.py`, `json_handler.py`, `txt_handler.py`, `xlsx_handler.py`, and PDF routines) call `build_table_noninteractive()` from `utils/table_builder.py`. This guarantees that table salvage, NLP tagging, pivoting, and canonical ordering behave the same way in the CLI and webapp.
- **Context hand-off**: Format handlers assemble a context payload (contest, state/county inference, session identifiers) before calling the builder, so shared utilities such as `safe_get`, `record_noise_suggestion`, and the coordinator feedback loop can reason about the run.
- **Shared heuristics**: `table_builder` relies on `shared_logic.py` for auditing helpers (`safe_append`, `safe_strip`, etc.) and the constants library for ballot-type normalization. Optimizations such as cached header normalization ensure the wide range of format inputs stay performant.
- **Testing parity**: Dedicated tests in `webapp/tests/` exercise the builder directly, mirroring the format handlers. This acts as a safety net for architecture changes and keeps `architecture.md` aligned with the actual pipeline.

---

## �🤖 ML, Context, and Web UI Integration

- **`health/`**
  - Correction, retraining, and automation health (see `health_router.py`).
  - Includes manual correction and retraining pipeline.

- **`Context_Integration/`**
  - Context, ML/NLP, and integrity modules:
    - `context_coordinator.py`: Orchestrates context analysis, NLP, and ML integrity checks.
    - `context_organizer.py`: Context enrichment, clustering, and persistent context library management.
    - `Integrity_check.py`: Election integrity and anomaly detection logic.
    - `librarian.py`: Manages context library loading/saving, and centralized filename parsing for location detection (state, county, contest, year) across all format handlers.

- **`context_library.json`**
  - Persistent context and feedback for smarter extraction and correction.
  - Learns from user feedback and corrections for future runs.

- **Web UI (`webapp/Smart_Elections_Parser_Webapp.py`)**
  - Flask-based web interface for managing URLs, running the parser, and reviewing output.
  - Real-time log streaming, data management, and user-friendly contest/table review.
  - All user prompts are modularized (`prompt_user_input`), allowing easy swap for a web interface.

---

## 🧬 Data Architecture: Context, Logs, and Database Flow

This project uses a modular, auditable pipeline for election data parsing, context management, and ML/NLP retraining. Below is a detailed breakdown of how context, logs, and databases interact, and recommendations for optimizing structure and paths.

---

### 1. **Core Data Flows and Roles**

#### **A. Context Library (`context_library.json`)

- **Purpose:**
  - The original, central source of contextual knowledge (states, counties, contests, patterns, etc.).
  - Used for lookups, normalization, and as a knowledge base for parsing and ML.
- **Location:**
  - `webapp/parser/Context_Integration/Context_Library/context_library.json`
- **Accessed by:**
  - `context_coordinator.py`, `context_organizer.py`, `librarian.py`, and ML health.

#### **B. Context Library DB (`context_library_db.json`)

- **Purpose:**
  - A more structured or expanded version of the context library, possibly for ML or audit.
  - Generated/updated by `manual_correction.py` and possibly others.
- **Location:**
  - Same directory as above.
- **Accessed by:**
  - Correction, possibly ML retraining scripts.

#### **C. Context DB (`context_elections.db`)

- **Purpose:**
  - Legacy SQLite DB, now mostly replaced by PostgreSQL.
  - May still be referenced for backward compatibility or migration.
- **Location:**
  - Same directory as above.
- **Accessed by:**
  - Should be phased out if you’re fully on PostgreSQL.

#### **D. PostgreSQL (`POSTGRES_URL`)

- **Purpose:**
  - The main, production-grade relational database for all structured data (contests, table structures, entities, etc.).
  - Used for robust querying, updates, and ML training data storage.
- **Accessed by:**
  - All SQLAlchemy-based models and session logic.

#### **E. Logs (`log/` directory)

- **Purpose:**
  - Store all extraction, correction, feedback, and anomaly logs as `.jsonl` files.
  - Serve as the audit trail and as a source for manual/ML correction and retraining.
- **Accessed by:**
  - `manual_correction.py`, `librarian.py`, retraining scripts, and context health.

#### **F. Fixtures Pipeline (`fixtures/`, `cache/`, `log/` → PostgreSQL)**

- **Purpose:**
  - Manage election data from URL downloads through local processing to warehouse storage.
  - Enable confidence-based filtering to ensure only validated data reaches PostgreSQL.
- **Data Flow:**

  ```txt
  Handler Downloads (URLs) → CSVs (local, gitignored)
                           → JSON Fixtures (fixtures/, committed)
                           → Handler Extraction
                           → Cache (short-term) + Log (append-only)
                           → Confidence Filtering (≥0.7)
                           → Index Builder (build_election_index.py)
                           → PostgreSQL Warehouse
  ```

- **Directories:**
  - **`webapp/parser/fixtures/`** (committed):
    - Handler-ready JSON/JSONL fixtures from CSV conversion
    - `election_results_index.json`, shards, schema
    - Triggers CI validation workflow on changes
  - **`Context_Library/cache/`** (gitignored):
    - Short-term: `context_cache.json`, `embedding_disk_cache.pkl`, `table_builder_cache/`
    - Cleared periodically; migrates to PostgreSQL when confidence ≥0.7
  - **`Context_Library/log/`** (gitignored):
    - Append-only JSONL: `field_selection_log.jsonl`, `navigation_learning_log.jsonl`, `integrity_monitor.jsonl`, `trust_history.jsonl`, session logs
    - Grows over time; periodic archival recommended; selective migration based on health/defense criteria
- **Source Priority (build_election_index.py):**
  1. CSVs (local-only, if present)
  2. Fixtures JSON/JSONL (committed)
  3. Cache JSON (`--include-cache` flag)
  4. Log JSONL (`--include-log` flag)
- **Confidence Filtering:**
  - Default threshold: 0.0 (no filtering)
  - Production recommended: ≥0.7 (`--min-confidence 0.7`)
  - Records below threshold logged to audit report but excluded from index
- **Migration Criteria (to PostgreSQL):**
  - Confidence threshold check (≥0.7)
  - Integrity monitor flags vs established trust patterns (via `Integrity_check.py`)
  - Deduplication and conflict resolution
  - Anomaly detection via ML models
  - Schema validation (`election_results_schema.json`)
- **Best Practices:**
  - Handlers: Download CSVs locally (never commit); convert to JSON; commit only validated JSON/JSONL with `source` URL and `confidence` score
  - Index Building: Use `--min-confidence 0.7 --include-cache --include-log` for production; review `fixture_audit_report.jsonl` for warnings
  - Maintenance: Archive old logs periodically; clear cache after migration; monitor PostgreSQL for duplicates

---

### 2. **How Data Flows Through the Pipeline**

#### **Step 1: Extraction & Parsing**

- **HTML and other sources are parsed** using Playwright and handlers.
- **Contextual clues** (state, county, contest, etc.) are extracted using the context library and ML/NLP.
- **Results and context** are logged to `.jsonl` files in `log/`.

#### **Step 2: Context Coordination & Organization**

- **`context_coordinator.py` and `context_organizer.py`:**
  - Use the context library for normalization and enrichment.
  - Organize parsed data, deduplicate, and run integrity checks.
  - May update the context library with new findings.

  **Enrichment Coordinator Handle**
  - Treat `context_coordinator.py` as the enrichment coordinator that stages work before `ContextOrganizer.organize_context` fires. Instead of flooding the organizer with every detected asset, the coordinator should batch entities by category (contests, locations, vote methods, etc.), attach provenance tags, and only submit the bundles that pass lightweight heuristics.
  - Each invocation of `organize_context` should emit a compact training snapshot: the categorized entities, the applied fixes, and the anomaly verdicts. Store these snapshots under `log/context_enrichment/*.jsonl` or directly in PostgreSQL so retraining jobs can ingest well-scoped examples rather than noisy global dumps.
  - When wiring new handlers (HTML or PDF), call into the coordinator’s enrichment handle first; let it decide whether to spawn DOM scans, NLP passes, or ML anomaly checks. This throttles unrelated tasks and keeps training data aligned with best practices (one category per pass, provenance recorded, audit-ready metadata attached).
  - The coordinator now builds an `enrichment_plan` (routes such as `dom`, `tables`, `ml`, `integrity`) and passes it to `ContextOrganizer`. The organizer honors that plan by skipping gated routes and records the resolved plan/decisions into `metadata.route_summary`. Every plan execution is appended to `log/context_enrichment/plan_snapshots.jsonl` for downstream training and auditing.
  - Format-aware routing: HTML/DOM runs keep panel/button scans, PDFs prioritize reconstruction, OCR/image inputs request DOM + ML cleanup, CSV/JSON/API/XML feeds bypass DOM entirely and focus on structured table + integrity routes. Each detected path is logged via `plan.dynamic_paths` so future training jobs understand why a subset of routes ran.

#### **Step 3: Logging & Feedback**

- **All field extractions, corrections, and feedback** are logged as `.jsonl` files.
- **Manual and ML-powered correction health** (`manual_correction.py`) review these logs, allow user or ML/LLM corrections, and update the context library and/or context_library_db.

#### **Step 4: Database Update**

- **Confirmed/corrected data** is written to PostgreSQL via SQLAlchemy models.
- **Table structures, contests, and entities** are upserted for robust querying and ML training.

#### **Step 5: ML/NLP Retraining**

- **NER and other models** are retrained using the corrected data from logs and the context library.
- **Retraining scripts** (e.g., `retrain_table_structure_models.py`) pull from both the context library and the database.

---

### 3. **Where Each File Fits**

| File/Module | Main Role | Reads From | Writes To |
| --- | --- | --- | --- |
| `context_library.json` | Central knowledge base for context, patterns, mappings | Used by all context code | Updated by librarian / correction health |
| `context_library_db.json` | Structured/expanded context for ML/audit (optional) | Correction health, ML | Correction health |
| `context_elections.db` | Legacy SQLite DB (should be phased out) | Legacy code | Legacy code |
| `POSTGRES_URL` (PostgreSQL) | Main relational DB for all structured data | SQLAlchemy models | SQLAlchemy models |
| `fixtures/*.json` | Handler-ready election result fixtures (committed) | CSVs (local conversion) | Used by handlers, index builder |
| `cache/*.json` | Short-term runtime context/table/embedding cache (gitignored) | Handlers, extractors | PostgreSQL (via migration script) |
| `log/*.jsonl` | All logs: extraction, correction, feedback, anomalies, navigation, integrity, trust, sessions | Correction health, ML | All pipeline components |
| `build_election_index.py` | Builds validated election index from CSV/JSON/JSONL/cache/log sources | Fixtures, cache, log | `election_results_index.json`, audit report |
| `election_fixtures.py` | Fixture loader for handlers | `fixtures/*.json` | In-memory data structures |
| `Integrity_check.py` | Validates data quality, deduplication, anomaly detection | Parsed data, trust history | `log/integrity_monitor.jsonl`, PostgreSQL |
| `manual_correction.py` | Reviews logs, allows corrections, updates context library and DB | `log/`, `context_library` | `context_library`, DB |
| `librarian.py` | Centralizes context knowledge, extends/updates context library | `context_library` | `context_library` |
| `context_coordinator.py` | Orchestrates context enrichment, integrity, and ML checks | `context_library`, DB | `log/` |
| `context_organizer.py` | Organizes parsed context, deduplicates, runs ML, updates DB | `context_library`, DB | DB, `log/` |
| `retrain_table_structure_models.py` | Retrains NER and other models using context and logs | `context_library`, DB, `log/` | model files, `log/` |
| `config.py` | Centralizes all paths and DB connection strings | `.env`, filesystem | N/A |

---

### 4. **Optimization Recommendations**

#### **A. Paths and Structure**

- **Single Source of Truth:**
  - Use `context_library.json` as the canonical context source.
  - Use `librarian.py` for all context extension and updates.
- **Phase Out Legacy DB:**
  - Remove all references to `context_elections.db` unless needed for migration.
- **Explicit Context Library DB:**
  - If you need a structured context DB for ML, always generate it from `context_library.json` and logs, not as a separate manual source.

#### **B. Database Usage**

- **PostgreSQL for All Structured Data:**
  - All confirmed contests, table structures, entities, etc. should be stored in PostgreSQL.
  - Use SQLAlchemy models for all DB access.
- **Context Library for Knowledge, Not Data:**
  - Use the context library for normalization, mapping, and as a knowledge base, not for storing raw data.

#### **C. Logging and Correction**

- **All logs go to `log/` directory** with clear naming conventions.
- **Manual/ML correction health** should always update both the context library and the DB as needed.

#### **D. ML/NLP Retraining**

- **Always pull training data from the latest context library and DB.**
- **Keep logs of all corrections and retraining sessions** for auditability.

#### **E. Migration and Maintenance**

- **Use migration scripts** (like `context_migration.py`) to move legacy data into PostgreSQL.
- **Document all paths and roles** in your README or a `docs/` folder for future maintainers.

#### **F. Runtime Compatibility Guards**

- **`sitecustomize.py`** runs before any project import and installs safe shims (e.g., aliasing `click.parser.split_arg_string`) so third-party updates from spaCy/weasel do not flood tests with deprecation warnings.
- Keep the shim in place until upstream libraries drop the deprecated import. When versions are bumped, remove the alias and rerun the table-builder pytest suite to confirm the warning-free contract still holds.

---

### 5. **Summary Diagram**

[Election URLs (handler downloads)] | v [CSVs (local-only, gitignored)] ---manual conversion---> [fixtures/*.json (committed)] | | v v [Parser/Handlers] ---> [cache/*.json (short-term)] | [log/*.jsonl (append-only)] | | +------------------------+-------------------------+ | | | v v v [context_organizer.py] <--- [context_library.json] <--- [librarian.py] | | v v [context_coordinator.py] <--- [manual_correction.py] | | +-------------------------------+ | | | v v v [build_election_index.py] ---confidence filtering (≥0.7)---> [PostgreSQL Warehouse] | | | v v [Integrity_check.py] [ML/NLP Retraining Scripts]

---

### 6. **Actionable Steps**

1. **Audit all code for references to `context_elections.db` and remove/replace with PostgreSQL.**
2. **Ensure all context knowledge is loaded from and saved to `context_library.json` via `librarian.py`.**
3. **Make sure all logs are written to `log/` and processed by correction health.**
4. **Use PostgreSQL as the only source of structured, confirmed data for ML and reporting.**
5. **Document all key paths and their roles in your project.**
6. **For fixture pipeline:** Convert CSVs to JSON locally; commit only validated JSON/JSONL to `fixtures/`; use `--min-confidence 0.7` for production index builds; monitor cache/log growth and archive periodically.

---

Contributions welcome! See `CONTRIBUTING.md` to get started.

## 📂 Data Flow Example

1. **User chooses URL** from `urls.txt` (prompted via `prompt_user_input`).
2. **Browser is launched** via `browser_utils` (Playwright-first; Selenium fallback only if optional dependency is installed).

   - Before any handler or download prompt runs, the `NavigationInstructionRunner` loads context-aware recipes from `webapp/parser/navigator/navigation_recipes.orjson`.  These recipes describe DOM markers, selectors, and optional parallel action groups that toggle hidden views, fire menus, or kick off context scans.  The runner merges any projected data (e.g., contests, inferred years) back into the orchestration context so downstream handlers inherit the dynamically detected state/county metadata.

3. **CAPTCHA page is detected**, `captcha_tools` attempts resolution.

4. HTML is scanned by `html_scanner` to gather:
   - Election year (e.g. 2022)
   - Race categories (e.g. Governor, Senate, Proposition)
   - County names (if present)

5. **Routing**:
   - If `state_router` detects a handler → delegate to `handlers/<state>.py`
   - Otherwise → delegate to `format_router`
   - Downloaded files selected via `format_router` stay in the same pipeline; their parsed results now continue through the HTML parser's integrity/AI/export stages instead of short-circuiting after the download completes.
6. The **handler parses and returns**: headers, data, contest, metadata.

7. **Table extraction** is performed using `table_core.py` and `dynamic_table_extractor.py`, with ML/NLP scoring and patching.

8. **Election integrity checks** are run via `Context_Integration/Integrity_check.py`.

9. **CSV and metadata are saved** in `output/<state>/<county>/<race>/`.

10. **Logs and audit trails** are written for transparency and reproducibility.

---

## 📥 Input Directory (`input/`)

The `input/` folder is used for:

- Live downloads triggered from the parser pipeline (e.g., PDFs, JSONs).
- Manual file drops for override parsing (supported via `.env` and `process_format_override()`).
- Testing new handlers or extraction logic with static files.

Files are placed here by `download_utils.py` or manually.  
Manual parsing is supported if you use the correct naming convention and trigger via override.

### 🧭 Dynamic Navigation Recipes

- File: `webapp/parser/navigator/navigation_recipes.orjson`
- Loader: `NavigationRecipeStore`
- Executor: `NavigationInstructionRunner`

Each recipe defines matching constraints (`state`, `county`, URL fragments, DOM markers) and a list of steps.  Supported actions include waiting for selectors, clicking buttons, running JavaScript, automatic scrolling, projecting results from `scan_html_for_context`, and spawning **parallel** sub-steps to emulate multi-threaded navigation.  Recipes can project values (e.g., contests, inferred years) directly into the parser context so routing, contest selection, and ML scoring all share the same dynamically detected signals.  The shared runner executes before format detection, which means traditional handlers and download-based flows both inherit the navigation side effects (toggled panes, expanded menus, etc.).

- Every execution streams structured telemetry (per-step status, selectors, scroll metadata) into `log/navigation_learning_log.jsonl` through `ContextCoordinator.record_navigation_feedback()`.  Use `webapp/parser/navigator/training_data.py` to pull that log into an orjson dataset for ML retraining or to bootstrap new recipes programmatically.
- `webapp/parser/health/navigation_feedback_ingest.py` tails the same navigation log and converts fresh entries into `navigation_feedback_selection_log.jsonl`, so the existing manual correction bot can review wins/losses, auto-accept high-signal patterns, or trigger retraining without any extra tooling.

---

## 🛠️ Extensibility Guidelines

- **Add a new state/county:**  
  Create `handlers/states/<state>.py` or `handlers/states/<state>/county/<county>.py` and register in `state_router.py`.
- **Add a new file format:**  
  Add to `handlers/formats/` and map in `format_router.py`.
- **Custom contest filtering:**  
  Pass `noisy_labels` and `noisy_label_patterns` to `select_contest()` in your handler.
- **Bot tasks:**  
  Add to `health/health_router.py` and enable with `ENABLE_BOT_TASKS=true` in `.env`.
- **Azure Health Control Center:**  
  The `/azure_health` route in `Smart_Elections_Parser_Webapp.py` exposes a control panel where high-impact scripts (manual correction modes, log/cache cleanup, full `BotPipeline.run`, retraining, Integrity_check summaries, dataset promotion, etc.) can be launched from the browser.  Each task streams stdout back into the UI so operators on Azure (or localhost) can supervise long-running health work without shell access.
- **Context and correction:**  
  Add new context patterns or feedback to `context_library.json` or extend `context_organizer.py`.
- **User prompts:**  
  Always use `prompt_user_input()` for future web UI compatibility.
- **Testing files:**  
  Use the `input/` directory for static HTML/PDF/JSON testing.

---

## 🛡️ Election Integrity & Transparency

- **ML/NER-powered anomaly detection:**  
  All extracted data is checked for anomalies and inconsistencies using ML/NLP models.
- **Persistent context library:**  
  User feedback and corrections are stored and used to improve future extraction.
- **Audit trails:**  
  Every extraction, correction, and output is logged with metadata for reproducibility.
- **Human-in-the-loop:**  
  Manual correction health and feedback loops ensure continuous improvement and transparency.

---

## ✅ Future Enhancements

- More granular exception logging and error recovery.
- Pluggable browser agent rotation and advanced anti-bot detection.
- Shared election terminology models (e.g., race aliases, candidate normalization).
- Plugin registry for state/community contributions.
- File-based ingestion workflow with filename inference for manual CSV/PDF drop-ins.
- Web UI for user prompts, batch management, and correction review.
- Automated retraining pipeline for ML/NLP models based on correction logs.

---

## 📐 Table pipeline contract (normalized → wide)

This project enforces light, explicit contracts for table shapes at two stages. The checks are logging-only and never drop data automatically, but they surface issues early for fast iteration.

- Normalized stage (pre-pivot):
  - Should contain either a candidate-like column (e.g., Candidate/Name) or both a Total column and one or more ballot-method columns.
  - Location/Precinct is recommended but not required.

- Wide stage (post-pivot):
  - Should contain at least one numeric vote column.
  - Candidate names are expected in the header row (after pivot).

When a stage looks incomplete, the builder emits a structured log event:

- event: schema_check
- stage: normalized | wide
- status: ok | weak
- counters: headers, rows, has_precinct, has_total, has_percent, candidates, ballots

## 🔤 Canonical column order and ballot methods

The builder applies a stable column order for consistency across formats:

1) Precinct
2) Candidate columns
3) Ballot-method columns ordered by `BALLOT_TYPES_SORT_ORDER`
4) Percent Reported
5) Total Vote / Grand Total
6) Remaining columns (as encountered)

All Absentee–Military variants are normalized to the canonical label “Absentee Military”, and duplicate/synonym ballot-method columns are merged by summing numeric values.

## 🧰 Troubleshooting schema warnings

- “normalized schema weak”
  - Ensure the table has a candidate-like column or a totals-plus-ballot combination.
  - Check header normalization: mixed-case and spacing are normalized; duplicates are deduped with suffixes.

- “wide schema weak”
  - Ensure the pipeline produced at least one numeric column; if not, pivot may have kept the table in a normalized shape (by design). Consider setting `context["skip_pivot"] = True` to inspect normalized output.

You can safely ignore these warnings when exploring new formats—they’re there to help you spot missing signals early.

---

## Auto Inventory (Regenerate)

Run this to rebuild the inventory section from source files:

```bash
python automate.py  # Runs all automated scripts including pipeline audit
# Or specifically for pipeline audit:
python -c "from webapp.parser.utils.shared_logic import generate_pipeline_map; generate_pipeline_map(project_root=r'.', out_markdown=r'docs/pipeline_map.md')"
```

The block below is auto-generated. Do not edit between markers.

<!-- AUTO-INVENTORY:START -->

Inventory summary: 160 files, ~195144 non-empty LOC

### Docs

- `docs/Election Integrity Guidelines.md` (loc: 28)
- `docs/architecture.md` (loc: 264)
- `docs/handlers.md` (loc: 145)
- `docs/index.md` (loc: 61)
- `docs/roadmap.md` (loc: 78)
- `docs/troubleshooting.md` (loc: 77)

### Misc

- `.Dockerfile` (loc: 60)
- `.env` (loc: 153)
- `.env.template` (loc: 165)
- `.gitattributes` (loc: 17)
- `.github/chatmodes/test.chatmode.md` (loc: 5)
- `.github/workflows/main_ballotlens.yml` (loc: 176)
- `.gitignore` (loc: 129)
- `.vscode/launch.json` (loc: 31)
- `.vscode/settings.json` (loc: 11)
- `CONTRIBUTING.md` (loc: 229)
- `input/.download_manifest.jsonl` (loc: 6)
- `input/export-GE2024Results.json` (loc: 1)
- `license.md` (loc: 17)
- `output/United_States_Senator__20251022_175606.csv` (loc: 3)
- `output/United_States_Senator__20251022_175606.metadata.json` (loc: 34)
- `output/United_States_Senator__20251022_175606.xlsx` (loc: 158)
- `output/new_york__new_york__Democratic_District_Attorney_New_York_2025__20251022_175411.csv` (loc: 3)
- `output/new_york__new_york__Democratic_District_Attorney_New_York_2025__20251022_175411.metadata.json` (loc: 43)
- `output/new_york__new_york__Democratic_District_Attorney_New_York_2025__20251022_175411.xlsx` (loc: 174)
- `output/ocr_debug/Democratic_District_Attorney_New_York_2025_p1_300dpi.png` (loc: 5835)
- `output/ocr_debug/Democratic_District_Attorney_New_York_2025_p2_300dpi.png` (loc: 9005)
- `readme.md` (loc: 279)
- `requirements.txt` (loc: 47)
- `uploads/01101200010New York Democratic District Attorney New York Recap.csv` (loc: 9144)
- `uploads/Democratic District Attorney New York 2025.pdf` (loc: 85736)

## Schema events (normalized and wide stages)

The table builder emits structured schema_check events to make pipeline validation observable and testable. These are informational and do not stop the pipeline; they indicate the strength of the current shape.

- Event name: schema_check
- Emitted twice per run: at normalized and wide stages
- Status values:
  - ok: heuristic expectations met for the stage
  - weak: expectations not met (actionable for tests/alerts), but processing continues
  - error: validator encountered an exception (rare; investigate)

Example payloads:

Normalized stage (pre-pivot):

```json
{
  "level": "INFO",
  "type": "builder",
  "message": {
    "event": "schema_check",
    "stage": "normalized",
    "status": "ok",
    "headers": 6,
    "rows": 124,
    "has_precinct": true,
    "has_total": true,
    "has_percent": false,
    "candidates": 1,
    "ballots": 3
  }
}
```

Wide stage (post-pivot):

```json
{
  "level": "INFO",
  "type": "builder",
  "message": {
    "event": "schema_check",
    "stage": "wide",
    "status": "ok",
    "headers": 28,
    "rows": 124,
    "has_precinct": true,
    "has_total": true,
    "has_percent": true,
    "candidates": 0,
    "ballots": 0
  }
}
```

Notes:

- Normalized stage is considered acceptable when either a Candidate-like column is present or a Total + one or more ballot-method columns exist.
- Wide stage expects at least one numeric vote column after pivoting.
- Tests can assert on presence of these events and their statuses to detect regressions without relying on exact header names.
- `uploads/export-GE2024Results.json` (loc: 1)
- `webapp/Smart_Elections_Parser_Webapp.py` (funcs: 69, classes: 1, loc: 1879)
- `webapp/__init__.py` (funcs: 0, classes: 0, loc: 0)
- `webapp/parser/Context_Integration/Context_Library/.processed_urls` (loc: 37)
- `webapp/parser/Context_Integration/Context_Library/cache/context_cache.json` (loc: 50)
- `webapp/parser/Context_Integration/Context_Library/cache/embedding_disk_cache.pkl` (loc: 1)
- `webapp/parser/Context_Integration/Context_Library/constants.py` (funcs: 11, classes: 0, loc: 2459)
- `webapp/parser/Context_Integration/Context_Library/context_library.json` (loc: 295)
- `webapp/parser/Context_Integration/Context_Library/context_library.json.20250915_123644.bak` (loc: 295)
- `webapp/parser/Context_Integration/Context_Library/context_library.json.20250917_181646.bak` (loc: 295)
- `webapp/parser/Context_Integration/Context_Library/context_library.json.20250917_181653.bak` (loc: 295)
- `webapp/parser/Context_Integration/Context_Library/context_library.json.20250917_181700.bak` (loc: 295)
- `webapp/parser/Context_Integration/Context_Library/context_library.json.20250917_181706.bak` (loc: 295)
- `webapp/parser/Context_Integration/Context_Library/log/dom_pattern_kb.jsonl` (loc: 599)
- `webapp/parser/Context_Integration/Context_Library/log/field_selection_log.jsonl` (loc: 986)
- `webapp/parser/Context_Integration/Context_Library/log/run_history.ndjson` (loc: 75)
- `webapp/parser/Context_Integration/Context_Library/log/sess_sess_77gkh9p97.ndjson` (loc: 1)
- `webapp/parser/Context_Integration/Context_Library/log/sess_sess_v07taenjq.ndjson` (loc: 36)
- `webapp/parser/Context_Integration/Integrity_check.py` (funcs: 18, classes: 0, loc: 369)
- `webapp/parser/Context_Integration/context.cs` (loc: 60)
- `webapp/parser/Context_Integration/context_coordinator.py` (funcs: 3, classes: 1, loc: 3137): context_coordinator.py
- `webapp/parser/Context_Integration/context_organizer.py` (funcs: 6, classes: 1, loc: 1751): context_organizer.py
- `webapp/parser/Context_Integration/librarian.py` (funcs: 34, classes: 0, loc: 597)
- `webapp/parser/Context_Integration/vocab/counties.txt` (loc: 0)
- `webapp/parser/Context_Integration/vocab/states.txt` (loc: 0)
- `webapp/parser/Context_Integration/vocab/types.txt` (loc: 0)
- `webapp/parser/Context_Integration/vocab/words.txt` (loc: 0)
- `webapp/parser/Context_Integration/vocab/years.txt` (loc: 0)
- `webapp/parser/config.py` (funcs: 3, classes: 0, loc: 394): Central configuration module for the Smart Elections Parser Webapp.
- `webapp/parser/data_manager.py` (funcs: 11, classes: 0, loc: 185)
- `webapp/parser/handlers/__init__.py` (funcs: 0, classes: 0, loc: 0)
- `webapp/parser/handlers/batch_handler.py` (funcs: 0, classes: 0, loc: 0)
- `webapp/parser/handlers/formats/__init__.py` (funcs: 0, classes: 0, loc: 0)
- `webapp/parser/handlers/formats/csv_handler.py` (funcs: 3, classes: 0, loc: 216)
- `webapp/parser/handlers/formats/html_handler.py` (funcs: 1, classes: 0, loc: 243)
- `webapp/parser/handlers/formats/json_handler.py` (funcs: 5, classes: 0, loc: 361)
- `webapp/parser/handlers/formats/pdf_handler.py` (funcs: 38, classes: 0, loc: 1976)
- `webapp/parser/handlers/states/arizona/__init__.py` (funcs: 0, classes: 0, loc: 2)
- `webapp/parser/handlers/states/arizona/arizona.py` (funcs: 1, classes: 0, loc: 168)
- `webapp/parser/handlers/states/example state/__init__.py` (funcs: 0, classes: 0, loc: 0)
- `webapp/parser/handlers/states/example state/example_county/__init__.py` (funcs: 0, classes: 0, loc: 0)
- `webapp/parser/handlers/states/example state/example_county/example_county.py` (funcs: 2, classes: 0, loc: 136)
- `webapp/parser/handlers/states/example state/example_state.py` (funcs: 2, classes: 0, loc: 161)
- `webapp/parser/handlers/states/new_york/__init__.py` (funcs: 0, classes: 0, loc: 0)
- `webapp/parser/handlers/states/new_york/county/__init__.py` (funcs: 0, classes: 0, loc: 0)
- `webapp/parser/handlers/states/new_york/county/rockland.py` (funcs: 1, classes: 0, loc: 315)
- `webapp/parser/handlers/states/new_york/new_york.py` (funcs: 1, classes: 0, loc: 37)
- `webapp/parser/handlers/states/pennsylvania/__init__.py` (funcs: 0, classes: 0, loc: 2)
- `webapp/parser/handlers/states/pennsylvania/pennsylvania.py` (funcs: 2, classes: 0, loc: 178)
- `webapp/parser/health/__init__.py` (funcs: 0, classes: 0, loc: 0)
- `webapp/parser/health/context_migration.py` (funcs: 8, classes: 0, loc: 219)
- `webapp/parser/health/health_router.py` (funcs: 3, classes: 1, loc: 493)
- `webapp/parser/health/log_cache_cleaner_bot.py` (funcs: 14, classes: 0, loc: 544): log_cache_cleaner_bot.py
- `webapp/parser/health/manual_correction_bot.py` (funcs: 37, classes: 0, loc: 1281): manual_correction.py
- `webapp/parser/health/retrain_table_structure_models.py` (funcs: 24, classes: 2, loc: 868)
- `webapp/parser/health/scan_misaligned_ner.py` (funcs: 4, classes: 0, loc: 154)
- `webapp/parser/html_election_parser.py` (funcs: 9, classes: 0, loc: 937)
- `webapp/parser/services/context_service.py` (funcs: 0, classes: 2, loc: 370)
- `webapp/parser/services/election_data_services.py` (funcs: 10, classes: 2, loc: 812): ElectionDataService: Service layer for all election DB operations.
- `webapp/parser/state_router.py` (funcs: 11, classes: 0, loc: 496)
- `webapp/parser/urls.txt` (loc: 19)
- `webapp/parser/utils/__init__.py` (funcs: 0, classes: 0, loc: 0)
- `webapp/parser/utils/browser_utils.py` (funcs: 27, classes: 1, loc: 642)
- `webapp/parser/utils/camelot_utils.py` (funcs: 6, classes: 0, loc: 116)
- `webapp/parser/utils/captcha_tools.py` (funcs: 5, classes: 4, loc: 139)
- `webapp/parser/utils/contest_selector.py` (funcs: 32, classes: 1, loc: 1088)
- `webapp/parser/utils/date_utils.py` (funcs: 1, classes: 0, loc: 14): date_utils.py
- `webapp/parser/utils/db_utils.py` (funcs: 28, classes: 0, loc: 375)
- `webapp/parser/utils/detect.py` (funcs: 21, classes: 2, loc: 365): detect.py
- `webapp/parser/utils/detector.py` (funcs: 2, classes: 2, loc: 169): detector.py
- `webapp/parser/utils/dom_extractor.py` (funcs: 5, classes: 0, loc: 162): dom_extractor.py
- `webapp/parser/utils/download_utils.py` (funcs: 10, classes: 0, loc: 143)
- `webapp/parser/utils/dynamic_table_extractor.py` (funcs: 25, classes: 0, loc: 1090)
- `webapp/parser/utils/embedding_cache.py` (funcs: 8, classes: 0, loc: 320)
- `webapp/parser/utils/extraction_strategies.py` (funcs: 11, classes: 0, loc: 266): extraction_strategies.py
- `webapp/parser/utils/format_router.py` (funcs: 9, classes: 0, loc: 573)
- `webapp/parser/utils/header_utils.py` (funcs: 2, classes: 0, loc: 49)
- `webapp/parser/utils/html_scanner.py` (funcs: 44, classes: 0, loc: 3096)
- `webapp/parser/utils/logger_singleton.py` (funcs: 2, classes: 0, loc: 24)
- `webapp/parser/utils/merge_utils.py` (funcs: 1, classes: 0, loc: 37): merge_utils.py
- `webapp/parser/utils/misc_utils.py` (funcs: 5, classes: 0, loc: 71)
- `webapp/parser/utils/ml_table_detector.py` (funcs: 11, classes: 0, loc: 386)
- `webapp/parser/utils/model_registry.py` (funcs: 4, classes: 4, loc: 550)
- `webapp/parser/utils/models.py` (funcs: 1, classes: 30, loc: 411)
- `webapp/parser/utils/output_utils.py` (funcs: 18, classes: 0, loc: 527)
- `webapp/parser/utils/pattern_extractor.py` (funcs: 2, classes: 0, loc: 90): pattern_extractor.py
- `webapp/parser/utils/pivot.py` (funcs: 28, classes: 0, loc: 1248): pivot.py
- `webapp/parser/utils/rawjson_utils.py` (funcs: 6, classes: 0, loc: 205)
- `webapp/parser/utils/salvage.py` (funcs: 3, classes: 0, loc: 126): salvage.py
- `webapp/parser/utils/seleniumbase_launcher.py` (funcs: 4, classes: 0, loc: 87)
- `webapp/parser/utils/shared_logger.py` (funcs: 1, classes: 3, loc: 559)
- `webapp/parser/utils/shared_logic.py` (funcs: 88, classes: 10, loc: 1571)
- `webapp/parser/utils/spacy_utils.py` (funcs: 26, classes: 0, loc: 243)
- `webapp/parser/utils/strategy_concurrency.py` (funcs: 2, classes: 0, loc: 115): strategy_concurrency.py
- `webapp/parser/utils/structure_cache.py` (funcs: 3, classes: 0, loc: 20): structure_cache.py
- `webapp/parser/utils/table_builder.py` (funcs: 17, classes: 0, loc: 942)
- `webapp/parser/utils/table_core.py` (funcs: 8, classes: 0, loc: 414): table_core.py (refactored orchestrator)
- `webapp/parser/utils/user_prompt.py` (funcs: 2, classes: 3, loc: 846)
- `webapp/parser/utils/xlsx_exporter.py` (funcs: 3, classes: 0, loc: 217)
- `webapp/parser/web_pipeline.py` (funcs: 4, classes: 1, loc: 257)
- `webapp/static/css/data_framework.css` (loc: 654)
- `webapp/static/css/history.css` (loc: 314)
- `webapp/static/css/main.css` (loc: 608)
- `webapp/static/css/ballot_lens_modern.css` (loc: 1722)
- `webapp/static/favicon.ico` (loc: 0)
- `webapp/static/icons/apple-touch-icon.png` (loc: 596)
- `webapp/static/icons/favicon-32.png` (loc: 66)
- `webapp/static/icons/icon-192.png` (loc: 810)
- `webapp/static/icons/icon-512.png` (loc: 3054)
- `webapp/static/icons/icon-maskable-192.png` (loc: 725)
- `webapp/static/icons/icon-maskable-512.png` (loc: 2831)
- `webapp/static/img/earth.png` (loc: 17764)
- `webapp/static/img/moon.png` (loc: 4032)
- `webapp/static/img/moon.svg` (loc: 226)
- `webapp/static/img/sun.png` (loc: 2892)
- `webapp/static/img/sun.svg` (loc: 1)
- `webapp/static/js/data_framework.js` (loc: 438)
- `webapp/static/js/history.js` (loc: 277)
- `webapp/static/js/main.js` (loc: 701)
- `webapp/static/js/nav_guard.js` (loc: 84)
- `webapp/static/js/ballot_lens_modern.js` (loc: 2469)
- `webapp/static/vendor/bootstrap-5.3.8.bundle.min.js` (loc: 7)
- `webapp/static/vendor/bootstrap-5.3.8.min.css` (loc: 6)
- `webapp/static/vendor/socket.io-4.7.5.min.js` (loc: 7)
- `webapp/templates/data_framework.html` (loc: 117)
- `webapp/templates/history.html` (loc: 254)
- `webapp/templates/index.html` (loc: 97)
- `webapp/templates/ballot_lens.html` (loc: 305)

### Tests

- `webapp/tests/test_header_normalization.py` (funcs: 4, classes: 0, loc: 45)

<!-- AUTO-INVENTORY:END -->

Contributions welcome! See `CONTRIBUTING.md` to get started.
