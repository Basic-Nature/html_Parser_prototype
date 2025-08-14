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
  - Launches Playwright or Selenium browser, supports headless/GUI, and user-agent spoofing.

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

---

## 🤖 ML, Context, and Web UI Integration

- **`health/`**
  - Correction, retraining, and automation health (see `health_router.py`).
  - Includes manual correction and retraining pipeline.

- **`Context_Integration/`**
  - Context, ML/NLP, and integrity modules:
    - `context_coordinator.py`: Orchestrates context analysis, NLP, and ML integrity checks.
    - `context_organizer.py`: Context enrichment, clustering, and persistent context library management.
    - `Integrity_check.py`: Election integrity and anomaly detection logic.

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

#### **A. Context Library (`context_library.json`)**

- **Purpose:**  
  - The original, central source of contextual knowledge (states, counties, contests, patterns, etc.).
  - Used for lookups, normalization, and as a knowledge base for parsing and ML.
- **Location:**  
  - `webapp/parser/Context_Integration/Context_Library/context_library.json`
- **Accessed by:**  
  - `context_coordinator.py`, `context_organizer.py`, `librarian.py`, and ML health.

#### **B. Context Library DB (`context_library_db.json`)**

- **Purpose:**  
  - A more structured or expanded version of the context library, possibly for ML or audit.
  - Generated/updated by `manual_correction.py` and possibly others.
- **Location:**  
  - Same directory as above.
- **Accessed by:**  
  - Correction, possibly ML retraining scripts.

#### **C. Context DB (`context_elections.db`)**

- **Purpose:**  
  - Legacy SQLite DB, now mostly replaced by PostgreSQL.
  - May still be referenced for backward compatibility or migration.
- **Location:**  
  - Same directory as above.
- **Accessed by:**  
  - Should be phased out if you’re fully on PostgreSQL.

#### **D. PostgreSQL (`POSTGRES_URL`)**

- **Purpose:**  
  - The main, production-grade relational database for all structured data (contests, table structures, entities, etc.).
  - Used for robust querying, updates, and ML training data storage.
- **Accessed by:**  
  - All SQLAlchemy-based models and session logic.

#### **E. Logs (`log/` directory)**

- **Purpose:**  
  - Store all extraction, correction, feedback, and anomaly logs as `.jsonl` files.
  - Serve as the audit trail and as a source for manual/ML correction and retraining.
- **Accessed by:**  
  - `manual_correction.py`, `librarian.py`, retraining scripts, and context health.

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

| File/Module                       | Main Role                                                                                   | Reads From                | Writes To                 |
|------------------------------------|--------------------------------------------------------------------------------------------|---------------------------|---------------------------|
| `context_library.json`             | Central knowledge base for context, patterns, mappings                                     | Used by all context code  | Updated by librarian/correction health |
| `context_library_db.json`          | Structured/expanded context for ML/audit (optional)                                        | Correction health, ML       | Correction health           |
| `context_elections.db`             | Legacy SQLite DB (should be phased out)                                                    | Legacy code               | Legacy code               |
| `POSTGRES_URL` (PostgreSQL)        | Main relational DB for all structured data                                                 | SQLAlchemy models         | SQLAlchemy models         |
| `log/*.jsonl`                      | All logs: extraction, correction, feedback, anomalies, etc.                                | Correction health, ML       | All pipeline components   |
| `manual_correction.py`         | Reviews logs, allows corrections, updates context library and DB                           | log/, context_library     | context_library, DB       |
| `librarian.py`                     | Centralizes context knowledge, extends/updates context library                             | context_library           | context_library           |
| `context_coordinator.py`           | Orchestrates context enrichment, integrity, and ML checks                                 | context_library, DB       | log/                      |
| `context_organizer.py`             | Organizes parsed context, deduplicates, runs ML, updates DB                               | context_library, DB       | DB, log/                  |
| `retrain_table_structure_models.py`| Retrains NER and other models using context and logs                                       | context_library, DB, log/ | model files, log/         |
| `config.py`                        | Centralizes all paths and DB connection strings                                            | .env, filesystem          | N/A                       |

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

---

### 5. **Summary Diagram**

[HTML/CSV/PDF] | v [Parser/Handlers] ---> [log/*.jsonl] <---+ | | v v [context_organizer.py] <--- [context_library.json] <--- [librarian.py] | | v v [context_coordinator.py] <--- [manual_correction.py] | | v v [PostgreSQL (SQLAlchemy models)] <--- [context_migration.py] | v [ML/NLP Retraining Scripts]

---

### 6. **Actionable Steps**

1. **Audit all code for references to `context_elections.db` and remove/replace with PostgreSQL.**
2. **Ensure all context knowledge is loaded from and saved to `context_library.json` via `librarian.py`.**
3. **Make sure all logs are written to `log/` and processed by correction health.**
4. **Use PostgreSQL as the only source of structured, confirmed data for ML and reporting.**
5. **Document all key paths and their roles in your project.**

---

Contributions welcome! See `CONTRIBUTING.md` to get started.

## 📂 Data Flow Example

1. **User chooses URL** from `urls.txt` (prompted via `prompt_user_input`).
2. **Browser is launched** via `browser_utils` (Playwright or Selenium).
3. **CAPTCHA page is detected**, `captcha_tools` attempts resolution.
4. HTML is scanned by `html_scanner` to gather:
   - Election year (e.g. 2022)
   - Race categories (e.g. Governor, Senate, Proposition)
   - County names (if present)
5. **Routing**:
   - If `state_router` detects a handler → delegate to `handlers/<state>.py`
   - Otherwise → delegate to `format_router`
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

Contributions welcome! See `CONTRIBUTING.md` to get started.
