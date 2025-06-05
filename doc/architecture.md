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
  - Each state script must export a `parse(page, html_context)` method and return `(headers, data, contest_title, metadata)`.
  - County-level handlers live in `handlers/states/<state>/county/`.

- **`handlers/formats/`**
  - Generic format parsers: `pdf_handler.py`, `json_handler.py`, `csv_handler.py`, `html_handler.py`.
  - Used when no specialized state handler exists.
  - Must also return `(headers, data, contest_title, metadata)`.

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

## 🤖 Bots, Context, and Web UI Integration

- **`bots/`**
  - Correction, retraining, and automation bots (see `bot_router.py`).
  - Includes manual correction bot and retraining pipeline.
  - Bots can be enabled via `.env` (`ENABLE_BOT_TASKS=true`).

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
6. The **handler parses and returns**: headers, data, contest_title, metadata.
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
  Add to `bots/bot_router.py` and enable with `ENABLE_BOT_TASKS=true` in `.env`.
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
  Manual correction bots and feedback loops ensure continuous improvement and transparency.

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
