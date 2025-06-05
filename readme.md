# Smart Elections Parser

## Overview

Smart Elections Parser is a robust, modular, and integrity-focused precinct-level election result scraper and analyzer. It is designed to adapt to the ever-changing landscape of U.S. election reporting, supporting both traditional and modern web formats, and is built for extensibility, transparency, and auditability.

---

## 🚀 What's New (2025)

### Major Additions

- **Dynamic Table Extraction & Structure Learning**
  - Centralized in `table_core.py` and `dynamic_table_extractor.py`
  - Multi-strategy extraction: HTML tables, repeated DOM, pattern-based, ML/LLM, and plugin-based
  - Table structure learning, harmonization, and feedback are now fully centralized
  - ML/NER-powered entity annotation and structure verification
  - Dynamic scoring and patching: extraction methods are scored and can "fill in the blanks" using information from other strategies

- **Context-Aware Orchestration**
  - `context_coordinator.py` and `context_organizer.py` orchestrate advanced context analysis, NLP, and ML integrity checks
  - Persistent context library (`context_library.json`) for learning from user feedback and corrections
  - Automated anomaly detection, clustering, and integrity checks (see `Integrity_check.py`)

- **Web UI & CLI Parity**
  - Flask-based web interface for managing URLs, running the parser, and reviewing output
  - Real-time log streaming via SocketIO
  - Data management dashboard for uploads, downloads, and URL hint management

- **Handler Architecture**
  - Modular state/county/format handlers in `handlers/`
  - Handlers can delegate to county-level or format-level logic
  - Shared logic and utilities for contest selection, table extraction, and output formatting

- **Election Integrity & Transparency**
  - ML/NER-based anomaly detection and cross-field validation
  - Persistent logs and feedback loops for user corrections and audit trails
  - Manual correction bot and retraining pipeline for continuous improvement
  - All outputs are saved with rich metadata and context for reproducibility

- **Security & Compliance**
  - Path traversal and injection protections on all file/database operations
  - .env-driven configuration for all sensitive settings
  - No credentials or session tokens are stored; web UI can be secured for public deployment

---

## 🧭 Design Philosophy

- **Single Source of Truth:** All table extraction, harmonization, and feedback logic is centralized for maintainability and learning.
- **Extensible & Pluggable:** New extraction strategies, handlers, and ML models can be added without breaking the pipeline.
- **Human-in-the-Loop:** User feedback is integrated at every stage, from contest selection to table correction.
- **Election Integrity First:** Every step is logged, auditable, and designed to surface anomalies or suspicious data.
- **Web & CLI Parity:** All features are available via both the command line and the web interface.

---

## 🔧 Features

- **Multi-Strategy Table Extraction:** HTML tables, repeated DOM, pattern-based, ML/LLM, plugin, and fallback NLP extraction.
- **Dynamic Scoring & Patching:** Extraction strategies are scored (ML/NER + heuristics); missing info is patched from other strategies when possible.
- **Persistent Context Library:** Learns from user corrections and feedback for smarter future extraction.
- **Contest & Handler Routing:** Dynamic state/county/format handler routing with fuzzy matching and context enrichment.
- **Election Integrity Checks:** ML/NER anomaly detection, cross-field validation, and audit logs.
- **Web UI:** Real-time log streaming, data management, and user-friendly contest/table review.
- **Batch & Parallel Processing:** Multiprocessing support for large-scale scraping.
- **Security:** Path safety, .env config, and no credential storage.

---

- **Headless or GUI Mode**: Browser launches headlessly by default unless CAPTCHA triggers a human interaction.
- **CAPTCHA-Resilient**: Dynamically detects and pauses for Cloudflare verification with a visible browser.
- **Race-Year Detection**: Scans HTML to find available election years and contests.
- **State-Aware Routing**: Automatically detects state context and delegates to appropriate handler module.
- **Format-Aware Fallback**: Supports CSV, JSON, PDF, and HTML formats with pluggable handlers.
- **Output Sorting**: Results saved in nested folders by state, county, and race.
- **URL Selection**: Loads URLs from `urls.txt` and lets users select specific targets.
- **.env Driven**: Easily override behavior such as CAPTCHA timeouts or headless preferences.
- **Bot Integration**: Enable `/bot` tasks (e.g., notifications, batch scans) via `.env`.
- **Web UI Ready**: All user prompts are modular for future web interface integration.

---

## 🖥️ Web UI (Optional)

**The Smart Elections Parser can be used in two ways:**

1. **Standalone Python Script:**  
   - Run `html_election_parser.py` directly from your IDE or terminal for full CLI control.
   - No web server required.

2. **Web UI (Optional):**  
   - A modern Flask-based web interface is included for users who prefer a graphical experience or are new to coding.
   - **Key Features of the Web UI:**
     - **Dashboard:** Overview of the parser and quick access to all tools.
     - **URL Hint Manager:** Add, edit, import/export, and validate custom URL-to-handler mappings.
     - **Change History:** View and restore previous configurations for transparency and auditability.
     - **Run Parser:** Trigger the parser from the browser and view real-time output in a styled terminal-like area.
     - **Live Feedback:** See parser logs as they happen (via WebSockets).
     - **Accessible:** Designed for both technical and non-technical users, making it ideal for teams, researchers, and those learning to code.
   - **How to Use the Web UI:**
     1. Install requirements:  
        `pip install -r requirements.txt`
        `python -m spacy download en_core_web_sm`
     2. Set up your `.env` file as needed.
     3. Start the web server:  
        `python webapp/Smart_Elections_Parser_Webapp.py`
     4. Open your browser to `http://localhost:5000`
   - The web UI is optional—**all core parser features remain available via the CLI**.

---

## How to Add a New State/County Handler, Format, or Bot Task

1. **State/County Handler:**  
   - Create a new handler in `handlers/states/` or `handlers/counties/`.
   - Implement a `parse(page, html_context)` function.
   - Register your handler in `state_router.py`.

2. **Bot Task:**  
   - Add your bot logic to `bot/bot_router.py`.
   - Implement a `run_bot_task(task_name, context)` function.
   - Enable with `ENABLE_BOT_TASKS=true` in `.env`.

3. **Custom Noisy Labels/Patterns:**  
   - In your handler, pass `noisy_labels` and `noisy_label_patterns` to `select_contest()` for contest filtering.

4. **Format Handler:**  
   - Add your handler to `utils/format_router.py` and register it in `route_format_handler`.

5. **User Prompts:**  
   - Use `prompt_user_input()` for all user input to allow easy web UI integration later.  
   - Example:

     ``python
     from utils.user_prompt import prompt_user_input
     url = prompt_user_input("Enter URL: ")
     ``

---

## 🗂 Folder Structure

``

```bash
html_Parser_prototype/
├── webapp/
│   ├── Smart_Elections_Parser_Webapp.py    # Flask web UI
│   ├── parser/
│   │   ├── html_election_parser.py         # Main CLI orchestrator
│   │   ├── state_router.py                 # Dynamic handler routing
│   │   ├── utils/
│   │   │   ├── table_core.py               # Centralized table extraction/learning
│   │   │   ├── dynamic_table_extractor.py  # Candidate table generator/scorer
│   │   │   ├── ml_table_detector.py        # ML/LLM table detection
│   │   │   ├── shared_logger.py            # Logging utilities
│   │   │   ├── user_prompt.py              # CLI/web prompt utilities
│   │   │   └── ...                         # (browser, captcha, etc.)
│   │   ├── Context_Integration/
│   │   │   ├── context_coordinator.py      # Context/NLP/ML orchestrator
│   │   │   ├── context_organizer.py        # Context enrichment, clustering, DB
│   │   │   └── Integrity_check.py          # Election integrity/anomaly checks
│   │   ├── handlers/
│   │   │   ├── states/                     # State/county handlers
│   │   │   ├── formats/                    # Format handlers (csv, pdf, json, html)
│   │   │   └── shared/                     # Shared handler logic
│   │   ├── bots/                           # Correction/retraining bots
│   │   ├── templates/                      # Web UI templates
│   │   ├── input/                          # Input data
│   │   ├── output/                         # Output data
│   │   ├── log/                            # Logs
│   │   ├── .env
│   │   ├── .env.template
│   │   └── requirements.txt

---
## 🧪 How to Use

1. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm

2. **Configure Settings (Optional)**
   Copy `.env.template` to `.env` and modify:

   ``bash
   cp .env.template .env
   ``

   Variables include:

   - `HEADLESS=false` (for debugging)
   - `CAPTCHA_TIMEOUT=300`
   - `CACHE_PROCESSED=true`
   - `ENABLE_BOT_TASKS=true` (to enable bot features)
   - `ENABLE_PARALLEL=true` (to enable multiprocessing)

3. **Add URLs**
   - Populate `urls.txt` with target election result URLs.
   - `url_hint_overrides.txt` is used in conjunction with `state_router.py` when dynamic state detection fails.

4. **Run Parser (CLI)**

   ``bash <--Terminal
   (uncomment the "")--
   `python -m webapp.parser.html_election_parser`
   if terminal already in root folder; otherwise,
    (replace full path with the actual path to the folder)
    "cd ...full path...\html_Parser_prototype"
   ``

5. **Run Parser (Web UI, Optional)**

   ``bash<Same as above with "" and folder path>
    `python -m webapp.Smart_Elections_Parser_Webapp`
   ``"cd ...full path...\html_Parser_prototype\"
   - Then visit [http://localhost:5000](http://localhost:5000) in your browser or more likely the printed to terminal IP address pasted into browser of choice.

---

## 📦 Output Format

All parsed results are saved in a structured, transparent, and auditable format:

### 📁 Directory Structure

```bash

output/{state}/{county}/{race}/{contest}_results.csv

```bash

**Example:**
```

output/arizona/maricopa/us_senate/kari_lake_results.csv

### 📄 Output Files

For each contest, the following files are generated:

- **CSV Results:**  
   Tabular results for the contest, ready for analysis.

- **Metadata JSON:**  
   Includes key information such as:
  - `state`
  - `county`
  - `year`
  - `race`
  - `contest`
  - `handler`
  - `timestamp`
  - Additional extraction context

- **Audit Trail:**  
   A detailed log of extraction steps, harmonization, user corrections, and any anomalies detected, ensuring full transparency and reproducibility.

---

## 🧩 Extending the Parser

Add New Extraction Strategies: Implement in table_core.py or as a plugin.
Add Handlers: Place new state/county/format handlers in handlers/.
Improve ML/NER: Retrain models using the correction bot and logs.
Election Integrity: All new logic should log decisions and support auditability.

- **Add New States**: Create a new file in `handlers/states/` (e.g. `georgia.py`) and implement a `parse()` method.
- **Add Format Support**: Add new file in `handlers/formats/` and map in `format_router.py`.
- **Shared Behavior**: Use `utils/shared_logic.py` for common race detection, total extraction, etc.
- **Add Bot Tasks**: Add new automation/notification logic in `bot/bot_router.py`.

---

## 🔐 Security & Integrity

- **Headless Scraping:** All scraping runs headlessly by default; a visible browser is launched only if CAPTCHA is triggered.
- **.env Protection:** Sensitive settings are managed via `.env`, which is excluded from version control (`.gitignore`).
- **No Credential Storage:** No credentials or session tokens are stored at any time.
- **Path Safety:** All file and database operations are path-safe and `.env`-configured to prevent injection or traversal attacks.
- **Web UI Security:** The web interface can be protected with authentication when deployed publicly.
- **Auditability:** All user feedback and corrections are logged for transparency and audit trails.
- **Election Integrity:** ML/NER-powered anomaly detection, cross-field validation, and persistent logs enforce data integrity.

---

## 🚧 Roadmap

- Multi-race selection prompt
- Retry logic for failed URLs
- Browser fingerprint obfuscation
- Contributor upload queue (for handler patches)
- YAML config option for handler metadata
- Web UI for user prompts and batch management

---

## 🛡️ Smart Elections Ambition

Smart Elections Parser is built to set a new standard for election data integrity and transparency. Every extraction, correction, and output is:

- **Auditable:** Full logs and metadata for every step.
- **Verifiable:** ML/NER-powered anomaly detection and structure validation.
- **Correctable:** Human-in-the-loop feedback at every stage.
- **Extensible:** Ready for new formats, handlers, and AI/ML improvements.
- **Secure:** Designed for safe, compliant, and transparent operation.

## 📄 License

MIT License (TBD)

---

## 🙋‍♀️ Contributors

- Lead Dev: [Juancarlos Barragan]
- Elections Research: TBD
- Format Extraction: TBD
