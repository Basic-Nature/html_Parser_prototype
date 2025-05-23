# Smart Elections Parser — Architecture Overview

This document provides a high-level overview of the architecture and responsibilities across modules in the Smart Elections parser repository.

---

## 🧱 Project Layers

### 1. **Entry Point**

- **`html_election_parser.py`**
  - Main orchestrator: delegates all specialized logic, never implements scraping/parsing directly.
  - Handles browser setup, CAPTCHA detection, user input (via `prompt_user_input`), and URL cycling.
  - Delegates parsing to state- or format-specific handlers.
  - Supports batch mode, multiprocessing, and bot/web integration.

### 2. **Router Layer**

- **`state_router.py`**
  - Matches URLs to a specific state handler in `handlers/`.
  - If no match is found, falls back to format detection.

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

- **`utils/browser_utils.py`**
  - Launches Playwright browser, supports headless/GUI, and user-agent spoofing.

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

- **`utils/format_router.py`**
  - Detects available formats and prompts user for selection (via `prompt_user_for_format`).

- **`utils/user_agents.py`**
  - List of randomized user agents for stealth navigation.

---

## 🤖 Bot & Web UI Integration

- **`bot/`**
  - Contains automation and notification logic (e.g., `bot_router.py`).
  - Can be triggered from the main pipeline if `ENABLE_BOT_TASKS=true` in `.env`.

- **Web UI Readiness**
  - All user prompts are modularized (`prompt_user_input`), allowing easy swap for a web interface.
  - Results and errors are logged and returned in structured formats for UI consumption.

---

## 📂 Data Flow Example

1. **User chooses URL** from `urls.txt` (prompted via `prompt_user_input`).
2. **Playwright browser** is launched via `browser_utils`.
3. **CAPTCHA page is detected**, `captcha_tools` attempts resolution.
4. HTML is scanned by `html_scanner` to gather:
   - Election year (e.g. 2022)
   - Race categories (e.g. Governor, Senate, Proposition)
   - County names (if present)
5. **Routing**:
   - If `state_router` detects a handler → delegate to `handlers/<state>.py`
   - Otherwise → delegate to `format_router`
6. The **handler parses and returns**: headers, data, contest_title, metadata.
7. **CSV is saved** in `output/<state>/<county>/<race>.csv`

---

## 📥 Input Directory (`input/`)

The `input/` folder is used for:

- Live downloads triggered from the parser pipeline (e.g., PDFs, JSONs).
- Manual file drops for override parsing (supported via `.env` and `process_format_override()`).

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
  Add to `bot/bot_router.py` and enable with `ENABLE_BOT_TASKS=true` in `.env`.
- **User prompts:**  
  Always use `prompt_user_input()` for future web UI compatibility.

---

## ✅ Future Enhancements

- More granular exception logging.
- Headless GUI fallback detection.
- Pluggable browser agent rotation.
- Shared election terminology models (e.g., race aliases).
- Plugin registry for state/community contributions.
- File-based ingestion workflow with filename inference for manual CSV/PDF drop-ins.
- Web UI for user prompts and batch management.

---

Contributions welcome! See `CONTRIBUTING.md` to get started.
