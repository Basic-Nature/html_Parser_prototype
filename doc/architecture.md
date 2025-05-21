# docs/architecture.md

# Smart Elections Parser — Architecture Overview

This document provides a high-level overview of the architecture and responsibilities across modules in the Smart Elections parser repository.

---

## 🧱 Project Layers

### 1. **Entry Point**
- **`html_election_parser.py`**
  - Main control flow of the scraper.
  - Handles browser setup, CAPTCHA detection, user input, and URL cycling.
  - Delegates parsing to state- or format-specific handlers.

### 2. **Router Layer**
- **`state_router.py`**
  - Matches URLs to a specific state handler in `handlers/`.
  - If no match is found, falls back to format detection.

- **`format_router.py`** (in `utils/`)
  - Uses `html_scanner.py` and link metadata to detect HTML, PDF, JSON, or CSV.
  - Dispatches to a format handler.

### 3. **Handlers**
- **`handlers/states`**
  - Contains one file per U.S. state (e.g., `arizona.py`, `new_york.py`).
    - Each state script should export a `parse(page, html_context)` method.
  - **`handlers/states/county`**
  - Each state script should export a `parse(page, html_context)` method.

- **`handlers/formats/`**
  - Generic format parsers: `pdf_handler.py`, `json_handler.py`, `csv_handler.py`, `html_handler.py`.
  - Used when no specialized state handler exists.


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

- **`utils/user_agents.py`**
  - List of randomized user agents for stealth navigation.

---

## 📂 Data Flow Example

1. **User chooses URL** from `urls.txt`.
2. **Playwright browser** is launched via `browser_utils`.
3. **CAPTCHA page is detected**, `captcha_tools` attempts resolution.
4. HTML is scanned by `html_scanner` to gather:
   - Election year (e.g. 2022)
   - Race categories (e.g. Governor, Senate, Proposition)
   - County names (if present)
5. **Routing**:
   - If `state_router` detects a handler → delegate to `handlers/<state>.py`
   - Otherwise → delegate to `format_router`
6. The **handler parses and returns**: contest_title, headers, row data.
7. **CSV is saved** in `output/<state>/<county>/<race>.csv`

---

## 📥 Input Directory (`input/`)

The `input/` folder is **primarily used for live downloads** triggered from the parser pipeline (e.g., downloaded PDFs or JSON files).

- Files are automatically placed here by `download_utils.py` when detected on a site.
- The parser then reads from `input/` to extract the relevant structured data.

⚠️ **Manual file drops into `input/` are not guaranteed to work out-of-the-box.**
To support this mode, you would need:
- A consistent naming convention (`<state>_<county>_<race>.<ext>`)
- A dedicated manual input loader (potential future enhancement)

---

## 🛠️ Extensibility Guidelines

- To support a new state: Add `handlers/<state>.py` and register in `state_router.py`.
- To support a new file format: Add to `handlers/formats/` and map in `format_router.py`.
- To override CAPTCHA or stealth logic: modify `captcha_tools.py` or `browser_utils.py`.

---

## ✅ Future Enhancements
- More granular exception logging.
- Headless GUI fallback detection.
- Pluggable browser agent rotation.
- Shared election terminology models (e.g., race aliases).
- Plugin registry for state/community contributions.
- File-based ingestion workflow with filename inference for manual CSV/PDF drop-ins.

---

Contributions welcome! See `CONTRIBUTING.md` to get started.
