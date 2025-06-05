# CONTRIBUTING.md

## Contributing to the Smart Elections Parser

We welcome contributions from developers, data analysts, civic technologists, and election transparency advocates!  
This project is designed to be scalable, readable, and resilient — please read below for how to help contribute meaningfully.

---

### 🧠 What You Can Help With

- Add or update a **state or county handler** in `handlers/states/` or `handlers/states/<state>/county/`.
- Improve or add **format handlers** under `handlers/formats/` (CSV, JSON, PDF, HTML).
- Contribute **test URLs** for election sites in `urls.txt`.
- Expand **race/year/contest detection** logic in `utils/html_scanner.py`.
- Optimize **CAPTCHA resilience** in `utils/captcha_tools.py`.
- Strengthen **modularity, orchestration, and UX** in `html_election_parser.py`.
- Add **bot tasks** in `bots/bot_router.py` for automation, correction, or notifications.
- Improve **shared utilities** in `utils/` or `handlers/shared/`.
- Enhance or document the **Web UI** (Flask app in `webapp/`) for a better user experience, especially for new coders or non-technical users.
- **Expand the context library**: Add new context patterns, feedback, or corrections in `context_library.json` or contribute to `Context_Integration/context_organizer.py`.
- **Improve ML/NLP extraction or entity recognition**: See `ml_table_detector.py` and `spacy_utils.py`.
- **Use or extend the correction bot**: See `bots/manual_correction_bot.py` and retraining scripts.
- **Tune dynamic table extraction**: Add or improve extraction strategies, scoring, or patching logic in `utils/table_core.py` and `utils/dynamic_table_extractor.py`.
- All corrections and feedback are logged for auditability and future learning.

---

### 🧠 Improving Context & Correction

- To add new context patterns or feedback, edit `context_library.json` or contribute to `Context_Integration/context_organizer.py`.
- To improve ML/NLP extraction or entity recognition, see `utils/ml_table_detector.py` and `utils/spacy_utils.py`.
- To use or extend the correction bot, see `bots/manual_correction_bot.py` and retraining scripts.
- All corrections and feedback are logged for auditability and future learning.

---

### 🤖 Adding Bots

- Place new bot scripts in `bots/` and register them in `bots/bot_router.py`.
- Bots can automate corrections, retraining, notifications, or data integrity checks.
- Enable bots via `.env` with `ENABLE_BOT_TASKS=true`.
- See `bots/manual_correction_bot.py` for an example of a correction/retraining bot.

---

### 🧩 Dynamic Table Extraction & Scoring

- Extraction is now multi-strategy and uses scoring/patching.
- To add or tune extraction strategies, edit `utils/table_core.py` or `utils/dynamic_table_extractor.py`.
- To expand the keyword libraries for locations, percent, etc., edit the keyword sets at the top of `table_core.py`.
- To contribute new scoring or patching logic, see the `extract_all_tables_with_location` function in `table_core.py`.

---

### 🧭 Handler Registration & Shared Utilities

- Handlers are modular and can delegate to shared/context logic.
- Use shared utilities and context-aware orchestration in new handlers.
- Register handlers for new states, counties, or formats in `state_router.py` or `utils/format_router.py`.

---

### 🛡️ Election Integrity & Auditability

- All outputs are auditable: logs, metadata, and correction trails are saved.
- To contribute to or extend integrity checks, see `Context_Integration/Integrity_check.py`.
- Ensure your handler or utility logs key decisions and supports auditability.

---

### 🛠️ Dev Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/SmartElections/parser.git
   cd parser
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create your `.env` file:

   ```bash
   cp .env.template .env
   ```

   Then edit `.env` as needed for HEADLESS mode, CAPTCHA_TIMEOUT, etc.

---

### 🧪 Running the Parser

**CLI (Recommended for advanced users):**

```bash
python -m webapp.parser.html_election_parser
```

You’ll be prompted to select from `urls.txt`, then walk through format/state handler detection, CAPTCHA solving, and CSV extraction.

**Web UI (Optional, recommended for new users or those who prefer a graphical interface):**

```bash
python webapp/Smart_Elections_Parser_Webapp.py
```

- Open your browser to [http://localhost:5000](http://localhost:5000) or the link printed in terminal (often the printed IP Address).
- The Web UI provides a dashboard, URL hint manager, change history, and a "Run Parser" page with real-time output.
- This is ideal for teams, researchers, and those learning to code—no Python experience required to use the main features!

---

### 🧭 How to Add a State or County Handler

- Add a new file in `handlers/states/<state>.py` or `handlers/states/<state>/county/<county>.py`.
- **Required:** Export a `parse(page, html_context)` function that returns:

  ```python
  return headers, data_rows, contest_title, metadata
  ```

  - `headers`: List of column headers
  - `data_rows`: List of row dicts or lists
  - `contest_title`: String describing the contest/race
  - `metadata`: Dict with at least `state`, `county`, and `race` (if available)

- **Optional:** Export `list_available_contests(page)` if the site supports user contest selection.
- **Always:** Use `prompt_user_input()` for any user prompts (import from `utils.user_prompt`).
- **Register** your handler in `state_router.py` for automatic routing.

**Example:**

```python
from utils.table_utils import extract_table_data
from utils.user_prompt import prompt_user_input

def parse(page, html_context):
    # Optionally prompt user for contest if needed
    # contest = prompt_user_input("Select contest: ")
    headers, data = extract_table_data(page)
    contest_title = "Some Contest"
    metadata = {
        "state": html_context.get("state", "Unknown"),
        "county": html_context.get("county", "Unknown"),
        "race": contest_title
    }
    return headers, data, contest_title, metadata
```

---

### 🧩 How to Add a Format Handler

- Add a new file in `handlers/formats/` (e.g., `csv_handler.py`, `pdf_handler.py`).
- Export a `parse(page, html_context)` or `parse(file_path, html_context)` function.
- Return the same `(headers, data, contest_title, metadata)` tuple.
- Register your handler in `utils/format_router.py`.

---

### 🧼 Coding Standards & Best Practices

- **Clarity over cleverness:** Write code that’s easy to read and maintain.
- **No hardcoded race/candidate strings:** Use shared logic or config where possible.
- **Always include all vote methods:** Even if count is 0, for comparability.
- **Uniform headers:** Use `utils.table_utils.normalize_headers()` for consistency.
- **Use `Pathlib`:** Prefer over `os.path` for file operations.
- **Logging:** Use the `logging` module, not `print`, for all output except user prompts.
- **User prompts:** Always use `prompt_user_input()` for CLI/web UI compatibility.
- **Docstrings and comments:** Document all functions and tricky logic.
- **Test in both headless and GUI modes:** Ensure browser automation works in both.
- **Return metadata:** Always return enough metadata for output routing (`output/<state>/<county>/<race>.csv`).
- **Reuse utilities:** Use tools from `utils/` or `handlers/shared/` instead of duplicating logic.
- **Document handler-specific config:** At the top of your handler file.

---

### 🖥️ Web UI Contributions

- The Web UI (in `webapp/`) is **optional** but highly valuable for users who prefer a graphical interface or are new to coding.
- You can contribute by:
  - Improving the dashboard, forms, or real-time output display.
  - Adding new features (e.g., search, filtering, user authentication).
  - Enhancing accessibility and documentation for non-technical users.
  - Writing clear instructions and tooltips to help new users understand each feature.
- The Web UI is designed to make the parser accessible to everyone, regardless of coding experience.

---

### 📂 Folder Structure (Quick Glance)

- `handlers/`: State and format-specific scrapers.
- `utils/`: Shared browser, captcha, and format logic.
- `bots/`: Correction/retraining/automation bots.
- `Context_Integration/`: Context, ML/NLP, and integrity modules.
- `input/`: Input files like PDFs or JSONs.
- `output/`: Where CSVs go.
- `urls.txt`: List of URLs to cycle.
- `.env`: Controls mode, timeouts, etc.
- `context_library.json`: Persistent context/feedback.
- `webapp/`: Flask-based Web UI (optional).

---

### 💡 Tips for Effective Contributions

- Test your handler with real and edge-case data.
- Use the troubleshooting guide (`docs/troubleshooting.md`) if you get stuck.
- Check logs for errors and tuple structure issues.
- Use `CACHE_RESET=true` in `.env` to clear processed URL cache if needed.
- For bot/automation, see `bots/bot_router.py` and enable with `ENABLE_BOT_TASKS=true` in `.env`.
- If contributing to the Web UI, test both CLI and web workflows to ensure compatibility.
- When contributing to context or correction, ensure your changes are logged and auditable.

---

### 💬 Questions?

File an issue or start a discussion. We're happy to walk you through a contribution!

Thanks for helping improve election transparency! 🗳️
