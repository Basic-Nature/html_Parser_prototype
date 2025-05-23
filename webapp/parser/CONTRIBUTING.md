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
- Add **bot tasks** in `bot/bot_router.py` for automation or notifications.
- Improve **shared utilities** in `utils/` or `handlers/shared/`.

---

### 🛠️ Dev Setup

1. Clone the repository:
   ``bash
   git clone https://github.com/SmartElections/parser.git
   cd parser
   ``
2. Install dependencies:
   ``bash
   pip install -r requirements.txt
   ``
3. Create your `.env` file:
   ``bash
   cp .env.template .env

Then edit `.env` as needed for HEADLESS mode, CAPTCHA_TIMEOUT, etc.

``

---

### 🧪 Running the Parser

``bash
python html_election_parser.py
``
You’ll be prompted to select from `urls.txt`, then walk through format/state handler detection, CAPTCHA solving, and CSV extraction.

---

### 🧭 How to Add a State or County Handler

- Add a new file in `handlers/states/<state>.py` or `handlers/states/<state>/county/<county>.py`.
- **Required:** Export a `parse(page, html_context)` function that returns:
  ``python
  return headers, data_rows, contest_title, metadata
  ``
  - `headers`: List of column headers
  - `data_rows`: List of row dicts or lists
  - `contest_title`: String describing the contest/race
  - `metadata`: Dict with at least `state`, `county`, and `race` (if available)

- **Optional:** Export `list_available_contests(page)` if the site supports user contest selection.
- **Always:** Use `prompt_user_input()` for any user prompts (import from `utils.user_prompt`).
- **Register** your handler in `state_router.py` for automatic routing.

**Example:**
``python
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
``

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

### 📦 Folder Structure (Quick Glance)

- `handlers/`: State and format-specific scrapers.
- `utils/`: Shared browser, captcha, and format logic.
- `input/`: Input files like PDFs or JSONs.
- `output/`: Where CSVs go.
- `urls.txt`: List of URLs to cycle.
- `.env`: Controls mode, timeouts, etc.

---

### 💡 Tips for Effective Contributions

- Test your handler with real and edge-case data.
- Use the troubleshooting guide (`docs/troubleshooting.md`) if you get stuck.
- Check logs for errors and tuple structure issues.
- Use `CACHE_RESET=true` in `.env` to clear processed URL cache if needed.
- For bot/automation, see `bot/bot_router.py` and enable with `ENABLE_BOT_TASKS=true` in `.env`.

---

### 💬 Questions?

File an issue or start a discussion. We're happy to walk you through a contribution!

Thanks for helping improve election transparency! 🗳️
