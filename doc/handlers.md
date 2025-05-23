# Handler Development Guide for Smart Elections Parser

This document outlines how to develop and maintain **state-level** and **format-level** handlers inside the `handlers/` directory.

---

## 🗂 Directory Layout

``text
handlers/
├── states/
│   ├── arizona/
│   │   ├── arizona.py
│   │   └── county/
│   ├── pennsylvania/
│   │   ├── pennsylvania.py
│   │   └── county/
│   ├── new_york/
│   │   ├── new_york.py
│   │   └── county/
│   │       ├── rockland.py
│   │       └── [county.py]
│   ├── georgia/
│   │   ├── georgia.py
│   │   └── county/
├── formats/                # Format-based fallback logic
│   ├── csv_handler.py
│   ├── json_handler.py
│   ├── pdf_handler.py
│   └── html_handler.py
├── shared/                 # Reusable modules across handlers
└── shared_logic.py         # Common logic for interpreting elections (race/year/etc.)
``

---

## 📘 State Handlers

Each state handler **must**:

- Export a `parse(page, html_context)` function.
- Return a tuple:
  ``python
  return headers, data_rows, contest_title, metadata
  ``
  - `headers`: List of column headers
  - `data_rows`: List of row dicts or lists
  - `contest_title`: String describing the contest/race
  - `metadata`: Dict with at least `state`, `county`, and `race` (if available)

- Optionally export `list_available_contests(page)` if the state site supports user contest selection.
- Pull `state`, `county`, and `race` metadata wherever possible.
- Set recommended output paths by providing structured metadata, e.g.:
  ``python
  metadata = {
    "state": "New York",
    "county": "Rockland",
    "race": "President"
  }
  ``

**Example: handlers/states/texas/texas.py**
``python
from utils.table_utils import extract_table_data

def parse(page, html_context):
    contest_title = "Governor - General Election"
    headers, data = extract_table_data(page)
    return headers, data, contest_title, {
        "state": "Texas",
        "county": html_context.get("county"),
        "race_type": "General"
    }
``

---

## 📦 Format Handlers (Fallback)

Used when no `state_router` match is found.

- Must export `parse(page, html_context)` or `parse(file_path, html_context)` depending on context.
- Return the same `(headers, data, contest_title, metadata)` tuple.
- Must extract metadata for state/county/race if possible for output directory routing.

**Example: handlers/formats/pdf_handler.py**
``python
def parse(page, html_context):
    from utils.pdf_utils import extract_pdf_tables
    tables = extract_pdf_tables(html_context["manual_file"])
    return tables[0].columns, tables[0].rows, "PDF Fallback", {
        "state": "Arizona",
        "race": "Senate"
    }
``

---

## 🔁 Reusable Helpers (handlers/shared)

Place logic used across multiple states in `handlers/shared/`. For example:

- OCR clean-up
- Column normalizers
- Candidate name mappers
- Shared vendor templates (like Enhanced Voting)

These are imported into individual state handlers as needed.

The file `shared_logic.py` is where general shared election-logic for parsing or interpreting race types, aliases, and year detection should reside.

---

## 🧩 Extending Handlers

- **Custom Noisy Labels/Patterns:**  
  Pass `noisy_labels` and `noisy_label_patterns` to `select_contest()` for contest filtering in your handler.
- **User Prompts:**  
  Use `prompt_user_input()` for all user input to allow easy CLI/web UI integration later.
- **Bot Tasks:**  
  Add automation/notification logic in `bot/bot_router.py` and enable with `ENABLE_BOT_TASKS=true` in `.env`.

---

## ✅ Best Practices

- Prefer clarity over cleverness.
- Avoid hardcoded strings for races/candidates where possible.
- Always include all vote methods, even if count is 0.
- Ensure cross-precinct comparability (e.g., uniform column headers).
- Use `Pathlib`, not `os.path`.
- Use shared tools from `utils/` or `shared/` instead of duplicating logic.
- Return metadata so results can be stored in `output/<state>/<county>/<race>.csv`.
- Modularize any user prompts using `prompt_user_input()` for future web UI compatibility.
- Document any handler-specific configuration at the top of your handler file.

---

## 🧪 Testing a Handler

Use the main runner:
``bash
python html_election_parser.py
``
Select the target URL tied to your handler.

For format handlers, place the file in `input/` and trigger parsing using the prompt.

To test locally with pre-saved HTML files, adjust the `page.set_content()` step to load from disk.

---

## 🧑‍💻 Example Handler Template

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

## 📫 Questions?

See `CONTRIBUTING.md` or open a GitHub issue. Happy parsing!
