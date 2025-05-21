# docs/handlers.md

# Handler Development Guide for Smart Elections Parser

This document outlines how to develop and maintain **state-level** and **format-level** handlers inside the `handlers/` directory.

---

## 🗂 Directory Layout

```text
handlers/
├──states arizona.py              # State-specific handler
│   ├── arizona   
│   │    ├── arizona.py
│   │    └──county   
│   ├── pennsylvania
│   │    ├── pennsylvania.py
│   │    └── county
│   ├── new_york
│   │    ├──new_york.py
│   │    └──county
│   │       ├──rockland.py
│   │       └──[county.py]
│   ├── georgia             # State-specific handler
│   │    ├──georgia.py
│        └──county
├── formats/                # Format-based fallback logic
│   ├── csv_handler.py
│   ├── json_handler.py
│   ├── pdf_handler.py
│   └── html_handler.py
├── shared/                 # Reusable modules across handlers
└── shared_logic.py         # Common logic for interpreting elections (race/year/etc.)
```

---

## 📘 State Handlers

Each state handler should:

- Export a `parse(page, html_context)` function.
- Return a tuple:
  ```python
  return contest_title, headers, data_rows, metadata
  ```
- Optionally export `list_available_contests(page)` if the state site supports user contest selection.
- Pull `state`, `county`, and `race` metadata wherever possible.
- Set recommended output paths by providing structured metadata, i.e.:
  ```python
  metadata = {
    "state": "New York",
    "county": "Rockland",
    "race": "President"
  }
  ```

**Example: handlers/texas.py**
```python
from utils.table_utils import extract_table_data

def parse(page, html_context):
    contest_title = "Governor - General Election"
    headers, data = extract_table_data(page)
    return contest_title, headers, data, {
        "state": "Texas",
        "county": html_context.get("county"),
        "race_type": "General"
    }
```

---

## 📦 Format Handlers (Fallback)

Used when no `state_router` match is found.

- Must export `parse(file_path)`
- Return same `(title, headers, data, metadata)` tuple.
- Must be updated to extract metadata for state/county/race if possible for output directory routing.

**Example: handlers/formats/pdf_handler.py**
```python
def parse(file_path):
    from utils.pdf_utils import extract_pdf_tables
    tables = extract_pdf_tables(file_path)
    return "PDF Fallback", tables[0].columns, tables[0].rows, {
        "state": "Arizona",
        "race": "Senate"
    }
```

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

## ✅ Best Practices

- Prefer clarity over cleverness.
- Avoid hardcoded strings for races/candidates where possible.
- Always include all vote methods, even if count is 0.
- Ensure cross-precinct comparability (e.g., uniform column headers).
- Use `Pathlib`, not `os.path`.
- Use shared tools from `utils/` or `shared/` instead of duplicating logic.
- Return metadata so results can be stored in `output/<state>/<county>/<race>.csv`.

---

## 🧪 Testing a Handler

Use the main runner:
```bash
python html_election_parser.py
```
Select the target URL tied to your handler.

For format handlers, place the file in `input/` and trigger parsing using the prompt.

To test locally with pre-saved HTML files, adjust the `page.set_content()` step to load from disk.

---

## 📫 Questions?
See `CONTRIBUTING.md` or open a GitHub issue. Happy parsing!
