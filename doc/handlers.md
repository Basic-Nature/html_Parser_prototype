# Handler Development Guide for Smart Elections Parser

This document outlines how to develop and maintain **state-level** and **format-level** handlers inside the `handlers/` directory.

---

## 🗂 Directory Layout

```text
handlers/
├── states/                # State-specific handlers
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
│   ├── example_state/
│   │   ├── example_state.py
│   │   └── example_county/
│   │       └── example_county.py
│   └── ...
├── formats/               # Format-based fallback handlers
│   ├── csv_handler.py
│   ├── json_handler.py
│   ├── pdf_handler.py
│   └── html_handler.py
├── utils/                # Reusable modules across handlers
└── shared_logic.py        # Common logic for interpreting elections (race/year/etc.)
```

---

## 📘 State Handlers

Each state handler **must**:

- Export a `parse(page, html_context)` function.
- Return a tuple:

  ```python
  return headers, data_rows, contest_title, metadata
  ```

  - `headers`: List of column headers
  - `data_rows`: List of row dicts or lists
  - `contest_title`: String describing the contest/race
  - `metadata`: Dict with at least `state`, `county`, and `race` (if available)

- Optionally export `list_available_contests(page)` if the state site supports user contest selection.
- Pull `state`, `county`, and `race` metadata wherever possible.
- Set recommended output paths by providing structured metadata, e.g.:

  ```python
  metadata = {
    "state": "New York",
    "county": "Rockland",
    "race": "President"
  }
  ```

---

## 📦 Format Handlers (Fallback)

Used when no `state_router` match is found.

- Must export `parse(page, html_context)` or `parse(file_path, html_context)` depending on context.
- Return the same `(headers, data, contest_title, metadata)` tuple.
- Must extract metadata for state/county/race if possible for output directory routing.
- Return a tuple:

  ```python
  return headers, data_rows, contest_title, metadata

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

### Custom Noisy Labels/Patterns

- Pass `noisy_labels` and `noisy_label_patterns` to `select_contest()` for advanced contest filtering within your handler.

### User Prompts

- Always use `prompt_user_input()` for user interactions to ensure seamless CLI and future web UI integration.

### Bot Tasks

- Add automation or notification logic in `bots/bot_router.py`.
- Enable bot tasks by setting `ENABLE_BOT_TASKS=true` in your `.env` file.

### Context-Aware Extraction

- Enhance extraction and validation by leveraging context enrichment and ML/NLP features.
- Use `context_coordinator.py` and `context_organizer.py` for smarter, context-driven data extraction.

### Dynamic Table Extraction

- For robust, multi-strategy table extraction, scoring, and patching, utilize `table_core.py` and `dynamic_table_extractor.py`.

---

## ✅ Best Practices

### Best Practices

- **Clarity First:**  
  Prefer clear, readable code over clever or obscure solutions.

- **Avoid Hardcoding:**  
  Do not hardcode race or candidate names; extract dynamically whenever possible.

- **Comprehensive Vote Methods:**  
  Always include all vote methods in your output, even if their count is zero.

- **Uniformity Across Precincts:**  
  Ensure consistent column headers and data formats for cross-precinct comparability.

- **Path Handling:**  
  Use `pathlib` for file and directory operations instead of `os.path`.

- **Reuse Shared Tools:**  
  Import utilities from `utils/` or `shared/` rather than duplicating logic.

- **Return Metadata:**  
  Always return metadata so results can be saved as `output/<state>/<county>/<race>.csv`.

- **Modular User Prompts:**  
  Use `prompt_user_input()` for all user interactions to support future web UI integration.

- **Document Configuration:**  
  Add handler-specific configuration details at the top of your handler file.

- **Audit Logging:**  
  Log key decisions and extraction steps using `shared_logger.py` for traceability.

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

```python
from utils.table_builder import build_dynamic_table
from utils.user_prompt import prompt_user_input

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .....Context_Integration.context_coordinator import ContextCoordinator
import numpy as e

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

## 🛡️ Election Integrity & Context

- Use context enrichment and ML/NLP validation (Context_Integration/) to improve extraction accuracy and integrity.
- All handler outputs are checked for anomalies and cross-field consistency.
- Corrections and feedback are logged and used to retrain extraction models and improve future runs.

---

## 📫 Questions?

See `CONTRIBUTING.md` or open a GitHub issue. Happy parsing!
