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
  return headers, data_rows, contest, metadata
  ```

  - `headers`: List of column headers
  - `data_rows`: List of row dicts or lists
  - `contest`: String describing the contest/race
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
- Return the same `(headers, data, contest, metadata)` tuple.
- Must extract metadata for state/county/race if possible for output directory routing.
- Return a tuple:

  ```python
  return headers, data_rows, contest, metadata

### Provided tables and skip_pivot

Format handlers (CSV/JSON/PDF) support a parity path that lets you supply pre-extracted table data and control whether the pivot-to-wide step runs. Pass these fields via `html_context`:

- `provided_tables`: list of `(headers, rows)` pairs, where `headers` is a list of column names and `rows` is a list of row dicts. Multiple tables will be merged and harmonized.
- `skip_pivot`: boolean; when true, the builder will not pivot to wide format and will emit the normalized table instead.

Example usage:

```python
html_context = {
  "state": "New York",
  "county": "New York",
  "contest": "District Attorney",
  "provided_tables": [
    (
      ["Precinct", "Candidate", "Votes"],
      [
        {"Precinct": "01-01", "Candidate": "Jane Doe", "Votes": 1234},
        {"Precinct": "01-01", "Candidate": "John Smith", "Votes": 1110},
      ],
    )
  ],
  "skip_pivot": True,
}

# CSV/JSON/PDF handlers will detect provided_tables and route through the
# unified builder and output pipeline accordingly.
headers, rows, contest, metadata = csv_handler.parse(None, html_context)
```

This path mirrors the HTML handler behavior and ensures that downstream schema events (normalized and wide) are emitted consistently. See Architecture docs for the Schema Events section.

## 🔁 Reusable Helpers (handlers/shared)

Place logic used across multiple states in `handlers/shared/`. For example:

- OCR clean-up
- Column normalizers
- Candidate name mappers
- Shared vendor templates (like Enhanced Voting)

These are imported into individual state handlers as needed.

The file `shared_logic.py` is where general shared election-logic for parsing or interpreting race types, aliases, and year detection should reside.

---

## 🎯 Real-Data Validation Snapshots (2025-11-07)

Recent full runs exposed schema behaviours that handler authors should respond to quickly:

- **JSON fast path (Rockland County, NY)** — `output/Orangetown_Town_Council__20251107_091920.csv`

  ```csv
  Precinct,Write-in - Total,CON Daniel W. Sullivan (Conservative) - Total,DEM Chrissy Knapp (Democratic) - Total,REP Daniel W. Sullivan (Republican) - Total,WOR Chrissy Knapp (Working Families) - Total,Grand Total
  Orangetown,1,26,605,199,25,856
  …
  All Precincts,13,1466,11981,12559,742,26761
  ```

  - Pivot + party/jurisdiction refactors now render stable wide output, but the party still lives inside the candidate label. Handlers that know party affiliations should populate a `Party` field per row before calling `build_dynamic_table` so the pivot can emit `<Candidate> - Party` columns.
  - Consider renaming the first column to the detected division type (for example `Town`) via the shared division helpers; Rockland County constants already define the correct mapping.

- **PDF contest selection (Hood River County, OR)** — `output/oregon__hood_river__US_Senator__20251107_082145.csv`

  ```csv
  Candidate,Party,Total Vote
  November,,2016
  ```

  - Contest detection succeeded but table extraction returned only boilerplate. Handlers should detect this condition and prompt the user to reselect or provide manual tables through `provided_tables` rather than emitting meaningless output.

- **PDF precinct aggregation (New York County DA)** — `output/new_york__new_york__Democratic_District_Attorney_New_York_2025__20251107_083504.csv`

  ```csv
  Precinct,Precinct Total Ballots,Precinct Total Applicable Ballots,…,Assembly District,Election District,…,Candidate,Votes
  AD 37 / Precinct 71 / Precinct 652,3543,3543,2891,…,37,,Patrick John Timmins,941
  ```

  - The composite “Precinct” cell blends assembly district, the precinct identifier, and stray affidavit/unrecorded totals. Implement a handler normalizer that splits these into discrete columns and ignores repeated totals per candidate row.
  - The repeated “New York County / New York State” strings should be fed into `context` so the centralized division heuristics can pick an appropriate jurisdiction header automatically.

**Immediate follow-up tasks**

1. Ensure handlers populate canonical `Party` values so party columns appear alongside each candidate in wide output.
2. Add PDF clean-up that separates assembly/election districts from precinct labels and pushes both into `context` and row data.
3. When extraction yields only boilerplate (for example “November 2016”), log a high-severity warning and abort instead of generating an empty CSV.

---

## 🧩 Extending Handlers

### Custom Noisy Labels/Patterns

- Pass `noisy_labels` and `noisy_label_patterns` to `select_contest()` for advanced contest filtering within your handler.

### User Prompts

- Always use `prompt_user_input()` for user interactions to ensure seamless CLI and future web UI integration.

### health Tasks

- Add automation or notification logic in `health/health_router.py`.

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

- The .... before a module `from` indicate the absolute folder structure relative to the
root directory of `webapp/parser/handlers/states/...` "4 dots in"

```python
from ....utils.table_builder import build_dynamic_table
from ....utils.user_prompt import prompt_user_input

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ....Context_Integration.context_coordinator import ContextCoordinator

def parse(page, html_context):
    # Optionally prompt user for contest if needed
    # contest = prompt_user_input("Select contest: ")
    headers, data = extract_table_data(page)
    contest = "Some Contest"
    metadata = {
        "state": html_context.get("state", "Unknown"),
        "county": html_context.get("county", "Unknown"),
        "race": contest
    }
    return headers, data, contest, metadata
```

## 🛡️ Election Integrity & Context

- Use context enrichment and ML/NLP validation (Context_Integration/) to improve extraction accuracy and integrity.
- All handler outputs are checked for anomalies and cross-field consistency.
- Corrections and feedback are logged and used to retrain extraction models and improve future runs.

---

## 📫 Questions?

See `CONTRIBUTING.md` or open a GitHub issue. Happy parsing!
