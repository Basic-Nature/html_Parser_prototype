# CONTRIBUTING.md

## Contributing to the Smart Elections Parser

We welcome contributions from developers, data analysts, civic technologists, and election transparency advocates!  
This project is designed to be scalable, readable, and resilient — please read below for how to help contribute meaningfully.

---
-Strategy going forward for mass handling of large datasets.
**Python + SQLAlchemy + PostgreSQL** and **C#/.NET + PostgreSQL** to maximize strengths, minimize weaknesses, and support scalable, high-performance batch election parsing and data warehousing:

---

## 1. Architectural Overview

- **PostgreSQL**: Central data warehouse for all parsed election data, metadata, and ML results.
- **Python (SQLAlchemy, FastAPI, ML stack)**: Handles HTML parsing, ML/NLP, rapid prototyping, and orchestration of batch jobs.
- **C#/.NET (Entity Framework Core, Dapper)**: Handles high-performance, parallel data ingestion, ETL, and analytics/reporting, especially for large-scale or Windows-centric deployments.

---

## 2. Division of Responsibilities

| Component                | Language/Stack                | Role/Strengths                                                                                  |
|--------------------------|-------------------------------|-------------------------------------------------------------------------------------------------|
| HTML Parsing, ML/NLP     | Python                        | Flexible, rapid dev, best for spaCy, transformers, and custom parsing logic                     |
| Batch Orchestration      | Python                        | Orchestrate batch jobs, manage queues, call C#/.NET for heavy ETL if needed                     |
| High-Performance ETL     | C#/.NET                       | Bulk data loading, parallel processing, data normalization, warehouse management                |
| Data Warehouse           | PostgreSQL                    | Central, normalized, scalable storage for all election data, accessible by both stacks          |
| API Layer                | Python (FastAPI) or C# (.NET WebAPI) | Expose data/services to UIs, dashboards, or external consumers                    |
| Analytics/Reporting      | C#/.NET or Python             | Use the best tool for the job: .NET for enterprise BI, Python for ad hoc analysis               |

---

## 3. Integration Points

- **Shared Database Schema**: Define a robust, version-controlled schema in PostgreSQL for all election data, results, and metadata.
- **Batch Processing**:
- Python parses HTML, extracts data, and writes to staging tables.
- C#/.NET services pick up batches from staging, perform high-speed ETL, normalization, and load into warehouse tables.
- **Parallelization**:
- Use Python’s multiprocessing for moderate parallelism (e.g., 10–50 concurrent jobs).
- For massive scale (100s–1000s of jobs), use C#/.NET for orchestrating and running parallel ETL, leveraging .NET’s async and threading strengths.
- **API/Service Layer**:
- Expose endpoints for triggering batch jobs, querying results, and monitoring status.
- Use FastAPI (Python) for ML/LM endpoints; use .NET WebAPI for enterprise integration if needed.

---

## 4. Strengths & Weaknesses

- **Python**: Flexible, great for ML and rapid iteration; less ideal for massive, concurrent, CPU-bound ETL.
- **C#/.NET**: High-throughput, strongly-typed, parallel ETL and analytics; more verbose, less flexible for ML/NLP.
- **PostgreSQL**: True data warehouse—partitioned tables, indexes, and analytics support.

---

## 5. Sample Workflow

1. **Python** parses thousands of county/state HTMLs, extracts raw results, and writes to `staging_election_results` in PostgreSQL.
2. **C#/.NET** service (triggered on schedule or by API) reads from staging, performs validation, normalization, and loads into `warehouse_election_results`.
3. **Python** ML jobs (e.g., anomaly detection, NER) run on warehouse data and write results back to PostgreSQL.
4. **APIs** (Python or .NET) expose data for dashboards, reporting, or further analysis.

---

## 6. Best Practices

- **Schema Management**: Use Alembic (Python) and EF Core Migrations (.NET) to keep schema in sync.
- **Data Contracts**: Define clear data models and document them for both stacks.
- **Batch IDs/Metadata**: Tag all data with batch IDs, source, and processing status for traceability.
- **Monitoring**: Use logging and monitoring in both stacks to track job status and performance.
- **Testing**: Integration tests to ensure both Python and .NET can read/write the same data correctly.

---

## 7. Scalability & Performance

- For moderate batch sizes, Python multiprocessing is sufficient.
- For very large-scale, use C#/.NET for ETL and parallelization, possibly with a job queue (e.g., RabbitMQ, Celery, or Hangfire for .NET).
- Use PostgreSQL features (partitioning, indexing, materialized views) to optimize warehouse queries.

---

## 8. Summary Table

| Task/Component         | Python         | C#/.NET         | PostgreSQL         |
|------------------------|---------------|-----------------|--------------------|
| HTML/ML Parsing        | ✔️             |                 |                    |
| ML/NER/AI              | ✔️             |                 |                    |
| Batch Orchestration    | ✔️ (small/med) | ✔️ (large)      |                    |
| High-Perf ETL          |               | ✔️              |                    |
| Data Warehouse         |               |                 | ✔️                 |
| API Layer              | ✔️/✔️           | ✔️/✔️            |                    |
| Analytics/Reporting    | ✔️/✔️           | ✔️/✔️            |                    |

---

**Next Steps:**

- Define your PostgreSQL schema and data contracts.
- Build your Python batch/ML pipeline and API.
- Build a C#/.NET ETL/analytics service for high-throughput needs.
- Use the database as the integration point.

*Let us know if you want a sample schema, API template, or batch orchestration code for either stack!*

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
  return headers, data_rows, contest, metadata
  ```

  - `headers`: List of column headers
  - `data_rows`: List of row dicts or lists
  - `contest`: String describing the contest/race
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
    contest = "Some Contest"
    metadata = {
        "state": html_context.get("state", "Unknown"),
        "county": html_context.get("county", "Unknown"),
        "race": contest
    }
    return headers, data, contest, metadata
```

---

### 🧩 How to Add a Format Handler

- Add a new file in `handlers/formats/` (e.g., `csv_handler.py`, `pdf_handler.py`).
- Export a `parse(page, html_context)` or `parse(file_path, html_context)` function.
- Return the same `(headers, data, contest, metadata)` tuple.
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
