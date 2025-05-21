# CONTRIBUTING.md

## Contributing to the Smart Elections Parser

We welcome contributions from developers, data analysts, civic technologists, and election transparency advocates! This project is designed to be scalable, readable, and resilient — please read below for how to help contribute meaningfully.

---

### 🧠 What You Can Help With
- Add or update a **state handler** in `handlers/` for a state you know.
- Improve **format handlers** under `handlers/formats/` (CSV, JSON, PDF, HTML).
- Contribute **test URLs** for election sites in `urls.txt`.
- Expand **race/year detection** logic in `utils/html_scanner.py`.
- Optimize **CAPTCHA resilience** in `utils/captcha_tools.py`.
- Strengthen **modularity and UX** in `html_election_parser.py`.

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
   # Then edit .env as needed for HEADLESS mode, CAPTCHA_TIMEOUT, etc.
   ```

---

### 🧪 Running the Parser
```bash
python html_election_parser.py
```
You’ll be prompted to select from `urls.txt`, then walk through format/state handler detection, CAPTCHA solving, and CSV extraction.

---

### 🧼 Code Standards
- Follow the format and logic in existing state and format handlers.
- Use `logging`, not just print statements.
- Prefer `Pathlib` over `os.path`.
- Include docstrings and inline comments.
- Test browser automation in both **headless** and **GUI** modes.

---

### 📦 Folder Structure (Quick Glance)
- `handlers/`: State and format-specific scrapers.
- `utils/`: Shared browser, captcha, and format logic.
- `input/`: Input files like PDFs or JSONs.
- `output/`: Where CSVs go.
- `urls.txt`: List of URLs to cycle.
- `.env`: Controls mode, timeouts, etc.

---

### 🧭 How to Add a State Handler
```bash
# Add to handlers/<state>.py
# Must export a `parse(page, html_context)` method
```
Example:
```python
def parse(page, html_context):
    # Extract race, county, and CSV rows from the page
    return contest_title, headers, data, metadata
```

---

### 💬 Questions?
File an issue or start a discussion. We're happy to walk you through a contribution!

Thanks for helping improve election transparency! 🗳️
