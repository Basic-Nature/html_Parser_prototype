# Smart Elections Parser

## Overview

A highly adaptable and modular precinct-level election result scraper. Designed to support structured parsing across various formats and state websites, it incorporates dynamic CAPTCHA handling, contest/race discovery, and file download capabilities.

---

🧭 **Design Philosophy**

- `html_election_parser.py` orchestrates the pipeline: it only delegates, never implements scraping/parsing logic.
- All specialized logic (browser, CAPTCHA, download, contest selection, format detection, etc.) is delegated to dedicated modules in `utils/` or `handlers/`.
- `format_router.py` takes over only if a downloadable file is explicitly confirmed by the user or automation.
- Each state/county handler (e.g., `pennsylvania.py`) should:
  - Dynamically interact with the current page
  - Parse content before assuming file-based fallback
  - Respect `html_scanner.py` flags but remain flexible
- All user prompts use `prompt_user_input()` for easy CLI/web UI swapping.
- `/bot` folder is reserved for automation and notification tasks, callable from the main pipeline.

---

## 🔧 Features

- **Headless or GUI Mode**: Browser launches headlessly by default unless CAPTCHA triggers a human interaction.
- **CAPTCHA-Resilient**: Dynamically detects and pauses for Cloudflare verification with a visible browser.
- **Race-Year Detection**: Scans HTML to find available election years and contests.
- **State-Aware Routing**: Automatically detects state context and delegates to appropriate handler module.
- **Format-Aware Fallback**: Supports CSV, JSON, PDF, and HTML formats with pluggable handlers.
- **Output Sorting**: Results saved in nested folders by state, county, and race.
- **URL Selection**: Loads URLs from `urls.txt` and lets users select specific targets.
- **.env Driven**: Easily override behavior such as CAPTCHA timeouts or headless preferences.
- **Bot Integration**: Enable `/bot` tasks (e.g., notifications, batch scans) via `.env`.
- **Web UI Ready**: All user prompts are modular for future web interface integration.

---

## How to Add a New State/County Handler, Format, or Bot Task

1. **State/County Handler:**  
   - Create a new handler in `handlers/states/` or `handlers/counties/`.
   - Implement a `parse(page, html_context)` function.
   - Register your handler in `state_router.py`.

2. **Bot Task:**  
   - Add your bot logic to `bot/bot_router.py`.
   - Implement a `run_bot_task(task_name, context)` function.
   - Enable with `ENABLE_BOT_TASKS=true` in `.env`.

3. **Custom Noisy Labels/Patterns:**  
   - In your handler, pass `noisy_labels` and `noisy_label_patterns` to `select_contest()` for contest filtering.

4. **Format Handler:**  
   - Add your handler to `utils/format_router.py` and register it in `route_format_handler`.

5. **User Prompts:**  
   - Use `prompt_user_input()` for all user input to allow easy web UI integration later.  
   - Example:

     ```python
     from utils.user_prompt import prompt_user_input
     url = prompt_user_input("Enter URL: ")
     ```

---

## 🗂 Folder Structure

``
html_Parser_prototype/
├── handlers/
│   ├── formats/
│   │   ├── csv_handler.py
│   │   ├── html_handler.py
│   │   ├── json_handler.py
│   │   └── pdf_handler.py
│   └── states/
│       └── county/
│
├── utils/
│   ├── browser_utils.py
│   ├── captcha_tools.py
│   ├── contest_selector.py
│   ├── download_utils.py
│   ├── format_router.py
│   ├── html_scanner.py
│   ├── output_utils.py
│   ├── shared_logger.py
│   ├── shared_logic.py
│   ├── user_agents.py
│   └── user_prompt.py
│
├── bot/
│   └── bot_router.py
│
├── state_router.py
├── html_election_parser.py
├── url_hint_overrides.txt
├── urls.txt
├── .env
├── .env.template
├── .gitignore
├── requirements.txt
├── seleniumbase_launcher.py
``

---

## 🧪 How to Use

1. **Install Requirements**

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Settings (Optional)**
   Copy `.env.template` to `.env` and modify:

   ```bash
   cp .env.template .env
   ```

   Variables include:

   - `HEADLESS=false` (for debugging)
   - `CAPTCHA_TIMEOUT=300`
   - `CACHE_PROCESSED=true`
   - `ENABLE_BOT_TASKS=true` (to enable bot features)
   - `ENABLE_PARALLEL=true` (to enable multiprocessing)

3. **Add URLs**
   - Populate `urls.txt` with target election result URLs.
   - `url_hint_overrides.txt` is used in conjunction with `state_router.py` when dynamic state detection fails.

4. **Run Parser**

   ```bash
   python html_election_parser.py
   ```

---

## 📦 Output Format

Parsed CSV files are saved as:

``
output/{state}/{county}/{race}/{contest}_results.csv
``

Example:

``
output/arizona/maricopa/us_senate/kari_lake_results.csv
``

---

## 🧩 Extending the Parser

- **Add New States**: Create a new file in `handlers/states/` (e.g. `georgia.py`) and implement a `parse()` method.
- **Add Format Support**: Add new file in `handlers/formats/` and map in `format_router.py`.
- **Shared Behavior**: Use `utils/shared_logic.py` for common race detection, total extraction, etc.
- **Add Bot Tasks**: Add new automation/notification logic in `bot/bot_router.py`.

---

## 🔐 Notes on Security

- All scraping runs headlessly unless CAPTCHA is triggered.
- `.env` is excluded from version control via `.gitignore`.
- No credentials or session tokens are stored.

---

## 🚧 Roadmap

- Multi-race selection prompt
- Retry logic on failed URLs
- Browser fingerprint obfuscation
- Contributor upload queue (for handler patches)
- YAML config option for handler metadata
- Web UI for user prompts and batch management

---

## 📄 License

MIT License (TBD)

---

## 🙋‍♀️ Contributors

- Lead Dev: [Juancarlos Barragan]
- Elections Research: TBD
- PDF Table Extraction: TBD
