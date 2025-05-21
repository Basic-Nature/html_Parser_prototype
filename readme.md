# Smart Elections Parser

## Overview

A highly adaptable and modular precinct-level election result scraper. Designed to support structured parsing across various formats and state websites, it incorporates dynamic CAPTCHA handling, contest/race discovery, and file download capabilities.

---

🧭 Design Philosophy
- html_election_parser.py should be focused on live, unofficial results; although, it will handle format detection and downloading various file extension types from a webpage to obtain official results to parse through.

- format_router.py should only take over if a downloadable file is explicitly confirmed.

- Each state handler (like pennsylvania.py) should:

   - Dynamically interact with the current page

   - Parse content on the page before assuming file-based fallback

   - Respect html_scanner.py flags but remain flexible



## 🔧 Features

* **Headless or GUI Mode**: Browser launches headlessly by default unless CAPTCHA triggers a human interaction.
* **CAPTCHA-Resilient**: Dynamically detects and pauses for Cloudflare verification with a visible browser.
* **Race-Year Detection**: Scans HTML to find available election years and contests.
* **State-Aware Routing**: Automatically detects state context and delegates to appropriate handler module.
* **Format-Aware Fallback**: Supports CSV, JSON, PDF, and HTML formats with pluggable handlers.
* **Output Sorting**: Results saved in nested folders by state, county, and race.
* **URL Selection**: Loads URLs from `urls.txt` and lets users select specific targets.
* **.env Driven**: Easily override behavior such as CAPTCHA timeouts or headless preferences.

---

## 🗂 Folder Structure

```
html_Parser_prototype/
├── handlers/
│   ├── formats/
│   │   ├── csv_handler.py
│   │   ├── html_handler.py 
│   │   ├── json_handler.py
│   │   └── pdf_handler.py
│   └── states/county
│
├── utils/
│   ├── browser_utils.py
│   ├── captcha_tools.py
│   ├── download_utils.py
│   ├── format_router.py
│   ├── html_scanner.py
│   ├── output_utils.py
│   ├── shared_logger.py
│   ├── shared_logic.py
│   └── user_agents.py
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
```

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

   * `HEADLESS=false` (for debugging)
   * `CAPTCHA_TIMEOUT=300`
   * `CACHE_PROCESSED=true`

3. **Add URLs**
   - Populate `urls.txt` with target election result URLs.
   - `url_hint_overrides.txt` is used in conjection with `state_router.py` when dynamic state detection fails.

4. **Run Parser**

   ```bash
   python html_election_parser.py
   ```

---

## 📦 Output Format

Parsed CSV files are saved as:

```
output/{state}/{county}/{race}/{contest}_results.csv
```

Example:

```
output/arizona/maricopa/us_senate/kari_lake_results.csv
```

---

## 🧩 Extending the Parser

* **Add New States**: Create a new file in `handlers/` (e.g. `georgia.py`) and implement a `parse()` method.
* **Add Format Support**: Add new file in `handlers/formats/` and map in `format_router.py`.
* **Shared Behavior**: Use `handlers/shared_logic.py` for common race detection, total extraction, etc.

---

## 🔐 Notes on Security

* All scraping runs headlessly unless CAPTCHA is triggered.
* `.env` is excluded from version control via `.gitignore`.
* No credentials or session tokens are stored.

---

## 🚧 Roadmap

* Multi-race selection prompt
* Retry logic on failed URLs
* Browser fingerprint obfuscation
* Contributor upload queue (for handler patches)
* YAML config option for handler metadata

---

## 📄 License

MIT License (TBD)

---

## 🙋‍♀️ Contributors

* Lead Dev: \[Juancarlos Barragan]
* Elections Research: TBD
* PDF Table Extraction: TBD
