# 📁 Smart Elections Documentation

Welcome to the developer and contributor guide for the **Smart Elections Parser**.  
This index links to all core documents and resources for building, extending, and maintaining the project.

---

## 🧭 Core Docs

- [`README.md`](../README.md): Project overview, install steps, and high-level architecture
- [`CONTRIBUTING.md`](../CONTRIBUTING.md): How to contribute, coding standards, and review process
- [`LICENSE`](../LICENSE): Open-source licensing and reuse terms

---

## 📄 Design & Development Documents

- [`docs/architecture.md`](architecture.md): System components, orchestration, and data flow
- [`docs/handlers.md`](handlers.md): How to build and extend state, county, and format handlers
- [`docs/roadmap.md`](roadmap.md): Planned features, enhancements, and future directions

---

## 🧩 Extensibility & Automation

- [`bot/`](../bot/): Automation and notification tasks (see `bot_router.py`)
- [`utils/`](../utils/): Shared utilities for browser, CAPTCHA, download, contest selection, and more
- [`handlers/`](../handlers/): All state/county and format-specific parsing logic

---

## 📦 Other Resources

- [`requirements.txt`](../requirements.txt): Required Python packages
- [`urls.txt`](../urls.txt): Starter list of known election result pages
- [`output/`](../output/): Parsed results (organized by state/county/race)
- [`input/`](../input/): Place files for manual/override parsing

---

## 🧪 Testing & Debugging

- Use `.env` variables like `HEADLESS=false` or `ENABLE_BOT_TASKS=true` to control behavior
- Try parsing pre-downloaded HTML or file formats using the `input/` directory
- Simulate CAPTCHA triggers for state/county sites
- Modular user prompts (`prompt_user_input`) allow easy CLI or web UI testing

---

## 🙋‍♀️ Getting Help

- See the [GitHub Issues](https://github.com/Basic-Nature/html_Parser_prototype) or Discussions tab for questions and support
- Refer to `handlers.md` for handler development, or `README.md` for general usage

---

Happy parsing!
