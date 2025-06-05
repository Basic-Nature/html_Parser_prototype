# 📁 Smart Elections Documentation

Welcome to the developer and contributor guide for the **Smart Elections Parser**.  
This index links to all core documents and resources for building, extending, and maintaining the project.

---

## 🧭 Core Docs

- [`README.md`](../README.md): Project overview, install steps, CLI and Web UI usage, and high-level architecture
- [`CONTRIBUTING.md`](../CONTRIBUTING.md): How to contribute, coding standards, and review process
- [`LICENSE`](../LICENSE): Open-source licensing and reuse terms

---

## 📄 Design & Development Documents

- [`architecture.md`](architecture.md): System components, orchestration, and data flow
- [`handlers.md`](handlers.md): How to build and extend state, county, and format handlers
- [`roadmap.md`](roadmap.md): Planned features, enhancements, and future directions

---

## 🖥️ Web UI (Optional)

The Smart Elections Parser includes an **optional Flask-based Web UI** for users who prefer a graphical experience or are new to coding.

**Web UI Features:**

- Dashboard for quick access to all tools
- URL Hint Manager for managing custom URL-to-handler mappings
- Change History for configuration transparency and auditability
- "Run Parser" page with real-time output and styled terminal-like area
- Live feedback via WebSockets
- Data management for uploads, downloads, and review

The Web UI is ideal for teams, researchers, and those learning to code—**all core parser features remain available via the CLI**.

---

## 🤖 Automation, Bots & Context

- [`bots/`](../bots/): Correction, retraining, and automation bots (see `bot_router.py`)
- [`Context_Integration/`](../Context_Integration/): Context, ML/NLP, and integrity modules (`context_coordinator.py`, `context_organizer.py`, `Integrity_check.py`)
- [`context_library.json`](../context_library.json): Persistent context and feedback for smarter extraction and correction

---

## 🧩 Extensibility & Utilities

- [`utils/`](../utils/): Shared utilities for browser automation, CAPTCHA, download, contest selection, table extraction, ML/NER, and more
- [`handlers/`](../handlers/): All state/county and format-specific parsing logic
- [`shared/`](../handlers/shared/): Shared handler logic for reuse

---

## 📦 Data & Resources

- [`requirements.txt`](../requirements.txt): Required Python packages
- [`urls.txt`](../urls.txt): Starter list of known election result pages
- [`output/`](../output/): Parsed results (organized by state/county/race)
- [`input/`](../input/): Place files for manual/override parsing
- [`log/`](../log/): Persistent logs and audit trails

---

## 🧪 Testing & Debugging

- Use `.env` variables like `HEADLESS=false`, `ENABLE_BOT_TASKS=true`, or `CACHE_RESET=true` to control behavior
- Try parsing pre-downloaded HTML or file formats using the `input/` directory
- Simulate CAPTCHA triggers for state/county sites
- Modular user prompts (`prompt_user_input`) allow easy CLI or web UI testing
- Correction and feedback bots help retrain extraction logic and improve future runs

---

## 🛡️ Election Integrity & Transparency

- All outputs are auditable: logs, metadata, and correction trails are saved
- ML/NER-powered anomaly detection and structure validation
- Human-in-the-loop feedback at every stage
- Persistent context library for smarter, more reliable extraction

---

## 🙋‍♀️ Getting Help

- See the [GitHub Issues](https://github.com/Basic-Nature/html_Parser_prototype) or Discussions tab for questions and support
- Refer to `handlers.md` for handler development, or `README.md` for general usage
- The Web UI is documented in the `README.md` and is fully optional—use the interface that best fits your workflow!

---

Happy parsing!
