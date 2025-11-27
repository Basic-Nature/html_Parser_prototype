---
layout: default
---

<!-- markdownlint-disable MD033 -->

# 📁 Smart Elections Documentation

Welcome to the developer and contributor guide for the **Smart Elections Parser**.
This index links to all core documents and resources for building, extending, and maintaining the project.

---

## 🧭 Quick Access

<div class="feature-list">
  <div class="feature" data-section="architecture">
    <h3>🏗️ Architecture</h3>
    <p>System components, orchestration, and data flow. Understand how the parser works end-to-end.</p>
    <a href="architecture.md">View Architecture →</a>
  </div>

  <div class="feature" data-section="audit">
    <h3>🔍 Project Audit</h3>
    <p>Comprehensive audit of webapp modules, dependencies, and cross-references with Mermaid diagrams.</p>
    <a href="project_audit.md">View Audit →</a>
  </div>

  <div class="feature" data-section="todos">
    <h3>📋 TODOs & Tasks</h3>
    <p>Index of all TODO/FIXME/WARN annotations across the codebase with progress tracking.</p>
    <a href="todos.md">View TODOs →</a>
  </div>

  <div class="feature" data-section="pipeline">
    <h3>🔄 Pipeline Map</h3>
    <p>Visual pipeline overview with module details and interactive Mermaid diagrams.</p>
    <a href="pipeline_map.md">View Pipeline →</a>
  </div>
</div>

---

## 📄 Core Documentation

<div class="mission-panel glossy">
  <h3>Essential Reading for Contributors</h3>
  <p>Start here to understand the project structure, contribution guidelines, and development workflow.</p>
</div>

- [`README.md`](../README.md): Project overview, install steps, CLI and Web UI usage, and high-level architecture
- [`CONTRIBUTING.md`](../CONTRIBUTING.md): How to contribute, coding standards, and review process
- [`handlers.md`](handlers.md): How to build and extend state, county, and format handlers
- [`roadmap.md`](roadmap.md): Planned features, enhancements, and future directions
- [`LICENSE`](../LICENSE): Open-source licensing and reuse terms

---

## 🔍 Automated Analysis & Health

<div class="progress-section">
  <h3>Project Health Dashboard</h3>
  <p>Automated analysis tools for maintaining code quality and tracking project progress.</p>
  <div class="progress-bar">
    <div class="progress-fill" style="width: 85%;"></div>
  </div>
  <p><small>85% of planned health checks implemented</small></p>
</div>

- [`noise_override_suggestions.md`](noise_override_suggestions.md): Suggested overrides for PDF/OCR noise filtering
- [`health/`](../webapp/parser/health/): Correction, retraining, and automation health (see `health_router.py`)
- [`Context_Integration/`](../webapp/parser/Context_Integration/): Context, ML/NLP, and integrity modules
- [`context_library.json`](../webapp/parser/context_library.json): Persistent context and feedback for smarter extraction

---

## 🖥️ Web UI (Optional)

<div class="mission-panel">
  <h3>Graphical Interface for Teams & Researchers</h3>
  <p>The Smart Elections Parser includes an optional Flask-based Web UI for users who prefer a graphical experience.</p>
</div>

**Web UI Features:**

- Dashboard for quick access to all tools
- URL Hint Manager for managing custom URL-to-handler mappings
- Change History for configuration transparency and auditability
- "Run Parser" page with real-time output and styled terminal-like area
- Live feedback via WebSockets
- Data management for uploads, downloads, and review

<div class="status-badge in-progress">Web UI Active</div>

---

## 🧩 Extensibility & Utilities

- [`utils/`](../webapp/parser/utils/): Shared utilities for browser automation, CAPTCHA, download, contest selection, table extraction, ML/NER, and more
- [`handlers/`](../webapp/parser/handlers/): All state/county and format-specific parsing logic
- [`shared/`](../webapp/parser/handlers/shared/): Shared handler logic for reuse

---

## 📦 Data & Resources

- [`requirements.txt`](../requirements.txt): Required Python packages
- [`urls.txt`](..webapp/parser/urls.txt): Starter list of known election result pages
- [`output/`](../output/): Parsed results (organized by state/county/race)
- [`input/`](../input/): Place files for manual/override parsing
- [`log/`](../webapp/parser/Context_Integration/Context_Library/log/): Persistent logs and audit trails

---

## 🧪 Testing & Debugging

<div class="feature-list">
  <div class="feature" data-section="architecture">
    <h3>🔧 Debug Mode</h3>
    <p>Use `.env` variables like `HEADLESS=false`, `ENABLE_BOT_TASKS=true`, or `CACHE_RESET=true` to control behavior.</p>
  </div>

  <div class="feature" data-section="pipeline">
    <h3>📁 Manual Testing</h3>
    <p>Try parsing pre-downloaded HTML or file formats using the `input/` directory and simulate CAPTCHA triggers.</p>
  </div>

  <div class="feature" data-section="audit">
    <h3>🔄 Modular Prompts</h3>
    <p>Modular user prompts (`prompt_user_input`) allow easy CLI or web UI testing with correction feedback.</p>
  </div>
</div>

---

## 🛡️ Election Integrity & Transparency

<div class="mission-panel glossy">
  <h3>Built for Trust & Accountability</h3>
  <p>All outputs are auditable with comprehensive logging, ML-powered anomaly detection, and human-in-the-loop feedback.</p>
</div>

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
