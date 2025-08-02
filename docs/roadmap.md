# Smart Elections Parser — Roadmap

This document tracks the progress and next steps for the Smart Elections Parser project.

---

## ✅ Completed Milestones

- **Modular Handler Architecture:**  
  State, county, and format handlers are fully modular and extensible.
- **Dynamic Table Extraction:**  
  Multi-strategy extraction (panel, section, ML/NER, plugin) with scoring and patching is implemented in `table_core.py` and `dynamic_table_extractor.py`.
- **Persistent Context Library:**  
  `context_library.json` and context enrichment modules are in place for smarter extraction and correction.
- **Bots & Automation:**  
  Correction and retraining bots are implemented (`bots/`), with `.env`-driven enable/disable.
- **Election Integrity Checks:**  
  ML/NER-based anomaly detection and cross-field validation are integrated (`Context_Integration/Integrity_check.py`).
- **Web UI (Flask):**  
  Web interface for running the parser, managing URLs, and reviewing output is live.
- **Unified Logging & Audit Trails:**  
  All actions and corrections are logged for transparency and reproducibility.
- **Batch & Parallel Processing:**  
  Multiprocessing and batch scraping are supported.
- **Security & Compliance:**  
  Path traversal protections, .env-driven config, and no credential storage.
- **User Prompt Abstraction:**  
  All user input is routed through `prompt_user_input()` for CLI/Web UI compatibility.
- **Format Handlers:**  
  CSV, PDF, JSON, and HTML fallback handlers are implemented and registered.
- **Shared Utilities:**  
  Centralized browser, CAPTCHA, download, and output logic in `utils/`.

---

## 🚧 Next Steps & Priorities

### 1. **ML/NLP Library & Training**

- Improve and expand the ML/NER models for table detection, entity recognition, and anomaly detection.
- Integrate more robust LLM (Large Language Model) support for structure learning and context inference.
- Build a retraining pipeline that leverages correction logs and user feedback for continuous improvement.
- Expand `spacy_utils.py` and `ml_table_detector.py` with new entity types and training data.

### 2. **Web UI & CLI Parity**

- Make the Web UI fully compatible with all CLI logic, including:
  - Contest selection and user prompts
  - Real-time feedback and correction workflows
  - Batch and parallel processing controls
- Add more robust error handling and user guidance in the Web UI.
- Enable upload and manual override of input files via the Web UI.

### 3. **LLM Integration**

- Improve reliability and fallback logic for LLM-based extraction.
- Add support for multiple LLM providers and local models.
- Allow handler and extraction logic to select or override LLM strategies as needed.

### 4. **Handler Expansion**

- Expand the number of state and county handlers, prioritizing high-impact or frequently requested jurisdictions.
- Add more format-specific handlers for edge-case PDFs, JSONs, and vendor-specific HTML.
- Encourage community contributions and provide templates for new handlers.

### 5. **Testing, Validation, and Documentation**

- Expand automated and manual test coverage for handlers and extraction logic.
- Add more sample URLs and edge cases to `urls.txt`.
- Improve documentation for handler development, context enrichment, and bot usage.
- Add troubleshooting and FAQ sections to the Web UI.

### 6. **Performance & Scalability**

- Optimize multiprocessing and memory usage for large-scale scraping.
- Add caching and smarter deduplication for processed URLs and files.
- Improve download and file management for large input datasets.

### 7. **Election Integrity & Transparency**

- Expand audit trail metadata and correction logging.
- Add more granular anomaly detection and reporting.
- Integrate with external election data sources for cross-validation.

### 8. **User Experience**

- Add more informative error messages and suggestions in both CLI and Web UI.
- Improve accessibility and onboarding for non-technical users.
- Provide more granular progress and status updates during batch runs.

---

## 📝 Additional Ideas & Stretch Goals

- **Plugin System:**  
  Allow third-party plugins for extraction, validation, or output formatting.
- **Automated Data Publishing:**  
  Integrate with open data portals or APIs for publishing results.
- **Crowdsourced Correction:**  
  Enable collaborative correction and feedback via the Web UI.
- **Advanced Visualization:**  
  Add basic charts or maps to the Web UI for quick data review.
- **Internationalization:**  
  Prepare for non-U.S. election formats and multilingual support.

---
