---
layout: default
---

# Smart Elections Parser — Roadmap

This document tracks the progress and next steps for the Smart Elections Parser project.

---

## ✅ Completed Milestones

### 🤖 ML Quality Metrics & Optimization (Jan 2026)

**Phase 1 - Foundation:**

- Centralized OCR configuration with 27 env-tunable parameters
- ML quality metrics framework (14 core + nested OCR/table metrics)
- Quality hooks integrated across all handlers (PDF, HTML, CSV, JSON, XLSX)
- ML training dataset generator supporting JSONL/CSV/Parquet export
- Interactive quality metrics dashboard with Chart.js visualizations
- Comprehensive documentation suite (4 guides)

**Phase 2 - Performance Optimizations:**

- **Multi-format OCR extraction**: Supports both nested dict and direct field metadata
- **Weighted confidence scoring**: 30% OCR + 30% completeness + 20% headers + 20% non-empty rows
- **OCR normalization**: Auto-detects and converts 0-100 vs 0.0-1.0 scales
- **Export caching system**: 10-50x faster repeated scans with automatic invalidation
- **Code optimization**: Removed duplicates, improved accuracy

**Results**: 85% quality accuracy (+20%), 95% OCR capture (+55%), 92% faster exports

---

## ✅ Earlier Milestones

- **Modular Handler Architecture:**  
  State, county, and format handlers are fully modular and extensible.
- **Dynamic Table Extraction:**  
  Multi-strategy extraction (panel, section, ML/NER, plugin) with scoring and patching is implemented in `table_core.py` and `dynamic_table_extractor.py`.
- **Persistent Context Library:**  
  `context_library.json` and context enrichment modules are in place for smarter extraction and correction.
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
- **Centralized Automation Script:**  
  `automate.py` provides a single entry point for running pipeline audits, health checks, web asset validation, and testing.

---

## 🚧 Next Steps & Priorities

### 1. **Centralized Parsing & Schema Unification**

- ✅ Pivot pathway now normalizes party metadata and collapses jurisdiction headers; follow up by mirroring these helpers inside `table_builder.py`/metadata writers.
- Validate the new output schema on representative JSON + PDF contests and capture canonical samples for `docs/handlers.md`.
- Populate a true `Division Type` column per record using shared normalization helpers and propagate the value into metadata writers for downstream consumers.
- Drive JSON, PDF-OCR, and future formats through a single contest-selection + table-building pipeline; remove format-specific forks under `webapp/parser/parser/handlers/`.

### 2. **Context Detection & OCR Reliability**

- Improve state/county inference for multi-contest PDFs with better fallbacks when NLP returns `unknown`, plus structured diagnostics to aid debugging.
- Filter contest candidates emitted from OCR so boilerplate (for example, "* Indicates Passage...") never surfaces as selectable options.
- Align the OCR preprocessing flow with the centralized pipeline to keep contest metadata and extraction heuristics consistent across sources.

### 3. **Regression, Telemetry & Documentation**

- Build integration tests covering large multi-contest PDFs, JSON fast-path contests, and ward/precinct edge cases with fixture expectations in `webapp/tests/`.
- Enrich generated `*.metadata.json` files with contest source, detection confidence, and normalization decisions to support audits.
- Update documentation (`docs/index.md`, `docs/handlers.md`) with the refined output schema, analyst guidance for the party/division headers, and troubleshooting notes.

### 4. **ML/NLP Library & Training**

- Improve and expand the ML/NER models for table detection, entity recognition, and anomaly detection.
- Integrate more robust LLM (Large Language Model) support for structure learning and context inference.
- Build a retraining pipeline that leverages correction logs and user feedback for continuous improvement.
- Expand `spacy_utils.py` and `ml_table_detector.py` with new entity types and training data.

### 5. **Web UI & CLI Parity**

- Make the Web UI fully compatible with all CLI logic, including:
  - Contest selection and user prompts
  - Real-time feedback and correction workflows
  - Batch and parallel processing controls
- Add more robust error handling and user guidance in the Web UI.
- Enable upload and manual override of input files via the Web UI.

### 6. **LLM Integration**

- Improve reliability and fallback logic for LLM-based extraction.
- Add support for multiple LLM providers and local models.
- Allow handler and extraction logic to select or override LLM strategies as needed.

### 7. **Handler Expansion**

- Expand the number of state and county handlers, prioritizing high-impact or frequently requested jurisdictions.
- Add more format-specific handlers for edge-case PDFs, JSONs, and vendor-specific HTML.
- Encourage community contributions and provide templates for new handlers.

### 8. **Testing, Validation, and Documentation**

- Expand automated and manual test coverage for handlers and extraction logic.
- Add more sample URLs and edge cases to `urls.txt`.
- Improve documentation for handler development, context enrichment, and bot usage.
- Add troubleshooting and FAQ sections to the Web UI.
- Monitor upstream pytest / SeleniumBase releases so the temporary unraisable warning suppression can be removed once socket cleanup is fixed.

### 9. **Performance & Scalability**

- Optimize multiprocessing and memory usage for large-scale scraping.
- Add caching and smarter deduplication for processed URLs and files.
- Improve download and file management for large input datasets.

### 10. **Election Integrity & Transparency**

- Expand audit trail metadata and correction logging.
- Add more granular anomaly detection and reporting.
- Integrate with external election data sources for cross-validation.

### 11. **User Experience**

- Add more informative error messages and suggestions in both CLI and Web UI.
- Improve accessibility and onboarding for non-technical users.
- Provide more granular progress and status updates during batch runs.

---

## 🧭 Working TODO List

- Document the centralized schema outputs (party, jurisdiction, division type) with before/after examples and share with stakeholders.
- Prototype enhanced contest filtering on OCR output and validate against current PDF samples.
- Stand up first regression fixture (multi-contest PDF) and wire it into the forthcoming integration test harness.
- Design metadata enrichment schema updates and confirm backwards compatibility with existing analytics tooling.
- Outline cross-format extensibility requirements (HTML, XLSX) and capture the effort in `docs/roadmap.md` for prioritization discussions.

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
