---
layout: default
title: System Architecture
---

## System Architecture

## Overview

This document provides the comprehensive architecture overview of the Smart Elections Parser system, including all major layers, components, responsibilities, and data flow.

> **Note**: This document consolidates content from:
>
> - [architecture.md](../architecture.md)
> - [handlers.md](../handlers.md)
> - [pipeline_map.md](../pipeline_map.md)
>
> For detailed information, consult the individual source documents linked above.

## 🧱 Project Layers

### 1. Entry Point

**`html_election_parser.py`** - Main orchestrator:

- Delegates all specialized logic, never implements scraping/parsing directly
- Handles browser setup, CAPTCHA detection, user input collection
- Delegates parsing to state- or format-specific handlers
- Supports batch mode, multiprocessing, and integration modes
- Logs all actions for auditability

**CLI branch**:

- Primary CLI entry path for batch runs, headless parsing, and automation
- Accepts file/URL lists, interactive prompts, and integration flags
- Shares the same routing and handler contracts as the web app

### 2. Router Layer

**CLI vs Web parity**:

- Both entry paths route through the same `state_router.py` and `format_router.py` logic
- Any routing change applies to CLI and web runs, ensuring consistent handler selection

#### State Router

**`state_router.py`**:

- Matches URLs to specific state handlers in `handlers/`
- Falls back to format detection if state match not found
- Handles dynamic routing for county-level and format-level delegation

#### Format Router

**`utils/format_router.py`**:

- Detects HTML, PDF, JSON, or CSV formats using `html_scanner.py`
- Handles user prompting for format selection via `prompt_user_for_format()`
- Dispatches to appropriate format handler

### 3. Handlers

#### State-Specific Handlers

**`handlers/states/`**:

- One handler per U.S. state (e.g., `arizona.py`, `new_york.py`)
- Each exports `parse(page, html_context)` → `(headers, data, contest, metadata)`
- County-level handlers in `handlers/states/<state>/county/`
- Implements state-specific validation, normalization, and extraction logic

#### Format Handlers

**`handlers/formats/`**:

- Generic format parsers: `pdf_handler.py`, `json_handler.py`, `csv_handler.py`, `html_handler.py`
- Fallback when no state handler exists
- Return standardized tuple: `(headers, data, contest, metadata)`

#### Shared Handler Logic

**`handlers/shared/`**:

- Reusable templates, normalizers, and validation functions
- Contest selection, header harmonization, data cleaning utilities
- Parity hook layer (`handlers/shared/parity_hooks.py`) safely passes router notes into handler outputs

### Internal NLP/ML Foundation (No External AI APIs)

**Design Philosophy**:

- **No external AI APIs**: System relies exclusively on local NLP models (spaCy) + ML framework (scikit-learn, sentence-transformers)
- **Reproducibility**: Consistent results for election integrity verification
- **Non-partisan**: Mathematical risk assessment without bias
- **Mathematical Framework**: 9-dimensional risk vector space (see `ALGORITHMIC_APPROACH_SUMMARY.md`)

**Core Components**:

- **spaCy NER** (`en_core_web_sm`): Entity recognition for candidates, parties, jurisdictions
- **sentence-transformers**: Embedding generation for semantic similarity and context matching
- **scikit-learn**: Clustering, outlier detection, and validation scoring
- **Risk Gates Calculus**: Three-gate threshold system (Confidence, Verification, Anomaly) with six derivative dimensions for rate-of-change analysis
- **HuggingFace pipelines**: Optional local transformer models for specialized tasks (no cloud dependencies)

**Key Files**:

- `webapp/parser/health/risk_gates.py`: Risk vector calculation and threshold enforcement
- `webapp/parser/health/risk_gates_calculus.py`: Derivative dimension computation
- `ALGORITHMIC_APPROACH_SUMMARY.md`: Mathematical foundation and 9-dimensional vector space specification

### 4. Core Utilities

#### Table Detection & Extraction

**`utils/table_core.py`**:

- Centralized table extraction, harmonization, and feedback
- Multi-strategy extraction: panel, section, ML/NER, plugin-based
- Dynamic scoring and patching from multiple extraction methods
- Keyword libraries for election-specific columns

**`utils/dynamic_table_extractor.py`**:

- Finds tables using panel and section heading strategies
- Plugin-based and ML/NER-powered extraction
- Returns candidate tables with context

**`utils/extraction_strategies.py`**:

- Layered table extraction strategies ordered by confidence and cost
- Heuristic HTML, heading, pattern, and selectolax fallbacks
- Strategy registry used by `table_core.py`

#### Table Processing

**`utils/table_builder.py`**:

- Normalizes, merges, annotates, and pivots tables
- Applied to CSV, JSON, TXT, PDF, and state pipeline formats
- Cached header normalization and row-salvage heuristics
- Ensures consistent downstream testing and exports

#### NLP/Entity Recognition

**`utils/spacy_utils.py`**:

- NLP-powered entity recognition
- Context enrichment and semantic analysis
- Supports extraction and enrichment alongside `table_core.py`

#### Browser & Network

**`utils/browser_utils.py`**:

- Launches Playwright (default) with optional Selenium fallback
- Supports headless and GUI modes
- User-agent spoofing and browser profile management

**`utils/download_utils.py`**:

- Handles file downloads and directory creation
- Manages temporary files and cleanup

#### Content Analysis

**`utils/html_scanner.py`**:

- Early-stage HTML scan for election year, races, counties
- Critical for routing and user prompt generation
- Detects format and content patterns

#### User Input & Output

**`utils/user_prompt.py`**:

- All user input routed through `prompt_user_input()` for CLI/web modularity
- Supports interactive selection, validation, and retry logic

**`utils/output_utils.py`**:

- Handles output formatting and metadata generation
- Audit trail and chain-of-custody tracking
- CSV, JSON, and report generation

**`utils/shared_logger.py`**:

- Centralized logging (all modules)
- CLI and Web UI support
- Structured logging for diagnostics

**`utils/shared_logic.py`**:

- Common validation, normalization, and transformation utilities
- Cross-module helper functions

#### Context & State Management

**`Context_Integration/` module**:

- Manages extraction context across handlers
- Handles state and contest selection
- Validates data integrity throughout pipeline

### 5. Web Application

**`Smart_Elections_Parser_Webapp.py`** - Flask application:

- web-based parsing UI
- Session management and state tracking
- Integration with parser backend
- Real-time progress and result display

**`static/js/` & `static/css/`**:

- Client-side logic for form handling, progress tracking
- UI state management and event delegation
- Responsive design and accessibility

### 6. Quality Assurance & Testing

**`webapp/tests/`**:

- Unit tests for all major components
- Integration tests for end-to-end workflows
- Test fixtures for common scenarios

**`health/`**:

- Health checks and automated validation
- Manual correction and feedback mechanisms

## 📊 Data Flow

```tree
URL Input
   ├→ CLI: html_election_parser.py
   │     ↓
   │   state_router.py (state match?)
   │     ├→ YES: handlers/states/<state>.py parse()
   │     └→ NO: format_router.py
   │           ├→ html_scanner.py (detect format)
   │           └→ handlers/formats/<type>_handler.py parse()
   └→ Web: Smart_Elections_Parser_Webapp.py
         ↓
       web_pipeline.process_urls_for_web()
         ↓
       state_router.py (state match?)
         ├→ YES: handlers/states/<state>.py parse()
         └→ NO: format_router.py
               ├→ html_scanner.py (detect format)
               └→ handlers/formats/<type>_handler.py parse()
   ↓
html_context (scout extraction paths)
   ├→ dynamic_table_extractor.py (find tables)
   ├→ extraction_strategies.py (ranked extraction strategies)
   └→ spacy_utils.py (entity recognition)
   ↓
table_core.py (harmonize + score)
   ↓
table_builder.py (normalize + pivot)
   ↓
contest_selector.py (select races)
   ↓
output_utils.py (format results)
   ↓
CSV/JSON Output + Metadata
```

## 🔄 Extraction Strategies

### 1. Panel-Based Strategy

- Identifies contiguous blocks of election data
- Effective for standardized election templates
- Quick and reliable for well-formatted sources

### 2. Section-Based Strategy

- Uses heading hierarchies and semantic structure
- Handles varied formatting and multiple sections
- Integrates content across hierarchical divisions

### 3. ML/NER Strategy

- Neural entity recognition for election-specific terms
- Context-aware extraction from free-form text
- Handles novel layouts and irregular sources

### 4. Plugin Strategy

- Extensible framework for custom extraction logic
- State-specific or format-specific plugins
- Allows rapid addition of new recognition patterns

## 🎯 Core Contracts

### Handler Interface

All handlers must implement:

```python
def parse(page, html_context):
    """Parse ballot data from page.

    Args:
        page: Playwright/Selenium page object
        html_context: Extracted table context

    Returns:
        (headers, data_rows, contest, metadata)
    """
```

### Return Tuple Structure

- **headers** (list[str]): Column names after normalization
- **data_rows** (list[dict]): Normalized data rows
- **contest** (dict): Selected races/contests with metadata
- **metadata** (dict): Extract confidence, source, version, etc.

## 📈 Scalability Considerations

### Modular Design

- Each handler can evolve independently
- Format routers allow new format support without core changes
- Shared utilities prevent code duplication

### Performance

- Browser pooling for batch processing
- Table caching and memoization strategies
- Async/await support for web integration

### Maintainability

- Clear separation of concerns
- Consistent logging throughout
- Comprehensive error handling

---

**Last Updated**: Consolidated from architecture, handlers, and pipeline documentation
**For Details**: See individual source documents in docs root directory
