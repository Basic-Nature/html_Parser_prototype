---
layout: default
title: Smart Elections Parser Documentation
---

<!-- markdownlint-disable MD033 -->

## 📁 Smart Elections Parser Documentation

Welcome to the comprehensive documentation for the **Smart Elections Parser** - a modular web scraper for U.S. election results supporting HTML, PDF, CSV, and JSON formats with browser automation, OCR, and ML-powered integrity checks.

---

## 🚀 Quick Navigation

<div class="feature-list">
  <div class="feature" data-section="architecture">
    <h3>🏗️ System Architecture</h3>
    <p>Complete system overview, data flow, and component interactions. Understand how the parser orchestrates multi-format extraction.</p>
    <a href="CORE/ARCHITECTURE.html">View Architecture →</a>
  </div>

  <div class="feature" data-section="audit">
    <h3>🔍 Project Audit</h3>
    <p>Automated analysis of all modules, dependencies, and cross-references with interactive Mermaid diagrams.</p>
    <a href="DEVELOPMENT/project_audit.html">View Audit →</a>
  </div>

  <div class="feature" data-section="roadmap">
    <h3>📋 Development Roadmap</h3>
    <p>Complete feature roadmap including planned features and development priorities.</p>
    <a href="IMPLEMENTATION-STATE.html">View Roadmap →</a>
  </div>

  <div class="feature" data-section="features">
    <h3>🚀 Active Features</h3>
    <p>Selenium NLP integration, multi-format support, and ML-powered validation.</p>
    <a href="FEATURES/GUIDES.html">Browse Features →</a>
  </div>

  <div class="feature" data-section="deployment">
    <h3>🚀 Deployment & Operations</h3>
    <p>Azure App Service setup, configuration, and best practices.</p>
    <a href="DEPLOYMENT/DEPLOYMENT.html">View Deployment →</a>
  </div>

  <div class="feature" data-section="ci-topology">
    <h3>🧭 CI Topology</h3>
    <p>See exactly how Azure dynamic deployment and GitHub Pages docs deployment are separated, triggered, and validated.</p>
    <a href="DEPLOYMENT/CI_TOPOLOGY.html">View CI Topology →</a>
  </div>

  <div class="feature" data-section="development">
    <h3>🔧 Development Guide</h3>
    <p>Local setup, handler development, and contribution workflows.</p>
    <a href="DEVELOPMENT/todos.html">Get Started →</a>
  </div>

  <div class="feature" data-section="quality">
    <h3>✅ Quality & Testing</h3>
    <p>Health checks, automation scripts, and validation procedures.</p>
    <a href="QUALITY/VERIFICATION.html">View Quality →</a>
  </div>

  <div class="feature" data-section="governance">
    <h3>📋 Governance & Standards</h3>
    <p>Code standards, decision logs, and project governance.</p>
    <a href="GOVERNANCE/GOVERNANCE.html">View Standards →</a>
  </div>
</div>

---

## 📖 Quick Reference

### 🎯 Getting Started

- **[Quick Start Guide](QUICK-START.html)** - Setup and first run
- **[Technical Reference](TECHNICAL-REFERENCE.html)** - API and technical details
- **[System State](IMPLEMENTATION-STATE.html)** - Current implementation status

### 📚 Core Documentation

- **[Executive Summary](EXECUTIVE-SUMMARY.html)** - High-level overview
- **[Session Summary](SESSION-SUMMARY.html)** - Recent work and decisions
- **[State Handler Integration](STATE_HANDLER_INTEGRATION.html)** - Handler patterns

### 🔗 Special Topics

- **[Storage Architecture](FEATURES/STORAGE_ARCHITECTURE.html)** - Storage and parser output architecture
- **[Election Integrity](FEATURES/INTEGRITY_GUIDELINES.html)** - Integrity practices
- **[Selenium NLP Integration](FEATURES/SELENIUM_NLP_INTEGRATION.html)** - Browser automation with NLP
- **[CI Topology](DEPLOYMENT/CI_TOPOLOGY.html)** - Azure workflow vs GitHub Pages workflow separation

---

## 📂 Documentation Structure

```markdown
docs/
├── CORE/                          # Core architecture & design
├── DEPLOYMENT/                    # Azure, Docker, production setup
├── DEVELOPMENT/                   # Local dev, handlers, testing
├── FEATURES/                      # Feature documentation & roadmaps
├── GOVERNANCE/                    # Standards, decisions, integrity
├── QUALITY/                       # Testing, health checks, validation
├── implementation-history/        # Archived session logs
├── implementation-phases/         # Phase documentation
└── [Standalone Docs]              # Summary and reference files
```

---

## 🔧 Development Quick Links

### For Developers

- **[TODO Overview](DEVELOPMENT/TODOS_OVERVIEW.html)** - Active development tasks and workflow
- **[Project Audit](DEVELOPMENT/project_audit.html)** - Module map and integration hotspots
- **[Verification Framework](QUALITY/VERIFICATION.html)** - QA workflow and test status

### For DevOps

- **[Azure Deployment](DEPLOYMENT/AZURE_CSP_DEPLOYMENT.html)** - Deploy to App Service
- **[Deployment Guide](DEPLOYMENT/DEPLOYMENT.html)** - Runtime configuration and environment model
- **[Post-Deploy Verification](DEPLOYMENT/POST_DEPLOY_VERIFICATION.html)** - Validate production health

### For Contributors

- **[Contributing Guide](https://github.com/Basic-Nature/html_Parser_prototype/blob/main/CONTRIBUTING.md)** - Contribution workflow
- **[Governance Guide](GOVERNANCE/GOVERNANCE.html)** - Standards and operating model
- **[Documentation Index](DOCUMENTATION-INDEX.html)** - Architecture and decision references

---

## 🚀 Recent Highlights

### Active Work (February 2026)

- **Socket.IO Multi-Instance Testing** - Validated broadcast propagation across clients
- **Azure mTLS Configuration** - HTTP/1.1 + TLS 1.2 for stable cert handling
- **Shared Auth Utilities** - Protected data_framework, health_dashboard, ballot_lens
- **Session Management** - Per-principal session caps with flexibility for autoscaling

### Key Features

- ✅ Multi-format support (HTML, PDF, CSV, JSON, XLSX)
- ✅ Browser automation with Selenium
- ✅ OCR with Tesseract and PDF extraction
- ✅ ML-powered anomaly detection
- ✅ Comprehensive audit logging

---

## 📊 Project Metrics

| Metric | Status |
| --- | --- |
| **Formats Supported** | HTML, PDF, CSV, JSON, TXT, XLSX |
| **States Covered** | Multiple state parsers implemented |
| **ML Integration** | Active context learning and anomaly detection |
| **Documentation** | Comprehensive technical reference |
| **Test Coverage** | Health checks, integration tests, automation |

---

## 🛡️ Election Integrity & Transparency

***Built for Trust & Accountability***

Every extraction, correction, and output is logged with comprehensive metadata for reproducibility and auditability.

- **Auditable Processing**: All operations logged with timestamps and context
- **ML-Powered Validation**: Anomaly detection and structure verification
- **Human-in-the-Loop**: Feedback systems for continuous improvement
- **Open Formats**: CSV/JSON outputs with full metadata preservation
- **Secure Multi-User**: Session isolation with certificate binding

---

## 🙋‍♀️ Need Help?

- **[GitHub Issues](https://github.com/Basic-Nature/html_Parser_prototype/issues)** - Report bugs or request features
- **[GitHub Discussions](https://github.com/Basic-Nature/html_Parser_prototype/discussions)** - Ask questions and get community help
- **[Full Documentation Index](DOCUMENTATION-INDEX.html)** - Browse all documentation
- **[Repository Contributing Guide](https://github.com/Basic-Nature/html_Parser_prototype/blob/main/CONTRIBUTING.md)** - How to contribute

---

## 📝 Documentation Status

| Aspect | Status |
| --- | --- |
| **Architecture** | ✅ Complete |
| **Development Guide** | ✅ Complete |
| **Deployment Guide** | ✅ Complete |
| **API Reference** | ✅ Complete |
| **Feature Docs** | ✅ Comprehensive |
| **Examples** | ✅ Available |

**Last updated**: February 16, 2026  
**Version**: Active Development  
**Maintenance**: Actively maintained
