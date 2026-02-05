# Smart Elections Parser Documentation

Welcome to the Smart Elections Parser documentation. This directory contains comprehensive guides for using, developing, and operating the parser system.

## 📚 Documentation Structure

Documentation is organized into 6 categories plus development resources:

### 🔷 [CORE/](./CORE/)
Architecture, data models, and constants reference
- [System Architecture](./CORE/ARCHITECTURE.md) - System design and components
- [Data Models & Schema](./CORE/DATA_MODELS.md) - Data structures and validation
- [Constants Reference](./CORE/CONSTANTS.md) - Enumerations and static data

### 🔶 [DEPLOYMENT/](./DEPLOYMENT/)
Deployment guides, security, and operations
- [Deployment Guide](./DEPLOYMENT/DEPLOYMENT.md) - Installation and setup procedures
- [Security & Authentication](./DEPLOYMENT/SECURITY.md) - Certificate-based access control
- [Operations Runbook](./DEPLOYMENT/OPERATIONS.md) - Operational procedures and troubleshooting

### 🔵 [QUALITY/](./QUALITY/)
Quality assurance, testing, and ML-powered improvements
- [Verification Framework](./QUALITY/VERIFICATION.md) - QA and testing procedures
- [Quarantine System](./QUALITY/QUARANTINE_SYSTEM.md) - Handling low-quality results
- [ML Framework](./QUALITY/ML_FRAMEWORK.md) - Machine learning quality improvements

### 🟢 [FEATURES/](./FEATURES/)
Feature guides, how-to documentation, and operational guidelines
- [Developer Guides](./FEATURES/GUIDES.md) - Handler development and architecture
- [FEC Fuzzy Matching](./FEATURES/FEC_FUZZY_MATCHING.md) - Candidate matching
- [Confidence Framework](./FEATURES/CONFIDENCE_FRAMEWORK.md) - Confidence scoring
- [Election Operations](./FEATURES/ELECTION_OPERATIONS.md) - Election day procedures
- [Integrity Guidelines](./FEATURES/INTEGRITY_GUIDELINES.md) - Election integrity standards

### 🟡 [GOVERNANCE/](./GOVERNANCE/)
System governance, principles, and decision-making
- [System Governance](./GOVERNANCE/GOVERNANCE.md) - Organizational structure and policies

### 🟣 [DEVELOPMENT/](./DEVELOPMENT/)
Auto-generated documentation and development resources
- [TODO Overview](./DEVELOPMENT/TODOS_OVERVIEW.md) - How-to guide for the TODO system
- [All TODOs](./DEVELOPMENT/todos.md) - Complete list of outstanding work items
- [High Priority](./DEVELOPMENT/todos_high.md) - Critical and urgent items
- [Medium Priority](./DEVELOPMENT/todos_medium.md) - Improvements and tech debt
- [Low Priority](./DEVELOPMENT/todos_low.md) - Future work and nice-to-haves
- [Project Audit](./DEVELOPMENT/project_audit.md) - Comprehensive module audit with Mermaid diagrams
- [Pipeline Map](./DEVELOPMENT/pipeline_map.md) - Detailed pipeline connections and architecture

> **Note**: Files in `DEVELOPMENT/` are auto-generated from code analysis. See [TODO Overview](./DEVELOPMENT/TODOS_OVERVIEW.md) for the TODO system.

---

## 🚀 Quick Start

**New to the project?**
1. Start with [System Architecture](./CORE/ARCHITECTURE.md)
2. Review [Data Models & Schema](./CORE/DATA_MODELS.md)
3. Check [Developer Guides](./FEATURES/GUIDES.md) if developing

**Deploying to production?**
1. Follow [Deployment Guide](./DEPLOYMENT/DEPLOYMENT.md)
2. Configure security per [Security & Authentication](./DEPLOYMENT/SECURITY.md)
3. Review [Operations Runbook](./DEPLOYMENT/OPERATIONS.md)

**Running elections?**
1. Prepare with [Election Operations](./FEATURES/ELECTION_OPERATIONS.md)
2. Monitor using [Operations Runbook](./DEPLOYMENT/OPERATIONS.md)
3. Review [Integrity Guidelines](./FEATURES/INTEGRITY_GUIDELINES.md)

**Working on quality?**
1. Understand [Verification Framework](./QUALITY/VERIFICATION.md)
2. Use [Quarantine System](./QUALITY/QUARANTINE_SYSTEM.md) for flagged results
3. Leverage [ML Framework](./QUALITY/ML_FRAMEWORK.md) for improvements

---

## 📖 Documentation Guidelines

### For Users
- Start with quick-start sections
- Follow step-by-step guides
- Check troubleshooting for common issues
- Refer to examples and code snippets

### For Developers
- Review architecture before coding
- Follow handler development patterns in [Developer Guides](./FEATURES/GUIDES.md)
- Ensure compliance with [Data Models](./CORE/DATA_MODELS.md)
- Test thoroughly per [Verification Framework](./QUALITY/VERIFICATION.md)

### For Operations
- Follow [Deployment Guide](./DEPLOYMENT/DEPLOYMENT.md) for installation
- Use [Operations Runbook](./DEPLOYMENT/OPERATIONS.md) for day-to-day ops
- Reference [Election Operations](./FEATURES/ELECTION_OPERATIONS.md) during elections
- Consult [Security & Authentication](./DEPLOYMENT/SECURITY.md) for security concerns

---

## 📞 Getting Help

To find information about a specific topic:

1. **Check the category headers above** - Documentation is organized by domain
2. **Use Ctrl+F to search** for specific keywords within this index
3. **Read the Related Documents** section at the bottom of each page
4. **Check the source documents** linked in consolidation notes

---

## 🔄 Documentation Maintenance

This documentation is actively maintained and includes consolidated content from 70+ source documents. The [DEVELOPMENT/](./DEVELOPMENT/) directory includes auto-generated TODO items that track outstanding work.

**Last Updated**: Consolidated documentation system (2024)
**Total Documentation**: 15 master files across 6 categories + development resources
**Coverage**: Core architecture, deployment, operations, quality, features, and governance

---

## 📋 Source Attribution

This consolidated documentation draws from the following source materials:

**CORE/**
- architecture.md, handlers.md, pipeline_map.md
- VERIFIED_DATA_SCHEMA.md, CONSTANTS_INVENTORY.md
- VERIFICATION_FRAMEWORK.md

**DEPLOYMENT/**
- DEPLOYMENT_GUIDE.md, AZURE_DEPLOYMENT_CHECKLIST.md
- AZURE_CERTIFICATE_AUTH_SETUP.md, CERT_AUTH_IMPLEMENTATION.md
- ELECTION_OPERATIONS_PLAYBOOK.md, troubleshooting.md
- INTEGRITY_MONITORING.md, WAREHOUSE_VERIFICATION_GUIDE.md

**QUALITY/**
- VERIFICATION_TESTING_GUIDE.md, VERIFICATION_SYNC_IMPLEMENTATION.md
- QUARANTINE_SYSTEM_GUIDE.md
- ML_QUICKSTART.md, ML_OPTIMIZATION_METRICS.md
- ML_QUALITY_METRICS_SUMMARY.md, ML_DEPLOYMENT_CHECKLIST.md

**FEATURES/**
- HANDLER_MIGRATION_GUIDE.md, MODERN_UI_FEATURES.md
- fec_fuzzy.md
- CONFIDENCE_CAUTION_FRAMEWORK.md
- ELECTION_OPERATIONS_PLAYBOOK.md
- Election_Integrity_Guidelines.md

**GOVERNANCE/**
- SYSTEM_GOVERNANCE.md

All source materials remain in the repository root for reference if needed.
