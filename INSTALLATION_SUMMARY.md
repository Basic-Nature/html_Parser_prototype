# Installation Resources - Summary

## Overview

Complete installation resources for the Smart Elections Parser cover production and development environments with scripts, guides, and quick references.

---

## Files Created

### Documentation

- **INSTALLATION_GUIDE.md** — comprehensive platform-specific guide (~400 lines)
- **INSTALL_QUICKSTART.md** — one-page quick reference

### Installation Scripts

- **install.py** — cross-platform Python installer with verification, optional tests, and security checks
- **install.bat** — Windows installer with dependency checks
- **install.sh** — Linux/macOS installer with system package helpers

**install.py usage:**

```bash
python install.py                  # Normal installation
python install.py --dev            # Development installation
python install.py --with-tests     # Include testing tools
python install.py --run-tests      # Run security tests after install
```

**install.bat usage:**

```batch
install.bat                # Normal installation
install.bat --dev          # Development installation
install.bat --with-tests   # Include testing tools
```

**install.sh usage:**

```bash
chmod +x install.sh
./install.sh                      # Normal installation
./install.sh --dev                # Development installation
./install.sh --with-tests         # Include testing tools
./install.sh --skip-system-deps   # Skip system packages
```

---

## Installation Methods

### Method 1: Automated (Recommended)

- **Windows:**

```batch
install.bat --dev
```

- **Linux/macOS:**

```bash
chmod +x install.sh
./install.sh --dev
```

- **Cross-platform:**

```bash
python install.py --dev --with-tests
```

### Method 2: Manual

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt -r requirements-dev.txt
pip install pytest pytest-cov pytest-mock

# Install Playwright browsers
playwright install chromium

# Verify
python -c "import flask_socketio, spacy; print('OK')"
```

### Method 3: One-liner

- **Production:**

```bash
pip install -r requirements.txt && playwright install chromium
```

- **Development:**

```bash
pip install -r requirements.txt -r requirements-dev.txt && pip install pytest pytest-cov pytest-mock && playwright install chromium
```

---

## Dependency Files

### requirements.txt (Production)

Key packages:

- Web Automation: playwright, selectolax, langdetect
- Web Framework: flask-socketio, gunicorn, eventlet
- File Processing: pandas, pytesseract, pdf2image, PyMuPDF, camelot-py
- Database: sqlalchemy, psycopg2
- NLP/ML: spacy, sentence_transformers, torch (CPU), nltk
- Utilities: rich, rapidfuzz, matplotlib, orjson

### requirements-dev.txt (Development)

Key packages:

- Linting: ruff
- Type Checking: mypy, types-requests
- Git Hooks: pre-commit

### Additional Testing Dependencies

- pytest
- pytest-cov
- pytest-mock

---

## System Dependencies Required

### Windows

1. **Poppler** — PDF processing. Download from <https://github.com/oschwartz10612/poppler-windows/releases> and add to PATH or set `POPPLER_PATH`.
2. **Tesseract OCR** — text recognition. Download from <https://github.com/UB-Mannheim/tesseract/wiki> and add to PATH.
3. **Ghostscript** — PostScript processing. Download from <https://ghostscript.com/releases/gsdnld.html>.

### Linux (Ubuntu/Debian)

```bash
sudo apt-get install poppler-utils tesseract-ocr ghostscript
```

### macOS

```bash
brew install poppler tesseract ghostscript
```

---

## Installation Features

### install.py

- Python version check (3.12+)
- Automatic pip upgrade
- Main + optional dev/testing dependencies
- Playwright browser install
- spaCy model verification
- System dependency reminders
- Optional security tests
- Colored output and robust error handling

### install.bat

- Python availability check
- pip upgrade
- Main + optional dev/testing dependencies
- Playwright install
- spaCy verification
- System dependency warnings
- Error handling with pauses

### install.sh

- Python version check
- pip upgrade
- System packages via apt/yum/brew
- Main + optional dev/testing dependencies
- Playwright install
- spaCy verification
- Optional security tests
- Colored output and robust error handling

---

## Verification Steps

### 1. Check Python Version

```bash
python --version  # Should be 3.12 or higher
```

### 2. Verify Core Imports

```bash
python -c "import flask_socketio, spacy, playwright; print('? Core imports OK')"
```

### 3. Verify Playwright

```bash
playwright --version
```

### 4. Verify spaCy Model

```bash
python -m spacy validate
```

### 5. Run Security Tests

```bash
pytest webapp/tests/test_path_security.py -v
pytest webapp/tests/test_manual_correction_security.py -v
pytest webapp/tests/test_librarian_security.py -v
```

### 6. Run Security Audit

```bash
python security_audit.py
```

---

## Common Installation Scenarios

### Scenario 1: New Developer Setup

```bash
# Clone repository
git clone https://github.com/Basic-Nature/html_Parser_prototype
cd html_Parser_prototype

# Run automated installation
python install.py --dev --with-tests --run-tests

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Verify everything works
pytest webapp/tests/test_*_security.py -v
python security_audit.py
```

### Scenario 2: Production Deployment

```bash
# Clone repository
git clone https://github.com/Basic-Nature/html_Parser_prototype
cd html_Parser_prototype

# Install production dependencies only
pip install -r requirements.txt
playwright install chromium

# Configure environment
cp .env.example .env
# Edit .env with production settings

# Verify installation
python -c "import flask_socketio, spacy; print('OK')"
```

### Scenario 3: CI/CD Pipeline

```bash
# Install dependencies
pip install -r requirements.txt -r requirements-dev.txt
pip install pytest pytest-cov

# Run tests
pytest webapp/tests -v --cov=webapp.parser

# Run security checks
pytest webapp/tests/test_*_security.py -v
python security_audit.py

# Run linting
ruff check webapp/

# Run type checking
mypy webapp/
```

### Scenario 4: Security Testing Only

```bash
# Minimal installation for security testing
pip install pytest pytest-mock

# Run security tests
pytest webapp/tests/test_path_security.py -v
pytest webapp/tests/test_manual_correction_security.py -v
pytest webapp/tests/test_librarian_security.py -v

# Run security audit
python security_audit.py --output security_report.txt
```

---

## Troubleshooting Quick Reference

| Issue | Solution |
| --- | --- |
| PyTorch fails to install | Install separately: `pip install torch==2.7.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu` |
| pdf2image error | Install Poppler (Windows) or `sudo apt-get install poppler-utils` (Linux) |
| Tesseract not found | Install Tesseract and add to PATH |
| psycopg2 build fails | Use binary: `pip install psycopg2-binary` |
| Ghostscript not found | Install from <https://ghostscript.com> or package manager |
| Playwright command not found | Run `python -m playwright install chromium` |
| spaCy model missing | Run `python -m spacy download en_core_web_sm` |

---

## Next Steps After Installation

1. **Configure Environment**

   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

2. **Run Security Tests**

   ```bash
   pytest webapp/tests/test_*_security.py -v
   ```

3. **Run Security Audit**

   ```bash
   python security_audit.py --output security_report.txt
   ```

4. **Review Documentation**

   - SECURITY_PATTERNS.md — security best practices
   - PATH_SECURITY_PROGRESS.md — security implementation status
   - PHASE_2_COMPLETION_SUMMARY.md — phase 2 completion details

5. **Start Development**

   ```bash
   python webapp/Smart_Elections_Parser_Webapp.py
   ```

---

## Quick Reference Card

**Install (Normal):**

```bash
pip install -r requirements.txt && playwright install chromium
```

**Install (Dev):**

```bash
python install.py --dev --with-tests
```

**Verify:**

```bash
python -c "import flask_socketio, spacy; print('OK')"
```

**Test:**

```bash
pytest webapp/tests/test_*_security.py -v
```

**Audit:**

```bash
python security_audit.py
```

---

## Resources

- Detailed Guide: INSTALLATION_GUIDE.md
- Quick Reference: INSTALL_QUICKSTART.md
- Security Patterns: SECURITY_PATTERNS.md
- Python Script: install.py
- Windows Script: install.bat
- Unix Script: install.sh

---

**Created:** 2025-12-31

**Status:** Ready for use

**Python Required:** 3.12+
