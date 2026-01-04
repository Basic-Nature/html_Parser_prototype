# Quick Installation Reference

## TL;DR - Fast Installation

### Production (Normal) Use

```bash
pip install -r requirements.txt
playwright install chromium
```

### Development Use

```bash
pip install -r requirements.txt -r requirements-dev.txt
pip install pytest pytest-cov pytest-mock
playwright install chromium
```

---

## Automated Installation

### Windows

```batch
# Run the batch script
install.bat

# Or for development
install.bat --dev

# Or with testing
install.bat --with-tests
```

### Linux/macOS

```bash
# Make executable
chmod +x install.sh

# Run the script
./install.sh

# Or for development
./install.sh --dev

# Or with testing
./install.sh --with-tests

# Skip system dependencies (if already installed)
./install.sh --skip-system-deps
```

### Python Script (Cross-platform)

```bash
# Normal installation
python install.py

# Development installation
python install.py --dev

# With tests
python install.py --with-tests

# Run security tests after install
python install.py --run-tests

# Skip Playwright browser installation
python install.py --skip-playwright
```

---

## Manual Installation

### Step 1: Virtual Environment (Recommended)

```bash
# Create
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/macOS)
source venv/bin/activate
```

### Step 2: Upgrade pip

```bash
python -m pip install --upgrade pip
```

### Step 3: Install Dependencies

```bash
# Production
pip install -r requirements.txt

# Development
pip install -r requirements.txt -r requirements-dev.txt

# Testing
pip install pytest pytest-cov pytest-mock
```

### Step 4: Install Playwright

```bash
playwright install chromium
```

### Step 5: Verify Installation

```bash
python -c "import flask_socketio, spacy, playwright; print('? OK')"
```

---

## System Dependencies

### Windows (System Dependencies)

Download and install:

1. **Poppler**: <https://github.com/oschwartz10612/poppler-windows/releases>
2. **Tesseract**: <https://github.com/UB-Mannheim/tesseract/wiki>
3. **Ghostscript**: <https://ghostscript.com/releases/gsdnld.html>

### Linux (System Dependencies)

```bash
sudo apt-get install poppler-utils tesseract-ocr ghostscript
```

### macOS (System Dependencies)

```bash
brew install poppler tesseract ghostscript
```

---

## Testing Installation

### Run Security Tests

```bash
pytest webapp/tests/test_path_security.py -v
pytest webapp/tests/test_manual_correction_security.py -v
pytest webapp/tests/test_librarian_security.py -v
```

### Run Security Audit

```bash
python security_audit.py
python security_audit.py --output report.txt
```

### Run All Tests

```bash
pytest webapp/tests -v
```

---

## Common Issues

### PyTorch Installation Fails

```bash
# Install separately with CPU-only version
pip install torch==2.7.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### pdf2image Error

```bash
# Windows: Set POPPLER_PATH
set POPPLER_PATH=C:\path\to\poppler\bin

# Linux/macOS: Install system package
sudo apt-get install poppler-utils  # Linux
brew install poppler                 # macOS
```

### Tesseract Not Found

```bash
# Windows: Add to PATH after installation
# Linux
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract
```

### psycopg2 Build Fails

```bash
pip install psycopg2-binary
```

---

## Dependency Categories

### Core (Required)

- Flask, Playwright, Pandas, spaCy, PyTorch

### File Processing

- pdf2image, PyMuPDF, camelot-py, pytesseract

### Database

- SQLAlchemy, psycopg2

### NLP

- spaCy

### Vision

- torch, torchvision

### Testing

- pytest, pytest-cov, pytest-mock

### Development

- ruff, mypy, pre-commit

---

## Verification Commands

```bash
# Python version
python --version  # Should be 3.12+

# Core imports
python -c "import flask_socketio, spacy, playwright; print('OK')"

# Playwright
playwright --version

# spaCy model
python -m spacy validate

# Run security tests
pytest webapp/tests/test_*_security.py -v

# Security audit
python security_audit.py
```

---

## Environment Setup

```bash
# Copy example .env
cp .env.example .env

# Edit .env with your settings
# Required:
#   - Database credentials
#   - POPPLER_PATH (Windows)
#   - API keys (if needed)
```

---

## Quick Commands

```bash
# Install everything (production)
pip install -r requirements.txt && playwright install chromium

# Install everything (development)
pip install -r requirements.txt -r requirements-dev.txt && \
pip install pytest pytest-cov pytest-mock && \
playwright install chromium

# Run Flask app (development)
python webapp/Smart_Elections_Parser_Webapp.py

# Run CLI parser
python webapp/parser/html_election_parser.py

# Run automated checks
python automate.py
```
