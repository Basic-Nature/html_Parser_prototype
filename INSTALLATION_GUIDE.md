# Python Dependencies Installation Guide

## Quick Start

### For Normal/Production Use
```bash
pip install -r requirements.txt
```

### For Development (includes testing, linting, type checking)
```bash
pip install -r requirements.txt -r requirements-dev.txt
```

---

## Detailed Installation Instructions

### Prerequisites

**Python Version Required:** 3.12+

**System Dependencies (for PDF processing):**

**Windows:**
```powershell
# Install Poppler for pdf2image (faster PDF processing)
# Download from: https://github.com/oschwartz10612/poppler-windows/releases
# Add to PATH or set POPPLER_PATH environment variable
```

**Linux/Ubuntu:**
```bash
sudo apt-get update
sudo apt-get install -y poppler-utils tesseract-ocr ghostscript
```

**macOS:**
```bash
brew install poppler tesseract ghostscript
```

---

## Step-by-Step Installation

### 1. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate

# Linux/macOS:
source venv/bin/activate
```

### 2. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 3. Install Dependencies

**Option A: Production/Normal Use**
```bash
pip install -r requirements.txt
```

**Option B: Development Environment**
```bash
# Install all dependencies including dev tools
pip install -r requirements.txt -r requirements-dev.txt

# Or install testing dependencies separately
pip install -r requirements.txt
pip install pytest pytest-cov pytest-mock
```

**Option C: With Optional Features**
```bash
# Install with Selenium (for CAPTCHA visual bypass)
pip install -r requirements.txt seleniumbase>=4.40.8

# Or use setuptools optional dependencies
pip install .[selenium]
```

### 4. Install Playwright Browsers (Required)

```bash
playwright install chromium
```

### 5. Download spaCy Language Model

The spaCy model is included in requirements.txt, but you can verify:

```bash
python -m spacy download en_core_web_sm
```

---

## Dependency Categories

### Core Production Dependencies

**Web Scraping & Browser Automation:**
- `playwright>=1.54.0` - Browser automation
- `selectolax>=0.3.32` - Fast HTML parsing
- `langdetect>=1.0.9` - Language detection

**Web Framework:**
- `flask-socketio>=5.5.1` - Real-time communication
- `gunicorn>=23.0.0` - Production WSGI server
- `eventlet>=0.40.2` - Async networking

**File Processing:**
- `pandas>=2.3.1` - Data manipulation
- `pytesseract>=0.3.13` - OCR capabilities
- `pdf2image>=1.17.0` - PDF to image conversion
- `PyMuPDF>=1.26.5` - PDF processing
- `camelot-py[cv]>=1.0.9` - Table extraction from PDFs
- `pdfminer.six>=20221105` - PDF text extraction
- `opencv-python>=4.12.0.88` - Computer vision

**Database:**
- `sqlalchemy>=2.0.42` - ORM
- `psycopg2>=2.9.10` - PostgreSQL adapter

**Data Export:**
- `openpyxl>=3.1.5` - Excel file handling

**NLP & Machine Learning:**
- `spacy>=3.8.7` - NLP framework
- `sentence_transformers>=5.1.0` - Embeddings
- `torch==2.7.1+cpu` - PyTorch (CPU version)
- `nltk>=3.9.1` - Natural language toolkit
- `spacy-lookups-data>=1.0.5` - spaCy data

**Utilities:**
- `rich>=14.1.0` - Terminal formatting
- `rapidfuzz>=3.13.0` - String matching
- `matplotlib>=3.10.5` - Visualization
- `orjson>=3.11.1` - Fast JSON parsing
- `azure_identity>=1.24.0` - Azure authentication
- `ghostscript>=0.8.1` - PostScript processing

### Development Dependencies

**Code Quality:**
- `ruff==0.6.9` - Fast linter and formatter
- `mypy==1.11.2` - Static type checker
- `types-requests==2.32.0.20241016` - Type stubs

**Git Hooks:**
- `pre-commit==3.8.0` - Git pre-commit hooks

### Testing Dependencies (Not in requirements files)

Add these for testing:
```bash
pip install pytest pytest-cov pytest-mock
```

---

## Installation Scenarios

### Scenario 1: Local Development

```bash
# Clone repository
git clone https://github.com/Basic-Nature/html_Parser_prototype
cd html_Parser_prototype

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# or: source venv/bin/activate  # Linux/macOS

# Install all dependencies
pip install -r requirements.txt -r requirements-dev.txt

# Install testing tools
pip install pytest pytest-cov pytest-mock

# Install Playwright browsers
playwright install chromium

# Verify installation
python -c "import flask_socketio; import spacy; print('? Core imports successful')"
```

### Scenario 2: Production Deployment

```bash
# Install only production dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install chromium

# Configure environment
cp .env.example .env
# Edit .env with your settings
```

### Scenario 3: CI/CD Pipeline

```bash
# Minimal install for testing
pip install -r requirements.txt -r requirements-dev.txt
pip install pytest pytest-cov

# Run tests
pytest webapp/tests -v

# Run linting
ruff check webapp/

# Run type checking
mypy webapp/
```

### Scenario 4: Security Testing Only

```bash
# Install base + security test dependencies
pip install -r requirements.txt
pip install pytest pytest-mock

# Run security tests
pytest webapp/tests/test_*_security.py -v

# Run security audit
python security_audit.py
```

---

## Verification Commands

After installation, verify everything works:

```bash
# Check Python version
python --version  # Should be 3.12+

# Verify core imports
python -c "import flask_socketio, spacy, playwright; print('? Core imports OK')"

# Verify Playwright
playwright --version

# Verify spaCy model
python -m spacy validate

# Run security tests
pytest webapp/tests/test_path_security.py -v

# Run security audit
python security_audit.py --dir webapp/parser
```

---

## Common Installation Issues

### Issue 1: PyTorch Installation Fails

**Solution:**
```bash
# Install PyTorch separately first
pip install torch==2.7.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
# Then install rest of requirements
pip install -r requirements.txt
```

### Issue 2: pdf2image "Unable to get page count"

**Solution:**
```bash
# Windows: Install Poppler
# Download from: https://github.com/oschwartz10612/poppler-windows/releases
# Extract and add bin/ directory to PATH
# Or set POPPLER_PATH environment variable

# Linux:
sudo apt-get install poppler-utils

# macOS:
brew install poppler
```

### Issue 3: Tesseract Not Found

**Solution:**
```bash
# Windows: Download installer
# https://github.com/UB-Mannheim/tesseract/wiki
# Add to PATH

# Linux:
sudo apt-get install tesseract-ocr

# macOS:
brew install tesseract
```

### Issue 4: psycopg2 Build Fails

**Solution:**
```bash
# Use binary wheel instead
pip install psycopg2-binary
```

### Issue 5: Ghostscript Not Found

**Solution:**
```bash
# Windows: Download installer
# https://ghostscript.com/releases/gsdnld.html

# Linux:
sudo apt-get install ghostscript

# macOS:
brew install ghostscript
```

---

## Dependency Update Strategy

### Check for Updates
```bash
pip list --outdated
```

### Update Specific Package
```bash
pip install --upgrade package-name
```

### Update All Dependencies (Careful!)
```bash
pip install --upgrade -r requirements.txt
```

### Freeze Current Environment
```bash
pip freeze > requirements-frozen.txt
```

---

## Minimal Installation (Testing Core Security Only)

If you only want to run security tests:

```bash
# Create venv
python -m venv venv
venv\Scripts\activate

# Install minimal dependencies
pip install pytest pytest-mock

# Run security tests
pytest webapp/tests/test_path_security.py -v
pytest webapp/tests/test_manual_correction_security.py -v
pytest webapp/tests/test_librarian_security.py -v
```

---

## Environment Configuration

After installing dependencies, set up your environment:

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings
# Required variables:
# - DB credentials (if using database)
# - POPPLER_PATH (Windows, for PDF processing)
# - Any API keys for external services
```

---

## Quick Commands Reference

```bash
# Normal installation
pip install -r requirements.txt

# Dev installation
pip install -r requirements.txt -r requirements-dev.txt

# With testing
pip install -r requirements.txt pytest pytest-cov pytest-mock

# Verify installation
python -c "import flask_socketio, spacy; print('OK')"

# Run security tests
pytest webapp/tests/test_*_security.py -v

# Run security audit
python security_audit.py
```

---

## Platform-Specific Notes

### Windows
- Use `venv\Scripts\activate` to activate virtual environment
- Install Poppler manually for PDF processing
- May need Visual C++ build tools for some packages

### Linux
- Use `source venv/bin/activate` to activate virtual environment
- Install system packages: `poppler-utils`, `tesseract-ocr`, `ghostscript`
- May need `python3-dev` for building some packages

### macOS
- Use `source venv/bin/activate` to activate virtual environment
- Use Homebrew for system dependencies
- May need Xcode command line tools

---

## Docker Installation (Alternative)

If you prefer Docker:

```dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    ghostscript \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright
RUN playwright install chromium --with-deps
```

---

**Last Updated:** 2025-12-31
