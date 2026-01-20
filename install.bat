@echo off
REM Smart Elections Parser - Windows Installation Script
REM Automates dependency installation on Windows

echo ======================================================================
echo    Smart Elections Parser - Windows Installation
echo ======================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.12 or higher from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [OK] Python found
echo.

REM Upgrade pip
echo ======================================================================
echo    Upgrading pip...
echo ======================================================================
python -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip
    pause
    exit /b 1
)
echo [OK] pip upgraded
echo.

REM Install main requirements
echo ======================================================================
echo    Installing production dependencies...
echo ======================================================================
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)
echo [OK] Production dependencies installed
echo.

REM Install dev requirements if requested
if "%1"=="--dev" (
    echo ======================================================================
    echo    Installing development dependencies...
    echo ======================================================================
    python -m pip install -r requirements-dev.txt
    if errorlevel 1 (
        echo [WARNING] Failed to install dev dependencies
    ) else (
        echo [OK] Development dependencies installed
    )
    echo.
)

REM Install testing dependencies if requested
if "%1"=="--with-tests" (
    echo ======================================================================
    echo    Installing testing dependencies...
    echo ======================================================================
    python -m pip install pytest pytest-cov pytest-mock
    if errorlevel 1 (
        echo [WARNING] Failed to install testing dependencies
    ) else (
        echo [OK] Testing dependencies installed
    )
    echo.
)

REM Install Playwright browsers
echo ======================================================================
echo    Installing Playwright browsers...
echo ======================================================================
playwright install chromium
if errorlevel 1 (
    echo [WARNING] Failed to install Playwright browsers
    echo You may need to run: playwright install chromium
) else (
    echo [OK] Playwright browsers installed
)
echo.

REM Verify spaCy model
echo ======================================================================
echo    Verifying spaCy model...
echo ======================================================================
python -m spacy validate >nul 2>&1
if errorlevel 1 (
    echo [INFO] Downloading spaCy model...
    python -m spacy download en_core_web_sm
) else (
    echo [OK] spaCy model is installed
)
echo.

REM System dependencies warning
echo ======================================================================
echo    System Dependencies Required
echo ======================================================================
echo.
echo Please ensure the following are installed on your system:
echo.
echo 1. Poppler (for PDF processing)
echo    Download: https://github.com/oschwartz10612/poppler-windows/releases
echo    Extract and add bin\ directory to PATH or set POPPLER_PATH
echo.
echo 2. Tesseract OCR
echo    Download: https://github.com/UB-Mannheim/tesseract/wiki
echo    Add to PATH after installation
echo.
echo 3. Ghostscript
echo    Download: https://ghostscript.com/releases/gsdnld.html
echo.

REM Final message
echo ======================================================================
echo    Installation Complete!
echo ======================================================================
echo.
echo Next steps:
echo   1. Configure your .env file
echo   2. Run security tests: pytest webapp\tests\test_path_security.py -v
echo   3. Run security audit: python security_audit.py
echo   4. Start application: python webapp\Smart_Elections_Parser_Webapp.py
echo.
echo Press any key to exit...
pause >nul
