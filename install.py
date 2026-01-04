#!/usr/bin/env python3
"""
Smart Elections Parser - Installation Script
Automates dependency installation with proper error handling
"""
import os
import platform
import subprocess
import sys
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    """Print formatted header"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text:^70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.ENDC}\n")


def print_success(text):
    """Print success message"""
    print(f"{Colors.OKGREEN}? {text}{Colors.ENDC}")


def print_warning(text):
    """Print warning message"""
    print(f"{Colors.WARNING}? {text}{Colors.ENDC}")


def print_error(text):
    """Print error message"""
    print(f"{Colors.FAIL}? {text}{Colors.ENDC}")


def print_info(text):
    """Print info message"""
    print(f"{Colors.OKCYAN}? {text}{Colors.ENDC}")


def check_python_version():
    """Check if Python version meets requirements"""
    print_header("Checking Python Version")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    print(f"Current Python version: {version_str}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 12):
        print_error("Python 3.12 or higher is required!")
        print_info("Please upgrade Python: https://www.python.org/downloads/")
        return False
    
    print_success("Python version is compatible")
    return True


def check_pip():
    """Check if pip is available and upgrade it"""
    print_header("Checking pip")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True)
        print_success("pip is available")
        
        print_info("Upgrading pip...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
                      check=True, capture_output=True)
        print_success("pip upgraded successfully")
        return True
    except subprocess.CalledProcessError:
        print_error("pip is not available!")
        return False


def install_requirements(dev_mode=False):
    """Install requirements from requirements.txt"""
    print_header(f"Installing {'Development' if dev_mode else 'Production'} Dependencies")
    
    project_root = Path(__file__).parent
    requirements_file = project_root / "requirements.txt"
    dev_requirements_file = project_root / "requirements-dev.txt"
    
    if not requirements_file.exists():
        print_error(f"requirements.txt not found at {requirements_file}")
        return False
    
    # Install main requirements
    print_info("Installing main dependencies...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
            check=True,
            capture_output=False
        )
        print_success("Main dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install main dependencies: {e}")
        return False
    
    # Install dev requirements if requested
    if dev_mode and dev_requirements_file.exists():
        print_info("Installing development dependencies...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(dev_requirements_file)],
                check=True,
                capture_output=False
            )
            print_success("Development dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            print_error(f"Failed to install dev dependencies: {e}")
            return False
    
    return True


def install_testing_deps():
    """Install testing dependencies"""
    print_header("Installing Testing Dependencies")
    
    test_packages = ["pytest", "pytest-cov", "pytest-mock"]
    
    print_info(f"Installing: {', '.join(test_packages)}")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install"] + test_packages,
            check=True,
            capture_output=False
        )
        print_success("Testing dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install testing dependencies: {e}")
        return False


def install_playwright_browsers():
    """Install Playwright browsers"""
    print_header("Installing Playwright Browsers")
    
    print_info("Installing Chromium browser...")
    try:
        subprocess.run(
            ["playwright", "install", "chromium"],
            check=True,
            capture_output=False
        )
        print_success("Playwright browsers installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install Playwright browsers: {e}")
        print_warning("You may need to install Playwright browsers manually:")
        print_warning("  playwright install chromium")
        return False
    except FileNotFoundError:
        print_warning("Playwright command not found in PATH")
        print_info("Trying alternative installation method...")
        try:
            subprocess.run(
                [sys.executable, "-m", "playwright", "install", "chromium"],
                check=True,
                capture_output=False
            )
            print_success("Playwright browsers installed successfully")
            return True
        except Exception as e:
            print_error(f"Failed to install Playwright browsers: {e}")
            return False


def verify_spacy_model():
    """Verify spaCy model is installed"""
    print_header("Verifying spaCy Model")
    
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
            print_success("spaCy model 'en_core_web_sm' is installed")
            return True
        except OSError:
            print_warning("spaCy model not found, attempting to download...")
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                check=True
            )
            print_success("spaCy model downloaded successfully")
            return True
    except ImportError:
        print_error("spaCy is not installed!")
        return False
    except Exception as e:
        print_error(f"Failed to verify spaCy model: {e}")
        return False


def check_system_dependencies():
    """Check for required system dependencies"""
    print_header("Checking System Dependencies")
    
    system = platform.system()
    
    if system == "Windows":
        print_info("Windows detected")
        print_warning("Please ensure the following are installed:")
        print_warning("  - Poppler (for PDF processing)")
        print_warning("    Download: https://github.com/oschwartz10612/poppler-windows/releases")
        print_warning("  - Tesseract OCR")
        print_warning("    Download: https://github.com/UB-Mannheim/tesseract/wiki")
        print_warning("  - Ghostscript")
        print_warning("    Download: https://ghostscript.com/releases/gsdnld.html")
    elif system == "Linux":
        print_info("Linux detected")
        print_info("Install system dependencies with:")
        print_info("  sudo apt-get install poppler-utils tesseract-ocr ghostscript")
    elif system == "Darwin":
        print_info("macOS detected")
        print_info("Install system dependencies with:")
        print_info("  brew install poppler tesseract ghostscript")
    
    return True


def verify_installation():
    """Verify that key packages are importable"""
    print_header("Verifying Installation")
    
    packages_to_test = [
        "flask_socketio",
        "spacy",
        "playwright",
        "pandas",
        "sqlalchemy",
    ]
    
    all_ok = True
    for package in packages_to_test:
        try:
            __import__(package)
            print_success(f"{package} imports successfully")
        except ImportError as e:
            print_error(f"{package} failed to import: {e}")
            all_ok = False
    
    return all_ok


def run_security_tests():
    """Run security tests to verify security features"""
    print_header("Running Security Tests")
    
    print_info("Running path security tests...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "webapp/tests/test_path_security.py", "-v"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print_success("Security tests passed!")
            return True
        else:
            print_warning("Some security tests failed")
            print(result.stdout)
            return False
    except Exception as e:
        print_warning(f"Could not run security tests: {e}")
        print_info("You can run them manually with:")
        print_info("  pytest webapp/tests/test_*_security.py -v")
        return False


def main():
    """Main installation flow"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Smart Elections Parser Installation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Install development dependencies"
    )
    parser.add_argument(
        "--with-tests",
        action="store_true",
        help="Install testing dependencies"
    )
    parser.add_argument(
        "--skip-playwright",
        action="store_true",
        help="Skip Playwright browser installation"
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Skip installation verification"
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run security tests after installation"
    )
    
    args = parser.parse_args()
    
    print_header("Smart Elections Parser - Installation")
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    if not check_pip():
        sys.exit(1)
    
    # Check system dependencies
    check_system_dependencies()
    
    # Install Python dependencies
    if not install_requirements(dev_mode=args.dev):
        print_error("Failed to install dependencies!")
        sys.exit(1)
    
    # Install testing dependencies if requested
    if args.with_tests or args.dev:
        install_testing_deps()
    
    # Install Playwright browsers
    if not args.skip_playwright:
        install_playwright_browsers()
    
    # Verify spaCy model
    verify_spacy_model()
    
    # Verify installation
    if not args.skip_verification:
        if not verify_installation():
            print_warning("Some packages failed to import")
            print_info("Installation may be incomplete")
    
    # Run tests if requested
    if args.run_tests:
        run_security_tests()
    
    # Final summary
    print_header("Installation Complete!")
    print_success("All dependencies have been installed")
    print_info("\nNext steps:")
    print_info("  1. Configure your .env file")
    print_info("  2. Run security tests: pytest webapp/tests/test_*_security.py -v")
    print_info("  3. Run security audit: python security_audit.py")
    print_info("  4. Start the application: python webapp/Smart_Elections_Parser_Webapp.py")


if __name__ == "__main__":
    main()
