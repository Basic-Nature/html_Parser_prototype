#!/bin/bash
# Smart Elections Parser - Linux/macOS Installation Script
# Automates dependency installation on Unix-like systems

set -e  # Exit on error

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${BLUE}======================================================================${NC}"
    echo -e "${BLUE}   $1${NC}"
    echo -e "${BLUE}======================================================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}? $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}? $1${NC}"
}

print_error() {
    echo -e "${RED}? $1${NC}"
}

print_info() {
    echo -e "${BLUE}? $1${NC}"
}

# Parse arguments
DEV_MODE=false
WITH_TESTS=false
SKIP_SYSTEM_DEPS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)
            DEV_MODE=true
            shift
            ;;
        --with-tests)
            WITH_TESTS=true
            shift
            ;;
        --skip-system-deps)
            SKIP_SYSTEM_DEPS=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dev] [--with-tests] [--skip-system-deps]"
            exit 1
            ;;
    esac
done

print_header "Smart Elections Parser - Installation"

# Check Python version
print_header "Checking Python Version"
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
print_info "Python version: $PYTHON_VERSION"

# Check if version is >= 3.12
REQUIRED_VERSION="3.12"
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 12) else 1)"; then
    print_error "Python 3.12 or higher is required!"
    print_info "Please upgrade Python: https://www.python.org/downloads/"
    exit 1
fi

print_success "Python version is compatible"

# Upgrade pip
print_header "Upgrading pip"
python3 -m pip install --upgrade pip
print_success "pip upgraded successfully"

# Install system dependencies
if [ "$SKIP_SYSTEM_DEPS" = false ]; then
    print_header "Installing System Dependencies"
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        print_info "Linux detected - Installing system packages"
        
        if command -v apt-get &> /dev/null; then
            print_info "Using apt-get package manager"
            sudo apt-get update
            sudo apt-get install -y poppler-utils tesseract-ocr ghostscript
            print_success "System dependencies installed"
        elif command -v yum &> /dev/null; then
            print_info "Using yum package manager"
            sudo yum install -y poppler-utils tesseract ghostscript
            print_success "System dependencies installed"
        else
            print_warning "Unknown package manager - please install manually:"
            print_warning "  - poppler-utils"
            print_warning "  - tesseract-ocr"
            print_warning "  - ghostscript"
        fi
        
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        print_info "macOS detected - Installing system packages"
        
        if command -v brew &> /dev/null; then
            brew install poppler tesseract ghostscript
            print_success "System dependencies installed"
        else
            print_warning "Homebrew not found - please install manually:"
            print_warning "  brew install poppler tesseract ghostscript"
        fi
    else
        print_warning "Unknown OS - please install system dependencies manually"
    fi
else
    print_info "Skipping system dependencies installation"
fi

# Install Python requirements
print_header "Installing Production Dependencies"
python3 -m pip install -r requirements.txt
print_success "Production dependencies installed"

# Install dev requirements if requested
if [ "$DEV_MODE" = true ]; then
    print_header "Installing Development Dependencies"
    if [ -f "requirements-dev.txt" ]; then
        python3 -m pip install -r requirements-dev.txt
        print_success "Development dependencies installed"
    else
        print_warning "requirements-dev.txt not found"
    fi
fi

# Install testing dependencies if requested
if [ "$WITH_TESTS" = true ] || [ "$DEV_MODE" = true ]; then
    print_header "Installing Testing Dependencies"
    python3 -m pip install pytest pytest-cov pytest-mock
    print_success "Testing dependencies installed"
fi

# Install Playwright browsers
print_header "Installing Playwright Browsers"
if command -v playwright &> /dev/null; then
    playwright install chromium
    print_success "Playwright browsers installed"
else
    print_warning "Playwright command not found, trying alternative..."
    python3 -m playwright install chromium || print_warning "Failed to install Playwright browsers"
fi

# Verify spaCy model
print_header "Verifying spaCy Model"
if python3 -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null; then
    print_success "spaCy model is installed"
else
    print_info "Downloading spaCy model..."
    python3 -m spacy download en_core_web_sm
    print_success "spaCy model downloaded"
fi

# Verify installation
print_header "Verifying Installation"
python3 -c "import flask_socketio, spacy, playwright; print('OK')" && \
    print_success "Core packages import successfully" || \
    print_error "Some packages failed to import"

# Run security tests if testing dependencies were installed
if [ "$WITH_TESTS" = true ] || [ "$DEV_MODE" = true ]; then
    print_header "Running Security Tests"
    if python3 -m pytest webapp/tests/test_path_security.py -v; then
        print_success "Security tests passed!"
    else
        print_warning "Some security tests failed"
    fi
fi

# Final summary
print_header "Installation Complete!"
print_success "All dependencies have been installed"
echo ""
print_info "Next steps:"
print_info "  1. Configure your .env file"
print_info "  2. Run security tests: pytest webapp/tests/test_*_security.py -v"
print_info "  3. Run security audit: python3 security_audit.py"
print_info "  4. Start application: python3 webapp/Smart_Elections_Parser_Webapp.py"
echo ""
