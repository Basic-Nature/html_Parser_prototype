#!/usr/bin/env python3
"""
Ballot Lens Modern UI - Verification Script
Validates that the modern parser UI implementation is correctly in place.

Note: The modern UI is now integrated into /ballot_lens (consolidated architecture).
The /ballot_lens_modern route redirects to the main ballot_lens page.
"""

import re
import sys
from pathlib import Path

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def check_file_exists(filepath):
    """Check if file exists."""
    return Path(filepath).exists()

def check_pattern_in_file(filepath, pattern, description):
    """Check if regex pattern exists in file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        found = bool(re.search(pattern, content))
        return found, content if found else None
    except Exception as e:
        return False, str(e)

def print_check(passed, description, details=""):
    """Print formatted check result."""
    status = f"{GREEN}✓ PASS{RESET}" if passed else f"{RED}✗ FAIL{RESET}"
    print(f"{status} | {description}")
    if details:
        print(f"     └─ {details}")

def main():
    print(f"\n{BLUE}{'='*70}")
    print("Ballot Lens Modern UI - Deployment Verification")
    print(f"{'='*70}{RESET}\n")
    
    checks = []
    
    # ==========================================
    # Flask Route Verification
    # ==========================================
    print(f"{YELLOW}Flask Route (/ballot_lens){RESET}")
    
    flask_file = Path("webapp/Smart_Elections_Parser_Webapp.py")
    
    if not check_file_exists(flask_file):
        print_check(False, "Flask app file exists", f"Not found: {flask_file}")
        checks.append(False)
    else:
        print_check(True, "Flask app file exists", str(flask_file))
        checks.append(True)
        
        # Check for main ballot_lens route
        found, _ = check_pattern_in_file(
            flask_file,
            r'@app\.route\("/ballot_lens"',
            "Flask route decorator @app.route(/ballot_lens)"
        )
        print_check(found, "Main route present", "Pattern: @app.route('/ballot_lens')")
        checks.append(found)
        
        # Check for redirect route
        found, _ = check_pattern_in_file(
            flask_file,
            r'@app\.route\("/ballot_lens_modern"',
            "Redirect route @app.route(/ballot_lens_modern)"
        )
        print_check(found, "Redirect route present", "/ballot_lens_modern redirects to /ballot_lens")
        checks.append(found)
    
    # ==========================================
    # Template Verification
    # ==========================================
    print(f"\n{YELLOW}Template (ballot_lens.html){RESET}")
    
    html_file = Path("webapp/templates/ballot_lens.html")
    
    if not check_file_exists(html_file):
        print_check(False, "ballot_lens.html exists", f"Not found: {html_file}")
        checks.append(False)
    else:
        print_check(True, "ballot_lens.html exists", str(html_file))
        checks.append(True)
        
        # Check for critical UI elements
        found, _ = check_pattern_in_file(
            html_file,
            r'btnRunParser2',
            "Run button element"
        )
        print_check(found, "Run button present", "id=\"btnRunParser2\"")
        checks.append(found)
        
        # Check for sidebar elements
        found, _ = check_pattern_in_file(
            html_file,
            r'id="sidebar"',
            "Sidebar element"
        )
        print_check(found, "Sidebar present", "id=\"sidebar\"")
        checks.append(found)
    
    # ==========================================
    # JavaScript and CSS Verification
    # ==========================================
    print(f"\n{YELLOW}Assets (JS and CSS){RESET}")
    
    js_file = Path("webapp/static/js/ballot_lens_modern.js")
    css_file = Path("webapp/static/css/ballot_lens_modern.css")
    
    if not check_file_exists(js_file):
        print_check(False, "JavaScript file exists", f"Not found: {js_file}")
        checks.append(False)
    else:
        print_check(True, "JavaScript file exists", str(js_file))
        checks.append(True)
        
        # Check for loadRealData function
        found, _ = check_pattern_in_file(
            js_file,
            r'async function loadRealData\(\)',
            "loadRealData() function"
        )
        print_check(found, "loadRealData() function present", "async function loadRealData()")
        checks.append(found)
        
        # Check for DOMContentLoaded initialization
        found, _ = check_pattern_in_file(
            js_file,
            r"document\.addEventListener\(['\"]DOMContentLoaded",
            "DOMContentLoaded initialization"
        )
        print_check(found, "DOMContentLoaded hooks present", "Multiple initialization hooks")
        checks.append(found)
    
    print_check(check_file_exists(css_file), "CSS file exists", str(css_file))
    checks.append(check_file_exists(css_file))
    
    # ==========================================
    # Summary
    # ==========================================
    print(f"\n{BLUE}{'='*70}")
    print(f"Summary{RESET}")
    print(f"{'='*70}\n")
    
    passed = sum(checks)
    total = len(checks)
    percentage = int((passed / total) * 100) if total > 0 else 0
    
    print(f"Total Checks: {total}")
    print(f"Passed:       {passed} {GREEN}✓{RESET}")
    print(f"Failed:       {total - passed} {RED}✗{RESET}")
    print(f"Success Rate: {percentage}%\n")
    
    if all(checks):
        print(f"{GREEN}✓ All checks passed! Modern UI is deployed.{RESET}\n")
        print("Next steps:")
        print("  1. Run UI tests: python tools/ui_robust_check.py")
        print("  2. Start server: python -m webapp.Smart_Elections_Parser_Webapp")
        print("  3. Visit http://localhost:5000/ballot_lens\n")
        return 0
    else:
        print(f"{RED}✗ Some checks failed. Review the output above.{RESET}\n")
        print("For detailed UI behavior tests, run:")
        print("  python tools/ui_robust_check.py\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
