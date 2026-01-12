#!/usr/bin/env python3
"""
Modern Parser UI - Deployment Verification Script
Validates that Steps 1-3 of the Quick Start implementation are correctly in place.
"""

import os
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
    print(f"Modern Parser UI - Deployment Verification")
    print(f"{'='*70}{RESET}\n")
    
    checks = []
    
    # ==========================================
    # Step 1: Flask Route Verification
    # ==========================================
    print(f"{YELLOW}STEP 1: Flask Route (/run_parser_modern){RESET}")
    
    flask_file = Path("webapp/Smart_Elections_Parser_Webapp.py")
    
    if not check_file_exists(flask_file):
        print_check(False, "Flask app file exists", f"Not found: {flask_file}")
        checks.append(False)
    else:
        print_check(True, "Flask app file exists", str(flask_file))
        checks.append(True)
        
        # Check for route decorator
        found, _ = check_pattern_in_file(
            flask_file,
            r'@app\.route\("/run_parser_modern"',
            "Flask route decorator @app.route(/run_parser_modern)"
        )
        print_check(found, "Route decorator present", f"Pattern: @app.route('/run_parser_modern')")
        checks.append(found)
        
        # Check for function definition
        found, _ = check_pattern_in_file(
            flask_file,
            r'def run_parser_modern\(\):',
            "Function run_parser_modern() defined"
        )
        print_check(found, "Function definition present", "def run_parser_modern():")
        checks.append(found)
        
        # Check for render_template call
        found, _ = check_pattern_in_file(
            flask_file,
            r'render_template\(\s*"run_parser_modern\.html"',
            "Template rendering"
        )
        print_check(found, "Template rendering present", 'render_template("run_parser_modern.html")')
        checks.append(found)
        
        # Check for error handling
        found, _ = check_pattern_in_file(
            flask_file,
            r'except Exception as e:.*logger\.error',
            "Error handling with logging"
        )
        print_check(found, "Error handling present", "Exception handling with logger")
        checks.append(found)
    
    # ==========================================
    # Step 2: Navigation Link Verification
    # ==========================================
    print(f"\n{YELLOW}STEP 2: Navigation Link (index.html){RESET}")
    
    html_file = Path("webapp/templates/index.html")
    
    if not check_file_exists(html_file):
        print_check(False, "index.html file exists", f"Not found: {html_file}")
        checks.append(False)
    else:
        print_check(True, "index.html file exists", str(html_file))
        checks.append(True)
        
        # Check for feature card
        found, _ = check_pattern_in_file(
            html_file,
            r'Parser Dashboard.*Beta',
            "Feature card with Beta label"
        )
        print_check(found, "Feature card present", "Parser Dashboard (Beta)")
        checks.append(found)
        
        # Check for URL helper
        found, _ = check_pattern_in_file(
            html_file,
            r'url_for\([\'"]run_parser_modern[\'"]\)',
            "URL helper function"
        )
        print_check(found, "URL helper present", "{{ url_for('run_parser_modern') }}")
        checks.append(found)
        
        # Check for description
        found, _ = check_pattern_in_file(
            html_file,
            r'real-time results grid|file preview|advanced filtering',
            "Feature description"
        )
        print_check(found, "Feature description present", "Describes dashboard capabilities")
        checks.append(found)
    
    # ==========================================
    # Step 3: JavaScript Data Integration
    # ==========================================
    print(f"\n{YELLOW}STEP 3: Real Data Integration (run_parser_modern.js){RESET}")
    
    js_file = Path("webapp/static/js/run_parser_modern.js")
    
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
        
        # Check for API fetch
        found, _ = check_pattern_in_file(
            js_file,
            r'fetch\([\'"]*/api/warehouse_election_results',
            "API endpoint fetch"
        )
        print_check(found, "API fetch present", "/api/warehouse_election_results endpoint")
        checks.append(found)
        
        # Check for schema transformation
        found, _ = check_pattern_in_file(
            js_file,
            r'items\.map\(\(item.*idx\).*=>\s*\({',
            "Schema transformation"
        )
        print_check(found, "Schema transformation present", "items.map(...) transformation")
        checks.append(found)
        
        # Check for fallback
        found, _ = check_pattern_in_file(
            js_file,
            r'loadSampleData\(\)',
            "Fallback to sample data"
        )
        print_check(found, "Fallback present", "loadSampleData() on error")
        checks.append(found)
        
        # Check for initialization call
        found, _ = check_pattern_in_file(
            js_file,
            r'document\.addEventListener\([\'"]DOMContentLoaded[\'"].*loadRealData\(\)',
            "Initialization hook"
        )
        print_check(found, "Initialization present", "loadRealData() called on DOMContentLoaded")
        checks.append(found)
    
    # ==========================================
    # Template Files Verification
    # ==========================================
    print(f"\n{YELLOW}Supporting Files{RESET}")
    
    modern_html = Path("webapp/templates/run_parser_modern.html")
    modern_css = Path("webapp/static/css/run_parser_modern.css")
    
    print_check(check_file_exists(modern_html), "run_parser_modern.html exists", str(modern_html))
    checks.append(check_file_exists(modern_html))
    
    print_check(check_file_exists(modern_css), "run_parser_modern.css exists", str(modern_css))
    checks.append(check_file_exists(modern_css))
    
    # ==========================================
    # Summary
    # ==========================================
    print(f"\n{BLUE}{'='*70}")
    print(f"Summary{RESET}")
    print(f"{'='*70}\n")
    
    passed = sum(checks)
    total = len(checks)
    percentage = int((passed / total) * 100)
    
    print(f"Total Checks: {total}")
    print(f"Passed:       {passed} {GREEN}✓{RESET}")
    print(f"Failed:       {total - passed} {RED}✗{RESET}")
    print(f"Success Rate: {percentage}%\n")
    
    if all(checks):
        print(f"{GREEN}✓ All checks passed! Implementation is complete.{RESET}\n")
        print(f"Next steps:")
        print(f"  1. Start Flask app: python -m flask run")
        print(f"  2. Visit http://localhost:5000/run_parser_modern")
        print(f"  3. Follow MODERN_UI_ROLLOUT_TESTING.md for full test suite\n")
        return 0
    else:
        print(f"{RED}✗ Some checks failed. Review the output above.{RESET}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
