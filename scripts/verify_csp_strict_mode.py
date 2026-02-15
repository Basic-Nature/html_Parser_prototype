#!/usr/bin/env python3
"""
CSP Local Verification Script
==============================
Validates that STRICT CSP mode works correctly in a local environment.
Run this BEFORE deploying to Azure.

Usage:
    python scripts/verify_csp_strict_mode.py [--port 5000] [--timeout 10]
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import requests


# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str) -> None:
    """Print a header section."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*60}")
    print(f"{text}")
    print(f"{'='*60}{Colors.ENDC}\n")


def print_check(passed: bool, message: str) -> None:
    """Print a check result."""
    status = f"{Colors.OKGREEN}✓ PASS{Colors.ENDC}" if passed else f"{Colors.FAIL}✗ FAIL{Colors.ENDC}"
    print(f"  [{status}] {message}")


def check_vendor_files() -> Tuple[bool, List[str]]:
    """Check that all required vendor files exist locally."""
    print_header("1. Checking Local Vendor Files")
    
    vendor_path = Path("webapp/static/vendor")
    required_files = [
        "bootstrap.min.css",
        "bootstrap.bundle.min.js",
        "chart.umd.js",
        "socket.io-4.7.5.min.js",
    ]
    
    if not vendor_path.exists():
        print(f"{Colors.FAIL}✗ Vendor directory not found: {vendor_path}{Colors.ENDC}")
        return False, []
    
    missing = []
    for filename in required_files:
        filepath = vendor_path / filename
        exists = filepath.exists()
        print_check(exists, f"{filepath}")
        if not exists:
            missing.append(filename)
    
    return len(missing) == 0, missing


def set_environment_variables() -> Dict[str, str]:
    """Set up environment variables for STRICT CSP mode."""
    print_header("2. Setting STRICT CSP Environment Variables")
    
    env = os.environ.copy()
    env['CSP_MODE'] = 'STRICT'
    env['ALLOW_STYLE_ATTR'] = '0'
    env['FLASK_ENV'] = 'development'
    
    print_check(True, f"CSP_MODE = {env['CSP_MODE']}")
    print_check(True, f"ALLOW_STYLE_ATTR = {env['ALLOW_STYLE_ATTR']}")
    print(f"  {Colors.OKBLUE}ℹ Note: Any other existing env vars are preserved{Colors.ENDC}")
    
    return env


def start_flask_server(env: Dict[str, str], port: int = 5000) -> subprocess.Popen:
    """Start the Flask development server in STRICT CSP mode."""
    print_header(f"3. Starting Flask Server (port {port})")
    
    print("  Starting: python -m webapp.Smart_Elections_Parser_Webapp")
    print(f"  {Colors.OKBLUE}ℹ Wait for 'Running on' message...{Colors.ENDC}\n")
    
    process = subprocess.Popen(
        [sys.executable, "-m", "webapp.Smart_Elections_Parser_Webapp"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Wait for server to start
    start_time = time.time()
    while time.time() - start_time < 15:
        line = process.stdout.readline()
        if line:
            print(f"    {line.rstrip()}")
        if "Running on" in line or "WARNING" in line:
            if "Running on" in line:
                print(f"  {Colors.OKGREEN}✓ Flask server started{Colors.ENDC}")
                return process
        if process.poll() is not None:
            print(f"  {Colors.FAIL}✗ Flask process exited unexpectedly{Colors.ENDC}")
            return None
    
    print(f"  {Colors.FAIL}✗ Server failed to start within 15 seconds{Colors.ENDC}")
    return None


def fetch_csp_header(base_url: str = "http://localhost:5000") -> Tuple[bool, str]:
    """Fetch and parse the CSP header from the server."""
    print_header("4. Checking Content-Security-Policy Header")
    
    try:
        response = requests.get(base_url, timeout=5)
        csp = response.headers.get('Content-Security-Policy', '')
        
        if not csp:
            print(f"  {Colors.FAIL}✗ No CSP header found in response{Colors.ENDC}")
            return False, ""
        
        print(f"  {Colors.OKGREEN}✓ CSP header present{Colors.ENDC}")
        print(f"\n  Full CSP Header:\n  {csp}\n")
        
        return True, csp
    except Exception as e:
        print(f"  {Colors.FAIL}✗ Failed to fetch CSP header: {e}{Colors.ENDC}")
        return False, ""


def validate_csp_content(csp: str) -> List[Tuple[bool, str]]:
    """Validate the CSP header content for STRICT mode requirements."""
    print_header("5. Validating STRICT CSP Requirements")
    
    checks = []
    
    # Check 1: No external CDNs
    has_cdn = any(cdn in csp for cdn in ["cdn.jsdelivr.net", "cdn.socket.io", "cdnjs"])
    check = not has_cdn
    checks.append((check, "No external CDN references (cdn.jsdelivr.net, cdn.socket.io)"))
    print_check(check, checks[-1][1])
    
    # Check 2: Has nonce for scripts
    has_nonce = "'nonce-" in csp
    check = has_nonce
    checks.append((check, "Contains nonce for inline scripts ('nonce-...')"))
    print_check(check, checks[-1][1])
    if has_nonce:
        nonce_part = [part for part in csp.split(";") if "nonce-" in part][0].strip()
        print(f"    {Colors.OKBLUE}ℹ Found: {nonce_part}{Colors.ENDC}")
    
    # Check 3: script-src 'self'
    has_self = "script-src 'self'" in csp or "script-src" in csp and "'self'" in csp
    check = has_self
    checks.append((check, "script-src includes 'self'"))
    print_check(check, checks[-1][1])
    
    # Check 4: style-src 'self'
    has_style = "style-src" in csp and "'self'" in csp
    check = has_style
    checks.append((check, "style-src includes 'self' (no external)"))
    print_check(check, checks[-1][1])
    
    # Check 5: No inline styles allowed
    has_no_style_attr = "style-src-attr 'none'" in csp
    check = has_no_style_attr
    checks.append((check, "style-src-attr set to 'none' (no inline styles)"))
    print_check(check, checks[-1][1])
    
    # Check 6: WebSocket support for real-time
    has_ws = "ws:" in csp or "wss:" in csp
    check = has_ws
    checks.append((check, "WebSocket support enabled (ws: or wss:)"))
    print_check(check, checks[-1][1])
    
    return checks


def test_page_load(base_url: str = "http://localhost:5000") -> Tuple[bool, str]:
    """Load a page and check for CSP violations in response."""
    print_header("6. Testing Page Load & Bootstrap Styling")
    
    try:
        response = requests.get(base_url, timeout=5)
        
        # Check HTTP status
        check = response.status_code == 200
        print_check(check, f"HTTP Status: {response.status_code}")
        
        # Check for Bootstrap CSS reference in HTML
        html = response.text
        has_bootstrap_css = "bootstrap" in html.lower()
        check = has_bootstrap_css
        print_check(check, "Bootstrap CSS referenced in HTML")
        
        # Check for Chart.js if quality page exists
        has_chart = "chart.umd.js" in html or "Chart" in html
        if has_chart:
            print_check(True, "Chart.js referenced in HTML")
        
        return response.status_code == 200, html
    except Exception as e:
        print(f"  {Colors.FAIL}✗ Failed to load page: {e}{Colors.ENDC}")
        return False, ""


def generate_verification_report(all_checks: Dict[str, bool]) -> None:
    """Generate a summary report."""
    print_header("7. VERIFICATION SUMMARY")
    
    passed = sum(1 for v in all_checks.values() if v)
    total = len(all_checks)
    
    print(f"  {Colors.BOLD}Results: {passed}/{total} checks passed{Colors.ENDC}\n")
    
    for check_name, result in all_checks.items():
        status = f"{Colors.OKGREEN}✓ PASS{Colors.ENDC}" if result else f"{Colors.FAIL}✗ FAIL{Colors.ENDC}"
        print(f"    [{status}] {check_name}")
    
    print()
    if passed == total:
        print(f"  {Colors.OKGREEN}{Colors.BOLD}✓ ALL CHECKS PASSED - READY FOR AZURE DEPLOYMENT{Colors.ENDC}")
    else:
        print(f"  {Colors.WARNING}{Colors.BOLD}⚠ {total - passed} CHECK(S) FAILED - REVIEW ABOVE{Colors.ENDC}")
    
    return passed == total


def main():
    parser = argparse.ArgumentParser(
        description="Verify STRICT CSP mode works locally before Azure deployment"
    )
    parser.add_argument("--port", type=int, default=5000, help="Flask port (default: 5000)")
    parser.add_argument("--timeout", type=int, default=10, help="Request timeout in seconds (default: 10)")
    parser.add_argument("--skip-server", action="store_true", help="Skip server startup (assume already running)")
    
    args = parser.parse_args()
    
    print(f"{Colors.BOLD}{Colors.HEADER}")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║           CSP STRICT MODE LOCAL VERIFICATION              ║")
    print("║                                                            ║")
    print("║  This script validates that STRICT CSP mode works         ║")
    print("║  correctly before deploying to Azure.                     ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}")
    
    checks = {}
    process = None
    
    try:
        # Step 1: Check vendor files
        vendor_ok, missing = check_vendor_files()
        checks["Vendor Files Present"] = vendor_ok
        if not vendor_ok:
            print(f"\n{Colors.WARNING}Missing files: {', '.join(missing)}")
            print(f"Please download them and retry.{Colors.ENDC}")
            return 1
        
        # Step 2: Set environment
        env = set_environment_variables()
        
        # Step 3: Start server (unless skipped)
        if not args.skip_server:
            process = start_flask_server(env, args.port)
            if not process:
                checks["Flask Server Start"] = False
                return 1
            checks["Flask Server Start"] = True
            time.sleep(2)  # Give server time to fully initialize
        
        # Step 4: Fetch CSP header
        base_url = f"http://localhost:{args.port}"
        csp_ok, csp_header = fetch_csp_header(base_url)
        checks["CSP Header Retrieved"] = csp_ok
        
        if not csp_ok:
            return 1
        
        # Step 5: Validate CSP content
        csp_checks = validate_csp_content(csp_header)
        checks["CSP Content Valid"] = all(check[0] for check in csp_checks)
        
        # Step 6: Test page load
        load_ok, html = test_page_load(base_url)
        checks["Page Load Success"] = load_ok
        
        # Step 7: Generate report
        all_passed = generate_verification_report(checks)
        
        if all_passed:
            print(f"\n{Colors.OKGREEN}Next Step: Deploy to Azure with CSP_MODE=STRICT{Colors.ENDC}")
            print("See docs/DEPLOYMENT/AZURE_CSP_DEPLOYMENT.md for instructions\n")
            return 0
        else:
            print(f"\n{Colors.FAIL}⚠ Fix the failed checks before deploying to Azure{Colors.ENDC}\n")
            return 1
    
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Verification interrupted by user{Colors.ENDC}")
        return 130
    except Exception as e:
        print(f"\n{Colors.FAIL}Unexpected error: {e}{Colors.ENDC}")
        return 1
    finally:
        # Cleanup: Stop Flask server if we started it
        if process and not args.skip_server:
            print(f"\n{Colors.OKBLUE}Stopping Flask server...{Colors.ENDC}")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    sys.exit(main())
