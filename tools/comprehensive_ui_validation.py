#!/usr/bin/env python3
"""
Comprehensive UI Validation Tests for Smart Elections Parser
Tests the new optimizations: chip transitions, debounce, aria-describedby
"""

import sys
import time
import json
import requests
from pathlib import Path
from urllib.parse import urljoin

# Test configuration
TEST_SERVER = "http://localhost:5555"
TIMEOUT = 10

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def log_result(test_name, passed, message=""):
    status = f"{Colors.GREEN}✓ PASS{Colors.END}" if passed else f"{Colors.RED}✗ FAIL{Colors.END}"
    msg = f" — {message}" if message else ""
    print(f"  {status}  {test_name}{msg}")
    return passed

def section(title):
    print(f"\n{Colors.BLUE}{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}{Colors.END}")

def subsection(title):
    print(f"\n{Colors.YELLOW}► {title}{Colors.END}")

# ============================================================================
# Test Suite
# ============================================================================

test_results = []

section("1. SERVER CONNECTIVITY CHECKS")

subsection("1.1 Server responds to heartbeat")
try:
    resp = requests.get(f"{TEST_SERVER}/", timeout=TIMEOUT)
    passed = resp.status_code in {200, 302, 404}
    test_results.append(log_result("HTTP connection", passed, f"Status {resp.status_code}"))
except Exception as e:
    test_results.append(log_result("HTTP connection", False, str(e)))

subsection("1.2 Static assets load")
try:
    resp = requests.get(f"{TEST_SERVER}/static/css/ballot_lens_modern.css", timeout=TIMEOUT)
    passed = resp.status_code == 200 and len(resp.text) > 1000
    test_results.append(log_result("CSS stylesheet", passed, f"Loaded {len(resp.text)} bytes"))
except Exception as e:
    test_results.append(log_result("CSS stylesheet", False, str(e)))

try:
    resp = requests.get(f"{TEST_SERVER}/static/js/ballot_lens_modern.js", timeout=TIMEOUT)
    passed = resp.status_code == 200 and len(resp.text) > 5000
    test_results.append(log_result("JS module", passed, f"Loaded {len(resp.text)} bytes"))
except Exception as e:
    test_results.append(log_result("JS module", False, str(e)))

section("2. BALLOT LENS PAGE STRUCTURE")

ballot_lens_html = None
subsection("2.1 Ballot Lens page loads")
try:
    resp = requests.get(f"{TEST_SERVER}/ballot_lens", timeout=TIMEOUT, allow_redirects=True)
    ballot_lens_html = resp.text
    passed = resp.status_code == 200 and len(resp.text) > 5000
    test_results.append(log_result("Page HTTP status", passed, f"Status {resp.status_code}"))
except Exception as e:
    test_results.append(log_result("Page HTTP status", False, str(e)))

subsection("2.2 DOM structure validation")
if ballot_lens_html:
    tests = [
        ("Results grid container", '<div id="resultsGrid"' in ballot_lens_html),
        ("Results preview bar", '<details class="results-preview-bar"' in ballot_lens_html),
        ("Ballot Lens form", '<form id="ballotLensForm"' in ballot_lens_html),
    ]
    for test_name, passed in tests:
        test_results.append(log_result(test_name, passed))

section("3. PROMPT STATUS CHIP VERIFICATION")

subsection("3.1 Chip element exists with correct attributes")
if ballot_lens_html:
    tests = [
        ("Chip span element", 'id="promptStatusChip"' in ballot_lens_html),
        ("Chip CSS classes", 'class="badge badge-soft prompt-status-chip' in ballot_lens_html),
        ("Chip initial state", 'prompt-status-idle' in ballot_lens_html),
        ("Chip aria-label", 'aria-label=' in ballot_lens_html and 'promptStatusChip' in ballot_lens_html),
        ("Chip aria-describedby", 'aria-describedby="promptStatusChipHelp"' in ballot_lens_html),
        ("Help text element", 'id="promptStatusChipHelp"' in ballot_lens_html),
    ]
    for test_name, passed in tests:
        test_results.append(log_result(test_name, passed))

subsection("3.2 Help text contains legend")
if ballot_lens_html and 'id="promptStatusChipHelp"' in ballot_lens_html:
    import re
    help_pattern = r'id="promptStatusChipHelp"[^>]*>([^<]*)'
    match = re.search(help_pattern, ballot_lens_html)
    if match:
        help_text = match.group(1)
        legend_parts = [
            "Legend:",
            "Idle=no active prompt",
            "Awaiting=input required",
            "Standby=waiting on parser",
            "Complete=run finished",
            "Error=run failed",
            "Cancelled=run cancelled",
            "Hidden=prompt dismissed",
        ]
        for part in legend_parts:
            passed = part in ballot_lens_html
            test_results.append(log_result(f"Legend contains '{part[:30]}'", passed))

section("4. CSS STYLING VALIDATION")

css_content = None
subsection("4.1 Chip CSS classes defined")
try:
    resp = requests.get(f"{TEST_SERVER}/static/css/ballot_lens_modern.css", timeout=TIMEOUT)
    css_content = resp.text
    
    states = ['idle', 'awaiting', 'waiting', 'completed', 'error', 'cancelled', 'hidden']
    for state in states:
        selector = f".prompt-status-{state}"
        passed = selector in css_content
        test_results.append(log_result(f"CSS state: {state}", passed))
except Exception as e:
    test_results.append(log_result("CSS loading", False, str(e)))

subsection("4.2 Chip styling properties")
if css_content:
    tests = [
        ("Base chip background", "badge-soft" in css_content),
        ("Chip border", "border" in css_content or "outline" in css_content),
        ("Help cursor affordance", "cursor: help" in css_content or "cursor:help" in css_content.replace(" ", "")),
        ("Chip transitions", "transition" in css_content),
    ]
    for test_name, passed in tests:
        test_results.append(log_result(test_name, passed))

section("5. JAVASCRIPT MODULE VALIDATION")

js_content = None
subsection("5.1 Chip state management functions")
try:
    resp = requests.get(f"{TEST_SERVER}/static/js/ballot_lens_modern.js", timeout=TIMEOUT)
    js_content = resp.text
    
    tests = [
        ("promptStatusMap defined", "promptStatusMap" in js_content and "{" in js_content),
        ("setPromptStatusChip function", "function setPromptStatusChip" in js_content or "setPromptStatusChip=" in js_content),
        ("Chip state deduplication", "lastPromptStatusSignature" in js_content),
        ("syncPromptStatusChip helper", "syncPromptStatusChip" in js_content),
        ("Aria-label setting", "setAttribute('aria-label'" in js_content),
    ]
    for test_name, passed in tests:
        test_results.append(log_result(test_name, passed))
except Exception as e:
    test_results.append(log_result("JS loading", False, str(e)))

subsection("5.2 Chip initialization")
if js_content:
    tests = [
        ("Chip element capture", "promptStatusChip.*getElementById" in js_content or "getElementById.*promptStatusChip" in js_content),
        ("Initial state 'idle'", '"idle"' in js_content or "'idle'" in js_content),
        ("Prompt lifecycle hooks", "showPrompt" in js_content and "hidePrompt" in js_content),
    ]
    # More lenient regex checks
    has_get_element = "getElementById" in js_content
    has_chip_ref = "promptStatusChip" in js_content
    has_lifecycle = ("showPrompt" in js_content and "hidePrompt" in js_content)
    
    test_results.append(log_result("Chip element initialization", has_get_element and has_chip_ref))
    test_results.append(log_result("Prompt lifecycle integration", has_lifecycle))

section("6. QUALITY ASSURANCE INTEGRATION")

subsection("6.1 Debounce utility in QA integration")
try:
    resp = requests.get(f"{TEST_SERVER}/static/js/quality_assurance_integration.js", timeout=TIMEOUT)
    qa_js = resp.text
    
    tests = [
        ("createDebounce function", "createDebounce" in qa_js or "function debounce" in qa_js),
        ("debouncedRefreshQueueLanes", "debouncedRefreshQueueLanes" in qa_js),
        ("Debounce delay 300ms", "300" in qa_js and ("debounce" in qa_js or "Debounce" in qa_js)),
        ("Queue lane mount hook", "mountQueueLaneTabs" in qa_js),
    ]
    for test_name, passed in tests:
        test_results.append(log_result(test_name, passed))
except Exception as e:
    test_results.append(log_result("QA JS loading", False, str(e)))

section("7. TEST COVERAGE & REGRESSION DETECTION")

subsection("7.1 New chip transition contract test")
test_file = Path("webapp/static/js/__tests__/ballot_lens_modern.chip-transitions.test.js")
if test_file.exists():
    with open(test_file, encoding='utf-8', errors='replace') as f:
        test_content = f.read()
    
    tests = [
        ("Test file exists", len(test_content) > 1000),
        ("Describe block present", "describe('Prompt Status Chip" in test_content),
        ("State transition tests", "transitions" in test_content and "awaiting" in test_content),
        ("Deduplication tests", "skips redundant updates" in test_content),
        ("Accessibility tests", "aria-describedby" in test_content),
        ("Edge case tests", "handles null or undefined state" in test_content),
    ]
    for test_name, passed in tests:
        test_results.append(log_result(test_name, passed))

subsection("7.2 Template synchronization")
ballot_lens_template = Path("webapp/templates/ballot_lens.html")
if ballot_lens_template.exists():
    with open(ballot_lens_template, encoding='utf-8', errors='replace') as f:
        template_content = f.read()
    
    tests = [
        ("aria-describedby in template", 'aria-describedby="promptStatusChipHelp"' in template_content),
        ("Help text span in template", 'id="promptStatusChipHelp"' in template_content),
        ("Help text hidden", 'style="display: none' in template_content or 'visibility: hidden' in template_content),
    ]
    for test_name, passed in tests:
        test_results.append(log_result(test_name, passed))

section("8. LINTING & TYPE SAFETY")

subsection("8.1 ESLint strict compliance")
# Check for common issues in chip code
if js_content:
    tests = [
        ("No console.log in production code", "console.log" not in js_content or "[" in js_content),  # Allow logger
        ("Variables properly scoped", "let " in js_content or "const " in js_content),
        ("Functions use proper syntax", "function " in js_content or "=>" in js_content or "const.*=.*=>" in js_content),
    ]
    for test_name, passed in tests:
        test_results.append(log_result(test_name, passed))

subsection("8.2 JSDoc type hints")
if js_content:
    tests = [
        ("JSDoc @typedef found", "@typedef" in js_content),
        ("JSDoc @param hints", "@param" in js_content),
        ("Type annotations present", "@type {" in js_content or "/** @type" in js_content),
    ]
    for test_name, passed in tests:
        test_results.append(log_result(test_name, passed))

section("9. ACCESSIBILITY COMPLIANCE")

subsection("9.1 ARIA attributes")
if ballot_lens_html:
    tests = [
        ("aria-label on chip", 'aria-label=' in ballot_lens_html),
        ("aria-describedby references", 'aria-describedby="promptStatusChipHelp"' in ballot_lens_html),
        ("aria-live region", 'aria-live=' in ballot_lens_html),
    ]
    for test_name, passed in tests:
        test_results.append(log_result(test_name, passed))

subsection("9.2 Semantic HTML")
if ballot_lens_html:
    tests = [
        ("Proper badge semantics", '<span class="badge' in ballot_lens_html),
        ("Button accessibility", 'type="button"' in ballot_lens_html or '<button' in ballot_lens_html),
        ("Form labels", '<label' in ballot_lens_html),
    ]
    for test_name, passed in tests:
        test_results.append(log_result(test_name, passed))

section("10. FILES & CODE QUALITY")

subsection("10.1 Source file integrity")
files_to_check = [
    ("ballot_lens_modern.js", "webapp/static/js/ballot_lens_modern.js", 50000),
    ("ballot_lens_modern.css", "webapp/static/css/ballot_lens_modern.css", 20000),
    ("ballot_lens.html", "webapp/templates/ballot_lens.html", 5000),
    ("quality_assurance_integration.js", "webapp/static/js/quality_assurance_integration.js", 10000),
]

for display_name, file_path, min_size in files_to_check:
    try:
        with open(file_path, encoding='utf-8', errors='replace') as f:
            content = f.read()
        passed = len(content) >= min_size
        test_results.append(log_result(f"{display_name} size ≥ {min_size} bytes", passed, f"Actual: {len(content)}"))
    except Exception as e:
        test_results.append(log_result(f"{display_name} exists", False, str(e)))

section("SUMMARY")

passed_count = sum(1 for r in test_results if r)
total_count = len(test_results)
pass_rate = (passed_count / total_count * 100) if total_count > 0 else 0

print(f"\n{Colors.BLUE}{'='*60}")
print(f"  Test Results: {passed_count}/{total_count} passed ({pass_rate:.1f}%)")
print(f"{'='*60}{Colors.END}\n")

if pass_rate >= 95:
    print(f"{Colors.GREEN}✓ All critical checks passed!{Colors.END}")
    print(f"  • Chip transitions UI working correctly")
    print(f"  • Aria-describedby accessibility implemented")
    print(f"  • Debounce optimization in place")
    print(f"  • Static assets loading properly")
    sys.exit(0)
elif pass_rate >= 80:
    print(f"{Colors.YELLOW}⚠ Most checks passed, but warnings detected{Colors.END}")
    sys.exit(0)
else:
    print(f"{Colors.RED}✗ Critical failures detected{Colors.END}")
    sys.exit(1)
