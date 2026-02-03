#!/usr/bin/env python3
"""Quick test script for Step 1 (Trust Scorer) + Step 2 (DOM Snapshot) integration.

Usage:
    python test_trust_snapshot.py <url>

Examples:
    # High-trust verified domain (score 90-100)
    python test_trust_snapshot.py https://elections.maryland.gov/results

    # Medium-trust gov domain (score 60-70)
    python test_trust_snapshot.py https://example.gov/results

    # Low-trust non-gov domain (score 30-40)
    python test_trust_snapshot.py https://elections-unofficial.com/results

    # Blocked suspicious domain (score 0-20)
    python test_trust_snapshot.py http://elections.xyz/results
"""

import sys

from webapp.parser.utils.url_trust_scorer import (
    compute_trust_score,
    should_quarantine,
    should_reject,
    should_use_snapshot_mode,
)


def test_url_trust_and_snapshot_decision(url: str) -> None:
    """Test trust scoring and snapshot mode decision for a URL."""
    print("=" * 70)
    print(f"Testing URL: {url}")
    print("=" * 70)
    
    # Build minimal context (state/county inference would happen in real pipeline)
    context = {
        "state": "Maryland" if "maryland" in url.lower() else None,
        "county": None,
    }
    
    # Compute trust score
    print("\n[Step 1: Trust Scoring]")
    trust_score, trust_factors = compute_trust_score(url, context, session_id=None)
    
    print(f"Trust Score: {trust_score}/100")
    print("\nTrust Factors:")
    for key, value in trust_factors.items():
        # Convert boolean/numeric to readable string
        if isinstance(value, bool):
            display = "✓ Yes" if value else "✗ No"
        elif isinstance(value, (int, float)):
            display = f"{value}"
        else:
            display = str(value)
        print(f"  - {key:.<30} {display}")
    
    # Determine action
    print("\n[Step 2: Access Control Decision]")
    if should_reject(trust_score, url):
        action = "🚫 REJECT (score < 30)"
        description = "URL blocked for security reasons. Will not process."
    elif should_quarantine(trust_score, url):
        action = "⚠️  QUARANTINE (score 30-49)"
        description = "URL flagged for manual review before processing."
    elif should_use_snapshot_mode(trust_score, url):
        action = "📸 DOM SNAPSHOT MODE (score 50-79)"
        description = "URL will be processed using safe snapshot extraction (no JS execution)."
    else:
        action = "✅ DIRECT NAVIGATION (score 80-100)"
        description = "URL trusted for full browser navigation with JavaScript."
    
    print(f"Action: {action}")
    print(f"Description: {description}")
    
    # Expected behavior
    print("\n[Expected Parser Behavior]")
    if should_reject(trust_score, url):
        print("  1. Log error: 'URL rejected due to low trust score'")
        print("  2. Mark URL as 'rejected' in .processed_urls")
        print("  3. Early return (no browser navigation)")
        print("  4. Log to trust_history.jsonl with action='reject'")
    elif should_quarantine(trust_score, url):
        print("  1. Log warning: 'URL quarantined for manual review'")
        print("  2. Mark URL as 'quarantined' in .processed_urls")
        print("  3. Early return (no processing until approved)")
        print("  4. Log to trust_history.jsonl with action='quarantine'")
        print("  5. TODO (Step 6): Add to quarantine review queue")
    elif should_use_snapshot_mode(trust_score, url):
        print("  1. Log info: 'Using DOM snapshot mode for medium-trust URL'")
        print("  2. Navigate to page with Playwright (no JS execution)")
        print("  3. Capture static HTML snapshot via page.content()")
        print("  4. Extract tables using selectolax (or fallback parser)")
        print("  5. Build metadata with snapshot_mode=True, trust_score={trust_score}")
        print("  6. Finalize output (same as full navigation)")
        print("  7. Mark URL as 'success' with snapshot_mode=True")
        print("  8. Early return (skip full navigation pipeline)")
    else:
        print("  1. Log info: 'High-trust URL - proceeding with direct navigation'")
        print("  2. Continue with full navigation pipeline (JS enabled)")
        print("  3. Execute navigation strategies (Playwright, optional Selenium)")
        print("  4. Run navigation recipes if available")
        print("  5. Autoscroll and wait for content")
        print("  6. Route to handler (state/county/format)")
        print("  7. Extract data with full browser automation")
    
    # Telemetry events
    print("\n[Telemetry Events Expected]")
    print("  - trust_score_computed (Step 1)")
    if should_use_snapshot_mode(trust_score, url):
        print("  - dom_snapshot_captured (Step 2)")
        print("  - snapshot_tables_extracted (Step 2)")
    else:
        print("  - navigation_start")
        print("  - navigation_complete")
        print("  - page_scrolled (if applicable)")
    
    print("\n" + "=" * 70)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nError: URL argument required")
        print("Usage: python test_trust_snapshot.py <url>")
        sys.exit(1)
    
    url = sys.argv[1].strip()
    if not url:
        print("Error: Empty URL provided")
        sys.exit(1)
    
    test_url_trust_and_snapshot_decision(url)


if __name__ == "__main__":
    main()
