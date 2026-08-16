"""
Comprehensive test for URL and Filename parsers against known patterns.

This test validates both parsers against common election file naming conventions
without requiring Google Sheets credentials.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from webapp.parser.url_parser import parse_url_simple
from webapp.parser.filename_parser import parse_filename_simple
from webapp.parser.utils.shared_logic import normalize_state_name


def _run_url_state_detection():
    """Test URL parser state detection accuracy"""

    test_cases = [
        # (URL, expected_state)
        ("https://results.sos.ga.gov/results/2024", "GA"),
        ("https://electionresults.sos.state.co.us/results", "CO"),
        ("https://results.vote.wa.gov/results/2024", "WA"),
        ("https://www.sos.alabama.gov/alabama-votes", "alabama"),
        ("https://results.enr.clarityelections.com/GA/Fulton/105430/", "GA"),
        ("https://www.electionreturns.pa.gov/ReportCenter/Reports", "PA"),
        ("https://elections.virginia.gov/resultsreports/", "virginia"),
        ("https://results.arizona.vote/default.html", "arizona"),
        ("https://results.elections.myflorida.com/", "florida"),
    ]

    print("=" * 80)
    print("URL PARSER STATE DETECTION TEST")
    print("=" * 80)
    print()

    passed = 0
    failed = 0

    for url, expected_state in test_cases:
        parsed = parse_url_simple(url)
        detected_state = parsed.get('state', '').upper() if parsed.get('state') else None
        expected_normalized = expected_state.upper() if expected_state else None

        # Handle both state codes and full names
        match = (
            detected_state == expected_normalized or
            (detected_state and expected_normalized and
             (detected_state in expected_normalized or expected_normalized in detected_state))
        )

        status = "✓" if match else "✗"
        if match:
            passed += 1
        else:
            failed += 1

        print(f"{status} URL: {url[:60]}...")
        print(f"  Expected: {expected_state}")
        print(f"  Detected: {parsed.get('state', 'None')}")
        if parsed.get('vendor_hint'):
            print(f"  Vendor: {parsed['vendor_hint']}")
        print()

    accuracy = (passed / len(test_cases)) * 100 if test_cases else 0
    print(f"State Detection Accuracy: {passed}/{len(test_cases)} ({accuracy:.1f}%)")
    print()

    return passed, failed


def test_url_state_detection():
    passed, failed = _run_url_state_detection()
    assert failed == 0, f"URL state detection mismatches: {failed}; passed: {passed}"


def _run_filename_parsing():
    """Test filename parser against common patterns"""

    test_cases = [
        # (filename, expected_state, expected_county, expected_year)
        ("Alabama_Jefferson_County_2024_General.pdf", "AL", "Jefferson", "2024"),
        ("GA-Fulton-President-2024.csv", "GA", None, "2024"),
        ("California_Alameda_Results_2024.xlsx", "CA", None, "2024"),
        ("2024_Florida_Statewide_Senate.pdf", "FL", None, "2024"),
        ("NewYork_Rockland_County_General_2024.csv", "NY", "Rockland", "2024"),
        ("PA_StLouis_Canvass_2024.pdf", "PA", "St Louis", "2024"),
        ("Washington_King_County_Results.pdf", "WA", "King", None),
        ("TX-Harris-General-Election-2024.csv", "TX", None, "2024"),
        ("Arizona_Maricopa_Primary_2024.xlsx", "AZ", None, "2024"),
    ]

    print("=" * 80)
    print("FILENAME PARSER TEST")
    print("=" * 80)
    print()

    passed = 0
    failed = 0

    for filename, expected_state, expected_county, expected_year in test_cases:
        parsed = parse_filename_simple(filename)

        state_match = parsed.get('state') == expected_state if expected_state else True
        county_match = parsed.get('county') == expected_county if expected_county else True
        year_match = parsed.get('year') == expected_year if expected_year else True

        all_match = state_match and county_match and year_match
        status = "✓" if all_match else "✗"

        if all_match:
            passed += 1
        else:
            failed += 1

        print(f"{status} Filename: {filename}")
        print(f"  Expected: State={expected_state}, County={expected_county}, Year={expected_year}")
        print(f"  Detected: State={parsed.get('state')}, County={parsed.get('county')}, Year={parsed.get('year')}")
        if parsed.get('contest_type'):
            print(f"  Contest: {parsed['contest_type']}")
        if not all_match:
            print(f"  MISMATCH: state={state_match}, county={county_match}, year={year_match}")
        print()

    accuracy = (passed / len(test_cases)) * 100 if test_cases else 0
    print(f"Filename Parsing Accuracy: {passed}/{len(test_cases)} ({accuracy:.1f}%)")
    print()

    return passed, failed


def test_filename_parsing():
    passed, failed = _run_filename_parsing()
    assert failed == 0, f"Filename parsing mismatches: {failed}; passed: {passed}"


def _run_url_vs_filename_consistency():
    """
    Test that both parsers produce consistent results for similar patterns.

    If a URL is from "GA" domain and a filename contains "GA", both should detect it.
    """

    print("=" * 80)
    print("URL vs FILENAME CONSISTENCY TEST")
    print("=" * 80)
    print()

    test_pairs = [
        # (URL, matching_filename)
        ("https://results.sos.ga.gov/results/2024/general", "GA_2024_General_Results.pdf"),
        ("https://results.vote.wa.gov/results/2024", "Washington_2024_Results.csv"),
        ("https://www.electionreturns.pa.gov/Reports", "Pennsylvania_Election_Returns.pdf"),
        ("https://results.arizona.vote/2024", "Arizona_2024_Election.xlsx"),
    ]

    consistent = 0
    inconsistent = 0

    for url, filename in test_pairs:
        url_parsed = parse_url_simple(url)
        file_parsed = parse_filename_simple(filename)

        url_state = url_parsed.get('state')
        url_state_canonical = normalize_state_name(url_state) if url_state else None
        file_state = file_parsed.get('state')
        file_state_canonical = normalize_state_name(file_state) if file_state else None

        # Both should detect a state
        both_detected = url_state_canonical and file_state_canonical

        if both_detected:
            # Compare canonical state identity, not parser representation.
            match = url_state_canonical == file_state_canonical

            if match:
                consistent += 1
                status = "✓"
            else:
                inconsistent += 1
                status = "✗ MISMATCH"
        else:
            inconsistent += 1
            status = "✗ MISSING"

        print(f"{status} Pair:")
        print(f"  URL: {url}")
        print(f"    → State: {url_parsed.get('state', 'None')}")
        print(f"  Filename: {filename}")
        print(f"    → State: {file_parsed.get('state', 'None')}")
        print()

    print(f"Consistency: {consistent}/{len(test_pairs)} pairs matched")
    print()

    return consistent, inconsistent


def test_url_vs_filename_consistency():
    consistent, inconsistent = _run_url_vs_filename_consistency()
    assert inconsistent == 0, f"URL/filename inconsistencies: {inconsistent}; consistent: {consistent}"


def test_edge_cases():
    """Test edge cases and challenging patterns"""

    print("=" * 80)
    print("EDGE CASES TEST")
    print("=" * 80)
    print()

    print("Testing ambiguous state codes...")

    # CO could be Colorado or Company
    parsed = parse_url_simple("https://www.co.jefferson.wa.us/elections")
    print(f"  URL with 'co' subdomain: {parsed.get('state')}")
    print(f"    (Should be WA from domain, not CO)")
    print()

    # Multi-word counties
    parsed_file = parse_filename_simple("New_York_St_Louis_County_2024.pdf")
    print(f"  Filename with multi-word county: {parsed_file.get('county')}")
    print(f"    (Should detect 'St Louis')")
    print()

    # Year in various formats
    test_year_formats = [
        "Election_2024_Results.pdf",
        "2024-General-Election.pdf",
        "Results2024.pdf",
        "ElectionYear2024.csv"
    ]

    print("  Testing year detection in various formats:")
    for fn in test_year_formats:
        parsed = parse_filename_simple(fn)
        year_detected = parsed.get('year') == '2024'
        status = "✓" if year_detected else "✗"
        print(f"    {status} {fn}: {parsed.get('year', 'None')}")
    print()


def run_all_tests():
    """Run all parser tests"""

    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "PARSER VALIDATION TEST SUITE" + " " * 30 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    # Run tests
    url_passed, url_failed = _run_url_state_detection()
    file_passed, file_failed = _run_filename_parsing()
    consistent, inconsistent = _run_url_vs_filename_consistency()
    test_edge_cases()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"URL Parser:")
    print(f"  ✓ Passed: {url_passed}")
    print(f"  ✗ Failed: {url_failed}")
    print()
    print(f"Filename Parser:")
    print(f"  ✓ Passed: {file_passed}")
    print(f"  ✗ Failed: {file_failed}")
    print()
    print(f"Consistency:")
    print(f"  ✓ Consistent: {consistent}")
    print(f"  ✗ Inconsistent: {inconsistent}")
    print()

    total_passed = url_passed + file_passed + consistent
    total_failed = url_failed + file_failed + inconsistent
    total_accuracy = (total_passed / (total_passed + total_failed)) * 100 if (total_passed + total_failed) > 0 else 0

    print(f"Overall Accuracy: {total_accuracy:.1f}%")
    print()

    if total_accuracy >= 80:
        print("✓ Parsers performing well (>80% accuracy)")
    elif total_accuracy >= 60:
        print("⚠ Parsers need improvement (60-80% accuracy)")
    else:
        print("✗ Parsers need significant work (<60% accuracy)")

    print()
    print("=" * 80)
    print()


if __name__ == "__main__":
    run_all_tests()
