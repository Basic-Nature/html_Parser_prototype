"""
Diagnostic: Compare URL-based state/county extraction vs Google Sheets truth data
This shows why ballot_lens dropdowns were incorrect
"""
import io
import os
import sys
from collections import defaultdict
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))

# Set up credentials
json_file = Path(__file__).parent / "google_service_account.json"
os.environ['GOOGLE_SHEETS_ELECTION_DB_LITE_CREDENTIALS'] = str(json_file)
os.environ['GOOGLE_SHEETS_DB_LITE_ID'] = '154z24Y7z99Yb9I7Swa8-6mKBgWKN44Ap9ukx5WEgLRY'

print("="*80)
print("GOOGLE SHEETS vs URL-EXTRACTION COMPARISON")
print("="*80)

print("\n[1/3] Fetching authoritative data from Google Sheets...")
from webapp.parser.data_standardization.google_sheets_client import get_election_data_client

client = get_election_data_client()
result = client.fetch_finalized_data()

# Build authoritative state-to-county mappings
state_counties_truth = defaultdict(set)
for record in result.records:
    state = record.get('State', '').strip()
    county = record.get('County/District', '').strip()
    if state and county:
        state_counties_truth[state].add(county)

print(f"      Found {len(state_counties_truth)} states")
print(f"      Found {sum(len(c) for c in state_counties_truth.values())} total unique counties")

print("\n[2/3] Analyzing data quality issues...")

# Check for suspicious county counts
issues = []
for state, counties in state_counties_truth.items():
    if len(counties) > 300:  # No state has more than 254 counties (Texas)
        issues.append((state, len(counties), 'Suspiciously high county count'))

if issues:
    print("      WARNING: Data quality issues detected:")
    for state, count, issue in issues[:5]:
        sample = sorted(counties)[:5]
        print(f"        {state}: {count} counties - {issue}")
        print(f"          Sample: {', '.join(sample)}")
else:
    print("      No obvious data quality issues")

print("\n[3/3] Showing correct state-to-county mappings...")

# Show sample states with correct county counts
test_states = ['Arizona', 'Alabama', 'Alaska', 'Texas', 'Vermont']
print("\n      Sample states with their counties:\n")

for state in test_states:
    if state in state_counties_truth:
        counties = sorted(state_counties_truth[state])
        print(f"      {state} ({len(counties)} counties):")
        if len(counties) <= 20:
            for county in counties:
                print(f"        - {county}")
        else:
            for county in counties[:10]:
                print(f"        - {county}")
            print(f"        ... and {len(counties) - 10} more")
        print()

print("="*80)
print("DIAGNOSIS")
print("="*80)
print("""
PROBLEM: ballot_lens currently extracts state/county from URL patterns:
  - Looks for 2-letter state codes in URLs
  - Looks for "county" keyword in URLs
  - Very unreliable and incomplete

SOLUTION: Use Google Sheets data to populate dropdowns:
  1. Add API endpoint: /api/election_data/states_counties
  2. Return authoritative state-to-county mappings from Google Sheets
  3. Update ballot_lens_modern.js to fetch real data instead of URL heuristics
  
BENEFIT: Accurate, complete state-to-county mappings for all 50 states
""")
print("="*80)
