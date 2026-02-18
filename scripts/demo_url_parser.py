"""
Quick demo: Parse URLs from the URL library and generate training data
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from webapp.parser.url_parser import parse_url_simple
from webapp.parser.config import URL_LIST_FILE, LOG_DIR
import json


def demo_parse_url_library():
    """Parse first 10 URLs from library and show structured output"""
    
    urls_file = URL_LIST_FILE
    if not urls_file.exists():
        print(f"URL library not found at {urls_file}")
        return
    
    print("=" * 80)
    print("URL PARSER DEMO: Processing URL Library")
    print("=" * 80)
    print()
    
    # Read URLs (handle both plain URLs and tab-delimited format)
    urls = []
    with open(urls_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            # Check if tab-delimited (year\tcontest\tstate\tscope\tformat\tnotes\turl)
            if "\t" in line:
                parts = line.split("\t")
                if len(parts) >= 7:
                    url = parts[6].strip()  # URL is last column
                else:
                    url = line  # Fallback to full line
            else:
                url = line
            
            # Validate it looks like a URL
            if url.startswith("http"):
                urls.append(url)
            
            if len(urls) >= 10:  # Limit to first 10
                break
    
    if not urls:
        print("No URLs found in library")
        return
    
    print(f"Processing {len(urls)} sample URLs...\n")
    
    # Parse and display
    parsed_data = []
    for i, url in enumerate(urls, 1):
        # Truncate display for long URLs
        display_url = url if len(url) < 80 else url[:77] + "..."
        print(f"{i}. {display_url}")
        
        try:
            parsed = parse_url_simple(url)
            parsed_data.append(parsed)
            
            # Show key info
            print(f"   └─ Root: {parsed['root_domain']}")
            print(f"   └─ Segments: {' / '.join(parsed['path_segments'][:3])}")
            if parsed['state']:
                print(f"   └─ State: {parsed['state']}")
            if parsed['county']:
                print(f"   └─ County: {parsed['county']}")
            if parsed['vendor_hint']:
                print(f"   └─ Vendor: {parsed['vendor_hint']}")
            if parsed['year']:
                print(f"   └─ Year: {parsed['year']}")
            
        except Exception as e:
            print(f"   └─ ERROR: {e}")
        
        print()
    
    # Summary statistics
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    states = [p['state'] for p in parsed_data if p.get('state')]
    counties = [p['county'] for p in parsed_data if p.get('county')]
    vendors = [p['vendor_hint'] for p in parsed_data if p.get('vendor_hint')]
    years = [p['year'] for p in parsed_data if p.get('year')]
    
    print(f"  Total Parsed: {len(parsed_data)}")
    print(f"  States Detected: {len(states)} unique states")
    if states:
        print(f"    → {', '.join(sorted(set(states)))}")
    print(f"  Counties Detected: {len(counties)}")
    if counties:
        print(f"    → {', '.join(sorted(set(counties)))}")
    print(f"  Vendors Detected: {len(vendors)}")
    if vendors:
        print(f"    → {', '.join(sorted(set(vendors)))}")
    print(f"  Years Detected: {len(years)}")
    if years:
        print(f"    → {', '.join(sorted(set(years)))}")
    print()
    
    # Show sample JSON for training
    if parsed_data:
        print("=" * 80)
        print("SAMPLE TRAINING DATA (JSON)")
        print("=" * 80)
        print(json.dumps(parsed_data[0], indent=2))
        print()
    
    print("=" * 80)
    print(f"To parse ALL URLs and save to training file, use:")
    print(f"  curl -X POST http://localhost:5000/api/urls/parse_all")
    print(f"Or call the API endpoint /api/urls/parse with store=true")
    print("=" * 80)


if __name__ == "__main__":
    demo_parse_url_library()
