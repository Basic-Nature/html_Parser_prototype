"""
Google Sheets Structure Discovery Script

Purpose: Explore the Google Sheets structure before writing migration logic.
         Maps all worksheets, columns, data types, and relationships.

Usage:
    python scripts/discover_google_sheets.py

Output:
    - Console: Pretty-printed worksheet structure
    - File: scripts/sheets_structure.json (detailed schema for migration planning)

Requirements:
    - GOOGLE_SERVICE_ACCOUNT_PATH in .env
    - GOOGLE_SHEETS_WORKBOOK_ID in .env
    - Google Service Account shared with the target sheet
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import gspread
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Configuration
SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets.readonly',
    'https://www.googleapis.com/auth/drive.readonly'
]

CREDS_PATH = os.getenv("GOOGLE_SERVICE_ACCOUNT_PATH")
WORKBOOK_ID = os.getenv("GOOGLE_SHEETS_WORKBOOK_ID")

# Output file
OUTPUT_FILE = Path(__file__).parent / "sheets_structure.json"


def categorize_column_type(values: List[Any]) -> str:
    """
    Analyze column values to determine data type.
    
    Args:
        values: List of cell values from a column (excluding header)
    
    Returns:
        Categorized type: url, date, boolean, numeric, text, reference
    """
    if not values:
        return "empty"
    
    # Sample up to 20 non-empty values
    sample = [v for v in values[:20] if v and str(v).strip()]
    if not sample:
        return "empty"
    
    # URL detection
    url_count = sum(1 for v in sample if isinstance(v, str) and 
                    ('http://' in v or 'https://' in v or 'drive.google.com' in v))
    if url_count > len(sample) * 0.7:
        return "url"
    
    # Date detection (common formats)
    date_patterns = ['/', '-', 'T']
    date_count = sum(1 for v in sample if isinstance(v, str) and 
                     any(p in v for p in date_patterns) and len(str(v)) > 6)
    if date_count > len(sample) * 0.7:
        return "date"
    
    # Boolean detection
    bool_values = {'true', 'false', 'yes', 'no', 'y', 'n', '1', '0', 'pass', 'fail'}
    bool_count = sum(1 for v in sample if str(v).lower().strip() in bool_values)
    if bool_count > len(sample) * 0.7:
        return "boolean"
    
    # Numeric detection
    try:
        numeric_count = sum(1 for v in sample if isinstance(v, (int, float)) or 
                           (isinstance(v, str) and v.replace('.', '', 1).replace('-', '', 1).isdigit()))
        if numeric_count > len(sample) * 0.7:
            return "numeric"
    except:
        pass
    
    # Reference detection (sheet name references, IDs)
    ref_keywords = ['sheet', 'dl1', 'dl2', 'data', 'id', 'form']
    ref_count = sum(1 for v in sample if isinstance(v, str) and 
                    any(kw in v.lower() for kw in ref_keywords))
    if ref_count > len(sample) * 0.5:
        return "reference"
    
    return "text"


def analyze_worksheet(worksheet: gspread.Worksheet) -> Dict[str, Any]:
    """
    Analyze a single worksheet structure.
    
    Args:
        worksheet: gspread Worksheet object
    
    Returns:
        Dictionary with worksheet metadata and column analysis
    """
    print(f"\n📊 Analyzing: {worksheet.title}")
    
    # Get all values
    all_values = worksheet.get_all_values()
    if not all_values:
        return {
            "title": worksheet.title,
            "row_count": 0,
            "col_count": 0,
            "headers": [],
            "columns": {},
            "category": "empty"
        }
    
    headers = all_values[0] if all_values else []
    data_rows = all_values[1:] if len(all_values) > 1 else []
    
    row_count = len(data_rows)
    col_count = len(headers)
    
    print(f"  Rows: {row_count} | Columns: {col_count}")
    
    # Analyze each column
    columns = {}
    for col_idx, header in enumerate(headers):
        if not header:
            continue
        
        # Extract column values
        column_values = [row[col_idx] if col_idx < len(row) else "" for row in data_rows]
        
        # Categorize
        col_type = categorize_column_type(column_values)
        
        # Count non-empty
        non_empty_count = sum(1 for v in column_values if v and str(v).strip())
        
        columns[header] = {
            "index": col_idx,
            "type": col_type,
            "non_empty_count": non_empty_count,
            "sample_values": [str(v)[:50] for v in column_values[:3] if v]
        }
        
        print(f"    {header:<40} | {col_type:>12} | {non_empty_count:>4} non-empty")
    
    # Categorize worksheet
    category = categorize_worksheet(worksheet.title, headers)
    
    return {
        "title": worksheet.title,
        "row_count": row_count,
        "col_count": col_count,
        "headers": headers,
        "columns": columns,
        "category": category
    }


def categorize_worksheet(title: str, headers: List[str]) -> str:
    """
    Categorize worksheet based on title and headers.
    
    Returns:
        tracking | dl1_data | dl2_data | lookup | other
    """
    title_lower = title.lower()
    
    # Tracking sheet patterns
    tracking_keywords = ['priority', 'sprint', 'status', 'progress', 'qc', 'workflow']
    if any(kw in title_lower for kw in ['tracking', 'workflow', 'qa', 'master']):
        return "tracking"
    if any(any(kw in h.lower() for kw in tracking_keywords) for h in headers):
        return "tracking"
    
    # DL1 data sheet
    if 'dl1' in title_lower and 'data' in title_lower:
        return "dl1_data"
    
    # DL2 data sheet
    if 'dl2' in title_lower and 'data' in title_lower:
        return "dl2_data"
    
    # Lookup tables
    if any(kw in title_lower for kw in ['lookup', 'reference', 'states', 'counties']):
        return "lookup"
    
    return "other"


def discover_google_sheets():
    """
    Main discovery function: connects to Google Sheets and analyzes structure.
    """
    # Validate configuration
    if not CREDS_PATH:
        print("❌ ERROR: GOOGLE_SERVICE_ACCOUNT_PATH not set in .env")
        return
    
    if not WORKBOOK_ID:
        print("❌ ERROR: GOOGLE_SHEETS_WORKBOOK_ID not set in .env")
        return
    
    if not Path(CREDS_PATH).exists():
        print(f"❌ ERROR: Credentials file not found: {CREDS_PATH}")
        return
    
    print("🔐 Authenticating with Google Sheets API...")
    
    # Authenticate
    try:
        credentials = Credentials.from_service_account_file(CREDS_PATH, scopes=SCOPES)
        client = gspread.authorize(credentials)
        print("✅ Authentication successful")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return
    
    # Open workbook
    print(f"\n📖 Opening workbook: {WORKBOOK_ID}")
    try:
        workbook = client.open_by_key(WORKBOOK_ID)
        print(f"✅ Opened: {workbook.title}")
    except Exception as e:
        print(f"❌ Failed to open workbook: {e}")
        return
    
    # List all worksheets
    worksheets = workbook.worksheets()
    print(f"\n📑 Found {len(worksheets)} worksheets")
    
    # Analyze each worksheet
    structure = {
        "discovery_date": datetime.now().isoformat(),
        "workbook_id": WORKBOOK_ID,
        "workbook_title": workbook.title,
        "worksheet_count": len(worksheets),
        "worksheets": {}
    }
    
    for ws in worksheets:
        analysis = analyze_worksheet(ws)
        structure["worksheets"][ws.title] = analysis
    
    # Categorization summary
    categories = {}
    for ws_name, ws_data in structure["worksheets"].items():
        cat = ws_data["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(ws_name)
    
    structure["categories"] = categories
    
    # Save to JSON
    print(f"\n💾 Saving structure to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(structure, f, indent=2, ensure_ascii=False)
    
    print("✅ Discovery complete!")
    
    # Print summary
    print("\n" + "="*70)
    print("📋 WORKSHEET CATEGORIZATION SUMMARY")
    print("="*70)
    for cat, sheets in sorted(categories.items()):
        print(f"\n{cat.upper()}:")
        for sheet in sheets:
            row_count = structure["worksheets"][sheet]["row_count"]
            print(f"  • {sheet} ({row_count} rows)")
    
    # Migration recommendations
    print("\n" + "="*70)
    print("💡 MIGRATION RECOMMENDATIONS")
    print("="*70)
    
    if "tracking" in categories:
        print("\n✅ TRACKING SHEET(S) FOUND:")
        for sheet in categories["tracking"]:
            print(f"  → {sheet}")
        print("  Recommendation: Migrate to PostgreSQL workflow.contests table")
    
    if "dl1_data" in categories:
        print("\n✅ DL1 DATA SHEET(S) FOUND:")
        for sheet in categories["dl1_data"]:
            print(f"  → {sheet}")
        print("  Recommendation: Migrate to PostgreSQL dl1.election_results table")
    
    if "dl2_data" in categories:
        print("\n✅ DL2 DATA SHEET(S) FOUND:")
        for sheet in categories["dl2_data"]:
            print(f"  → {sheet}")
        print("  Recommendation: Migrate to PostgreSQL dl2.election_results table")
    
    if "other" in categories or "lookup" in categories:
        print("\n⚠️  OTHER SHEETS DETECTED:")
        other_sheets = categories.get("other", []) + categories.get("lookup", [])
        for sheet in other_sheets:
            print(f"  → {sheet}")
        print("  Recommendation: Review manually before migration")
    
    print("\n" + "="*70)
    print(f"📄 Full structure saved to: {OUTPUT_FILE}")
    print("="*70)


if __name__ == "__main__":
    discover_google_sheets()
