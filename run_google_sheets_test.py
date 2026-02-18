"""
Quick test script to pull data from Google Sheets using the google_service_account.json file
This will show you what data is in the sheets and help debug state/county dropdown issues
"""

import os
import sys
import io
from pathlib import Path

# Force UTF-8 encoding for console output on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Set up to use the google_service_account.json file
json_file = Path(__file__).parent / "google_service_account.json"
if json_file.exists():
    os.environ['GOOGLE_SHEETS_ELECTION_DB_LITE_CREDENTIALS'] = str(json_file)
    print(f"[OK] Using credentials from: {json_file}\n")
else:
    print(f"[ERROR] google_service_account.json not found at: {json_file}\n")
    sys.exit(1)

# You need to set the Google Sheets IDs
# Replace these with your actual spreadsheet IDs:
WORKLIST_ID = "1AnKXIi7fkP3FNzFSbPABSj_QYPY8WGu4ZGzwyW4A_Ac"  # SE Data Standardization - Full Worklist
DB_LITE_ID = "154z24Y7z99Yb9I7Swa8-6mKBgWKN44Ap9ukx5WEgLRY"   # SMART Elections Database-Lite

os.environ['GOOGLE_SHEETS_WORKLIST_ID'] = WORKLIST_ID
os.environ['GOOGLE_SHEETS_DB_LITE_ID'] = DB_LITE_ID
os.environ['GOOGLE_SHEETS_WORKLIST_OVERVIEW_SHEET'] = 'Overview'

print(f"📊 Testing with:")
print(f"   Worklist ID: {WORKLIST_ID}")
print(f"   DB-Lite ID:  {DB_LITE_ID}\n")

# Now run the full test suite
import test_google_sheets_data_pull
