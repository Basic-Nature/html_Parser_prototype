"""
Import DL1 and DL2 Election Data from Google Drive

Purpose: Import verified (DL1) and parser-extracted (DL2) election data into PostgreSQL.
         Links each Google Sheet to its workflow.contests record and imports all candidate results.

Architecture:
    DL1 Folder: 1ZwsL_Ui2qFyV-EJ1OZ_9lyhMeX8d4v9N (Ground truth, manually verified)
    DL2 Folder: 1wQcC_UEIFQrIYyRhyfgY2rr5RkiCBQ7V (Parser-extracted, for comparison)
    
    Filename pattern: "YYYY - RACE - STATE - DL1/DL2"
    Example: "2012 - Attorney General - Oregon - DL1"
    
    Sheet structure: "Standardized" worksheet with columns:
        - County
        - Ballot Candidate Name
        - Ballot Party
        - Uncategorized Votes (or vote type columns)
        - Write-In flag

Usage:
    python scripts/import_dl_data.py --match-files       # Match sheets to contests
    python scripts/import_dl_data.py --import-dl1 --limit 5   # Import first 5 DL1 sheets
    python scripts/import_dl_data.py --import-dl2 --limit 5   # Import first 5 DL2 sheets
    python scripts/import_dl_data.py --import-all         # Import all (279 sheets each)
    python scripts/import_dl_data.py --verify             # Verify import results

Options:
    --match-files   : Match Google Sheets to workflow.contests
    --import-dl1    : Import DL1 ground truth data
    --import-dl2    : Import DL2 parser-extracted data
    --import-all    : Import both DL1 and DL2
    --limit N       : Only import first N sheets (for testing)
    --verify        : Show import statistics
"""

import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import gspread
import psycopg2
from dotenv import load_dotenv
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

# Load environment
load_dotenv()

# Google API configuration
SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets.readonly',
    'https://www.googleapis.com/auth/drive.readonly'
]
CREDS_PATH = os.getenv("GOOGLE_SERVICE_ACCOUNT_PATH") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
RATE_LIMIT_SECONDS = float(os.getenv("DL_IMPORT_RATE_LIMIT_SECONDS", "1.1"))
MAX_SHEETS_RETRIES = int(os.getenv("DL_IMPORT_MAX_RETRIES", "5"))

# Folder IDs
DL1_FOLDER_ID = os.getenv("DL1_FOLDER_ID", "1ZwsL_Ui2qFyV-EJ1OZ_9lyhMeX8d4v9N")
DL2_FOLDER_ID = os.getenv("DL2_FOLDER_ID", "1wQcC_UEIFQrIYyRhyfgY2rr5RkiCBQ7V")

# PostgreSQL configuration
DB_CONFIG = {
    'dbname': os.getenv('POSTGRES_DB', 'warehouse_election_results'),
    'user': os.getenv('POSTGRES_USER', 'postgres'),
    'password': os.getenv('POSTGRES_PASSWORD'),
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': os.getenv('POSTGRES_PORT', '5432')
}


def parse_filename(filename: str) -> Optional[Dict[str, str]]:
    """
    Parse DL1/DL2 filename to extract metadata.
    
    Format: "YYYY - RACE - STATE - DL1/DL2"
    Example: "2012 - Attorney General - Oregon - DL1"
    
    Returns:
        Dict with year, race, state, dl_type or None if parse fails
    """
    # Remove file extension if present
    filename = filename.replace('.gsheet', '').strip()
    
    # Pattern: YEAR - RACE - STATE - DL#
    match = re.match(r'^(\d{4})\s*-\s*(.+?)\s*-\s*(.+?)\s*-\s*(DL[12])$', filename)
    
    if match:
        return {
            'year': int(match.group(1)),
            'race': match.group(2).strip(),
            'state': match.group(3).strip(),
            'dl_type': match.group(4)
        }
    
    return None


def list_drive_folder(drive_service, folder_id: str) -> List[Dict[str, Any]]:
    """
    List all Google Sheets in a Drive folder.
    """
    try:
        query = f"'{folder_id}' in parents and trashed=false"
        results = drive_service.files().list(
            q=query,
            fields="files(id, name, mimeType)",
            pageSize=1000,
            orderBy='name'
        ).execute()
        
        files = results.get('files', [])
        
        # Filter to only spreadsheets
        sheets = [f for f in files if 'spreadsheet' in f.get('mimeType', '')]
        
        return sheets
        
    except Exception as e:
        print(f"❌ Error listing folder: {e}")
        return []


def match_sheet_to_contest(sheet_meta: Dict[str, str], conn) -> Optional[int]:
    """
    Match a Google Sheet to its workflow.contests record.
    
    Args:
        sheet_meta: Parsed filename metadata (year, race, state)
        conn: PostgreSQL connection
    
    Returns:
        contest_id if matched, None otherwise
    """
    with conn.cursor() as cur:
        # Try exact match first
        cur.execute("""
            SELECT id FROM workflow.contests
            WHERE year = %s
              AND LOWER(race) = LOWER(%s)
              AND LOWER(state) = LOWER(%s)
              AND status = 'PROD Loaded'
            LIMIT 1
        """, (sheet_meta['year'], sheet_meta['race'], sheet_meta['state']))
        
        result = cur.fetchone()
        if result:
            return result[0]
        
        # Try fuzzy match (race might have variations)
        cur.execute("""
            SELECT id, race FROM workflow.contests
            WHERE year = %s
              AND LOWER(state) = LOWER(%s)
              AND status = 'PROD Loaded'
        """, (sheet_meta['year'], sheet_meta['state']))
        
        # Simple fuzzy matching on race name
        for row in cur.fetchall():
            contest_id, contest_race = row
            if sheet_meta['race'].lower() in contest_race.lower() or \
               contest_race.lower() in sheet_meta['race'].lower():
                return contest_id
        
    return None


def _should_skip_existing(conn, table_name: str, contest_id: int, source_name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT COUNT(*) FROM {table_name} WHERE contest_id = %s AND source_name = %s",
            (contest_id, source_name)
        )
        return cur.fetchone()[0] > 0


def _delete_existing(conn, table_name: str, contest_id: int, source_name: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            f"DELETE FROM {table_name} WHERE contest_id = %s AND source_name = %s",
            (contest_id, source_name)
        )
        return cur.rowcount


def _get_all_values_with_retry(ws) -> List[List[str]]:
    delay = 2.0
    for attempt in range(1, MAX_SHEETS_RETRIES + 1):
        try:
            return ws.get_all_values()
        except gspread.exceptions.APIError as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code == 429 or "429" in str(exc):
                print(f"  ⏳ Rate limit hit (attempt {attempt}/{MAX_SHEETS_RETRIES}). Sleeping {delay:.1f}s...")
                time.sleep(delay)
                delay *= 2
                continue
            raise
    raise RuntimeError("Exceeded retry budget while reading sheet data")


def import_sheet_data(gspread_client, sheet_id: str, sheet_name: str, contest_id: int, 
                     dl_type: str, conn, verified_by: str = None, replace_existing: bool = False) -> int:
    """
    Import data from a single Google Sheet into dl1 or dl2 table.
    
    Args:
        gspread_client: Authenticated gspread client
        sheet_id: Google Sheet ID
        sheet_name: Sheet name for logging
        contest_id: workflow.contests.id to link to
        dl_type: 'DL1' or 'DL2'
        conn: PostgreSQL connection
        verified_by: Name of person who verified (for DL1)
    
    Returns:
        Number of rows imported
    """
    table_name = "dl1.election_results" if dl_type == "DL1" else "dl2.election_results"
    
    try:
        if _should_skip_existing(conn, table_name, contest_id, sheet_name):
            if not replace_existing:
                return -1
            deleted = _delete_existing(conn, table_name, contest_id, sheet_name)
            print(f"  🧹 Removed {deleted} existing rows before re-import")
        # Open sheet
        sheet = gspread_client.open_by_key(sheet_id)
        
        # Get "Standardized" worksheet
        try:
            ws = sheet.worksheet("Standardized")
        except gspread.WorksheetNotFound:
            # Try first worksheet
            ws = sheet.get_worksheet(0)
        
        # Get all data (with retry/backoff for 429s)
        all_values = _get_all_values_with_retry(ws)
        
        if not all_values or len(all_values) < 2:
            print("  ⚠️  No data found in sheet")
            return 0
        
        headers = all_values[0]
        data_rows = all_values[1:]
        
        # Find key columns
        col_map = {}
        for i, header in enumerate(headers):
            h_lower = header.lower().strip()
            if 'county' in h_lower:
                col_map['county'] = i
            elif 'candidate' in h_lower and 'name' in h_lower:
                col_map['candidate'] = i
            elif 'party' in h_lower:
                col_map['party'] = i
            elif 'uncategorized' in h_lower and 'vote' in h_lower:
                col_map['votes'] = i
            elif 'write' in h_lower:
                col_map['write_in'] = i
        
        # Import rows
        imported = 0
        with conn.cursor() as cur:
            for row in data_rows:
                if not any(row):  # Skip empty rows
                    continue
                
                # Extract values
                county = row[col_map.get('county', 0)] if col_map.get('county') is not None else None
                candidate = row[col_map.get('candidate', 1)] if col_map.get('candidate') is not None else None
                party = row[col_map.get('party', 2)] if col_map.get('party') is not None else None
                votes_str = row[col_map.get('votes', 3)] if col_map.get('votes') is not None else '0'
                write_in_str = row[col_map.get('write_in', -1)] if col_map.get('write_in') is not None else 'FALSE'
                
                if not candidate or not votes_str:
                    continue
                
                # Parse votes (remove commas, convert to int)
                try:
                    votes = int(votes_str.replace(',', '').replace(' ', '').strip())
                except (AttributeError, TypeError, ValueError):
                    votes = 0
                
                # Parse write-in flag
                is_write_in = write_in_str.upper() in ('TRUE', 'YES', '1')
                
                # Insert into appropriate table
                if dl_type == "DL1":
                    cur.execute("""
                        INSERT INTO dl1.election_results (
                            contest_id, state, county, year, office, election_date, candidate_name, 
                            candidate_party, votes_total, write_in_votes,
                            verified_by, verified_date, confidence_score,
                            source_name, standardization_method
                        ) VALUES (
                            %s, 
                            (SELECT state FROM workflow.contests WHERE id = %s),
                            %s,
                            (SELECT year FROM workflow.contests WHERE id = %s),
                            (SELECT race FROM workflow.contests WHERE id = %s),
                            ((SELECT year FROM workflow.contests WHERE id = %s) || '-11-05')::date,
                            %s, %s, %s,
                            %s,
                            %s, NOW(), 1.0,
                            %s, 'manual_standardization'
                        )
                    """, (
                        contest_id, contest_id, county, contest_id, contest_id, contest_id,
                        candidate, party, votes,
                        votes if is_write_in else None,
                        verified_by, sheet_name
                    ))
                else:  # DL2
                    cur.execute("""
                        INSERT INTO dl2.election_results (
                            contest_id, state, county, year, office, election_date, candidate_name,
                            candidate_party, votes_total, write_in_votes,
                            extracted_by, extraction_date, confidence_score,
                            source_name, parser_version
                        ) VALUES (
                            %s,
                            (SELECT state FROM workflow.contests WHERE id = %s),
                            %s,
                            (SELECT year FROM workflow.contests WHERE id = %s),
                            (SELECT race FROM workflow.contests WHERE id = %s),
                            ((SELECT year FROM workflow.contests WHERE id = %s) || '-11-05')::date,
                            %s, %s, %s,
                            %s,
                            'parser', NOW(), NULL,
                            %s, '1.0'
                        )
                    """, (
                        contest_id, contest_id, county, contest_id, contest_id, contest_id,
                        candidate, party, votes,
                        votes if is_write_in else None,
                        sheet_name
                    ))
                
                imported += 1
        
        conn.commit()
        return imported
        
    except Exception as e:
        print(f"  ❌ Error importing {sheet_name}: {e}")
        conn.rollback()
        return 0


def match_files_to_contests(drive_service, gspread_client, conn, dl_type: str):
    """
    Match all Google Sheets in a folder to workflow.contests records.
    """
    folder_id = DL1_FOLDER_ID if dl_type == "DL1" else DL2_FOLDER_ID
    folder_name = "DL1 (Ground Truth)" if dl_type == "DL1" else "DL2 (Parser-Extracted)"
    
    print(f"\n{'='*70}")
    print(f"🔗 Matching {folder_name} Sheets to Contests")
    print(f"{'='*70}")
    
    # List all sheets in folder
    sheets = list_drive_folder(drive_service, folder_id)
    print(f"\n✅ Found {len(sheets)} Google Sheets")
    
    # Match each to a contest
    matched = 0
    unmatched = []
    
    for sheet in sheets:
        sheet_name = sheet['name']
        # Parse filename
        meta = parse_filename(sheet_name)
        if not meta:
            print(f"\n⚠️  Could not parse: {sheet_name}")
            unmatched.append(sheet_name)
            continue
        
        # Find matching contest
        contest_id = match_sheet_to_contest(meta, conn)
        
        if contest_id:
            print(f"\n✅ {sheet_name}")
            print(f"   → Contest #{contest_id}: {meta['year']} {meta['state']} {meta['race']}")
            matched += 1
        else:
            print(f"\n❌ No match: {sheet_name}")
            unmatched.append(sheet_name)
    
    print(f"\n{'='*70}")
    print("📊 Matching Results:")
    print(f"   Matched:   {matched}/{len(sheets)} sheets")
    print(f"   Unmatched: {len(unmatched)}")
    
    if unmatched and len(unmatched) <= 10:
        print("\n   Unmatched sheets:")
        for name in unmatched:
            print(f"      - {name}")
    
    print(f"{'='*70}\n")


def import_dl_data(
    drive_service,
    gspread_client,
    conn,
    dl_type: str,
    limit: Optional[int] = None,
    rate_limit_seconds: float = RATE_LIMIT_SECONDS,
    replace_existing: bool = False
):
    """
    Import DL1 or DL2 data from Google Drive folder.
    """
    folder_id = DL1_FOLDER_ID if dl_type == "DL1" else DL2_FOLDER_ID
    folder_name = "DL1 (Ground Truth)" if dl_type == "DL1" else "DL2 (Parser-Extracted)"
    
    print(f"\n{'='*70}")
    print(f"📥 Importing {folder_name} Data")
    print(f"{'='*70}")
    
    # List all sheets
    sheets = list_drive_folder(drive_service, folder_id)
    if limit:
        sheets = sheets[:limit]
        print(f"\n⚠️  Limited to first {limit} sheets for testing")
    
    print(f"\n✅ Processing {len(sheets)} Google Sheets")
    
    # Import each sheet
    total_imported = 0
    total_sheets = 0
    skipped = 0
    
    for i, sheet in enumerate(sheets, 1):
        sheet_name = sheet['name']
        sheet_id = sheet['id']
        
        print(f"\n[{i}/{len(sheets)}] {sheet_name}")
        
        # Parse filename
        meta = parse_filename(sheet_name)
        if not meta:
            print("  ⚠️  Could not parse filename, skipping")
            skipped += 1
            continue
        
        # Find matching contest
        contest_id = match_sheet_to_contest(meta, conn)
        if not contest_id:
            print("  ⚠️  No matching contest found, skipping")
            skipped += 1
            continue
        
        print(f"  Contest #{contest_id}: {meta['year']} {meta['state']} {meta['race']}")
        
        # Get verified_by from contest if DL1
        verified_by = None
        if dl_type == "DL1":
            with conn.cursor() as cur:
                cur.execute("SELECT work_in_progress_dl1 FROM workflow.contests WHERE id = %s", (contest_id,))
                result = cur.fetchone()
                if result:
                    verified_by = result[0]
        
        # Import data
        rows_imported = import_sheet_data(
            gspread_client,
            sheet_id,
            sheet_name,
            contest_id,
            dl_type,
            conn,
            verified_by,
            replace_existing=replace_existing
        )
        
        if rows_imported == -1:
            print("  ⏭️  Already imported, skipping")
            skipped += 1
        elif rows_imported > 0:
            print(f"  ✅ Imported {rows_imported} rows")
            total_imported += rows_imported
            total_sheets += 1
        else:
            print("  ⚠️  No data imported")
            skipped += 1
        
        if rate_limit_seconds > 0:
            time.sleep(rate_limit_seconds)
    
    print(f"\n{'='*70}")
    print("📊 Import Results:")
    print(f"   Sheets processed: {total_sheets}")
    print(f"   Total rows:       {total_imported}")
    print(f"   Skipped:          {skipped}")
    print(f"{'='*70}\n")


def verify_import(conn):
    """
    Show statistics on imported data.
    """
    print(f"\n{'='*70}")
    print("📊 IMPORT VERIFICATION")
    print(f"{'='*70}")
    
    with conn.cursor() as cur:
        # DL1 stats
        cur.execute("""
            SELECT 
                COUNT(*) as total_rows,
                COUNT(DISTINCT contest_id) as contests,
                COUNT(DISTINCT state) as states,
                COUNT(DISTINCT year) as years,
                SUM(votes_total) as total_votes
            FROM dl1.election_results
        """)
        result = cur.fetchone()
        
        print("\n✅ DL1 (Ground Truth):")
        print(f"   Total rows:    {result[0]:,}")
        print(f"   Contests:      {result[1]}")
        print(f"   States:        {result[2]}")
        print(f"   Years:         {result[3]}")
        print(f"   Total votes:   {result[4]:,}" if result[4] else "   Total votes:   0")
        
        # DL2 stats
        cur.execute("""
            SELECT 
                COUNT(*) as total_rows,
                COUNT(DISTINCT contest_id) as contests,
                COUNT(DISTINCT state) as states,
                COUNT(DISTINCT year) as years,
                SUM(votes_total) as total_votes
            FROM dl2.election_results
        """)
        result = cur.fetchone()
        
        print("\n✅ DL2 (Parser-Extracted):")
        print(f"   Total rows:    {result[0]:,}")
        print(f"   Contests:      {result[1]}")
        print(f"   States:        {result[2]}")
        print(f"   Years:         {result[3]}")
        print(f"   Total votes:   {result[4]:,}" if result[4] else "   Total votes:   0")
        
        # Sample comparison
        print("\n📋 Sample DL1 vs DL2 Match:")
        cur.execute("""
            SELECT 
                c.year, c.state, c.race,
                COUNT(DISTINCT dl1.id) as dl1_rows,
                COUNT(DISTINCT dl2.id) as dl2_rows
            FROM workflow.contests c
            LEFT JOIN dl1.election_results dl1 ON dl1.contest_id = c.id
            LEFT JOIN dl2.election_results dl2 ON dl2.contest_id = c.id
            WHERE c.status = 'PROD Loaded'
              AND (dl1.id IS NOT NULL OR dl2.id IS NOT NULL)
            GROUP BY c.id, c.year, c.state, c.race
            ORDER BY c.year DESC, c.state
            LIMIT 10
        """)
        
        for row in cur.fetchall():
            year, state, race, dl1_count, dl2_count = row
            print(f"\n   {year} {state} {race}")
            print(f"      DL1: {dl1_count} rows | DL2: {dl2_count} rows")
    
    print(f"\n{'='*70}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Import DL1/DL2 Election Data")
    parser.add_argument('--match-files', action='store_true', help='Match sheets to contests')
    parser.add_argument('--import-dl1', action='store_true', help='Import DL1 data')
    parser.add_argument('--import-dl2', action='store_true', help='Import DL2 data')
    parser.add_argument('--import-all', action='store_true', help='Import both DL1 and DL2')
    parser.add_argument('--verify', action='store_true', help='Verify import statistics')
    parser.add_argument('--limit', type=int, help='Limit import to first N sheets')
    parser.add_argument('--rate-limit-seconds', type=float, default=RATE_LIMIT_SECONDS,
                        help='Sleep between sheets to avoid API rate limits (default: 1.1)')
    parser.add_argument('--replace-existing', action='store_true',
                        help='Delete and re-import if rows already exist for a sheet')
    args = parser.parse_args()
    
    # Validate configuration
    if not CREDS_PATH or not Path(CREDS_PATH).exists():
        print(f"❌ ERROR: Google Service Account credentials not found: {CREDS_PATH}")
        return
    
    # Connect to Google APIs
    print("🔐 Authenticating with Google APIs...")
    credentials = Credentials.from_service_account_file(CREDS_PATH, scopes=SCOPES)
    drive_service = build('drive', 'v3', credentials=credentials)
    gspread_client = gspread.authorize(credentials)
    print("✅ Authentication successful")
    
    # Connect to PostgreSQL
    print("\n🐘 Connecting to PostgreSQL...")
    conn = psycopg2.connect(**DB_CONFIG)
    print("✅ Database connected")
    
    # Execute requested operations
    if args.match_files or (not args.import_dl1 and not args.import_dl2 and not args.import_all and not args.verify):
        match_files_to_contests(drive_service, gspread_client, conn, "DL1")
        match_files_to_contests(drive_service, gspread_client, conn, "DL2")
    
    if args.import_dl1 or args.import_all:
        import_dl_data(
            drive_service,
            gspread_client,
            conn,
            "DL1",
            limit=args.limit,
            rate_limit_seconds=args.rate_limit_seconds,
            replace_existing=args.replace_existing
        )
    
    if args.import_dl2 or args.import_all:
        import_dl_data(
            drive_service,
            gspread_client,
            conn,
            "DL2",
            limit=args.limit,
            rate_limit_seconds=args.rate_limit_seconds,
            replace_existing=args.replace_existing
        )
    
    if args.verify:
        verify_import(conn)
    
    # Cleanup
    conn.close()
    print("\n✅ Process complete!\n")


if __name__ == "__main__":
    main()
