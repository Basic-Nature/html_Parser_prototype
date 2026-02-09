"""
Google Sheets "Overview" → PostgreSQL Migration Script

Purpose: Migrate the "Overview" worksheet which contains the complete QA tracking data.
         Handles the special two-row header structure (row 1 = section titles, row 2 = actual headers)

Usage:
    python scripts/migrate_overview_sheet.py [--dry-run] [--limit N]

Options:
    --dry-run  : Preview migration without writing to database
    --limit N  : Only migrate first N records (for testing)

Requirements:
    - PostgreSQL running (POSTGRES_* env vars in .env)
    - Google Service Account configured
    - "Overview" worksheet in workbook 1AnKXIi7fkP3FNzFSbPABSj_QYPY8WGu4ZGzwyW4A_Ac
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import gspread
import psycopg2
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Google Sheets configuration
SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets.readonly',
    'https://www.googleapis.com/auth/drive.readonly'
]
CREDS_PATH = os.getenv("GOOGLE_SERVICE_ACCOUNT_PATH")
WORKBOOK_ID = os.getenv("GOOGLE_SHEETS_WORKBOOK_ID")

# PostgreSQL configuration
DB_CONFIG = {
    'dbname': os.getenv('POSTGRES_DB', 'warehouse_election_results'),
    'user': os.getenv('POSTGRES_USER', 'postgres'),
    'password': os.getenv('POSTGRES_PASSWORD'),
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': os.getenv('POSTGRES_PORT', '5432')
}


def create_workflow_schema(conn):
    """
    Create workflow schema and contests table.
    """
    print("\n📦 Creating workflow schema and tables...")
    
    with conn.cursor() as cur:
        # Create schema
        cur.execute("CREATE SCHEMA IF NOT EXISTS workflow;")
        
        # Drop existing table if it exists (for clean migration)
        # Uncomment if you want to start fresh each time:
        # cur.execute("DROP TABLE IF EXISTS workflow.contests CASCADE;")
        
        # Create workflow.contests table with flexible columns
        cur.execute("""
            CREATE TABLE IF NOT EXISTS workflow.contests (
                id SERIAL PRIMARY KEY,
                
                -- Priority and status
                priority VARCHAR(50),
                sprint VARCHAR(50),
                status VARCHAR(100),
                hold VARCHAR(50),
                
                -- Work assignments
                work_in_progress_dl1 VARCHAR(100),
                work_in_progress_dl2 VARCHAR(100),
                
                -- Election metadata
                year INTEGER,
                race VARCHAR(300),
                state VARCHAR(100),
                county VARCHAR(100),
                
                -- Downloads and sources
                download_1 TEXT,
                download_2 TEXT,
                source_link TEXT,
                
                -- DL1 tracking
                dl1_data_sheet VARCHAR(200),
                data_source_dl1 VARCHAR(200),
                dl1_complete VARCHAR(100),
                qc_id VARCHAR(100),
                
                -- DL2 tracking  
                data_source_dl2 VARCHAR(200),
                dl2_complete VARCHAR(100),
                
                -- QA automation results - DL1
                run_candidate_check_dl1 VARCHAR(100),
                candidate_check_results_dl1 VARCHAR(200),
                candidates_reviewed_dl1 VARCHAR(200),
                
                -- QA automation results - DL2
                run_candidate_check_dl2 VARCHAR(100),
                candidate_check_results_dl2 VARCHAR(200),
                candidates_reviewed_dl2 VARCHAR(200),
                
                -- Pre-check
                run_pre_check VARCHAR(100),
                pre_qc_results VARCHAR(200),
                
                -- QC Forms and database status
                qc_1_form VARCHAR(200),
                database_upload VARCHAR(200),
                qc_2_form VARCHAR(200),
                qc_db_loaded VARCHAR(200),
                
                -- Additional fields (flexible for unknowns)
                additional_data JSONB,
                
                -- Migration metadata
                migrated_from_google_sheets BOOLEAN DEFAULT TRUE,
                migrated_at TIMESTAMP DEFAULT NOW(),
                google_sheets_row_number INTEGER,
                
                -- Timestamps
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            );
        """)
        
        # Create indexes
        cur.execute("CREATE INDEX IF NOT EXISTS idx_contests_state_year ON workflow.contests(state, year);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_contests_status ON workflow.contests(status);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_contests_priority ON workflow.contests(priority);")
        
        conn.commit()
        print("✅ Workflow schema and tables created")


def extract_headers_from_overview(worksheet: gspread.Worksheet) -> List[str]:
    """
    Extract actual column headers from row 2 of Overview worksheet.
    Row 1 contains section groupings (merged cells), row 2 has actual headers.
    
    Returns:
        List of column headers from row 2
    """
    all_values = worksheet.get_all_values()
    
    if len(all_values) < 2:
        raise ValueError("Overview worksheet doesn't have enough rows")
    
    # Row 2 (index 1) contains the actual headers
    headers = all_values[1]
    
    print(f"\n📋 Extracted {len([h for h in headers if h])} non-empty headers from row 2")
    print(f"   Sample headers: {[h for h in headers[:10] if h]}")
    
    return headers


def migrate_overview_worksheet(workbook: gspread.Spreadsheet, conn, dry_run: bool = False, limit: Optional[int] = None):
    """
    Migrate the Overview worksheet to workflow.contests table.
    
    Special handling:
    - Row 1: Section titles (skip)
    - Row 2: Column headers (use these)
    - Row 3+: Actual data
    """
    print("\n📋 Migrating 'Overview' worksheet...")
    
    # Get Overview worksheet
    try:
        worksheet = workbook.worksheet("Overview")
        print(f"  ✅ Found worksheet: {worksheet.title}")
    except gspread.WorksheetNotFound:
        print("  ❌ ERROR: 'Overview' worksheet not found!")
        return
    
    # Get all values
    all_values = worksheet.get_all_values()
    
    if len(all_values) < 3:
        print("  ⚠️  Not enough rows in Overview worksheet")
        return
    
    # Row 1: Section titles (skip)
    section_headers = all_values[0]
    
    # Row 2: Actual column headers
    headers = all_values[1]
    
    # Row 3+: Data
    data_rows = all_values[2:]
    
    print(f"  Section headers (row 1): {len([h for h in section_headers if h])} non-empty")
    print(f"  Column headers (row 2): {len([h for h in headers if h])} non-empty")
    print(f"  Data rows: {len(data_rows)}")
    
    # Apply limit if specified
    if limit:
        data_rows = data_rows[:limit]
        print(f"  ⚠️  Limited to first {limit} records for testing")
    
    # Build records as dictionaries
    records = []
    for row_idx, row in enumerate(data_rows, start=3):  # Start at 3 (row 1=sections, row 2=headers, row 3=first data)
        record = {}
        for col_idx, header in enumerate(headers):
            if header and col_idx < len(row):
                record[header] = row[col_idx] if row[col_idx] else None
            record['_row_number'] = row_idx
        records.append(record)
    
    print(f"  Processed {len(records)} records")
    
    if dry_run:
        print("\n  🔍 DRY RUN - Sample record:")
        if records:
            import json
            sample = {k: v for k, v in records[0].items() if v}  # Only show non-empty values
            print(json.dumps(sample, indent=2, default=str))
        
        print(f"\n  🔍 DRY RUN - All unique headers found:")
        unique_headers = sorted(set(h for h in headers if h))
        for i, h in enumerate(unique_headers, 1):
            print(f"    {i:2d}. {h}")
        
        return
    
    # Column name mapping (Google Sheets → PostgreSQL)
    COLUMN_MAP = {
        'Priority': 'priority',
        'Sprint': 'sprint',
        'Status': 'status',
        'Hold': 'hold',
        'Work in Progress - DL1': 'work_in_progress_dl1',
        'Work in Progress - DL2': 'work_in_progress_dl2',
        'Year': 'year',
        'Race': 'race',
        'State': 'state',
        'County': 'county',
        'Download 1': 'download_1',
        'Download 2': 'download_2',
        'Source Link': 'source_link',
        'DL1 Data Sheet': 'dl1_data_sheet',
        'Data Source': 'data_source_dl1',
        'DL1 Complete': 'dl1_complete',
        'QC ID': 'qc_id',
        'DL2 Complete': 'dl2_complete',
        'Run Candidate Check DL1': 'run_candidate_check_dl1',
        'Candidate Check Results DL1': 'candidate_check_results_dl1',
        'Candidates Reviewed DL1': 'candidates_reviewed_dl1',
        'Run Candidate Check DL2': 'run_candidate_check_dl2',
        'Candidate Check Results DL2': 'candidate_check_results_dl2',
        'Candidates Reviewed DL2': 'candidates_reviewed_dl2',
        'Run Pre-Check': 'run_pre_check',
        'Pre-QC Results': 'pre_qc_results',
        'QC 1 Form': 'qc_1_form',
        'Database Upload': 'database_upload',
        'QC 2 Form': 'qc_2_form',
        'QC DB Loaded': 'qc_db_loaded',
    }
    
    # Insert into database
    inserted = 0
    skipped = 0
    
    with conn.cursor() as cur:
        for record in records:
            # Convert Year to integer if present
            year_val = record.get('Year')
            if year_val and year_val.isdigit():
                year_val = int(year_val)
            else:
                year_val = None
            
            # Build mapped record
            mapped = {
                'year': year_val,
                'google_sheets_row_number': record.get('_row_number'),
            }
            
            # Map known columns
            for sheets_col, pg_col in COLUMN_MAP.items():
                if sheets_col in record:
                    mapped[pg_col] = record[sheets_col]
            
            # Store unmapped columns in JSONB
            additional = {}
            for sheets_col, value in record.items():
                if sheets_col not in COLUMN_MAP and sheets_col != '_row_number' and value:
                    additional[sheets_col] = value
            
            if additional:
                import json
                mapped['additional_data'] = json.dumps(additional)
            
            # Skip completely empty records
            if not any(v for k, v in mapped.items() if k not in ['google_sheets_row_number', 'additional_data']):
                skipped += 1
                continue
            
            try:
                # Build dynamic INSERT query
                columns = ', '.join(mapped.keys())
                placeholders = ', '.join(['%s'] * len(mapped))
                query = f"INSERT INTO workflow.contests ({columns}) VALUES ({placeholders})"
                
                cur.execute(query, list(mapped.values()))
                inserted += 1
                
            except Exception as e:
                print(f"  ❌ Row {record.get('_row_number')}: {e}")
    
    conn.commit()
    print(f"\n✅ Migration complete:")
    print(f"   Inserted: {inserted} records")
    print(f"   Skipped:  {skipped} empty records")


def main():
    parser = argparse.ArgumentParser(description="Migrate Google Sheets 'Overview' to PostgreSQL")
    parser.add_argument('--dry-run', action='store_true', help='Preview migration without database writes')
    parser.add_argument('--limit', type=int, help='Only migrate first N records (for testing)')
    args = parser.parse_args()
    
    # Validate configuration
    if not CREDS_PATH or not Path(CREDS_PATH).exists():
        print(f"❌ ERROR: Google Service Account credentials not found: {CREDS_PATH}")
        sys.exit(1)
    
    if not WORKBOOK_ID:
        print("❌ ERROR: GOOGLE_SHEETS_WORKBOOK_ID not set in .env")
        sys.exit(1)
    
    # Connect to Google Sheets
    print("🔐 Authenticating with Google Sheets API...")
    credentials = Credentials.from_service_account_file(CREDS_PATH, scopes=SCOPES)
    client = gspread.authorize(credentials)
    
    print(f"📖 Opening workbook: {WORKBOOK_ID}")
    workbook = client.open_by_key(WORKBOOK_ID)
    print(f"✅ Opened: {workbook.title}")
    
    # Connect to PostgreSQL
    if not args.dry_run:
        if not DB_CONFIG['password']:
            print("❌ ERROR: POSTGRES_PASSWORD not set in .env")
            sys.exit(1)
        
        print(f"\n🐘 Connecting to PostgreSQL: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}")
        conn = psycopg2.connect(**DB_CONFIG)
        print("✅ Database connected")
        
        # Create schema
        create_workflow_schema(conn)
    else:
        print("\n🔍 DRY RUN MODE - No database connection")
        conn = None
    
    # Migrate Overview worksheet
    migrate_overview_worksheet(workbook, conn, dry_run=args.dry_run, limit=args.limit)
    
    # Cleanup
    if conn:
        conn.close()
    
    print("\n✅ Process complete!")


if __name__ == "__main__":
    main()
