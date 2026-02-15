"""
Google Sheets → PostgreSQL Migration Script

Purpose: Migrate Google Sheets election data to local PostgreSQL database.
         Preserves QA workflow metadata and creates DL1/DL2 datasets.

Usage:
    python scripts/migrate_google_sheets.py [--dry-run] [--tracking-only] [--data-only]

Options:
    --dry-run        : Preview migration without writing to database
    --tracking-only  : Only migrate tracking sheet (workflow.contests)
    --data-only      : Only migrate DL1/DL2 data sheets

Requirements:
    - PostgreSQL running (POSTGRES_* env vars in .env)
    - Google Service Account configured (GOOGLE_SERVICE_ACCOUNT_PATH)
    - Run discover_google_sheets.py first to understand structure

Database Schema:
    - workflow.contests       : QA tracking (Priority, Sprint, Status, QC metadata)
    - dl1.election_results    : Ground truth verified data
    - dl2.election_results    : Parser-extracted data
"""

import argparse
import os
import sys
from pathlib import Path

import gspread
import psycopg2
from dotenv import load_dotenv
from google.oauth2.service_account import Credentials

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


def create_schemas(conn):
    """
    Create PostgreSQL schemas for workflow, dl1, and dl2.
    """
    print("\n📦 Creating database schemas...")
    
    with conn.cursor() as cur:
        # Create schemas
        cur.execute("CREATE SCHEMA IF NOT EXISTS workflow;")
        cur.execute("CREATE SCHEMA IF NOT EXISTS dl1;")
        cur.execute("CREATE SCHEMA IF NOT EXISTS dl2;")
        
        # Create workflow.contests table (QA tracking)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS workflow.contests (
                id SERIAL PRIMARY KEY,
                
                -- Election identification
                year INTEGER,
                race VARCHAR(300),
                state VARCHAR(100),
                county VARCHAR(100),
                
                -- Workflow metadata
                priority VARCHAR(50),
                sprint VARCHAR(50),
                status VARCHAR(100),
                
                -- Work assignments
                work_in_progress_dl1 VARCHAR(100),
                work_in_progress_dl2 VARCHAR(100),
                
                -- Downloads and sources
                download_1 TEXT,
                download_2 TEXT,
                source_link TEXT,
                
                -- DL1 tracking
                dl1_data_sheet VARCHAR(200),
                dl1_data_source VARCHAR(200),
                dl1_complete BOOLEAN,
                
                -- DL2 tracking
                dl2_data_source VARCHAR(200),
                dl2_complete BOOLEAN,
                
                -- QA automation results
                run_candidate_check_dl1 BOOLEAN,
                candidate_check_results_dl1 VARCHAR(100),
                candidates_reviewed_dl1 VARCHAR(100),
                
                run_candidate_check_dl2 BOOLEAN,
                candidate_check_results_dl2 VARCHAR(100),
                candidates_reviewed_dl2 VARCHAR(100),
                
                run_pre_check BOOLEAN,
                pre_qc_results VARCHAR(100),
                
                -- QC Forms and status
                qc_1_form VARCHAR(200),
                database_upload BOOLEAN,
                qc_2_form VARCHAR(200),
                qc_db_loaded VARCHAR(100),
                
                -- Metadata
                migrated_from_google_sheets BOOLEAN DEFAULT TRUE,
                migrated_at TIMESTAMP DEFAULT NOW(),
                google_sheets_row_number INTEGER,
                
                -- Timestamps
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            );
        """)
        
        # Create dl1.election_results table (ground truth)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS dl1.election_results (
                id SERIAL PRIMARY KEY,
                
                -- Link to workflow tracking
                contest_id INTEGER REFERENCES workflow.contests(id),
                
                -- Election identification
                state VARCHAR(50) NOT NULL,
                county VARCHAR(100),
                office VARCHAR(200) NOT NULL,
                contest_name VARCHAR(300),
                election_date DATE NOT NULL,
                
                -- Candidate/option details
                candidate_name VARCHAR(200) NOT NULL,
                party VARCHAR(100),
                
                -- Results
                votes INTEGER NOT NULL,
                percentage DECIMAL(5,2),
                
                -- Verification metadata
                verified_by VARCHAR(100) NOT NULL,
                verified_date TIMESTAMP NOT NULL,
                confidence_score DECIMAL(3,2) DEFAULT 1.0,
                
                -- Source tracking
                source_sheet VARCHAR(100),
                source_row_number INTEGER,
                original_source_url TEXT,
                
                -- Timestamps
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            );
        """)
        
        # Create dl2.election_results table (parser-extracted)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS dl2.election_results (
                id SERIAL PRIMARY KEY,
                
                -- Link to workflow tracking
                contest_id INTEGER REFERENCES workflow.contests(id),
                
                -- Election identification (same as dl1)
                state VARCHAR(50) NOT NULL,
                county VARCHAR(100),
                office VARCHAR(200) NOT NULL,
                contest_name VARCHAR(300),
                election_date DATE NOT NULL,
                
                -- Candidate/option details
                candidate_name VARCHAR(200) NOT NULL,
                party VARCHAR(100),
                
                -- Results
                votes INTEGER NOT NULL,
                percentage DECIMAL(5,2),
                
                -- Extraction metadata
                extracted_by VARCHAR(100) DEFAULT 'parser',
                extraction_date TIMESTAMP NOT NULL,
                confidence_score DECIMAL(3,2),
                
                -- Source tracking
                source_sheet VARCHAR(100),
                source_row_number INTEGER,
                parser_version VARCHAR(50),
                
                -- Timestamps
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            );
        """)
        
        # Create indexes
        cur.execute("CREATE INDEX IF NOT EXISTS idx_contests_state_year ON workflow.contests(state, year);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_dl1_state_election_date ON dl1.election_results(state, election_date);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_dl2_state_election_date ON dl2.election_results(state, election_date);")
        
        conn.commit()
        print("✅ Schemas and tables created successfully")


def migrate_tracking_sheet(workbook: gspread.Spreadsheet, conn, dry_run: bool = False):
    """
    Migrate the main tracking sheet to workflow.contests table.
    
    Expected headers:
        Priority, Sprint, Status, Work in Progress - DL1, Work in Progress - DL2,
        Year, Race, State, Download 1, Download 2, Source Link, DL1 Data Sheet,
        Data Source, DL1 Complete, QC ID, Download 2, Data Source, DL2 Complete,
        Run Candidate Check DL1, Candidate Check Results DL1, Candidates Reviewed DL1,
        Run Candidate Check DL2, Candidate Check Results DL2, Candidates Reviewed DL2,
        Run Pre-Check, Pre-QC Results, QC 1 Form, Database Upload, QC 2 Form, QC DB Loaded
    """
    print("\n📋 Migrating tracking sheet...")
    
    # Find tracking sheet (usually first sheet or named "Tracking" / "Master")
    worksheet = workbook.get_worksheet(0)  # Assume first sheet is tracking
    print(f"  Using worksheet: {worksheet.title}")
    
    # Get all records
    records = worksheet.get_all_records()
    print(f"  Found {len(records)} records")
    
    if dry_run:
        print("\n  🔍 DRY RUN - Sample record:")
        if records:
            import json
            print(json.dumps(records[0], indent=2, default=str))
        return
    
    # Insert into database
    inserted = 0
    with conn.cursor() as cur:
        for idx, record in enumerate(records, start=2):  # Start at 2 (row 1 is header)
            try:
                cur.execute("""
                    INSERT INTO workflow.contests (
                        year, race, state, county,
                        priority, sprint, status,
                        work_in_progress_dl1, work_in_progress_dl2,
                        download_1, download_2, source_link,
                        dl1_data_sheet, dl1_data_source, dl1_complete,
                        dl2_data_source, dl2_complete,
                        run_candidate_check_dl1, candidate_check_results_dl1, candidates_reviewed_dl1,
                        run_candidate_check_dl2, candidate_check_results_dl2, candidates_reviewed_dl2,
                        run_pre_check, pre_qc_results,
                        qc_1_form, database_upload, qc_2_form, qc_db_loaded,
                        google_sheets_row_number
                    ) VALUES (
                        %(Year)s, %(Race)s, %(State)s, %(County)s,
                        %(Priority)s, %(Sprint)s, %(Status)s,
                        %(Work in Progress - DL1)s, %(Work in Progress - DL2)s,
                        %(Download 1)s, %(Download 2)s, %(Source Link)s,
                        %(DL1 Data Sheet)s, %(Data Source)s, %(DL1 Complete)s,
                        %(Data Source.1)s, %(DL2 Complete)s,
                        %(Run Candidate Check DL1)s, %(Candidate Check Results DL1)s, %(Candidates Reviewed DL1)s,
                        %(Run Candidate Check DL2)s, %(Candidate Check Results DL2)s, %(Candidates Reviewed DL2)s,
                        %(Run Pre-Check)s, %(Pre-QC Results)s,
                        %(QC 1 Form)s, %(Database Upload)s, %(QC 2 Form)s, %(QC DB Loaded)s,
                        %s
                    )
                """, {**record, 'google_sheets_row_number': idx})
                inserted += 1
            except KeyError as e:
                print(f"  ⚠️  Row {idx}: Missing column {e} - skipping")
            except Exception as e:
                print(f"  ❌ Row {idx}: Error - {e}")
    
    conn.commit()
    print(f"✅ Migrated {inserted}/{len(records)} tracking records")


def main():
    parser = argparse.ArgumentParser(description="Migrate Google Sheets to PostgreSQL")
    parser.add_argument('--dry-run', action='store_true', help='Preview migration without database writes')
    parser.add_argument('--tracking-only', action='store_true', help='Only migrate tracking sheet')
    parser.add_argument('--data-only', action='store_true', help='Only migrate DL1/DL2 data')
    args = parser.parse_args()
    
    # Validate configuration
    if not CREDS_PATH or not Path(CREDS_PATH).exists():
        print(f"❌ ERROR: Google Service Account credentials not found: {CREDS_PATH}")
        sys.exit(1)
    
    if not WORKBOOK_ID:
        print("❌ ERROR: GOOGLE_SHEETS_WORKBOOK_ID not set in .env")
        sys.exit(1)
    
    if not DB_CONFIG['password']:
        print("❌ ERROR: POSTGRES_PASSWORD not set in .env")
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
        print(f"\n🐘 Connecting to PostgreSQL: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}")
        conn = psycopg2.connect(**DB_CONFIG)
        print("✅ Database connected")
        
        # Create schemas
        create_schemas(conn)
    else:
        print("\n🔍 DRY RUN MODE - No database connection")
        conn = None
    
    # Migrate tracking sheet
    if not args.data_only:
        migrate_tracking_sheet(workbook, conn, dry_run=args.dry_run)
    
    # TODO: Migrate DL1/DL2 data sheets (requires discovery first to identify them)
    if not args.tracking_only and not args.dry_run:
        print("\n⚠️  DL1/DL2 data migration not yet implemented")
        print("   Run discover_google_sheets.py first to identify data sheets")
    
    # Cleanup
    if conn:
        conn.close()
    
    print("\n✅ Migration complete!")


if __name__ == "__main__":
    main()
