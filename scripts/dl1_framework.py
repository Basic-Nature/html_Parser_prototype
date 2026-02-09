"""
DL1 Ground Truth Framework

Purpose: Define structure and tracking for the DL1 ground truth dataset.
         DL1 = Manually verified, approved election results (PROD Loaded status)

What We Know:
    - 279 contests have PROD Loaded status = verified and approved
    - Each has an official source URL (Source Link column)
    - Each has verification metadata (QC approvals, owner names, timestamps)
    
What We Need:
    - Actual election results data (candidate names, vote counts, etc.)
    - Location/source of this standardized data
    - How to link each contest to its verified data

Usage:
    python scripts/dl1_framework.py --create-schema
    python scripts/dl1_framework.py --import-data [--source-type google_sheets|csv|api]
    python scripts/dl1_framework.py --validate
    python scripts/dl1_framework.py --export-fixtures
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    'dbname': os.getenv('POSTGRES_DB', 'warehouse_election_results'),
    'user': os.getenv('POSTGRES_USER', 'postgres'),
    'password': os.getenv('POSTGRES_PASSWORD'),
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': os.getenv('POSTGRES_PORT', '5432')
}

OUTPUT_DIR = Path("webapp/parser/fixtures/dl1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_dl1_schema(conn):
    """
    Create dl1 schema and tables for storing ground truth election data.
    """
    print("\n📦 Creating DL1 Ground Truth Schema...")
    
    with conn.cursor() as cur:
        # Create DL1 schema
        cur.execute("CREATE SCHEMA IF NOT EXISTS dl1;")
        
        # DL1 Ground Truth Election Results
        cur.execute("""
            CREATE TABLE IF NOT EXISTS dl1.election_results (
                id SERIAL PRIMARY KEY,
                
                -- Link to workflow tracking
                contest_id INTEGER REFERENCES workflow.contests(id),
                
                -- Election identification  
                state VARCHAR(50) NOT NULL,
                county VARCHAR(100),
                year INTEGER NOT NULL,
                office VARCHAR(200) NOT NULL,
                race_type VARCHAR(100),
                election_date DATE NOT NULL,
                
                -- Candidate/option details
                candidate_name VARCHAR(200) NOT NULL,
                candidate_party VARCHAR(100),
                is_winner BOOLEAN,
                
                -- Results (ground truth)
                votes_total INTEGER NOT NULL,
                votes_percentage DECIMAL(5,2),
                write_in_votes INTEGER,
                
                -- Verification metadata
                verified_by VARCHAR(100) NOT NULL,
                verified_date TIMESTAMP NOT NULL DEFAULT NOW(),
                confidence_score DECIMAL(3,2) DEFAULT 1.0,
                confidence_notes TEXT,
                
                -- Source tracking
                source_name VARCHAR(200),
                source_url TEXT,
                source_document_type VARCHAR(50),  -- pdf, csv, website, etc.
                source_row_number INTEGER,
                
                -- Data standardization
                standardization_method VARCHAR(100),  -- manual_entry, ocr_extracted, etc.
                standardization_notes TEXT,
                
                -- Timestamps
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            );
        """)
        
        # DL1 Source Tracking (links contests to their source files)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS dl1.sources (
                id SERIAL PRIMARY KEY,
                
                -- Contest reference
                contest_id INTEGER NOT NULL REFERENCES workflow.contests(id),
                
                -- Source details
                source_name VARCHAR(200) NOT NULL,
                source_url TEXT,
                source_type VARCHAR(50),  -- pdf, csv, website, google_sheet, api, etc.
                
                -- Download/access details
                local_file_path TEXT,
                blob_storage_url TEXT,
                
                -- Metadata
                downloaded_at TIMESTAMP,
                downloaded_by VARCHAR(100),
                file_size_bytes INTEGER,
                file_hash VARCHAR(64),  -- SHA256 for integrity checking
                
                -- Verification
                source_verified BOOLEAN DEFAULT FALSE,
                source_verified_by VARCHAR(100),
                source_verified_at TIMESTAMP,
                
                -- Timestamps
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            );
        """)
        
        # DL1 Audit Trail (tracks changes and verifications)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS dl1.audit_trail (
                id SERIAL PRIMARY KEY,
                
                -- What was changed
                table_name VARCHAR(100),
                record_id INTEGER,
                action VARCHAR(50),  -- INSERT, UPDATE, VERIFICATION, etc.
                
                -- Who/When
                changed_by VARCHAR(100),
                changed_at TIMESTAMP DEFAULT NOW(),
                
                -- Details
                changes JSONB,
                notes TEXT,
                
                -- Timestamps
                created_at TIMESTAMP DEFAULT NOW()
            );
        """)
        
        # Create indexes for common queries
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_dl1_results_contest_id ON dl1.election_results(contest_id);
            CREATE INDEX IF NOT EXISTS idx_dl1_results_state_year ON dl1.election_results(state, year);
            CREATE INDEX IF NOT EXISTS idx_dl1_results_office ON dl1.election_results(office);
            CREATE INDEX IF NOT EXISTS idx_dl1_sources_contest_id ON dl1.sources(contest_id);
            CREATE INDEX IF NOT EXISTS idx_dl1_audit_table_record ON dl1.audit_trail(table_name, record_id);
        """)
        
        conn.commit()
        print("✅ DL1 schema created successfully")


def show_dl1_planning(conn):
    """
    Show planning information for DL1 data integration.
    """
    print("\n" + "="*70)
    print("📊 DL1 GROUND TRUTH DATASET PLANNING")
    print("="*70)
    
    with conn.cursor() as cur:
        # Get PROD records with their sources
        cur.execute("""
            SELECT 
                id, year, state, race,
                source_link, work_in_progress_dl1,
                qc_1_form, qc_2_form
            FROM workflow.contests 
            WHERE status = 'PROD Loaded'
            ORDER BY year DESC, state
            LIMIT 10
        """)
        
        print("\n📋 Sample PROD Loaded Records (first 10):")
        print("   Ready to link to verified election data")
        
        for row in cur.fetchall():
            cid, year, state, race, source, owner, qc1, qc2 = row
            print(f"\n   Contest #{cid}: {year} {state} {race}")
            print(f"      Source: {source[:60]}..." if len(str(source)) > 60 else f"      Source: {source}")
            print(f"      Verified by: {owner} (QC1: {qc1}, QC2: {qc2})")
    
    print("\n" + "="*70)
    print("🔧 DATA INTEGRATION WORKFLOW")
    print("="*70)
    print("""
STEP 1: Identify Data Source
    Where is the standardized election results stored?
    - Google Sheet ID? (provide spreadsheet URL)
    - CSV files? (provide folder path)
    - API endpoint? (provide documentation)
    - Already in database? (provide table name)
    
STEP 2: Import Data
    python scripts/dl1_framework.py --import-data --source-type [google_sheets|csv|api|database]
    
STEP 3: Validate Data
    - Check data integrity
    - Verify candidate names match official records
    - Validate vote totals
    
STEP 4: Create Audit Trail
    - Record who verified each record
    - Track sources and timestamps
    - Document standardization method
    
STEP 5: Export Fixtures
    - Create version-controlled JSON snapshots
    - For DATA_COMPARISON_ROADMAP verification
    - For DL1 vs DL2 accuracy tracking
    """)
    
    print("="*70)
    print("\n❓ QUESTION FOR YOU:")
    print("   Where should we get the actual verified election results data?")
    print("   (candidate names, vote counts, percentages for each PROD record)")
    print("\n   Please provide:")
    print("   1. Location: Google Sheet ID, CSV folder, API endpoint, etc.")
    print("   2. Structure: How are records organized?")
    print("   3. Format: Column names, data types")
    print("="*70 + "\n")


def export_dl1_index(conn):
    """
    Export index of PROD Loaded records ready for DL1 data integration.
    """
    print("\n📄 Exporting DL1 Index...")
    
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 
                id, year, state, county, race,
                source_link, work_in_progress_dl1,
                qc_1_form, qc_2_form,
                google_sheets_row_number
            FROM workflow.contests 
            WHERE status = 'PROD Loaded'
            ORDER BY year DESC, state, race
        """)
        
        columns = [desc[0] for desc in cur.description]
        records = [dict(zip(columns, row)) for row in cur.fetchall()]
    
    index = {
        "dataset": "DL1 Ground Truth Election Results",
        "version": "1.0",
        "created_at": datetime.now().isoformat(),
        "total_records": len(records),
        "status_filter": "PROD Loaded only",
        "description": "Production-ready, fully-validated election contests ready for data integration",
        "records": records
    }
    
    index_path = OUTPUT_DIR / "dl1_index.json"
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"✅ Index exported to: {index_path}")
    print(f"   Records ready for data import: {len(records)}")
    
    # Summary statistics
    years = sorted(set(r['year'] for r in records))
    states = sorted(set(r['state'] for r in records))
    owners = sorted(set(r['work_in_progress_dl1'] for r in records if r['work_in_progress_dl1']))
    
    print(f"   Years: {years}")
    print(f"   States: {len(states)}")
    print(f"   Verification team: {', '.join(owners)}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="DL1 Ground Truth Framework")
    parser.add_argument('--create-schema', action='store_true', help='Create DL1 schema')
    parser.add_argument('--plan', action='store_true', help='Show planning information')
    parser.add_argument('--export-index', action='store_true', help='Export DL1 index')
    args = parser.parse_args()
    
    conn = psycopg2.connect(**DB_CONFIG)
    
    if args.create_schema:
        create_dl1_schema(conn)
    
    if args.plan or not any([args.create_schema, args.export_index]):
        show_dl1_planning(conn)
    
    if args.export_index:
        export_dl1_index(conn)
    
    conn.close()


if __name__ == "__main__":
    main()
