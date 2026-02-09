"""
Build DL1 Ground Truth Dataset

Purpose: Extract only PROD-Loaded (validated) records and prepare them for DL1 dataset.
         DL1 = Ground truth, manually verified, production-ready election data.

Logic:
    - DL1 records must have status = "PROD Loaded"
    - Only 279 out of 408 records qualify
    - Other records are in-progress and have validation issues

Usage:
    python scripts/build_dl1_dataset.py [--analyze] [--export-manifest]

Options:
    --analyze            : Show which records qualify for DL1
    --export-manifest    : Export JSON manifest of DL1-ready records
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any

import psycopg2
from dotenv import load_dotenv

# Load environment
load_dotenv()

# PostgreSQL configuration
DB_CONFIG = {
    'dbname': os.getenv('POSTGRES_DB', 'warehouse_election_results'),
    'user': os.getenv('POSTGRES_USER', 'postgres'),
    'password': os.getenv('POSTGRES_PASSWORD'),
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': os.getenv('POSTGRES_PORT', '5432')
}

OUTPUT_DIR = Path("webapp/parser/fixtures/dl1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def analyze_dl1_readiness():
    """
    Analyze which records are ready for DL1 dataset.
    """
    print("\n" + "="*70)
    print("📊 DL1 GROUND TRUTH DATASET READINESS ANALYSIS")
    print("="*70)
    
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Get PROD Loaded records
    cur.execute("""
        SELECT id, year, race, state, 
               priority, status, 
               work_in_progress_dl1, work_in_progress_dl2,
               dl1_complete, dl2_complete,
               qc_1_form, qc_2_form,
               download_1, source_link
        FROM workflow.contests 
        WHERE status = 'PROD Loaded'
        ORDER BY year DESC, state, race
    """)
    
    prod_records = cur.fetchall()
    
    print(f"\n✅ PROD Loaded Records: {len(prod_records)}")
    print(f"   These are validated, production-ready ground truth records")
    
    # Breakdown by year
    print("\n📋 PROD Loaded by Year:")
    cur.execute("""
        SELECT year, COUNT(*) as count
        FROM workflow.contests 
        WHERE status = 'PROD Loaded'
        GROUP BY year
        ORDER BY year DESC
    """)
    for row in cur.fetchall():
        print(f"   {row[0]}: {row[1]} records")
    
    # Breakdown by state
    print("\n📋 PROD Loaded by State (Top 10):")
    cur.execute("""
        SELECT state, COUNT(*) as count
        FROM workflow.contests 
        WHERE status = 'PROD Loaded'
        GROUP BY state
        ORDER BY count DESC
        LIMIT 10
    """)
    for row in cur.fetchall():
        print(f"   {row[0]}: {row[1]}")
    
    # QC completion for PROD records
    print("\n✅ QC Status for PROD Loaded Records:")
    cur.execute("""
        SELECT 
            COUNT(CASE WHEN dl1_complete = 'TRUE' THEN 1 END) as dl1_done,
            COUNT(CASE WHEN dl2_complete = 'TRUE' THEN 1 END) as dl2_done,
            COUNT(CASE WHEN qc_1_form IS NOT NULL AND qc_1_form != '' THEN 1 END) as qc1_done,
            COUNT(CASE WHEN qc_2_form IS NOT NULL AND qc_2_form != '' THEN 1 END) as qc2_done
        FROM workflow.contests
        WHERE status = 'PROD Loaded'
    """)
    result = cur.fetchone()
    total = len(prod_records)
    print(f"   DL1 Complete:  {result[0]}/{total} ({result[0]*100//total}%)")
    print(f"   DL2 Complete:  {result[1]}/{total} ({result[1]*100//total}%)")
    print(f"   QC 1 Done:     {result[2]}/{total} ({result[2]*100//total}%)")
    print(f"   QC 2 Done:     {result[3]}/{total} ({result[3]*100//total}%)")
    
    # In-Progress records (not yet ready)
    print(f"\n⏳ Other Records (Not Yet Production Ready): {408 - len(prod_records)}")
    cur.execute("""
        SELECT status, COUNT(*) as count
        FROM workflow.contests 
        WHERE status != 'PROD Loaded'
        GROUP BY status
        ORDER BY count DESC
    """)
    for row in cur.fetchall():
        print(f"   {row[0]}: {row[1]}")
    
    # Sample PROD records
    print("\n📋 Sample PROD Loaded Records:")
    for i, record in enumerate(prod_records[:5], 1):
        rid, year, race, state, priority, status, dl1_owner, dl2_owner, dl1_done, dl2_done, qc1, qc2, dl1_ref, source = record
        print(f"\n   {i}. {year} | {race} | {state}")
        print(f"      DL1 Owner: {dl1_owner} | DL2 Owner: {dl2_owner}")
        print(f"      DL1 Ref: {dl1_ref}")
    
    conn.close()
    
    print("\n" + "="*70)
    print("💡 NEXT STEPS FOR DL1 DATASET")
    print("="*70)
    print("""
1. Extract actual election data from "Download 1" references
   - These point to DL1 Data Sheet or similar
   - Could be separate Google Sheets or CSV files
   
2. Create dl1.election_results table with validated data:
   - state, year, race, office, candidate_name, votes, percentage
   - verified_by, verified_date, confidence_score
   - source_sheet, original_source_url
   
3. Link to workflow.contests via contest_id foreign key
   
4. Version control the DL1 dataset in fixtures/dl1/
   - JSON snapshots of ground truth data
   - Migration manifest
   - QA approval metadata
   
5. Build comparison system (DL1 vs DL2) for accuracy tracking
    """)
    print("="*70 + "\n")


def export_dl1_manifest():
    """
    Export manifest of DL1-ready records to JSON.
    """
    print("\n📄 Exporting DL1 Manifest...")
    
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Get PROD Loaded records
    cur.execute("""
        SELECT 
            id, year, race, state, county,
            priority, status,
            work_in_progress_dl1, work_in_progress_dl2,
            dl1_complete, dl2_complete,
            qc_1_form, qc_2_form,
            download_1, download_2, source_link,
            pre_qc_results, google_sheets_row_number
        FROM workflow.contests 
        WHERE status = 'PROD Loaded'
        ORDER BY year DESC, state, race
    """)
    
    columns = [desc[0] for desc in cur.description]
    records = [dict(zip(columns, row)) for row in cur.fetchall()]
    
    manifest = {
        "dataset_name": "DL1 - Ground Truth Election Results",
        "description": "Production-loaded, fully-validated election data approved for full pseudo database",
        "total_records": len(records),
        "status": "PROD Loaded only",
        "created_at": json.dumps(None, default=str),  # Will be filled by script
        "records": records
    }
    
    # Save manifest
    manifest_path = OUTPUT_DIR / "dl1_manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"✅ Manifest exported to: {manifest_path}")
    print(f"   Records: {len(records)}")
    
    # Create summary
    print("\n📊 DL1 Dataset Summary by Year:")
    by_year = {}
    for record in records:
        year = record['year']
        if year not in by_year:
            by_year[year] = 0
        by_year[year] += 1
    
    for year in sorted(by_year.keys(), reverse=True):
        print(f"   {year}: {by_year[year]} records")
    
    conn.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build DL1 Ground Truth Dataset")
    parser.add_argument('--analyze', action='store_true', help='Analyze DL1 readiness')
    parser.add_argument('--export-manifest', action='store_true', help='Export DL1 manifest to JSON')
    args = parser.parse_args()
    
    if args.analyze or not args.export_manifest:
        analyze_dl1_readiness()
    
    if args.export_manifest:
        export_dl1_manifest()


if __name__ == "__main__":
    main()
