#!/usr/bin/env python3
"""
Quick verification of DL1 import data.
"""

import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()

# PostgreSQL configuration (same as import_dl_data.py)
DB_CONFIG = {
    'dbname': os.getenv('POSTGRES_DB', 'warehouse_election_results'),
    'user': os.getenv('POSTGRES_USER', 'postgres'),
    'password': os.getenv('POSTGRES_PASSWORD'),
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': os.getenv('POSTGRES_PORT', '5432')
}

def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Check total rows
    cur.execute("SELECT COUNT(*) FROM dl1.election_results")
    total = cur.fetchone()[0]
    print(f"\n📊 Total DL1 rows: {total}")
    
    # Sample records
    cur.execute("""
        SELECT state, county, year, office, election_date, 
               candidate_name, candidate_party, votes_total, 
               verified_by 
        FROM dl1.election_results 
        ORDER BY county 
        LIMIT 10
    """)
    
    print("\n📋 Sample records:")
    print(f"{'State':<12} {'County':<20} {'Year':<8} {'Office':<25} {'Election':<12} {'Candidate':<30} {'Party':<15} {'Votes':>10} {'Verified':<10}")
    print("=" * 180)
    
    for row in cur.fetchall():
        print(f"{row[0]:<12} {row[1]:<20} {row[2]:<8} {row[3]:<25} {str(row[4]):<12} {row[5]:<30} {str(row[6] or ''):<15} {row[7]:>10,} {str(row[8] or ''):<10}")
    
    # Check by contest
    cur.execute("""
        SELECT c.year, c.state, c.race, COUNT(*) as rows
        FROM dl1.election_results dl1
        JOIN workflow.contests c ON dl1.contest_id = c.id
        GROUP BY c.year, c.state, c.race
        ORDER BY c.year, c.state, c.race
    """)
    
    print("\n📊 Imported contests:")
    for row in cur.fetchall():
        print(f"  {row[0]} {row[1]:20} {row[2]:30} → {row[3]} rows")
    
    conn.close()

if __name__ == "__main__":
    main()
