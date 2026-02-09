"""Quick verification script for migration"""
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    dbname=os.getenv('POSTGRES_DB', 'warehouse_election_results'),
    user=os.getenv('POSTGRES_USER', 'postgres'),
    password=os.getenv('POSTGRES_PASSWORD'),
    host=os.getenv('POSTGRES_HOST', 'localhost'),
    port=os.getenv('POSTGRES_PORT', '5432')
)

cur = conn.cursor()

# Overall stats
cur.execute("""
    SELECT 
        COUNT(*) as total,
        COUNT(DISTINCT state) as states,
        COUNT(DISTINCT year) as years,
        COUNT(DISTINCT status) as statuses,
        COUNT(DISTINCT priority) as priorities
    FROM workflow.contests
""")
result = cur.fetchone()

print("\n" + "="*70)
print("📊 MIGRATION VERIFICATION")
print("="*70)
print(f"   Total contests:      {result[0]}")
print(f"   Unique states:       {result[1]}")
print(f"   Unique years:        {result[2]}")
print(f"   Unique statuses:     {result[3]}")
print(f"   Unique priorities:   {result[4]}")

# Priority breakdown
print("\n📋 Breakdown by Priority:")
cur.execute("""
    SELECT priority, COUNT(*) as count 
    FROM workflow.contests 
    WHERE priority IS NOT NULL
    GROUP BY priority 
    ORDER BY priority
""")
for row in cur.fetchall():
    print(f"   {row[0]}: {row[1]} contests")

# Status breakdown
print("\n📋 Breakdown by Status:")
cur.execute("""
    SELECT status, COUNT(*) as count 
    FROM workflow.contests 
    WHERE status IS NOT NULL
    GROUP BY status 
    ORDER BY count DESC
    LIMIT 10
""")
for row in cur.fetchall():
    print(f"   {row[0]}: {row[1]} contests")

# Year breakdown
print("\n📋 Breakdown by Year:")
cur.execute("""
    SELECT year, COUNT(*) as count 
    FROM workflow.contests 
    WHERE year IS NOT NULL
    GROUP BY year 
    ORDER BY year DESC
""")
for row in cur.fetchall():
    print(f"   {row[0]}: {row[1]} contests")

# Sample records
print("\n📋 Sample Records (first 5):")
cur.execute("""
    SELECT priority, status, year, race, state, work_in_progress_dl1, work_in_progress_dl2
    FROM workflow.contests 
    ORDER BY id
    LIMIT 5
""")
for row in cur.fetchall():
    print(f"\n   Priority: {row[0]} | Status: {row[1]}")
    print(f"   {row[2]} | {row[3]} | {row[4]}")
    print(f"   DL1: {row[5]} | DL2: {row[6]}")

# QC completion stats
print("\n📋 QC Completion Status:")
cur.execute("""
    SELECT 
        COUNT(CASE WHEN dl1_complete = 'TRUE' THEN 1 END) as dl1_complete,
        COUNT(CASE WHEN dl2_complete = 'TRUE' THEN 1 END) as dl2_complete,
        COUNT(CASE WHEN qc_1_form IS NOT NULL AND qc_1_form != '' THEN 1 END) as qc1_done,
        COUNT(CASE WHEN qc_2_form IS NOT NULL AND qc_2_form != '' THEN 1 END) as qc2_done
    FROM workflow.contests
""")
result = cur.fetchone()
print(f"   DL1 Complete:  {result[0]}/{408} ({result[0]*100//408}%)")
print(f"   DL2 Complete:  {result[1]}/{408} ({result[1]*100//408}%)")
print(f"   QC 1 Done:     {result[2]}/{408} ({result[2]*100//408}%)")
print(f"   QC 2 Done:     {result[3]}/{408} ({result[3]*100//408}%)")

print("\n" + "="*70)
print("✅ Migration verification complete!")
print("="*70 + "\n")

conn.close()
