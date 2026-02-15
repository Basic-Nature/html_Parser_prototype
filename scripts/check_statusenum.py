import os

import psycopg2
from dotenv import load_dotenv

env_path = os.path.join(os.getcwd(), ".env")
load_dotenv(env_path)

conn = psycopg2.connect(
    dbname=os.getenv("POSTGRES_DB", "warehouse_election_results"),
    user=os.getenv("POSTGRES_USER", "postgres"),
    password=os.getenv("POSTGRES_PASSWORD"),
    host=os.getenv("POSTGRES_HOST", "localhost"),
    port=os.getenv("POSTGRES_PORT", "5432"),
)
with conn, conn.cursor() as cur:
    cur.execute(
        "SELECT enumlabel FROM pg_enum JOIN pg_type ON pg_enum.enumtypid = pg_type.oid "
        "WHERE pg_type.typname = 'statusenum' ORDER BY enumsortorder;"
    )
    print("statusenum values:", [row[0] for row in cur.fetchall()])
conn.close()
