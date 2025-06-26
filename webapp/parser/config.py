import os
from dotenv import load_dotenv

load_dotenv()
import psycopg2
from psycopg2 import sql, OperationalError

# PROJECT_ROOT is the parent directory of 'webapp', i.e., the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
# BASE_DIR points to .../webapp
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTEXT_DB_PATH = os.path.join(BASE_DIR, "parser", "Context_Integration", "Context_Library", "context_elections.db")
CONTEXT_LIBRARY_PATH = os.path.join(
    BASE_DIR, "parser", "Context_Integration", "Context_Library", "context_library.json"
)
CONTEXT_LIBRARY_DIR = os.path.dirname(CONTEXT_LIBRARY_PATH)

# Ensure log and cache directories exist inside Context_Library
LOG_DIR = os.path.join(CONTEXT_LIBRARY_DIR, "log")
CACHE_DIR = os.path.join(CONTEXT_LIBRARY_DIR, "cache")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
MODEL_DIR = os.path.dirname(BASE_DIR)
# Usage: for subprocesses, set cwd=PROJECT_ROOT and ensure PROJECT_ROOT is in PYTHONPATH

# Build the PostgreSQL URL from .env variables
POSTGRES_USER = os.getenv("POSTGRES_USER", "")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
POSTGRES_DB = os.getenv("POSTGRES_DB", "")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "")

POSTGRES_URL = os.getenv(
    "POSTGRES_URL",
    f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

def ensure_postgres_db():
    """Ensure the target database exists, create if not."""
    if not all([POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB, POSTGRES_HOST, POSTGRES_PORT]):
        raise RuntimeError("PostgreSQL credentials are not fully set in the environment variables.")

    # Try connecting to the target DB
    try:
        conn = psycopg2.connect(
            dbname=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
        )
        conn.close()
        return  # DB exists and is accessible
    except OperationalError as e:
        if "does not exist" not in str(e):
            raise RuntimeError(f"Could not connect to PostgreSQL: {e}")

    # If we get here, DB does not exist: connect to 'postgres' and create it
    try:
        conn = psycopg2.connect(
            dbname="postgres",
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
        )
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(POSTGRES_DB)))
        conn.close()
        print(f"[INFO] Database '{POSTGRES_DB}' created.")
    except Exception as e:
        raise RuntimeError(f"Failed to create database '{POSTGRES_DB}': {e}")

# Ensure DB exists before anything else uses POSTGRES_URL
ensure_postgres_db()

if __name__ == "__main__":
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("BASE_DIR:", BASE_DIR)
    print("POSTGRES_URL:", POSTGRES_URL)