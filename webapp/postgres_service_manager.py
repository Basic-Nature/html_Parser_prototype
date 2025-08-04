"""
postgres_service_manager.py
-----------------------------------------
Standalone manager for Postgres service.
- Start/stop service safely.
- Health checks.
- Designed for use by other scripts.
-----------------------------------------
"""

import subprocess
import atexit
import time
import psycopg2
from psycopg2 import sql, OperationalError
from webapp.parser.config import (
    POSTGRES_SERVICE_NAME, POSTGRES_USER, POSTGRES_PASSWORD, 
    POSTGRES_DB, POSTGRES_HOST, POSTGRES_PORT
)

def ensure_postgres_db():
    """Ensure the target database exists, create if not."""
    try:
        from webapp.parser.utils.logger_singleton import logger
    except ImportError:
        import logging
        logger = logging.getLogger("postgres_fallback")
    if not all([POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB, POSTGRES_HOST, POSTGRES_PORT]):
        raise RuntimeError("PostgreSQL credentials are not fully set in the environment variables.")
    try:
        conn = psycopg2.connect(
            dbname=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
        )
        conn.close()
        logger.info(f"[INFO] Database '{POSTGRES_DB}' exists and is accessible.")
        return
    except OperationalError as e:
        if "does not exist" not in str(e):
            raise RuntimeError(f"Could not connect to PostgreSQL: {e}")
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
        logger.info(f"[INFO] Database '{POSTGRES_DB}' created.")
    except Exception as e:
        raise RuntimeError(f"Failed to create database '{POSTGRES_DB}': {e}")

def start_postgres_service():
    try:
        result = subprocess.run(["sc", "start", POSTGRES_SERVICE_NAME], capture_output=True, text=True)
        if "START_PENDING" in result.stdout or "RUNNING" in result.stdout:
            print("[Postgres] Service start requested.")
            return True
        print("[Postgres] Service start failed:", result.stdout)
        return False
    except Exception as e:
        print("[Postgres] Exception during start:", e)
        return False

def stop_postgres_service():
    try:
        result = subprocess.run(["sc", "stop", POSTGRES_SERVICE_NAME], capture_output=True, text=True)
        if "STOP_PENDING" in result.stdout or "STOPPED" in result.stdout:
            print("[Postgres] Service stop requested.")
            return True
        print("[Postgres] Service stop failed:", result.stdout)
        return False
    except Exception as e:
        print("[Postgres] Exception during stop:", e)
        return False

def postgres_service_status():
    try:
        result = subprocess.run(["sc", "query", POSTGRES_SERVICE_NAME], capture_output=True, text=True)
        if "RUNNING" in result.stdout:
            return "running"
        elif "STOPPED" in result.stdout:
            return "stopped"
        else:
            return "unknown"
    except Exception as e:
        print("[Postgres] Exception during status check:", e)
        return "error"

def health_check():
    status = postgres_service_status()
    print(f"[Postgres] Service status: {status}")
    return status == "running"

def safe_exit():
    print("[Postgres] Safe exit: stopping service if running...")
    if postgres_service_status() == "running":
        stop_postgres_service()

atexit.register(safe_exit)

if __name__ == "__main__":
    print("[Postgres] Starting service...")
    started = start_postgres_service()
    time.sleep(2)
    health_check()
    ensure_postgres_db()
    print("[Postgres] Service is running. Press Enter to stop and exit, or type 'exit' then Enter.")
    try:
        while True:
            user_input = input("> ")
            if user_input.strip().lower() in ("exit", ""):
                break
            print("Type 'exit' or press Enter to stop the service and exit.")
    except KeyboardInterrupt:
        print("\n[Postgres] Keyboard interrupt received. Exiting...")
    print("[Postgres] Stopping service...")
    stop_postgres_service()
    health_check()