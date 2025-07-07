import subprocess
import os
from .utils.shared_logger import log_info, log_error
POSTGRES_SERVICE_NAME = os.getenv("POSTGRES_SERVICE_NAME", "postgresql-x64-17")

def start_postgres_service(service_name=None):
    if service_name is None:
        service_name = POSTGRES_SERVICE_NAME
    try:
        result = subprocess.run(
            ["net", "start", service_name],
            check=True,
            capture_output=True,
            text=True
        )
        log_info(f"[INFO] PostgreSQL service '{service_name}' started.")
        log_info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        if e.stderr and "already been started" in e.stderr:
            log_info(f"[INFO] PostgreSQL service '{service_name}' is already running.")
            return True
        log_error(f"[ERROR] Could not start PostgreSQL service: {e}")
        log_info("STDOUT:", e.stdout)
        log_info("STDERR:", e.stderr)
        return False

def stop_postgres_service(service_name=None):
    if service_name is None:
        service_name = POSTGRES_SERVICE_NAME
    try:
        subprocess.run(["net", "stop", service_name], check=True, capture_output=True)
        log_info(f"[INFO] PostgreSQL service '{service_name}' stopped.")
        return True
    except subprocess.CalledProcessError as e:
        log_error(f"[ERROR] Could not stop PostgreSQL service: {e}")
        return False