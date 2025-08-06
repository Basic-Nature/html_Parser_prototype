import subprocess
import os
from .utils.logger_singleton import logger, console
from .config import POSTGRES_SERVICE_NAME

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
        logger.info(f"[INFO] PostgreSQL service '{service_name}' started.")
        console.print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        if e.stderr and "already been started" in e.stderr:
            logger.info(f"[INFO] PostgreSQL service '{service_name}' is already running.")
            return True
        logger.error(f"[ERROR] Could not start PostgreSQL service: {e}")
        logger.info("STDOUT:", e.stdout)
        logger.info("STDERR:", e.stderr)
        return False

def stop_postgres_service(service_name=None):
    if service_name is None:
        service_name = POSTGRES_SERVICE_NAME
    try:
        subprocess.run(["net", "stop", service_name], check=True, capture_output=True)
        logger.info(f"[INFO] PostgreSQL service '{service_name}' stopped.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"[ERROR] Could not stop PostgreSQL service: {e}")
        return False