import subprocess
import os

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
        print(f"[INFO] PostgreSQL service '{service_name}' started.")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        if e.stderr and "already been started" in e.stderr:
            print(f"[INFO] PostgreSQL service '{service_name}' is already running.")
            return True
        print(f"[ERROR] Could not start PostgreSQL service: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

def stop_postgres_service(service_name=None):
    if service_name is None:
        service_name = POSTGRES_SERVICE_NAME
    try:
        subprocess.run(["net", "stop", service_name], check=True, capture_output=True)
        print(f"[INFO] PostgreSQL service '{service_name}' stopped.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Could not stop PostgreSQL service: {e}")
        return False