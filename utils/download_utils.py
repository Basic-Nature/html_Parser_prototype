# utils/download_utils.py
# ==============================================================
# Utilities for downloading file links (JSON, CSV, PDF) from election result pages.
# Ensures files are placed in the /input folder for processing.
# ==============================================================

import os
import requests
from urllib.parse import urljoin

# Ensure the /input directory exists for incoming file downloads
def ensure_input_directory():
    """Ensure the 'input' directory exists."""
    """Create the 'input' directory if it doesn't exist."""
    os.makedirs("input", exist_ok=True)
# # Ensure the /output directory exists for when format route is chosen
def ensure_output_directory():
    """Ensure the 'output' directory exists."""
    """Create the 'output' directory if it doesn't exist."""
    os.makedirs("output", exist_ok=True)


# Download a specific linked file and save to /input
# Used for direct href-based downloads after format detection
def download_file(page_url, href):
    """
    Download the linked file and save it into the input directory.
    Returns the full path of the saved file.
    """
    ensure_input_directory()
    filename = os.path.basename(href)
    save_path = os.path.join("input", filename)

    file_url = urljoin(page_url, href)
    try:
        response = requests.get(file_url)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"[INFO] Downloaded: {filename} -> input/")
        return save_path
    except Exception as e:
        print(f"[ERROR] Failed to download {file_url}: {e}")
        return None

def download_confirmed_file(file_url: str, page_url: str):
    """
    Given a confirmed file link (e.g., from detect_format_from_links),
    download it and save to /input folder. Returns full path.
    """
    return download_file(page_url, file_url)
