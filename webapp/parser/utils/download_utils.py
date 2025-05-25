import os
import requests
from urllib.parse import urljoin

def ensure_input_directory():
    """Ensure the 'input' directory exists."""
    os.makedirs("input", exist_ok=True)

def ensure_output_directory():
    """Ensure the 'output' directory exists."""
    os.makedirs("output", exist_ok=True)

def download_file(page_url, href):
    """
    Download the linked file and save it into the input directory.
    Returns the full path of the saved file, or None on failure.
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

def download_multiple_files(page_url, href_list, confirmed: bool = True):
    """
    Download multiple files (given as a list of hrefs) to the input directory.
    Returns a list of file paths for successfully downloaded files.
    """
    if not confirmed or not href_list:
        print("[INFO] Multiple download skipped by user or empty list.")
        return []
    ensure_input_directory()
    downloaded_files = []
    for href in href_list:
        file_path = download_file(page_url, href)
        if file_path:
            downloaded_files.append(file_path)
    return downloaded_files

def download_confirmed_file(file_url: str, page_url: str, confirmed: bool = True):
    """
    Download the file if confirmed by the user.
    If not confirmed, return None so the pipeline can skip to HTML handler.
    """
    if not confirmed:
        print("[INFO] Download skipped by user.")
        return None
    return download_file(page_url, file_url)