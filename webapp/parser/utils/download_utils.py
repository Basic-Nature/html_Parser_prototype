import os
import requests
import hashlib
import json
from urllib.parse import urljoin
from datetime import datetime
from ..utils.logger_instance import logger
from ..Context_Integration.context_organizer import append_to_context_library

DOWNLOAD_MANIFEST = "input/.download_manifest.jsonl"

def ensure_input_directory():
    """Ensure the 'input' directory exists."""
    os.makedirs("input", exist_ok=True)

def ensure_output_directory():
    """Ensure the 'output' directory exists."""
    os.makedirs("output", exist_ok=True)

def file_hash(filepath, algo="sha256", blocksize=65536):
    """Compute the hash of a file for deduplication/integrity."""
    h = hashlib.new(algo)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(blocksize), b""):
            h.update(chunk)
    return h.hexdigest()

def load_download_manifest():
    """Load the download manifest as a dict: url or filename -> metadata."""
    if not os.path.exists(DOWNLOAD_MANIFEST):
        return {}
    manifest = {}
    with open(DOWNLOAD_MANIFEST, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                manifest[entry.get("url") or entry.get("filename")] = entry
            except Exception:
                continue
    return manifest

def update_download_manifest(entry):
    """Append a new entry to the download manifest."""
    with open(DOWNLOAD_MANIFEST, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

def is_already_downloaded(url, filename=None, check_hash=False):
    """Check if a file has already been downloaded (by URL or filename, optionally by hash)."""
    manifest = load_download_manifest()
    if url in manifest:
        entry = manifest[url]
        if filename and os.path.exists(filename):
            if not check_hash or entry.get("hash") == file_hash(filename):
                return True
    if filename and os.path.exists(filename):
        # Check by filename only
        for entry in manifest.values():
            if entry.get("filename") == filename:
                if not check_hash or entry.get("hash") == file_hash(filename):
                    return True
    return False

def download_file(page_url, href, context_info=None, check_hash=False):
    """
    Download the linked file and save it into the input directory.
    Returns the full path of the saved file, or None on failure.
    Prevents re-downloading if already present (by URL or filename/hash).
    Optionally updates the context library with download info.
    """
    ensure_input_directory()
    filename = os.path.basename(href)
    save_path = os.path.join("input", filename)
    file_url = urljoin(page_url, href)

    # Prevent re-download if already present
    if is_already_downloaded(file_url, save_path, check_hash=check_hash):
        logger.info(f"[DOWNLOAD] Skipping already downloaded file: {filename}")
        return save_path

    try:
        response = requests.get(file_url)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(response.content)
        filehash = file_hash(save_path)
        logger.info(f"[DOWNLOAD] Downloaded: {filename} -> input/")
        # Update manifest
        entry = {
            "url": file_url,
            "filename": save_path,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "hash": filehash,
            "status": "success"
        }
        update_download_manifest(entry)
        # Optionally update context library
        if context_info:
            append_to_context_library({"downloads": [entry]})
        return save_path
    except Exception as e:
        logger.error(f"[ERROR] Failed to download {file_url}: {e}")
        entry = {
            "url": file_url,
            "filename": save_path,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "fail",
            "error": str(e)
        }
        update_download_manifest(entry)
        return None

def download_multiple_files(page_url, href_list, confirmed: bool = True, context_info=None, check_hash=False):
    """
    Download multiple files (given as a list of hrefs) to the input directory.
    Returns a list of file paths for successfully downloaded files.
    """
    if not confirmed or not href_list:
        logger.info("[DOWNLOAD] Multiple download skipped by user or empty list.")
        return []
    ensure_input_directory()
    downloaded_files = []
    for href in href_list:
        file_path = download_file(page_url, href, context_info=context_info, check_hash=check_hash)
        if file_path:
            downloaded_files.append(file_path)
    return downloaded_files

def download_confirmed_file(file_url: str, page_url: str, confirmed: bool = True, context_info=None, check_hash=False):
    """
    Download the file if confirmed by the user.
    If not confirmed, return None so the pipeline can skip to HTML handler.
    """
    if not confirmed:
        logger.info("[DOWNLOAD] Download skipped by user.")
        return None
    return download_file(page_url, file_url, context_info=context_info, check_hash=check_hash)

def summarize_downloads():
    """Print a summary of all downloads from the manifest."""
    manifest = load_download_manifest()
    print("\n[DOWNLOAD SUMMARY]")
    for entry in manifest.values():
        print(f"  {entry.get('filename')} | {entry.get('url')} | {entry.get('status')} | {entry.get('timestamp')}")

def get_downloaded_files_by_status(status="success"):
    """Return a list of filenames for downloads with the given status."""
    manifest = load_download_manifest()
    return [entry["filename"] for entry in manifest.values() if entry.get("status") == status]