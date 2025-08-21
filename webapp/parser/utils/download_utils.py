from __future__ import annotations
# webapp/parser/utils/download_utils.py
# ---------------------------------------------------------------
# Download utility functions for Smart Elections Parser Webapp
# ---------------------------------------------------------------
import os
import requests
import orjson
from urllib.parse import urljoin
from datetime import datetime
from ..utils.logger_singleton import logger
from ..utils.shared_logic import safe_get
from ..Context_Integration.context_organizer import ContextOrganizer
from ..utils.misc_utils import file_hash
from ..config import INPUT_DIR, OUTPUT_DIR, DOWNLOAD_MANIFEST

def ensure_input_directory():
    """Ensure the 'input' directory exists."""
    os.makedirs(INPUT_DIR, exist_ok=True)

def ensure_output_directory():
    """Ensure the 'output' directory exists."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_download_manifest():
    """Load the download manifest as a dict: url or filename -> metadata."""
    if not os.path.exists(DOWNLOAD_MANIFEST):
        return {}
    manifest = {}
    with open(DOWNLOAD_MANIFEST, "rb") as f:
        for line in f:
            try:
                entry = orjson.loads(line)
                key = safe_get(entry, "url") or safe_get(entry, "filename")
                if key:
                    manifest[key] = entry
            except Exception:
                continue
    return manifest

def update_download_manifest(entry):
    """Append a new entry to the download manifest."""
    with open(DOWNLOAD_MANIFEST, "ab") as f:
        f.write(orjson.dumps(entry) + b"\n")

def is_already_downloaded(url, filename=None, check_hash=False):
    """Check if a file has already been downloaded (by URL or filename, optionally by hash)."""
    manifest = load_download_manifest()
    entry = safe_get(manifest, url)
    if entry and filename and os.path.exists(filename):
        entry_hash = safe_get(entry, "hash")
        file_hash_val = file_hash(filename)
        if not check_hash or (entry_hash and file_hash_val and entry_hash == file_hash_val):
            return True
    if filename and os.path.exists(filename):
        # Check by filename only
        for entry in manifest.values():
            entry_filename = safe_get(entry, "filename")
            entry_hash = safe_get(entry, "hash")
            file_hash_val = file_hash(filename)
            if entry_filename == filename:
                if not check_hash or (entry_hash and file_hash_val and entry_hash == file_hash_val):
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
    save_path = os.path.join(INPUT_DIR, filename)
    file_url = urljoin(page_url, href)
    logger.info(f"[DEBUG][download_file] page_url={page_url}, href={href}, file_url={file_url}, save_path={save_path}")
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
        logger.info(f"[DOWNLOAD] Downloaded: {filename} -> {INPUT_DIR}/")
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
            organizer = ContextOrganizer()
            organizer.append_to_context_library({"downloads": [entry]})
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
    logger.info("\n[DOWNLOAD SUMMARY]")
    for entry in manifest.values():
        filename = safe_get(entry, "filename")
        url = safe_get(entry, "url")
        status = safe_get(entry, "status")
        timestamp = safe_get(entry, "timestamp")
        logger.info(f"  {filename} | {url} | {status} | {timestamp}")

def get_downloaded_files_by_status(status="success"):
    """Return a list of filenames for downloads with the given status."""
    manifest = load_download_manifest()
    return [
        safe_get(entry, "filename")
        for entry in manifest.values()
        if safe_get(entry, "status") == status and safe_get(entry, "filename") is not None
    ]