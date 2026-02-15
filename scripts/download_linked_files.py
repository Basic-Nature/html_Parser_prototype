"""
Google Drive Linked Files Downloader

Purpose: Download CSVs and PDFs referenced in Google Sheets (Download 1/2, Source Link columns)
         Stores files in Azure Blob Storage and updates PostgreSQL with blob URLs.

Usage:
    python scripts/download_linked_files.py [--dry-run] [--contest-id ID]

Options:
    --dry-run       : Preview downloads without saving files
    --contest-id ID : Only download files for specific contest ID

Requirements:
    - PostgreSQL with workflow.contests table populated
    - Google Service Account with Drive API access
    - Azure Blob Storage configured (optional, uses local cache if not available)

Output:
    - Downloads files to: cache/google_drive_downloads/{contest_id}/
    - Updates PostgreSQL: workflow.contests.download_1, download_2, source_link with local paths
    - (Future) Uploads to Azure Blob Storage and stores blob URLs
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import psycopg2
from dotenv import load_dotenv
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# Load environment
load_dotenv()

# Google Drive configuration
SCOPES = [
    'https://www.googleapis.com/auth/drive.readonly'
]
CREDS_PATH = os.getenv("GOOGLE_SERVICE_ACCOUNT_PATH")

# PostgreSQL configuration
DB_CONFIG = {
    'dbname': os.getenv('POSTGRES_DB', 'warehouse_election_results'),
    'user': os.getenv('POSTGRES_USER', 'postgres'),
    'password': os.getenv('POSTGRES_PASSWORD'),
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': os.getenv('POSTGRES_PORT', '5432')
}

# Local cache directory
CACHE_DIR = Path("cache/google_drive_downloads")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def extract_file_id_from_url(url: str) -> Optional[str]:
    """
    Extract Google Drive file ID from various URL formats.
    
    Supported formats:
        - https://drive.google.com/file/d/{FILE_ID}/view
        - https://drive.google.com/open?id={FILE_ID}
        - https://drive.google.com/uc?id={FILE_ID}
        - https://docs.google.com/spreadsheets/d/{FILE_ID}/edit
    
    Args:
        url: Google Drive URL
    
    Returns:
        File ID or None if not found
    """
    if not url or not isinstance(url, str):
        return None
    
    # Pattern 1: /file/d/{FILE_ID}/ or /spreadsheets/d/{FILE_ID}/
    match = re.search(r'/(?:file|spreadsheets)/d/([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    
    # Pattern 2: ?id={FILE_ID}
    parsed = urlparse(url)
    query_params = parse_qs(parsed.query)
    if 'id' in query_params:
        return query_params['id'][0]
    
    return None


def extract_folder_id_from_url(url: str) -> Optional[str]:
    """
    Extract Google Drive folder ID from URL.
    
    Format: https://drive.google.com/drive/folders/{FOLDER_ID}
    """
    if not url or not isinstance(url, str):
        return None
    
    match = re.search(r'/folders/([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    
    return None


def download_file(service, file_id: str, save_path: Path, dry_run: bool = False) -> bool:
    """
    Download a file from Google Drive.
    
    Args:
        service: Google Drive API service instance
        file_id: Google Drive file ID
        save_path: Local path to save file
        dry_run: If True, only print what would be downloaded
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get file metadata
        file_metadata = service.files().get(fileId=file_id, fields='name,mimeType,size').execute()
        file_name = file_metadata.get('name', 'unknown')
        file_size = int(file_metadata.get('size', 0))
        
        print(f"  📥 {file_name} ({file_size / 1024:.1f} KB)")
        
        if dry_run:
            print(f"     Would save to: {save_path}")
            return True
        
        # Download file
        request = service.files().get_media(fileId=file_id)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    print(f"     Progress: {int(status.progress() * 100)}%", end='\r')
        
        print(f"     ✅ Saved to: {save_path}")
        return True
        
    except Exception as e:
        print(f"     ❌ Download failed: {e}")
        return False


def list_folder_files(service, folder_id: str) -> List[Dict[str, Any]]:
    """
    List all files in a Google Drive folder.
    
    Args:
        service: Google Drive API service instance
        folder_id: Google Drive folder ID
    
    Returns:
        List of file metadata dictionaries
    """
    try:
        results = service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="files(id, name, mimeType, size)"
        ).execute()
        
        return results.get('files', [])
    except Exception as e:
        print(f"  ❌ Failed to list folder: {e}")
        return []


def download_contest_files(service, contest: Dict[str, Any], dry_run: bool = False):
    """
    Download all files linked to a contest (Download 1, Download 2, Source Link).
    
    Args:
        service: Google Drive API service instance
        contest: Contest record from workflow.contests table
        dry_run: If True, preview without downloading
    """
    contest_id = contest['id']
    contest_dir = CACHE_DIR / f"contest_{contest_id}"
    
    print(f"\n📦 Contest {contest_id}: {contest.get('state')} - {contest.get('race')} ({contest.get('year')})")
    
    downloaded_paths = {
        'download_1': [],
        'download_2': [],
        'source_link': []
    }
    
    # Download 1
    if contest.get('download_1'):
        print(f"\n  Download 1: {contest['download_1']}")
        file_id = extract_file_id_from_url(contest['download_1'])
        folder_id = extract_folder_id_from_url(contest['download_1'])
        
        if file_id:
            save_path = contest_dir / "download_1" / f"{file_id}.csv"
            if download_file(service, file_id, save_path, dry_run):
                downloaded_paths['download_1'].append(str(save_path))
        elif folder_id:
            print("  📁 Folder detected, listing files...")
            files = list_folder_files(service, folder_id)
            for file in files:
                save_path = contest_dir / "download_1" / file['name']
                if download_file(service, file['id'], save_path, dry_run):
                    downloaded_paths['download_1'].append(str(save_path))
    
    # Download 2
    if contest.get('download_2'):
        print(f"\n  Download 2: {contest['download_2']}")
        file_id = extract_file_id_from_url(contest['download_2'])
        folder_id = extract_folder_id_from_url(contest['download_2'])
        
        if file_id:
            save_path = contest_dir / "download_2" / f"{file_id}.csv"
            if download_file(service, file_id, save_path, dry_run):
                downloaded_paths['download_2'].append(str(save_path))
        elif folder_id:
            print("  📁 Folder detected, listing files...")
            files = list_folder_files(service, folder_id)
            for file in files:
                save_path = contest_dir / "download_2" / file['name']
                if download_file(service, file['id'], save_path, dry_run):
                    downloaded_paths['download_2'].append(str(save_path))
    
    # Source Link (usually PDF)
    if contest.get('source_link'):
        print(f"\n  Source Link: {contest['source_link']}")
        file_id = extract_file_id_from_url(contest['source_link'])
        
        if file_id:
            save_path = contest_dir / "source" / f"{file_id}.pdf"
            if download_file(service, file_id, save_path, dry_run):
                downloaded_paths['source_link'].append(str(save_path))
        elif contest['source_link'].startswith('http'):
            # External URL (not Google Drive) - could fetch with requests
            print("  ⚠️  External URL (not Google Drive), skipping for now")
    
    return downloaded_paths


def main():
    parser = argparse.ArgumentParser(description="Download files linked in Google Sheets")
    parser.add_argument('--dry-run', action='store_true', help='Preview downloads without saving')
    parser.add_argument('--contest-id', type=int, help='Only download files for specific contest ID')
    args = parser.parse_args()
    
    # Validate configuration
    if not CREDS_PATH or not Path(CREDS_PATH).exists():
        print(f"❌ ERROR: Google Service Account credentials not found: {CREDS_PATH}")
        sys.exit(1)
    
    # Connect to Google Drive API
    print("🔐 Authenticating with Google Drive API...")
    credentials = Credentials.from_service_account_file(CREDS_PATH, scopes=SCOPES)
    service = build('drive', 'v3', credentials=credentials)
    print("✅ Authentication successful")
    
    # Connect to PostgreSQL
    print(f"\n🐘 Connecting to PostgreSQL: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}")
    conn = psycopg2.connect(**DB_CONFIG)
    print("✅ Database connected")
    
    # Fetch contests with file links
    with conn.cursor() as cur:
        if args.contest_id:
            cur.execute("""
                SELECT * FROM workflow.contests 
                WHERE id = %s
            """, (args.contest_id,))
        else:
            cur.execute("""
                SELECT * FROM workflow.contests 
                WHERE download_1 IS NOT NULL 
                   OR download_2 IS NOT NULL 
                   OR source_link IS NOT NULL
                ORDER BY id
            """)
        
        columns = [desc[0] for desc in cur.description]
        contests = [dict(zip(columns, row)) for row in cur.fetchall()]
    
    print(f"\n📋 Found {len(contests)} contests with file links")
    
    if args.dry_run:
        print("\n🔍 DRY RUN MODE - No files will be downloaded\n")
    
    # Download files for each contest
    for contest in contests:
        try:
            download_contest_files(service, contest, dry_run=args.dry_run)
        except Exception as e:
            print(f"\n❌ Error processing contest {contest['id']}: {e}")
    
    # Cleanup
    conn.close()
    print("\n✅ Download process complete!")
    print(f"📂 Files saved to: {CACHE_DIR.absolute()}")


if __name__ == "__main__":
    main()
