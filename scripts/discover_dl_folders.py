"""
Discover DL1 and DL2 Google Drive Folders

Purpose: Explore the DL1 and DL2 Google Drive folders to understand file structure.
         DL1 = Manually verified, standardized election data
         DL2 = Parser-extracted election data for comparison

Folders:
    DL1: https://drive.google.com/drive/u/4/folders/1ZwsL_Ui2qFyV-EJ1OZ_9lyhMeX8d4v9N
    DL2: https://drive.google.com/drive/u/4/folders/1wQcC_UEIFQrIYyRhyfgY2rr5RkiCBQ7V

Usage:
    python scripts/discover_dl_folders.py [--dl1] [--dl2] [--details] [--limit N]

Options:
    --dl1       : Show DL1 folder contents
    --dl2       : Show DL2 folder contents
    --details   : Show detailed file metadata
    --limit N   : Limit results to first N files
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Google Drive configuration
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
CREDS_PATH = os.getenv("GOOGLE_SERVICE_ACCOUNT_PATH")

# Folder IDs extracted from URLs
DL1_FOLDER_ID = "1ZwsL_Ui2qFyV-EJ1OZ_9lyhMeX8d4v9N"
DL2_FOLDER_ID = "1wQcC_UEIFQrIYyRhyfgY2rr5RkiCBQ7V"

OUTPUT_DIR = Path("scripts")


def list_folder_contents(service, folder_id: str, folder_name: str, details: bool = False, limit: int = None) -> List[Dict[str, Any]]:
    """
    List all files in a Google Drive folder.
    
    Args:
        service: Google Drive API service instance
        folder_id: Google Drive folder ID
        folder_name: Human-readable name for display
        details: Include detailed metadata
        limit: Maximum number of files to return
    
    Returns:
        List of file metadata dictionaries
    """
    print(f"\n{'='*70}")
    print(f"📁 {folder_name} FOLDER CONTENTS")
    print(f"{'='*70}")
    
    try:
        # Query files in folder
        query = f"'{folder_id}' in parents and trashed=false"
        fields = "files(id, name, mimeType, size, modifiedTime, createdTime, webViewLink)"
        
        results = service.files().list(
            q=query,
            fields=fields,
            pageSize=1000,  # Max per page
            orderBy='name'
        ).execute()
        
        files = results.get('files', [])
        
        if limit:
            files = files[:limit]
        
        print(f"\n✅ Found {len(files)} files")
        
        # Categorize by type
        file_types = {}
        for file in files:
            mime = file.get('mimeType', 'unknown')
            if mime not in file_types:
                file_types[mime] = []
            file_types[mime].append(file)
        
        print(f"\n📊 File Types:")
        for mime, type_files in sorted(file_types.items()):
            count = len(type_files)
            mime_display = mime.split('.')[-1] if '.' in mime else mime
            print(f"   {mime_display:40} | {count:4} files")
        
        # Sample files
        print(f"\n📋 Sample Files (first 10):")
        for i, file in enumerate(files[:10], 1):
            name = file['name']
            size = int(file.get('size', 0)) if file.get('size') else 0
            size_kb = size / 1024 if size > 0 else 0
            mime = file['mimeType'].split('.')[-1] if '.' in file['mimeType'] else file['mimeType']
            
            print(f"\n   {i}. {name}")
            print(f"      Type: {mime}")
            if size_kb > 0:
                print(f"      Size: {size_kb:.1f} KB")
            print(f"      ID: {file['id']}")
            
            if details:
                print(f"      Modified: {file.get('modifiedTime')}")
                print(f"      Created: {file.get('createdTime')}")
                print(f"      Link: {file.get('webViewLink')}")
        
        # Look for patterns in filenames
        print(f"\n🔍 Filename Patterns:")
        
        # Extract years from filenames
        years = set()
        states = set()
        for file in files:
            name = file['name']
            # Look for 4-digit years
            import re
            year_matches = re.findall(r'\b(20\d{2}|19\d{2})\b', name)
            years.update(year_matches)
            
            # Look for state names (common patterns)
            state_patterns = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 
                            'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia',
                            'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas',
                            'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts',
                            'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana',
                            'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico',
                            'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma',
                            'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina',
                            'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont',
                            'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']
            
            for state in state_patterns:
                if state in name:
                    states.add(state)
        
        if years:
            print(f"   Years found in filenames: {sorted(years)}")
        if states:
            print(f"   States found in filenames ({len(states)}): {', '.join(sorted(list(states)[:10]))}")
            if len(states) > 10:
                print(f"      ... and {len(states) - 10} more")
        
        return files
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return []


def analyze_file_structure(files: List[Dict[str, Any]], folder_name: str):
    """
    Analyze common file naming patterns and structure.
    """
    print(f"\n{'='*70}")
    print(f"🔬 {folder_name} FILE STRUCTURE ANALYSIS")
    print(f"{'='*70}")
    
    # Common naming patterns
    patterns = {
        'has_year': 0,
        'has_state': 0,
        'has_race': 0,
        'is_csv': 0,
        'is_sheet': 0,
        'is_xlsx': 0,
    }
    
    state_keywords = ['Alabama', 'Alaska', 'Arizona', 'California', 'Florida', 'Texas']  # Sample
    race_keywords = ['President', 'Senate', 'House', 'Governor', 'Representative']
    
    for file in files:
        name = file['name'].lower()
        mime = file['mimeType']
        
        import re
        if re.search(r'\b(20\d{2}|19\d{2})\b', name):
            patterns['has_year'] += 1
        
        if any(state.lower() in name for state in state_keywords):
            patterns['has_state'] += 1
        
        if any(race.lower() in name for race in race_keywords):
            patterns['has_race'] += 1
        
        if 'csv' in mime or name.endswith('.csv'):
            patterns['is_csv'] += 1
        
        if 'spreadsheet' in mime or 'sheet' in mime:
            patterns['is_sheet'] += 1
        
        if 'excel' in mime or name.endswith('.xlsx'):
            patterns['is_xlsx'] += 1
    
    total = len(files)
    print(f"\nNaming Patterns ({total} total files):")
    for pattern, count in sorted(patterns.items()):
        pct = (count * 100 // total) if total > 0 else 0
        print(f"   {pattern:20} | {count:4} files ({pct}%)")
    
    print(f"\n💡 Recommendations:")
    if patterns['is_csv'] > total * 0.5:
        print("   • Primary format: CSV files")
        print("   • Use pandas.read_csv() for import")
    
    if patterns['is_sheet'] > total * 0.5:
        print("   • Primary format: Google Sheets")
        print("   • Use gspread to read data")
    
    if patterns['has_year'] > total * 0.7:
        print("   • Filenames include year - can extract from filename")
    
    if patterns['has_state'] > total * 0.7:
        print("   • Filenames include state - can extract from filename")


def export_folder_index(files: List[Dict[str, Any]], folder_name: str):
    """
    Export folder contents to JSON for reference.
    """
    output_path = OUTPUT_DIR / f"{folder_name.lower().replace(' ', '_')}_index.json"
    
    index = {
        "folder_name": folder_name,
        "total_files": len(files),
        "files": files
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n📄 Index exported to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Discover DL1 and DL2 Google Drive folders")
    parser.add_argument('--dl1', action='store_true', help='Show DL1 folder contents')
    parser.add_argument('--dl2', action='store_true', help='Show DL2 folder contents')
    parser.add_argument('--details', action='store_true', help='Show detailed file metadata')
    parser.add_argument('--limit', type=int, help='Limit results to first N files')
    args = parser.parse_args()
    
    # Default to both if none specified
    if not args.dl1 and not args.dl2:
        args.dl1 = True
        args.dl2 = True
    
    # Validate configuration
    if not CREDS_PATH or not Path(CREDS_PATH).exists():
        print(f"❌ ERROR: Google Service Account credentials not found: {CREDS_PATH}")
        return
    
    # Connect to Google Drive API
    print("🔐 Authenticating with Google Drive API...")
    credentials = Credentials.from_service_account_file(CREDS_PATH, scopes=SCOPES)
    service = build('drive', 'v3', credentials=credentials)
    print("✅ Authentication successful")
    
    # Explore DL1 folder
    if args.dl1:
        dl1_files = list_folder_contents(service, DL1_FOLDER_ID, "DL1 (Ground Truth)", 
                                         details=args.details, limit=args.limit)
        if dl1_files:
            analyze_file_structure(dl1_files, "DL1")
            export_folder_index(dl1_files, "DL1")
    
    # Explore DL2 folder
    if args.dl2:
        dl2_files = list_folder_contents(service, DL2_FOLDER_ID, "DL2 (Parser-Extracted)", 
                                         details=args.details, limit=args.limit)
        if dl2_files:
            analyze_file_structure(dl2_files, "DL2")
            export_folder_index(dl2_files, "DL2")
    
    print("\n" + "="*70)
    print("✅ Discovery complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
