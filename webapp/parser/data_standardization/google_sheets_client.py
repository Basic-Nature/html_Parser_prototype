"""
Google Sheets API Client
Fetches election data from Smart Elections Database-Lite workbooks
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    from google.auth.exceptions import GoogleAuthError
    from google.oauth2 import service_account
    GOOGLE_SHEETS_AVAILABLE = True
except ImportError:
    GOOGLE_SHEETS_AVAILABLE = False

logger = logging.getLogger(__name__)


def _build_service_account_json_from_env() -> Optional[Dict[str, str]]:
    """
    Construct service account JSON from individual environment variables.
    
    Returns JSON dict if all required fields are present, None otherwise.
    This approach is preferred for Azure App Settings where multi-line JSON
    strings can be problematic.
    """
    required_fields = {
        'type': 'GOOGLE_SHEETS_SA_TYPE',
        'project_id': 'GOOGLE_SHEETS_SA_PROJECT_ID',
        'private_key_id': 'GOOGLE_SHEETS_SA_PRIVATE_KEY_ID',
        'private_key': 'GOOGLE_SHEETS_SA_PRIVATE_KEY',
        'client_email': 'GOOGLE_SHEETS_SA_CLIENT_EMAIL',
        'client_id': 'GOOGLE_SHEETS_SA_CLIENT_ID',
        'auth_uri': 'GOOGLE_SHEETS_SA_AUTH_URI',
        'token_uri': 'GOOGLE_SHEETS_SA_TOKEN_URI',
        'auth_provider_x509_cert_url': 'GOOGLE_SHEETS_SA_AUTH_PROVIDER_CERT_URL',
        'client_x509_cert_url': 'GOOGLE_SHEETS_SA_CLIENT_CERT_URL',
        'universe_domain': 'GOOGLE_SHEETS_SA_UNIVERSE_DOMAIN',
    }
    
    creds_dict = {}
    missing_fields = []
    
    for json_key, env_var in required_fields.items():
        value = os.getenv(env_var, '').strip()
        if not value:
            missing_fields.append(env_var)
        else:
            creds_dict[json_key] = value
    
    if missing_fields:
        logger.debug(f"Service account env vars missing: {', '.join(missing_fields)}")
        return None
    
    # Handle private_key newline restoration (Azure strips \n)
    if 'private_key' in creds_dict:
        # Replace literal \n with actual newlines
        creds_dict['private_key'] = creds_dict['private_key'].replace('\\n', '\n')
    
    return creds_dict


def _load_credentials_from_file(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Load credentials JSON from a file path.
    
    Returns parsed JSON dict if successful, None otherwise.
    """
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                creds = json.load(f)
                logger.info(f"✓ Loaded credentials from file: {file_path}")
                return creds
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.debug(f"Could not load credentials from {file_path}: {e}")
    
    return None


@dataclass
class SheetFetchResult:
    """Result of fetching a sheet"""
    success: bool
    sheet_name: str
    records: List[Dict[str, Any]]
    row_count: int
    fetch_time: datetime
    error: Optional[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class GoogleSheetsElectionClient:
    """
    Client for fetching election data from Smart Elections spreadsheets.
    """
    
    # Expected sheet names and their expected column counts
    EXPECTED_SHEETS = {
        'finalized data': 18,  # A-R
        'down-ballot calculations': 8,  # A-H
        'races': 5,  # A-E
        'mass update log': 4,  # A-D
        'mass updates': 7,  # A-G
        'chain of custody': 5,  # A-E
    }
    
    # Column mappings for primary sheets
    FINALIZED_DATA_COLUMNS = {
        'A': 'County/District',
        'B': 'Ballot Candidate Name',
        'C': 'Ballot Party',
        'D': 'Uncategorized Votes',
        'E': 'Early Votes',
        'F': 'Election Day Votes',
        'G': 'Mail in Votes',
        'H': 'Provisional Votes',
        'I': 'Is Write In',
        'J': 'Candidate',
        'K': 'Year',
        'L': 'Office',
        'M': 'State',
        'N': 'Party',
        'O': 'FEC ID',
        'P': 'Source Data URL',
        'Q': 'RACE ID',
        'R': 'Total Votes',
    }
    
    DOWN_BALLOT_COLUMNS = {
        'A': 'Year',
        'B': 'State',
        'C': 'County',
        'D': 'Party',
        'E': 'Presidential Votes',
        'F': 'Down-Ballot Votes',
        'G': 'Drop-off %',
        'H': 'Office',
    }
    
    def __init__(self, credentials_json: Optional[str] = None, spreadsheet_id: Optional[str] = None):
        """
        Initialize Google Sheets client.
        
        Credentials Loading Priority (first match wins):
            1. credentials_json parameter (file path or JSON string)
            2. Individual env vars (GOOGLE_SHEETS_SA_*) - Recommended for Azure
            3. GOOGLE_SHEETS_ELECTION_DB_LITE_CREDENTIALS env var (file/JSON string)
            4. GOOGLE_APPLICATION_CREDENTIALS env var (points to file)
            5. google_service_account.json in project root (local dev convenience)
        
        Spreadsheet ID Loading:
            1. spreadsheet_id parameter
            2. GOOGLE_SHEETS_DB_LITE_ID env var
        
        Args:
            credentials_json: Optional file path or JSON string for credentials
            spreadsheet_id: Optional override for spreadsheet ID
        """
        if not GOOGLE_SHEETS_AVAILABLE:
            raise ImportError("Google Sheets API client requires 'google-auth' and 'google-auth-oauthlib' packages")
        
        self.credentials = None
        self.service = None
        self.spreadsheet_id = spreadsheet_id or os.getenv('GOOGLE_SHEETS_DB_LITE_ID', '').strip()
        
        # Attempt to load credentials using priority chain
        creds_data = None
        credentials_source = None
        
        # Priority 1: credentials_json parameter (explicit parameter)
        if credentials_json:
            credentials_source = credentials_json
        
        # Priority 2: Individual env vars (Azure recommended)
        if not credentials_source and not creds_data:
            creds_data = _build_service_account_json_from_env()
            if creds_data:
                logger.info("✓ Using credentials from individual env vars (Azure recommended)")
        
        # Priority 3: GOOGLE_SHEETS_ELECTION_DB_LITE_CREDENTIALS env var (legacy)
        if not credentials_source and not creds_data:
            credentials_source = os.getenv('GOOGLE_SHEETS_ELECTION_DB_LITE_CREDENTIALS', '').strip()
            if credentials_source:
                logger.debug("Using credentials from GOOGLE_SHEETS_ELECTION_DB_LITE_CREDENTIALS")
        
        # Priority 4: GOOGLE_APPLICATION_CREDENTIALS env var (local/GCP standard)
        if not credentials_source and not creds_data:
            credentials_source = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', '').strip()
            if credentials_source:
                logger.debug(f"Using credentials from GOOGLE_APPLICATION_CREDENTIALS: {credentials_source}")
        
        # Priority 5: google_service_account.json in project root (local dev convenience)
        if not credentials_source and not creds_data:
            default_json_path = 'google_service_account.json'
            if os.path.exists(default_json_path):
                credentials_source = default_json_path
                logger.debug(f"Using credentials from {default_json_path} (local dev)")
        
        # Validation: Did we find any credentials?
        if not credentials_source and not creds_data:
            raise ValueError(
                "Google Sheets credentials not configured. Configure one of:\n\n"
                "  OPTION 1 (Azure Recommended) - Individual environment variables:\n"
                "    GOOGLE_SHEETS_SA_TYPE, GOOGLE_SHEETS_SA_PROJECT_ID,\n"
                "    GOOGLE_SHEETS_SA_PRIVATE_KEY_ID, GOOGLE_SHEETS_SA_PRIVATE_KEY,\n"
                "    GOOGLE_SHEETS_SA_CLIENT_EMAIL, GOOGLE_SHEETS_SA_CLIENT_ID,\n"
                "    GOOGLE_SHEETS_SA_AUTH_URI, GOOGLE_SHEETS_SA_TOKEN_URI,\n"
                "    GOOGLE_SHEETS_SA_AUTH_PROVIDER_CERT_URL, GOOGLE_SHEETS_SA_CLIENT_CERT_URL,\n"
                "    GOOGLE_SHEETS_SA_UNIVERSE_DOMAIN\n\n"
                "  OPTION 2 (Local Dev) - JSON file:\n"
                "    A) GOOGLE_APPLICATION_CREDENTIALS=/path/to/google_service_account.json\n"
                "    B) Place google_service_account.json in project root\n\n"
                "  OPTION 3 (Legacy) - Complete JSON string:\n"
                "    GOOGLE_SHEETS_ELECTION_DB_LITE_CREDENTIALS='{...}'\n"
            )

        if not self.spreadsheet_id:
            raise ValueError(
                "Spreadsheet ID not configured. Set GOOGLE_SHEETS_DB_LITE_ID environment variable."
            )
        
        try:
            if creds_data:
                # Use pre-built JSON dict from individual env vars
                self.credentials = service_account.Credentials.from_service_account_info(creds_data)
                logger.info("✓ Google Sheets authentication successful (from env vars)")
            elif os.path.exists(credentials_source):
                # File path - try loading JSON
                creds_data = _load_credentials_from_file(credentials_source)
                if creds_data:
                    self.credentials = service_account.Credentials.from_service_account_info(creds_data)
                    logger.info(f"✓ Google Sheets authentication successful (from file: {credentials_source})")
                else:
                    raise FileNotFoundError(f"Could not parse JSON from {credentials_source}")
            else:
                # Assume it's a JSON string
                creds_data = json.loads(credentials_source)
                self.credentials = service_account.Credentials.from_service_account_info(creds_data)
                logger.info("✓ Google Sheets authentication successful (from JSON string)")
            
            # Add required scopes
            self.credentials = self.credentials.with_scopes(
                ['https://www.googleapis.com/auth/spreadsheets.readonly']
            )
            
        except (FileNotFoundError, json.JSONDecodeError, GoogleAuthError) as e:
            logger.error(f"✗ Failed to initialize Google Sheets client: {e}")
            raise
    
    def fetch_sheet(self, sheet_name: str, skip_header: bool = True) -> SheetFetchResult:
        """
        Fetch data from a sheet.
        
        Args:
            sheet_name: Name of sheet (case-insensitive)
            skip_header: Whether to skip first row (assumed to be headers)
            
        Returns:
            SheetFetchResult with records and metadata
        """
        start_time = datetime.utcnow()
        
        try:
            from googleapiclient.discovery import build
            
            # Build the Sheets service
            service = build('sheets', 'v4', credentials=self.credentials)
            
            # Normalize sheet name
            normalized_name = sheet_name.lower().strip()
            
            # Get sheet metadata to find correct sheet ID
            sheet_metadata = service.spreadsheets().get(
                spreadsheetId=self.spreadsheet_id,
                fields='sheets(properties(sheetId,title))'
            ).execute()
            
            sheet_id = None
            actual_sheet_name = None
            
            for sheet in sheet_metadata.get('sheets', []):
                if sheet['properties']['title'].lower() == normalized_name:
                    sheet_id = sheet['properties']['sheetId']
                    actual_sheet_name = sheet['properties']['title']
                    break
            
            if sheet_id is None:
                return SheetFetchResult(
                    success=False,
                    sheet_name=sheet_name,
                    records=[],
                    row_count=0,
                    fetch_time=datetime.utcnow(),
                    error=f"Sheet '{sheet_name}' not found in workbook. Available sheets: {[s['properties']['title'] for s in sheet_metadata.get('sheets', [])]}",
                )
            
            # Fetch the sheet data
            result = service.spreadsheets().values().get(
                spreadsheetId=self.spreadsheet_id,
                range=f"'{actual_sheet_name}'!A:Z"  # Fetch up to column Z
            ).execute()
            
            rows = result.get('values', [])
            
            if not rows:
                return SheetFetchResult(
                    success=True,
                    sheet_name=actual_sheet_name,
                    records=[],
                    row_count=0,
                    fetch_time=datetime.utcnow(),
                    warnings=["Sheet is empty"],
                )
            
            # Extract headers
            headers = rows[0] if rows else []
            data_rows = rows[1:] if skip_header and len(rows) > 1 else rows
            
            # Convert rows to dictionaries
            records = []
            for row_idx, row in enumerate(data_rows, start=2):  # Start at 2 (after header)
                # Pad row with empty strings to match header length
                padded_row = row + [''] * (len(headers) - len(row))
                
                record = {}
                for col_idx, header in enumerate(headers):
                    if col_idx < len(padded_row):
                        record[header] = padded_row[col_idx]
                
                # Skip completely empty rows
                if any(record.values()):
                    records.append(record)
            
            fetch_duration = (datetime.utcnow() - start_time).total_seconds()
            
            logger.info(
                f"✓ Fetched '{actual_sheet_name}': "
                f"{len(records)} records, {len(headers)} columns in {fetch_duration:.2f}s"
            )
            
            return SheetFetchResult(
                success=True,
                sheet_name=actual_sheet_name,
                records=records,
                row_count=len(records),
                fetch_time=start_time,
            )
            
        except Exception as e:
            logger.error(f"✗ Failed to fetch sheet '{sheet_name}': {e}")
            return SheetFetchResult(
                success=False,
                sheet_name=sheet_name,
                records=[],
                row_count=0,
                fetch_time=datetime.utcnow(),
                error=str(e),
            )
    
    def fetch_all_sheets(self) -> Dict[str, SheetFetchResult]:
        """
        Fetch all expected sheets from the workbook.
        
        Returns:
            Dictionary mapping sheet names to SheetFetchResult objects
        """
        results = {}
        
        for sheet_name in self.EXPECTED_SHEETS.keys():
            results[sheet_name] = self.fetch_sheet(sheet_name)
        
        # Summary logging
        successful = sum(1 for r in results.values() if r.success)
        total_records = sum(r.row_count for r in results.values() if r.success)
        
        logger.info(f"✓ Fetched {successful}/{len(self.EXPECTED_SHEETS)} sheets with {total_records} total records")
        
        return results
    
    def fetch_finalized_data(self) -> SheetFetchResult:
        """Convenience method to fetch Finalized Data sheet"""
        return self.fetch_sheet('finalized data')
    
    def fetch_down_ballot_calculations(self) -> SheetFetchResult:
        """Convenience method to fetch Down-Ballot Calculations sheet"""
        return self.fetch_sheet('down-ballot calculations')
    
    def validate_columns(self, sheet_name: str, records: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """
        Validate that fetched sheet has expected columns.
        
        Returns:
            (is_valid, list of missing/extra columns)
        """
        if not records:
            return True, []
        
        expected_columns = None
        
        if sheet_name.lower() == 'finalized data':
            expected_columns = set(self.FINALIZED_DATA_COLUMNS.values())
        elif sheet_name.lower() == 'down-ballot calculations':
            expected_columns = set(self.DOWN_BALLOT_COLUMNS.values())
        
        if not expected_columns:
            return True, []  # No validation for unknown sheets
        
        actual_columns = set(records[0].keys())
        
        missing = expected_columns - actual_columns
        extra = actual_columns - expected_columns
        
        issues = []
        if missing:
            issues.append(f"Missing columns: {missing}")
        if extra:
            issues.append(f"Unexpected columns: {extra}")
        
        return len(issues) == 0, issues


# Convenience function for module usage
def get_election_data_client(credentials_json: Optional[str] = None) -> GoogleSheetsElectionClient:
    """Create a GoogleSheetsElectionClient instance"""
    return GoogleSheetsElectionClient(credentials_json)


def get_worklist_client(credentials_json: Optional[str] = None) -> GoogleSheetsElectionClient:
    """
    Create a GoogleSheetsElectionClient instance for the worklist sheet.

    Uses GOOGLE_SHEETS_WORKLIST_ID by default.
    """
    worklist_id = os.getenv('GOOGLE_SHEETS_WORKLIST_ID', '').strip()
    if not worklist_id:
        raise ValueError("GOOGLE_SHEETS_WORKLIST_ID not configured")
    return GoogleSheetsElectionClient(credentials_json, spreadsheet_id=worklist_id)


def fetch_worklist_overview(
    credentials_json: Optional[str] = None,
    sheet_name: Optional[str] = None,
) -> SheetFetchResult:
    """
    Fetch the overview sheet from the worklist spreadsheet.

    Uses GOOGLE_SHEETS_WORKLIST_OVERVIEW_SHEET when sheet_name is not provided.
    Defaults to "Overview" if the env var is not set.
    """
    client = get_worklist_client(credentials_json)
    overview_name = sheet_name or os.getenv("GOOGLE_SHEETS_WORKLIST_OVERVIEW_SHEET", "Overview")
    return client.fetch_sheet(overview_name)
