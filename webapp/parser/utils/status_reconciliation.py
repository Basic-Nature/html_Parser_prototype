"""
Status Reconciliation System

Maps between:
1. Parser status (.processed_urls) - URL-based, technical
2. Worklist status (Google Sheets) - Contest/state-based, workflow

Purpose: Determine the TRUE authoritative status for display
"""
from __future__ import annotations

from typing import Optional, Dict, Any, Tuple
from datetime import datetime


class StatusReconciliation:
    """
    Reconcile parser vs worklist status.
    
    Authority hierarchy:
    1. Parser status (if URL was processed)
    2. Worklist status (if contest/state is tracked)
    3. Default to "pending" (if no data)
    """
    
    # Parser status values (authoritative when present)
    PARSER_STATUSES = {
        'success': 'Parsed',
        'fail': 'Failed',
        'error': 'Error',
        'partial': 'Partial',
        'cancelled': 'Cancelled',
        'rejected': 'Rejected',
        'quarantined': 'Quarantined',
        'skipped_data_exists': 'Skipped (Exists)',
        'pending': 'Pending'
    }
    
    # Worklist status values (from Google Sheets)
    WORKLIST_STATUSES = {
        'PROD Loaded': 'Production',
        'QC Loaded': 'QC Complete',
        'QC2 Fail/Fix': 'QC2 Failed',
        'QC1 Fail/Fix': 'QC1 Failed',
        'Pre-QC Fail/Fix': 'Pre-QC Failed',
        'Cand Check DL1': 'Candidate Check',
        'Download Needed': 'Download Needed',
        'DL1 Processing': 'DL1 Processing',
        'DL2 Processing': 'DL2 Processing',
        'Draft': 'Draft',
    }
    
    # Status badges with colors and icons
    STATUS_BADGES = {
        # Parser statuses
        'success': {'icon': '✅', 'label': 'Success', 'badge_class': 'success', 'priority': 1},
        'fail': {'icon': '❌', 'label': 'Failed', 'badge_class': 'error', 'priority': 5},
        'error': {'icon': '⚠️', 'label': 'Error', 'badge_class': 'error', 'priority': 6},
        'partial': {'icon': '🔸', 'label': 'Partial', 'badge_class': 'warning', 'priority': 4},
        'cancelled': {'icon': '⏹️', 'label': 'Cancelled', 'badge_class': 'warning', 'priority': 7},
        'rejected': {'icon': '🚫', 'label': 'Rejected', 'badge_class': 'danger', 'priority': 8},
        'quarantined': {'icon': '⚠️', 'label': 'Quarantine', 'badge_class': 'warning', 'priority': 9},
        'skipped_data_exists': {'icon': '⏭️', 'label': 'Skipped', 'badge_class': 'info', 'priority': 2},
        'pending': {'icon': '⏳', 'label': 'Pending', 'badge_class': 'secondary', 'priority': 10},
        
        # Worklist statuses
        'production': {'icon': '📦', 'label': 'Production', 'badge_class': 'success', 'priority': 1},
        'qc_complete': {'icon': '✓', 'label': 'QC Complete', 'badge_class': 'success', 'priority': 2},
        'qc2_failed': {'icon': '❌', 'label': 'QC2 Failed', 'badge_class': 'error', 'priority': 5},
        'qc1_failed': {'icon': '❌', 'label': 'QC1 Failed', 'badge_class': 'error', 'priority': 5},
        'preqc_failed': {'icon': '❌', 'label': 'Pre-QC Failed', 'badge_class': 'error', 'priority': 5},
        'candidate_check': {'icon': '🔍', 'label': 'Candidate Check', 'badge_class': 'info', 'priority': 3},
        'download_needed': {'icon': '📥', 'label': 'Download Needed', 'badge_class': 'warning', 'priority': 6},
        'dl1_processing': {'icon': '⚙️', 'label': 'DL1 Processing', 'badge_class': 'info', 'priority': 4},
        'dl2_processing': {'icon': '⚙️', 'label': 'DL2 Processing', 'badge_class': 'info', 'priority': 4},
        'draft': {'icon': '📝', 'label': 'Draft', 'badge_class': 'secondary', 'priority': 7},
    }
    
    @staticmethod
    def reconcile(
        url: str,
        parser_status: Optional[str] = None,
        worklist_status: Optional[str] = None,
        production_source: Optional[str] = None,
        last_processed: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Determine the TRUE status by reconciling parser and worklist.
        
        Returns:
            (status_key, status_info) where:
            - status_key: Canonical status identifier
            - status_info: Dict with icon, label, badge_class, source, authority
        """
        
        authority = None
        canonical_status = None
        
        # 1. Parser status is AUTHORITATIVE if present
        if parser_status and parser_status in StatusReconciliation.PARSER_STATUSES:
            canonical_status = parser_status
            authority = 'parser'
            
            # Special case: if skipped_data_exists, mark it
            if parser_status == 'skipped_data_exists':
                return 'skipped_data_exists', {
                    **StatusReconciliation.STATUS_BADGES.get('skipped_data_exists', {}),
                    'source': production_source or 'database',
                    'authority': 'parser',
                    'reason': 'Data already in production'
                }
        
        # 2. If no parser status but worklist status exists, use worklist
        elif worklist_status:
            worklist_normalized = StatusReconciliation._normalize_worklist_status(worklist_status)
            canonical_status = worklist_normalized
            authority = 'worklist'
        
        # 3. Default to pending
        else:
            canonical_status = 'pending'
            authority = 'default'
        
        # Build status info
        status_info = StatusReconciliation.STATUS_BADGES.get(
            canonical_status,
            {'icon': '❓', 'label': canonical_status, 'badge_class': 'secondary', 'priority': 99}
        )
        
        return canonical_status, {
            **status_info,
            'source': parser_status or worklist_status or 'not_processed',
            'authority': authority,
            'last_processed': last_processed,
            'parsed': parser_status is not None,
            'in_worklist': worklist_status is not None,
        }
    
    @staticmethod
    def _normalize_worklist_status(worklist_status: str) -> str:
        """
        Convert worklist status string to canonical key.
        """
        if not worklist_status:
            return 'pending'
        
        status_lower = worklist_status.strip().lower()
        
        # Direct mappings
        mappings = {
            'prod loaded': 'production',
            'qc loaded': 'qc_complete',
            'qc2 fail/fix': 'qc2_failed',
            'qc1 fail/fix': 'qc1_failed',
            'pre-qc fail/fix': 'preqc_failed',
            'cand check dl1': 'candidate_check',
            'download needed': 'download_needed',
            'dl1 processing': 'dl1_processing',
            'dl2 processing': 'dl2_processing',
            'draft': 'draft',
        }
        
        return mappings.get(status_lower, 'pending')
    
    @staticmethod
    def get_status_priority(status: str) -> int:
        """
        Get priority for sorting (lower = higher priority).
        Used to show most important statuses first.
        """
        badge_info = StatusReconciliation.STATUS_BADGES.get(status, {})
        return badge_info.get('priority', 99)
    
    @staticmethod
    def status_requires_action(status: str) -> bool:
        """
        Determine if status requires manual action.
        """
        action_required = {
            'fail', 'error', 'rejected', 'quarantined',
            'qc2_failed', 'qc1_failed', 'preqc_failed',
            'download_needed', 'candidate_check'
        }
        return status in action_required
    
    @staticmethod
    def status_is_complete(status: str) -> bool:
        """
        Determine if status means processing is done.
        """
        complete_statuses = {
            'success', 'production', 'qc_complete',
            'skipped_data_exists'
        }
        return status in complete_statuses


class WorklistParser:
    """
    Parse and filter Google Sheets worklist data.
    
    Removes PII columns (DL1/DL2 assignees) from public view.
    """
    
    # Columns that should never be exposed publicly (contain PII)
    PII_COLUMNS = [
        'Work in Progress - DL1',
        'Work in Progress - DL2',
        'DL1',
        'DL2',
        'Assigned To',
        'Email',
        'Phone',
    ]
    
    @staticmethod
    def sanitize_row(row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove PII columns from worklist row.
        """
        return {k: v for k, v in row.items() if k not in WorklistParser.PII_COLUMNS}
    
    @staticmethod
    def extract_contest_key(
        year: Optional[str],
        state: Optional[str],
        race: Optional[str]
    ) -> Optional[str]:
        """
        Create a contest key for matching against parser URLs.
        """
        if not (year and state and race):
            return None
        
        # Normalize and create key
        year = str(year).strip()
        state = state.strip()
        race = race.strip()
        
        # Normalize state (handle abbreviations)
        state_norm = _normalize_state(state)
        
        # Normalize race/contest name
        race_norm = race.lower().replace(' of representatives', '').replace('statewide', '').strip()
        
        return f"{year}_{state_norm}_{race_norm}".lower()
    
    @staticmethod
    def get_public_columns() -> list:
        """
        Get list of columns safe to expose publicly.
        """
        all_columns = [
            'Priority', 'Sprint', 'Status',
            'Year', 'Race', 'State',
            'Download 1', 'Download 2', 'Source Link'
        ]
        return [col for col in all_columns if col not in WorklistParser.PII_COLUMNS]


def _normalize_state(state: str) -> str:
    """
    Normalize state name to standard format.
    """
    state_lower = state.strip().lower()
    
    # Common mappings
    state_map = {
        'al': 'alabama', 'ak': 'alaska', 'az': 'arizona', 'ar': 'arkansas',
        'ca': 'california', 'co': 'colorado', 'ct': 'connecticut', 'de': 'delaware',
        'fl': 'florida', 'ga': 'georgia', 'hi': 'hawaii', 'id': 'idaho',
        'il': 'illinois', 'in': 'indiana', 'ia': 'iowa', 'ks': 'kansas',
        'ky': 'kentucky', 'la': 'louisiana', 'me': 'maine', 'md': 'maryland',
        'ma': 'massachusetts', 'mi': 'michigan', 'mn': 'minnesota', 'ms': 'mississippi',
        'mo': 'missouri', 'mt': 'montana', 'ne': 'nebraska', 'nv': 'nevada',
        'nh': 'new hampshire', 'nj': 'new jersey', 'nm': 'new mexico', 'ny': 'new york',
        'nc': 'north carolina', 'nd': 'north dakota', 'oh': 'ohio', 'ok': 'oklahoma',
        'or': 'oregon', 'pa': 'pennsylvania', 'ri': 'rhode island', 'sc': 'south carolina',
        'sd': 'south dakota', 'tn': 'tennessee', 'tx': 'texas', 'ut': 'utah',
        've': 'vermont', 'va': 'virginia', 'wa': 'washington', 'wv': 'west virginia',
        'wi': 'wisconsin', 'wy': 'wyoming',
    }
    
    # If abbreviation, convert to full name
    if state_lower in state_map:
        return state_map[state_lower]
    
    # If already full name, return as-is
    return state_lower.replace(' ', '_')
