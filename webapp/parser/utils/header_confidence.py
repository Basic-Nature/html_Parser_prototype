"""
Header mapping confidence scoring and validation.
Ensures only high-quality headers are migrated to the warehouse.
"""

from typing import Dict, Optional, Tuple

import logging

from ..config import HEADER_CONFIDENCE_THRESHOLD, HEADER_INSERT_CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)


# Known column aliases with confidence weights
COLUMN_ALIASES = {
    'candidate': {
        'exact': ['candidate', 'ballot_candidate', 'choice', 'nominee', 'name'],
        'fuzzy': ['cand', 'person', 'person_name', 'contestant'],
    },
    'party': {
        'exact': ['party', 'ballot_party', 'affiliation', 'party_affiliation'],
        'fuzzy': ['party_name', 'political_party', 'affil'],
    },
    'votes': {
        'exact': ['votes', 'total_votes', 'vote_count', 'reported_votes', 'vote_total'],
        'fuzzy': ['ballots', 'total', 'count', 'vote'],
    },
    'precinct': {
        'exact': ['precinct', 'ward', 'district', 'division', 'jurisdiction'],
        'fuzzy': ['precinct_id', 'location', 'area'],
    },
}


def get_header_confidence(header: str, target_column: str, weights: Optional[Dict] = None) -> float:
    """
    Score 0.0–1.0 based on how confident we are a header maps to the target column.
    
    Confidence levels:
    - 1.0 = exact match (case-insensitive)
    - 0.85+ = fuzzy match with high similarity
    - 0.70-0.84 = plausible fuzzy match (low similarity or broad category)
    - < 0.70 = no reliable match
    
    Args:
        header: The CSV header string to evaluate
        target_column: Target column name ('candidate', 'party', 'votes', 'precinct')
        weights: Optional custom weighting (not yet implemented)
    
    Returns:
        Confidence score 0.0–1.0
    """
    if not header or not target_column:
        return 0.0
    
    header_clean = header.strip().lower().replace('_', ' ')
    target_clean = target_column.strip().lower().replace('_', ' ')
    
    if target_column not in COLUMN_ALIASES:
        return 0.0
    
    aliases = COLUMN_ALIASES[target_column]
    
    # Exact match (highest confidence)
    if header_clean in [a.replace('_', ' ') for a in aliases.get('exact', [])]:
        return 1.0
    
    # Fuzzy substring match
    for exact_alias in aliases.get('exact', []):
        if header_clean == exact_alias.replace('_', ' '):
            return 1.0
    
    # Partial/fuzzy match: check if header contains key words
    for exact_alias in aliases.get('exact', []):
        exact_normalized = exact_alias.replace('_', ' ').lower()
        if exact_normalized in header_clean or header_clean in exact_normalized:
            # Strong fuzzy match
            return 0.85
    
    # Broader fuzzy category match
    for fuzzy_alias in aliases.get('fuzzy', []):
        fuzzy_normalized = fuzzy_alias.replace('_', ' ').lower()
        if fuzzy_normalized in header_clean or header_clean in fuzzy_normalized:
            return 0.70
    
    # No reliable match
    return 0.0


def validate_row_headers(
    headers: list[str],
    critical_columns: list[str],
    confidence_threshold: float = HEADER_CONFIDENCE_THRESHOLD,
) -> Tuple[bool, Dict[str, float], list[str]]:
    """
    Validate a set of CSV headers against critical columns.
    
    Args:
        headers: List of CSV header strings
        critical_columns: List of required column names (e.g., ['candidate', 'party', 'votes'])
        confidence_threshold: Minimum confidence score required (default from config)
    
    Returns:
        Tuple of (all_critical_found: bool, confidence_scores: dict, flagged_headers: list)
    """
    confidence_scores = {}
    flagged = []
    
    for col in critical_columns:
        best_header = None
        best_score = 0.0
        
        for header in headers:
            score = get_header_confidence(header, col)
            if score > best_score:
                best_score = score
                best_header = header
        
        confidence_scores[col] = best_score
        
        if best_score < confidence_threshold:
            flagged.append(f"{col}: {best_score:.2f} (best_match={best_header})")
    
    all_critical_found = all(confidence_scores.get(col, 0.0) >= confidence_threshold 
                             for col in critical_columns)
    
    return all_critical_found, confidence_scores, flagged


def should_insert_row(
    mapped_row: dict,
    confidence_scores: dict,
    confidence_threshold: float = HEADER_INSERT_CONFIDENCE_THRESHOLD,
) -> bool:
    """
    Determine if a row should be inserted based on header confidence.
    
    Returns True only if all critical fields exceed threshold.
    """
    critical = ['candidate', 'party', 'votes']
    for col in critical:
        if confidence_scores.get(col, 0.0) < confidence_threshold:
            return False
    
    # Also check that actual values are present
    for col in critical:
        if not mapped_row.get(col):
            return False
    
    return True
