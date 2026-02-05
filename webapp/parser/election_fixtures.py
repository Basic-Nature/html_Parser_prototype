"""
Election results fixture loader with lazy caching (mirrors fec_lookup.py pattern).

Provides O(1) lookups for election result indices while supporting:
- Per-state fixture data access
- Fuzzy candidate name matching
- Metrics tracking (cache hits, mismatches)
- Thread-safe operations
"""

import json
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

try:
    from rapidfuzz import fuzz as fuzzy_fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

# Global lock for thread-safe cache operations
_CACHE_LOCK = threading.RLock()

# Global cache state
_ELECTION_RESULTS_INDEX: Optional[Dict[str, Any]] = None
_ELECTION_RESULTS_SHARDS: Dict[str, Dict[str, Any]] = {}
_CACHE_LOADED = False
_CACHE_METRICS = {
    'hits': 0,
    'misses': 0,
    'fuzzy_matches': 0,
    'fuzzy_mismatches': 0,
}


def _get_fixture_dir() -> Path:
    """Determine fixture directory path (relative to this file)."""
    return Path(__file__).parent.parent / 'webapp' / 'parser' / 'fixtures'


def load_election_results_index(force_reload: bool = False) -> Dict[str, Any]:
    """
    Lazy-load election results index with global caching.
    
    Returns main index or empty dict if not found.
    Uses thread-safe locking to prevent race conditions.
    """
    global _ELECTION_RESULTS_INDEX, _CACHE_LOADED, _CACHE_METRICS
    
    with _CACHE_LOCK:
        if _CACHE_LOADED and not force_reload:
            _CACHE_METRICS['hits'] += 1
            return _ELECTION_RESULTS_INDEX or {}
        
        fixture_dir = _get_fixture_dir()
        index_path = fixture_dir / 'election_results_index.json'
        
        if not index_path.exists():
            _CACHE_LOADED = True
            _ELECTION_RESULTS_INDEX = {}
            return {}
        
        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                _ELECTION_RESULTS_INDEX = json.load(f)
            _CACHE_LOADED = True
            _CACHE_METRICS['misses'] += 1
            return _ELECTION_RESULTS_INDEX or {}
        except Exception as e:
            print(f"[ERROR] Failed to load election index: {e}")
            _ELECTION_RESULTS_INDEX = {}
            _CACHE_LOADED = True
            return {}


def load_election_results_shards(force_reload: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Lazy-load sharded election results (by state).
    
    Returns dict: {state → {contest_key → record}}
    """
    global _ELECTION_RESULTS_SHARDS, _CACHE_METRICS
    
    with _CACHE_LOCK:
        if _ELECTION_RESULTS_SHARDS and not force_reload:
            _CACHE_METRICS['hits'] += 1
            return _ELECTION_RESULTS_SHARDS
        
        fixture_dir = _get_fixture_dir()
        shard_dir = fixture_dir / 'election_results_shards'
        
        if not shard_dir.exists():
            return {}
        
        try:
            shards = {}
            for shard_file in shard_dir.glob('election_results_*.json'):
                state = shard_file.stem.replace('election_results_', '').upper()
                with open(shard_file, 'r', encoding='utf-8') as f:
                    shards[state] = json.load(f)
            _ELECTION_RESULTS_SHARDS = shards
            _CACHE_METRICS['misses'] += 1
            return shards
        except Exception as e:
            print(f"[ERROR] Failed to load shards: {e}")
            _ELECTION_RESULTS_SHARDS = {}
            return {}


def get_results_by_state(
    state: str,
    year: Optional[int] = None,
    include_data_source: bool = True,
) -> List[Dict[str, Any]]:
    """
    Get election results for a state (and optional year).
    
    Args:
        state: Two-letter state code (e.g., 'TX')
        year: Optional year filter
        include_data_source: Whether to add 'data_source: "fixture"' to each record
    
    Returns:
        List of result records matching filters
    """
    state = state.upper()
    main_index = load_election_results_index()
    shards = load_election_results_shards()
    
    results = []
    
    # Check main index first
    for key, record in main_index.items():
        if key.startswith(f"{state}_"):
            parts = key.split('_', 2)
            if len(parts) >= 2:
                try:
                    rec_year = int(parts[1])
                    if year is None or rec_year == year:
                        record_copy = dict(record)
                        if include_data_source:
                            record_copy['data_source'] = 'fixture'
                        results.append(record_copy)
                except ValueError:
                    pass
    
    # Check sharded index
    if state in shards:
        for key, record in shards[state].items():
            parts = key.split('_', 2)
            if len(parts) >= 2:
                try:
                    rec_year = int(parts[1])
                    if year is None or rec_year == year:
                        record_copy = dict(record)
                        if include_data_source:
                            record_copy['data_source'] = 'fixture'
                        results.append(record_copy)
                except ValueError:
                    pass
    
    return results


def get_results_by_contest(
    state: str,
    year: int,
    contest: str,
    include_data_source: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Get election results for a specific contest.
    
    Args:
        state: Two-letter state code
        year: Election year
        contest: Contest name/identifier
        include_data_source: Whether to add 'data_source' field
    
    Returns:
        Result record or None
    """
    state = state.upper()
    contest_key = f"{state}_{year}_{contest}"
    
    main_index = load_election_results_index()
    shards = load_election_results_shards()
    
    # Try main index
    if contest_key in main_index:
        record = dict(main_index[contest_key])
        if include_data_source:
            record['data_source'] = 'fixture'
        return record
    
    # Try shards
    if state in shards and contest_key in shards[state]:
        record = dict(shards[state][contest_key])
        if include_data_source:
            record['data_source'] = 'fixture'
        return record
    
    return None


def find_candidate_by_name(
    name: str,
    state: Optional[str] = None,
    year: Optional[int] = None,
    threshold: int = 70,
) -> List[Dict[str, Any]]:
    """
    Fuzzy-match candidate name across fixture data.
    
    Args:
        name: Candidate name to search
        state: Optional state filter
        year: Optional year filter
        threshold: Fuzzy match score threshold (0-100)
    
    Returns:
        List of matching candidates with scores
    """
    main_index = load_election_results_index()
    shards = load_election_results_shards()
    
    matches = []
    name_clean = name.strip().upper()
    
    def _search_in_dict(idx_dict: Dict[str, Any]):
        for key, record in idx_dict.items():
            # Parse key for state/year filter
            parts = key.split('_', 2)
            if len(parts) >= 2:
                rec_state = parts[0].upper()
                try:
                    rec_year = int(parts[1])
                except ValueError:
                    continue
                
                if state and rec_state != state.upper():
                    continue
                if year and rec_year != year:
                    continue
                
                # Search candidates
                if 'candidates' in record:
                    for candidate in record['candidates']:
                        cand_name = candidate.get('name', '').upper()
                        
                        if HAS_RAPIDFUZZ:
                            score = fuzzy_fuzz.token_sort_ratio(name_clean, cand_name)
                        else:
                            from difflib import SequenceMatcher
                            ratio = SequenceMatcher(None, name_clean, cand_name).ratio()
                            score = int(ratio * 100)
                        
                        if score >= threshold:
                            matches.append({
                                'state': rec_state,
                                'year': rec_year,
                                'contest': record.get('contest'),
                                'candidate': dict(candidate),
                                'match_score': score,
                                'data_source': 'fixture',
                            })
                            _CACHE_METRICS['fuzzy_matches'] += 1
    
    _search_in_dict(main_index)
    for state_code, shard_dict in shards.items():
        _search_in_dict(shard_dict)
    
    # Sort by score descending
    matches.sort(key=lambda x: x['match_score'], reverse=True)
    
    if not matches:
        _CACHE_METRICS['fuzzy_mismatches'] += 1
    
    return matches


def get_cache_metrics() -> Dict[str, int]:
    """Return cache performance metrics."""
    with _CACHE_LOCK:
        return dict(_CACHE_METRICS)


def clear_cache():
    """Clear all cached data (for testing/reloading)."""
    global _ELECTION_RESULTS_INDEX, _ELECTION_RESULTS_SHARDS, _CACHE_LOADED
    
    with _CACHE_LOCK:
        _ELECTION_RESULTS_INDEX = None
        _ELECTION_RESULTS_SHARDS = {}
        _CACHE_LOADED = False


def reset_metrics():
    """Reset performance metrics."""
    global _CACHE_METRICS
    
    with _CACHE_LOCK:
        _CACHE_METRICS = {
            'hits': 0,
            'misses': 0,
            'fuzzy_matches': 0,
            'fuzzy_mismatches': 0,
        }


# ============================================================================
# Test / Demo
# ============================================================================

if __name__ == '__main__':
    print("Loading election results fixtures...")
    idx = load_election_results_index()
    print(f"  Main index: {len(idx)} entries")
    
    shards = load_election_results_shards()
    print(f"  Shards: {len(shards)} states")
    
    # Test by state
    for state in ['TX', 'CA', 'NY']:
        results = get_results_by_state(state)
        print(f"  {state}: {len(results)} results")
    
    # Test by contest
    result = get_results_by_contest('TX', 2020, '2020 General Election - Statewide')
    print(f"  TX 2020 General: {result is not None}")
    
    # Test fuzzy match
    matches = find_candidate_by_name('democratic', state='TX', threshold=50)
    print(f"  Fuzzy matches for 'democratic' in TX: {len(matches)}")
    
    # Print metrics
    print(f"\nCache metrics: {get_cache_metrics()}")
