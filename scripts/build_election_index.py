#!/usr/bin/env python3
"""
Build election results fixture index from CSV sources.

Usage:
  python scripts/build_election_index.py \\
    --src webapp/parser/fixtures \\
    --out webapp/parser/fixtures/election_results_index.json \\
    [--audit-against-fec] \\
    [--max-file-size-mb 10] \\
    [--shard-threshold-mb 8] \\
    [--include-cache] \\
    [--include-log] \\
    [--min-confidence 0.7]

Data Flow Architecture:
  1. Handlers download election data from URLs → CSV format
  2. CSV → JSON conversion (local, manual) → webapp/parser/fixtures/
  3. Handler extraction → Cache (webapp/parser/Context_Integration/Context_Library/cache/)
  4. Append-only logs → Log (webapp/parser/Context_Integration/Context_Library/log/)
  5. Index builder reads from fixtures, cache, log → merges with confidence filtering
  6. Final validated data → PostgreSQL warehouse (long-term storage)

Sources (priority order):
  - Primary: webapp/parser/fixtures/ (CSVs, JSON, JSONL, shards)
  - Cache: Context_Library/cache/ (short-term, ready for migration based on confidence)
  - Logs: Context_Library/log/ (append-only JSONL, slightly more persistent)

Features:
  - Reads SMART Elections Database CSVs (Finalized Data, Down-Ballot Calculations)
  - Reads JSON/JSONL from cache and log directories (handler outputs)
  - Organizes by state + year + contest
  - Validates against JSON Schema
  - Deduplicates and cleans data
  - Filters by confidence threshold (default: no filter, use --min-confidence to set)
  - Monitors file size; shards by state if exceeds threshold
  - Optional FEC candidate name fuzzy matching (--audit-against-fec)
  - Generates audit report (fixture_audit_report.jsonl)
"""

import argparse
import csv
import gzip
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False
    jsonschema = None  # type: ignore[assignment]

try:
    from rapidfuzz import fuzz as fuzzy_fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False
    try:
        from difflib import SequenceMatcher
    except ImportError:
        pass


# ============================================================================
# Logging & Utilities
# ============================================================================

class Auditor:
    """Track fixture audit events for reporting."""
    def __init__(self):
        self.events = []
    
    def log(self, event_type: str, row: int, file: str, message: str, **extra):
        self.events.append({
            "type": event_type,
            "row": row,
            "file": file,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **extra
        })
    
    def save(self, out_path: str):
        """Write audit log as JSONL."""
        with open(out_path, 'w', encoding='utf-8') as f:
            for evt in self.events:
                f.write(json.dumps(evt) + '\n')
        print(f"[AUDIT] Wrote {len(self.events)} events to {out_path}")


def fuzzy_match_candidate(name: str, candidates_index: Dict) -> Optional[Tuple[str, int]]:
    """
    Fuzzy match candidate name against FEC index.
    Returns (cand_id, score) or None.
    """
    if not candidates_index:
        return None
    
    name_clean = name.strip().upper()
    best_id, best_score = None, 0
    
    if HAS_RAPIDFUZZ:
        for cand_id, cand_data in candidates_index.items():
            cand_name = cand_data.get("CLYMER", "").upper()
            score = fuzzy_fuzz.token_sort_ratio(name_clean, cand_name)
            if score > best_score:
                best_score = score
                best_id = cand_id
    else:
        # Fallback: difflib
        for cand_id, cand_data in candidates_index.items():
            cand_name = cand_data.get("CLYMER", "").upper()
            ratio = SequenceMatcher(None, name_clean, cand_name).ratio()
            score = int(ratio * 100)
            if score > best_score:
                best_score = score
                best_id = cand_id
    
    if best_score >= 70:  # Threshold
        return best_id, best_score
    return None


# ============================================================================
# CSV Parsing
# ============================================================================

def parse_votes(value: str) -> Optional[int]:
    """Parse vote count from string."""
    if not value or isinstance(value, str) and value.strip().upper() in ('', 'NA', 'N/A', 'NULL'):
        return None
    try:
        return int(float(str(value).replace(',', '').strip()))
    except (ValueError, TypeError):
        return None


def parse_party(value: str) -> Optional[str]:
    """Normalize party affiliation."""
    if not value:
        return None
    val = str(value).strip().upper()
    party_map = {
        'DEMOCRAT': 'DEM',
        'DEMOCRATIC': 'DEM',
        'REPUBLICAN': 'REP',
        'INDEPENDENT': 'IND',
        'LIBERTARIAN': 'LIB',
        'GREEN': 'GRN',
        'PROGRESSIVE': 'PRO',
        'WORKING FAMILIES': 'WFM',
    }
    return party_map.get(val, val) if val else None


def read_smart_database_csv(csv_path: str, auditor: Auditor) -> Dict[str, List[Dict[str, Any]]]:
    """
    Read SMART Elections Database CSV.
    Returns dict: {state_year: [records]}
    """
    records_by_state_year = defaultdict(list)
    
    try:
        with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                auditor.log('error', 0, csv_path, "No CSV headers found")
                return records_by_state_year
            
            for row_num, row in enumerate(reader, start=2):  # Start at 2 (header = 1)
                # Extract state, year
                state = (row.get('State') or '').strip().upper()
                year_str = (row.get('Year') or '').strip()
                county = (row.get('County') or '').strip()
                party = parse_party(row.get('Party'))
                
                if not state or not year_str:
                    auditor.log('skip', row_num, csv_path, 
                                "Missing state or year; skipping row",
                                row_data={k: v for k, v in row.items() if v})
                    continue
                
                try:
                    year = int(year_str)
                except ValueError:
                    auditor.log('skip', row_num, csv_path, 
                                f"Invalid year format: {year_str}",
                                row_data=dict(row))
                    continue
                
                # Parse data
                pres_votes = parse_votes(row.get('Presidential Votes'))
                ballot_votes = parse_votes(row.get('Down-Ballot Votes'))
                
                if pres_votes is None and ballot_votes is None:
                    auditor.log('skip', row_num, csv_path, 
                                "No vote data found",
                                row_data=dict(row))
                    continue
                
                key = f"{state}_{year}"
                records_by_state_year[key].append({
                    'state': state,
                    'year': year,
                    'county': county,
                    'party': party,
                    'presidential_votes': pres_votes,
                    'down_ballot_votes': ballot_votes,
                    'row_num': row_num,
                    'source': Path(csv_path).name,
                })
    
    except Exception as e:
        auditor.log('error', 0, csv_path, f"Failed to read CSV: {e}")
    
    return records_by_state_year


# ============================================================================
# JSON/JSONL Parsing (fixture cache sources)
# ============================================================================

def _open_text_file(path: str):
    if path.endswith('.gz'):
        return gzip.open(path, 'rt', encoding='utf-8')
    return open(path, 'r', encoding='utf-8')


def _merge_index(target: Dict[str, Dict[str, Any]], source: Dict[str, Dict[str, Any]],
                 auditor: Auditor, source_name: str) -> None:
    for key, record in source.items():
        if key in target:
            auditor.log('warning', 0, source_name, f"Duplicate key in index: {key}")
        target[key] = record


def read_json_index_file(path: str, auditor: Auditor) -> Dict[str, Dict[str, Any]]:
    try:
        with _open_text_file(path) as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload
        auditor.log('error', 0, path, "JSON index is not an object")
        return {}
    except Exception as e:
        auditor.log('error', 0, path, f"Failed to read JSON index: {e}")
        return {}


def read_jsonl_index_file(path: str, auditor: Auditor) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    try:
        with _open_text_file(path) as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as e:
                    auditor.log('error', line_num, path, f"Invalid JSONL: {e}")
                    continue
                if not isinstance(entry, dict):
                    auditor.log('skip', line_num, path, "JSONL entry is not an object")
                    continue

                key = entry.get('key') or entry.get('id')
                record = entry.get('record') if isinstance(entry.get('record'), dict) else None
                if record is None:
                    record = {k: v for k, v in entry.items() if k not in ('key', 'id')}

                if not key:
                    meta = record.get('metadata') if isinstance(record, dict) else None
                    if isinstance(meta, dict):
                        state = meta.get('state')
                        year = meta.get('year')
                        contest = record.get('contest')
                        if state and year and contest:
                            key = f"{state}_{year}_{contest}"

                if not key or not isinstance(record, dict):
                    auditor.log('skip', line_num, path, "Missing key or record in JSONL entry")
                    continue

                if key in index:
                    auditor.log('warning', line_num, path, f"Duplicate key in JSONL: {key}")
                index[key] = record
    except Exception as e:
        auditor.log('error', 0, path, f"Failed to read JSONL index: {e}")
    return index


def load_index_from_json_sources(
    src_dir: str,
    auditor: Auditor,
    include_cache: bool = True,
    include_log: bool = True,
    min_confidence: float = 0.0,
) -> Dict[str, Dict[str, Any]]:
    """
    Load fixture index from JSON/JSONL sources in fixtures, cache, and log directories.
    
    Args:
        src_dir: Primary fixtures directory
        auditor: Audit tracker
        include_cache: Whether to load from cache directory
        include_log: Whether to load from log directory
        min_confidence: Minimum confidence threshold (0.0-1.0) for including records
    
    Returns:
        Merged index dict
    """
    index: Dict[str, Dict[str, Any]] = {}
    src_path = Path(src_dir)

    # Primary fixtures sources
    index_path = src_path / 'election_results_index.json'
    if index_path.exists():
        _merge_index(index, read_json_index_file(str(index_path), auditor), auditor, str(index_path))

    shard_dir = src_path / 'election_results_shards'
    if shard_dir.exists():
        for shard_file in shard_dir.glob('election_results_*.json'):
            _merge_index(index, read_json_index_file(str(shard_file), auditor), auditor, str(shard_file))

    jsonl_path = src_path / 'election_results_index.jsonl'
    if jsonl_path.exists():
        _merge_index(index, read_jsonl_index_file(str(jsonl_path), auditor), auditor, str(jsonl_path))

    jsonl_gz_path = src_path / 'election_results_index.jsonl.gz'
    if jsonl_gz_path.exists():
        _merge_index(index, read_jsonl_index_file(str(jsonl_gz_path), auditor), auditor, str(jsonl_gz_path))

    # Context_Library cache sources (short-term data ready for migration)
    if include_cache:
        cache_dir = src_path.parent.parent / 'Context_Integration' / 'Context_Library' / 'cache'
        if cache_dir.exists():
            for cache_file in cache_dir.glob('*.json'):
                if cache_file.name.startswith('election') or cache_file.name.startswith('context'):
                    _merge_index(index, read_json_index_file(str(cache_file), auditor), auditor, str(cache_file))

    # Context_Library log sources (append-only JSONL logs)
    if include_log:
        log_dir = src_path.parent.parent / 'Context_Integration' / 'Context_Library' / 'log'
        if log_dir.exists():
            for log_file in log_dir.glob('*.jsonl'):
                if any(kw in log_file.name for kw in ['field_selection', 'navigation_learning', 'integrity', 'trust']):
                    _merge_index(index, read_jsonl_index_file(str(log_file), auditor), auditor, str(log_file))

    # Apply confidence threshold filter
    if min_confidence > 0.0:
        filtered_index = {}
        for key, record in index.items():
            confidence = record.get('confidence', 1.0)
            if isinstance(confidence, (int, float)) and confidence >= min_confidence:
                filtered_index[key] = record
            else:
                auditor.log('skip', 0, key, f"Below confidence threshold: {confidence} < {min_confidence}")
        index = filtered_index

    return index


# ============================================================================
# Index Building
# ============================================================================

def build_election_index(
    records_by_state_year: Dict[str, List[Dict]],
    candidates_index: Optional[Dict] = None,
    auditor: Optional[Auditor] = None,
    audit_against_fec: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Transform records into state_year keyed index with contests.
    """
    if auditor is None:
        auditor = Auditor()
    
    index = {}
    
    for state_year, records in records_by_state_year.items():
        state, year = state_year.split('_')
        year = int(year)
        
        # Group by contest (using party + county as proxy for now)
        # In real data, this would be determined contest type
        contests = defaultdict(lambda: defaultdict(int))
        
        for rec in records:
            party = rec.get('party') or 'UNKNOWN'
            county = rec.get('county') or 'Statewide'
            contest_key = f"{year} General Election - {county}"
            
            votes_to_use = rec.get('presidential_votes') or rec.get('down_ballot_votes') or 0
            
            # Accumulate votes by party
            contests[contest_key][party] += votes_to_use
        
        # Build index entry for this state_year
        state_year_index = {}
        
        for contest_name, party_votes in contests.items():
            candidates = []
            total_votes = sum(party_votes.values())
            
            for party, votes in sorted(party_votes.items(), key=lambda x: x[1], reverse=True):
                pct = (votes / total_votes * 100) if total_votes > 0 else 0
                
                # Determine winner (highest votes)
                is_winner = votes == max(party_votes.values())
                
                # Optional FEC audit
                fec_id = None
                if audit_against_fec and candidates_index:
                    match = fuzzy_match_candidate(party, candidates_index)
                    if match:
                        fec_id, score = match
                        if score < 85:
                            auditor.log('fuzzy_match', 0, state_year, 
                                      f"Fuzzy matched {party} to {fec_id} ({score}%)",
                                      party=party, fec_id=fec_id, score=score)
                
                candidate_record = {
                    'name': party,
                    'party': party,
                    'votes': votes,
                    'percent': round(pct, 2),
                    'winner': is_winner,
                }
                if fec_id:
                    candidate_record['fec_id'] = fec_id
                
                candidates.append(candidate_record)
            
            state_year_index[contest_name] = {
                'contest': contest_name,
                'candidates': candidates,
                'source': 'SMART Elections Database',
                'last_updated': datetime.now(timezone.utc).isoformat(),
                'confidence': 0.9,  # High confidence for verified fixtures
                'metadata': {
                    'state': state,
                    'year': year,
                    'total_votes': total_votes,
                    'num_candidates': len(candidates),
                }
            }
        
        # Flatten: state_year_contest_name → record
        for contest_name, record in state_year_index.items():
            key = f"{state}_{year}_{contest_name}"
            index[key] = record
    
    return index


# ============================================================================
# File Management & Sharding
# ============================================================================

def calculate_file_size(data: Dict) -> float:
    """Calculate JSON size in MB."""
    json_str = json.dumps(data)
    return len(json_str.encode('utf-8')) / (1024 * 1024)


def shard_index_by_state(
    index: Dict[str, Dict[str, Any]],
    max_file_size_mb: float = 10,
    shard_threshold_mb: float = 8,
) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """
    Split index into main (unsplit) and sharded (by state) portions.
    
    Returns:
      (main_index, sharded_indices_by_state)
    """
    main_index = dict(index)
    size_mb = calculate_file_size(main_index)
    
    if size_mb < shard_threshold_mb:
        return main_index, {}
    
    # Need to shard; organize by state
    print(f"[SHARD] Index size {size_mb:.2f}MB exceeds threshold {shard_threshold_mb}MB; sharding by state...")
    
    sharded = defaultdict(dict)
    for key, record in index.items():
        # Extract state from key (format: STATE_YEAR_CONTEST)
        parts = key.split('_', 2)
        if len(parts) >= 1:
            state = parts[0]
            sharded[state][key] = record
    
    # Validate shard sizes
    oversized_states = []
    for state, state_index in sharded.items():
        state_size_mb = calculate_file_size(state_index)
        if state_size_mb > max_file_size_mb:
            oversized_states.append((state, state_size_mb))
    
    if oversized_states:
        print(f"[WARN] Oversized state shards (max {max_file_size_mb}MB):")
        for state, size_mb in oversized_states:
            print(f"  {state}: {size_mb:.2f}MB")
    
    return {}, dict(sharded)


def write_index_files(
    main_index: Dict[str, Dict],
    sharded_indices: Dict[str, Dict],
    output_dir: str,
    compress: bool = False,
) -> Tuple[List[str], float]:
    """
    Write main index and sharded indices.
    
    Returns:
      (list_of_written_files, total_size_mb)
    """
    written = []
    total_size_mb = 0
    
    # Main index
    if main_index:
        main_path = os.path.join(output_dir, 'election_results_index.json')
        with open(main_path, 'w', encoding='utf-8') as f:
            json.dump(main_index, f, indent=2)
        size_mb = calculate_file_size(main_index)
        written.append(main_path)
        total_size_mb += size_mb
        print(f"[WRITE] {main_path} ({size_mb:.2f}MB)")
    
    # Sharded indices
    if sharded_indices:
        shard_dir = os.path.join(output_dir, 'election_results_shards')
        os.makedirs(shard_dir, exist_ok=True)
        
        for state, state_index in sharded_indices.items():
            shard_path = os.path.join(shard_dir, f'election_results_{state.lower()}.json')
            with open(shard_path, 'w', encoding='utf-8') as f:
                json.dump(state_index, f, indent=2)
            size_mb = calculate_file_size(state_index)
            written.append(shard_path)
            total_size_mb += size_mb
            print(f"[WRITE] {shard_path} ({size_mb:.2f}MB)")
    
    return written, total_size_mb


# ============================================================================
# Schema Validation
# ============================================================================

def validate_against_schema(
    index: Dict[str, Dict],
    schema_path: str,
    auditor: Auditor,
) -> bool:
    """Validate index entries against JSON Schema."""
    if not HAS_JSONSCHEMA:
        print("[WARN] jsonschema not installed; skipping validation")
        return True
    
    if not os.path.exists(schema_path):
        print(f"[WARN] Schema file not found: {schema_path}")
        return True
    
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema = json.load(f)
    
    errors = 0
    for key, record in index.items():
        try:
            jsonschema.validate(record, schema)
        except jsonschema.ValidationError as e:
            errors += 1
            auditor.log('validation_error', 0, key, str(e))
    
    if errors:
        print(f"[VALIDATE] {errors} validation errors (see audit report)")
        return False
    
    print(f"[VALIDATE] All {len(index)} entries valid against schema")
    return True


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--src', default='webapp/parser/fixtures',
                       help='Source directory containing CSV files')
    parser.add_argument('--out', default='webapp/parser/fixtures/election_results_index.json',
                       help='Output index file path')
    parser.add_argument('--schema', default='webapp/parser/fixtures/election_results_schema.json',
                       help='JSON Schema file for validation')
    parser.add_argument('--audit-against-fec', action='store_true',
                       help='Fuzzy match candidates against FEC candidate summary index')
    parser.add_argument('--fec-index', default='webapp/parser/fixtures/candidate_summary_index.json',
                       help='Path to FEC candidate summary index (for --audit-against-fec)')
    parser.add_argument('--max-file-size-mb', type=float, default=10,
                       help='Maximum file size in MB before sharding')
    parser.add_argument('--shard-threshold-mb', type=float, default=8,
                       help='Threshold to trigger sharding (default: 80% of max)')
    parser.add_argument('--audit-report', default='webapp/parser/fixtures/fixture_audit_report.jsonl',
                       help='Output audit report path')
    parser.add_argument('--include-cache', action='store_true', default=True,
                       help='Include cache directory JSON sources')
    parser.add_argument('--include-log', action='store_true', default=True,
                       help='Include log directory JSONL sources')
    parser.add_argument('--min-confidence', type=float, default=0.0,
                       help='Minimum confidence threshold for including records (0.0-1.0)')
    
    args = parser.parse_args()
    
    # Initialize auditor
    auditor = Auditor()
    
    print("[START] Building election results index...")
    print(f"  Source: {args.src}")
    print(f"  Output: {args.out}")
    print(f"  Max size: {args.max_file_size_mb}MB")
    print(f"  Shard threshold: {args.shard_threshold_mb}MB")
    
    # Load FEC index if audit mode
    candidates_index = None
    if args.audit_against_fec:
        if os.path.exists(args.fec_index):
            with open(args.fec_index, 'r', encoding='utf-8') as f:
                candidates_index = json.load(f)
            print(f"[FEC] Loaded {len(candidates_index)} candidate records")
        else:
            print(f"[WARN] FEC index not found: {args.fec_index}")
    
    # Read CSV files
    records_by_state_year = defaultdict(list)
    csv_files = [f for f in os.listdir(args.src) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"[CSV] No CSV files found in {args.src}; checking JSON/JSONL sources...")
        index = load_index_from_json_sources(
            args.src,
            auditor,
            include_cache=args.include_cache,
            include_log=args.include_log,
            min_confidence=args.min_confidence,
        )
        if not index:
            print(f"[ERROR] No CSV or JSON sources found in {args.src}")
            return 1
        print(f"[JSON] Loaded index with {len(index)} entries (min confidence: {args.min_confidence})")
    else:
        print(f"[CSV] Found {len(csv_files)} CSV files")
        
        for csv_file in csv_files:
            csv_path = os.path.join(args.src, csv_file)
            print(f"[CSV] Reading {csv_file}...")
            file_records = read_smart_database_csv(csv_path, auditor)
            for key, recs in file_records.items():
                records_by_state_year[key].extend(recs)
        
        print(f"[CSV] Parsed {sum(len(r) for r in records_by_state_year.values())} records")
        
        # Build index
        index = build_election_index(
            dict(records_by_state_year),
            candidates_index=candidates_index,
            auditor=auditor,
            audit_against_fec=args.audit_against_fec,
        )
    
    print(f"[INDEX] Built index with {len(index)} entries")
    
    # Validate
    if os.path.exists(args.schema):
        validate_against_schema(index, args.schema, auditor)
    
    # Shard if needed
    main_index, sharded_indices = shard_index_by_state(
        index,
        max_file_size_mb=args.max_file_size_mb,
        shard_threshold_mb=args.shard_threshold_mb,
    )
    
    # Write files
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    written_files, total_size_mb = write_index_files(
        main_index,
        sharded_indices,
        os.path.dirname(args.out) or '.',
    )
    
    # Save audit report
    os.makedirs(os.path.dirname(args.audit_report) or '.', exist_ok=True)
    auditor.save(args.audit_report)
    
    print(f"[COMPLETE] Wrote {len(written_files)} files ({total_size_mb:.2f}MB total)")
    print(f"[AUDIT] Saved report to {args.audit_report}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
