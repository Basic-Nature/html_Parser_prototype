#!/usr/bin/env python3
"""Scan CSV/XLSX FEC export files and augment header alias & party maps.

Usage:
  python scripts/scan_fec_exports.py --dir path/to/dir [--apply]

By default this will run in dry-run mode and print suggested additions. Use
`--apply` to update the JSON maps; backups are created automatically.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from datetime import datetime
from typing import Dict, List, Set

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, '..'))
ALIASES_PATH = os.path.join(ROOT, 'webapp', 'parser', 'handlers', 'fec_header_aliases.json')
PARTY_MAP_PATH = os.path.join(ROOT, 'webapp', 'parser', 'handlers', 'fec_party_map.json')

try:
    import pandas as pd
except Exception:
    pd = None


def load_json(path: str) -> Dict:
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            return json.load(fh)
    except Exception:
        return {}


def write_json_backup(path: str) -> str:
    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    bak = f"{path}.{ts}.bak"
    shutil.copy2(path, bak)
    return bak


def normalize_token(t: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", t.strip()).strip('_')


def scan_file(path: str) -> Dict[str, Set[str]]:
    headers_found: Set[str] = set()
    party_tokens: Set[str] = set()
    ext = os.path.splitext(path)[1].lower().lstrip('.')
    if ext in ('xlsx', 'xls'):
        if pd is None:
            print(f"pandas not available; skipping Excel: {path}")
            return {"headers": set(), "parties": set()}
        try:
            df = pd.read_excel(path, sheet_name=0, dtype=str)
            headers_found.update([str(c).strip() for c in df.columns.tolist()])
            # try to find likely party column by header name
            candidate_cols = [c for c in df.columns if re.search(r'party|affil|affiliation', str(c), re.I)]
            for c in candidate_cols:
                vals = df[c].dropna().astype(str).unique().tolist()
                for v in vals:
                    party_tokens.add(v.strip())
        except Exception as e:
            print(f"Failed reading Excel {path}: {e}")
    else:
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as fh:
                reader = csv.reader(fh)
                header = next(reader, [])
                headers_found.update([str(h).strip() for h in header])
                # sample first 200 rows to collect party tokens
                party_idx = None
                for i, h in enumerate(header):
                    if re.search(r'party|affil|affiliation', str(h), re.I):
                        party_idx = i
                        break
                if party_idx is not None:
                    for i, row in enumerate(reader):
                        if i >= 200:
                            break
                        if len(row) > party_idx:
                            party_tokens.add(row[party_idx].strip())
        except Exception as e:
            print(f"Failed reading CSV {path}: {e}")
    return {"headers": headers_found, "parties": party_tokens}


def merge_aliases(existing: Dict[str, List[str]], observed: Set[str]) -> Dict[str, List[str]]:
    additions = {}
    # lowercase variant index
    variant_index = {}
    for canon, variants in existing.items():
        for v in variants:
            variant_index[v.lower()] = canon

    for h in sorted(observed):
        if not h:
            continue
        low = h.lower()
        if low in variant_index:
            continue
        # if header looks like an existing canonical key
        n = normalize_token(h).lower()
        if n in existing:
            existing[n].append(h)
            additions.setdefault(n, []).append(h)
            continue
        # try substring match to a canonical
        matched = False
        for canon in existing.keys():
            if canon.lower() in low or low in canon.lower():
                existing[canon].append(h)
                additions.setdefault(canon, []).append(h)
                matched = True
                break
        if matched:
            continue
        # safe fallback: attach to a generic key guess
        guess = None
        if re.search(r'party|affil', low):
            guess = 'party'
        elif re.search(r'cand.*id|candidate.*id|id$', low):
            guess = 'candidate_id'
        elif re.search(r'name|candidate', low):
            guess = 'candidate_name'
        elif re.search(r'receipt|receipt|total', low):
            guess = 'total_receipts'
        if guess and guess in existing:
            existing[guess].append(h)
            additions.setdefault(guess, []).append(h)
        else:
            # create a new canonical using normalized token
            key = normalize_token(h)
            if key in existing:
                existing[key].append(h)
                additions.setdefault(key, []).append(h)
            else:
                existing[key] = [h]
                additions.setdefault(key, []).append(h)
    return additions


def merge_party_map(existing: Dict[str, str], observed: Set[str]) -> Dict[str, str]:
    additions = {}
    for p in sorted(observed):
        if not p:
            continue
        up = p.strip().upper()
        if up in existing:
            continue
        # heuristics
        if up in ('D', 'DEM', 'DEMOCRAT', 'DEMOCRATIC'):
            existing[up] = 'DEM'
            additions[up] = 'DEM'
            continue
        if up in ('R', 'REP', 'GOP', 'REPUBLICAN'):
            existing[up] = 'REP'
            additions[up] = 'REP'
            continue
        if up in ('I', 'IND', 'INDEPENDENT'):
            existing[up] = 'IND'
            additions[up] = 'IND'
            continue
        # otherwise map to OTHER but record for review
        existing[up] = 'OTHER'
        additions[up] = 'OTHER'
    return additions


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', help='Directory to scan for CSV/XLSX files', required=True)
    p.add_argument('--apply', help='Apply changes to maps (default: dry-run)', action='store_true')
    args = p.parse_args()

    scan_dir = os.path.abspath(args.dir)
    if not os.path.isdir(scan_dir):
        print('Not a directory:', scan_dir)
        return

    files = [os.path.join(scan_dir, f) for f in os.listdir(scan_dir) if os.path.isfile(os.path.join(scan_dir, f)) and os.path.splitext(f)[1].lower() in ('.csv', '.xlsx', '.xls')]
    if not files:
        print('No CSV/XLSX files found in', scan_dir)
        return

    observed_headers: Set[str] = set()
    observed_parties: Set[str] = set()
    for fpath in files:
        print('Scanning', fpath)
        out = scan_file(fpath)
        observed_headers.update(out.get('headers', set()))
        observed_parties.update(out.get('parties', set()))

    print('\nObserved headers (sample):')
    for h in sorted(list(observed_headers)[:40]):
        print('  ', h)
    print('\nObserved party tokens (sample):')
    for p in sorted(list(observed_parties)[:40]):
        print('  ', p)

    aliases = load_json(ALIASES_PATH)
    party_map = load_json(PARTY_MAP_PATH)

    suggested_alias_adds = merge_aliases(aliases, observed_headers)
    suggested_party_adds = merge_party_map(party_map, observed_parties)

    print('\nSuggested alias additions:')
    for k, vals in suggested_alias_adds.items():
        print(f'  {k}: {vals}')

    print('\nSuggested party map additions:')
    for k, v in suggested_party_adds.items():
        print(f'  {k} -> {v}')

    if args.apply:
        # backups
        if os.path.exists(ALIASES_PATH):
            bak = write_json_backup(ALIASES_PATH)
            print('Backed up aliases to', bak)
        if os.path.exists(PARTY_MAP_PATH):
            bak2 = write_json_backup(PARTY_MAP_PATH)
            print('Backed up party map to', bak2)
        with open(ALIASES_PATH, 'w', encoding='utf-8') as fh:
            json.dump(aliases, fh, indent=2, sort_keys=True)
        with open(PARTY_MAP_PATH, 'w', encoding='utf-8') as fh:
            json.dump(party_map, fh, indent=2, sort_keys=True)
        print('\nApplied changes to maps.')
    else:
        print('\nDry-run complete. Rerun with --apply to write changes (backups will be created).')


if __name__ == '__main__':
    main()
